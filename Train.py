# Entrenamiento de LUMEN V2.
# El bucle recibe batches ya tokenizados: no presupone formato ni descarga corpus.
import json
import math
from pathlib import Path

import torch

try:
    from Model import LumenV2
except ImportError:
    from model import LumenV2


def get_optimizer(model, slow_lr=5e-4, fast_lr=1e-3, weight_decay=0.01):
    """AdamW con dos ritmos y parámetros únicos (embedding/head tied)."""
    slow, fast = [], []
    seen = set()

    for index, block in enumerate(model.blocks):
        destination = fast if index >= len(model.blocks) - 2 else slow
        for parameter in block.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                destination.append(parameter)
                seen.add(id(parameter))

    for module in (model.tok, model.norm, model.head):
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                fast.append(parameter)
                seen.add(id(parameter))

    groups = []
    if slow:
        groups.append({"params": slow, "lr": slow_lr, "base_lr": slow_lr})
    if fast:
        groups.append({"params": fast, "lr": fast_lr, "base_lr": fast_lr})
    return torch.optim.AdamW(
        groups, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay
    )


def lr_schedule(step, warmup=200, total=100000, min_ratio=0.01):
    """Multiplicador común de LR: warmup lineal y cosine decay."""
    if total <= 0 or warmup < 0:
        raise ValueError("total must be positive and warmup non-negative.")
    if step < 0:
        raise ValueError("step must be non-negative.")
    if warmup and step < warmup:
        return max(1e-8, (step + 1) / warmup)
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def set_scheduled_lr(optimizer, step, warmup, total):
    multiplier = lr_schedule(step, warmup=warmup, total=total)
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * multiplier
    return multiplier


class EMA:
    """Exponential moving average de parámetros y buffers flotantes."""
    def __init__(self, model, decay=0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be between 0 and 1.")
        self.decay = decay
        self.shadow = {
            key: value.detach().clone().float()
            for key, value in model.state_dict().items()
            if value.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model):
        state = model.state_dict()
        for key, shadow_value in self.shadow.items():
            current = state[key].detach().to(device=shadow_value.device, dtype=torch.float32)
            shadow_value.mul_(self.decay).add_(current, alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        backup = {}
        state = model.state_dict()
        for key, average in self.shadow.items():
            tensor = state[key]
            backup[key] = tensor.detach().clone()
            tensor.copy_(average.to(device=tensor.device, dtype=tensor.dtype))
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        state = model.state_dict()
        for key, value in backup.items():
            state[key].copy_(value)


def save_sharded(model, out_dir, shard_mb=24):
    """Guarda state_dict en shards .pt con nombres y manifiesto deterministas."""
    if shard_mb <= 0:
        raise ValueError("shard_mb must be positive.")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    limit = shard_mb * 1024 * 1024
    items = list(model.state_dict().items())
    groups, current, current_bytes = [], {}, 0

    # Conserva los dtypes originales: no degrada buffers al convertirlos a BF16.
    for key, tensor in items:
        tensor_bytes = tensor.numel() * tensor.element_size()
        if current and current_bytes + tensor_bytes > limit:
            groups.append(current)
            current, current_bytes = {}, 0
        current[key] = tensor.detach().cpu().contiguous()
        current_bytes += tensor_bytes
    if current:
        groups.append(current)

    total = len(groups)
    manifest = {"format": "pytorch_state_dict", "shard_count": total, "shards": []}
    for index, shard in enumerate(groups, start=1):
        filename = f"model-{index:05d}-of-{total:05d}.pt"
        torch.save(shard, output / filename)
        manifest["shards"].append({
            "file": filename,
            "keys": list(shard.keys()),
            "size_bytes": sum(t.numel() * t.element_size() for t in shard.values()),
        })

    manifest_path = output / "model-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {total} shard(s); manifest: {manifest_path}")
    return manifest


def unpack_batch(batch, device):
    if isinstance(batch, dict):
        inputs = batch.get("input_ids")
        targets = batch.get("targets", batch.get("labels"))
    elif isinstance(batch, (tuple, list)) and len(batch) == 2:
        inputs, targets = batch
    else:
        raise TypeError("Each batch must be (input_ids, targets) or a matching dict.")
    if inputs is None or targets is None:
        raise ValueError("Batch must contain input_ids and targets/labels.")
    return inputs.to(device=device, dtype=torch.long), targets.to(device=device, dtype=torch.long)


def train_steps(
    model, batches, steps, optimizer=None, device=None, warmup=200,
    total_steps=None, grad_clip=1.0, ema=None, log_every=100, amp_dtype=None,
):
    """Ejecuta updates sobre un iterador de batches ya tokenizados.

    El caller controla dataset, batching, checkpoints y validación.
    """
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if grad_clip is not None and grad_clip <= 0:
        raise ValueError("grad_clip must be positive or None.")

    device = torch.device(device or next(model.parameters()).device)
    model.to(device)
    optimizer = optimizer or get_optimizer(model)
    total = total_steps or steps
    if total < steps:
        raise ValueError("total_steps cannot be smaller than this training run.")

    model.train()
    iterator = iter(batches)
    history = []

    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise RuntimeError(
                f"Batch iterator ended after {step} of {steps} steps."
            ) from exc

        inputs, targets = unpack_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        multiplier = set_scheduled_lr(optimizer, step, warmup, total)

        use_amp = amp_dtype is not None and device.type in {"cuda", "cpu"}
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(inputs, targets=targets)

        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {step + 1}: {loss}")

        loss.backward()
        grad_norm = None
        if grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm at step {step + 1}.")
        optimizer.step()
        if ema is not None:
            ema.update(model)

        value = float(loss.detach().item())
        history.append(value)
        if log_every and ((step + 1) % log_every == 0 or step == 0 or step + 1 == steps):
            grad_value = float(grad_norm.item()) if grad_norm is not None else float("nan")
            print(
                f"step {step + 1}/{steps} loss {value:.6f} "
                f"lr_scale {multiplier:.5f} grad_norm {grad_value:.4f}",
                flush=True,
            )

    return {"steps": steps, "losses": history, "last_loss": history[-1]}


def save_checkpoint(path, model, optimizer=None, ema=None, step=0, extra=None):
    """Checkpoint reanudable: pesos, optimizer, EMA y metadatos de paso."""
    payload = {"model": model.state_dict(), "step": int(step), "extra": extra or {}}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if ema is not None:
        payload["ema"] = {"decay": ema.decay, "shadow": ema.shadow}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return str(path)


if __name__ == "__main__":
    model = LumenV2()
    optimizer = get_optimizer(model)
    unique_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(unique_ids) == len(set(unique_ids)), "Duplicate optimizer parameters"
    print(f"Model parameters: {model.num_params():,}")
    print(f"Optimizer groups: {[len(g['params']) for g in optimizer.param_groups]}")
    print("Preflight OK. Import train_steps() with a tokenized batch iterator to train.")
