# Entrenamiento de LUMEN V2.
# El bucle recibe batches ya tokenizados: no presupone formato ni descarga corpus.
import json
import math
import time
from pathlib import Path

import torch

try:
    from Model import LumenV2, active_layers_for_step, seq_for_step
except ImportError:
    from model import LumenV2, active_layers_for_step, seq_for_step


def configure_cpu_runtime(threads=None, interop_threads=1):
    """Configura hilos CPU; por defecto respeta OMP_NUM_THREADS/MKL_NUM_THREADS."""
    import os

    if threads is None:
        threads = int(os.environ.get("OMP_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "1")))
    if threads <= 0:
        raise ValueError("CPU thread count must be positive.")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, int(interop_threads)))
    except RuntimeError:
        # PyTorch no permite cambiar interop threads después de iniciar trabajo.
        pass
    return {"intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads()}


def prepare_cpu_training(model, optimizer=None, use_ipex=False, compile_model=False):
    """Optimizaciones CPU opcionales; no cambia la arquitectura ni se activa sola.

    IPEX es opcional y requiere una build compatible con la versión de PyTorch.
    torch.compile también queda opt-in porque su coste de compilación puede
    superar el beneficio en runs cortos o con operadores dinámicos Top-K.
    """
    if optimizer is None:
        optimizer = get_optimizer(model)
    if use_ipex:
        try:
            import intel_extension_for_pytorch as ipex
        except ImportError as exc:
            raise RuntimeError(
                "IPEX requested but intel-extension-for-pytorch is not installed."
            ) from exc
        model, optimizer = ipex.optimize(
            model, optimizer=optimizer, dtype=torch.bfloat16, inplace=False
        )
    if compile_model:
        if not hasattr(torch, "compile"):
            raise RuntimeError("This PyTorch version does not provide torch.compile.")
        model = torch.compile(model)
    return model, optimizer


def get_optimizer(model, slow_lr=5e-4, fast_lr=2e-3, weight_decay=0.01,
                  prefer_8bit=True):
    """AdamW con dos grupos LR, parámetros deduplicados y fallback 8-bit."""
    slow, fast, seen = [], [], set()
    for index, block in enumerate(model.blocks):
        destination = fast if index >= len(model.blocks) - 2 else slow
        for parameter in block.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                destination.append(parameter)
                seen.add(id(parameter))

    # head.weight comparte almacenamiento/parámetro con tok.weight: no añadir head.
    for module in (model.tok, model.norm):
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                fast.append(parameter)
                seen.add(id(parameter))

    groups = []
    if slow:
        groups.append({"params": slow, "lr": slow_lr, "base_lr": slow_lr,
                       "weight_decay": weight_decay})
    if fast:
        groups.append({"params": fast, "lr": fast_lr, "base_lr": fast_lr,
                       "weight_decay": weight_decay})

    optimizer_class = torch.optim.AdamW
    if prefer_8bit:
        try:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
            print("Optimizer: bitsandbytes AdamW8bit", flush=True)
        except Exception as exc:
            print(f"Optimizer: AdamW vanilla (8-bit unavailable: {type(exc).__name__})",
                  flush=True)

    # foreach batches AdamW tensor operations, reducing Python/kernel-launch
    # overhead without changing optimizer mathematics. Fall back on older builds.
    try:
        optimizer = optimizer_class(
            groups, betas=(0.9, 0.95), eps=1e-8, foreach=True
        )
        print("AdamW foreach=True", flush=True)
    except TypeError:
        optimizer = optimizer_class(groups, betas=(0.9, 0.95), eps=1e-8)
        print("AdamW foreach unavailable; using standard implementation", flush=True)
    # Algunos optimizadores exponen los grupos sin base_lr; el scheduler lo necesita.
    for group in optimizer.param_groups:
        group.setdefault("base_lr", group["lr"])
    return optimizer


def homeostatic_scaling(model, max_norm=5.0):
    """Limita normas por fila en las matrices de entrada de las neuronas MLP.

    Evita escalar indiscriminadamente embeddings, atención y matrices completas.
    Se aplica únicamente cuando una fila supera el umbral.
    """
    if max_norm <= 0:
        raise ValueError("max_norm must be positive.")
    scaled = 0
    with torch.no_grad():
        for block in model.blocks:
            mlp = block.mlp
            for name in ("gate_up", "gate_up_A", "gate_up_B"):
                layer = getattr(mlp, name, None)
                weight = getattr(layer, "weight", None) if layer is not None else None
                if weight is None or weight.ndim != 2:
                    continue
                norms = weight.norm(dim=1, keepdim=True)
                factors = (max_norm / norms.clamp_min(1e-12)).clamp(max=1.0)
                changed = int((factors < 1.0).sum().item())
                if changed:
                    weight.mul_(factors)
                    scaled += changed
    return scaled


def prune_dead_neurons(model, threshold=0.001):
    """Pruning compatible con gate_up fusionado; no asume low-rank inexistente."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative.")
    n_pruned = 0
    with torch.no_grad():
        for block in model.blocks:
            mlp = block.mlp
            dead = mlp.excitability < threshold
            if not dead.any():
                continue
            n_pruned += int(dead.sum().item())
            width = mlp.d_ff
            low_rank = getattr(mlp, "low_rank", None)

            dead_indices = dead.nonzero(as_tuple=True)[0]
            if low_rank is None:
                # gate_up = [gate | up]; borrar las dos filas de cada neurona.
                rows = torch.cat((dead_indices, dead_indices + width))
                mlp.gate_up.weight[rows] = 0
                mlp.down.weight[:, dead] = 0
            else:
                # Compatibilidad futura con la parametrización low-rank si se añade.
                for attr, columns in (
                    ("gate_up_B", torch.cat((dead_indices, dead_indices + width))),
                    ("down_B", dead_indices),
                ):
                    tensor = getattr(mlp, attr, None)
                    if tensor is not None:
                        tensor[:, columns] = 0
                mlp.excitability[dead] = 0
    return n_pruned

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
    total_steps=None, start_step=0, grad_clip=1.0, ema=None, log_every=100, amp_dtype=None,
    mode="original", ema_every=10, clip_every=5, layer_schedule=None,
    max_step_seconds=None,
):
    """Ejecuta updates sobre un iterador de batches ya tokenizados.

    El caller controla dataset, batching, checkpoints y validación.
    """
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if grad_clip is not None and grad_clip <= 0:
        raise ValueError("grad_clip must be positive or None.")
    if mode not in {"original", "v21", "v22"}:
        raise ValueError("mode must be 'original', 'v21' or 'v22'.")
    if ema_every <= 0 or clip_every <= 0:
        raise ValueError("ema_every and clip_every must be positive.")
    if mode == "original":
        ema_every, clip_every = 1, 1

    device = torch.device(device or next(model.parameters()).device)
    model.to(device)
    optimizer = optimizer or get_optimizer(model)
    total = total_steps or (start_step + steps)
    if start_step < 0 or total < start_step + steps:
        raise ValueError("total_steps must cover start_step + steps; start_step must be non-negative.")

    model.train()
    iterator = iter(batches)
    history = []
    timing_history = []

    for step in range(steps):
        step_started = time.perf_counter()
        batch_started = step_started
        try:
            batch = next(iterator)
        except StopIteration as exc:
            raise RuntimeError(
                f"Batch iterator ended after {step} of {steps} steps."
            ) from exc

        batch_seconds = time.perf_counter() - batch_started
        inputs, targets = unpack_batch(batch, device)
        global_step = start_step + step
        if mode == "v21":
            requested_seq = seq_for_step(global_step, "v21")
            actual_seq = min(requested_seq, inputs.shape[1])
            inputs = inputs[:, :actual_seq].contiguous()
            targets = targets[:, :actual_seq].contiguous()
        else:
            # V2.2: preserve the complete formatted conversation. Do not apply
            # the V2.1 128-token curriculum to system+user+assistant sequences.
            actual_seq = inputs.shape[1]
        optimizer.zero_grad(set_to_none=True)
        multiplier = set_scheduled_lr(optimizer, global_step, warmup, total)

        use_amp = amp_dtype is not None and device.type in {"cuda", "cpu"}
        if layer_schedule is None:
            active_layers = active_layers_for_step(global_step, mode, len(model.blocks))
            if len(active_layers) == len(model.blocks):
                # Full-depth stages are constant: pass a range rather than
                # rebuilding a list and a set on every forward.
                active_layers = range(len(model.blocks))
        else:
            matches = [
                layers for start, end, layers in layer_schedule
                if start <= global_step < end
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"layer_schedule must define exactly one range for step {global_step}; "
                    f"matched {len(matches)}"
                )
            active_layers = matches[0]
            if not active_layers or any(
                index < 0 or index >= len(model.blocks) for index in active_layers
            ):
                raise ValueError(f"Invalid active layer indices: {active_layers}")
        forward_started = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss = model(
                inputs, targets=targets, active_layers=active_layers,
                return_logits=False,
            )
        forward_seconds = time.perf_counter() - forward_started

        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at step {step + 1}: {loss}")

        backward_started = time.perf_counter()
        loss.backward()
        backward_seconds = time.perf_counter() - backward_started
        optimizer_started = time.perf_counter()
        grad_norm = None
        do_clip = grad_clip is not None and ((global_step + 1) % clip_every == 0)
        if do_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite gradient norm at step {step + 1}.")
        optimizer.step()
        if ema is not None and ((global_step + 1) % ema_every == 0):
            ema.update(model)

        optimizer_seconds = time.perf_counter() - optimizer_started
        value = float(loss.detach().item())
        history.append(value)
        step_seconds = time.perf_counter() - step_started
        timing_history.append(step_seconds)
        if max_step_seconds is not None and step_seconds > max_step_seconds:
            raise RuntimeError(
                f"CPU performance guard: step {step + 1} took {step_seconds:.3f}s "
                f"(limit {max_step_seconds:.3f}s); timings "
                f"batch={batch_seconds:.3f}s fwd={forward_seconds:.3f}s "
                f"bwd={backward_seconds:.3f}s opt_ema={optimizer_seconds:.3f}s. "
                "Aborting before wasting the remaining block; parent checkpoint is unchanged."
            )
        if log_every and ((step + 1) % log_every == 0 or step == 0 or step + 1 == steps):
            grad_value = float(grad_norm.item()) if grad_norm is not None else float("nan")
            print(
                f"step {step + 1}/{steps} loss {value:.6f} "
                f"lr_scale {multiplier:.5f} grad_norm {grad_value:.4f} "
                f"seq {actual_seq} layers {len(active_layers)}/{len(model.blocks)} "
                f"ema_every {ema_every} clip_every {clip_every} "
                f"step_s {step_seconds:.3f} batch_s {batch_seconds:.3f} "
                f"fwd_s {forward_seconds:.3f} bwd_s {backward_seconds:.3f} "
                f"opt_ema_s {optimizer_seconds:.3f} "
                f"rolling10_s {sum(timing_history[-10:]) / len(timing_history[-10:]):.3f}",
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
