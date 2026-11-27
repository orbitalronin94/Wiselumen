"""Fast-path continuation trainer: loads pre-tokenized V5 pools and streams batches."""
import json
import os
import platform
import random
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from Model import LumenV2, active_layers_for_step
from Train import EMA, configure_cpu_runtime, get_optimizer, save_checkpoint, train_steps
from scripts.train_v22_v5_blocks import SEED, MAX_LEN, BATCH_SIZE, LOGIT_CHUNK_SIZE
from data import collate_chat

START_STEP = int(os.environ.get("START_STEP", "30000"))
STEPS = int(os.environ.get("STEPS", "20"))
TOTAL_STEPS = START_STEP + STEPS
SCHEDULE_TOTAL_STEPS = int(os.environ.get("SCHEDULE_TOTAL_STEPS", "50000"))
PARENT_CHECKPOINT = Path(os.environ.get("PARENT_CHECKPOINT", "artifacts/lumen-v22-step-30000.pt"))
PREPARED_DATA = Path(os.environ.get("PREPARED_DATA", "artifacts/prepared-v5-training-data.pt"))
LR_SCALE = float(os.environ.get("LR_SCALE", "0.80"))


def cpu_bf16_fast_supported():
    """Only enable CPU BF16 autocast when native BF16 instructions are advertised."""
    flags = set()
    model_name = platform.processor() or "unknown"
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines()
        for line in cpuinfo:
            if line.lower().startswith("model name"):
                model_name = line.split(":", 1)[1].strip()
                break
        for line in cpuinfo:
            if line.lower().startswith("flags") or line.lower().startswith("features"):
                flags.update(line.split(":", 1)[1].lower().split())
                break
    except OSError:
        pass
    native = bool({"avx512_bf16", "amx_bf16"} & flags)
    selected = sorted(flags & {"avx512_bf16", "amx_bf16", "avx512_vnni", "avx512f"})
    return native, model_name, selected


def timed(label, fn):
    started = time.perf_counter()
    result = fn()
    print(f"LOAD_PHASE {label} seconds={time.perf_counter() - started:.3f}", flush=True)
    return result


def load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def iter_v5_batches(data, steps, seed):
    """Streaming equivalent of sample_v5_batches; never materializes the full block."""
    if steps <= 0 or steps % 10:
        raise ValueError("Exact V5 sampling requires a positive multiple of 10 steps")
    pools = {"v4": data["v4"], "lore": data["lore"], "conflict": data["conflict"]}
    if any(not pools[k] for k in pools):
        raise RuntimeError("Prepared V5 pool is empty")
    rng = random.Random(seed)
    counts = (("v4", 27), ("lore", 3), ("conflict", 10))
    for _cycle in range(steps // 10):
        tags = [kind for kind, count in counts for _ in range(count)]
        rng.shuffle(tags)
        for offset in range(0, len(tags), BATCH_SIZE):
            chosen = [(kind, rng.choice(pools[kind])) for kind in tags[offset:offset + BATCH_SIZE]]
            rng.shuffle(chosen)
            yield collate_chat(
                [sample for _kind, sample in chosen],
                pad_id=int(data["pad_id"]),
            )


def main():
    if STEPS <= 0 or STEPS % 10:
        raise ValueError("STEPS must be a positive multiple of 10")
    if SCHEDULE_TOTAL_STEPS < START_STEP + STEPS:
        raise ValueError(
            f"SCHEDULE_TOTAL_STEPS={SCHEDULE_TOTAL_STEPS} must cover target step {START_STEP + STEPS}"
        )
    # Paired benchmark winner: 9 intra-op / 8 inter-op.
    # Keep these fixed for comparable V5 continuation runs rather than
    # inheriting older workflow defaults.
    cpu_runtime = configure_cpu_runtime(threads=9, interop_threads=8)
    print("cpu_runtime", cpu_runtime, flush=True)

    data = timed("load_prepared_dataset", lambda: load_torch(PREPARED_DATA))
    if data.get("format") != "lumen_v22_v5_prepared_v1":
        raise RuntimeError("Prepared dataset format/version mismatch")
    if data.get("corpus_v5_sha256") != "8b56fcace9c025880193d55c13ceaec01b09f99e8055f9961b558da00dd788ba":
        raise RuntimeError("Prepared dataset corpus SHA256 mismatch")

    payload = timed("load_parent_checkpoint", lambda: load_torch(PARENT_CHECKPOINT))
    parent_step = int(payload.get("step", -1))
    if parent_step != START_STEP:
        raise RuntimeError(f"Expected parent step {START_STEP}, got {parent_step}")
    if "optimizer" not in payload or "ema" not in payload:
        raise RuntimeError("Parent checkpoint lacks optimizer or EMA state")
    if not 0.0 < LR_SCALE <= 1.0:
        raise ValueError(f"LR_SCALE must be in (0, 1], got {LR_SCALE}")

    model = LumenV2(vocab=8192, ctx=MAX_LEN, loss_mode="experimental")
    # V5 batches come from the frozen, SHA-verified tokenizer/corpus pools.
    # Avoid the redundant token min/max reduction during trusted training.
    model.skip_token_range_scan = True
    model.logit_chunk_size = LOGIT_CHUNK_SIZE
    optimizer = get_optimizer(
        model, slow_lr=5e-4, fast_lr=2e-3, weight_decay=0.01, prefer_8bit=False
    )
    ema = EMA(model, decay=0.99)

    timed("restore_model", lambda: model.load_state_dict(payload["model"], strict=True))
    timed("restore_optimizer", lambda: optimizer.load_state_dict(payload["optimizer"]))
    # The measured winner is AdamW foreach=True; never force CPU fused AdamW.
    for group in optimizer.param_groups:
        group["foreach"] = True
        group["fused"] = None
    if not all(group.get("foreach") is True and group.get("fused") is None for group in optimizer.param_groups):
        raise RuntimeError("Could not restore the benchmarked foreach AdamW configuration")

    # Apply the 20% reduction once, not once per 2.5k continuation block.
    # The optimizer state already carries scaled base_lr in subsequent checkpoints.
    parent_extra = payload.get("extra", {})
    previous_scale = float(parent_extra.get("lr_scale", 1.0))
    if abs(previous_scale - LR_SCALE) > 1e-12:
        for group in optimizer.param_groups:
            restored_base_lr = float(group.get("base_lr", group["lr"]))
            group["base_lr"] = restored_base_lr * LR_SCALE
            group["lr"] = float(group["lr"]) * LR_SCALE
        print(f"LR_SCALE_APPLIED previous={previous_scale:.4f} current={LR_SCALE:.4f}", flush=True)
    else:
        print(f"LR_SCALE_ALREADY_PRESENT scale={previous_scale:.4f}; preserving optimizer LR", flush=True)
    lr_groups = [float(group["base_lr"]) for group in optimizer.param_groups]

    ema.decay = float(payload["ema"]["decay"])
    ema.shadow = payload["ema"]["shadow"]
    bf16_native, cpu_model, cpu_features = cpu_bf16_fast_supported()
    amp_dtype = torch.bfloat16 if bf16_native else None
    bf16_label = "enabled" if bf16_native else "disabled-fp32"
    print(
        f"CPU_FEATURES model={cpu_model!r} native_bf16={bf16_native} "
        f"features={cpu_features} torch_capability={torch.backends.cpu.get_cpu_capability()}",
        flush=True,
    )
    print(
        f"FASTPATH_CONFIG bf16={bf16_label} "
        f"adamw_foreach=true fused=false lr_scale={LR_SCALE:.2f} base_lrs={lr_groups} "
        f"logit_chunk_size={LOGIT_CHUNK_SIZE} compile=false ipex=false "
        f"batch={BATCH_SIZE} context={MAX_LEN}",
        flush=True,
    )
    print(
        f"RESUME_OK parent_step={parent_step} target_step={TOTAL_STEPS} "
        f"active_layers={len(active_layers_for_step(START_STEP, 'v22', len(model.blocks)))} "
        f"v4_share=0.675 lore_share=0.075 synthetic_conflict_share=0.250",
        flush=True,
    )

    batches = iter_v5_batches(data, STEPS, SEED + START_STEP + 1)
    started = time.perf_counter()
    result = train_steps(
        model, batches, steps=STEPS, optimizer=optimizer, device="cpu",
        warmup=10, total_steps=SCHEDULE_TOTAL_STEPS, start_step=START_STEP,
        grad_clip=1.0, ema=ema, log_every=1, amp_dtype=amp_dtype,
        mode="v22", ema_every=10, clip_every=5, max_step_seconds=20.0,
    )
    elapsed = time.perf_counter() - started

    checkpoint_path = Path(f"artifacts/lumen-v22-v5-fastpath-step-{TOTAL_STEPS}.pt")
    checkpoint = save_checkpoint(
        str(checkpoint_path), model, optimizer=optimizer, ema=ema,
        step=TOTAL_STEPS,
        extra={
            "seed": SEED,
            "corpus_version": data["corpus_version"],
            "corpus_v5_sha256": data["corpus_v5_sha256"],
            "corpus_v4_sha256": data["corpus_v4_sha256"],
            "tokenizer_original_sha256": data["tokenizer_original_sha256"],
            "tokenizer_derived_sha256": data["tokenizer_derived_sha256"],
            "sampling_shares": data["sampling_shares"],
            "batch_size": BATCH_SIZE, "max_len": MAX_LEN,
            "mode": "v22", "loss_mode": "experimental",
            "amp_dtype": "bfloat16" if bf16_native else "float32", "adamw_foreach": True,
            "fused_adamw": False, "logit_chunk_size": LOGIT_CHUNK_SIZE,
            "lr_scale": LR_SCALE, "base_lrs": lr_groups,
            "schedule_total_steps": SCHEDULE_TOTAL_STEPS,
            "compile": False, "ipex": False,
        },
    )
    metrics = {
        "status": "completed", "steps": TOTAL_STEPS,
        "parent_step": START_STEP, "additional_steps": STEPS,
        "last_loss": result["last_loss"],
        "mean_first_10": sum(result["losses"][:10]) / min(10, len(result["losses"])),
        "mean_last_10": sum(result["losses"][-10:]) / min(10, len(result["losses"])),
        "elapsed_seconds": elapsed, "seconds_per_step": elapsed / STEPS,
        "steps_per_second": STEPS / elapsed,
        "corpus_version": data["corpus_version"],
        "corpus_v5_sha256": data["corpus_v5_sha256"],
        "tokenizer_derived_sha256": data["tokenizer_derived_sha256"],
        "checkpoint": checkpoint, "checkpoint_path": str(checkpoint_path),
        "model_parameters": model.num_params(),
        "cpu_runtime": {"intraop_threads": torch.get_num_threads(),
                        "interop_threads": torch.get_num_interop_threads()},
        "amp_dtype": "bfloat16", "adamw_foreach": True, "fused_adamw": False,
        "logit_chunk_size": LOGIT_CHUNK_SIZE,
        "lr_scale": LR_SCALE, "base_lrs": lr_groups,
        "schedule_total_steps": SCHEDULE_TOTAL_STEPS,
    }
    Path(f"artifacts/metrics-v5-fastpath-step-{TOTAL_STEPS}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("FASTPATH_RESULT " + json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
