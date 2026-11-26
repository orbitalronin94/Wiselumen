"""Paired real-checkpoint CPU benchmark: main vs combined candidate.

Both arms use one physical batch of 4 per update, with no gradient accumulation.
Both arms consume identical pre-tokenized V5 examples in identical order.
"""
import copy
import hashlib
import json
import os
import platform
import random
import resource
import subprocess
import time
import zipfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tokenizers import AddedToken, Tokenizer

from Model import LumenV2, active_layers_for_step
from Train import EMA, configure_cpu_runtime, get_optimizer, set_scheduled_lr
from data import ROLE_IDS, collate_chat, format_v4_row, validate_tokenizer
from scripts.train_v22_v5_mixed import (
    V5_SHA256, V4_SHA256, TOKENIZER_SHA256, SEED, RESERVOIR_SIZE,
    load_v5_bundle, derive_tokenizer, verify_corpus_and_reservoir,
)
from scripts.v5_sampling import format_lore_row, sample_v5_batches

START_STEP = 20000
STEPS = 100
MICRO_BS = 4
EFFECTIVE_BS = 4
MICROBATCHES_PER_UPDATE = EFFECTIVE_BS // MICRO_BS
MAX_LEN = 512
CHUNK_SIZE = int(os.environ.get("LOGIT_CHUNK_SIZE", "2048"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_batches():
    root = Path("artifacts")
    root.mkdir(exist_ok=True)
    archive = root / "corpus-v4-nested.zip"
    v5_archive = root / "wise-ronin-corpus-v5.0.0.zip"
    if sha256_file(v5_archive) != V5_SHA256:
        raise RuntimeError("V5 archive SHA256 mismatch")
    manifest, lore, conflicts = load_v5_bundle(v5_archive, archive)
    with zipfile.ZipFile(archive) as z:
        v4_manifest = json.loads(z.read("corpus/manifest.json"))
        tok_bytes = z.read("corpus/tokenizer.json")
        tokenizer, original_sha, derived_sha = derive_tokenizer(
            tok_bytes, root / "tokenizer-v22-derived.json"
        )
    if original_sha != TOKENIZER_SHA256:
        raise RuntimeError("Tokenizer provenance mismatch")
    meta = validate_tokenizer(tokenizer)
    reservoir = verify_corpus_and_reservoir(
        archive, v4_manifest, random.Random(SEED)
    )
    system_prompt = Path("system_prompt_short.txt").read_text(encoding="utf-8").strip()
    v4_rows = []
    for row in reservoir:
        try:
            v4_rows.append(format_v4_row(row, tokenizer, system_prompt, max_len=MAX_LEN))
        except ValueError:
            continue
    lore_rows = []
    for row in lore:
        try:
            lore_rows.append(format_lore_row(row, tokenizer, system_prompt, max_len=MAX_LEN))
        except ValueError:
            continue
    conflict_rows = [
        format_v4_row(row, tokenizer, system_prompt, max_len=MAX_LEN)
        for row in conflicts
    ]
    if len(v4_rows) < 100 or len(lore_rows) < 20 or len(conflict_rows) != 10:
        raise RuntimeError("Insufficient V5 training pools")
    # 10 identical physical-batch-4 updates; no accumulation.
    sampled = sample_v5_batches(
        v4_rows, lore_rows, conflict_rows,
        steps=STEPS * MICROBATCHES_PER_UPDATE,
        rng=random.Random(SEED + START_STEP + 1),
        batch_size=MICRO_BS,
    )
    micro, physical = [], []
    for i in range(STEPS):
        grouped_samples = sampled[i * MICROBATCHES_PER_UPDATE:(i + 1) * MICROBATCHES_PER_UPDATE]
        flat_samples = [
            sample for one_microbatch in grouped_samples
            for _kind, sample in one_microbatch
        ]
        # Collate once so both arms use identical padding width and token positions.
        full_batch = collate_chat(flat_samples, pad_id=meta["pad_id"])
        # Fixed 512-token shape avoids compile recompilation on variable batch widths.
        width = full_batch[0].shape[1]
        if width < MAX_LEN:
            full_batch = (
                torch.nn.functional.pad(full_batch[0], (0, MAX_LEN - width), value=meta["pad_id"]),
                torch.nn.functional.pad(full_batch[1], (0, MAX_LEN - width), value=-100),
            )
        physical.append(full_batch)
        for j in range(MICROBATCHES_PER_UPDATE):
            start, end = j * MICRO_BS, (j + 1) * MICRO_BS
            micro.append((
                full_batch[0][start:end],
                full_batch[1][start:end],
            ))
    return micro, physical, {
        "corpus_v5_sha256": V5_SHA256,
        "corpus_v4_sha256": V4_SHA256,
        "tokenizer_original_sha256": original_sha,
        "tokenizer_derived_sha256": derived_sha,
        "v4_pool": len(v4_rows), "lore_pool": len(lore_rows),
        "conflict_pool": len(conflict_rows), "updates": STEPS,
        "effective_batch": EFFECTIVE_BS, "context": MAX_LEN,
    }


def load_arm(checkpoint, fused=False):
    model = LumenV2(vocab=8192, ctx=MAX_LEN, loss_mode="experimental")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer = get_optimizer(
        model, slow_lr=5e-4, fast_lr=2e-3, weight_decay=0.01,
        prefer_8bit=False,
    )
    if fused:
        groups = optimizer.param_groups
        try:
            # Probe the fused CPU kernel itself, not just constructor acceptance.
            probe_param = torch.nn.Parameter(torch.ones(2))
            probe_optimizer = torch.optim.AdamW([probe_param], fused=True)
            probe_optimizer.zero_grad(set_to_none=True)
            probe_param.square().sum().backward()
            probe_optimizer.step()
            optimizer = torch.optim.AdamW(
                groups, betas=(0.9, 0.95), eps=1e-8, fused=True
            )
            fused_status = "enabled"
        except Exception as exc:
            fused_status = f"unsupported:{type(exc).__name__}:{exc}"
            optimizer = get_optimizer(
                model, slow_lr=5e-4, fast_lr=2e-3, weight_decay=0.01,
                prefer_8bit=False,
            )
    else:
        fused_status = "main-foreach"
    optimizer.load_state_dict(checkpoint["optimizer"])
    # Checkpoint optimizer state may restore constructor flags. Pin the tested
    # CPU path explicitly: foreach=True, fused=False for BOTH arms.
    for group in optimizer.param_groups:
        group["foreach"] = True
        group["fused"] = None
    if not all(g.get("foreach") is True and g.get("fused") is None for g in optimizer.param_groups):
        raise RuntimeError("AdamW foreach CPU path was not retained after state restore")
    ema = EMA(model, decay=float(checkpoint["ema"]["decay"]))
    ema.shadow = {
        k: v.detach().clone()
        for k, v in checkpoint["ema"]["shadow"].items()
    }
    return model, optimizer, ema, fused_status


def one_update(model, optimizer, ema, batch, global_step, amp_dtype=None):
    timings = {"forward": 0.0, "backward": 0.0, "optimizer": 0.0}
    t = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    lr_scale = set_scheduled_lr(optimizer, global_step, warmup=10, total=30000)
    active = active_layers_for_step(global_step, "v22", len(model.blocks))
    ids, targets = batch[0].to(dtype=torch.long), batch[1].to(dtype=torch.long)
    enabled = amp_dtype is not None
    t0 = time.perf_counter()
    with torch.autocast("cpu", dtype=amp_dtype or torch.float32, enabled=enabled):
        _, loss = model(ids, targets=targets, active_layers=active, return_logits=False)
    timings["forward"] = time.perf_counter() - t0
    if loss is None or not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss at global step {global_step}")
    t0 = time.perf_counter()
    loss.backward()
    timings["backward"] = time.perf_counter() - t0
    if (global_step + 1) % 5 == 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite grad norm at {global_step}")
    else:
        grad_norm = torch.tensor(float("nan"))
    t0 = time.perf_counter()
    optimizer.step()
    if (global_step + 1) % 10 == 0:
        ema_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        ema.update(ema_model)
    timings["optimizer"] = time.perf_counter() - t0
    timings["total"] = time.perf_counter() - t
    return float(loss.detach()), float(grad_norm), lr_scale, timings


def run_arm(label, checkpoint, batches, variant):
    cpu_runtime = configure_cpu_runtime(threads=4, interop_threads=1)
    if cpu_runtime["intraop_threads"] != 4 or cpu_runtime["interop_threads"] != 1:
        raise RuntimeError(f"Expected intra-op=4/inter-op=1, got {cpu_runtime}")
    setup_started = time.perf_counter()
    model, optimizer, ema, fused_status = load_arm(checkpoint, fused=False)
    details = {
        "fused_adamw": fused_status, "cpu_runtime": cpu_runtime,
        "variant": variant, "compile": "not-requested",
    }
    amp_dtype = None
    if variant in ("bf16_chunked", "inductor"):
        model.logit_chunk_size = CHUNK_SIZE
        try:
            probe = torch.randn(32, 32)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                _ = probe @ probe
            amp_dtype = torch.bfloat16
            details["bf16"] = "enabled"
        except Exception as exc:
            raise RuntimeError(f"BF16 CPU probe failed: {exc}") from exc
    if variant == "inductor":
        model = torch.compile(model, backend="inductor", mode="default", fullgraph=False)
        details["compile"] = "inductor-default-graph-breaks-allowed"
    setup_seconds = time.perf_counter() - setup_started
    losses, grad_norms, phase = [], [], {"forward": [], "backward": [], "optimizer": [], "total": []}
    step_seconds = []
    started = time.perf_counter()
    for i, batch in enumerate(batches):
        step_start = time.perf_counter()
        loss, grad, _, timing = one_update(
            model, optimizer, ema, batch, START_STEP + i, amp_dtype
        )
        step_seconds.append(time.perf_counter() - step_start)
        losses.append(loss)
        grad_norms.append(grad)
        for key in phase:
            phase[key].append(timing[key])
        if (i + 1) % 10 == 0:
            print(f"{label} step={i+1}/{STEPS} loss={loss:.6f}", flush=True)
    elapsed = time.perf_counter() - started
    warm_n = min(5, STEPS) if variant == "inductor" else 0
    steady = step_seconds[warm_n:]
    tokens_processed = sum(int(batch[0].numel()) for batch in batches)
    return {
        "label": label, "variant": variant, "seconds": elapsed,
        "setup_seconds": setup_seconds, "end_to_end_seconds": setup_seconds + elapsed,
        "updates": STEPS, "effective_batch": EFFECTIVE_BS,
        "tokens_processed": tokens_processed,
        "tokens_per_second": tokens_processed / elapsed,
        "end_to_end_tokens_per_second": tokens_processed / (setup_seconds + elapsed),
        "updates_per_second": STEPS / elapsed,
        "mean_loss": sum(losses) / len(losses),
        "first10_loss": sum(losses[:10]) / min(10, len(losses)),
        "last10_loss": sum(losses[-10:]) / min(10, len(losses)),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "step_seconds_first5": step_seconds[:5],
        "step_seconds_steady_mean": sum(steady) / len(steady) if steady else None,
        "phase_seconds_mean": {k: sum(v) / len(v) for k, v in phase.items()},
        "phase_seconds_first5": {k: sum(v[:5]) / min(5, len(v)) for k, v in phase.items()},
        "phase_seconds_after5": {k: sum(v[5:]) / max(1, len(v[5:])) for k, v in phase.items()},
        "max_grad_norm": max((x for x in grad_norms if x == x), default=float("nan")),
        "details": details,
    }

def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    print("cpu", platform.processor(), flush=True)
    try:
        print(subprocess.check_output(["lscpu"], text=True), flush=True)
    except Exception:
        pass
    print("cpu_runtime_initial", configure_cpu_runtime(threads=4, interop_threads=1), flush=True)
    ckpt_path = Path("artifacts/lumen-v22-step-20000.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("step", -1)) != START_STEP:
        raise RuntimeError(f"Expected step-20000 checkpoint, got {checkpoint.get('step')}")
    if not all(k in checkpoint for k in ("model", "optimizer", "ema")):
        raise RuntimeError("Checkpoint missing model/optimizer/EMA state")
    micro, physical, provenance = prepare_batches()
    # All three arms receive the exact same 100 physical batches, in the same order.
    # Main gets batch-4 tensors; the other arms use those identical tensors too.
    batches = physical
    baseline = run_arm("main-fp32-unchunked", checkpoint, batches, "main")
    bf16_chunked = run_arm("bf16-chunked-eager", checkpoint, batches, "bf16_chunked")
    inductor = run_arm("bf16-chunked-inductor", checkpoint, batches, "inductor")
    result = {
        "status": "completed", "checkpoint_step": START_STEP,
        "checkpoint_artifact_sha256": "5a5c20fed63238b031188295e1db9eae876951e1e63137b9e969eacb22574fb",
        "steps_per_arm": STEPS, "same_effective_batch": EFFECTIVE_BS,
        "thread_configuration": {"all_arms": "intra=4,inter=1"},
        "same_examples_and_order": True, "provenance": provenance,
        "baseline": baseline, "bf16_chunked_eager": bf16_chunked, "bf16_chunked_inductor": inductor,
        "bf16_chunked_vs_main_gain_pct": (bf16_chunked["tokens_per_second"] / baseline["tokens_per_second"] - 1) * 100,
        "inductor_vs_bf16_chunked_gain_pct": (inductor["tokens_per_second"] / bf16_chunked["tokens_per_second"] - 1) * 100,
        "inductor_vs_main_gain_pct": (inductor["tokens_per_second"] / baseline["tokens_per_second"] - 1) * 100,
        "optimizer_note": "All arms explicitly use AdamW foreach=True and fused=False after loading checkpoint state.",
        "attribution": "Main vs BF16+chunked isolates precision/projection; BF16+chunked eager vs Inductor isolates compilation.",
        "timing_note": "Compilation is lazy. Inductor first-five update times and setup time are reported separately from post-warmup updates. End-to-end throughput includes setup and compilation.",
        "phase_note": "Forward, backward, optimizer-step and total update timings are instrumented separately.",
        "gradient_checkpointing": "disabled",
    }
    Path("artifacts/combined_cpu_benchmark.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("COMBINED_CPU_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
