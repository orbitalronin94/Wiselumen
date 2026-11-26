"""Paired real-checkpoint CPU benchmark: main vs combined candidate.

The baseline emulates the production batch-4 update using 8 microbatches of 4.
The candidate uses one physical batch of 32 for the same 32 examples/update.
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
EFFECTIVE_BS = 32
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
    # 800 x batch-4 = 100 identical effective-batch-32 updates.
    sampled = sample_v5_batches(
        v4_rows, lore_rows, conflict_rows,
        steps=STEPS * (EFFECTIVE_BS // MICRO_BS),
        rng=random.Random(SEED + START_STEP + 1),
        batch_size=MICRO_BS,
    )
    micro, physical = [], []
    for i in range(STEPS):
        grouped_samples = sampled[i * 8:(i + 1) * 8]
        flat_samples = [
            sample for one_microbatch in grouped_samples
            for _kind, sample in one_microbatch
        ]
        # Collate once so both arms use identical padding width and token positions.
        full_batch = collate_chat(flat_samples, pad_id=meta["pad_id"])
        physical.append(full_batch)
        for j in range(8):
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
    ema = EMA(model, decay=float(checkpoint["ema"]["decay"]))
    ema.shadow = {
        k: v.detach().clone()
        for k, v in checkpoint["ema"]["shadow"].items()
    }
    return model, optimizer, ema, fused_status


def one_update(model, optimizer, ema, batches, global_step, amp_dtype=None):
    optimizer.zero_grad(set_to_none=True)
    lr_scale = set_scheduled_lr(optimizer, global_step, warmup=10, total=30000)
    active = active_layers_for_step(global_step, "v22", len(model.blocks))
    losses = []
    token_counts = [int(batch[1].ne(-100).sum()) for batch in batches]
    total_tokens = sum(token_counts)
    if total_tokens <= 0:
        raise RuntimeError("effective batch contains no supervised tokens")
    for batch, token_count in zip(batches, token_counts):
        ids = batch[0].to(dtype=torch.long)
        targets = batch[1].to(dtype=torch.long)
        enabled = amp_dtype is not None
        with torch.autocast("cpu", dtype=amp_dtype or torch.float32, enabled=enabled):
            _, loss = model(ids, targets=targets, active_layers=active, return_logits=False)
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at global step {global_step}")
        (loss * (token_count / total_tokens)).backward()
        losses.append((float(loss.detach()), token_count))
    grad_norm = torch.tensor(float("nan"))
    if (global_step + 1) % 5 == 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite grad norm at {global_step}")
    optimizer.step()
    if (global_step + 1) % 10 == 0:
        ema_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        ema.update(ema_model)
    mean_loss = sum(value * count for value, count in losses) / total_tokens
    return mean_loss, float(grad_norm), lr_scale


def run_arm(label, checkpoint, batches, micro_accum, candidate=False):
    setup_started = time.perf_counter()
    model, optimizer, ema, fused_status = load_arm(checkpoint, fused=candidate)
    details = {"fused_adamw": fused_status, "ipex": "not-requested", "compile": "eager"}
    amp_dtype = None
    if candidate:
        # Candidate-only exact-objective CE chunking; main keeps the default path.
        model.logit_chunk_size = CHUNK_SIZE
        details["logit_chunk_size"] = CHUNK_SIZE
        # CPU BF16 is tested on this exact runner; unsupported hardware is recorded.
        try:
            probe = torch.randn(32, 32)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                _ = probe @ probe
            amp_dtype = torch.bfloat16
            details["bf16"] = "enabled"
        except Exception as exc:
            details["bf16"] = f"unsupported:{type(exc).__name__}"
        try:
            import intel_extension_for_pytorch as ipex
            model, optimizer = ipex.optimize(
                model, optimizer=optimizer, dtype=torch.bfloat16, inplace=False
            )
            details["ipex"] = getattr(ipex, "__version__", "installed")
        except Exception as exc:
            details["ipex"] = f"unavailable:{type(exc).__name__}:{exc}"
        model.logit_chunk_size = CHUNK_SIZE
        try:
            model = torch.compile(
                model, mode="max-autotune", fullgraph=True
            )
            details["compile"] = "max-autotune-fullgraph"
        except Exception as exc:
            details["compile"] = f"setup-failed:{type(exc).__name__}:{exc}"
    setup_seconds = time.perf_counter() - setup_started
    started = time.perf_counter()
    losses, grad_norms = [], []
    compile_failed_seconds = 0.0
    for i in range(STEPS):
        group = batches[i] if not micro_accum else batches[i * 8:(i + 1) * 8]
        try:
            loss, grad, _ = one_update(
                model, optimizer, ema, group, START_STEP + i, amp_dtype
            )
        except Exception as exc:
            if candidate and details["compile"] != "eager":
                # Retry from the exact parent state in eager mode; no partial candidate
                # result is retained after a compile/IPEX graph failure.
                details["compile_runtime_error"] = f"{type(exc).__name__}:{exc}"
                compile_failed_seconds = time.perf_counter() - started
                model, optimizer, ema, fused_status = load_arm(checkpoint, fused=True)
                model.logit_chunk_size = CHUNK_SIZE
                details["fused_adamw"] = fused_status
                details["compile_fallback"] = "eager-restart-from-parent"
                details["ipex"] = "discarded-after-compile-failure"
                losses, grad_norms = [], []
                started = time.perf_counter()
                for j in range(STEPS):
                    loss, grad, _ = one_update(
                        model, optimizer, ema, batches[j], START_STEP + j, amp_dtype
                    )
                    losses.append(loss); grad_norms.append(grad)
                break
            raise
        losses.append(loss); grad_norms.append(grad)
        if (i + 1) % 10 == 0:
            print(f"{label} step={i+1}/{STEPS} loss={loss:.6f}", flush=True)
    elapsed = time.perf_counter() - started
    total_elapsed = setup_seconds + compile_failed_seconds + elapsed
    tokens_processed = sum(int(batch[0].numel()) for batch in batches)
    return {
        "label": label, "seconds": elapsed, "setup_seconds": setup_seconds,
        "compile_failed_seconds": compile_failed_seconds,
        "end_to_end_seconds": total_elapsed, "updates": STEPS,
        "effective_examples": STEPS * EFFECTIVE_BS,
        "tokens_processed": tokens_processed,
        "tokens_per_second": tokens_processed / elapsed,
        "updates_per_second": STEPS / elapsed,
        "end_to_end_tokens_per_second": tokens_processed / total_elapsed,
        "mean_loss": sum(losses) / len(losses),
        "first10_loss": sum(losses[:10]) / min(10, len(losses)),
        "last10_loss": sum(losses[-10:]) / min(10, len(losses)),
        "max_grad_norm": max((x for x in grad_norms if x == x), default=float("nan")),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
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
    print("cpu_runtime", configure_cpu_runtime(threads=4, interop_threads=1), flush=True)
    ckpt_path = Path("artifacts/lumen-v22-step-20000.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("step", -1)) != START_STEP:
        raise RuntimeError(f"Expected step-20000 checkpoint, got {checkpoint.get('step')}")
    if not all(k in checkpoint for k in ("model", "optimizer", "ema")):
        raise RuntimeError("Checkpoint missing model/optimizer/EMA state")
    micro, physical, provenance = prepare_batches()
    baseline = run_arm("main-batch4-accum8", checkpoint, micro, True, False)
    candidate = run_arm("combined-batch32", checkpoint, physical, False, True)
    result = {
        "status": "completed", "checkpoint_step": START_STEP,
        "checkpoint_artifact_sha256": "5a5c20fed63238b031188295e1db9eae876951e1e63137b9e969eacb22574fb",
        "steps_per_arm": STEPS, "same_effective_batch": EFFECTIVE_BS,
        "same_examples_and_order": True, "provenance": provenance,
        "baseline": baseline, "candidate": candidate,
        "candidate_vs_main_tokens_per_second_gain_pct": (
            candidate["tokens_per_second"] / baseline["tokens_per_second"] - 1
        ) * 100,
        "candidate_vs_main_end_to_end_gain_pct": (
            candidate["end_to_end_tokens_per_second"] / baseline["end_to_end_tokens_per_second"] - 1
        ) * 100,
        "loss_comparison_note": "Same examples and effective batch per update; BF16/compile/IPEX/fused AdamW may change numerics.",
        "chunked_logits_note": "Candidate enables exact-objective chunked projection/CE over supervised positions (chunk size 2048); main keeps the existing supervised-only, unchunked path. Neither materializes full [B,T,V].",
        "muon_note": "Muon excluded: it changes optimizer mathematics and cannot be treated as a transparent CPU kernel optimization.",
        "gradient_checkpointing": "disabled",
        "checkpoint_policy": "only final benchmark artifact; no periodic training checkpoint in a 100-step benchmark",
    }
    Path("artifacts/combined_cpu_benchmark.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("COMBINED_CPU_RESULT " + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
