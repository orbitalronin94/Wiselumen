"""Parallel CPU hypothesis benchmark for LUMEN V2.2.

Every runner executes the same seeded model, batches, warmup and measured steps.
Only CPU thread settings and the explicitly named backend hypothesis vary.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

# GitHub executes this file from scripts/; make the repository root importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from Model import LumenV2
from Train import EMA, configure_cpu_runtime, get_optimizer, train_steps

SEED = 20261126
BATCH = 8
SEQ = 413
WARMUP = 5
MEASURE = 40


def main():
    threads = int(os.environ["BENCH_THREADS"])
    interop = int(os.environ["BENCH_INTEROP"])
    hypothesis = os.environ["BENCH_ID"]
    mkldnn = os.environ.get("BENCH_MKLDNN", "true").lower() == "true"

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.mkldnn.enabled = mkldnn
    runtime = configure_cpu_runtime(threads=threads, interop_threads=interop)

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    batches = []
    for _ in range(WARMUP + MEASURE):
        x = torch.randint(4, 4096, (BATCH, SEQ), generator=generator)
        y = torch.randint(4, 4096, (BATCH, SEQ), generator=generator)
        y[:, : int(SEQ * 0.60)] = -100
        batches.append((x, y))

    model = LumenV2(vocab=8192, ctx=512, loss_mode="experimental")
    optimizer = get_optimizer(
        model, slow_lr=5e-4, fast_lr=2e-3, weight_decay=0.01,
        prefer_8bit=False,
    )
    ema = EMA(model, decay=0.99)

    # Identical warmup work is excluded from the measured interval.
    train_steps(
        model, batches[:WARMUP], steps=WARMUP, optimizer=optimizer,
        device="cpu", warmup=10, total_steps=1000, start_step=0,
        grad_clip=1.0, ema=ema, log_every=0, amp_dtype=None, mode="v22",
        ema_every=10, clip_every=5,
    )
    started = time.perf_counter()
    result = train_steps(
        model, batches[WARMUP:], steps=MEASURE, optimizer=optimizer,
        device="cpu", warmup=10, total_steps=1000, start_step=WARMUP,
        grad_clip=1.0, ema=ema, log_every=0, amp_dtype=None, mode="v22",
        ema_every=10, clip_every=5,
    )
    elapsed = time.perf_counter() - started
    supervised_tokens = BATCH * (SEQ - int(SEQ * 0.60)) * MEASURE
    metrics = {
        "hypothesis": hypothesis,
        "threads_intra": threads,
        "threads_inter": interop,
        "mkldnn_enabled": mkldnn,
        "batch_size": BATCH,
        "sequence_length": SEQ,
        "warmup_steps": WARMUP,
        "measured_steps": MEASURE,
        "elapsed_seconds": elapsed,
        "steps_per_second": MEASURE / elapsed,
        "supervised_tokens_per_second": supervised_tokens / elapsed,
        "last_loss": result["last_loss"],
        "model_parameters": model.num_params(),
        "torch_version": torch.__version__,
        "cpu_runtime": runtime,
    }
    Path("benchmark-result.json").write_text(json.dumps(metrics, indent=2))
    print("BENCH_RESULT " + json.dumps(metrics), flush=True)
    summary = Path(os.environ["GITHUB_STEP_SUMMARY"])
    summary.write_text(
        "# LUMEN V2.2 CPU benchmark\\n\\n"
        f"- Hypothesis: `{hypothesis}`\\n"
        f"- Runtime: {threads}/{interop}; MKLDNN={mkldnn}\\n"
        f"- Steps/s: **{metrics['steps_per_second']:.5f}**\\n"
        f"- Supervised tokens/s: **{metrics['supervised_tokens_per_second']:.2f}**\\n"
        f"- Elapsed: {elapsed:.2f}s; final loss: {result['last_loss']:.5f}\\n"
    )


if __name__ == "__main__":
    main()
