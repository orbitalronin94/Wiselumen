"""CPU throughput benchmark for LUMEN V2.2; no corpus I/O in timed region."""
import json
import os
import time
from pathlib import Path
import sys

# Running a file under scripts/ puts that directory, not the repository root,
# at sys.path[0]. Add the project root so Model.py and Train.py resolve reliably.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from Model import LumenV2
from Train import EMA, configure_cpu_runtime, get_optimizer, train_steps

THREADS = int(os.environ.get("BENCH_THREADS", "2"))
BATCH = int(os.environ.get("BENCH_BATCH", "8"))
SEQ = int(os.environ.get("BENCH_SEQ", "413"))
WARMUP_STEPS = int(os.environ.get("BENCH_WARMUP", "3"))
MEASURE_STEPS = int(os.environ.get("BENCH_STEPS", "25"))
SEED = 20261126


def make_batches(count):
    generator = torch.Generator(device="cpu").manual_seed(SEED + BATCH + THREADS)
    batches = []
    for _ in range(count):
        x = torch.randint(4, 4096, (BATCH, SEQ), generator=generator)
        y = torch.randint(4, 4096, (BATCH, SEQ), generator=generator)
        # Approximate assistant-only supervision: leave the leading 60% masked.
        y[:, : int(SEQ * 0.60)] = -100
        batches.append((x, y))
    return batches


def main():
    torch.manual_seed(SEED)
    runtime = configure_cpu_runtime(threads=THREADS)
    print("BENCH_RUNTIME", json.dumps(runtime), flush=True)
    batches = make_batches(WARMUP_STEPS + MEASURE_STEPS)

    model = LumenV2(vocab=8192, ctx=512, loss_mode="experimental")
    optimizer = get_optimizer(model, prefer_8bit=False)
    ema = EMA(model, decay=0.99)

    train_steps(
        model, batches[:WARMUP_STEPS], steps=WARMUP_STEPS,
        optimizer=optimizer, device="cpu", warmup=10, total_steps=1000,
        start_step=0, grad_clip=1.0, ema=ema, log_every=0,
        amp_dtype=None, mode="v22", ema_every=10, clip_every=5,
    )

    start = time.perf_counter()
    result = train_steps(
        model, batches[WARMUP_STEPS:], steps=MEASURE_STEPS,
        optimizer=optimizer, device="cpu", warmup=10, total_steps=1000,
        start_step=WARMUP_STEPS, grad_clip=1.0, ema=ema, log_every=0,
        amp_dtype=None, mode="v22", ema_every=10, clip_every=5,
    )
    elapsed = time.perf_counter() - start
    supervised_tokens = BATCH * (SEQ - int(SEQ * 0.60)) * MEASURE_STEPS
    metrics = {
        "threads": THREADS,
        "batch_size": BATCH,
        "sequence_length": SEQ,
        "warmup_steps": WARMUP_STEPS,
        "measured_steps": MEASURE_STEPS,
        "elapsed_seconds": elapsed,
        "steps_per_second": MEASURE_STEPS / elapsed,
        "supervised_tokens_per_second": supervised_tokens / elapsed,
        "last_loss": result["last_loss"],
        "model_parameters": model.num_params(),
        "torch_version": torch.__version__,
        "cpu_runtime": runtime,
    }
    Path("benchmark-result.json").write_text(json.dumps(metrics, indent=2))
    print("BENCH_RESULT " + json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
