"""Paired CPU benchmark: PyTorch intra-op thread count vs 4-thread main."""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import benchmark_v22_cpu_hypotheses as bench

BASELINE_THREADS = 4
CONFIGS = list(range(1, 17)) + [24]

def run(candidate_threads):
    bench.configure_cpu_runtime(BASELINE_THREADS, 1)
    torch.backends.mkldnn.enabled = True
    data, batch_sha = bench.get_batches()
    baseline, candidate = bench.make_pair("14-skip-disabled-autocast")
    baseline_times, candidate_times = [], []
    baseline_losses, candidate_losses = [], []

    def measured(bundle, batch, step_index, threads):
        torch.set_num_threads(threads)
        start = time.perf_counter()
        loss = bench.step(bundle, batch, step_index, None)
        return time.perf_counter() - start, loss

    for i in range(bench.WARMUP):
        arms = [(baseline, BASELINE_THREADS), (candidate, candidate_threads)]
        if i % 2:
            arms.reverse()
        for bundle, threads in arms:
            torch.set_num_threads(threads)
            bench.step(bundle, data[i], i, None)

    for i in range(bench.MEASURE):
        arms = [("baseline", baseline, BASELINE_THREADS),
                ("candidate", candidate, candidate_threads)]
        if i % 2:
            arms.reverse()
        for label, bundle, threads in arms:
            elapsed, loss = measured(bundle, data[bench.WARMUP + i],
                                     bench.WARMUP + i, threads)
            if label == "baseline":
                baseline_times.append(elapsed)
                baseline_losses.append(loss)
            else:
                candidate_times.append(elapsed)
                candidate_losses.append(loss)

    base_elapsed, cand_elapsed = sum(baseline_times), sum(candidate_times)
    base_loss = sum(baseline_losses) / len(baseline_losses)
    cand_loss = sum(candidate_losses) / len(candidate_losses)
    speed_gain = (base_elapsed / cand_elapsed - 1.0) * 100.0
    loss_delta = (base_loss - cand_loss) / max(abs(base_loss), 1e-12) * 100.0
    return {
        "test": "intra_op_threads",
        "baseline_threads": BASELINE_THREADS,
        "candidate_threads": candidate_threads,
        "interop_threads": 1,
        "paired_interleaved": True,
        "warmup_steps_each": bench.WARMUP,
        "measured_steps_each": bench.MEASURE,
        "seed": bench.SEED,
        "batch_manifest_sha256": batch_sha,
        "batch_size": bench.BATCH,
        "sequence_length": bench.SEQ,
        "start_step": bench.START,
        "torch_version": torch.__version__,
        "baseline": {"elapsed_seconds": base_elapsed,
                     "steps_per_second": bench.MEASURE / base_elapsed,
                     "mean_loss": base_loss},
        "candidate": {"elapsed_seconds": cand_elapsed,
                      "steps_per_second": bench.MEASURE / cand_elapsed,
                      "mean_loss": cand_loss},
        "speed_gain_percent": speed_gain,
        "mean_loss_delta_percent": loss_delta,
        "passed_screen": speed_gain >= 1.0 and abs(loss_delta) < 0.1,
        "decision": "10-step screen only; confirm winner with longer paired run",
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, required=True, choices=CONFIGS)
    args = parser.parse_args()
    result = run(args.threads)
    Path("thread-benchmark-result.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print("THREAD_BENCH_RESULT " + json.dumps(result), flush=True)
    Path(os.getenv("GITHUB_STEP_SUMMARY", "bench.md")).write_text(
        f"# Intra-op threads: {args.threads} vs 4\n\n"
        f"Baseline: {result['baseline']['steps_per_second']:.4f} steps/s "
        f"(loss {result['baseline']['mean_loss']:.6f})\n\n"
        f"Candidate: {result['candidate']['steps_per_second']:.4f} steps/s "
        f"(loss {result['candidate']['mean_loss']:.6f})\n\n"
        f"Speed: {result['speed_gain_percent']:.3f}% | "
        f"loss delta: {result['mean_loss_delta_percent']:.6f}% | "
        f"passed_screen={result['passed_screen']}\n")

if __name__ == "__main__":
    main()
