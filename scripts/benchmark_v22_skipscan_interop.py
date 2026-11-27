"""Paired inter-op thread sweep for the 9-thread skip-token-range-scan winner."""
import argparse, json, os, time
from pathlib import Path
import torch
import benchmark_v22_cpu_hypotheses as bench

INTRA_THREADS = 9
WINNER = "09-skip-token-range-scan"

def run(interop_threads):
    runtime = bench.configure_cpu_runtime(INTRA_THREADS, interop_threads)
    torch.backends.mkldnn.enabled = True
    bench.WARMUP = 5
    bench.MEASURE = 50
    data, batch_sha = bench.get_batches()
    baseline, candidate = bench.make_pair("14-skip-disabled-autocast")
    bench.patch(candidate[0], candidate[2], WINNER)

    for i in range(bench.WARMUP):
        arms = [("main", baseline, None), ("winner", candidate, WINNER)]
        if i % 2:
            arms.reverse()
        for _, bundle, name in arms:
            bench.step(bundle, data[i], i, name)

    times = {"main": [], "winner": []}
    losses = {"main": [], "winner": []}
    for i in range(bench.MEASURE):
        arms = [("main", baseline, None), ("winner", candidate, WINNER)]
        if i % 2:
            arms.reverse()
        for label, bundle, name in arms:
            t0 = time.perf_counter()
            loss = bench.step(bundle, data[bench.WARMUP+i], bench.WARMUP+i, name)
            times[label].append(time.perf_counter()-t0)
            losses[label].append(loss)

    main_elapsed = sum(times["main"])
    win_elapsed = sum(times["winner"])
    main_loss = sum(losses["main"])/len(losses["main"])
    win_loss = sum(losses["winner"])/len(losses["winner"])
    speed_delta = (main_elapsed/win_elapsed-1.0)*100.0
    loss_delta = (main_loss-win_loss)/max(abs(main_loss),1e-12)*100.0
    result = {
        "intraop_threads": torch.get_num_threads(),
        "interop_threads_requested": interop_threads,
        "interop_threads_actual": torch.get_num_interop_threads(),
        "runtime": runtime,
        "paired_interleaved": True,
        "warmup_steps_each": bench.WARMUP,
        "measured_steps_each": bench.MEASURE,
        "seed": bench.SEED,
        "batch_manifest_sha256": batch_sha,
        "batch_size": bench.BATCH,
        "sequence_length": bench.SEQ,
        "start_step": bench.START,
        "main": {"elapsed_seconds": main_elapsed, "steps_per_second": bench.MEASURE/main_elapsed, "mean_loss": main_loss},
        "winner": {"name": WINNER, "elapsed_seconds": win_elapsed, "steps_per_second": bench.MEASURE/win_elapsed, "mean_loss": win_loss},
        "speed_gain_percent_vs_main": speed_delta,
        "mean_loss_improvement_percent_vs_main": loss_delta,
        "passed_screen": speed_delta >= 1.0 and abs(loss_delta) < 0.1,
        "note": "50-step paired confirmation per inter-op setting; compare each winner against its same-job MAIN."
    }
    Path("interop-benchmark-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("INTEROP_BENCH_RESULT "+json.dumps(result), flush=True)
    Path(os.getenv("GITHUB_STEP_SUMMARY","interop.md")).write_text(
        f"# {WINNER}: {INTRA_THREADS} intra-op / {interop_threads} inter-op\n\n"
        f"MAIN: {result['main']['steps_per_second']:.4f} steps/s, loss {main_loss:.6f}\n\n"
        f"Winner: {result['winner']['steps_per_second']:.4f} steps/s, loss {win_loss:.6f}\n\n"
        f"Delta vs same-job MAIN: {speed_delta:+.3f}% speed | {loss_delta:+.6f}% loss improvement\n",
        encoding="utf-8"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interop", type=int, required=True, choices=[1,2,4,8,16])
    run(parser.parse_args().interop)

if __name__ == "__main__":
    main()
