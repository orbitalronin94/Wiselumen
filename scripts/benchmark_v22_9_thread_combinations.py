"""Paired benchmark of individual and composable CPU optimizations at 9 threads vs MAIN."""
import argparse, json, os, time
from pathlib import Path
import torch
import benchmark_v22_cpu_hypotheses as bench

THREADS = 9
CASES = {
    "01-topk-embedding-projection": ["01-topk-embedding-projection"],
    "02-topk-einsum-projection": ["02-topk-einsum-projection"],
    "03-excitability-scatter-add": ["03-excitability-scatter-add"],
    "04-rope-cast-cache": ["04-rope-cast-cache"],
    "05-rope-no-stack": ["05-rope-no-stack"],
    "06-rmsnorm-square": ["06-rmsnorm-square"],
    "07-qkv-unbind-first": ["07-qkv-unbind-first"],
    "08-manual-causal-attention": ["08-manual-causal-attention"],
    "09-skip-token-range-scan": ["09-skip-token-range-scan"],
    "10-ema-foreach": ["10-ema-foreach"],
    "11-cache-clip-parameters": ["11-cache-clip-parameters"],
    "12-cache-active-layer-set": ["12-cache-active-layer-set"],
    "13-logsumexp-loss-weights": ["13-logsumexp-loss-weights"],
    "14-skip-disabled-autocast": ["14-skip-disabled-autocast"],
    "combo-top3-04-06-11": ["04-rope-cast-cache","06-rmsnorm-square","11-cache-clip-parameters"],
    "combo-rope-nostack-rms-ema-clip": ["04-rope-cast-cache","05-rope-no-stack","06-rmsnorm-square","10-ema-foreach","11-cache-clip-parameters"],
    "combo-qkv-rope-rms-ema-clip": ["04-rope-cast-cache","05-rope-no-stack","06-rmsnorm-square","07-qkv-unbind-first","10-ema-foreach","11-cache-clip-parameters"],
    "combo-scatter-qkv-rope-rms-ema-clip": ["03-excitability-scatter-add","04-rope-cast-cache","05-rope-no-stack","06-rmsnorm-square","07-qkv-unbind-first","10-ema-foreach","11-cache-clip-parameters"],
    "combo-skipscan-qkv-rope-rms-ema-clip": ["09-skip-token-range-scan","04-rope-cast-cache","05-rope-no-stack","06-rmsnorm-square","07-qkv-unbind-first","10-ema-foreach","11-cache-clip-parameters"],
    "combo-logsum-qkv-rope-rms-ema-clip": ["13-logsumexp-loss-weights","04-rope-cast-cache","05-rope-no-stack","06-rmsnorm-square","07-qkv-unbind-first","10-ema-foreach","11-cache-clip-parameters"],
    "combo-active-qkv-rope-rms-ema-clip": ["12-cache-active-layer-set","04-rope-cast-cache","05-rope-no-stack","06-rmsnorm-square","07-qkv-unbind-first","10-ema-foreach","11-cache-clip-parameters"],
    "combo-mlp-scatter-attn-manual-rope-rms-ema-clip": ["03-excitability-scatter-add","08-manual-causal-attention","04-rope-cast-cache","05-rope-no-stack","06-rmsnorm-square","10-ema-foreach","11-cache-clip-parameters"],
    "combo-mlp-scatter-attn-qkv-rope-rms-ema-clip": ["03-excitability-scatter-add","07-qkv-unbind-first","04-rope-cast-cache","05-rope-no-stack","06-rmsnorm-square","10-ema-foreach","11-cache-clip-parameters"],
    "combo-safe-max": ["03-excitability-scatter-add","04-rope-cast-cache","05-rope-no-stack","06-rmsnorm-square","07-qkv-unbind-first","10-ema-foreach","11-cache-clip-parameters"],
}

def train_step(bundle, batch, i, features):
    model, optimizer, ema = bundle
    x, y = batch
    gs = bench.START + i
    optimizer.zero_grad(set_to_none=True)
    bench.set_scheduled_lr(optimizer, gs, 10, bench.TOTAL)
    layers = model._active if "12-cache-active-layer-set" in features else bench.active_layers_for_step(gs, "v22", len(model.blocks))
    _, loss = model(x, targets=y, active_layers=layers, return_logits=False)
    if loss is None or not torch.isfinite(loss):
        raise FloatingPointError("nonfinite loss")
    loss.backward()
    if (gs + 1) % 5 == 0:
        params = model._clip_params if "11-cache-clip-parameters" in features else model.parameters()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(gn):
            raise FloatingPointError("nonfinite grad")
    optimizer.step()
    if (gs + 1) % 10 == 0:
        ema.update(model)
    return float(loss.detach().item())

def run(case):
    features = CASES[case]
    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.backends.mkldnn.enabled = True
    data, batch_sha = bench.get_batches()
    baseline, candidate = bench.make_pair("14-skip-disabled-autocast")
    for feature in features:
        bench.patch(candidate[0], candidate[2], feature)

    for i in range(bench.WARMUP):
        arms = [("main", baseline, []), ("candidate", candidate, features)]
        if i % 2:
            arms.reverse()
        for _, bundle, fs in arms:
            train_step(bundle, data[i], i, fs)

    baseline_times, candidate_times = [], []
    baseline_losses, candidate_losses = [], []
    for i in range(bench.MEASURE):
        arms = [("main", baseline, []), ("candidate", candidate, features)]
        if i % 2:
            arms.reverse()
        for label, bundle, fs in arms:
            started = time.perf_counter()
            loss = train_step(bundle, data[bench.WARMUP+i], bench.WARMUP+i, fs)
            elapsed = time.perf_counter() - started
            if label == "main":
                baseline_times.append(elapsed); baseline_losses.append(loss)
            else:
                candidate_times.append(elapsed); candidate_losses.append(loss)

    bt, ct = sum(baseline_times), sum(candidate_times)
    bl, cl = sum(baseline_losses)/len(baseline_losses), sum(candidate_losses)/len(candidate_losses)
    speed = (bt/ct-1.0)*100.0
    loss_delta = (bl-cl)/max(abs(bl),1e-12)*100.0
    result = {
        "case": case, "features": features, "threads": THREADS, "baseline_threads": THREADS,
        "interop_threads": 1, "paired_interleaved": True,
        "warmup_steps_each": bench.WARMUP, "measured_steps_each": bench.MEASURE,
        "seed": bench.SEED, "batch_manifest_sha256": batch_sha,
        "batch_size": bench.BATCH, "sequence_length": bench.SEQ, "start_step": bench.START,
        "baseline": {"elapsed_seconds": bt, "steps_per_second": bench.MEASURE/bt, "mean_loss": bl},
        "candidate": {"elapsed_seconds": ct, "steps_per_second": bench.MEASURE/ct, "mean_loss": cl},
        "speed_gain_percent_vs_main": speed, "mean_loss_improvement_percent_vs_main": loss_delta,
        "passed_screen": speed >= 1.0 and abs(loss_delta) < 0.1,
        "note": "Short 10-step paired screen; no merge decision without longer confirmation."
    }
    Path("combo-benchmark-result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("COMBO_BENCH_RESULT "+json.dumps(result), flush=True)
    Path(os.getenv("GITHUB_STEP_SUMMARY","bench.md")).write_text(
        f"# {case} at {THREADS} threads vs MAIN\n\nFeatures: {', '.join(features)}\n\n"
        f"MAIN: {result['baseline']['steps_per_second']:.4f} steps/s, loss {bl:.6f}\n\n"
        f"Candidate: {result['candidate']['steps_per_second']:.4f} steps/s, loss {cl:.6f}\n\n"
        f"Speed delta: {speed:+.3f}% | loss improvement: {loss_delta:+.6f}% | passed={result['passed_screen']}\n",
        encoding="utf-8")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    run(parser.parse_args().case)

if __name__=="__main__":
    main()
