"""Paired 100-update CPU screening: frozen main baseline vs one hypothesis."""
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path
import torch
from Model import LumenV2, active_layers_for_step
from Train import EMA, configure_cpu_runtime, get_optimizer, set_scheduled_lr

SEED = 20261126
BASELINE_SHA = "c27c61982b87611f7c809f6aa25f3b06da5975d7"
STEPS = 100
WARMUP = 5
BATCH = 2
SEQ = 192
VOCAB = 8192
START_STEP = 10000
ACTIVE_MAIN = active_layers_for_step(START_STEP, "v22", 8)

def sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def make_batches():
    gen = torch.Generator(device="cpu").manual_seed(SEED + 99)
    batches, manifest = [], []
    for _ in range(WARMUP + STEPS):
        x = torch.randint(4, 4096, (BATCH, SEQ), generator=gen)
        y = torch.randint(4, 4096, (BATCH, SEQ), generator=gen)
        y[:, : int(SEQ * 0.60)] = -100
        batches.append((x, y))
        manifest.append({
            "input_sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(),
            "target_sha256": hashlib.sha256(y.numpy().tobytes()).hexdigest(),
        })
    return batches, sha256_json(manifest)

def make_model():
    torch.manual_seed(SEED)
    return LumenV2(vocab=VOCAB, ctx=512, loss_mode="experimental")

def run_arm(model, batches, candidate):
    model.train()
    optimizer = get_optimizer(model, slow_lr=5e-4, fast_lr=2e-3,
                              weight_decay=0.01, prefer_8bit=False)
    ema = EMA(model, decay=0.99)
    losses = []
    rng = random.Random(SEED + 7001)
    ema_every = 25 if candidate == "ema_every_25" else 10
    eval_x, eval_y = batches[-1]
    started = None
    for step, (inputs, targets) in enumerate(batches):
        if step == WARMUP:
            started = time.perf_counter()
        global_step = START_STEP + step
        optimizer.zero_grad(set_to_none=True)
        set_scheduled_lr(optimizer, global_step, warmup=200, total=30000)
        active = list(ACTIVE_MAIN)
        if candidate == "layerdrop_4of8":
            active = [0, 1, 6, 7]
        elif candidate == "stochastic_depth_4of6":
            active = sorted(rng.sample(ACTIVE_MAIN, 4))
        elif candidate == "topk_192":
            for block in model.blocks:
                block.mlp.top_k = 192
        elif candidate == "topk_128":
            for block in model.blocks:
                block.mlp.top_k = 128
        _, loss = model(inputs, targets=targets, active_layers=active, return_logits=False)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at update {step}")
        loss.backward()
        if (global_step + 1) % 5 == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite grad norm at update {step}")
        optimizer.step()
        if (global_step + 1) % ema_every == 0:
            ema.update(model)
        if step >= WARMUP:
            losses.append(float(loss.detach()))
    elapsed = time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        eval_layers = [0, 1, 6, 7] if candidate == "layerdrop_4of8" else ACTIVE_MAIN
        _, eval_loss = model(eval_x, targets=eval_y, active_layers=eval_layers, return_logits=False)
    return {
        "elapsed_seconds": elapsed,
        "steps_per_second": STEPS / elapsed,
        "mean_first10_loss": sum(losses[:10]) / 10,
        "mean_last10_loss": sum(losses[-10:]) / 10,
        "eval_ce_diagnostic": float(eval_loss),
        "max_step_loss": max(losses),
        "min_step_loss": min(losses),
        "ema_every": ema_every,
        "active_layers_main": ACTIVE_MAIN,
    }

def main():
    candidate = os.environ["HYPOTHESIS"]
    allowed = {"layerdrop_4of8", "stochastic_depth_4of6", "topk_192",
               "topk_128", "ema_every_25"}
    if candidate not in allowed:
        raise ValueError(f"Unknown candidate: {candidate}")
    runtime = configure_cpu_runtime(
        threads=int(os.environ.get("OMP_NUM_THREADS", "4")),
        interop_threads=int(os.environ.get("TORCH_INTEROP_THREADS", "2")),
    )
    torch.backends.mkldnn.enabled = True
    batches, batch_sha = make_batches()
    initial = make_model()
    initial_state = copy.deepcopy(initial.state_dict())
    baseline = make_model()
    baseline.load_state_dict(initial_state, strict=True)
    variant = make_model()
    variant.load_state_dict(initial_state, strict=True)
    order = os.environ.get("PAIR_ORDER", "AB")
    if order == "AB":
        baseline_result = run_arm(baseline, batches, "main")
        variant_result = run_arm(variant, batches, candidate)
    else:
        variant_result = run_arm(variant, batches, candidate)
        baseline_result = run_arm(baseline, batches, "main")
    speedup = baseline_result["elapsed_seconds"] / variant_result["elapsed_seconds"]
    improvement_pct = (speedup - 1.0) * 100.0
    eval_delta = variant_result["eval_ce_diagnostic"] - baseline_result["eval_ce_diagnostic"]
    passed = improvement_pct >= 3.0 and eval_delta <= 0.02
    result = {
        "candidate": candidate,
        "baseline_sha": BASELINE_SHA,
        "seed": SEED,
        "steps_per_arm": STEPS,
        "warmup_per_arm": WARMUP,
        "pair_order": order,
        "batch_manifest_sha256": batch_sha,
        "baseline": baseline_result,
        "variant": variant_result,
        "speedup_ratio": speedup,
        "time_improvement_pct": improvement_pct,
        "eval_ce_delta": eval_delta,
        "screen_pass": passed,
        "runtime": runtime,
        "torch_version": torch.__version__,
        "threads": int(os.environ.get("OMP_NUM_THREADS", "4")),
        "interop_threads": int(os.environ.get("TORCH_INTEROP_THREADS", "2")),
    }
    Path("paired-benchmark-result.json").write_text(json.dumps(result, indent=2))
    print("PAIRED_RESULT " + json.dumps(result), flush=True)
    summary = Path(os.environ["GITHUB_STEP_SUMMARY"])
    summary.write_text(
        "# Paired LUMEN benchmark: " + candidate + "\n\n"
        + "- Baseline SHA: " + BASELINE_SHA + "\n"
        + "- Paired updates: 100 per arm; warmup: 5\n"
        + "- Order: " + order + "; batch SHA256: " + batch_sha + "\n"
        + "- Baseline: " + f"{baseline_result['elapsed_seconds']:.2f}s "
          + f"({baseline_result['steps_per_second']:.3f} steps/s)\n"
        + "- Variant: " + f"{variant_result['elapsed_seconds']:.2f}s "
          + f"({variant_result['steps_per_second']:.3f} steps/s)\n"
        + "- Time improvement: " + f"{improvement_pct:.2f}%\n"
        + "- Diagnostic eval CE delta: " + f"{eval_delta:.5f}\n"
        + "- Promotion screen: " + ("PASS" if passed else "NO PASS") + "\n\n"
        + "Synthetic random-token batches are only a controlled throughput screen; "
          "they do not establish language-model quality or generalization.\n"
    )

if __name__ == "__main__":
    main()
