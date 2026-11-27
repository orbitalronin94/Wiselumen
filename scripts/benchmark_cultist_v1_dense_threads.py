#!/usr/bin/env python3
"""Paired 50-step throughput tournament: current main (4/4) vs one thread pair."""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/"scripts/train_cultist_v1_dense.py"
OUT=ROOT/"artifacts/thread-tournament"
STEPS=50
MAIN=(4,4)

def run(label: str, intra: int, inter: int) -> dict:
    out=OUT/label
    env=os.environ.copy()
    env.update({
        "OMP_NUM_THREADS":str(intra),
        "MKL_NUM_THREADS":str(intra),
        "OPENBLAS_NUM_THREADS":str(intra),
        "TORCH_INTEROP_THREADS":str(inter),
        "STEPS":str(STEPS),
        "TOTAL_TRAIN_STEPS":str(STEPS),
        "WARMUP":"50",
        "BASE_LR_MAIN":"0.0008",
        "BASE_LR_EMBED":"0.0004",
        "BATCH_SIZE":"4",
        "CTX":"256",
        "SAMPLED_NEGATIVES":"1024",
        "PROFILE":"0",
        "PARENT_STEP_OFFSET":"0",
        "OUT_DIR":str(out),
        "PYTHONUNBUFFERED":"1",
    })
    subprocess.run([sys.executable,str(TRAIN)],cwd=ROOT,env=env,check=True)
    metrics_path=out/f"cultistlumen-v1-dense-metrics-step-{STEPS}.json"
    metrics=json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "label":label,"intra_threads":intra,"interop_threads":inter,
        "steps":metrics["steps"],"elapsed_seconds":metrics["elapsed_seconds"],
        "steps_per_second":metrics["steps_per_second"],
        "last_loss":metrics["last_loss"],
        "mean_first_10":metrics["mean_first_10"],
        "mean_last_10":metrics["mean_last_10"],
        "phase_timing_seconds":metrics["phase_timing_seconds"],
        "checkpoint":metrics["checkpoint"],
    }

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--intra",type=int,required=True)
    p.add_argument("--interop",type=int,required=True)
    a=p.parse_args()
    if not (1<=a.intra<=20 and 1<=a.interop<=20):
        raise SystemExit("Thread counts must be between 1 and 20")
    OUT.mkdir(parents=True,exist_ok=True)
    candidate=(a.intra,a.interop)
    # Alternate order to reduce systematic first-run / second-run bias.
    if (a.intra+a.interop)%2:
        cand=run("candidate",*candidate); base=run("main",*MAIN)
    else:
        base=run("main",*MAIN); cand=run("candidate",*candidate)
    speedup=cand["steps_per_second"]/base["steps_per_second"] if base["steps_per_second"] else None
    result={
        "status":"completed","benchmark":"CULTISTLUMEN-V1-DENSE paired thread tournament",
        "steps_per_arm":STEPS,"main_config":{"intra_threads":MAIN[0],"interop_threads":MAIN[1]},
        "candidate_config":{"intra_threads":a.intra,"interop_threads":a.interop},
        "main":base,"candidate":cand,"speedup_vs_main":speedup,
        "faster_than_main":bool(speedup is not None and speedup>1.0),
        "loss_delta_candidate_minus_main":cand["mean_last_10"]-base["mean_last_10"],
        "method":"same runner, same code, frozen tokenizer and prebuilt V5 pool, same seed and 50 steps per arm; run order alternated by parity",
    }
    out=OUT/"paired-result.json"
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("THREAD_TOURNAMENT_RESULT "+json.dumps(result,ensure_ascii=False),flush=True)

if __name__=="__main__": main()
