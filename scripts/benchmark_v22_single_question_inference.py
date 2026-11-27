"""Single-question, same-runner inference benchmark for LUMEN V2.2."""
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from tokenizers import Tokenizer
from Model import LumenV2, active_layers_for_step
from data import ROLE_IDS
from export_v22_torchscript import InferenceGraph

CHECKPOINT = "v22/lumen-v22-v5-fastpath-step-32500.pt"
TOKENIZER = "v22/tokenizer-v22-derived.json"
SYSTEM = "system_prompt_short.txt"
QUESTIONS = "oldrepo/data/freeform_identity_general_100.jsonl"
MAX_LEN = 512
MAX_NEW = 48
WARMUPS = 1
REPEATS = 5
THREADS = (1, 2, 4)
MODES = ("eager-recompute", "torchscript-recompute", "torchscript-kv-cache")


def read_question():
    for line in Path(QUESTIONS).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            prompt = str(row.get("prompt", "")).strip()
            if prompt:
                return row.get("id"), prompt
    raise RuntimeError("No usable question in the frozen 100-question set")


def build_prefix(tok, system, prompt):
    system_ids = tok.encode(system.strip(), add_special_tokens=False).ids
    user_ids = tok.encode(prompt, add_special_tokens=False).ids
    if any(x >= 8184 for x in system_ids + user_ids):
        raise RuntimeError("Text collides with reserved role token IDs")
    ids = (
        [ROLE_IDS["<|bos|>"], ROLE_IDS["<|system|>"]]
        + system_ids
        + [ROLE_IDS["<|end_system|>"], ROLE_IDS["<|user|>"]]
        + user_ids
        + [ROLE_IDS["<|end_user|>"], ROLE_IDS["<|assistant|>"]]
    )
    return ids[-MAX_LEN:]


def next_token(logits):
    return int(torch.argmax(logits[0, -1], dim=-1).item())


def should_stop(token, eos, pad):
    return token in (eos, pad, ROLE_IDS["<|end_assistant|>"])


def run_eager(model, prefix, active, eos, pad):
    ids = list(prefix)
    generated = []
    t0 = time.perf_counter()
    logits = model(torch.tensor([ids], dtype=torch.long),
                   active_layers=active, return_logits=True)[0]
    prefill = time.perf_counter() - t0
    decode = 0.0
    calls = 0
    while len(generated) < MAX_NEW:
        token = next_token(logits)
        if should_stop(token, eos, pad):
            break
        generated.append(token)
        if len(generated) >= MAX_NEW or len(ids) + 1 >= MAX_LEN:
            break
        ids.append(token)
        t = time.perf_counter()
        logits = model(torch.tensor([ids], dtype=torch.long),
                       active_layers=active, return_logits=True)[0]
        decode += time.perf_counter() - t
        calls += 1
    return generated, prefill, decode, calls


def run_script_recompute(graph, prefix, eos, pad):
    ids = list(prefix)
    generated = []
    t0 = time.perf_counter()
    logits = graph(torch.tensor([ids], dtype=torch.long))
    prefill = time.perf_counter() - t0
    decode = 0.0
    calls = 0
    while len(generated) < MAX_NEW:
        token = next_token(logits)
        if should_stop(token, eos, pad):
            break
        generated.append(token)
        if len(generated) >= MAX_NEW or len(ids) + 1 >= MAX_LEN:
            break
        ids.append(token)
        t = time.perf_counter()
        logits = graph(torch.tensor([ids], dtype=torch.long))
        decode += time.perf_counter() - t
        calls += 1
    return generated, prefill, decode, calls


def run_script_cache(graph, prefix, eos, pad, cache_shape):
    ids = list(prefix)
    generated = []
    ck = torch.zeros(cache_shape, dtype=torch.float32)
    cv = torch.zeros(cache_shape, dtype=torch.float32)
    t0 = time.perf_counter()
    logits, ck, cv = graph.forward_cached(
        torch.tensor([ids], dtype=torch.long), ck, cv, 0
    )
    prefill = time.perf_counter() - t0
    decode = 0.0
    calls = 0
    while len(generated) < MAX_NEW:
        token = next_token(logits)
        if should_stop(token, eos, pad):
            break
        generated.append(token)
        if len(generated) >= MAX_NEW or len(ids) + 1 >= MAX_LEN:
            break
        ids.append(token)
        t = time.perf_counter()
        logits, ck, cv = graph.forward_cached(
            torch.tensor([[token]], dtype=torch.long), ck, cv, len(ids) - 1
        )
        decode += time.perf_counter() - t
        calls += 1
    return generated, prefill, decode, calls


def median_run(fn):
    fn()  # warmup
    samples = []
    outputs = []
    for _ in range(REPEATS):
        out, prefill, decode, calls = fn()
        samples.append({
            "total_seconds": prefill + decode,
            "prefill_seconds": prefill,
            "decode_seconds": decode,
            "decode_calls": calls,
            "generated_tokens": len(out),
        })
        outputs.append(out)
    if any(x != outputs[0] for x in outputs[1:]):
        raise RuntimeError("Non-deterministic token IDs across repeats")
    return {
        "median_total_seconds": statistics.median(x["total_seconds"] for x in samples),
        "median_prefill_seconds": statistics.median(x["prefill_seconds"] for x in samples),
        "median_decode_seconds": statistics.median(x["decode_seconds"] for x in samples),
        "median_decode_calls": statistics.median(x["decode_calls"] for x in samples),
        "generated_tokens": len(outputs[0]),
        "token_ids": outputs[0],
        "samples": samples,
    }


def main():
    torch.set_num_interop_threads(1)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    step = int(state["step"])
    if step != 32500:
        raise RuntimeError(f"Expected step 32500 checkpoint, got {step}")
    model = LumenV2(vocab=8192, ctx=MAX_LEN, loss_mode="experimental")
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    active = active_layers_for_step(step, "v22", len(model.blocks))
    graph = torch.jit.freeze(
        torch.jit.script(InferenceGraph(model, active).eval()),
        preserved_attrs=["forward_cached"],
    ).eval()

    tok = Tokenizer.from_file(TOKENIZER)
    eos = tok.token_to_id("[EOS]")
    pad = tok.token_to_id("[PAD]")
    if eos is None or pad is None:
        raise RuntimeError("Tokenizer missing [EOS] or [PAD]")
    question_id, question = read_question()
    system = Path(SYSTEM).read_text(encoding="utf-8")
    prefix = build_prefix(tok, system, question)
    cache_shape = (
        len(model.blocks), 1, model.blocks[0].attn.n_head,
        MAX_LEN, model.blocks[0].attn.d_head
    )

    results = []
    with torch.inference_mode():
        for threads in THREADS:
            torch.set_num_threads(threads)
            fns = {
                "eager-recompute": lambda: run_eager(model, prefix, active, eos, pad),
                "torchscript-recompute": lambda: run_script_recompute(graph, prefix, eos, pad),
                "torchscript-kv-cache": lambda: run_script_cache(
                    graph, prefix, eos, pad, cache_shape
                ),
            }
            group = {}
            for mode in MODES:
                group[mode] = median_run(fns[mode])
            reference = group["eager-recompute"]["token_ids"]
            for mode, data in group.items():
                data["token_ids_identical_to_eager"] = data["token_ids"] == reference
                if not data["token_ids_identical_to_eager"]:
                    raise RuntimeError(f"Output parity failure: {mode}, threads={threads}")
                data.pop("token_ids")
                data.pop("samples")
                results.append({"threads_intra": threads, "mode": mode, **data})

    checkpoint_hash = hashlib.sha256(Path(CHECKPOINT).read_bytes()).hexdigest()
    report = {
        "checkpoint_step": step,
        "checkpoint_sha256": checkpoint_hash,
        "tokenizer_sha256": hashlib.sha256(Path(TOKENIZER).read_bytes()).hexdigest(),
        "question_id": question_id,
        "question": question,
        "question_token_count": len(prefix),
        "max_new_tokens": MAX_NEW,
        "warmups_per_cell": WARMUPS,
        "timed_repeats_per_cell": REPEATS,
        "cpu": platform.processor(),
        "torch_version": torch.__version__,
        "active_layers": active,
        "interop_threads": 1,
        "results": results,
    }
    # Add within-thread speedups against eager recomputation.
    for row in report["results"]:
        baseline = next(
            x for x in report["results"]
            if x["threads_intra"] == row["threads_intra"]
            and x["mode"] == "eager-recompute"
        )
        row["speedup_vs_eager"] = (
            baseline["median_total_seconds"] / row["median_total_seconds"]
        )
        n = max(0, row["generated_tokens"] - 1)
        row["decode_tokens_per_second"] = (
            n / row["median_decode_seconds"] if n and row["median_decode_seconds"] > 0 else None
        )
    Path("inference-single-question-benchmark.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
