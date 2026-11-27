#!/usr/bin/env python3
"""Train CULTISTLUMEN-V1 from scratch on V5: natural file-size mix + 10% full Marper lore."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import torch
from tokenizers import AddedToken, Tokenizer

from scripts.cultistlumen_v1 import CultistLumenV1

SEED = 20261127
STEPS = int(os.environ.get("STEPS", "1000"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
CTX = int(os.environ.get("CTX", "128"))
WARMUP = int(os.environ.get("WARMUP", "200"))
THREADS = int(os.environ.get("OMP_NUM_THREADS", "4"))
MAX_CHUNKS_PER_FILE = int(os.environ.get("MAX_CHUNKS_PER_FILE", "2048"))
V5_PATH = Path(os.environ.get("V5_ARCHIVE", "artifacts/wise-ronin-corpus-v5.0.0.zip"))
OUT = Path("artifacts")
MARPER_FILES = [
    "lore/Marper-Codice-Juntado (1).txt",
    "lore/lore_seed_v0.6.jsonl",
    "lore/palimpsesto_topic_map_v0.3.jsonl",
    "lore/palimpsesto_index_v0.1_part1.jsonl",
    "lore/palimpsesto_index_v0.1_part2.jsonl",
    "lore/palimpsesto_index_v0.1_part3.jsonl",
    "lore/palimpsesto_index_v0.1_part4.jsonl",
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_tokenizer(zf: zipfile.ZipFile) -> tuple[Tokenizer, str, str]:
    original = zf.read("corpus/v4/tokenizer.json")
    original_sha = sha256(original)
    path = OUT / "cultist-tokenizer-original.json"
    path.write_bytes(original)
    tokenizer = Tokenizer.from_file(str(path))
    original_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    if original_vocab != 4096:
        raise RuntimeError(f"Expected V4 tokenizer vocab=4096, got {original_vocab}")
    if tokenizer.token_to_id("[PAD]") != 0 or tokenizer.token_to_id("[EOS]") != 3:
        raise RuntimeError("Unexpected V4 PAD/EOS IDs")
    filler = [
        AddedToken(f"<|cultist_reserved_{i:04d}|>", special=True)
        for i in range(8192 - original_vocab)
    ]
    added = tokenizer.add_special_tokens(filler)
    if added != len(filler) or tokenizer.get_vocab_size(with_added_tokens=True) != 8192:
        raise RuntimeError("Could not extend frozen V4 tokenizer to CULTIST vocab=8192")
    derived = OUT / "cultist-tokenizer-v1-8192.json"
    tokenizer.save(str(derived))
    return tokenizer, original_sha, sha256(derived.read_bytes())


def make_chunks(tokenizer: Tokenizer, text: str, ctx: int, cap: int, reservoir_rng: random.Random):
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    width = ctx + 1
    chunks = []
    total = 0
    for start in range(0, len(ids) - width + 1, width):
        chunk = ids[start:start + width]
        total += 1
        if len(chunks) < cap:
            chunks.append(chunk)
        else:
            slot = reservoir_rng.randrange(total)
            if slot < cap:
                chunks[slot] = chunk
    return chunks, total


def build_file_pools(zf, manifest, tokenizer, nested_bytes):
    pools = {"world": {}, "marper": {}}
    weights = {"world": {}, "marper": {}}
    rng = random.Random(SEED + 91)
    import io
    nested = zipfile.ZipFile(io.BytesIO(nested_bytes))

    # Natural V4 training files. Holdout rows are excluded before tokenization.
    for meta in manifest["shards"]:
        member = "corpus/" + meta["file"]
        chunks = []
        buffer = []
        seen_rows = 0
        seen_chunks = 0
        with nested.open(member) as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                if row.get("split") == "holdout" or row.get("source") == "holdout_500.jsonl":
                    continue
                prompt = row.get("prompt")
                completion = row.get("completion")
                if not isinstance(prompt, str) or not isinstance(completion, str):
                    continue
                ids = tokenizer.encode(prompt.strip() + "\\n" + completion.strip(), add_special_tokens=False).ids
                buffer.extend(ids)
                buffer.append(3)  # frozen V4 EOS
                seen_rows += 1
                while len(buffer) >= CTX + 1:
                    chunk = buffer[:CTX + 1]
                    del buffer[:CTX + 1]
                    seen_chunks += 1
                    if len(chunks) < MAX_CHUNKS_PER_FILE:
                        chunks.append(chunk)
                    else:
                        slot = rng.randrange(seen_chunks)
                        if slot < MAX_CHUNKS_PER_FILE:
                            chunks[slot] = chunk
        if not chunks:
            raise RuntimeError(f"No usable training chunks in V4 shard {member}")
        pools["world"][member] = chunks
        weights["world"][member] = int(meta["bytes"])
        print(f"POOL world file={member} bytes={weights['world'][member]} rows={seen_rows} chunks={len(chunks)}", flush=True)

    # All seven Marper/Codex sources form one 10% lore pool.
    for member in MARPER_FILES:
        raw = zf.read(member)
        chunks, total = make_chunks(tokenizer, raw.decode("utf-8"), CTX, MAX_CHUNKS_PER_FILE, rng)
        if not chunks:
            raise RuntimeError(f"No usable token chunks in Marper source {member}")
        pools["marper"][member] = chunks
        weights["marper"][member] = len(raw)
        print(f"POOL marper file={member} bytes={len(raw)} chunks={len(chunks)} total_chunks={total}", flush=True)

    nested.close()
    if not pools["world"] or not pools["marper"]:
        raise RuntimeError("Both world and full Marper pools must be non-empty")
    return pools, weights

def weighted_file_sample(pool: dict, weights: dict, rng: random.Random):
    names = list(pool)
    total = sum(weights[name] for name in names)
    point = rng.randrange(total)
    for name in names:
        point -= weights[name]
        if point < 0:
            return rng.choice(pool[name])
    return rng.choice(pool[names[-1]])


def build_step_sources(steps: int, batch_size: int, rng: random.Random):
    if steps % 10 != 0:
        raise ValueError("STEPS must be a multiple of 10 for exact 10% Marper exposure")
    schedule = []
    for _ in range(steps // 10):
        tags = ["world"] * (batch_size * 9) + ["marper"] * batch_size
        rng.shuffle(tags)
        schedule.extend([tags[i:i + batch_size] for i in range(0, len(tags), batch_size)])
    return schedule


def main():
    if STEPS <= 0 or BATCH_SIZE <= 0 or CTX <= 0:
        raise ValueError("STEPS, BATCH_SIZE and CTX must be positive")
    if STEPS % 10:
        raise ValueError("STEPS must be divisible by 10")
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(SEED)
    torch.manual_seed(SEED)
    rng = random.Random(SEED + 1000)
    print(f"CULTISTLUMEN_V1_CONFIG steps={STEPS} batch={BATCH_SIZE} ctx={CTX} warmup={WARMUP} threads={THREADS}", flush=True)

    with zipfile.ZipFile(V5_PATH) as outer:
        v5_manifest = json.loads(outer.read("corpus/v5/manifest.json"))
        if v5_manifest.get("version") != "v5.0.0" or not v5_manifest.get("training_authorized"):
            raise RuntimeError("V5 manifest is not authorized")
        nested_bytes = outer.read("corpus/v4/wise-ronin-corpus-v4.0.0.zip")
        if sha256(nested_bytes) != v5_manifest["v4"]["zip_sha256"]:
            raise RuntimeError("Nested V4 archive checksum mismatch")
        import io
        with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
            manifest = json.loads(nested.read("corpus/manifest.json"))
            if manifest.get("version") != "v4.0.0" or manifest.get("status") != "FROZEN_TRAINABLE" or manifest.get("training_authorized") is not True:
                raise RuntimeError("V4 manifest is not authorized")
            tokenizer, tokenizer_original_sha, tokenizer_derived_sha = load_tokenizer(nested)
            pools, weights = build_file_pools(outer, manifest, tokenizer, nested_bytes)

    model = CultistLumenV1(vocab=8192, d_model=512, d_ff=1024, n_experts=64, ctx=CTX)
    optimizer = torch.optim.AdamW([
        {"params": list(model.moe.parameters()), "lr": 5e-4, "base_lr": 5e-4, "weight_decay": 0.01},
        {"params": list(model.mix.parameters()) + list(model.tok.parameters()) +
                   list(model.norm1.parameters()) + list(model.norm2.parameters()) +
                   list(model.norm.parameters()), "lr": 2e-3, "base_lr": 2e-3, "weight_decay": 0.01},
    ], betas=(0.9, 0.95), eps=1e-8)
    print(f"MODEL total_params={model.num_params():,} active_per_token={model.active_params_per_token():,} experts={model.moe.n_experts}", flush=True)

    schedule = build_step_sources(STEPS, BATCH_SIZE, rng)
    started = time.perf_counter()
    losses = []
    model.train()
    for step, tags in enumerate(schedule, 1):
        if step <= WARMUP:
            scale = step / max(1, WARMUP)
        else:
            progress = (step - WARMUP) / max(1, STEPS - WARMUP)
            scale = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        for group in optimizer.param_groups:
            group["lr"] = group["base_lr"] * scale

        batch = []
        for tag in tags:
            chunk = weighted_file_sample(pools[tag], weights[tag], rng)
            batch.append(chunk)
        x = torch.tensor([sample[:-1] for sample in batch], dtype=torch.long)
        y = torch.tensor([sample[1:] for sample in batch], dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, targets=y)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        value = float(loss.detach())
        losses.append(value)
        if step == 1 or step % 10 == 0 or step == STEPS:
            elapsed = time.perf_counter() - started
            used = int((model.moe.expert_usage > 0).sum().item())
            print(f"STEP {step}/{STEPS} loss={value:.6f} lr_moe={optimizer.param_groups[0]['lr']:.7g} lr_fast={optimizer.param_groups[1]['lr']:.7g} steps_per_sec={step/max(elapsed,1e-9):.3f} experts_used={used}/64", flush=True)

    elapsed = time.perf_counter() - started
    checkpoint_path = OUT / f"cultistlumen-v1-step-{STEPS}.pt"
    torch.save({
        "model": model.state_dict(),
        "step": STEPS,
        "architecture": "CULTISTLUMEN-V1",
        "vocab": 8192,
        "ctx": CTX,
        "seed": SEED,
        "corpus_version": "v5.0.0",
        "tokenizer_original_sha256": tokenizer_original_sha,
        "tokenizer_derived_sha256": tokenizer_derived_sha,
        "sampling": {"world": 0.90, "full_marper_lore": 0.10, "conflicts": 0.0, "weighting": "uncompressed source-file byte size within each pool"},
    }, checkpoint_path)
    metrics = {
        "status": "completed",
        "architecture": "CULTISTLUMEN-V1",
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "context": CTX,
        "last_loss": losses[-1],
        "mean_first_10": sum(losses[:10]) / min(10, len(losses)),
        "mean_last_10": sum(losses[-10:]) / min(10, len(losses)),
        "elapsed_seconds": elapsed,
        "steps_per_second": STEPS / elapsed,
        "model_parameters": model.num_params(),
        "active_parameters_per_token": model.active_params_per_token(),
        "experts_used": int((model.moe.expert_usage > 0).sum().item()),
        "expert_usage": model.moe.expert_usage.tolist(),
        "sampling": {"world": 0.90, "full_marper_lore": 0.10, "conflicts": 0.0, "weighting": "uncompressed source-file byte size within each pool"},
        "file_weights_bytes": weights,
        "tokenizer_original_sha256": tokenizer_original_sha,
        "tokenizer_derived_sha256": tokenizer_derived_sha,
        "checkpoint": str(checkpoint_path),
    }
    (OUT / f"cultistlumen-v1-metrics-step-{STEPS}.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("CULTISTLUMEN_RESULT " + json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
