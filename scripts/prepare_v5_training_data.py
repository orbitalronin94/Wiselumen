"""One-time, verified preprocessing for the LUMEN V2.2 V5 training corpus."""
import json
import random
import time
import zipfile
from pathlib import Path

import torch
from tokenizers import Tokenizer

from scripts.train_v22_v5_blocks import (
    SEED, V5_SHA256, V4_SHA256, TOKENIZER_SHA256, MAX_LEN,
    load_v5_bundle, derive_tokenizer, verify_corpus_and_reservoir,
)
from scripts.v5_sampling import format_lore_row
from data import collate_chat, format_v4_row, validate_tokenizer

V5_ARCHIVE = Path("artifacts/wise-ronin-corpus-v5.0.0.zip")
OUTPUT = Path("artifacts/prepared-v5-training-data.pt")


def timed(label, fn):
    started = time.perf_counter()
    result = fn()
    print(f"PREP_PHASE {label} seconds={time.perf_counter() - started:.3f}", flush=True)
    return result


def main():
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    if not V5_ARCHIVE.is_file():
        raise FileNotFoundError(V5_ARCHIVE)

    v4_path = artifacts / "corpus-v4-nested.zip"
    v5_manifest, lore_rows, conflict_rows = timed(
        "verify_and_extract_v5",
        lambda: load_v5_bundle(V5_ARCHIVE, v4_path),
    )

    def load_tokenizer():
        with zipfile.ZipFile(v4_path) as zf:
            manifest = json.loads(zf.read("corpus/manifest.json"))
            if manifest.get("version") != "v4.0.0":
                raise RuntimeError("Unexpected V4 corpus version")
            if manifest.get("status") != "FROZEN_TRAINABLE" or manifest.get("training_authorized") is not True:
                raise RuntimeError("V4 manifest does not authorize training")
            tokenizer_bytes = zf.read("corpus/tokenizer.json")
        tokenizer, original_sha, derived_sha = derive_tokenizer(
            tokenizer_bytes, artifacts / "tokenizer-v22-derived.json"
        )
        return manifest, tokenizer, original_sha, derived_sha

    manifest, tokenizer, original_sha, derived_sha = timed("derive_tokenizer", load_tokenizer)
    metadata = validate_tokenizer(tokenizer)
    print(f"PREP_TOKENIZER {metadata}", flush=True)

    reservoir = timed(
        "verify_all_v4_shards_and_build_reservoir",
        lambda: verify_corpus_and_reservoir(v4_path, manifest, random.Random(SEED)),
    )
    system_prompt = Path("system_prompt_short.txt").read_text(encoding="utf-8").strip()

    def encode_v4():
        rows, rejected = [], 0
        for index, row in enumerate(reservoir, 1):
            try:
                rows.append(format_v4_row(row, tokenizer, system_prompt, max_len=MAX_LEN))
            except ValueError as exc:
                if "needs" in str(exc) or "max_len" in str(exc):
                    rejected += 1
                    continue
                raise
            if index % 256 == 0:
                print(f"PREP_ENCODE v4={index}/{len(reservoir)} accepted={len(rows)}", flush=True)
        if len(rows) < 100:
            raise RuntimeError(f"Only {len(rows)} V4 rows fit; rejected={rejected}")
        return rows, rejected

    encoded_v4, rejected_v4 = timed("tokenize_v4_reservoir", encode_v4)

    def encode_lore():
        rows, rejected = [], 0
        for index, row in enumerate(lore_rows, 1):
            try:
                rows.append(format_lore_row(row, tokenizer, system_prompt, max_len=MAX_LEN))
            except ValueError as exc:
                if "needs" in str(exc) or "max_len" in str(exc):
                    rejected += 1
                    continue
                raise
            if index % 64 == 0:
                print(f"PREP_ENCODE lore={index}/{len(lore_rows)} accepted={len(rows)}", flush=True)
        if len(rows) < 20:
            raise RuntimeError(f"Only {len(rows)} lore rows fit; rejected={rejected}")
        return rows, rejected

    encoded_lore, rejected_lore = timed("tokenize_lore", encode_lore)

    def encode_conflicts():
        rows = [format_v4_row(row, tokenizer, system_prompt, max_len=MAX_LEN) for row in conflict_rows]
        if len(rows) != 10:
            raise RuntimeError(f"Expected 10 conflicts, got {len(rows)}")
        return rows

    encoded_conflicts = timed("tokenize_conflicts", encode_conflicts)

    payload = {
        "format": "lumen_v22_v5_prepared_v1",
        "corpus_version": v5_manifest["version"],
        "corpus_v5_sha256": V5_SHA256,
        "corpus_v4_sha256": V4_SHA256,
        "tokenizer_original_sha256": original_sha,
        "tokenizer_derived_sha256": derived_sha,
        "pad_id": int(metadata["pad_id"]),
        "v4": encoded_v4,
        "lore": encoded_lore,
        "conflict": encoded_conflicts,
        "rejected_v4_long": rejected_v4,
        "rejected_lore_long": rejected_lore,
        "reservoir_rows": len(reservoir),
        "lore_rows": len(lore_rows),
        "conflict_rows": len(conflict_rows),
        "sampling_shares": {"v4": 0.675, "lore": 0.075, "synthetic_conflict": 0.25},
        "seed": SEED,
    }
    timed("serialize_prepared_dataset", lambda: torch.save(payload, OUTPUT))
    print(
        f"PREPARED_OK path={OUTPUT} bytes={OUTPUT.stat().st_size} "
        f"v4={len(encoded_v4)} lore={len(encoded_lore)} conflicts={len(encoded_conflicts)} "
        f"v5_sha256={V5_SHA256} tokenizer_sha256={derived_sha}",
        flush=True,
    )


if __name__ == "__main__":
    main()
