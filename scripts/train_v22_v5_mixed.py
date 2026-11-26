"""Continue LUMEN V2.2 using the frozen V5 67.5/7.5/25 mixed corpus."""
import hashlib
import json
import os
import random
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tokenizers import AddedToken, Tokenizer

from Model import LumenV2
from Train import EMA, configure_cpu_runtime, get_optimizer, save_checkpoint, train_steps
from data import ROLE_IDS, collate_chat, format_v4_row, validate_tokenizer
from scripts.v5_sampling import format_lore_row, sample_v5_batches

SEED = 20261126
V5_SHA256 = "44bc36457a58a1743f5f1d7f3374ee8850d381177f85b52a89930aaaa2869dff"
V4_SHA256 = "535ae5a1c067f30a93e1ad20e0a8b7e332df5b14558d1dfac7a3e44193de3e9a"
TOKENIZER_SHA256 = "d7869c11809a28085d5156638086c4833a1dcbacfda4a60b69ec71e26ed3fefe"
CONTROL_TOKENS = list(ROLE_IDS)
START_STEP = int(os.environ.get("START_STEP", "10000"))
STEPS = int(os.environ.get("STEPS", "10000"))
TOTAL_STEPS = START_STEP + STEPS
PARENT_CHECKPOINT = Path(os.environ.get("PARENT_CHECKPOINT", f"artifacts/lumen-v22-step-{START_STEP}.pt"))
V5_ARCHIVE = Path(os.environ.get("V5_ARCHIVE", "artifacts/wise-ronin-corpus-v5.0.0.zip"))
BATCH_SIZE = 4
MAX_LEN = 512
RESERVOIR_SIZE = 2048


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_v5_bundle(path, v4_path):
    if not path.exists():
        raise FileNotFoundError(f"Missing V5 archive: {path}")
    actual = sha256_file(path)
    print(f"corpus_v5_sha256={actual}", flush=True)
    if actual != V5_SHA256:
        raise RuntimeError(f"V5 ZIP SHA256 mismatch: {actual}")
    with zipfile.ZipFile(path) as outer:
        v5_manifest = json.loads(outer.read("corpus/v5/manifest.json"))
        if v5_manifest.get("version") != "v5.0.0" or not v5_manifest.get("training_authorized"):
            raise RuntimeError("V5 manifest does not authorize training")
        v4_bytes = outer.read("corpus/v4/wise-ronin-corpus-v4.0.0.zip")
        if hashlib.sha256(v4_bytes).hexdigest() != V4_SHA256:
            raise RuntimeError("Nested V4 archive SHA256 mismatch")
        lore_rows = [json.loads(line) for line in outer.read("lore/lore_seed_v0.6.jsonl").decode("utf-8").splitlines() if line.strip()]
        conflict_rows = [json.loads(line) for line in outer.read("synthetic/system_conflicts.jsonl").decode("utf-8").splitlines() if line.strip()]
    v4_path.parent.mkdir(parents=True, exist_ok=True)
    v4_path.write_bytes(v4_bytes)
    if len(lore_rows) != 289 or len(conflict_rows) != 10:
        raise RuntimeError(f"V5 pool size mismatch: lore={len(lore_rows)} conflicts={len(conflict_rows)}")
    return v5_manifest, lore_rows, conflict_rows


def derive_tokenizer(original_bytes, output_path):
    original_sha = hashlib.sha256(original_bytes).hexdigest()
    if original_sha != TOKENIZER_SHA256:
        raise RuntimeError(f"Frozen tokenizer SHA256 mismatch: {original_sha}")

    source_path = Path("artifacts/tokenizer-v4-original.json")
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(original_bytes)
    tokenizer = Tokenizer.from_file(str(source_path))
    original_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if original_size != 4096:
        raise RuntimeError(f"Expected frozen V4 vocab 4096, got {original_size}")
    if tokenizer.token_to_id("[PAD]") != 0 or tokenizer.token_to_id("[EOS]") != 3:
        raise RuntimeError("Frozen V4 [PAD]/[EOS] IDs differ from the verified layout")

    # Keep every original V4 token ID unchanged. Fill the unused ID interval
    # with inert special tokens so the eight requested controls land at 8184..8191.
    filler = [
        AddedToken(f"<|lumen_reserved_{index:04d}|>", special=True)
        for index in range(8184 - original_size)
    ]
    added_fillers = tokenizer.add_special_tokens(filler)
    if added_fillers != len(filler):
        raise RuntimeError(f"Added {added_fillers}/{len(filler)} reserved filler tokens")
    controls = [AddedToken(token, special=True) for token in CONTROL_TOKENS]
    added_controls = tokenizer.add_special_tokens(controls)
    if added_controls != len(controls):
        raise RuntimeError(f"Added {added_controls}/{len(controls)} control tokens")

    actual_size = tokenizer.get_vocab_size(with_added_tokens=True)
    if actual_size != 8192:
        raise RuntimeError(f"Derived tokenizer vocab must be 8192, got {actual_size}")
    for token, expected_id in ROLE_IDS.items():
        actual_id = tokenizer.token_to_id(token)
        if actual_id != expected_id:
            raise RuntimeError(f"{token}: expected ID {expected_id}, got {actual_id}")
    if tokenizer.token_to_id("[PAD]") != 0 or tokenizer.token_to_id("[EOS]") != 3:
        raise RuntimeError("Derived tokenizer changed frozen PAD/EOS IDs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    derived_sha = sha256_file(output_path)
    print(
        f"tokenizer_original_vocab={original_size} derived_vocab={actual_size} "
        f"original_sha256={original_sha} derived_sha256={derived_sha}",
        flush=True,
    )
    return tokenizer, original_sha, derived_sha


def verify_corpus_and_reservoir(archive_path, manifest, reservoir_rng):
    reservoir = []
    seen = 0
    total_rows = 0
    for shard_index, shard_meta in enumerate(manifest["shards"]):
        member = "corpus/" + shard_meta["file"]
        digest = hashlib.sha256()
        shard_rows = 0
        with zipfile.ZipFile(archive_path) as zf:
            with zf.open(member) as handle:
                for raw_line in handle:
                    digest.update(raw_line)
                    shard_rows += 1
                    total_rows += 1
                    if not raw_line.strip():
                        continue
                    row = json.loads(raw_line)
                    if row.get("split") == "holdout" or row.get("source") == "holdout_500.jsonl":
                        continue
                    if row.get("language") != "es":
                        continue
                    if not isinstance(row.get("prompt"), str) or not isinstance(row.get("completion"), str):
                        continue
                    seen += 1
                    if len(reservoir) < RESERVOIR_SIZE:
                        reservoir.append(row)
                    else:
                        slot = reservoir_rng.randrange(seen)
                        if slot < RESERVOIR_SIZE:
                            reservoir[slot] = row

        actual_bytes = zf.getinfo(member).file_size
        actual_sha = digest.hexdigest()
        if shard_rows != shard_meta["rows"]:
            raise RuntimeError(
                f"{member}: row count mismatch {shard_rows} != {shard_meta['rows']}"
            )
        if actual_bytes != shard_meta["bytes"]:
            raise RuntimeError(
                f"{member}: byte count mismatch {actual_bytes} != {shard_meta['bytes']}"
            )
        if actual_sha != shard_meta["sha256"]:
            raise RuntimeError(f"{member}: SHA256 mismatch {actual_sha}")
        print(
            f"verified_shard={shard_index + 1}/{len(manifest['shards'])} "
            f"rows={shard_rows} sha256={actual_sha}",
            flush=True,
        )

    if total_rows != manifest["total_rows"]:
        raise RuntimeError(
            f"Total row count mismatch {total_rows} != {manifest['total_rows']}"
        )
    print(
        f"verified_rows={total_rows} eligible_spanish_rows={seen} "
        f"reservoir={len(reservoir)}",
        flush=True,
    )
    if len(reservoir) < 100:
        raise RuntimeError("Not enough eligible Spanish training rows")
    return reservoir


def load_conflicts(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    interop_threads = int(os.environ.get("TORCH_INTEROP_THREADS", "2"))
    print("cpu_runtime", configure_cpu_runtime(threads=threads, interop_threads=interop_threads), flush=True)

    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    archive_path = artifacts / "corpus-v4-nested.zip"
    v5_manifest, lore_rows, conflict_rows = load_v5_bundle(V5_ARCHIVE, archive_path)

    with zipfile.ZipFile(archive_path) as zf:
        manifest = json.loads(zf.read("corpus/manifest.json"))
        if manifest.get("version") != "v4.0.0":
            raise RuntimeError("Unexpected corpus version")
        if manifest.get("status") != "FROZEN_TRAINABLE" or manifest.get("training_authorized") is not True:
            raise RuntimeError("Corpus manifest does not authorize training")
        tokenizer_bytes = zf.read("corpus/tokenizer.json")
        tokenizer, original_tokenizer_sha, derived_tokenizer_sha = derive_tokenizer(
            tokenizer_bytes, artifacts / "tokenizer-v22-derived.json"
        )

    metadata = validate_tokenizer(tokenizer)
    print("tokenizer_validation", metadata, flush=True)
    reservoir = verify_corpus_and_reservoir(
        archive_path, manifest, random.Random(SEED)
    )

    system_prompt = Path("system_prompt_short.txt").read_text(encoding="utf-8").strip()
    encoded_corpus = []
    rejected_long = 0
    for row in reservoir:
        try:
            encoded_corpus.append(
                format_v4_row(row, tokenizer, system_prompt, max_len=MAX_LEN)
            )
        except ValueError as exc:
            if "needs" in str(exc) or "max_len" in str(exc):
                rejected_long += 1
                continue
            raise
    if len(encoded_corpus) < 100:
        raise RuntimeError(
            f"Only {len(encoded_corpus)} reservoir rows fit the chat format; "
            "need at least 100. Rejected long rows: {rejected_long}"
        )

    encoded_lore = [
        format_lore_row(row, tokenizer, system_prompt, max_len=MAX_LEN)
        for row in lore_rows
    ]
    if len(encoded_lore) != 289:
        raise RuntimeError(f"Expected 289 curated lore records, got {len(encoded_lore)}")

    encoded_conflicts = [
        format_v4_row(row, tokenizer, system_prompt, max_len=MAX_LEN)
        for row in conflict_rows
    ]
    if len(encoded_conflicts) != 10:
        raise RuntimeError(f"Expected 10 conflict examples, got {len(encoded_conflicts)}")

    batch_rng = random.Random(SEED + START_STEP + 1)
    mixed_batches = sample_v5_batches(
        encoded_corpus, encoded_lore, encoded_conflicts,
        steps=STEPS, rng=batch_rng, batch_size=BATCH_SIZE,
    )
    batches = [
        collate_chat([sample for _kind, sample in batch], pad_id=metadata["pad_id"])
        for batch in mixed_batches
    ]

    model = LumenV2(vocab=8192, ctx=MAX_LEN, loss_mode="experimental")
    optimizer = get_optimizer(
        model, slow_lr=5e-4, fast_lr=2e-3, weight_decay=0.01, prefer_8bit=False
    )
    ema = EMA(model, decay=0.99)
    if not PARENT_CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing parent checkpoint: {PARENT_CHECKPOINT}")
    try:
        payload = torch.load(PARENT_CHECKPOINT, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(PARENT_CHECKPOINT, map_location="cpu")
    parent_step = int(payload.get("step", -1))
    if parent_step != START_STEP:
        raise RuntimeError(f"Expected parent step {START_STEP}, got {parent_step}")
    model.load_state_dict(payload["model"], strict=True)
    if "optimizer" not in payload or "ema" not in payload:
        raise RuntimeError("Parent checkpoint lacks optimizer or EMA state")
    optimizer.load_state_dict(payload["optimizer"])
    ema.decay = float(payload["ema"]["decay"])
    ema.shadow = payload["ema"]["shadow"]
    print(
        f"RESUME_OK parent_step={parent_step} batch_size={BATCH_SIZE} "
        f"additional_steps={STEPS} target_step={TOTAL_STEPS} context={MAX_LEN} "
        f"active_layers=4/8 v4_share=0.675 lore_share=0.075 synthetic_conflict_share=0.250",
        flush=True,
    )

    started = time.perf_counter()
    result = train_steps(
        model,
        batches,
        steps=STEPS,
        optimizer=optimizer,
        device="cpu",
        warmup=10,
        total_steps=TOTAL_STEPS,
        start_step=START_STEP,
        grad_clip=1.0,
        ema=ema,
        log_every=10,
        amp_dtype=None,
        mode="v22",
        ema_every=10,
        clip_every=5,
    )
    elapsed = time.perf_counter() - started
    checkpoint = save_checkpoint(
        f"artifacts/lumen-v22-v5-step-{TOTAL_STEPS}.pt",
        model,
        optimizer=optimizer,
        ema=ema,
        step=TOTAL_STEPS,
        extra={
            "seed": SEED,
            "corpus_version": v5_manifest["version"],
            "corpus_v5_sha256": V5_SHA256,
            "corpus_v4_sha256": V4_SHA256,
            "sampling_shares": {"v4": 0.675, "lore": 0.075, "synthetic_conflict": 0.25},
            "tokenizer_original_sha256": original_tokenizer_sha,
            "tokenizer_derived_sha256": derived_tokenizer_sha,
            "system_prompt": "system_prompt_short.txt",
            "batch_size": BATCH_SIZE,
            "max_len": MAX_LEN,
            "mode": "v22",
            "loss_mode": "experimental",
        },
    )
    metrics = {
        "status": "completed",
        "steps": TOTAL_STEPS,
        "corpus_version": v5_manifest["version"],
        "sampling_shares": {"v4": 0.675, "lore": 0.075, "synthetic_conflict": 0.25},
        "lore_rows_encoded": len(encoded_lore),
        "rejected_lore_long_rows": rejected_lore_long,
        "additional_steps": STEPS,
        "parent_step": START_STEP,
        "last_loss": result["last_loss"],
        "mean_first_10": sum(result["losses"][:10]) / 10,
        "mean_last_10": sum(result["losses"][-10:]) / 10,
        "elapsed_seconds": elapsed,
        "steps_per_second": STEPS / elapsed,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "max_len": MAX_LEN,
        "reservoir_rows_encoded": len(encoded_corpus),
        "rejected_long_rows": rejected_long,
        "synthetic_conflict_rows": len(encoded_conflicts),
        "corpus_v5_sha256": V5_SHA256,
        "corpus_v4_sha256": V4_SHA256,
        "tokenizer_original_sha256": original_tokenizer_sha,
        "tokenizer_derived_sha256": derived_tokenizer_sha,
        "checkpoint": checkpoint,
        "model_parameters": model.num_params(),
        "active_layers": 4,
        "parent_checkpoint": str(PARENT_CHECKPOINT),
        "loss_mode": "experimental",
    }
    Path(f"artifacts/metrics-v5-step-{TOTAL_STEPS}.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("TRAIN_RESULT " + json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
