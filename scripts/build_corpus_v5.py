#!/usr/bin/env python3
"""Build a reproducible V5 bundle: frozen V4 + curated Concordia/Marper sources.

The 16,848 source revelations are preserved as provenance/index material. Training
samples come from the 289 curated v0.6 records, avoiding 400x repetition of topics.
"""
from __future__ import annotations
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
V4_URL = "https://github.com/orbitalronin94/WISE-RONIN/releases/download/corpus-v4.0.0-public-20261125/wise-ronin-corpus-v4.0.0.zip"
V4_SHA256 = "535ae5a1c067f30a93e1ad20e0a8b7e332df5b14558d1dfac7a3e44193de3e9a"
LORE_BASE = "https://raw.githubusercontent.com/orbitalronin94/Wiselumen/lore/archivo-de-las-contradicciones/"
SOURCES = {
    "lore/Marper-Codice-Juntado (1).txt": ("Marper-Codice-Juntado (1).txt", "b0eb703c5237d1d461321a30a89525e154f0219d", 20401800),
    "lore/lore_seed_v0.6.jsonl": ("data/lore_seed_v0.6.jsonl", "c237f6db97c63cf77e6d080e87b36d990f22fddd", 402427),
    "lore/palimpsesto_topic_map_v0.3.jsonl": ("data/palimpsesto_topic_map_v0.3.jsonl", "68460cddabbe0f8fd2ec3f613b316f27b1f53031", 34238),
    "lore/palimpsesto_index_v0.1_part1.jsonl": ("data/palimpsesto_index_v0.1_part1.jsonl", "055715c642bc71a5dfccb89737de830f6b21553b", 1864031),
    "lore/palimpsesto_index_v0.1_part2.jsonl": ("data/palimpsesto_index_v0.1_part2.jsonl", "0a29c8931193e85ce049a7081e9ff1dd30bd9a03", 1867801),
    "lore/palimpsesto_index_v0.1_part3.jsonl": ("data/palimpsesto_index_v0.1_part3.jsonl", "d0898b68191407fe5148fd3f2b2b28cfd871a8c8", 1870292),
    "lore/palimpsesto_index_v0.1_part4.jsonl": ("data/palimpsesto_index_v0.1_part4.jsonl", "bfae3f9cd2451bf7d8742f50e6ea1176a162de9a", 1880221),
}
CONFLICT_URL = "https://raw.githubusercontent.com/orbitalronin94/Wiselumen/main/data/system_conflicts.jsonl"
CONFLICT_GIT_BLOB = "e5febcfedcd18cbf6f3249ebf9a3f0d216706bac"
SAMPLING = {
    "version": "v5.0.0",
    "unit": "training examples, not bytes or corpus rows",
    "batch_size": 4,
    "cycle_steps": 10,
    "cycle_examples": 40,
    "shares": {"v4_factual": 0.675, "marper_concordia_seed": 0.075, "synthetic_conflicts": 0.25},
    "exact_cycle_counts": {"v4_factual": 27, "marper_concordia_seed": 3, "synthetic_conflicts": 10},
    "holdout_policy": "The V4 archive is embedded byte-for-byte; holdout rows remain excluded by the existing V4 formatter.",
    "lore_policy": "Train from the 289 curated v0.6 records (fact/lore/bridge); preserve the 16,848-revelation source and indexes as provenance, not duplicated training rows.",
}

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()

def download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Wiselumen-Corpus-Builder/5.0"})
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read()

def verify_source(label: str, data: bytes, expected_blob: str, expected_size: int) -> dict:
    actual_blob = git_blob_sha(data)
    if len(data) != expected_size:
        raise RuntimeError(f"{label}: byte count {len(data)} != {expected_size}")
    if actual_blob != expected_blob:
        raise RuntimeError(f"{label}: Git blob SHA mismatch {actual_blob} != {expected_blob}")
    return {"bytes": len(data), "git_blob_sha1": actual_blob, "sha256": sha256(data)}

def jsonl_rows(data: bytes, label: str) -> list[dict]:
    rows = []
    for line_no, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{label}:{line_no}: {exc}") from exc
    return rows

def write_deterministic(zf: zipfile.ZipFile, name: str, data: bytes, compression: int) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = compression
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    if compression == zipfile.ZIP_DEFLATED:
        zf.writestr(info, data, compress_type=compression, compresslevel=6)
    else:
        zf.writestr(info, data, compress_type=compression)

def main() -> None:
    OUT.mkdir(exist_ok=True)
    v4_path = OUT / "wise-ronin-corpus-v4.0.0.zip"
    if not v4_path.exists() or sha256(v4_path.read_bytes()) != V4_SHA256:
        print("Downloading frozen V4 archive...", flush=True)
        v4_path.write_bytes(download(V4_URL))
    v4_bytes = v4_path.read_bytes()
    if sha256(v4_bytes) != V4_SHA256:
        raise RuntimeError("Frozen V4 ZIP SHA256 mismatch")

    with zipfile.ZipFile(v4_path) as zf:
        v4_manifest = json.loads(zf.read("corpus/manifest.json"))
        if v4_manifest.get("version") != "v4.0.0":
            raise RuntimeError("Unexpected V4 version")
        if v4_manifest.get("status") != "FROZEN_TRAINABLE" or v4_manifest.get("training_authorized") is not True:
            raise RuntimeError("V4 manifest does not authorize training")
        if "corpus/tokenizer.json" not in zf.namelist():
            raise RuntimeError("V4 tokenizer missing")

    assets: dict[str, bytes] = {}
    asset_meta: dict[str, dict] = {}
    for archive_name, (source_path, expected_blob, expected_size) in SOURCES.items():
        data = download(LORE_BASE + source_path.replace(" ", "%20"))
        asset_meta[archive_name] = verify_source(archive_name, data, expected_blob, expected_size)
        assets[archive_name] = data

    seed = jsonl_rows(assets["lore/lore_seed_v0.6.jsonl"], "lore_seed_v0.6")
    counts = {kind: sum(row.get("type") == kind for row in seed) for kind in ("fact", "lore", "bridge")}
    if len(seed) != 289 or counts != {"fact": 62, "lore": 129, "bridge": 98}:
        raise RuntimeError(f"Unexpected v0.6 seed distribution: {len(seed)} {counts}")
    ids = [row.get("id") for row in seed]
    if len(set(ids)) != len(ids) or any(not x for x in ids):
        raise RuntimeError("Lore seed has missing or duplicate IDs")
    seed_ids = set(ids)
    for row in seed:
        missing = [link for link in row.get("links", []) if link not in seed_ids]
        if missing:
            raise RuntimeError(f"{row.get('id')} has unresolved seed links: {missing}")

    topic_map = jsonl_rows(assets["lore/palimpsesto_topic_map_v0.3.jsonl"], "topic_map_v0.3")
    if len(topic_map) != 41 or sum(int(row.get("occurrences", 0)) for row in topic_map) != 16848:
        raise RuntimeError("Topic map must contain 41 topics and 16,848 total occurrences")
    index_count = sum(len(jsonl_rows(assets[f"lore/palimpsesto_index_v0.1_part{i}.jsonl"], f"index_part{i}")) for i in range(1, 5))
    if index_count != 16848:
        raise RuntimeError(f"Expected 16,848 indexed revelations, got {index_count}")

    conflict_data = download(CONFLICT_URL)
    conflict_meta = verify_source("synthetic_conflicts", conflict_data, CONFLICT_GIT_BLOB, len(conflict_data))
    conflicts = jsonl_rows(conflict_data, "system_conflicts")
    if len(conflicts) != 10:
        raise RuntimeError(f"Expected 10 frozen conflict examples, got {len(conflicts)}")
    assets["synthetic/system_conflicts.jsonl"] = conflict_data
    asset_meta["synthetic/system_conflicts.jsonl"] = conflict_meta

    manifest = {
        "version": "v5.0.0",
        "status": "FROZEN_TRAINABLE_BUNDLE",
        "training_authorized": True,
        "v4": {"version": v4_manifest["version"], "zip_bytes": len(v4_bytes), "zip_sha256": V4_SHA256, "manifest_total_rows": v4_manifest.get("total_rows")},
        "lore": {"seed_records": len(seed), "seed_type_counts": counts, "topic_count": len(topic_map), "indexed_revelations": index_count, "source_policy": "The raw source and index are provenance; only curated seed rows enter the 7.5% sampler."},
        "synthetic_conflicts": {"records": len(conflicts), **conflict_meta},
        "sampling": SAMPLING,
        "assets": asset_meta,
    }
    output = OUT / "wise-ronin-corpus-v5.0.0.zip"
    with zipfile.ZipFile(output, "w", allowZip64=True) as zf:
        write_deterministic(zf, "corpus/v4/wise-ronin-corpus-v4.0.0.zip", v4_bytes, zipfile.ZIP_STORED)
        for name, data in assets.items():
            write_deterministic(zf, name, data, zipfile.ZIP_DEFLATED)
        zf.writestr("corpus/v5/sampling.json", json.dumps(SAMPLING, ensure_ascii=False, indent=2) + "\n")
        zf.writestr("corpus/v5/manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    result = {"output": str(output), "bytes": output.stat().st_size, "sha256": sha256(output.read_bytes()), "manifest": manifest}
    (OUT / "v5-build-report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    main()
