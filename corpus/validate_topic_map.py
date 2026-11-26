#!/usr/bin/env python3
"""Validate the 41-topic map against the curated v0.5 dataset."""
import json
import re
import sys
from pathlib import Path

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def main():
    topic_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/palimpsesto_topic_map_v0.2.jsonl")
    seed_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/lore_seed_v0.5.jsonl")
    people_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("lore/PERSONAJES.md")
    topics = read_jsonl(topic_path)
    records = read_jsonl(seed_path)
    errors = []
    ids = {r["id"] for r in records}
    people_text = people_path.read_text(encoding="utf-8")
    character_ids = set(re.findall(r"^\\| (P\\d{3}) \\|", people_text, flags=re.M))
    names = [r.get("topic") for r in topics]
    if len(topics) != 41:
        errors.append(f"expected 41 topic mappings, got {len(topics)}")
    if len(names) != len(set(names)):
        errors.append("duplicate source topic")
    if sum(r.get("occurrences", 0) for r in topics) != 16848:
        errors.append("topic occurrence counts do not sum to 16848")
    for item in topics:
        if item.get("assigned_character_id") not in character_ids:
            errors.append(f"{item.get('topic')}: curator ID missing from PERSONAJES.md")
        if not item.get("assigned_character_name"):
            errors.append(f"{item.get('topic')}: missing curator name")
        for field in ("fact_record_ids", "lore_record_ids", "bridge_record_ids"):
            for target in item.get(field, []):
                if target not in ids:
                    errors.append(f"{item.get('topic')}: missing linked record {target}")
        if item.get("source_blob_sha") != "b0eb703c5237d1d461321a30a89525e154f0219d":
            errors.append(f"{item.get('topic')}: source blob SHA mismatch")
    if errors:
        print("TOPIC MAP INVALID")
        for error in errors:
            print("-", error)
        return 1
    print(f"TOPIC MAP VALID: {len(topics)} topics, {sum(x['occurrences'] for x in topics)} revelations, {len(ids)} curated records")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
