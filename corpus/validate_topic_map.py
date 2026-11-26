#!/usr/bin/env python3
"""Validate the complete 41-topic map, source excerpts, and dedicated adaptations."""
import json
import re
import sys
from pathlib import Path

HEADER = re.compile(r"^### Revelación del Códice de Marper\s*#\s*(\d+)\s*$", re.M)

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def extract_source(path):
    text = Path(path).read_text(encoding="utf-8")
    matches = list(HEADER.finditer(text))
    if len(matches) != 16848:
        raise ValueError(f"expected 16848 source headings, found {len(matches)}")
    modules = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[match.end():end].strip()
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", block) if p.strip()]
        if len(paragraphs) < 2:
            raise ValueError(f"source revelation {match.group(1)} has no thematic paragraph")
        passage = paragraphs[1]
        topic = passage.split(":", 1)[0].strip()
        if topic not in modules:
            modules[topic] = {"first_source_id": f"PM{int(match.group(1)):05d}", "excerpt": passage, "occurrences": 0}
        modules[topic]["occurrences"] += 1
    if len(modules) != 41:
        raise ValueError(f"expected 41 distinct topics, found {len(modules)}")
    return modules

def main():
    topic_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/palimpsesto_topic_map_v0.3.jsonl")
    seed_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/lore_seed_v0.6.jsonl")
    people_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("lore/PERSONAJES.md")
    source_path = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("Marper-Codice-Juntado (1).txt")
    topics = read_jsonl(topic_path)
    records = read_jsonl(seed_path)
    errors = []
    by_id = {r["id"]: r for r in records}
    people_text = people_path.read_text(encoding="utf-8")
    character_ids = set(re.findall(r"^\| (P\d{3}) \|", people_text, flags=re.M))
    source = extract_source(source_path)
    if len(topics) != 41:
        errors.append(f"expected 41 topic mappings, got {len(topics)}")
    if len({r.get("topic") for r in topics}) != len(topics):
        errors.append("duplicate source topic")
    if sum(r.get("occurrences", 0) for r in topics) != 16848:
        errors.append("topic occurrence counts do not sum to 16848")
    for item in topics:
        topic = item.get("topic")
        source_item = source.get(topic)
        if source_item is None:
            errors.append(f"{topic}: not found in original source")
            continue
        if item.get("occurrences") != source_item["occurrences"]:
            errors.append(f"{topic}: occurrence count differs from source")
        if item.get("first_source_id") != source_item["first_source_id"]:
            errors.append(f"{topic}: first source ID differs from source")
        if item.get("assigned_character_id") not in character_ids:
            errors.append(f"{topic}: curator ID missing from PERSONAJES.md")
        if not item.get("assigned_character_name"):
            errors.append(f"{topic}: missing curator name")
        for field in ("fact_record_ids", "lore_record_ids", "bridge_record_ids"):
            for target in item.get(field, []):
                if target not in by_id:
                    errors.append(f"{topic}: missing linked record {target}")
        dedicated = item.get("adaptation_record_ids", [])
        if len(dedicated) != 2:
            errors.append(f"{topic}: expected one dedicated lore and one bridge")
            continue
        lore_id, bridge_id = dedicated
        lore, bridge = by_id.get(lore_id), by_id.get(bridge_id)
        if not lore or lore.get("record_class") != "palimpsest_adaptation":
            errors.append(f"{topic}: dedicated lore record missing or mistyped")
            continue
        if not bridge or bridge.get("record_class") != "palimpsest_source_bridge":
            errors.append(f"{topic}: dedicated epistemic bridge missing or mistyped")
            continue
        for record in (lore, bridge):
            if record.get("source_topic") != topic:
                errors.append(f"{topic}: adaptation source_topic mismatch in {record['id']}")
            if record.get("source_revelation_id") != source_item["first_source_id"]:
                errors.append(f"{topic}: source revelation mismatch in {record['id']}")
            if record.get("source_repetition_count") != source_item["occurrences"]:
                errors.append(f"{topic}: repetition count mismatch in {record['id']}")
        if lore.get("source_excerpt") != source_item["excerpt"]:
            errors.append(f"{topic}: source excerpt is not verbatim in {lore_id}")
        if lore.get("factual_claims"):
            errors.append(f"{topic}: fictional adaptation {lore_id} must not assert unverified source claims as facts")
        if lore_id not in item.get("lore_record_ids", []):
            errors.append(f"{topic}: dedicated lore absent from lore_record_ids")
        if bridge_id not in item.get("bridge_record_ids", []):
            errors.append(f"{topic}: dedicated bridge absent from bridge_record_ids")
        if item.get("source_blob_sha") != "b0eb703c5237d1d461321a30a89525e154f0219d":
            errors.append(f"{topic}: source blob SHA mismatch")
    if errors:
        print("TOPIC MAP INVALID")
        for error in errors:
            print("-", error)
        return 1
    print(f"TOPIC MAP VALID: {len(topics)} topics, 16848 source revelations, 41 dedicated lore records, 41 dedicated bridges, {len(records)} curated records")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
