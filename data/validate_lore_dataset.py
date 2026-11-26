#!/usr/bin/env python3
"""Validate the Concordia lore JSONL seed without third-party dependencies."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ALLOWED_TYPES = {"fact", "lore", "bridge"}
REQUIRED = {"id", "type", "domain", "title", "text", "characters", "places", "factual_claims", "uncertainty", "links", "canon_status"}


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("lore_seed_v0.3.jsonl")
    records = []
    errors = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON: {exc}")
            continue
        missing = REQUIRED - item.keys()
        if missing:
            errors.append(f"line {line_no} ({item.get('id', '?')}): missing {sorted(missing)}")
        if item.get("type") not in ALLOWED_TYPES:
            errors.append(f"line {line_no} ({item.get('id', '?')}): invalid type {item.get('type')!r}")
        if item.get("type") == "fact" and not item.get("factual_claims"):
            errors.append(f"line {line_no} ({item.get('id', '?')}): fact needs factual_claims")
        if item.get("type") == "lore" and not item.get("source_form"):
            errors.append(f"line {line_no} ({item.get('id', '?')}): lore needs source_form")
        if item.get("type") == "lore" and not item.get("author"):
            errors.append(f"line {line_no} ({item.get('id', '?')}): lore needs author")
        records.append(item)

    ids = [r.get("id") for r in records]
    seen = set()
    for record_id in ids:
        if record_id in seen:
            errors.append(f"duplicate id: {record_id}")
        seen.add(record_id)

    for item in records:
        for linked_id in item.get("links", []):
            if linked_id not in seen:
                errors.append(f"{item.get('id')}: unresolved link {linked_id}")

    counts = {kind: sum(r.get("type") == kind for r in records) for kind in sorted(ALLOWED_TYPES)}
    print(f"records={len(records)} counts={counts}")
    if errors:
        print("\n".join(f"ERROR: {e}" for e in errors), file=sys.stderr)
        return 1
    print("OK: JSONL, schema basics, IDs and links are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
