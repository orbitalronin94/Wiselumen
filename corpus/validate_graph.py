#!/usr/bin/env python3
"""Validate the sparse heterogeneous graph exported for Wiselumen."""
import argparse
import json
from pathlib import Path

def validate(path: Path, require_complete: bool = False) -> list[str]:
    graph = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    nodes = graph.get("nodes", [])
    ids = [n.get("id") for n in nodes]
    if len(ids) != len(set(ids)):
        errors.append("duplicate node IDs")
    index = graph.get("node_index", [])
    if len(index) != len(ids) or set(index) != set(ids):
        errors.append("node_index does not match nodes")
    known = set(ids)
    n = len(index)
    for relation, matrix in graph.get("matrices", {}).items():
        if matrix.get("rows") != n or matrix.get("cols") != n:
            errors.append(f"{relation}: matrix dimensions do not match node_index")
        seen = set()
        for entry in matrix.get("entries", []):
            if len(entry) != 3:
                errors.append(f"{relation}: malformed COO entry {entry!r}")
                continue
            row, col, weight = entry
            if not isinstance(row, int) or not isinstance(col, int):
                errors.append(f"{relation}: non-integer matrix index")
                continue
            if not (0 <= row < n and 0 <= col < n):
                errors.append(f"{relation}: index out of bounds")
            if not isinstance(weight, (int, float)) or weight <= 0:
                errors.append(f"{relation}: weight must be positive")
            if (row, col) in seen:
                errors.append(f"{relation}: duplicate COO coordinate {(row, col)}")
            seen.add((row, col))
    unresolved = graph.get("unresolved_record_links", [])
    if unresolved:
        errors.append(f"{len(unresolved)} unresolved record links")
    completeness = graph.get("completeness", {})
    if require_complete and completeness.get("status") != "complete":
        errors.append("graph is partial; complete source corpus is required")
    if graph.get("edge_count") is None:
        errors.append("missing edge_count")
    return errors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("graph", nargs="?", default="corpus/graph_v0.3.json")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    problems = validate(Path(args.graph), args.require_complete)
    if problems:
        print("GRAPH INVALID")
        for problem in problems:
            print("-", problem)
        raise SystemExit(1)
    print("GRAPH VALID")
