#!/usr/bin/env python3
"""Build a deterministic sparse heterogeneous graph from lore_seed_v0.3.jsonl."""
import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

SOURCE = {
    "id": "source:palimpsesto-de-marper",
    "type": "source",
    "label": "El Palimpsesto de Marper",
    "file": "Marper-Codice-Juntado (1).txt",
    "blob_sha": "b0eb703c5237d1d461321a30a89525e154f0219d",
    "size_bytes": 20401800,
    "recorded_revelations": 16848,
    "content_status": "unavailable_via_contents_api",
    "provenance": "Mayoría de material original de Isabel con pequeñas adaptaciones al lore; comparación exacta pendiente.",
}

def slug(value):
    value = unicodedata.normalize("NFD", str(value))
    value = "".join(c for c in value if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")

def build(dataset_path: Path, output_path: Path, dataset_blob_sha: str):
    records = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    nodes, edges = {}, []

    def add_node(node_id, node_type, label, **metadata):
        candidate = {"id": node_id, "type": node_type, "label": label, **metadata}
        previous = nodes.get(node_id)
        if previous and (previous["type"] != node_type or previous["label"] != label):
            raise ValueError(f"Node identity collision: {node_id}")
        nodes[node_id] = candidate
        return node_id

    def add_edge(source, relation, target):
        edges.append({"from": source, "to": target, "relation": relation, "weight": 1})

    for record in records:
        add_node(f"record:{record['id']}", "record", record.get("title", record["id"]),
                 record_type=record.get("type"), domain=record.get("domain"))

    for record in records:
        source = f"record:{record['id']}"
        scalar_fields = [
            ("domain", "domain", "domain", "has_domain"),
            ("author", "author", "author", "authored_by"),
            ("source_form", "form", "source_form", "uses_form"),
            ("sect", "sect", "sect", "belongs_to_sect"),
            ("deity", "deity", "deity", "invokes_deity"),
            ("record_class", "record_class", "record_class", "has_record_class"),
            ("truth_scope", "truth_scope", "truth_scope", "has_truth_scope"),
        ]
        for field, prefix, node_type, relation in scalar_fields:
            value = record.get(field)
            if value:
                target = add_node(f"{prefix}:{slug(value)}", node_type, value)
                add_edge(source, relation, target)
        for field, prefix, node_type, relation in [
            ("characters", "character", "character", "mentions_character"),
            ("places", "place", "place", "located_at"),
            ("factual_claims", "claim", "factual_claim", "asserts_claim"),
        ]:
            for value in record.get(field, []) or []:
                target = add_node(f"{prefix}:{slug(value)[:100]}", node_type, value)
                add_edge(source, relation, target)
        for target_id in record.get("links", []) or []:
            add_edge(source, "references_record", f"record:{target_id}")

    add_node(SOURCE["id"], SOURCE["type"], SOURCE["label"],
             **{k: v for k, v in SOURCE.items() if k not in {"id", "type", "label"}})
    ordered_nodes = sorted(nodes.values(), key=lambda item: item["id"])
    node_index = [item["id"] for item in ordered_nodes]
    index = {node_id: i for i, node_id in enumerate(node_index)}

    relations, unresolved = defaultdict(list), []
    for edge in edges:
        if edge["from"] not in index:
            raise ValueError(f"Missing edge source: {edge}")
        if edge["to"] not in index:
            if edge["relation"] == "references_record":
                unresolved.append(edge)
                continue
            raise ValueError(f"Missing edge target: {edge}")
        relations[edge["relation"]].append([index[edge["from"]], index[edge["to"]], edge["weight"]])

    matrices = {}
    for relation, entries in relations.items():
        matrices[relation] = {
            "rows": len(node_index), "cols": len(node_index), "format": "COO",
            "entries": sorted(entries, key=lambda item: (item[0], item[1])),
        }

    # Projections indicate shared metadata only, never agreement or causality.
    for entity_type, relation in [
        ("character", "mentions_character"), ("place", "located_at"), ("domain", "has_domain")
    ]:
        buckets = defaultdict(set)
        for edge in edges:
            if edge["relation"] == relation and edge["to"] in index:
                buckets[edge["to"]].add(index[edge["from"]])
        weights = defaultdict(int)
        for ids in buckets.values():
            ids = sorted(ids)
            for i, left in enumerate(ids):
                for right in ids[i + 1:]:
                    weights[(left, right)] += 1
        entries = []
        for (left, right), weight in sorted(weights.items()):
            entries.extend([[left, right, weight], [right, left, weight]])
        matrices[f"projection_shared_{entity_type}"] = {
            "rows": len(node_index), "cols": len(node_index), "format": "COO",
            "entries": entries,
        }

    graph = {
        "schema_version": "1.0.0",
        "graph_type": "heterogeneous_directed_multirelational",
        "generated_from": {
            "dataset": str(dataset_path), "dataset_blob_sha": dataset_blob_sha,
            "record_count": len(records),
        },
        "completeness": {
            "status": "partial",
            "reason": "El texto íntegro del Palimpsesto no está disponible para parseo; sus revelaciones no se han inventado.",
            "missing_source_nodes": [SOURCE["id"]],
            "requires_original_text": True,
        },
        "node_index": node_index,
        "nodes": ordered_nodes,
        "matrices": matrices,
        "edge_count": len(edges) - len(unresolved),
        "unresolved_record_links": unresolved,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"records={len(records)} nodes={len(ordered_nodes)} edges={graph['edge_count']} unresolved={len(unresolved)} status=partial")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/lore_seed_v0.3.jsonl")
    parser.add_argument("--output", default="corpus/graph_v0.3.json")
    parser.add_argument("--dataset-blob-sha", default="b9bdf1d799fcc3cccb68d043662be2e93516372e")
    args = parser.parse_args()
    build(Path(args.dataset), Path(args.output), args.dataset_blob_sha)
