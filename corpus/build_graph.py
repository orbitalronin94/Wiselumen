#!/usr/bin/env python3
"""Build the complete structural Corpus Master graph from v0.4 and the Palimpsest index."""
import argparse, json, re, unicodedata
from collections import defaultdict
from pathlib import Path

SOURCE = {
    "id": "source:palimpsesto-de-marper", "type": "source",
    "label": "El Palimpsesto de Marper",
    "file": "Marper-Codice-Juntado (1).txt",
    "blob_sha": "b0eb703c5237d1d461321a30a89525e154f0219d",
    "size_bytes": 20401800, "recorded_revelations": 16848,
    "content_status": "indexed_all_revelations",
    "provenance": "Mayoría de material original de Isabel con pequeñas adaptaciones al lore; proporción exacta pendiente.",
}

def slug(value):
    value = unicodedata.normalize("NFD", str(value))
    value = "".join(c for c in value if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")

def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def build(dataset_path, index_paths, output_path, dataset_blob_sha):
    records = read_jsonl(Path(dataset_path))
    source_records = []
    for path in index_paths:
        source_records.extend(read_jsonl(Path(path)))
    if len(source_records) != 16848:
        raise ValueError(f"Expected 16848 indexed revelations, found {len(source_records)}")
    expected = [f"PM{i:05d}" for i in range(1, 16849)]
    actual = [r["id"] for r in source_records]
    if actual != expected:
        raise ValueError("Palimpsest index IDs are missing, duplicated, or out of order")

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

    add_node(SOURCE["id"], SOURCE["type"], SOURCE["label"],
             **{k:v for k,v in SOURCE.items() if k not in {"id","type","label"}})
    for record in records:
        add_node(f"record:{record['id']}", "record", record.get("title", record["id"]),
                 record_type=record.get("type"), domain=record.get("domain"))
    for record in records:
        source = f"record:{record['id']}"
        scalar_fields = [
            ("domain","domain","domain","has_domain"), ("author","author","author","authored_by"),
            ("source_form","form","source_form","uses_form"), ("sect","sect","sect","belongs_to_sect"),
            ("deity","deity","deity","invokes_deity"), ("record_class","record_class","record_class","has_record_class"),
            ("truth_scope","truth_scope","truth_scope","has_truth_scope"),
            ("presented_by","character","character","presented_by"),
        ]
        for field,prefix,kind,relation in scalar_fields:
            value=record.get(field)
            if value:
                add_edge(source,relation,add_node(f"{prefix}:{slug(value)}",kind,value))
        for field,prefix,kind,relation in [
            ("characters","character","character","mentions_character"),
            ("places","place","place","located_at"),
            ("factual_claims","claim","factual_claim","asserts_claim"),
        ]:
            for value in record.get(field,[]) or []:
                add_edge(source,relation,add_node(f"{prefix}:{slug(value)[:100]}",kind,value))
        for src in record.get("sources",[]) or []:
            url=src.get("url")
            if url:
                target=add_node(f"reference:{slug(url)}","reference",url,uri=url)
                add_edge(source,"supported_by_source",target)
        for target_id in record.get("links",[]) or []:
            add_edge(source,"references_record",f"record:{target_id}")

    for item in source_records:
        source=f"palimpsest:{item['id']}"
        add_node(source,"source_record",item["title"],
                 source_revelation_number=item["source_revelation_number"],
                 source_line_start=item["source_line_start"],source_line_end=item["source_line_end"],
                 topic=item["topic"],topic_group=item["topic_group"],
                 truth_scope=item["truth_scope"])
        add_edge(source,"derived_from",SOURCE["id"])
        topic=add_node(f"topic:{slug(item['topic'])}","topic",item["topic"])
        group=add_node(f"topic_group:{item['topic_group']}","topic_group",item["topic_group"])
        add_edge(source,"has_topic",topic)
        add_edge(source,"has_topic_group",group)

    ordered=sorted(nodes.values(),key=lambda x:x["id"])
    node_index=[x["id"] for x in ordered]
    idx={v:i for i,v in enumerate(node_index)}
    relations=defaultdict(list); unresolved=[]
    for edge in edges:
        if edge["from"] not in idx: raise ValueError(f"Missing source node: {edge}")
        if edge["to"] not in idx:
            if edge["relation"]=="references_record": unresolved.append(edge); continue
            raise ValueError(f"Missing target node: {edge}")
        relations[edge["relation"]].append([idx[edge["from"]],idx[edge["to"]],edge["weight"]])
    matrices={}
    for rel,entries in relations.items():
        matrices[rel]={"rows":len(idx),"cols":len(idx),"format":"COO",
                       "entries":sorted(entries,key=lambda x:(x[0],x[1]))}

    # Shared-entity projections are computed only over curated records, not all repeated revelations.
    for entity_type,rel in [("character","mentions_character"),("place","located_at"),("domain","has_domain")]:
        buckets=defaultdict(set)
        for edge in edges:
            if edge["relation"]==rel and edge["from"].startswith("record:"):
                buckets[edge["to"]].add(idx[edge["from"]])
        weights=defaultdict(int)
        for ids in buckets.values():
            ids=sorted(ids)
            for i,left in enumerate(ids):
                for right in ids[i+1:]: weights[(left,right)]+=1
        entries=[]
        for (left,right),weight in sorted(weights.items()):
            entries.extend([[left,right,weight],[right,left,weight]])
        matrices[f"projection_shared_{entity_type}"]={"rows":len(idx),"cols":len(idx),"format":"COO","entries":entries}

    graph={
      "schema_version":"1.1.0","graph_type":"heterogeneous_directed_multirelational",
      "generated_from":{"dataset":str(dataset_path),"dataset_blob_sha":dataset_blob_sha,
                        "record_count":len(records),"palimpsest_index_records":len(source_records)},
      "completeness":{"status":"complete_structural_index",
        "reason":"Los 16.848 registros fuente están indexados por ID, tema y rango de líneas; las afirmaciones del Palimpsesto siguen sin auditoría factual individual.",
        "requires_original_text":False,"requires_claim_review":True},
      "node_index":node_index,"nodes":ordered,"matrices":matrices,
      "edge_count":sum(len(v["entries"]) for k,v in matrices.items() if not k.startswith("projection_")),
      "unresolved_record_links":unresolved
    }
    output_path=Path(output_path);output_path.parent.mkdir(parents=True,exist_ok=True)
    output_path.write_text(json.dumps(graph,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"seed_records={len(records)} palimpsest_records={len(source_records)} nodes={len(ordered)} edges={graph['edge_count']} unresolved={len(unresolved)} status={graph['completeness']['status']}")

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",default="data/lore_seed_v0.4.jsonl")
    p.add_argument("--index-dir",default="data")
    p.add_argument("--output",default="corpus/graph_v0.4.json")
    p.add_argument("--dataset-blob-sha",default="bfe381b1819217c19fb208b0dd0fdadc15a55814")
    a=p.parse_args()
    indexes=[Path(a.index_dir)/f"palimpsesto_index_v0.1_part{i}.jsonl" for i in range(1,5)]
    build(a.dataset,indexes,a.output,a.dataset_blob_sha)
