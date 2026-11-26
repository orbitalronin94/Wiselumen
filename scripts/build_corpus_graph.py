#!/usr/bin/env python3
"""Build a sparse, typed, multi-relational graph for the unified master corpus."""
from __future__ import annotations
import argparse, gzip, json, re, unicodedata
from collections import defaultdict, Counter
from pathlib import Path

def slug(value):
    value = unicodedata.normalize("NFD", str(value))
    value = "".join(c for c in value if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--master",default="build/corpus_master_v1.0.jsonl")
    ap.add_argument("--manifest",default="build/corpus_master_manifest_v1.0.json")
    ap.add_argument("--out",default="build/corpus_graph_master_v1.0.json.gz")
    args=ap.parse_args()
    records=[json.loads(line) for line in Path(args.master).read_text(encoding="utf-8").splitlines() if line.strip()]
    nodes={}
    edges=[]
    def node(node_id,kind,label,**meta):
        old=nodes.get(node_id)
        if old and (old["type"]!=kind or old["label"]!=label):
            raise ValueError(f"Node identity collision: {node_id}")
        nodes[node_id]={"id":node_id,"type":kind,"label":label,**meta}
        return node_id
    def edge(a,rel,b):
        edges.append((a,rel,b,1))
    manifest=json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    src=manifest["source"]
    node("source:S001","source","El Palimpsesto de Marper",
         path=src["path"],blob_sha1=src["git_blob_sha1"],bytes=src["bytes"],
         revelations=src["revelations"],provenance="Material mayoritariamente original de Isabel con adaptaciones parciales al lore.")
    for r in records:
        node("record:"+r["id"],"record",r.get("title",r["id"]),
             record_type=r.get("type"),record_class=r.get("record_class"),
             truth_scope=r.get("truth_scope"))
    for r in records:
        a="record:"+r["id"]
        if r.get("source_id")=="S001":
            edge(a,"derived_from","source:S001")
        scalar=[
            ("domain","domain","has_domain"),("author","author","authored_by"),
            ("source_form","source_form","uses_form"),("sect","sect","belongs_to_sect"),
            ("deity","deity","invokes_deity"),("record_class","record_class","has_record_class"),
            ("truth_scope","truth_scope","has_truth_scope")
        ]
        for field,kind,rel in scalar:
            val=r.get(field)
            if val:
                edge(a,rel,node(kind+":"+slug(val),kind,val))
        for field,kind,rel in [
            ("characters","character","mentions_character"),
            ("places","place","located_at"),
            ("factual_claims","factual_claim","asserts_claim")
        ]:
            for val in r.get(field,[]) or []:
                edge(a,rel,node(kind+":"+slug(val)[:120],kind,val))
        for target in r.get("links",[]) or []:
            edge(a,"references_record","record:"+target)
    ordered=sorted(nodes.values(),key=lambda x:x["id"])
    index={n["id"]:i for i,n in enumerate(ordered)}
    rel_entries=defaultdict(list)
    unresolved=[]
    for a,rel,b,w in edges:
        if a not in index or b not in index:
            unresolved.append({"from":a,"relation":rel,"to":b})
            continue
        rel_entries[rel].append([index[a],index[b],w])
    matrices={}
    for rel,entries in rel_entries.items():
        matrices[rel]={"rows":len(ordered),"cols":len(ordered),"format":"COO",
                       "entries":sorted(entries,key=lambda x:(x[0],x[1]))}
    # Co-occurrence projections are capped: high-degree hubs stay as incidence nodes.
    skipped={}
    for kind,rel in [("character","mentions_character"),("place","located_at"),("domain","has_domain")]:
        buckets=defaultdict(set)
        for a,r,b,w in edges:
            if r==rel and a in index and b in index:
                buckets[b].add(index[a])
        entries=[]
        skipped_count=0
        for entity,ids in buckets.items():
            ids=sorted(ids)
            if len(ids)>32:
                skipped_count+=1
                continue
            for i,left in enumerate(ids):
                for right in ids[i+1:]:
                    entries.extend(([left,right,1],[right,left,1]))
        matrices["projection_shared_"+kind]={"rows":len(ordered),"cols":len(ordered),"format":"COO",
                                             "entries":sorted(entries,key=lambda x:(x[0],x[1]))}
        skipped[kind]=skipped_count
    graph={
        "schema_version":"1.0.0","graph_type":"heterogeneous_directed_multirelational",
        "generated_from":{"dataset":args.master,"records":len(records),"source_blob_sha1":src["git_blob_sha1"]},
        "completeness":{"status":"complete_for_declared_inputs","source_revelations":src["revelations"],
                        "source_preface_included":True,"unresolved_links":len(unresolved),
                        "factual_review":"source claims remain unverified; separate fact layer carries review status"},
        "node_index":[n["id"] for n in ordered],"nodes":ordered,"matrices":matrices,
        "edge_count":len(edges)-len(unresolved),"unresolved_edges":unresolved,
        "projection_policy":{"maximum_entity_degree":32,"high_degree_entities_skipped":skipped,
                             "meaning":"Co-occurrence is structural only; it does not imply truth, agreement or causality."}
    }
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(out,"wt",encoding="utf-8",compresslevel=9) as f:
        json.dump(graph,f,ensure_ascii=False,separators=(",",":"))
    summary={"nodes":len(ordered),"edges":graph["edge_count"],"unresolved":len(unresolved),
             "matrix_entries":{k:len(v["entries"]) for k,v in matrices.items()},
             "projection_skips":skipped,"compressed_graph_bytes":out.stat().st_size}
    Path("build/corpus_graph_summary_v1.0.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    manifest["graph"]=summary
    Path(args.manifest).write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False))

if __name__=="__main__":
    main()
