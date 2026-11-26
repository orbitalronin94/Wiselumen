#!/usr/bin/env python3
"""Validate full source preservation, corpus schema, graph matrices and measured sizes."""
from __future__ import annotations
import argparse, gzip, hashlib, json, re
from pathlib import Path

EXPECTED_SOURCE_BLOB="b0eb703c5237d1d461321a30a89525e154f0219d"
HEADER=re.compile(r"^### Revelación del Códice de Marper\s*#\s*(\d+)\s*$",re.MULTILINE)
REQUIRED={"id","type","domain","title","text","characters","places","factual_claims","uncertainty","links","canon_status"}

def git_blob_sha(data):
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()

def fail(errors,msg):
    errors.append(msg)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",default="Marper-Codice-Juntado (1).txt")
    ap.add_argument("--master",default="build/corpus_master_v1.0.jsonl")
    ap.add_argument("--manifest",default="build/corpus_master_manifest_v1.0.json")
    ap.add_argument("--graph",default="build/corpus_graph_master_v1.0.json.gz")
    args=ap.parse_args()
    errors=[]
    source_path,master_path,manifest_path,graph_path=map(Path,(args.source,args.master,args.manifest,args.graph))
    raw=source_path.read_bytes()
    source=raw.decode("utf-8")
    blob=git_blob_sha(raw)
    if blob!=EXPECTED_SOURCE_BLOB: fail(errors,f"source blob SHA mismatch: {blob}")
    headers=list(HEADER.finditer(source))
    if len(headers)!=16848: fail(errors,f"expected 16848 revelations, found {len(headers)}")
    if [int(m.group(1)) for m in headers]!=list(range(1,16849)): fail(errors,"revelation sequence is not contiguous")
    records=[]
    for line_no,line in enumerate(master_path.read_text(encoding="utf-8").splitlines(),1):
        try: records.append(json.loads(line))
        except json.JSONDecodeError as exc: fail(errors,f"line {line_no}: invalid JSON {exc}")
    ids=[r.get("id") for r in records]
    if len(ids)!=len(set(ids)): fail(errors,"duplicate record IDs")
    known=set(ids)
    for r in records:
        missing=REQUIRED-set(r)
        if missing: fail(errors,f"{r.get('id')}: missing {sorted(missing)}")
        if r.get("type") not in {"fact","lore","bridge"}: fail(errors,f"{r.get('id')}: invalid type")
        if r.get("type")=="fact" and not r.get("factual_claims"): fail(errors,f"{r.get('id')}: fact without factual_claims")
        if r.get("type")=="lore" and (not r.get("author") or not r.get("source_form")): fail(errors,f"{r.get('id')}: lore missing author/source_form")
        if r.get("record_class")=="sectarian_dogma" and not all(r.get(k) for k in ("sect","deity","truth_scope")): fail(errors,f"{r.get('id')}: incomplete dogma metadata")
        for target in r.get("links",[]):
            if target not in known: fail(errors,f"{r.get('id')}: unresolved link {target}")
    by_id={r.get("id"):r for r in records}
    pm=[r for r in records if r.get("record_class") in {"palimpsest_preface","palimpsest_revelation"}]
    if len(pm)!=16849: fail(errors,f"expected 16849 Palimpsest segments including preface, found {len(pm)}")
    if "PM00000" not in by_id: fail(errors,"missing Palimpsest preface PM00000")
    reconstructed=by_id.get("PM00000",{}).get("text","")+"".join(by_id.get(f"PM{i:05d}",{}).get("text","") for i in range(1,16849))
    if reconstructed!=source: fail(errors,"verbatim reconstruction differs from original source")
    for i in range(1,16849):
        if f"PM{i:05d}" not in by_id: fail(errors,f"missing PM record {i:05d}"); break
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    master_bytes=master_path.stat().st_size
    if manifest["source"]["git_blob_sha1"]!=blob: fail(errors,"manifest source SHA differs")
    if manifest["output"]["bytes"]!=master_bytes: fail(errors,"manifest master byte size differs")
    if manifest["output"]["sha256"]!=hashlib.sha256(master_path.read_bytes()).hexdigest(): fail(errors,"manifest master SHA256 differs")
    if manifest["output"]["records"]!=len(records): fail(errors,"manifest record count differs")
    if not graph_path.exists(): fail(errors,"graph artifact missing")
    else:
        with gzip.open(graph_path,"rt",encoding="utf-8") as f: graph=json.load(f)
        node_ids=[n["id"] for n in graph["nodes"]]
        n=len(node_ids)
        if len(node_ids)!=len(set(node_ids)): fail(errors,"duplicate graph node IDs")
        if graph["node_index"]!=node_ids: fail(errors,"graph node_index mismatch")
        if graph["generated_from"]["records"]!=len(records): fail(errors,"graph record count mismatch")
        for relation,matrix in graph["matrices"].items():
            if matrix["rows"]!=n or matrix["cols"]!=n: fail(errors,f"{relation}: dimensions mismatch")
            seen=set()
            for row,col,weight in matrix["entries"]:
                if not (isinstance(row,int) and isinstance(col,int) and 0<=row<n and 0<=col<n): fail(errors,f"{relation}: invalid COO index")
                if not isinstance(weight,(int,float)) or weight<=0: fail(errors,f"{relation}: invalid weight")
                if (row,col) in seen: fail(errors,f"{relation}: duplicate COO coordinate")
                seen.add((row,col))
        if graph["unresolved_edges"]: fail(errors,f"graph has {len(graph['unresolved_edges'])} unresolved edges")
        manifest_graph=manifest.get("graph",{})
        if manifest_graph.get("nodes")!=n: fail(errors,"manifest graph node count mismatch")
    if errors:
        print("CORPUS INVALID")
        for error in errors[:100]: print("ERROR:",error)
        print("total_errors=",len(errors))
        raise SystemExit(1)
    print(json.dumps({
        "status":"VALID","records":len(records),
        "types":{t:sum(r["type"]==t for r in records) for t in ("fact","lore","bridge")},
        "palimpsest_revelations":len(headers),"verbatim_source_reconstruction":True,
        "source_bytes":len(raw),"master_bytes":master_bytes,
        "master_MB":round(master_bytes/1_000_000,3),"master_MiB":round(master_bytes/1048576,3),
        "graph_nodes":len(graph["nodes"]),"graph_edges":graph["edge_count"],
        "graph_matrices":len(graph["matrices"]),"unresolved_links":0
    },ensure_ascii=False))

if __name__=="__main__":
    main()
