#!/usr/bin/env python3
"""Build a sharded sparse heterogeneous graph for the complete structural Corpus Master."""
import argparse, json, re, unicodedata
from collections import defaultdict
from pathlib import Path

SOURCE = {
 "id":"source:palimpsesto-de-marper","type":"source","label":"El Palimpsesto de Marper",
 "file":"Marper-Codice-Juntado (1).txt","blob_sha":"b0eb703c5237d1d461321a30a89525e154f0219d",
 "size_bytes":20401800,"recorded_revelations":16848,"content_status":"indexed_all_revelations",
 "provenance":"Mayoría de material original de Isabel con pequeñas adaptaciones al lore; proporción exacta pendiente."
}
SIMPLE_RELATIONS={"asserts_claim","authored_by","has_domain","mentions_character","references_record","uses_form","curated_by_character","topic_linked_fact","topic_adapted_as_lore","topic_separated_by_bridge"}

def slug(value):
 value=unicodedata.normalize("NFD",str(value))
 value="".join(c for c in value if unicodedata.category(c)!="Mn")
 return re.sub(r"[^a-z0-9]+","-",value.lower()).strip("-")

def read_jsonl(path):
 return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def build(dataset_path,index_paths,topic_map_path,output_manifest,dataset_blob_sha):
 records=read_jsonl(dataset_path)
 topic_map=read_jsonl(topic_map_path)
 if len(topic_map)!=41: raise ValueError(f"Expected 41 Palimpsest topic mappings, got {len(topic_map)}")
 if sum(x.get("occurrences",0) for x in topic_map)!=16848: raise ValueError("Topic map must account for all 16848 revelations")
 record_ids={x["id"] for x in records}
 for item in topic_map:
  if item.get("assigned_character_id") is None or item.get("assigned_character_name") is None: raise ValueError(f"Missing curator for {item.get('topic')}")
  for field in ("fact_record_ids","lore_record_ids","bridge_record_ids"):
   for linked in item.get(field,[]):
    if linked not in record_ids: raise ValueError(f"Topic map {item['topic']} links missing record {linked}")
 source_records=[]
 for path in index_paths: source_records.extend(read_jsonl(path))
 if len(source_records)!=16848: raise ValueError(f"Expected 16848 source records, got {len(source_records)}")
 expected=[f"PM{i:05d}" for i in range(1,16849)]
 if [x["id"] for x in source_records]!=expected: raise ValueError("Palimpsest index IDs are not complete and sequential")
 nodes,edges={},[]
 def add_node(node_id,kind,label,**meta):
  old=nodes.get(node_id)
  if old and (old["type"]!=kind or old["label"]!=label): raise ValueError(f"Node collision: {node_id}")
  nodes[node_id]={"id":node_id,"type":kind,"label":label,**meta}
  return node_id
 def add_edge(a,rel,b): edges.append((a,rel,b,1))
 add_node(SOURCE["id"],SOURCE["type"],SOURCE["label"],**{k:v for k,v in SOURCE.items() if k not in {"id","type","label"}})
 for r in records: add_node(f"record:{r['id']}","record",r.get("title",r["id"]),record_type=r.get("type"),domain=r.get("domain"))
 for r in records:
  src=f"record:{r['id']}"
  for field,prefix,kind,rel in [
   ("domain","domain","domain","has_domain"),("author","author","author","authored_by"),
   ("source_form","form","source_form","uses_form"),("sect","sect","sect","belongs_to_sect"),
   ("deity","deity","deity","invokes_deity"),("record_class","record_class","record_class","has_record_class"),
   ("truth_scope","truth_scope","truth_scope","has_truth_scope"),("presented_by","character","character","presented_by")]:
   value=r.get(field)
   if value: add_edge(src,rel,add_node(f"{prefix}:{slug(value)}",kind,value))
  for field,prefix,kind,rel in [
   ("characters","character","character","mentions_character"),("places","place","place","located_at"),
   ("factual_claims","claim","factual_claim","asserts_claim")]:
   for value in r.get(field,[]) or []: add_edge(src,rel,add_node(f"{prefix}:{slug(value)[:100]}",kind,value))
  for item in r.get("sources",[]) or []:
   url=item.get("url")
   if url: add_edge(src,"supported_by_source",add_node(f"reference:{slug(url)}","reference",url,uri=url))
  for target in r.get("links",[]) or []: add_edge(src,"references_record",f"record:{target}")
 for item in source_records:
  src=f"palimpsest:{item['id']}"
  add_node(src,"source_record",item["title"],source_revelation_number=item["source_revelation_number"],
           source_line_start=item["source_line_start"],source_line_end=item["source_line_end"],
           topic=item["topic"],topic_group=item["topic_group"],truth_scope=item["truth_scope"])
  add_edge(src,"derived_from",SOURCE["id"])
  add_edge(src,"has_topic",add_node(f"topic:{slug(item['topic'])}","topic",item["topic"]))
  add_edge(src,"has_topic_group",add_node(f"topic_group:{item['topic_group']}","topic_group",item["topic_group"]))
 for item in topic_map:
  topic_node=f"topic:{slug(item['topic'])}"
  add_edge(topic_node,"curated_by_character",add_node(f"character:{slug(item['assigned_character_name'])}","character",item["assigned_character_name"],character_id=item["assigned_character_id"]))
  for target in item.get("fact_record_ids",[]): add_edge(topic_node,"topic_linked_fact",f"record:{target}")
  for target in item.get("lore_record_ids",[]): add_edge(topic_node,"topic_adapted_as_lore",f"record:{target}")
  for target in item.get("bridge_record_ids",[]): add_edge(topic_node,"topic_separated_by_bridge",f"record:{target}")


 ordered=sorted(nodes.values(),key=lambda x:x["id"])
 node_ids=[x["id"] for x in ordered]; index={v:i for i,v in enumerate(node_ids)}
 relations=defaultdict(list); unresolved=[]
 for a,rel,b,w in edges:
  if a not in index: raise ValueError(f"Missing edge source {a}")
  if b not in index:
   if rel=="references_record": unresolved.append((a,rel,b,w)); continue
   raise ValueError(f"Missing edge target {b}")
  relations[rel].append([index[a],index[b],w])
 for entries in relations.values(): entries.sort(key=lambda x:(x[0],x[1]))

 # Co-occurrence projections are computed only over the curated records, not all repeated source revelations.
 for entity,rel in [("character","mentions_character"),("place","located_at"),("domain","has_domain")]:
  buckets=defaultdict(set)
  for a,r,b,w in edges:
   if r==rel and a.startswith("record:"): buckets[b].add(index[a])
  weights=defaultdict(int)
  for ids in buckets.values():
   ids=sorted(ids)
   for i,left in enumerate(ids):
    for right in ids[i+1:]: weights[(left,right)]+=1
  entries=[]
  for (left,right),weight in sorted(weights.items()): entries.extend([[left,right,weight],[right,left,weight]])
  relations[f"projection_shared_{entity}"]=entries

 out=Path(output_manifest).parent
 graph_dir=out/Path(output_manifest).stem
 matrix_dir=graph_dir/"matrices"
 graph_dir.mkdir(parents=True,exist_ok=True); matrix_dir.mkdir(parents=True,exist_ok=True)
 node_shards=[]
 shard_size=4000
 shard_count=(len(ordered)+shard_size-1)//shard_size
 for part in range(shard_count):
  chunk=ordered[part*shard_size:(part+1)*shard_size]
  path=graph_dir/f"nodes_part{part+1}.jsonl"
  path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in chunk),encoding="utf-8")
  node_shards.append({"path":str(path),"count":len(chunk)})

 matrix_meta={}
 remaining=[]
 for rel,entries in sorted(relations.items()):
  if rel in SIMPLE_RELATIONS:
   path=matrix_dir/f"{rel}.coo.jsonl"
   path.write_text("".join(json.dumps(x)+"\n" for x in entries),encoding="utf-8")
   matrix_meta[rel]={"path":str(path),"rows":len(ordered),"cols":len(ordered),"format":"COO","nnz":len(entries)}
  else:
   for entry in entries: remaining.append(json.dumps({"relation":rel,"entry":entry}))
 combined_path=matrix_dir/"remaining_relations.coo.jsonl"
 combined_path.write_text("\n".join(remaining)+("\n" if remaining else ""),encoding="utf-8")
 for rel,entries in sorted(relations.items()):
  if rel not in SIMPLE_RELATIONS:
   matrix_meta[rel]={"path":str(combined_path),"rows":len(ordered),"cols":len(ordered),"format":"COO_RELATION_ROWS","nnz":len(entries)}
 base_edges=sum(len(v) for k,v in relations.items() if not k.startswith("projection_"))
 manifest={
  "schema_version":"1.1.0","graph_type":"heterogeneous_directed_multirelational",
  "generated_from":{"dataset":str(dataset_path),"dataset_blob_sha":dataset_blob_sha,
                    "record_count":len(records),"palimpsest_index_records":len(source_records),"index_shards":len(index_paths),"topic_map":str(topic_map_path),"topic_map_records":len(topic_map),"topic_occurrences":sum(x["occurrences"] for x in topic_map)},
  "completeness":{"status":"complete_structural_index",
   "reason":"Los 16.848 registros fuente están indexados por ID, tema y rango de líneas; las afirmaciones del Palimpsesto siguen sin auditoría factual individual.",
   "requires_original_text":False,"requires_claim_review":True},
  "node_count":len(ordered),
  "node_index_order":"Concatenar nodes_part1..partN en orden; orden lexicográfico por id; el índice matricial es la posición base cero.",
  "nodes_shards":node_shards,"matrices":matrix_meta,"edge_count":base_edges,"topic_map_record_count":len(topic_map),"topic_map_occurrences":sum(x["occurrences"] for x in topic_map),
  "unresolved_record_links":[{"from":a,"relation":r,"to":b} for a,r,b,w in unresolved],
  "projection_semantics":"Las proyecciones compartidas se calculan solo sobre los registros curados y expresan coaparición de metadatos; no implican causalidad ni acuerdo."
 }
 Path(output_manifest).write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 print(f"seed_records={len(records)} source_records={len(source_records)} nodes={len(ordered)} edges={base_edges} unresolved={len(unresolved)} matrices={len(matrix_meta)}")

if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("--dataset",default="data/lore_seed_v0.6.jsonl")
 p.add_argument("--index-dir",default="data")
 p.add_argument("--topic-map",default="data/palimpsesto_topic_map_v0.3.jsonl")
 p.add_argument("--output",default="corpus/graph_v0.6.json")
 p.add_argument("--dataset-blob-sha",default="10d0aa1feddca898031740f709993dc11a0c710c")
 a=p.parse_args()
 indexes=[Path(a.index_dir)/f"palimpsesto_index_v0.1_part{i}.jsonl" for i in range(1,5)]
 build(a.dataset,indexes,a.topic_map,a.output,a.dataset_blob_sha)
