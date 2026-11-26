#!/usr/bin/env python3
"""Validate the sharded sparse graph manifest and all COO matrix shards."""
import argparse,json
from pathlib import Path

def read_jsonl(path):
 return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def validate(manifest_path,require_complete=False):
 manifest_path=Path(manifest_path)
 graph=json.loads(manifest_path.read_text(encoding="utf-8"))
 errors=[]
 nodes=[]
 for shard in graph.get("nodes_shards",[]):
  path=Path(shard["path"])
  try: part=read_jsonl(path)
  except Exception as exc:
   errors.append(f"cannot read node shard {path}: {exc}");continue
  if len(part)!=shard.get("count"): errors.append(f"{path}: node count mismatch")
  nodes.extend(part)
 ids=[n.get("id") for n in nodes]
 if len(ids)!=len(set(ids)): errors.append("duplicate node IDs")
 if ids!=sorted(ids): errors.append("node shards are not globally sorted by id")
 if len(nodes)!=graph.get("node_count"): errors.append("node_count mismatch")
 n=len(nodes); known=set(ids); id_by_index=ids; id_to_index={node_id:i for i,node_id in enumerate(ids)}
 if graph.get("unresolved_record_links"): errors.append(f"{len(graph['unresolved_record_links'])} unresolved record links")

 source_records=[x for x in nodes if x.get("type")=="source_record"]
 if len(source_records)!=16848: errors.append(f"expected 16848 Palimpsest source records, got {len(source_records)}")
 for i,item in enumerate(source_records,1):
  if item.get("source_revelation_number")!=i: errors.append(f"source revelation sequence breaks at {i}")
  if not isinstance(item.get("source_line_start"),int) or not isinstance(item.get("source_line_end"),int) or item["source_line_end"]<item["source_line_start"]:
   errors.append(f"invalid source line range for {item.get('id')}")

 matrix_cache={}
 total_edges=0
 for relation,spec in graph.get("matrices",{}).items():
  if spec.get("rows")!=n or spec.get("cols")!=n: errors.append(f"{relation}: dimensions do not match node_count")
  path=spec.get("path")
  if path not in matrix_cache:
   try:
    if spec.get("format")=="COO_RELATION_ROWS":
     grouped={}
     for row in read_jsonl(path): grouped.setdefault(row["relation"],[]).append(row["entry"])
     matrix_cache[path]=grouped
    else: matrix_cache[path]=read_jsonl(path)
   except Exception as exc:
    errors.append(f"{relation}: cannot read matrix file {path}: {exc}")
    continue
  if spec.get("format")=="COO_RELATION_ROWS":
   entries=matrix_cache[path].get(relation,[])
  elif spec.get("format")=="COO":
   entries=matrix_cache[path]
  else:
   errors.append(f"{relation}: unknown matrix format {spec.get('format')}")
   continue
  if len(entries)!=spec.get("nnz"): errors.append(f"{relation}: nnz mismatch")
  seen=set()
  for entry in entries:
   if not isinstance(entry,list) or len(entry)!=3:
    errors.append(f"{relation}: malformed COO entry {entry!r}");continue
   row,col,weight=entry
   if not isinstance(row,int) or not isinstance(col,int) or not (0<=row<n and 0<=col<n):
    errors.append(f"{relation}: index out of bounds");continue
   if not isinstance(weight,(int,float)) or weight<=0: errors.append(f"{relation}: non-positive weight")
   if (row,col) in seen: errors.append(f"{relation}: duplicate coordinate {(row,col)}")
   seen.add((row,col))
  if not relation.startswith("projection_"): total_edges+=len(entries)

 if total_edges!=graph.get("edge_count"): errors.append("edge_count does not match non-projection matrices")
 completeness=graph.get("completeness",{})
 if require_complete and completeness.get("status")!="complete_structural_index":
  errors.append("graph is not a complete structural index")
 if completeness.get("requires_claim_review") is not True:
  errors.append("graph must disclose pending factual claim review")
 if "references_record" in graph.get("matrices",{}):
  spec=graph["matrices"]["references_record"]
  path=spec["path"]
  entries=matrix_cache.get(path)
  if entries is None:
   try: entries=read_jsonl(path);matrix_cache[path]=entries
   except Exception: entries=[]
  for row,col,_ in entries:
   if row<n and col<n and not id_by_index[col].startswith("record:"):
    errors.append("references_record target is not a curated record node")
    break
 # Validate the topic map and exact provenance edges for all adapted lore records.
 topic_map_path=Path("data/palimpsesto_topic_map_v0.2.jsonl")
 try:
  topic_map=read_jsonl(topic_map_path)
 except Exception as exc:
  errors.append(f"cannot read topic map {topic_map_path}: {exc}")
  topic_map=[]
 if len(topic_map)!=41:
  errors.append(f"expected 41 topic map rows, got {len(topic_map)}")
 if sum(item.get("occurrences",0) for item in topic_map)!=16848:
  errors.append("topic map occurrence total must equal 16848")
 record_nodes={node["id"].removeprefix("record:"):node for node in nodes if node.get("type")=="record"}
 for item in topic_map:
  for field in ("fact_record_ids","lore_record_ids","bridge_record_ids"):
   for linked in item.get(field,[]):
    if linked not in record_nodes:
     errors.append(f"topic map {item.get('topic')}: missing {field} record {linked}")
  if not item.get("fact_record_ids") or not item.get("lore_record_ids") or not item.get("bridge_record_ids"):
   errors.append(f"topic map {item.get('topic')}: missing fact/lore/bridge link")
 source_by_topic={}
 for node in source_records:
  source_by_topic.setdefault(node.get("topic"),[]).append(node["id"])
 adaptations=[node for node in nodes if node.get("type")=="record" and node.get("record_class")=="palimpsest_adaptation"]
 adapted_topics={topic for node in adaptations for topic in node.get("source_topics",[])}
 if adapted_topics!=set(source_by_topic):
  errors.append("adapted lore does not cover exactly the 41 indexed source topics")
 expected_pairs=set()
 for node in adaptations:
  if not node.get("source_topics"):
   errors.append(f"{node['id']}: palimpsest_adaptation missing source_topics")
  for topic in node.get("source_topics",[]):
   if topic not in source_by_topic:
    errors.append(f"{node['id']}: unknown source topic {topic}")
    continue
   for source_id in source_by_topic[topic]:
    expected_pairs.add((id_to_index[node["id"]],id_to_index[source_id]))
 adapt_spec=graph.get("matrices",{}).get("adapted_from")
 if adapt_spec is None:
  errors.append("missing adapted_from matrix")
 else:
  entries=matrix_cache.get(adapt_spec["path"],[])
  actual_pairs={(row[0],row[1]) for row in entries if isinstance(row,list) and len(row)==3}
  if actual_pairs!=expected_pairs:
   errors.append(f"adapted_from mismatch: expected {len(expected_pairs)} links, found {len(actual_pairs)}")
 for item in topic_map:
  mapped=set(item.get("lore_record_ids",[]))
  actual={node["id"].removeprefix("record:") for node in adaptations if item["topic"] in node.get("source_topics",[])}
  if not actual.issubset(mapped):
   errors.append(f"topic map {item['topic']}: adapted lore missing from lore_record_ids")
 return errors

if __name__=="__main__":
 p=argparse.ArgumentParser()
 p.add_argument("graph",nargs="?",default="corpus/graph_v0.5.json")
 p.add_argument("--require-complete",action="store_true")
 a=p.parse_args()
 problems=validate(a.graph,a.require_complete)
 if problems:
  print("GRAPH INVALID")
  for item in problems: print("-",item)
  raise SystemExit(1)
 print("GRAPH VALID")
