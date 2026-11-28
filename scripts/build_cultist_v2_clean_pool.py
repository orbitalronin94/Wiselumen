#!/usr/bin/env python3
"""Build a balanced clean pool from the frozen pre-Marper pool, excluding OCR-heavy shards."""
import gzip,pickle,hashlib,json,random
from pathlib import Path
SRC=Path("artifacts/cultist-v1-pool-ctx128-premarper.pkl.gz")
OUT=Path("artifacts/cultist-v2-clean-pool.pkl.gz")
KEEP=("corpus-v4-00000.jsonl","corpus-v4-00001.jsonl","corpus-v4-00002.jsonl","corpus-v4-00021.jsonl","corpus-v4-00022.jsonl","corpus-v4-00023.jsonl")
with gzip.open(SRC,"rb") as f: p=pickle.load(f)
world=p["pools"]["world"]; selected={k:v for k,v in world.items() if any(k.endswith(x) for x in KEEP)}
if len(selected)!=len(KEEP): raise RuntimeError(f"Expected {len(KEEP)} clean shards, got {list(selected)}")
clean={}
for name,chunks in selected.items():
    rows=[list(map(int,c)) for c in chunks if len(c)==129 and all(0<=int(x)<4096 for x in c)]
    if not rows: raise RuntimeError(f"No valid chunks in {name}")
    clean[name]=rows
meta={"format":"cultist-clean-pool-v2","source_pool_sha256":hashlib.sha256(SRC.read_bytes()).hexdigest(),"excluded_shards":"corpus-v4-00003..00020.jsonl","included_shards":list(clean),"context":128,"active_vocab":4096,"sampling":"uniform over six retained shards; uniform chunk within shard","boundary_note":"Retains EOS separators present in source pool; source pool is prepacked, so original record boundaries cannot be reconstructed."}
payload={"metadata":meta,"shards":clean}
OUT.parent.mkdir(exist_ok=True)
with gzip.open(OUT,"wb",compresslevel=1) as f: pickle.dump(payload,f,protocol=4)
meta["pool_sha256"]=hashlib.sha256(OUT.read_bytes()).hexdigest()
Path("artifacts/cultist-v2-clean-pool-manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
print("CLEAN_POOL_READY",json.dumps(meta),flush=True)
