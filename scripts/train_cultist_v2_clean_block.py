#!/usr/bin/env python3
"""Continue CULTISTLUMEN V2 CLEAN from a checkpoint for one 25k block."""
import argparse, os, random, time, json, hashlib, gzip, pickle
from pathlib import Path
import torch
from tokenizers import Tokenizer
from cultistlumen_v2_clean import CultistLumenClean

TOKENIZER="artifacts/frozen/cultist-tokenizer-v1-8192.json"
TOKENIZER_SHA="91dfa25c95f07ce9e6f3eee94caeb723712d6eaa656a18162adb0d4a047e6d9e"
POOL="artifacts/cultist-v2-clean-pool.pkl.gz"
CTX=128; VOCAB=4096; BATCH=2

p=argparse.ArgumentParser()
p.add_argument("--resume", required=True)
p.add_argument("--steps", type=int, default=25000)
p.add_argument("--output", default="artifacts")
p.add_argument("--seed", type=int, default=20261128)
a=p.parse_args()
if a.steps != 25000:
    raise ValueError("This continuation workflow is fixed to 25,000-step blocks")
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS","2")))
torch.set_num_interop_threads(1)
torch.backends.mkldnn.enabled=True
random.seed(a.seed); torch.manual_seed(a.seed)

tokenizer_raw=Path(TOKENIZER).read_bytes()
tokenizer_sha=hashlib.sha256(tokenizer_raw).hexdigest()
if tokenizer_sha != TOKENIZER_SHA:
    raise RuntimeError("Frozen tokenizer SHA mismatch")
Tokenizer.from_file(TOKENIZER)
with gzip.open(POOL,"rb") as f:
    data=pickle.load(f)
meta=data["metadata"]; shards=data["shards"]
if meta["context"] != CTX or meta["active_vocab"] != VOCAB:
    raise RuntimeError("Pool contract mismatch")
pool_sha=hashlib.sha256(Path(POOL).read_bytes()).hexdigest()
if len(shards) != 6:
    raise RuntimeError(f"Expected six curated shards, got {len(shards)}")
for name, rows in shards.items():
    if any(max(c)>=VOCAB or min(c)<0 for c in rows):
        raise RuntimeError(f"Inactive token in {name}")
names=list(shards)

ck=torch.load(a.resume,map_location="cpu",weights_only=False)
cfg=ck["model_config"]
if ck.get("pool_sha256") != pool_sha or ck.get("tokenizer_sha256") != tokenizer_sha:
    raise RuntimeError("Checkpoint/pool/tokenizer provenance mismatch")
model=CultistLumenClean(vocab=cfg["vocab"],d=cfg["d_model"],heads=cfg["heads"],ff=cfg["d_ff"],layers=cfg["layers"],ctx=cfg["ctx"])
model.load_state_dict(ck["model"])
opt=torch.optim.AdamW(model.parameters(),lr=0.0003,betas=(0.9,0.95),weight_decay=0.01,foreach=True)
opt.load_state_dict(ck["optimizer"])
start_step=int(ck["step"])
end_step=start_step+a.steps

def batch():
    xs=[]; ys=[]
    for _ in range(BATCH):
        n=random.choice(names); c=random.choice(shards[n])
        xs.append(c[:-1]); ys.append(c[1:])
    return torch.tensor(xs,dtype=torch.long),torch.tensor(ys,dtype=torch.long)

def save(path, step):
    torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"step":step,
      "architecture":"CULTISTLUMEN-V2-CAUSAL-TRANSFORMER",
      "model_config":cfg,"pool_sha256":pool_sha,"tokenizer_sha256":tokenizer_sha},path)

model.train(); losses=[]; start=time.perf_counter()
out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
for local_step in range(1,a.steps+1):
    x,y=batch(); opt.zero_grad(set_to_none=True)
    _,loss=model(x,y)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Nonfinite loss at global step {start_step+local_step}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,foreach=True)
    opt.step()
    losses.append(float(loss.detach()))
    if local_step==1 or local_step%500==0 or local_step==a.steps:
        elapsed=time.perf_counter()-start
        mean100=sum(losses[-100:])/min(100,len(losses))
        print(f"STEP {start_step+local_step} loss={losses[-1]:.6f} mean100={mean100:.6f} steps_per_sec={local_step/elapsed:.3f}",flush=True)

elapsed=time.perf_counter()-start
ck_path=out/f"cultistlumen-v2-clean-step-{end_step}.pt"
save(ck_path,end_step)
metrics={"status":"completed","start_step":start_step,"end_step":end_step,"block_steps":a.steps,
 "last_loss":losses[-1],"mean_first_100":sum(losses[:100])/100,"mean_last_100":sum(losses[-100:])/100,
 "elapsed_seconds":elapsed,"steps_per_second":a.steps/elapsed,"context":CTX,"vocab":VOCAB,
 "architecture":"2-layer causal Transformer, 4 heads, d_model=256, d_ff=768",
 "parameters":model.num_params(),"pool_sha256":pool_sha,"tokenizer_sha256":tokenizer_sha}
(out/f"cultistlumen-v2-clean-metrics-{end_step}.json").write_text(json.dumps(metrics,indent=2)+"\\n")
print("BLOCK_COMPLETE",json.dumps(metrics),flush=True)
