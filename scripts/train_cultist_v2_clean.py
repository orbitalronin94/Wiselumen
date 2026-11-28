#!/usr/bin/env python3
"""Train clean causal CULTISTLUMEN v2 from step zero."""
import os,random,time,json,hashlib,gzip,pickle
from pathlib import Path
import torch
from tokenizers import Tokenizer
from cultistlumen_v2_clean import CultistLumenClean
STEPS=10000; BATCH=2; CTX=128; VOCAB=4096; SEED=20261128
TOKENIZER="artifacts/frozen/cultist-tokenizer-v1-8192.json"
TOKENIZER_SHA="91dfa25c95f07ce9e6f3eee94caeb723712d6eaa656a18162adb0d4a047e6d9e"
POOL="artifacts/cultist-v2-clean-pool.pkl.gz"
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS","2"))); torch.set_num_interop_threads(1)
torch.backends.mkldnn.enabled=True
random.seed(SEED); torch.manual_seed(SEED)
raw=Path(TOKENIZER).read_bytes()
if hashlib.sha256(raw).hexdigest()!=TOKENIZER_SHA: raise RuntimeError("Frozen tokenizer SHA mismatch")
Tokenizer.from_file(TOKENIZER)
with gzip.open(POOL,"rb") as f: data=pickle.load(f)
meta=data["metadata"]; shards=data["shards"]
if meta["context"]!=CTX or meta["active_vocab"]!=VOCAB: raise RuntimeError("Pool contract mismatch")
pool_sha=hashlib.sha256(Path(POOL).read_bytes()).hexdigest()
names=list(shards)
if len(names)!=6: raise RuntimeError(f"Expected six curated shards, got {len(names)}")
for name,rows in shards.items():
    if any(max(c)>=VOCAB or min(c)<0 for c in rows): raise RuntimeError(f"Inactive token in {name}")
model=CultistLumenClean(vocab=VOCAB,d=256,heads=4,ff=768,layers=2,ctx=CTX)
opt=torch.optim.AdamW(model.parameters(),lr=0.0003,betas=(0.9,0.95),weight_decay=0.01,foreach=True)
def batch():
    xs=[];ys=[]
    for _ in range(BATCH):
        n=random.choice(names); c=random.choice(shards[n]); xs.append(c[:-1]);ys.append(c[1:])
    return torch.tensor(xs,dtype=torch.long),torch.tensor(ys,dtype=torch.long)
def save(path,step):
    torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),"step":step,"architecture":"CULTISTLUMEN-V2-CAUSAL-TRANSFORMER","model_config":{"vocab":VOCAB,"d_model":256,"heads":4,"layers":2,"d_ff":768,"ctx":CTX,"weight_tying":True},"pool_sha256":pool_sha,"tokenizer_sha256":TOKENIZER_SHA},path)
model.train(); losses=[]; start=time.perf_counter(); out=Path("artifacts");out.mkdir(exist_ok=True)
for step in range(1,STEPS+1):
    x,y=batch(); opt.zero_grad(set_to_none=True)
    _,loss=model(x,y)
    if not torch.isfinite(loss): raise RuntimeError(f"Nonfinite loss at {step}")
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,foreach=True);opt.step()
    losses.append(float(loss.detach()))
    if step==1 or step%100==0 or step==STEPS:
        elapsed=time.perf_counter()-start
        print(f"STEP {step}/{STEPS} loss={losses[-1]:.6f} mean100={sum(losses[-100:])/min(100,len(losses)):.6f} steps_per_sec={step/elapsed:.3f}",flush=True)
    if step in (2500,5000,7500):
        save(out/f"cultistlumen-v2-clean-step-{step}.pt",step)
elapsed=time.perf_counter()-start
save(out/"cultistlumen-v2-clean-step-10000.pt",10000)
metrics={"status":"completed","step":10000,"parameters":model.num_params(),"last_loss":losses[-1],"mean_first_100":sum(losses[:100])/100,"mean_last_100":sum(losses[-100:])/100,"elapsed_seconds":elapsed,"steps_per_second":10000/elapsed,"context":CTX,"vocab":VOCAB,"architecture":"2-layer causal Transformer, 4 heads, d_model=256, d_ff=768","pool_sha256":pool_sha,"tokenizer_sha256":TOKENIZER_SHA,"shards":names}
(out/"cultistlumen-v2-clean-metrics-10000.json").write_text(json.dumps(metrics,indent=2)+"\n")
print("TRAIN_COMPLETE",json.dumps(metrics),flush=True)
