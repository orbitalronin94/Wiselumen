#!/usr/bin/env python3
"""Train CULTISTLUMEN-V1-DENSE from scratch on the frozen V5 90/10 corpus."""
from __future__ import annotations
import argparse, bisect, gzip, hashlib, json, math, os, pickle, random, time, zipfile, io
from pathlib import Path
import torch
from tokenizers import Tokenizer
from cultistlumen_v1_dense import CultistLumenV1Dense

SEED=20261127
STEPS=int(os.environ.get("STEPS","1000"))
TOTAL_TRAIN_STEPS=int(os.environ.get("TOTAL_TRAIN_STEPS",str(STEPS)))
BATCH_SIZE=int(os.environ.get("BATCH_SIZE","4"))
CTX=int(os.environ.get("CTX","256"))
WARMUP=int(os.environ.get("WARMUP","50"))
BASE_LR_MAIN=float(os.environ.get("BASE_LR_MAIN","8e-4"))
BASE_LR_EMBED=float(os.environ.get("BASE_LR_EMBED","4e-4"))
THREADS=int(os.environ.get("OMP_NUM_THREADS","4"))
INTEROP_THREADS=int(os.environ.get("TORCH_INTEROP_THREADS","4"))
PROFILE=os.environ.get("PROFILE","1")=="1"
SAMPLED_NEGATIVES=int(os.environ.get("SAMPLED_NEGATIVES","1024"))
MAX_CHUNKS_PER_FILE=int(os.environ.get("MAX_CHUNKS_PER_FILE","2048"))
PARENT_CHECKPOINT=Path(os.environ.get("PARENT_CHECKPOINT","")) if os.environ.get("PARENT_CHECKPOINT") else None
V5_PATH=Path(os.environ.get("V5_ARCHIVE","artifacts/wise-ronin-corpus-v5.0.0.zip"))
TOKENIZER_PATH=Path(os.environ.get("FROZEN_TOKENIZER","artifacts/frozen/cultist-tokenizer-v1-8192.json"))
TOKENIZER_SHA256="91dfa25c95f07ce9e6f3eee94caeb723712d6eaa656a18162adb0d4a047e6d9e"
OUT=Path("artifacts")
POOL_PATH=Path(os.environ.get("POOL_ARCHIVE","artifacts/cultist-v1-pool-ctx256.pkl.gz"))
MARPER_FILES=[
 "lore/Marper-Codice-Juntado (1).txt","lore/lore_seed_v0.6.jsonl",
 "lore/palimpsesto_topic_map_v0.3.jsonl","lore/palimpsesto_index_v0.1_part1.jsonl",
 "lore/palimpsesto_index_v0.1_part2.jsonl","lore/palimpsesto_index_v0.1_part3.jsonl",
 "lore/palimpsesto_index_v0.1_part4.jsonl"]

def sha256(data:bytes)->str: return hashlib.sha256(data).hexdigest()

def load_frozen_tokenizer():
    raw=TOKENIZER_PATH.read_bytes()
    actual=sha256(raw)
    if actual!=TOKENIZER_SHA256:
        raise RuntimeError(f"Frozen tokenizer SHA256 mismatch: {actual}")
    tok=Tokenizer.from_file(str(TOKENIZER_PATH))
    vocab=tok.get_vocab_size(with_added_tokens=True)
    if vocab!=8192 or tok.token_to_id("[PAD]")!=0 or tok.token_to_id("[EOS]")!=3:
        raise RuntimeError(f"Frozen tokenizer contract mismatch: vocab={vocab}")
    print(f"FROZEN_TOKENIZER sha256={actual} vocab={vocab}",flush=True)
    return tok

def make_chunks(tokenizer,text,ctx,cap,rng):
    ids=tokenizer.encode(text,add_special_tokens=False).ids
    width=ctx+1; chunks=[]; total=0
    for start in range(0,len(ids)-width+1,width):
        chunk=ids[start:start+width]; total+=1
        if len(chunks)<cap: chunks.append(chunk)
        else:
            slot=rng.randrange(total)
            if slot<cap: chunks[slot]=chunk
    return chunks,total

def build_file_pools(zf,manifest,tokenizer,nested_bytes):
    pools={"world":{},"marper":{}}; weights={"world":{},"marper":{}}
    rng=random.Random(SEED+91)
    nested=zipfile.ZipFile(io.BytesIO(nested_bytes))
    for meta in manifest["shards"]:
        member="corpus/"+meta["file"]; chunks=[]; buffer=[]; seen_rows=0; seen_chunks=0
        with nested.open(member) as handle:
            for raw in handle:
                if not raw.strip(): continue
                row=json.loads(raw)
                if row.get("split")=="holdout" or row.get("source")=="holdout_500.jsonl": continue
                prompt=row.get("prompt"); completion=row.get("completion")
                if not isinstance(prompt,str) or not isinstance(completion,str): continue
                buffer.extend(tokenizer.encode(prompt.strip()+" "+completion.strip(),add_special_tokens=False).ids)
                buffer.append(3)
                seen_rows+=1
                while len(buffer)>=CTX+1:
                    chunk=buffer[:CTX+1]; del buffer[:CTX+1]; seen_chunks+=1
                    if len(chunks)<MAX_CHUNKS_PER_FILE: chunks.append(chunk)
                    else:
                        slot=rng.randrange(seen_chunks)
                        if slot<MAX_CHUNKS_PER_FILE: chunks[slot]=chunk
        if not chunks: raise RuntimeError(f"No usable chunks in {member}")
        pools["world"][member]=chunks; weights["world"][member]=int(meta["bytes"])
        print(f"POOL world file={member} bytes={weights['world'][member]} rows={seen_rows} chunks={len(chunks)}",flush=True)
    for member in MARPER_FILES:
        raw=zf.read(member)
        chunks,total=make_chunks(tokenizer,raw.decode("utf-8"),CTX,MAX_CHUNKS_PER_FILE,rng)
        if not chunks: raise RuntimeError(f"No chunks in Marper source {member}")
        pools["marper"][member]=chunks; weights["marper"][member]=len(raw)
        print(f"POOL marper file={member} bytes={len(raw)} chunks={len(chunks)} total_chunks={total}",flush=True)
    nested.close()
    if not pools["world"] or not pools["marper"]: raise RuntimeError("Both pools must be non-empty")
    return pools,weights


def save_pool_archive(pools,weights,corpus_sha):
    metadata={"format":"cultist-token-pool-v1","context":CTX,"tokenizer_sha256":TOKENIZER_SHA256,
              "corpus_version":"v5.0.0","corpus_v5_sha256":corpus_sha,
              "sampling":{"world":0.90,"marper":0.10,"conflicts":0.0},
              "weighting":"uncompressed source-file byte size within each pool",
              "max_chunks_per_file":MAX_CHUNKS_PER_FILE,"seed":SEED+91}
    with gzip.open(POOL_PATH,"wb",compresslevel=1) as handle:
        pickle.dump({"metadata":metadata,"pools":pools,"weights":weights},handle,protocol=4)
    digest=hashlib.sha256(POOL_PATH.read_bytes()).hexdigest()
    (OUT/"cultist-v1-pool-manifest.json").write_text(json.dumps({**metadata,"pool_sha256":digest,"pool_file":str(POOL_PATH)},indent=2)+"\\n",encoding="utf-8")
    print(f"POOL_ARCHIVE_READY file={POOL_PATH} sha256={digest} corpus_sha256={corpus_sha} bytes={POOL_PATH.stat().st_size}",flush=True)

def load_pool_archive():
    if not POOL_PATH.is_file(): raise RuntimeError(f"Reusable pool artifact missing: {POOL_PATH}; prepare it in the pool job first")
    with gzip.open(POOL_PATH,"rb") as handle: payload=pickle.load(handle)
    meta=payload["metadata"]
    if meta.get("format")!="cultist-token-pool-v1" or meta.get("context")!=CTX or meta.get("tokenizer_sha256")!=TOKENIZER_SHA256 or meta.get("corpus_version")!="v5.0.0" or meta.get("sampling")!={"world":0.90,"marper":0.10,"conflicts":0.0}:
        raise RuntimeError(f"Reusable pool contract mismatch: {meta}")
    digest=hashlib.sha256(POOL_PATH.read_bytes()).hexdigest()
    print(f"POOL_ARCHIVE_REUSED file={POOL_PATH} sha256={digest} corpus_sha256={meta['corpus_v5_sha256']} bytes={POOL_PATH.stat().st_size}",flush=True)
    return payload["pools"],payload["weights"],meta,digest

class WeightedPoolSampler:
    # Precompute nombres y sumas acumuladas una sola vez; no reconstruir
    # list(pool), sum(weights) ni recorrer todos los ficheros en cada muestra.
    def __init__(self,pool,weights):
        self.names=tuple(pool)
        total=0; self.cumulative=[]
        for name in self.names:
            total+=int(weights[name]); self.cumulative.append(total)
        self.total=total
        self.pool=pool
    def sample(self,rng):
        point=rng.randrange(self.total)
        index=bisect.bisect_right(self.cumulative,point)
        return rng.choice(self.pool[self.names[index]])

def build_step_sources(steps,batch_size,rng):
    if steps%10: raise ValueError("STEPS must be a multiple of 10 for exact 90/10 schedule")
    schedule=[]
    for _ in range(steps//10):
        tags=["world"]*(batch_size*9)+["marper"]*batch_size
        rng.shuffle(tags)
        schedule.extend([tags[i:i+batch_size] for i in range(0,len(tags),batch_size)])
    return schedule

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--build-pool-only',action='store_true')
    args=parser.parse_args()
    if STEPS<=0 or BATCH_SIZE<=0 or CTX<=0 or STEPS%10 or TOTAL_TRAIN_STEPS<=0: raise ValueError("Invalid training dimensions")
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(THREADS)
    torch.set_num_interop_threads(INTEROP_THREADS)
    torch.backends.mkldnn.enabled=True
    random.seed(SEED); torch.manual_seed(SEED)
    rng=random.Random(SEED+1000+int(os.environ.get("PARENT_STEP_OFFSET","0")))
    print(f"CULTISTLUMEN_V1_DENSE_CONFIG steps={STEPS} batch={BATCH_SIZE} ctx={CTX} warmup={WARMUP} lr_main={BASE_LR_MAIN} lr_embed={BASE_LR_EMBED} intra={THREADS} inter={INTEROP_THREADS} ffn_compute=fp32 master_weights=fp32 sampled_negatives={SAMPLED_NEGATIVES}",flush=True)
    tokenizer=load_frozen_tokenizer()
    if args.build_pool_only:
        if not V5_PATH.is_file(): raise RuntimeError(f"V5 corpus bundle missing: {V5_PATH}")
        with zipfile.ZipFile(V5_PATH) as outer:
            v5_manifest=json.loads(outer.read("corpus/v5/manifest.json"))
            if v5_manifest.get("version")!="v5.0.0" or not v5_manifest.get("training_authorized"):
                raise RuntimeError("V5 manifest is not authorized")
            expected={"v4_factual":0.90,"full_marper_lore":0.10,"synthetic_conflicts":0.0}
            if v5_manifest.get("sampling",{}).get("shares")!=expected: raise RuntimeError("V5 sampling policy mismatch")
            nested_bytes=outer.read("corpus/v4/wise-ronin-corpus-v4.0.0.zip")
            if sha256(nested_bytes)!=v5_manifest["v4"]["zip_sha256"]: raise RuntimeError("Nested V4 checksum mismatch")
            with zipfile.ZipFile(io.BytesIO(nested_bytes)) as nested:
                manifest=json.loads(nested.read("corpus/manifest.json"))
                if manifest.get("version")!="v4.0.0" or manifest.get("status")!="FROZEN_TRAINABLE" or manifest.get("training_authorized") is not True:
                    raise RuntimeError("V4 manifest is not authorized")
                pools,weights=build_file_pools(outer,manifest,tokenizer,nested_bytes)
        corpus_sha=hashlib.sha256(V5_PATH.read_bytes()).hexdigest()
        save_pool_archive(pools,weights,corpus_sha)
        if args.build_pool_only: return
    else:
        pools,weights,pool_meta,pool_sha=load_pool_archive()
    model=CultistLumenV1Dense(vocab=8192,d_model=512,d_ff=1024,ctx=CTX)
    parent_meta=None
    parent_payload=None
    start_step=0
    if PARENT_CHECKPOINT is not None:
        if not PARENT_CHECKPOINT.is_file(): raise RuntimeError(f"Parent checkpoint missing: {PARENT_CHECKPOINT}")
        parent=torch.load(PARENT_CHECKPOINT,map_location="cpu",weights_only=False)
        parent_payload=parent
        required={"architecture":"CULTISTLUMEN-V1-DENSE","vocab":8192,"ctx":CTX,"corpus_version":"v5.0.0","tokenizer_sha256":TOKENIZER_SHA256,"pool_sha256":pool_sha,"corpus_v5_sha256":pool_meta["corpus_v5_sha256"]}
        for key,value in required.items():
            if parent.get(key)!=value: raise RuntimeError(f"Parent checkpoint contract mismatch for {key}: {parent.get(key)!r} != {value!r}")
        start_step=int(parent.get("step",0))
        if start_step<=0: raise RuntimeError(f"Invalid parent step: {start_step}")
        model.load_state_dict(parent["model"],strict=True)
        parent_meta={k:parent.get(k) for k in ("step","architecture","vocab","ctx","corpus_version","tokenizer_sha256","pool_sha256","corpus_v5_sha256","sampling","precision")}
        print(f"RESUMED_PARENT checkpoint={PARENT_CHECKPOINT} start_step={start_step} pool_sha256={pool_sha}",flush=True)
    ffn_params=list(model.ffn.parameters())
    mix_params=list(model.mix.parameters())+list(model.norm1.parameters())+list(model.norm2.parameters())+list(model.norm.parameters())
    embed_params=list(model.tok.parameters())
    optimizer=torch.optim.AdamW([
      {"params":ffn_params+mix_params,"lr":BASE_LR_MAIN,"base_lr":BASE_LR_MAIN,"weight_decay":0.01},
      {"params":embed_params,"lr":BASE_LR_EMBED,"base_lr":BASE_LR_EMBED,"weight_decay":0.01},
    ],betas=(0.9,0.95),eps=1e-8,foreach=True)
    optimizer_state_restored=False
    if parent_payload is not None and parent_payload.get("optimizer") is not None:
        optimizer.load_state_dict(parent_payload["optimizer"])
        optimizer_state_restored=True
        print(f"OPTIMIZER_STATE_RESTORED parent_step={start_step}",flush=True)
    elif parent_payload is not None:
        print(f"OPTIMIZER_STATE_NOT_PRESENT parent_step={start_step}; initializing AdamW state for this first continuation block",flush=True)
    print(f"MODEL architecture=CULTISTLUMEN-V1-DENSE total_params={model.num_params():,} experts=0",flush=True)
    schedule=build_step_sources(STEPS,BATCH_SIZE,rng)
    samplers={tag:WeightedPoolSampler(pools[tag],weights[tag]) for tag in ("world","marper")}
    from torch.profiler import profile, ProfilerActivity, record_function
    def make_batch(tags):
        samples=[samplers[tag].sample(rng) for tag in tags]
        return (torch.tensor([sample[:-1] for sample in samples],dtype=torch.long),
                torch.tensor([sample[1:] for sample in samples],dtype=torch.long))
    # Perfilado de diagnóstico fuera del cronómetro: una pasada forward/backward
    # sin optimizer.step; la primera batch se reutiliza para el paso 1 real.
    first_batch=make_batch(schedule[0])
    if PROFILE:
        optimizer.zero_grad(set_to_none=True)
        with profile(activities=[ProfilerActivity.CPU],record_shapes=False,profile_memory=False) as prof:
            with record_function("train/forward"):
                _,probe_loss=model(first_batch[0],targets=first_batch[1],sampled_negatives=SAMPLED_NEGATIVES)
            with record_function("train/backward"):
                probe_loss.backward()
        print("CPU_OPERATOR_PROFILE\\n"+prof.key_averages().table(sort_by="self_cpu_time_total",row_limit=24),flush=True)
        optimizer.zero_grad(set_to_none=True)
    started=time.perf_counter(); losses=[]
    phase={"batch":0.0,"forward":0.0,"backward":0.0,"clip":0.0,"optimizer":0.0}
    model.train()
    for step,tags in enumerate(schedule,1):
        global_step=start_step+step
        if global_step<=WARMUP: scale=global_step/max(1,WARMUP)
        else:
            progress=(global_step-WARMUP)/max(1,TOTAL_TRAIN_STEPS-WARMUP)
            scale=0.5*(1.0+math.cos(math.pi*min(1.0,progress)))
        for group in optimizer.param_groups: group["lr"]=group["base_lr"]*scale
        t0=time.perf_counter()
        if step==1: x,y=first_batch
        else: x,y=make_batch(tags)
        phase["batch"]+=time.perf_counter()-t0
        optimizer.zero_grad(set_to_none=True)
        t0=time.perf_counter()
        with record_function("train/forward"):
            _,loss=model(x,targets=y,sampled_negatives=SAMPLED_NEGATIVES)
        phase["forward"]+=time.perf_counter()-t0
        if loss is None or not math.isfinite(float(loss.detach())):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss}")
        t0=time.perf_counter()
        with record_function("train/backward"):
            loss.backward()
        phase["backward"]+=time.perf_counter()-t0
        t0=time.perf_counter()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,foreach=True)
        phase["clip"]+=time.perf_counter()-t0
        t0=time.perf_counter()
        with record_function("train/optimizer"):
            optimizer.step()
        phase["optimizer"]+=time.perf_counter()-t0
        value=float(loss.detach()); losses.append(value)
        if step==1 or step%10==0 or step==STEPS:
            elapsed=time.perf_counter()-started
            print(f"STEP {step}/{STEPS} loss={value:.6f} lr_main={optimizer.param_groups[0]['lr']:.7g} lr_embed={optimizer.param_groups[1]['lr']:.7g} steps_per_sec={step/max(elapsed,1e-9):.3f}",flush=True)
    elapsed=time.perf_counter()-started
    phase_total=sum(phase.values())
    print("PHASE_TIMING_SECONDS "+json.dumps({k:round(v,4) for k,v in phase.items()}|{"measured_total":round(phase_total,4),"wall_total":round(elapsed,4)},sort_keys=True),flush=True)
    final_step=start_step+STEPS
    if final_step>TOTAL_TRAIN_STEPS: raise RuntimeError(f"Block ends at {final_step}, beyond TOTAL_TRAIN_STEPS={TOTAL_TRAIN_STEPS}")
    checkpoint=OUT/f"cultistlumen-v1-dense-step-{final_step}.pt"
    torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":final_step,"architecture":"CULTISTLUMEN-V1-DENSE","vocab":8192,"ctx":CTX,"seed":SEED,"corpus_version":"v5.0.0","tokenizer_sha256":TOKENIZER_SHA256,"pool_sha256":pool_sha,"corpus_v5_sha256":pool_meta["corpus_v5_sha256"],"sampling":{"world":0.90,"full_marper_lore":0.10,"conflicts":0.0,"weighting":"uncompressed source-file byte size within each pool"},"precision":{"master_weights":"float32","ffn_compute":"float32","mixer_solver":"float32","loss":"float32","vocab_head":"sampled_softmax"},"parent_checkpoint":str(PARENT_CHECKPOINT) if PARENT_CHECKPOINT else None,"parent_step":start_step},checkpoint)
    metrics={"status":"completed","architecture":"CULTISTLUMEN-V1-DENSE","steps":STEPS,"start_step":start_step,"global_step":final_step,"total_train_steps":TOTAL_TRAIN_STEPS,"optimizer_state_restored":optimizer_state_restored,"parent_checkpoint":str(PARENT_CHECKPOINT) if PARENT_CHECKPOINT else None,"parent_step":start_step,"batch_size":BATCH_SIZE,"context":CTX,"warmup":WARMUP,"base_lr_main":BASE_LR_MAIN,"base_lr_embed":BASE_LR_EMBED,"intraop_threads":THREADS,"interop_threads":INTEROP_THREADS,"precision":{"master_weights":"float32","ffn_compute":"float32","mixer_solver":"float32","loss":"float32","vocab_head":"sampled_softmax"},"last_loss":losses[-1],"mean_first_10":sum(losses[:10])/min(10,len(losses)),"mean_last_10":sum(losses[-10:])/min(10,len(losses)),"elapsed_seconds":elapsed,"steps_per_second":STEPS/elapsed,"model_parameters":model.num_params(),"experts":0,"sampled_negatives":SAMPLED_NEGATIVES,"training_objective":"sampled softmax: all batch targets plus uniform sampled negatives; inference uses full vocabulary","phase_timing_seconds":phase,"sampling":{"world":0.90,"full_marper_lore":0.10,"conflicts":0.0},"file_weights_bytes":weights,"tokenizer_sha256":TOKENIZER_SHA256,"pool_sha256":pool_sha,"corpus_v5_sha256":pool_meta["corpus_v5_sha256"],"checkpoint":str(checkpoint)}
    (OUT/f"cultistlumen-v1-dense-metrics-step-{final_step}.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("CULTISTLUMEN_DENSE_RESULT "+json.dumps(metrics,ensure_ascii=False),flush=True)

if __name__=="__main__": main()
