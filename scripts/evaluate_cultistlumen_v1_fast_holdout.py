#!/usr/bin/env python3
"""Fixed 120-prompt generative holdout for CULTISTLUMEN-V1-FAST."""
from __future__ import annotations
import argparse, hashlib, json, re, time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from cultistlumen_v1_fast import CultistLumenV1Fast

TOKENIZER_SHA256="91dfa25c95f07ce9e6f3eee94caeb723712d6eaa656a18162adb0d4a047e6d9e"
MODEL_CONFIG={"vocab":8192,"d_model":512,"d_ff":2048,"n_head":0,"rank":64,"ctx":128,"conv_kernel":4,"mixer":"CausalDepthwiseConvLowRank","ffn":"DenseGatedFFN","weight_tying":True,"sampled_negatives":256}

def load_general(path):
    text=path.read_text(encoding="utf-8")
    blocks=re.split(r"(?m)^###\s+\d{3}\.\s+",text)[1:]
    rows=[]
    for block in blocks:
        m=re.search(r"(?s)^([^\n]+).*?\*\*Pregunta\*\*\s*\n(.*?)\n\s*\*\*Respuesta del modelo\*\*",block)
        if not m: continue
        heading=m.group(1).strip()
        prompt=re.sub(r"\s+"," ",m.group(2)).strip()
        if prompt:
            rows.append({"id":heading.split("—")[0].strip(),"category":heading.split("—",1)[1].strip() if "—" in heading else "general","prompt":prompt})
    if len(rows)!=100: raise RuntimeError(f"Expected 100 general prompts; got {len(rows)}")
    return rows

def generate(model,tok,prompt,context,max_new):
    # Match the training serialization: prompt + one separating space + completion.
    ids=tok.encode(prompt.strip()+" ",add_special_tokens=False).ids
    generated=[]; eos=tok.token_to_id("[EOS]"); pad=tok.token_to_id("[PAD]"); ended_with_eos=False
    with torch.inference_mode():
        for _ in range(max_new):
            logits,_=model(torch.tensor([ids[-context:]],dtype=torch.long),targets=None)
            nxt=int(torch.argmax(logits[0,-1]).item())
            if nxt in (eos,pad):
                ended_with_eos=(nxt==eos)
                break
            generated.append(nxt); ids.append(nxt)
    return tok.decode(generated,skip_special_tokens=True).strip(),generated,ended_with_eos

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True); p.add_argument("--tokenizer",required=True)
    p.add_argument("--general-source",default="HOLDOUT_LUMEN_V22_40K.md")
    p.add_argument("--generative",default="eval/v5_generative_holdout_20.jsonl")
    p.add_argument("--out-dir",default="artifacts/holdout")
    p.add_argument("--checkpoint-step",type=int,default=10000)
    p.add_argument("--threads",type=int,default=2); p.add_argument("--context",type=int,default=128)
    p.add_argument("--max-new-tokens",type=int,default=64); a=p.parse_args()
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    raw=Path(a.tokenizer).read_bytes(); actual=hashlib.sha256(raw).hexdigest()
    if actual!=TOKENIZER_SHA256: raise RuntimeError(f"Tokenizer SHA mismatch: {actual}")
    tok=Tokenizer.from_file(a.tokenizer)
    state=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    if state.get("architecture")!="CULTISTLUMEN-V1-FAST" or int(state.get("step",-1))!=a.checkpoint_step:
        raise RuntimeError(f"Wrong checkpoint: architecture={state.get('architecture')} step={state.get('step')}")
    expected={"model_config":MODEL_CONFIG,"vocab":8192,"ctx":128,"corpus_version":"v4.0.0-pre-marper","tokenizer_sha256":TOKENIZER_SHA256,"sampling":{"world":1.0,"full_marper_lore":0.0,"conflicts":0.0,"weighting":"uncompressed source-file byte size; factual V4 only"}}
    for key,value in expected.items():
        if state.get(key)!=value: raise RuntimeError(f"Checkpoint contract mismatch for {key}: {state.get(key)!r}")
    model=CultistLumenV1Fast(vocab=8192,d_model=512,d_ff=2048,ctx=128,n_head=0,rank=64)
    model.load_state_dict(state["model"],strict=True); model.eval()
    general=load_general(Path(a.general_source))
    generative=[json.loads(x) for x in Path(a.generative).read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(generative)!=20 or len({r["id"] for r in generative})!=20: raise RuntimeError(f"Expected 20 unique generative probes, got {len(generative)}")
    tasks=[("general",r) for r in general]+[("generative",r) for r in generative]
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True); results=[]; started=time.time()
    for i,(kind,row) in enumerate(tasks,1):
        answer,ids,ended_with_eos=generate(model,tok,str(row["prompt"]),a.context,a.max_new_tokens)
        results.append({"id":row["id"],"set":kind,"category":row.get("category","generative"),"prompt":row["prompt"],"response":answer,"token_ids":ids,"ended_with_eos":ended_with_eos})
        print(f"HOLDOUT {i}/120 set={kind} id={row['id']} chars={len(answer)} response={answer[:140]!r}",flush=True)
    checkpoint_sha=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    unique_responses=len({r["response"] for r in results})
    eos_terminated=sum(1 for r in results if r["ended_with_eos"])
    capped=sum(1 for r in results if len(r["token_ids"])>=a.max_new_tokens)
    empty=sum(1 for r in results if not r["response"])
    report={"architecture":"CULTISTLUMEN-V1-FAST","checkpoint_step":a.checkpoint_step,"checkpoint_sha256":checkpoint_sha,"model_config":MODEL_CONFIG,"parameter_count":sum(p.numel() for p in model.parameters()),"sample_size":len(results),"general_count":len(general),"generative_count":len(generative),"unique_response_count":unique_responses,"eos_terminated_count":eos_terminated,"max_token_cap_count":capped,"empty_response_count":empty,"decoding":f"greedy argmax; prompt+space prefix matching train serialization; max_new_tokens={a.max_new_tokens}","context":a.context,"tokenizer_sha256":actual,"general_source_sha256":hashlib.sha256(Path(a.general_source).read_bytes()).hexdigest(),"generative_source_sha256":hashlib.sha256(Path(a.generative).read_bytes()).hexdigest(),"elapsed_seconds":round(time.time()-started,2),"results":results}
    step_label=f"{a.checkpoint_step//1000}K" if a.checkpoint_step%1000==0 else str(a.checkpoint_step)
    (out/f"cultistlumen-v1-fast-step-{a.checkpoint_step}-holdout.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=[f"# CULTISTLUMEN-V1-FAST — Holdout generativo, checkpoint {a.checkpoint_step:,}","",f"120 prompts fijos: 100 generales históricos + 20 probes V5. Decodificación greedy, contexto 128, máximo {a.max_new_tokens} tokens nuevos. Prefijo prompt+espacio, igual que la serialización de entrenamiento; sin system prompt.","",f"- Respuestas únicas: {unique_responses}/120",f"- EOS emitido: {eos_terminated}/120",f"- Alcanzaron el límite de tokens: {capped}/120",f"- Respuestas vacías: {empty}/120",""]
    for i,r in enumerate(results,1):
        lines += [f"## {i:03d}. {r['id']} — {r['set']} / {r['category']}","","**Pregunta**",r["prompt"],"","**Respuesta del modelo**",r["response"] or "∅","","---",""]
    (out/f"HOLDOUT_CULTISTLUMEN_V1_FAST_{step_label}.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("HOLDOUT_COMPLETE "+json.dumps({k:v for k,v in report.items() if k!="results"},ensure_ascii=False),flush=True)

if __name__=="__main__": main()
