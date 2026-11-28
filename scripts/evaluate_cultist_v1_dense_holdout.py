#!/usr/bin/env python3
"""Fixed 120-prompt generative holdout for CULTISTLUMEN-V1-DENSE.
This evaluator deliberately uses the dense model's plain-text training format.
"""
from __future__ import annotations
import argparse, hashlib, json, re, time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from cultistlumen_v1_dense import CultistLumenV1Dense

def load_general(path: Path):
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"(?m)^###\s+\d{3}\.\s+", text)[1:]
    rows = []
    for block in blocks:
        m = re.search(r"(?s)^([^\n]+).*?\*\*Pregunta\*\*\s*\n(.*?)\n\s*\*\*Respuesta del modelo\*\*", block)
        if not m: continue
        heading = m.group(1).strip()
        prompt = re.sub(r"\s+", " ", m.group(2)).strip()
        if prompt:
            rows.append({"id": heading.split("—")[0].strip(), "category": heading.split("—",1)[1].strip() if "—" in heading else "general", "prompt": prompt})
    if len(rows) != 100: raise RuntimeError(f"Expected 100 general prompts; got {len(rows)}")
    return rows

def generate(model, tok, prompt, max_ctx, max_new):
    ids = tok.encode(f"Pregunta: {prompt.strip()}\nRespuesta:", add_special_tokens=False).ids
    generated=[]; eos=tok.token_to_id("[EOS]"); pad=tok.token_to_id("[PAD]")
    with torch.inference_mode():
        for _ in range(max_new):
            logits,_=model(torch.tensor([ids[-max_ctx:]],dtype=torch.long),targets=None)
            nxt=int(torch.argmax(logits[0,-1]).item())
            if nxt in (eos,pad): break
            generated.append(nxt); ids.append(nxt)
    return tok.decode(generated,skip_special_tokens=True).strip(), generated

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True); p.add_argument("--tokenizer",required=True)
    p.add_argument("--general-source",default="HOLDOUT_LUMEN_V22_40K.md")
    p.add_argument("--generative",default="eval/v5_generative_holdout_20.jsonl")
    p.add_argument("--out-dir",default=None)
    p.add_argument("--checkpoint-step",type=int,default=1000)
    p.add_argument("--threads",type=int,default=4); p.add_argument("--context",type=int,default=256)
    p.add_argument("--max-new-tokens",type=int,default=64); a=p.parse_args()
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    raw=Path(a.tokenizer).read_bytes(); expected="91dfa25c95f07ce9e6f3eee94caeb723712d6eaa656a18162adb0d4a047e6d9e"
    actual=hashlib.sha256(raw).hexdigest()
    if actual!=expected: raise RuntimeError(f"Tokenizer SHA mismatch: {actual}")
    tok=Tokenizer.from_file(a.tokenizer); state=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    if state.get("architecture")!="CULTISTLUMEN-V1-DENSE" or int(state.get("step",-1))!=a.checkpoint_step: raise RuntimeError(f"Wrong checkpoint: {state.get('architecture')} step={state.get('step')} expected={a.checkpoint_step}")
    if int(state.get("ctx",-1))!=a.context: raise RuntimeError(f"Checkpoint ctx mismatch: {state.get('ctx')}")
    model_config=state.get("model_config",{"vocab":8192,"d_model":512,"d_ff":1024,"n_head":8,"rank":128,"ctx":a.context})
    if int(model_config.get("ctx",a.context))!=a.context: raise RuntimeError(f"Checkpoint model_config ctx mismatch: {model_config}")
    model=CultistLumenV1Dense(vocab=int(model_config["vocab"]),d_model=int(model_config["d_model"]),d_ff=int(model_config["d_ff"]),ctx=a.context,n_head=int(model_config["n_head"]),rank=int(model_config["rank"]))
    model.load_state_dict(state["model"],strict=True); model.eval()
    general=load_general(Path(a.general_source))
    generative=[json.loads(x) for x in Path(a.generative).read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(generative)!=20 or len({r["id"] for r in generative})!=20: raise RuntimeError(f"Expected 20 unique generative probes, got {len(generative)}")
    tasks=[("general",r) for r in general]+[("generative",r) for r in generative]
    out=Path(a.out_dir or f"artifacts/holdout-{a.checkpoint_step}"); out.mkdir(parents=True,exist_ok=True); results=[]; started=time.time()
    for i,(kind,row) in enumerate(tasks,1):
        answer,ids=generate(model,tok,str(row["prompt"]),a.context,a.max_new_tokens)
        results.append({"id":row["id"],"set":kind,"category":row.get("category","generative"),"prompt":row["prompt"],"response":answer,"token_ids":ids})
        print(f"HOLDOUT {i}/120 set={kind} id={row['id']} chars={len(answer)} response={answer[:140]!r}",flush=True)
    report={"architecture":"CULTISTLUMEN-V1-DENSE","checkpoint_step":a.checkpoint_step,"model_config":model_config,"parameter_count":model.num_params(),"model_config":model_config,"sample_size":len(results),"general_count":len(general),"generative_count":len(generative),"decoding":"greedy argmax; plain-text prompt prefix; no system prompt; max_new_tokens="+str(a.max_new_tokens),"context":a.context,"tokenizer_sha256":actual,"general_source_sha256":hashlib.sha256(Path(a.general_source).read_bytes()).hexdigest(),"generative_source_sha256":hashlib.sha256(Path(a.generative).read_bytes()).hexdigest(),"elapsed_seconds":round(time.time()-started,2),"results":results}
    (out/f"cultistlumen-v1-dense-step-{a.checkpoint_step}-holdout.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=[f"# CULTISTLUMEN-V1-DENSE — Holdout generativo, checkpoint {a.checkpoint_step:,}".replace(",", "."),"","120 prompts fijos: 100 generales reutilizados del holdout histórico + 20 probes generativas V5. Decodificación greedy, contexto 256, máximo 64 tokens nuevos. Prefijo de texto plano Pregunta/Respuesta, sin roles ni system prompt de V2.2.",""]
    for i,r in enumerate(results,1): lines += [f"## {i:03d}. {r['id']} — {r['set']} / {r['category']}","","**Pregunta**",r["prompt"],"","**Respuesta del modelo**",r["response"] or "∅","","---",""]
    (out/f"HOLDOUT_CULTISTLUMEN_V1_DENSE_{a.checkpoint_step}.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("HOLDOUT_COMPLETE "+json.dumps({k:v for k,v in report.items() if k!="results"},ensure_ascii=False),flush=True)

if __name__=="__main__": main()
