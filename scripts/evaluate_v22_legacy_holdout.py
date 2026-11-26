"""Run the legacy 2,431-item MC and 100-question freeform holdouts on LUMEN V2.2."""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from Model import LumenV2, active_layers_for_step
from data import ROLE_IDS, validate_tokenizer

def load_rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

def normalize_answer(answer, choices):
    if isinstance(answer, int):
        return answer if 0 <= answer < len(choices) else None
    if isinstance(answer, str):
        s=answer.strip()
        if s.isdigit() and 0 <= int(s) < len(choices): return int(s)
        if len(s)==1 and s.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            i=ord(s.upper())-65
            if i<len(choices): return i
        for i,c in enumerate(choices):
            if s==str(c).strip(): return i
    return None

def chat_prefix(tok, system_ids, prompt):
    user_ids=tok.encode(str(prompt).strip(),add_special_tokens=False).ids
    if any(i>=8184 for i in user_ids): raise ValueError("Ordinary text collided with reserved role IDs")
    return [ROLE_IDS["<|bos|>"],ROLE_IDS["<|system|>"]]+system_ids+[ROLE_IDS["<|end_system|>"],ROLE_IDS["<|user|>"]]+user_ids+[ROLE_IDS["<|end_user|>"],ROLE_IDS["<|assistant|>"]]

def candidate_nll(model,tok,prefix,choice,active,max_len):
    suffix=tok.encode("\n"+str(choice),add_special_tokens=False).ids
    if not suffix: return float("inf")
    if len(suffix)>=max_len: suffix=suffix[:max_len-1]
    context=prefix[-(max_len-len(suffix)):]
    ids=context+suffix
    if len(ids)<2: return float("inf")
    x=torch.tensor([ids],dtype=torch.long)
    logits,_=model(x,active_layers=active,return_logits=True)
    start=len(context)-1
    pred=logits[0,start:start+len(suffix),:]
    target=torch.tensor(suffix,dtype=torch.long)
    return float(F.cross_entropy(pred,target,reduction="mean").item())

def generate(model,tok,prefix,active,max_len,max_new):
    generated=[]
    eos=tok.token_to_id("[EOS]")
    pad=tok.token_to_id("[PAD]")
    for _ in range(max_new):
        context=(prefix+generated)[-max_len:]
        x=torch.tensor([context],dtype=torch.long)
        logits,_=model(x,active_layers=active,return_logits=True)
        nxt=int(torch.argmax(logits[0,-1]).item())
        if nxt in (eos,pad,ROLE_IDS["<|end_assistant|>"]): break
        generated.append(nxt)
    return tok.decode(generated,skip_special_tokens=True).strip(),generated

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True); p.add_argument("--tokenizer",required=True)
    p.add_argument("--system",required=True); p.add_argument("--mc-holdout",required=True)
    p.add_argument("--freeform-holdout",required=True); p.add_argument("--out-dir",required=True)
    p.add_argument("--step",type=int,default=10000); p.add_argument("--threads",type=int,default=4)
    p.add_argument("--max-len",type=int,default=512); p.add_argument("--max-new-tokens",type=int,default=64)
    a=p.parse_args()
    torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
    tok=Tokenizer.from_file(a.tokenizer); validate_tokenizer(tok)
    state=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    assert int(state["step"])==a.step, state.get("step")
    model=LumenV2(vocab=8192,ctx=a.max_len,loss_mode="experimental")
    model.load_state_dict(state["model"],strict=True); model.eval()
    active=active_layers_for_step(a.step,"v22",len(model.blocks))
    system=Path(a.system).read_text(encoding="utf-8").strip()
    sys_ids=tok.encode(system,add_special_tokens=False).ids
    if any(i>=8184 for i in sys_ids): raise ValueError("System prompt collides with reserved IDs")
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)
    mc=load_rows(a.mc_holdout); mc_results=[]; correct=0; total=0; nll_sum=0.0
    with torch.inference_mode():
        for n,row in enumerate(mc,1):
            prompt=str(row.get("prompt","")).strip()
            choices=row.get("choices",[])
            if isinstance(choices,dict): choices=[choices[k] for k in sorted(choices)]
            gold=normalize_answer(row.get("answer"),choices)
            if not prompt or not isinstance(choices,list) or len(choices)<2 or gold is None: continue
            prefix=chat_prefix(tok,sys_ids,prompt)
            scores=[candidate_nll(model,tok,prefix,c,active,a.max_len) for c in choices]
            pred=min(range(len(scores)),key=scores.__getitem__)
            total+=1; correct+=int(pred==gold); nll_sum+=scores[gold]
            mc_results.append({"id":row.get("id",n),"gold":gold,"pred":pred,"scores":scores})
            if n%100==0: print(f"mc_progress {n}/{len(mc)}",flush=True)
    mc_report={"checkpoint_step":a.step,"rows_in_source":len(mc),"scored":total,"correct":correct,
       "accuracy":correct/max(1,total),"mean_gold_answer_nll":nll_sum/max(1,total),
       "holdout_sha256":hashlib.sha256(Path(a.mc_holdout).read_bytes()).hexdigest(),
       "tokenizer_sha256":hashlib.sha256(Path(a.tokenizer).read_bytes()).hexdigest(),
       "active_layers":active,"method":"legacy frozen 2431-question MC set; conditional answer NLL; V2.2 system/user/assistant framing; greedy argmin"}
    (out/"official_holdout_2431_v22_step10000.json").write_text(json.dumps({**mc_report,"predictions":mc_results},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("MC_RESULT "+json.dumps(mc_report,ensure_ascii=False),flush=True)
    free=load_rows(a.freeform_holdout)
    if len(free)!=100: raise ValueError(f"Expected 100 legacy freeform prompts, got {len(free)}")
    free_results=[]
    with torch.inference_mode():
        for n,row in enumerate(free,1):
            prefix=chat_prefix(tok,sys_ids,row["prompt"])
            answer,ids=generate(model,tok,prefix,active,a.max_len,a.max_new_tokens)
            free_results.append({"id":row.get("id",n),"source":row.get("source"),"prompt":row["prompt"],"response":answer,"token_ids":ids})
            if n%10==0: print(f"freeform_progress {n}/100",flush=True)
    free_report={"checkpoint_step":a.step,"sample_size":len(free_results),"active_layers":active,
       "decoding":"greedy argmax; max_new_tokens=%d; context=%d"%(a.max_new_tokens,a.max_len),
       "prompt_sha256":hashlib.sha256(Path(a.freeform_holdout).read_bytes()).hexdigest(),"results":free_results}
    (out/"freeform_100_v22_step10000.json").write_text(json.dumps(free_report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# LUMEN V2.2 — 100 respuestas libres (paso 10.000)","",
      "- Mismas 100 preguntas abiertas del holdout histórico.",
      "- Prompt de sistema V2.2 incorporado; generación greedy; máximo 64 tokens.",""]
    for i,r in enumerate(free_results,1):
        lines += [f"## {i}. {r['id']}","","**Pregunta**",r["prompt"],"","**V2.2**",r["response"] or "∅","","---",""]
    (out/"freeform_100_v22_step10000.md").write_text("\n".join(lines),encoding="utf-8")
    print("FREEFORM_RESULT "+json.dumps({"rows":len(free_results),"active_layers":active}),flush=True)
if __name__=="__main__": main()
