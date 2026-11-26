"""Generate a frozen 20-prompt V5-corpus probe on a LUMEN V2.2 checkpoint."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tokenizers import Tokenizer
from Model import LumenV2, active_layers_for_step
from data import ROLE_IDS, validate_tokenizer

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--tokenizer",required=True)
    p.add_argument("--system",required=True)
    p.add_argument("--prompts",default="eval/v5_generative_holdout_20.jsonl")
    p.add_argument("--out-dir",required=True)
    p.add_argument("--step",type=int,required=True)
    p.add_argument("--threads",type=int,default=4)
    p.add_argument("--max-len",type=int,default=512)
    p.add_argument("--max-new-tokens",type=int,default=96)
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    tok=Tokenizer.from_file(a.tokenizer)
    validate_tokenizer(tok)
    state=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    assert int(state["step"])==a.step, state.get("step")
    model=LumenV2(vocab=8192,ctx=a.max_len,loss_mode="experimental")
    model.load_state_dict(state["model"],strict=True)
    model.eval()
    active=active_layers_for_step(a.step,"v22",len(model.blocks))
    system=Path(a.system).read_text(encoding="utf-8").strip()
    sys_ids=tok.encode(system,add_special_tokens=False).ids
    if any(i>=8184 for i in sys_ids):
        raise ValueError("System prompt collides with reserved IDs")
    rows=[json.loads(line) for line in Path(a.prompts).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows)!=20:
        raise ValueError(f"Expected exactly 20 V5 probes, got {len(rows)}")
    if len({r["id"] for r in rows})!=20:
        raise ValueError("V5 probe IDs must be unique")
    eos=tok.token_to_id("[EOS]")
    pad=tok.token_to_id("[PAD]")
    results=[]
    with torch.inference_mode():
        for n,row in enumerate(rows,1):
            user_ids=tok.encode(str(row["prompt"]).strip(),add_special_tokens=False).ids
            if any(i>=8184 for i in user_ids):
                raise ValueError(f"Prompt {row['id']} collides with reserved role IDs")
            ids=[ROLE_IDS["<|bos|>"],ROLE_IDS["<|system|>"]]+sys_ids+[ROLE_IDS["<|end_system|>"],ROLE_IDS["<|user|>"]]+user_ids+[ROLE_IDS["<|end_user|>"],ROLE_IDS["<|assistant|>"]]
            generated=[]
            for _ in range(a.max_new_tokens):
                x=torch.tensor([ids[-a.max_len:]],dtype=torch.long)
                logits,_=model(x,active_layers=active,return_logits=True)
                nxt=int(torch.argmax(logits[0,-1]).item())
                if nxt in (eos,pad,ROLE_IDS["<|end_assistant|>"]):
                    break
                generated.append(nxt)
                ids.append(nxt)
            answer=tok.decode(generated,skip_special_tokens=True).strip()
            results.append({"id":row["id"],"category":row["category"],"prompt":row["prompt"],"response":answer,"token_ids":generated})
            print(f"v5_probe {n}/20 id={row['id']} category={row['category']} response={answer[:180]!r}",flush=True)
    out=Path(a.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    report={"checkpoint_step":a.step,"sample_size":len(results),"categories":{c:sum(r["category"]==c for r in results) for c in sorted({r["category"] for r in results})},"active_layers":active,"decoding":f"greedy argmax; max_new_tokens={a.max_new_tokens}; context={a.max_len}","prompt_sha256":hashlib.sha256(Path(a.prompts).read_bytes()).hexdigest(),"results":results}
    (out/"v5_generative_holdout_20.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=[f"# LUMEN V2.2 — 20 probes generativas del corpus V5 (checkpoint {a.step})","","Holdout fijo, independiente de los ejemplos de entrenamiento. Generación greedy; sin puntuación automática de verdad factual o coherencia.",""]
    for i,r in enumerate(results,1):
        lines += [f"## {i}. {r['id']} — {r['category']}","","**Pregunta**",r["prompt"],"","**Respuesta generada**",r["response"] or "∅","","---",""]
    (out/"v5_generative_holdout_20.md").write_text("\n".join(lines),encoding="utf-8")
    print("V5_GENERATIVE_HOLDOUT "+json.dumps({k:v for k,v in report.items() if k!="results"},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
