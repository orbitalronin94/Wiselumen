"""Greedy identity/adversarial holdout for LUMEN V2.2 checkpoints."""
import argparse, json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from Model import LumenV2, active_layers_for_step
from Train import configure_cpu_runtime
from data import ROLE_IDS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--tokenizer",required=True)
    ap.add_argument("--system",default="system_prompt_short.txt")
    ap.add_argument("--holdout",default="eval/identity_attacks.jsonl")
    ap.add_argument("--out",default="results/identity_holdout_step10000.json")
    ap.add_argument("--threads",type=int,default=4)
    ap.add_argument("--step",type=int,default=10000)
    ap.add_argument("--max-new-tokens",type=int,default=96)
    a=ap.parse_args()
    configure_cpu_runtime(a.threads,1)
    tok=Tokenizer.from_file(a.tokenizer)
    eos=tok.token_to_id("[EOS]")
    pad=tok.token_to_id("[PAD]")
    assert tok.get_vocab_size(with_added_tokens=True)==8192
    assert pad==0 and eos is not None
    for t,i in ROLE_IDS.items(): assert tok.token_to_id(t)==i,(t,tok.token_to_id(t))
    payload=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    assert int(payload["step"])==a.step, payload.get("step")
    model=LumenV2(vocab=8192,ctx=512,loss_mode="experimental")
    model.load_state_dict(payload["model"],strict=True)
    model.eval()
    sys_ids=tok.encode(Path(a.system).read_text(encoding="utf-8").strip(),add_special_tokens=False).ids
    rows=[json.loads(x) for x in Path(a.holdout).read_text(encoding="utf-8").splitlines() if x.strip()]
    active=active_layers_for_step(a.step, "v22", len(model.blocks))
    results=[]
    for n,row in enumerate(rows):
        user_ids=tok.encode(row["prompt"],add_special_tokens=False).ids
        ids=[ROLE_IDS["<|bos|>"],ROLE_IDS["<|system|>"]]+sys_ids+[ROLE_IDS["<|end_system|>"],ROLE_IDS["<|user|>"]]+user_ids+[ROLE_IDS["<|end_user|>"],ROLE_IDS["<|assistant|>"]]
        generated=[]
        with torch.inference_mode():
            for _ in range(a.max_new_tokens):
                x=torch.tensor([ids[-512:]],dtype=torch.long)
                logits,_=model(x,active_layers=active,return_logits=True)
                nxt=int(torch.argmax(logits[0,-1]).item())
                if nxt in (ROLE_IDS["<|end_assistant|>"],eos): break
                generated.append(nxt); ids.append(nxt)
        text=tok.decode(generated,skip_special_tokens=True).strip()
        low=text.casefold()
        bad=[s for s in row.get("must_not",[]) if s.casefold() in low]
        expected=row.get("expected_identity","LUMEN")
        identity_ok=expected.casefold() in low
        results.append({"id":n+1,"prompt":row["prompt"],"response":text,"expected_identity":expected,"identity_ok":identity_ok,"violations":bad,"pass":identity_ok and not bad})
        print(f"holdout {n+1}/{len(rows)} pass={identity_ok and not bad} response={text[:180]!r}",flush=True)
    report={"checkpoint_step":a.step,"rows":len(rows),"passed":sum(x["pass"] for x in results),"accuracy":sum(x["pass"] for x in results)/len(rows) if rows else 0.0,"active_layers":active,"method":"greedy autoregressive; frozen identity/adversarial prompts; exact substring checks","results":results}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("HOLDOUT_RESULT "+json.dumps({k:v for k,v in report.items() if k!="results"},ensure_ascii=False),flush=True)
if __name__=="__main__": main()
