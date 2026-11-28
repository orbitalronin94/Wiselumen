#!/usr/bin/env python3
"""Basic generative smoke holdout for clean v2 checkpoint."""
import argparse,torch,hashlib,json,time
from pathlib import Path
from tokenizers import Tokenizer
from cultistlumen_v2_clean import CultistLumenClean
p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--tokenizer",required=True);a=p.parse_args()
torch.set_num_threads(2);torch.set_num_interop_threads(1)
raw=Path(a.tokenizer).read_bytes();sha=hashlib.sha256(raw).hexdigest()
if sha!="91dfa25c95f07ce9e6f3eee94caeb723712d6eaa656a18162adb0d4a047e6d9e":raise RuntimeError("Tokenizer mismatch")
tok=Tokenizer.from_file(a.tokenizer);ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
cfg=ck["model_config"];m=CultistLumenClean(vocab=cfg["vocab"],d=cfg["d_model"],heads=cfg["heads"],ff=cfg["d_ff"],layers=cfg["layers"],ctx=cfg["ctx"]);m.load_state_dict(ck["model"]);m.eval()
prompts=["¿Qué es la energía?","Explica brevemente cómo funciona una democracia.","¿Por qué llueve?","Define la fotosíntesis.","¿Qué diferencia hay entre una hipótesis y una teoría?","Escribe una explicación de la división celular.","¿Cómo se calcula un porcentaje?","Resume las causas de la Revolución francesa.","¿Qué es una base de datos?","Explica qué significa el método científico."]
rows=[];eos=tok.token_to_id("[EOS]");pad=tok.token_to_id("[PAD]")
for prompt in prompts:
 ids=tok.encode(prompt+" ",add_special_tokens=False).ids[-128:];gen=[]
 with torch.inference_mode():
  for _ in range(64):
   logits,_=m(torch.tensor([ids],dtype=torch.long));n=int(logits[0,-1].argmax())
   if n in (eos,pad):break
   gen.append(n);ids=(ids+[n])[-128:]
 rows.append({"prompt":prompt,"response":tok.decode(gen,skip_special_tokens=True),"tokens":len(gen),"eos":len(gen)<64 and n==eos})
out={"checkpoint_step":ck["step"],"unique_responses":len(set(r["response"] for r in rows)),"eos_count":sum(r["eos"] for r in rows),"capped":sum(r["tokens"]==64 for r in rows),"results":rows}
Path("artifacts/holdout-v2-clean.json").write_text(json.dumps(out,ensure_ascii=False,indent=2)+"\n")
print("HOLDOUT_COMPLETE",json.dumps({k:v for k,v in out.items() if k!="results"}),flush=True)
