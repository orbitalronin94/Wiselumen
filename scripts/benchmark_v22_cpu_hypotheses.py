"""Paired CPU tournament: 50 baseline + 50 hypothesis steps per isolated runner."""
import argparse,json,os,random,sys,time,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from Model import LumenV2,active_layers_for_step
from Train import EMA,configure_cpu_runtime,get_optimizer,set_scheduled_lr
SEED,BATCH,SEQ,WARMUP,MEASURE=20261126,4,512,3,50
START,TOTAL=20000,30000
NAMES=["01-topk-embedding-projection","02-topk-einsum-projection","03-excitability-scatter-add","04-rope-cast-cache","05-rope-no-stack","06-rmsnorm-square","07-qkv-unbind-first","08-manual-causal-attention","09-skip-token-range-scan","10-ema-foreach","11-cache-clip-parameters","12-cache-active-layer-set","13-logsumexp-loss-weights","14-skip-disabled-autocast"]

def mlp_f(self,x,mode):
 g,u=self.gate_up(x).chunk(2,-1); h=F.silu(g)*u
 h=h+(self.excitability-float(self.top_k)/self.d_ff).to(h.dtype)*self.excitability_strength
 ix=torch.topk(h,k=self.top_k,dim=-1).indices; v=h.gather(-1,ix)
 if mode=="01": out=(v.unsqueeze(-1)*F.embedding(ix,self.down.weight.t())).sum(-2)
 elif mode=="02": out=torch.einsum("btk,btkd->btd",v,self.down.weight[:,ix].permute(1,2,3,0))
 else: out=self.down(torch.zeros_like(h).scatter_(-1,ix,v))
 if self.training:
  with torch.no_grad():
   if mode=="03":
    counts=torch.zeros_like(self.excitability); counts.scatter_add_(0,ix.reshape(-1),torch.ones(ix.numel(),device=ix.device,dtype=counts.dtype))
   else: counts=torch.bincount(ix.reshape(-1),minlength=self.d_ff).to(self.excitability.dtype)
   self.excitability.mul_(self.excitability_decay).add_(counts/ix.numel(),alpha=1-self.excitability_decay)
 return out

def rope_cache(self,q,k):
 n=q.shape[-2]; key=(n,q.device,q.dtype); cache=getattr(self,"_bench_cache",{})
 if key not in cache: cache[key]=(self.cos[:n].to(q.device,q.dtype)[None,None,:,:],self.sin[:n].to(q.device,q.dtype)[None,None,:,:]); self._bench_cache=cache
 c,s=cache[key]; return self._rotate_one(q,c,s),self._rotate_one(k,c,s)
def rotate_no_stack(self,x,c,s):
 e,o=x[...,0::2],x[...,1::2]; y=torch.empty_like(x)
 y[...,0::2]=e*c[...,0::2]-o*s[...,0::2]; y[...,1::2]=e*s[...,0::2]+o*c[...,0::2]; return y
def norm_square(self,x):
 r=torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.eps); return x*r.to(x.dtype)*self.weight
def attn_unbind(self,x):
 b,t,c=x.shape; z=self.qkv(x).view(b,t,3,self.n_head,self.d_head); q,k,v=(p.transpose(1,2) for p in z.unbind(2)); q,k=self.rope.rotate(q,k)
 y=F.scaled_dot_product_attention(q,k,v,is_causal=True); return self.out(y.transpose(1,2).contiguous().view(b,t,c))
def attn_manual(self,x):
 b,t,c=x.shape; z=self.qkv(x).view(b,t,3,self.n_head,self.d_head); q,k,v=z.permute(2,0,3,1,4).unbind(0); q,k=self.rope.rotate(q,k)
 a=q@k.transpose(-2,-1)*(self.d_head**-.5); m=torch.ones(t,t,dtype=torch.bool,device=x.device).triu(1)
 y=F.softmax(a.masked_fill(m,torch.finfo(a.dtype).min),-1)@v
 return self.out(y.transpose(1,2).contiguous().view(b,t,c))
def fast_forward(self,idx,targets=None,active_layers=None,return_logits=True,skip=False,cached=False,logsum=False):
 if idx.ndim!=2: raise ValueError("idx must be rank 2")
 if idx.shape[1]>self.ctx: raise ValueError("context exceeded")
 if not skip and idx.numel() and (idx.min()<0 or idx.max()>=self.vocab): raise ValueError("token out of range")
 x=self.tok(idx)
 if active_layers is None: active_layers=range(len(self.blocks))
 elif not(cached and isinstance(active_layers,(set,frozenset))):
  active_layers=set(active_layers)
  if active_layers.difference(range(len(self.blocks))): raise ValueError("invalid active layers")
 for i,b in enumerate(self.blocks):
  if i in active_layers: x=b(x)
 x=self.norm(x); loss=logits=None
 if targets is not None:
  if targets.shape!=idx.shape: raise ValueError("target shape mismatch")
  ft=targets.reshape(-1); valid=ft.ne(-100)
  if not torch.any(valid): raise ValueError("no supervised tokens")
  h=x.reshape(-1,self.d_model)[valid]; y=ft[valid]; z=F.linear(h,self.head.weight)
  ce=F.cross_entropy(z,y,reduction="none",label_smoothing=.05)
  if self.loss_mode=="cross_entropy": loss=ce.mean()
  else:
   with torch.no_grad():
    zf=z.float()
    if logsum:
     lz=torch.logsumexp(zf,-1); pc=(zf.gather(1,y[:,None]).squeeze(1)-lz).exp(); mp=(zf.max(-1).values-lz).exp()
    else:
     p=F.softmax(zf,-1); pc=p.gather(1,y[:,None]).squeeze(1); mp=p.max(-1).values
    ew=1-pc; oc=((mp>.7)&(pc<.3)).to(ce.dtype)*.5
   loss=(ce*(ew+oc)).mean()
 if return_logits or targets is None: logits=self.head(x)
 return logits,loss

def patch(m,e,n):
 if n[:2] in ("01","02","03"):
  mode=n[:2]
  for b in m.blocks: b.mlp.forward=types.MethodType(lambda self,x,mode=mode:mlp_f(self,x,mode),b.mlp)
 elif n.startswith("04"): m.rope.rotate=types.MethodType(rope_cache,m.rope)
 elif n.startswith("05"): m.rope._rotate_one=types.MethodType(rotate_no_stack,m.rope)
 elif n.startswith("06"):
  for b in m.blocks:
   b.norm1.forward=types.MethodType(norm_square,b.norm1); b.norm2.forward=types.MethodType(norm_square,b.norm2)
  m.norm.forward=types.MethodType(norm_square,m.norm)
 elif n.startswith("07"):
  for b in m.blocks: b.attn.forward=types.MethodType(attn_unbind,b.attn)
 elif n.startswith("08"):
  for b in m.blocks: b.attn.forward=types.MethodType(attn_manual,b.attn)
 elif n.startswith("09"): m.forward=types.MethodType(lambda s,i,targets=None,active_layers=None,return_logits=True:fast_forward(s,i,targets,active_layers,return_logits,skip=True),m)
 elif n.startswith("10"):
  def upd(self,model):
   st=model.state_dict(); keys=list(self.shadow); a=[self.shadow[k] for k in keys]; b=[st[k].detach().float() for k in keys]
   torch._foreach_mul_(a,self.decay); torch._foreach_add_(a,b,alpha=1-self.decay)
  e.update=types.MethodType(upd,e)
 elif n.startswith("11"): m._clip_params=[p for p in m.parameters() if p.requires_grad]
 elif n.startswith("12"):
  m._active=frozenset(active_layers_for_step(START,"v22",len(m.blocks)))
  m.forward=types.MethodType(lambda s,i,targets=None,active_layers=None,return_logits=True:fast_forward(s,i,targets,s._active,return_logits,cached=True),m)
 elif n.startswith("13"): m.forward=types.MethodType(lambda s,i,targets=None,active_layers=None,return_logits=True:fast_forward(s,i,targets,active_layers,return_logits,logsum=True),m)

def get_batches():
 g=torch.Generator().manual_seed(SEED+99); a=[]
 for _ in range(WARMUP+MEASURE):
  x=torch.randint(4,8184,(BATCH,SEQ),generator=g); y=torch.randint(4,8184,(BATCH,SEQ),generator=g); y[:,:int(SEQ*.6)]=-100; a.append((x,y))
 return a
def make_pair(name):
 torch.manual_seed(SEED); a=LumenV2(vocab=8192,ctx=512,loss_mode="experimental"); sd={k:v.detach().clone() for k,v in a.state_dict().items()}
 b=LumenV2(vocab=8192,ctx=512,loss_mode="experimental"); b.load_state_dict(sd,strict=True)
 ao=get_optimizer(a,prefer_8bit=False); bo=get_optimizer(b,prefer_8bit=False); ae=EMA(a,.99); be=EMA(b,.99); patch(b,be,name)
 return (a,ao,ae),(b,bo,be)
def step(bundle,batch,i,name=None):
 m,o,e=bundle; x,y=batch; gs=START+i; o.zero_grad(set_to_none=True); set_scheduled_lr(o,gs,10,TOTAL)
 layers=m._active if name and name.startswith("12") else active_layers_for_step(gs,"v22",len(m.blocks))
 if name and name.startswith("14"): _,loss=m(x,targets=y,active_layers=layers,return_logits=False)
 else:
  with torch.autocast(device_type="cpu",dtype=torch.bfloat16,enabled=False): _,loss=m(x,targets=y,active_layers=layers,return_logits=False)
 if loss is None or not torch.isfinite(loss): raise FloatingPointError("nonfinite loss")
 loss.backward()
 if (gs+1)%5==0:
  gn=torch.nn.utils.clip_grad_norm_(m._clip_params if name and name.startswith("11") else m.parameters(),1.)
  if not torch.isfinite(gn): raise FloatingPointError("nonfinite grad")
 o.step()
 if (gs+1)%10==0: e.update(m)
 return float(loss.detach().item())
def run(name):
 configure_cpu_runtime(int(os.getenv("OMP_NUM_THREADS","4")),int(os.getenv("TORCH_INTEROP_THREADS","1"))); torch.backends.mkldnn.enabled=True
 data=get_batches(); base,cand=make_pair(name)
 for i in range(WARMUP):
  order=[(base,None),(cand,name)]
  if i%2: order.reverse()
  for z,n in order: step(z,data[i],i,n)
 bt=[];ct=[];bl=[];cl=[]
 for i in range(MEASURE):
  order=[("b",base,None),("c",cand,name)]
  if i%2: order.reverse()
  for label,z,n in order:
   t=time.perf_counter(); v=step(z,data[WARMUP+i],WARMUP+i,n); dt=time.perf_counter()-t
   if label=="b": bt.append(dt);bl.append(v)
   else: ct.append(dt);cl.append(v)
 be,ce=sum(bt),sum(ct); bm,cm=sum(bl)/MEASURE,sum(cl)/MEASURE; btail,ctail=sum(bl[-10:])/10,sum(cl[-10:])/10
 sg=(be/ce-1)*100; lg=(bm-cm)/max(abs(bm),1e-12)*100; tg=(btail-ctail)/max(abs(btail),1e-12)*100
 return {"hypothesis":name,"seed":SEED,"paired_interleaved":True,"warmup_each":WARMUP,"measured_steps_each":MEASURE,"batch_size":BATCH,"sequence_length":SEQ,"start_step":START,"active_layers":len(active_layers_for_step(START,"v22",8)),"threads":torch.get_num_threads(),"torch_version":torch.__version__,"baseline":{"elapsed_seconds":be,"steps_per_second":MEASURE/be,"mean_loss":bm,"mean_last_10":btail},"hypothesis_result":{"elapsed_seconds":ce,"steps_per_second":MEASURE/ce,"mean_loss":cm,"mean_last_10":ctail},"speed_gain_percent":sg,"mean_loss_improvement_percent":lg,"last_10_loss_improvement_percent":tg,"accepted_for_merge":sg>=1 and lg>0 and tg>0}
def main():
 p=argparse.ArgumentParser();p.add_argument("--hypothesis",required=True,choices=NAMES);n=p.parse_args().hypothesis;r=run(n)
 Path("benchmark-result.json").write_text(json.dumps(r,indent=2),encoding="utf-8");print("PAIRED_BENCH_RESULT "+json.dumps(r),flush=True)
 Path(os.getenv("GITHUB_STEP_SUMMARY","bench.md")).write_text(f"# {n}\n\nBaseline {r['baseline']['steps_per_second']:.4f} steps/s, loss {r['baseline']['mean_loss']:.6f}\n\nHypothesis {r['hypothesis_result']['steps_per_second']:.4f} steps/s, loss {r['hypothesis_result']['mean_loss']:.6f}\n\nSpeed {r['speed_gain_percent']:.3f}%; mean loss improvement {r['mean_loss_improvement_percent']:.6f}%; tail improvement {r['last_10_loss_improvement_percent']:.6f}%; accepted={r['accepted_for_merge']}\n")
if __name__=="__main__":main()
