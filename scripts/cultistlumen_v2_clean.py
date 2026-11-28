"""CULTISTLUMEN clean v2: two-layer causal Transformer with real 128-token context."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalBlock(nn.Module):
    def __init__(self, d=256, heads=4, ff=768, ctx=128):
        super().__init__()
        self.heads=heads; self.hd=d//heads
        self.n1=nn.LayerNorm(d); self.qkv=nn.Linear(d,3*d,bias=False); self.proj=nn.Linear(d,d,bias=False)
        self.n2=nn.LayerNorm(d); self.up=nn.Linear(d,2*ff,bias=False); self.down=nn.Linear(ff,d,bias=False)
        self.register_buffer("causal",torch.triu(torch.ones(ctx,ctx,dtype=torch.bool),diagonal=1),persistent=False)
    def forward(self,x):
        b,t,d=x.shape
        z=self.qkv(self.n1(x)).view(b,t,3,self.heads,self.hd).permute(2,0,3,1,4)
        q,k,v=z.unbind(0)
        a=F.scaled_dot_product_attention(q,k,v,attn_mask=None,is_causal=True)
        x=x+self.proj(a.transpose(1,2).contiguous().view(b,t,d))
        g,u=self.up(self.n2(x)).chunk(2,-1)
        return x+self.down(F.silu(g)*u)

class CultistLumenClean(nn.Module):
    def __init__(self,vocab=4096,d=256,heads=4,ff=768,layers=2,ctx=128):
        super().__init__()
        self.vocab=vocab; self.ctx=ctx
        self.tok=nn.Embedding(vocab,d); self.pos=nn.Embedding(ctx,d)
        self.blocks=nn.ModuleList([CausalBlock(d,heads,ff,ctx) for _ in range(layers)])
        self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,vocab,bias=False); self.head.weight=self.tok.weight
        self.apply(self._init)
        nn.init.normal_(self.pos.weight,mean=0.0,std=0.02)
    def _init(self,m):
        if isinstance(m,(nn.Linear,nn.Embedding)):
            if not (isinstance(m,nn.Linear) and m.weight is self.tok.weight):
                nn.init.normal_(m.weight,mean=0.0,std=0.02)
            if isinstance(m,nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self,idx,targets=None):
        b,t=idx.shape
        if t>self.ctx: raise ValueError(f"sequence length {t} exceeds context {self.ctx}")
        x=self.tok(idx)+self.pos(torch.arange(t,device=idx.device))[None,:,:]
        for block in self.blocks: x=block(x)
        logits=self.head(self.norm(x))
        loss=F.cross_entropy(logits.reshape(-1,self.vocab),targets.reshape(-1)) if targets is not None else None
        return logits,loss
    def num_params(self): return sum(p.numel() for p in self.parameters())
