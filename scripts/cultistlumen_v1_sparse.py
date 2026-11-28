# CULTISTLUMEN-V1-SPARSE: mismo mixer y FFN gated, ejecución estructural top-2/8.
# Los bancos FFN son particiones de una única FFN, no redes expertas completas.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self,d):
        super().__init__(); self.w=nn.Parameter(torch.ones(d))
    def forward(self,x):
        r=torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+1e-6)
        return x*r.to(x.dtype)*self.w

class GatedDeltaNetLR(nn.Module):
    def __init__(self,d_model=4096,n_head=64,chunk=64,conv_kernel=4,rank=512):
        super().__init__(); self.d_model=d_model; self.n_head=n_head
        self.d_head=d_model//n_head; self.chunk=chunk; self.rank=rank
        self.qkv_A=nn.Parameter(torch.randn(d_model,rank)*0.02)
        self.qkv_B=nn.Parameter(torch.randn(rank,d_model*3)*0.02)
        self.b_a=nn.Linear(d_model,n_head,bias=False); self.b_beta=nn.Linear(d_model,n_head,bias=False)
        self.dt_bias=nn.Parameter(torch.ones(n_head)*2.0); self.A_log=nn.Parameter(torch.log(torch.ones(n_head)*0.5))
        self.conv=nn.Conv1d(d_model,d_model,kernel_size=conv_kernel,groups=d_model,bias=True,padding=conv_kernel-1)
        self.out_A=nn.Parameter(torch.randn(d_model,rank)*0.02); self.out_B=nn.Parameter(torch.randn(rank,d_model)*0.02)
        self.scale=self.d_head**-0.5
    def forward(self,x):
        B,T,C=x.shape
        xc=F.silu(self.conv(x.transpose(1,2))[:,:,:T].transpose(1,2))
        qkv=(xc@self.qkv_A)@self.qkv_B
        qkv=qkv.view(B,T,3,self.n_head,self.d_head).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        q=F.normalize(q,p=2,dim=-1)*self.scale; k=F.normalize(k,p=2,dim=-1)
        a=self.b_a(xc); beta=torch.sigmoid(self.b_beta(xc))
        g=torch.exp(-torch.exp(self.A_log)[None,None,:]*F.softplus(a+self.dt_bias[None,None,:]))
        L=self.chunk; N=(T+L-1)//L; pad=N*L-T
        if pad:
            q=torch.cat([q,torch.zeros(B,self.n_head,pad,self.d_head,device=x.device,dtype=x.dtype)],2)
            k=torch.cat([k,torch.zeros(B,self.n_head,pad,self.d_head,device=x.device,dtype=x.dtype)],2)
            v=torch.cat([v,torch.zeros(B,self.n_head,pad,self.d_head,device=x.device,dtype=x.dtype)],2)
            g=torch.cat([g,torch.ones(B,pad,self.n_head,device=x.device,dtype=x.dtype)],1)
            beta=torch.cat([beta,torch.zeros(B,pad,self.n_head,device=x.device,dtype=x.dtype)],1)
        q_c=q.view(B,self.n_head,N,L,self.d_head); k_c=k.view(B,self.n_head,N,L,self.d_head); v_c=v.view(B,self.n_head,N,L,self.d_head)
        g_c=g.permute(0,2,1).view(B,self.n_head,N,L); beta_c=beta.permute(0,2,1).view(B,self.n_head,N,L)
        tril=torch.tril(torch.ones(L,L,device=x.device,dtype=torch.bool),diagonal=-1); eye=torch.eye(L,device=x.device,dtype=torch.float32)
        S=torch.zeros(B,self.n_head,self.d_head,self.d_head,device=x.device,dtype=x.dtype); outs=[]
        for n in range(N):
            qc=q_c[:,:,n]; kc=k_c[:,:,n]; vc=v_c[:,:,n]; gc=g_c[:,:,n]; bc=beta_c[:,:,n]
            cum_g=torch.cumprod(gc,dim=-1); k_s=kc*cum_g.unsqueeze(-1); q_s=qc/cum_g.unsqueeze(-1).clamp(min=1e-6)
            A=torch.matmul(k_s,k_s.transpose(-1,-2))*tril.unsqueeze(0).unsqueeze(0)
            M=eye+A.float()*bc.float().unsqueeze(-1)
            Tmat=torch.linalg.solve_triangular(M,eye.expand(B,self.n_head,L,L),upper=False)
            w=torch.matmul(Tmat,(vc*bc.unsqueeze(-1)).float()).to(x.dtype)
            u=torch.matmul(Tmat.transpose(-1,-2),(k_s*bc.unsqueeze(-1)).float()).to(x.dtype)
            o_inter=torch.matmul(q_s,S); qu=torch.matmul(q_s,u.transpose(-1,-2))*tril.unsqueeze(0).unsqueeze(0)
            diag_qk=torch.sum(q_s*k_s,dim=-1)
            outs.append(o_inter+torch.matmul(qu,w)+(diag_qk*bc).unsqueeze(-1)*w)
            S=S*cum_g[:,:,-1].unsqueeze(-1).unsqueeze(-1)+torch.matmul(u.transpose(-1,-2),w)
        o=torch.cat(outs,2)[:,:,:T].permute(0,2,1,3).reshape(B,T,C)
        return (o@self.out_A)@self.out_B

class BlockSparseGatedFFN(nn.Module):
    """Una FFN particionada en 8 bancos; cada token ejecuta solo top-2 bancos."""
    def __init__(self,d_model=4096,d_ff=16544,banks=8,top_k=2):
        super().__init__()
        if d_ff%banks: raise ValueError("d_ff must be divisible by banks")
        self.d_model=d_model; self.d_ff=d_ff; self.banks=banks; self.top_k=top_k; self.bank_width=d_ff//banks
        self.router=nn.Linear(d_model,banks,bias=False)
        self.gate_up=nn.Parameter(torch.randn(banks,d_model,self.bank_width*2)*0.02)
        self.down=nn.Parameter(torch.randn(banks,self.bank_width,d_model)*0.02)
        self.last_active_fraction=0.0; self.last_load=[0.0]*banks; self.balance_loss=None
    def forward(self,x):
        shape=x.shape; flat=x.reshape(-1,self.d_model)
        logits=self.router(flat).float()
        top_values,top_indices=torch.topk(logits,self.top_k,dim=-1)
        weights=torch.softmax(top_values,dim=-1).to(x.dtype)
        out=torch.zeros_like(flat)
        load=torch.bincount(top_indices.reshape(-1),minlength=self.banks).float()
        self.last_load=(load/load.sum().clamp_min(1)).detach().cpu().tolist()
        importance=torch.softmax(logits,dim=-1).mean(0)
        hard_load=load/load.sum().clamp_min(1)
        self.balance_loss=self.banks*torch.sum(importance*hard_load)
        for bank in range(self.banks):
            token_slot=(top_indices==bank).nonzero(as_tuple=False)
            if token_slot.numel()==0: continue
            token_ids=token_slot[:,0]; rank_ids=token_slot[:,1]
            xb=flat.index_select(0,token_ids)
            gu=xb@self.gate_up[bank]; gate,up=gu.chunk(2,dim=-1)
            y=(F.silu(gate)*up)@self.down[bank]
            y=y*weights[token_ids,rank_ids].unsqueeze(-1)
            out.index_add_(0,token_ids,y)
        self.last_active_fraction=self.top_k/self.banks
        return out.view(shape)

class CultistLumenV1Sparse(nn.Module):
    def __init__(self,vocab=8192,d_model=4096,d_ff=16544,ctx=128,n_head=64,rank=512,banks=8,top_k=2):
        super().__init__()
        if d_model%n_head: raise ValueError("d_model must divide n_head")
        self.ctx=ctx; self.d_model=d_model; self.vocab=vocab
        self.tok=nn.Embedding(vocab,d_model); self.norm1=RMSNorm(d_model)
        self.mix=GatedDeltaNetLR(d_model,n_head,64,4,rank); self.norm2=RMSNorm(d_model)
        self.ffn=BlockSparseGatedFFN(d_model,d_ff,banks,top_k); self.norm=RMSNorm(d_model)
        self.head=nn.Linear(d_model,vocab,bias=False); self.head.weight=self.tok.weight
        self.apply(self._init)
    def _init(self,m):
        if isinstance(m,nn.Linear):
            if m.weight is not self.tok.weight: nn.init.normal_(m.weight,0.0,0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        if isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0.0,0.02)
        if isinstance(m,nn.Conv1d):
            nn.init.normal_(m.weight,0.0,0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self,idx,targets=None,sampled_negatives=1024):
        x=self.tok(idx); x=x+self.mix(self.norm1(x))
        x=x+self.ffn(self.norm2(x)); x=self.norm(x)
        if targets is None: return self.head(x),None
        flat_x=x.reshape(-1,self.d_model).float(); tgt=targets.reshape(-1)
        positives=torch.unique(tgt); n=min(int(sampled_negatives),self.vocab-positives.numel())
        perm=torch.randperm(self.vocab,device=idx.device)
        negatives=perm[~torch.isin(perm,positives)][:n]
        candidates=torch.sort(torch.cat((positives,negatives))).values
        logits=F.linear(flat_x,self.tok.weight[candidates].float())
        positions=torch.searchsorted(candidates,tgt); target_logits=logits.gather(1,positions[:,None]).squeeze(1)
        mask=torch.isin(candidates,negatives); prob=n/max(1,self.vocab-positives.numel())
        corrected=logits.clone(); corrected[:,mask]-=math.log(max(prob,1e-12))
        logz=torch.logsumexp(corrected,dim=-1)
        ce=logz-0.95*target_logits-0.05*corrected.mean(-1)
        with torch.no_grad():
            pc=torch.exp(target_logits-logz); mp=torch.exp(corrected.max(-1).values-logz)
            weight=1.0-pc; over=((mp>0.7)&(pc<0.3)).float()*0.5
        loss=(ce*weight+ce*over).mean()
        if self.ffn.balance_loss is not None: loss=loss+0.01*self.ffn.balance_loss
        return corrected,loss
    def num_params(self): return sum(p.numel() for p in self.parameters())
