# CULTISTLUMEN-V1-DENSE: mismo mixer, FFN densa, sin MoE.
# CPU-first: FP32 throughout, chunk-local solver, sampled vocabulary loss.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        r = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        return x * r.to(x.dtype) * self.w

class GatedDeltaNetLR(nn.Module):
    def __init__(self, d_model=512, n_head=8, chunk=64, conv_kernel=4, rank=128):
        super().__init__()
        self.d_model=d_model; self.n_head=n_head
        self.d_head=d_model//n_head; self.chunk=chunk; self.rank=rank
        self.qkv_A=nn.Parameter(torch.randn(d_model,rank)*0.02)
        self.qkv_B=nn.Parameter(torch.randn(rank,d_model*3)*0.02)
        self.b_a=nn.Linear(d_model,n_head,bias=False)
        self.b_beta=nn.Linear(d_model,n_head,bias=False)
        self.dt_bias=nn.Parameter(torch.ones(n_head)*2.0)
        self.A_log=nn.Parameter(torch.log(torch.ones(n_head)*0.5))
        self.conv=nn.Conv1d(d_model,d_model,kernel_size=conv_kernel,groups=d_model,bias=True,padding=conv_kernel-1)
        self.out_A=nn.Parameter(torch.randn(d_model,rank)*0.02)
        self.out_B=nn.Parameter(torch.randn(rank,d_model)*0.02)
        self.scale=self.d_head**-0.5

    def forward(self,x):
        B,T,C=x.shape
        xc=F.silu(self.conv(x.transpose(1,2))[:,:,:T].transpose(1,2))
        qkv=(xc@self.qkv_A)@self.qkv_B
        qkv=qkv.view(B,T,3,self.n_head,self.d_head).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        q=F.normalize(q,p=2,dim=-1)*self.scale
        k=F.normalize(k,p=2,dim=-1)
        a=self.b_a(xc)
        beta=torch.sigmoid(self.b_beta(xc))
        g=torch.exp(-torch.exp(self.A_log)[None,None,:]*F.softplus(a+self.dt_bias[None,None,:]))
        L=self.chunk; N=(T+L-1)//L; pad=N*L-T
        if pad:
            q=torch.cat([q,torch.zeros(B,self.n_head,pad,self.d_head,device=x.device,dtype=x.dtype)],dim=2)
            k=torch.cat([k,torch.zeros(B,self.n_head,pad,self.d_head,device=x.device,dtype=x.dtype)],dim=2)
            v=torch.cat([v,torch.zeros(B,self.n_head,pad,self.d_head,device=x.device,dtype=x.dtype)],dim=2)
            g=torch.cat([g,torch.ones(B,pad,self.n_head,device=x.device,dtype=x.dtype)],dim=1)
            beta=torch.cat([beta,torch.zeros(B,pad,self.n_head,device=x.device,dtype=x.dtype)],dim=1)
        q_c=q.view(B,self.n_head,N,L,self.d_head)
        k_c=k.view(B,self.n_head,N,L,self.d_head)
        v_c=v.view(B,self.n_head,N,L,self.d_head)
        g_c=g.permute(0,2,1).view(B,self.n_head,N,L)
        beta_c=beta.permute(0,2,1).view(B,self.n_head,N,L)
        # Solver por bloque: limita matrices auxiliares vivas y el pico de autograd.
        tril=torch.tril(torch.ones(L,L,device=x.device,dtype=torch.bool),diagonal=-1)
        eye=torch.eye(L,device=x.device,dtype=torch.float32)
        S=torch.zeros(B,self.n_head,self.d_head,self.d_head,device=x.device,dtype=x.dtype)
        outs=[]
        for n in range(N):
            qc=q_c[:,:,n]; kc=k_c[:,:,n]; vc=v_c[:,:,n]
            gc=g_c[:,:,n]; bc=beta_c[:,:,n]
            cum_g=torch.cumprod(gc,dim=-1)
            k_s=kc*cum_g.unsqueeze(-1)
            q_s=qc/cum_g.unsqueeze(-1).clamp(min=1e-6)
            A=torch.matmul(k_s,k_s.transpose(-1,-2))
            A=A*tril.unsqueeze(0).unsqueeze(0)
            M=eye+A.float()*bc.float().unsqueeze(-1)
            Tmat=torch.linalg.solve_triangular(M,eye.expand(B,self.n_head,L,L),upper=False)
            w=torch.matmul(Tmat,(vc*bc.unsqueeze(-1)).float()).to(x.dtype)
            u=torch.matmul(Tmat.transpose(-1,-2),(k_s*bc.unsqueeze(-1)).float()).to(x.dtype)
            o_inter=torch.matmul(q_s,S)
            qu=torch.matmul(q_s,u.transpose(-1,-2))
            qu=qu*tril.unsqueeze(0).unsqueeze(0)
            diag_qk=torch.sum(q_s*k_s,dim=-1)
            o=o_inter+torch.matmul(qu,w)+(diag_qk*bc).unsqueeze(-1)*w
            outs.append(o)
            cum_last=cum_g[:,:,-1]
            S=S*cum_last.unsqueeze(-1).unsqueeze(-1)+torch.matmul(u.transpose(-1,-2),w)
        o_cat=torch.cat(outs,dim=2)[:,:,:T,:]
        o_cat=o_cat.permute(0,2,1,3).reshape(B,T,C)
        return (o_cat@self.out_A)@self.out_B

class DenseGatedFFN(nn.Module):
    def __init__(self,d_model=512,d_ff=1024):
        super().__init__()
        self.gate_up=nn.Parameter(torch.randn(d_model,d_ff*2)*0.02)
        self.down=nn.Parameter(torch.randn(d_ff,d_model)*0.02)
    def forward(self,x):
        # FP32 evita conversiones y kernels FP16 lentos en CPU x86.
        gu=x@self.gate_up
        gate,up=gu.chunk(2,dim=-1)
        return (F.silu(gate)*up)@self.down

class CultistLumenV1Dense(nn.Module):
    def __init__(self,vocab=8192,d_model=512,d_ff=1024,ctx=256,n_head=8,rank=128):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_head={n_head}")
        self.ctx=ctx; self.d_model=d_model; self.vocab=vocab
        self.tok=nn.Embedding(vocab,d_model)
        self.norm1=RMSNorm(d_model)
        self.mix=GatedDeltaNetLR(d_model=d_model,n_head=n_head,chunk=64,conv_kernel=4,rank=rank)
        self.norm2=RMSNorm(d_model)
        self.ffn=DenseGatedFFN(d_model=d_model,d_ff=d_ff)
        self.norm=RMSNorm(d_model)
        self.head=nn.Linear(d_model,vocab,bias=False)
        self.head.weight=self.tok.weight
        self.apply(self._init)
    def _init(self,m):
        if isinstance(m,nn.Linear):
            # head comparte tok: no inicializar dos veces el mismo parámetro.
            if m.weight is not self.tok.weight:
                nn.init.normal_(m.weight,mean=0.0,std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        if isinstance(m,nn.Embedding):
            nn.init.normal_(m.weight,mean=0.0,std=0.02)
        if isinstance(m,nn.Conv1d):
            nn.init.normal_(m.weight,mean=0.0,std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self,idx,targets=None,sampled_negatives=1024):
        from torch.profiler import record_function
        with record_function("model/mixer_forward"):
            x=self.tok(idx)
            x=x+self.mix(self.norm1(x))
        with record_function("model/ffn_forward"):
            x=x+self.ffn(self.norm2(x))
        x=self.norm(x)
        if targets is None:
            with record_function("model/full_vocab_head"):
                return self.head(x),None
        # Softmax muestral: todos los positivos del batch y negativos uniformes.
        # La inferencia sigue usando el vocabulario completo y exacto.
        with record_function("model/sampled_vocab_head"):
            flat_x=x.reshape(-1,self.d_model).float()
            tgt_f=targets.reshape(-1)
            positive_ids=torch.unique(tgt_f)
            negative_count=min(int(sampled_negatives),self.vocab-positive_ids.numel())
            permutation=torch.randperm(self.vocab,device=idx.device)
            negative_ids_all=permutation[~torch.isin(permutation,positive_ids)]
            negative_ids=negative_ids_all[:negative_count]
            candidates=torch.sort(torch.cat((positive_ids,negative_ids))).values
            logits_f=F.linear(flat_x,self.tok.weight[candidates].float())
            target_positions=torch.searchsorted(candidates,tgt_f)
            target_logits=logits_f.gather(1,target_positions.unsqueeze(1)).squeeze(1)
            negative_mask=torch.isin(candidates,negative_ids)
            inclusion_prob=negative_count/max(1,self.vocab-positive_ids.numel())
            corrected_logits=logits_f.clone()
            corrected_logits[:,negative_mask]-=math.log(max(inclusion_prob,1e-12))
            log_z=torch.logsumexp(corrected_logits,dim=-1)
            ce=log_z-0.95*target_logits-0.05*corrected_logits.mean(dim=-1)
            with torch.no_grad():
                p_correct=torch.exp(target_logits-log_z)
                max_p=torch.exp(corrected_logits.max(dim=-1).values-log_z)
                weight=1.0-p_correct
                overconf=((max_p>0.7)&(p_correct<0.3)).float()*0.5
            loss=(ce*weight+ce*overconf).mean()
        return corrected_logits,loss
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
