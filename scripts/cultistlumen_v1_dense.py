# CULTISTLUMEN-V1-DENSE: mismo mixer, FFN densa, sin MoE.
# FP32 master weights; FFN GEMMs en FP16; recurrent solver y loss en FP32.
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
        self.qkv_B=nn.Parameter(torch.zeros(rank,d_model*3))
        self.b_a=nn.Linear(d_model,n_head,bias=False)
        self.b_beta=nn.Linear(d_model,n_head,bias=False)
        self.dt_bias=nn.Parameter(torch.ones(n_head)*2.0)
        self.A_log=nn.Parameter(torch.log(torch.ones(n_head)*0.5))
        self.conv=nn.Conv1d(d_model,d_model,kernel_size=conv_kernel,groups=d_model,bias=True,padding=conv_kernel-1)
        self.out_A=nn.Parameter(torch.randn(d_model,rank)*0.02)
        self.out_B=nn.Parameter(torch.zeros(rank,d_model))
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
        # Resolver todos los bloques en una sola llamada batched; el estado S
        # sigue siendo recurrente y se actualiza secuencialmente debajo.
        tril=torch.tril(torch.ones(L,L,device=x.device,dtype=torch.bool),diagonal=-1)
        eye=torch.eye(L,device=x.device,dtype=torch.float32)
        cum_g_all=torch.cumprod(g_c,dim=-1)
        k_s_all=k_c*cum_g_all.unsqueeze(-1)
        q_s_all=q_c/cum_g_all.unsqueeze(-1).clamp(min=1e-6)
        A_all=torch.matmul(k_s_all,k_s_all.transpose(-1,-2))
        A_all=A_all*tril.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        M_all=eye+A_all.float()*beta_c.float().unsqueeze(-1)
        eye_batch=eye.expand(B,self.n_head,N,L,L)
        T_all=torch.linalg.solve_triangular(M_all,eye_batch,upper=False)
        w_all=torch.matmul(T_all,(v_c*beta_c.unsqueeze(-1)).float()).to(x.dtype)
        u_all=torch.matmul(T_all.transpose(-1,-2),(k_s_all*beta_c.unsqueeze(-1)).float()).to(x.dtype)
        S=torch.zeros(B,self.n_head,self.d_head,self.d_head,device=x.device,dtype=x.dtype)
        outs=[]
        for n in range(N):
            qc=q_s_all[:,:,n]; kc=k_s_all[:,:,n]; bc=beta_c[:,:,n]
            w=w_all[:,:,n]; u=u_all[:,:,n]
            o_inter=torch.matmul(qc,S)
            qu=torch.matmul(qc,u.transpose(-1,-2))
            qu=qu*tril.unsqueeze(0).unsqueeze(0)
            diag_qk=torch.sum(qc*kc,dim=-1)
            o=o_inter+torch.matmul(qu,w)+(diag_qk*bc).unsqueeze(-1)*w
            outs.append(o)
            cum_last=cum_g_all[:,:,n,-1]
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
        # Parámetros maestros FP32; GEMMs de la FFN en FP16.
        x16=x.to(torch.float16)
        gu=x16@self.gate_up.to(torch.float16)
        gate,up=gu.chunk(2,dim=-1)
        h=F.silu(gate)*up
        return (h@self.down.to(torch.float16)).to(torch.float32)

class CultistLumenV1Dense(nn.Module):
    def __init__(self,vocab=8192,d_model=512,d_ff=1024,ctx=256):
        super().__init__()
        self.ctx=ctx; self.d_model=d_model; self.vocab=vocab
        self.tok=nn.Embedding(vocab,d_model)
        self.norm1=RMSNorm(d_model)
        self.mix=GatedDeltaNetLR(d_model=d_model,n_head=8,chunk=64,conv_kernel=4,rank=128)
        self.norm2=RMSNorm(d_model)
        self.ffn=DenseGatedFFN(d_model=d_model,d_ff=d_ff)
        self.norm=RMSNorm(d_model)
        self.head=nn.Linear(d_model,vocab,bias=False)
        self.head.weight=self.tok.weight
        self.apply(self._init)
    def _init(self,m):
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,mean=0.0,std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        if isinstance(m,nn.Embedding):
            nn.init.normal_(m.weight,mean=0.0,std=0.02)
        if isinstance(m,nn.Conv1d):
            nn.init.normal_(m.weight,mean=0.0,std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self,idx,targets=None):
        x=self.tok(idx)
        x=x+self.mix(self.norm1(x))
        x=x+self.ffn(self.norm2(x))
        x=self.norm(x)
        logits=self.head(x)
        loss=None
        if targets is not None:
            logits_f=logits.view(-1,logits.size(-1)).float()
            tgt_f=targets.view(-1)
            # Una sola logsumexp sirve para CE, p(correcta) y max-prob.
            log_z=torch.logsumexp(logits_f,dim=-1)
            target_logits=logits_f.gather(1,tgt_f.unsqueeze(1)).squeeze(1)
            ce=log_z-(1.0-0.05)*target_logits-0.05*logits_f.mean(dim=-1)
            with torch.no_grad():
                p_correct=torch.exp(target_logits-log_z)
                max_p=torch.exp(logits_f.max(dim=-1).values-log_z)
                w=1.0-p_correct
                overconf=((max_p>0.7)&(p_correct<0.3)).float()*0.5
            loss=(ce*w+ce*overconf).mean()
        return logits,loss
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
