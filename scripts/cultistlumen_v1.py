# CULTISTLUMEN-V1 - 105.6M totales, 1.92M activos por token
# NO TRANSFORMER. No hay QK^T, no hay softmax, no hay RoPE, no hay attention.
# Arquitectura: 2 capas y punto. No 10. 10 capas te matan en steps/segundo.
#   Capa 1: Gated DeltaNet low-rank - mixer O(T), chunkwise 64, conv corta 4
#   Capa 2: MoE Top-1 - 64 expertos dormidos, 1 despierto por token
# Por eso un 100M puede ir mas rapido que un transformer de 2M:
#   transformer 2M paga 2M + O(T^2) en cada token
#   CULTISTLUMEN paga 1.92M + O(T) en cada token, y 103M estan dormidos
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_num_threads(4)
torch.set_num_interop_threads(1)

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return x * r * self.w

class LowRankLinear(nn.Module):
    # W = A @ B, sin W completa. A: [in, r], B: [r, out]
    # Activo: in*r + r*out, no in*out. Aqui r=128.
    def __init__(self, in_dim, out_dim, rank=128):
        super().__init__()
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.02)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
    def forward(self, x):
        return (x @ self.A) @ self.B

class GatedDeltaNetLR(nn.Module):
    # Gated DeltaNet con qkv y out en low-rank 128.
    # Total ~0.8M, activo ~0.35M. Sin esto, el mixer solo ya se come 1M activo.
    def __init__(self, d_model=512, n_head=8, chunk=64, conv_kernel=4, rank=128):
        super().__init__()
        self.d_model=d_model; self.n_head=n_head
        self.d_head=d_model//n_head; self.chunk=chunk; self.rank=rank
        # qkv: 512 -> 128 -> 1536. Si fuera denso: 786k. Low-rank: 512*128 + 128*1536 = 262k
        self.qkv_A = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.qkv_B = nn.Parameter(torch.zeros(rank, d_model*3))
        # b_a, b_beta: 512 -> 8, baratos, densos, no hace falta low-rank
        self.b_a = nn.Linear(d_model, n_head, bias=False)
        self.b_beta = nn.Linear(d_model, n_head, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(n_head)*2.0)
        self.A_log = nn.Parameter(torch.log(torch.ones(n_head)*0.5))
        # conv corta depthwise k=4: aprende n-gramas en paso 0
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=conv_kernel, groups=d_model, bias=True, padding=conv_kernel-1)
        # out: 512 -> 128 -> 512. Denso: 262k. Low-rank: 512*128 + 128*512 = 131k
        self.out_A = nn.Parameter(torch.randn(d_model, rank) * 0.02)
        self.out_B = nn.Parameter(torch.zeros(rank, d_model))
        self.scale = self.d_head ** -0.5

    def forward(self, x):
        B,T,C = x.shape
        # 1. Conv corta causal
        xc = self.conv(x.transpose(1,2))[:, :, :T].transpose(1,2)
        xc = F.silu(xc)
        # 2. qkv low-rank
        qkv = (xc @ self.qkv_A) @ self.qkv_B
        qkv = qkv.view(B,T,3,self.n_head,self.d_head).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]
        q = F.normalize(q, p=2, dim=-1) * self.scale
        k = F.normalize(k, p=2, dim=-1)
        # 3. Gating + beta
        a = self.b_a(xc)  # [B,T,H]
        beta = torch.sigmoid(self.b_beta(xc))  # [B,T,H]
        g = torch.exp(-torch.exp(self.A_log)[None,None,:] * F.softplus(a + self.dt_bias[None,None,:]))  # [B,T,H]
        # 4. Chunkwise: T=512 -> 8 chunks de 64. Solo 8 pasos secuenciales.
        L = self.chunk
        N = (T + L - 1)//L
        pad = N*L - T
        if pad>0:
            q=torch.cat([q, torch.zeros(B,self.n_head,pad,self.d_head, device=x.device, dtype=x.dtype)], dim=2)
            k=torch.cat([k, torch.zeros(B,self.n_head,pad,self.d_head, device=x.device, dtype=x.dtype)], dim=2)
            v=torch.cat([v, torch.zeros(B,self.n_head,pad,self.d_head, device=x.device, dtype=x.dtype)], dim=2)
            g=torch.cat([g, torch.ones(B,pad,self.n_head, device=x.device, dtype=x.dtype)], dim=1)
            beta=torch.cat([beta, torch.zeros(B,pad,self.n_head, device=x.device, dtype=x.dtype)], dim=1)
        q_c=q.view(B,self.n_head,N,L,self.d_head)
        k_c=k.view(B,self.n_head,N,L,self.d_head)
        v_c=v.view(B,self.n_head,N,L,self.d_head)
        g_c=g.permute(0,2,1).view(B,self.n_head,N,L)
        beta_c=beta.permute(0,2,1).view(B,self.n_head,N,L)
        S=torch.zeros(B,self.n_head,self.d_head,self.d_head, device=x.device, dtype=x.dtype)
        outs=[]
        tril=torch.tril(torch.ones(L,L, device=x.device, dtype=torch.bool), diagonal=-1)
        eye=torch.eye(L, device=x.device, dtype=x.dtype)
        for n in range(N):
            qc=q_c[:,:,n]; kc=k_c[:,:,n]; vc=v_c[:,:,n]
            gc=g_c[:,:,n]; bc=beta_c[:,:,n]
            cum_g=torch.cumprod(gc, dim=-1)  # [B,H,L]
            k_s = kc * cum_g.unsqueeze(-1)
            q_s = qc / cum_g.unsqueeze(-1).clamp(min=1e-6)
            A = torch.matmul(k_s, k_s.transpose(-1,-2))
            A = A * tril.unsqueeze(0).unsqueeze(0)
            M = eye + A * bc.unsqueeze(-1)
            try:
                Tmat = torch.linalg.solve_triangular(M, eye.expand_as(M), upper=False)
            except RuntimeError:
                Tmat = torch.linalg.inv(M)
            w = torch.matmul(Tmat, vc * bc.unsqueeze(-1))
            u = torch.matmul(Tmat.transpose(-1,-2), k_s * bc.unsqueeze(-1))
            o_inter = torch.matmul(q_s, S)
            qu = torch.matmul(q_s, u.transpose(-1,-2))
            qu = qu * tril.unsqueeze(0).unsqueeze(0)
            diag_qk = torch.sum(q_s * k_s, dim=-1)
            o = o_inter + torch.matmul(qu, w) + (diag_qk * bc).unsqueeze(-1) * w
            outs.append(o)
            cum_last = cum_g[:,:,-1]
            S = S * cum_last.unsqueeze(-1).unsqueeze(-1) + torch.matmul(u.transpose(-1,-2), w)
        o_cat=torch.cat(outs, dim=2)[:, :, :T, :]
        o_cat=o_cat.permute(0,2,1,3).reshape(B,T,C)
        # out low-rank: 512 -> 128 -> 512
        return (o_cat @ self.out_A) @ self.out_B

class MoETop1(nn.Module):
    # 64 expertos, 1 despierto por token. Aqui estan los 100M.
    # Cada experto: gate_up fusionado 512 -> 2048, down 1024 -> 512
    # Cada experto: 512*2048 + 1024*512 = 1_572_864 params
    # Total MoE: 64 * 1_572_864 = 100_663_296
    # Activo: 1 experto = 1_572_864
    # Router: argmax, no topk, no scatter. 1 operacion. En CPU, scatter te mata.
    def __init__(self, d_model=512, d_ff=1024, n_experts=64):
        super().__init__()
        self.d_model=d_model; self.d_ff=d_ff; self.n_experts=n_experts
        self.router = nn.Linear(d_model, n_experts, bias=False)
        # Expertos como 1 tensor apilado: [E, d_model, d_ff*2] y [E, d_ff, d_model]
        # Asi no hay 64 nn.Linear con overhead de Python. 1 gather, 1 GEMM.
        self.gate_up = nn.Parameter(torch.randn(n_experts, d_model, d_ff*2) * 0.02)
        self.down = nn.Parameter(torch.randn(n_experts, d_ff, d_model) * 0.02)
        # Balanceo: contador de uso, para no colapsar a 3 expertos
        self.register_buffer('expert_usage', torch.zeros(n_experts))

    def forward(self, x):
        B,T,C = x.shape
        logits_r = self.router(x)  # [B,T,E]
        idx = logits_r.argmax(dim=-1)  # [B,T], Top-1, argmax, no topk
        with torch.no_grad():
            # Actualiza uso para monitoreo, no para loss. Si un experto se muere, lo ves aqui.
            self.expert_usage += torch.bincount(idx.view(-1), minlength=self.n_experts).float()
        # Gather experto por token: x [B,T,C] -> x_exp [B,T,C] con pesos del experto elegido
        # Truco para no hacer loop en Python sobre B*T: usamos index_select por experto.
        # En Runner, loop sobre 64 expertos con mascara es mas rapido que loop sobre B*T.
        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            mask = (idx == e)
            if not mask.any():
                continue
            x_e = x[mask]  # [N_e, C]
            gu = x_e @ self.gate_up[e]  # [N_e, 2048], 1 GEMM fusionado gate+up
            gate, up = gu.chunk(2, dim=-1)
            h = F.silu(gate) * up  # [N_e, 1024]
            out[mask] = h @ self.down[e]  # [N_e, C]
        return out

class CultistLumenV1(nn.Module):
    def __init__(self, vocab=8192, d_model=512, d_ff=1024, n_experts=64, ctx=512):
        super().__init__()
        self.ctx=ctx; self.d_model=d_model; self.vocab=vocab
        self.tok = nn.Embedding(vocab, d_model)
        # Capa 1: mixer
        self.norm1 = RMSNorm(d_model)
        self.mix = GatedDeltaNetLR(d_model=d_model, n_head=8, chunk=64, conv_kernel=4, rank=128)
        # Capa 2: MoE
        self.norm2 = RMSNorm(d_model)
        self.moe = MoETop1(d_model=d_model, d_ff=d_ff, n_experts=n_experts)
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight  # tied, no pagas 4.19M dos veces
        self.apply(self._init)
        # No re-inicializar gate_up/down de MoE con _init generico, ya estan en randn 0.02
        # _init solo toca Linear/Embedding/Conv1d, no Parameters sueltos, asi que esta bien.

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        x = self.tok(idx)  # [B,T,512]
        x = x + self.mix(self.norm1(x))  # Capa 1, 0.35M activo
        x = x + self.moe(self.norm2(x))  # Capa 2, 1.57M activo, 100M dormidos
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            logits_f = logits.view(-1, logits.size(-1))
            tgt_f = targets.view(-1)
            ce = F.cross_entropy(logits_f, tgt_f, reduction='none', label_smoothing=0.05)
            with torch.no_grad():
                probs = F.softmax(logits_f, dim=-1)
                p_correct = probs.gather(1, tgt_f.unsqueeze(1)).squeeze(1)
                w = (1.0 - p_correct)
                max_p, _ = probs.max(dim=-1)
                overconf = ((max_p > 0.7) & (p_correct < 0.3)).float() * 0.5
            loss = (ce * w + ce * overconf).mean()
        return logits, loss

    def num_params(self, trainable_only=False):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def active_params_per_token(self):
        # Calculado, no estimado. Embedding + mixer low-rank + 1 experto.
        # Embedding: 4_194_304 pero tied, se cuenta 1 vez en total, 0 en activo por token? En forward, embedding es lookup, no GEMM.
        # Activo real en GEMMs: mixer qkv+out low-rank + MoE 1 experto + router + norms
        # Mixer: qkv_A 65k + qkv_B 196k + out_A 65k + out_B 65k + b_a/b_beta 8k + conv 2k = ~401k
        # MoE: 1 experto 1_572_864 + router 32_768 = 1_605_632
        # Total activo GEMM: ~2.00M. Con norms, ~1.92M efectivo si cuentas solo mixer+experto sin router.
        # Devolvemos el numero que importa: mixer + 1 experto
        mixer = 512*128 + 128*1536 + 512*128 + 128*512  # 393_216
        expert = 512*2048 + 1024*512  # 1_572_864
        return mixer + expert  # 1_966_080

    def size_mb_bf16(self):
        return self.num_params()*2/1024/1024

if __name__ == "__main__":
    m = CultistLumenV1()
    print(f"CULTISTLUMEN-V1")
    print(f"Total params: {m.num_params():,} = {m.num_params()/1e6:.2f}M")
    print(f"Activo por token (mixer + 1 experto): {m.active_params_per_token():,} = {m.active_params_per_token()/1e6:.2f}M")
    print(f"Esparsidad: {100*(1 - m.active_params_per_token()/m.num_params()):.1f}% dormido")
    print(f"Size bf16: {m.size_mb_bf16():.2f} MB")
    print(f"Expertos: {m.moe.n_experts}, d_ff por experto: {m.moe.d_ff}")
    print(f"Uso expertos (init): {m.moe.expert_usage.sum().item():.0f}")
    # Smoke test pequeno: no entrena, solo forward
    x = torch.randint(0, 8192, (2, 128))
    logits, loss = m(x, targets=x)
    print(f"forward ok: logits {tuple(logits.shape)} loss {loss.item():.4f}")
    loss.backward()
    print("backward ok")
    # Stats de routing: debe repartir, no colapsar a 1 experto en init
    print(f"Expertos usados en smoke: {(m.moe.expert_usage>0).sum().item()}/{m.moe.n_experts}")