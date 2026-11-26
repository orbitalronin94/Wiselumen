
# LUMEN V2 - picado a mano, bloque a bloque
# Sin transformers, sin magia. Solo torch.
# Trucos: Top-K esparso, excitabilidad CREB, scaling homeostatico, 2 LRs
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        # RMSNorm: mas rapido que LayerNorm, estabiliza LR agresivo
        r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return x * r * self.w

class RoPE:
    # RoPE picado a mano, sin libreria
    def __init__(self, d_head, max_len=2048):
        inv = 1.0 / (10000 ** (torch.arange(0, d_head, 2).float() / d_head))
        pos = torch.arange(max_len).float()
        freqs = torch.outer(pos, inv)
        self.cos = torch.cos(freqs).repeat_interleave(2, dim=-1)  # [T, d_head]
        self.sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
    def apply(self, q, k):
        # q,k: [B, H, T, D]
        T = q.shape[2]
        cos = self.cos[:T].to(q.device)[None, None, :, :]
        sin = self.sin[:T].to(q.device)[None, None, :, :]
        def rot(x):
            x1 = x[..., 0::2]
            x2 = x[..., 1::2]
            # interleave rotado
            out = torch.stack([x1*cos[...,0::2]-x2*sin[...,0::2], x1*sin[...,0::2]+x2*cos[...,0::2]], dim=-1)
            return out.flatten(-2)
        return rot(q), rot(k)

class SparseMLP(nn.Module):
    # Truco 5: Sparse Coding + Truco 1: Excitabilidad CREB
    # Solo Top-K neuronas activas. Las que ganaron antes tienen bonus.
    def __init__(self, d_model, d_ff, top_k=256):
        super().__init__()
        self.d_ff = d_ff
        self.top_k = top_k
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        # excitabilidad: buffer no entrenable, decae con el tiempo
        self.register_buffer('excitability', torch.zeros(d_ff))
        self.decay = 0.99
    def forward(self, x):
        # SwiGLU
        h = F.silu(self.gate(x)) * self.up(x)  # [B,T,FF]
        # Excitabilidad: bonus a las que ganaron antes
        h = h + self.excitability * 0.1
        # Top-K: solo las K mas grandes sobreviven
        vals, idx = torch.topk(h, k=self.top_k, dim=-1)
        mask = torch.zeros_like(h).scatter(-1, idx, 1.0)
        h_sparse = h * mask
        # Actualizar excitabilidad: las ganadoras suben, resto decae
        with torch.no_grad():
            win = mask.mean(dim=(0,1))  # [FF] cuanto gano cada neurona
            self.excitability.mul_(self.decay).add_(win * 0.5)
        return self.down(h_sparse)

class Attention(nn.Module):
    def __init__(self, d_model, n_head, rope):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.rope = rope
        self.qkv = nn.Linear(d_model, d_model*3, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
    def forward(self, x):
        B,T,C = x.shape
        qkv = self.qkv(x).view(B,T,3,self.n_head,self.d_head).transpose(1,3) # [B,H,T,3,D]
        q,k,v = qkv[:,:, :,0], qkv[:,:, :,1], qkv[:,:, :,2]
        q,k = self.rope.apply(q,k)
        # SDPA = Flash Attention, obligatorio para pasos/seg
        y = F.scaled_dot_product_attention(q,k,v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B,T,C)
        return self.out(y)

class Block(nn.Module):
    def __init__(self, d_model, n_head, d_ff, top_k, rope):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = Attention(d_model, n_head, rope)
        self.norm2 = RMSNorm(d_model)
        self.mlp = SparseMLP(d_model, d_ff, top_k)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class LumenV2(nn.Module):
    # Arquitectura completa, picada a mano
    def __init__(self, vocab=8192, d_model=512, n_layer=8, n_head=8, d_ff=1536, top_k=256, ctx=512):
        super().__init__()
        self.ctx = ctx
        self.d_model = d_model
        self.tok = nn.Embedding(vocab, d_model)
        self.rope = RoPE(d_model//n_head, max_len=2048)
        self.blocks = nn.ModuleList([Block(d_model, n_head, d_ff, top_k, self.rope) for _ in range(n_layer)])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        # weight tying: head = tok, ahorra 4M params y 8MB
        self.head.weight = self.tok.weight
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        # idx: [B,T]
        x = self.tok(idx)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            # Truco 4: Error de prediccion - ponderar por (1 - p_correcta)
            # + penalizacion de confianza cuando falla
            logits_f = logits.view(-1, logits.size(-1))
            tgt_f = targets.view(-1)
            ce = F.cross_entropy(logits_f, tgt_f, reduction='none', label_smoothing=0.05)
            with torch.no_grad():
                probs = F.softmax(logits_f, dim=-1)
                p_correct = probs.gather(1, tgt_f.unsqueeze(1)).squeeze(1)  # [N]
                # si ya acierta, casi no mueve pesos. Si falla con confianza, mueve mucho
                w = (1.0 - p_correct)  # 0..1
                # penalizacion extra si confia y falla: p_correct < 0.3 y max_prob > 0.7
                max_p, _ = probs.max(dim=-1)
                overconf = ((max_p > 0.7) & (p_correct < 0.3)).float() * 0.5
            loss = (ce * w + ce * overconf).mean()
        return logits, loss
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    def size_mb_bf16(self):
        return self.num_params()*2/1024/1024

if __name__ == "__main__":
    m = LumenV2()
    print(f"Params: {m.num_params()/1e6:.2f}M")
    print(f"Size bf16: {m.size_mb_bf16():.2f} MB")
    print(f"Bloques de 24MB necesarios: {int(m.size_mb_bf16()//24)+1}")
    x = torch.randint(0,8192,(2,32))
    logits, loss = m(x, targets=x)
    print(f"Logits {logits.shape} Loss {loss.item():.4f}")
