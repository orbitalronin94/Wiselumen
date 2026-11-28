# CULTISTLUMEN-V1-FAST: CPU-oriented ablation for throughput benchmarking.
# Replaces chunkwise triangular solves and per-bank scatter with causal depthwise
# convolution + low-rank projection and a single dense gated FFN.
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

class FastConvMixer(nn.Module):
    def __init__(self, d_model=1024, rank=128, kernel=4):
        super().__init__()
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel, groups=d_model, bias=True)
        self.in_proj = nn.Linear(d_model, rank, bias=False)
        self.out_proj = nn.Linear(rank, d_model, bias=False)
        self.kernel = kernel
    def forward(self, x):
        y = F.pad(x.transpose(1, 2), (self.kernel - 1, 0))
        y = F.silu(self.dw(y).transpose(1, 2))
        return self.out_proj(F.silu(self.in_proj(y)))

class DenseGatedFFN(nn.Module):
    def __init__(self, d_model=1024, d_ff=4096):
        super().__init__()
        self.gate_up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.active_fraction = 1.0
    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)

class CultistLumenV1Fast(nn.Module):
    def __init__(self, vocab=8192, d_model=1024, d_ff=4096, ctx=128, n_head=16, rank=128):
        super().__init__()
        if d_model % n_head:
            raise ValueError("d_model must divide n_head")
        self.ctx, self.d_model, self.vocab = ctx, d_model, vocab
        self.tok = nn.Embedding(vocab, d_model)
        self.norm1 = RMSNorm(d_model)
        self.mix = FastConvMixer(d_model, rank, kernel=4)
        self.norm2 = RMSNorm(d_model)
        self.ffn = DenseGatedFFN(d_model, d_ff)
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            if m.weight is not self.tok.weight:
                nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    def forward(self, idx, targets=None, sampled_negatives=256):
        x = self.tok(idx)
        x = x + self.mix(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        x = self.norm(x)
        if targets is None:
            return self.head(x), None
        flat_x = x.reshape(-1, self.d_model).float()
        tgt = targets.reshape(-1)
        positives = torch.unique(tgt)
        n = min(int(sampled_negatives), self.vocab - positives.numel())
        perm = torch.randperm(self.vocab, device=idx.device)
        negatives = perm[~torch.isin(perm, positives)][:n]
        candidates = torch.sort(torch.cat((positives, negatives))).values
        logits = F.linear(flat_x, self.tok.weight[candidates].float())
        positions = torch.searchsorted(candidates, tgt)
        target_logits = logits.gather(1, positions[:, None]).squeeze(1)
        mask = torch.isin(candidates, negatives)
        prob = n / max(1, self.vocab - positives.numel())
        corrected = logits.clone()
        corrected[:, mask] -= math.log(max(prob, 1e-12))
        logz = torch.logsumexp(corrected, dim=-1)
        ce = logz - 0.95 * target_logits - 0.05 * corrected.mean(-1)
        with torch.no_grad():
            pc = torch.exp(target_logits - logz)
            mp = torch.exp(corrected.max(-1).values - logz)
            weight = 1.0 - pc
            over = ((mp > 0.7) & (pc < 0.3)).float() * 0.5
        loss = (ce * weight + ce * over).mean()
        return corrected, loss
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
