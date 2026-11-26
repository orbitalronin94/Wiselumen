# LUMEN V2 — arquitectura experimental implementada en PyTorch.
# Mantiene: RoPE, RMSNorm, SwiGLU, Top-K, excitabilidad y loss experimental.
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * rms.to(dtype=x.dtype)) * self.weight


class RoPE:
    """Rotary positional embeddings. Tensors are kept on CPU and moved to q/k's device."""
    def __init__(self, head_dim, max_len=2048, base=10000.0):
        if head_dim % 2:
            raise ValueError(f"RoPE requires an even head dimension, got {head_dim}.")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_len).float()
        freqs = torch.outer(positions, inv_freq)
        self.cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1)
        self.sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1)

    def apply(self, q, k):
        seq_len = q.shape[-2]
        if seq_len > self.cos.shape[0]:
            raise ValueError(f"Sequence length {seq_len} exceeds RoPE limit {self.cos.shape[0]}.")
        cos = self.cos[:seq_len].to(device=q.device, dtype=q.dtype)[None, None, :, :]
        sin = self.sin[:seq_len].to(device=q.device, dtype=q.dtype)[None, None, :, :]

        def rotate(x):
            x_even, x_odd = x[..., 0::2], x[..., 1::2]
            rotated = torch.stack(
                (x_even * cos[..., 0::2] - x_odd * sin[..., 0::2],
                 x_even * sin[..., 0::2] + x_odd * cos[..., 0::2]),
                dim=-1,
            )
            return rotated.flatten(-2)

        return rotate(q), rotate(k)


class SparseMLP(nn.Module):
    """SwiGLU with Top-K activation and bounded, frequency-based excitability."""
    def __init__(self, d_model, d_ff, top_k=256, excitability_strength=0.1, excitability_decay=0.99):
        super().__init__()
        if not 1 <= top_k <= d_ff:
            raise ValueError(f"top_k must be in [1, {d_ff}], got {top_k}.")
        self.d_ff = d_ff
        self.top_k = top_k
        self.excitability_strength = excitability_strength
        self.excitability_decay = excitability_decay
        # Una GEMM para las dos proyecciones SwiGLU: misma función matemática,
        # menos lecturas de x y mejor aprovechamiento de GEMM en CPU.
        self.gate_up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        # EMA de frecuencia de selección por neurona (no es un parámetro entrenable).
        self.register_buffer("excitability", torch.zeros(d_ff))

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        h = F.silu(gate) * up

        # Centrar respecto a la frecuencia uniforme esperada evita que el bonus
        # crezca sin límite y que domine progresivamente a la activación.
        baseline = float(self.top_k) / float(self.d_ff)
        centered_exc = self.excitability - baseline
        h = h + centered_exc.to(dtype=h.dtype) * self.excitability_strength

        indices = torch.topk(h, k=self.top_k, dim=-1).indices
        mask = torch.zeros_like(h).scatter_(-1, indices, 1.0)
        sparse_h = h * mask

        if self.training:
            with torch.no_grad():
                win_rate = mask.float().mean(dim=tuple(range(mask.ndim - 1)))
                self.excitability.mul_(self.excitability_decay).add_(
                    win_rate.to(self.excitability.dtype), alpha=1.0 - self.excitability_decay
                )

        return self.down(sparse_h)


class Attention(nn.Module):
    def __init__(self, d_model, n_head, rope):
        super().__init__()
        if d_model % n_head:
            raise ValueError("d_model must be divisible by n_head.")
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.rope = rope
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch, seq_len, channels = x.shape
        # [B,T,3,H,D] -> [3,B,H,T,D]. La dimensión 3 debe separarse
        # antes de seleccionar Q, K y V (el orden anterior indexaba T por error).
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_head, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        q, k = self.rope.apply(q, k)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, channels)
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


def seq_for_step(step, mode="v21"):
    """Context curriculum; caller must request batches of the returned length."""
    if mode == "original":
        return 256
    if step < 15000:
        return 128
    if step < 30000:
        return 256
    return 512


def active_layers_for_step(step, mode="v21", n_layer=8):
    """Skip intermediate blocks without deleting parameters or changing checkpoints."""
    if mode == "original":
        return list(range(n_layer))
    if step < 10000:
        return [i for i in (0, 1, 6, 7) if i < n_layer]
    if step < 25000:
        return [i for i in (0, 1, 2, 5, 6, 7) if i < n_layer]
    return list(range(n_layer))


class LumenV2(nn.Module):
    def __init__(
        self, vocab=8192, d_model=512, n_layer=8, n_head=8,
        d_ff=1536, top_k=256, ctx=512, rope_max_len=2048,
        loss_mode="experimental",
    ):
        super().__init__()
        if vocab <= 0 or d_model <= 0 or n_layer <= 0 or d_ff <= 0:
            raise ValueError("vocab, d_model, n_layer and d_ff must be positive.")
        if d_model % n_head:
            raise ValueError("d_model must be divisible by n_head.")
        if (d_model // n_head) % 2:
            raise ValueError("d_model // n_head must be even for RoPE.")
        if ctx <= 0 or ctx > rope_max_len:
            raise ValueError(f"ctx must be in [1, {rope_max_len}].")
        if loss_mode not in {"experimental", "cross_entropy"}:
            raise ValueError("loss_mode must be 'experimental' or 'cross_entropy'.")

        self.ctx = ctx
        self.d_model = d_model
        self.vocab = vocab
        self.loss_mode = loss_mode
        self.tok = nn.Embedding(vocab, d_model)
        self.rope = RoPE(d_model // n_head, max_len=rope_max_len)
        self.blocks = nn.ModuleList([
            Block(d_model, n_head, d_ff, top_k, self.rope) for _ in range(n_layer)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.tok.weight  # weight tying real: un único parámetro
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None, active_layers=None):
        if idx.ndim != 2:
            raise ValueError(f"idx must have shape [batch, sequence], got {tuple(idx.shape)}.")
        if idx.shape[1] > self.ctx:
            raise ValueError(f"Sequence length {idx.shape[1]} exceeds configured context {self.ctx}.")
        if idx.numel() and (idx.min() < 0 or idx.max() >= self.vocab):
            raise ValueError(f"Token IDs must be in [0, {self.vocab - 1}].")

        x = self.tok(idx)
        if active_layers is None:
            active_layers = range(len(self.blocks))
        else:
            active_layers = set(active_layers)
            invalid = active_layers.difference(range(len(self.blocks)))
            if invalid:
                raise ValueError(f"active_layers contains invalid indices: {sorted(invalid)}")
        for layer_index, block in enumerate(self.blocks):
            if layer_index in active_layers:
                x = block(x)
        logits = self.head(self.norm(x))
        loss = None

        if targets is not None:
            if targets.shape != idx.shape:
                raise ValueError("targets must have the same [batch, sequence] shape as idx.")
            flat_logits = logits.reshape(-1, self.vocab)
            flat_targets = targets.reshape(-1)
            # -100 marca posiciones sin supervisión (system/user/padding).
            # Se filtran antes de CE y gather: p_correct nunca recibe una etiqueta
            # inválida, y el gradiente solo nace en posiciones objetivo del assistant.
            valid = flat_targets.ne(-100)
            if not torch.any(valid):
                raise ValueError("targets must contain at least one supervised token.")
            supervised_logits = flat_logits[valid]
            supervised_targets = flat_targets[valid]
            ce = F.cross_entropy(
                supervised_logits, supervised_targets,
                reduction="none", label_smoothing=0.05,
            )

            if self.loss_mode == "cross_entropy":
                loss = ce.mean()
            else:
                # Hipótesis experimental original, calculada solo en tokens supervisados.
                with torch.no_grad():
                    probs = F.softmax(supervised_logits.float(), dim=-1)
                    p_correct = probs.gather(1, supervised_targets.unsqueeze(1)).squeeze(1)
                    max_prob = probs.max(dim=-1).values
                    error_weight = 1.0 - p_correct
                    overconfidence = (
                        (max_prob > 0.7) & (p_correct < 0.3)
                    ).to(ce.dtype) * 0.5
                loss = (ce * (error_weight + overconfidence)).mean()

        return logits, loss

    def num_params(self):
        # parameters() elimina referencias duplicadas del weight tying.
        return sum(parameter.numel() for parameter in self.parameters())

    def size_mb_bf16(self):
        return self.num_params() * 2 / (1024 ** 2)


if __name__ == "__main__":
    model = LumenV2()
    tokens = torch.randint(0, model.vocab, (2, 32))
    logits, loss = model(tokens, targets=tokens)
    logits.sum().backward()
    print(f"Params: {model.num_params() / 1e6:.3f}M")
    print(f"BF16 weights: {model.size_mb_bf16():.2f} MiB")
    print(f"Logits: {tuple(logits.shape)} | Loss: {loss.item():.4f}")
    print("Forward + backward: OK")
