"""Export LUMEN V2.2 full-sequence and incremental KV-cache TorchScript methods."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from Model import LumenV2, active_layers_for_step

class InferenceGraph(torch.nn.Module):
    def __init__(self, model, active_layers):
        super().__init__()
        self.tok = model.tok
        self.blocks = model.blocks
        self.norm = model.norm
        self.head = model.head
        self.d_model = model.d_model
        self.active_layers = active_layers
        self.register_buffer("rope_cos", model.rope.cos.clone())
        self.register_buffer("rope_sin", model.rope.sin.clone())

    def _rotate(self, x, cos, sin):
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even*cos[..., 0::2]-odd*sin[..., 0::2],
                            even*sin[..., 0::2]+odd*cos[..., 0::2]), dim=-1).flatten(-2)

    def forward(self, token_ids):
        batch, seq_len = token_ids.shape
        x = self.tok(token_ids)
        cos = self.rope_cos.narrow(0, 0, seq_len).to(dtype=x.dtype)
        sin = self.rope_sin.narrow(0, 0, seq_len).to(dtype=x.dtype)
        for layer_index, block in enumerate(self.blocks):
            if layer_index in self.active_layers:
                normalized = block.norm1(x)
                qkv = block.attn.qkv(normalized).view(batch, seq_len, 3,
                    block.attn.n_head, block.attn.d_head)
                q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
                q = self._rotate(q, cos, sin)
                k = self._rotate(k, cos, sin)
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
                y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
                x = x + block.attn.out(y)
                x = x + block.mlp(block.norm2(x))
        return self.head(self.norm(x))

    @torch.jit.export
    def forward_cached(self, token_ids: torch.Tensor, cache_k: torch.Tensor,
                       cache_v: torch.Tensor, position: int):
        batch, seq_len = token_ids.shape
        x = self.tok(token_ids)
        cos = self.rope_cos.narrow(0, position, seq_len).to(dtype=x.dtype)
        sin = self.rope_sin.narrow(0, position, seq_len).to(dtype=x.dtype)
        for layer_index, block in enumerate(self.blocks):
            if layer_index in self.active_layers:
                normalized = block.norm1(x)
                qkv = block.attn.qkv(normalized).view(batch, seq_len, 3,
                    block.attn.n_head, block.attn.d_head)
                q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
                q = self._rotate(q, cos, sin)
                k = self._rotate(k, cos, sin)
                past_k = cache_k[layer_index].narrow(2, 0, position)
                past_v = cache_v[layer_index].narrow(2, 0, position)
                all_k = torch.cat((past_k, k), dim=2)
                all_v = torch.cat((past_v, v), dim=2)
                cache_k[layer_index, :, :, position:position+seq_len, :] = k
                cache_v[layer_index, :, :, position:position+seq_len, :] = v
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, all_k, all_v, is_causal=(seq_len > 1))
                y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
                x = x + block.attn.out(y)
                x = x + block.mlp(block.norm2(x))
        logits = self.head(self.norm(x))
        return logits, cache_k, cache_v

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--step",type=int,required=True)
    p.add_argument("--threads",type=int,default=2)
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    state=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    if int(state["step"]) != a.step:
        raise ValueError(f"checkpoint step mismatch: {state['step']} != {a.step}")
    model=LumenV2(vocab=8192,ctx=512,loss_mode="experimental")
    model.load_state_dict(state["model"],strict=True)
    model.eval()
    active=active_layers_for_step(a.step,"v22",len(model.blocks))
    graph=InferenceGraph(model,active).eval()
    scripted=torch.jit.script(graph)
    scripted=torch.jit.freeze(scripted,preserved_attrs=["forward_cached"])
    example=torch.zeros((1,16),dtype=torch.long)
    with torch.inference_mode():
        expected=model(example, active_layers=active, return_logits=True)[0]
        actual=scripted(example)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        cache_shape=(len(model.blocks),1,model.blocks[0].attn.n_head,model.ctx,model.blocks[0].attn.d_head)
        ck=torch.zeros(cache_shape); cv=torch.zeros(cache_shape)
        full=scripted(example)
        inc,ck,cv=scripted.forward_cached(example,ck,cv,0)
        torch.testing.assert_close(inc,full,rtol=2e-4,atol=2e-5)
        next_id=torch.tensor([[17]],dtype=torch.long)
        cached_next,_,_=scripted.forward_cached(next_id,ck,cv,16)
        full_next=scripted(torch.cat((example,next_id),dim=1))
        torch.testing.assert_close(cached_next[:,-1,:],full_next[:,-1,:],rtol=3e-4,atol=3e-5)
        scripted.save(a.out)
    print({"step":a.step,"active_layers":active,"output":a.out,
           "logits_shape":list(actual.shape),"kv_cache_shape":list(cache_shape),
           "checkpoint_sha256":__import__("hashlib").sha256(Path(a.checkpoint).read_bytes()).hexdigest()})
if __name__=="__main__": main()
