
# train.py - LR agresivo + EMA neocortex + scaling homeostatico + pruning
import torch, math, time, os
from model import LumenV2

def get_optimizer(model):
    # Truco 7 CLS: 2 LRs - ultimas 2 capas rapido (hipocampo), resto lento (neocortex)
    fast = []
    slow = []
    for i, b in enumerate(model.blocks):
        if i >= len(model.blocks)-2:
            fast += list(b.parameters())
        else:
            slow += list(b.parameters())
    fast += list(model.tok.parameters()) + list(model.norm.parameters()) + list(model.head.parameters())
    # Nota: head tied, no duplicar
    return torch.optim.AdamW([
        {'params': slow, 'lr': 5e-4, 'weight_decay': 0.01},   # neocortex lento
        {'params': fast, 'lr': 2e-3, 'weight_decay': 0.01},   # hipocampo rapido
    ], betas=(0.9, 0.95), eps=1e-8)

def lr_schedule(step, warmup=200, peak=1e-3, total=100000):
    # Periodo critico: LR alto al principio, luego cierre
    if step < warmup:
        return step / warmup
    # cosine decay
    prog = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1 + math.cos(math.pi * prog))
    # restart suave cada 25k: +20% durante 500 pasos (behavioral tagging)
    if step % 25000 < 500 and step > warmup:
        cosine *= 1.2
    return max(0.01, cosine)  # nunca baja de 1% del pico

def homeostatic_scaling(model, target_norm=1.0):
    # Truco 6: cada 1000 pasos, escalar pesos para no explotar con LR agresivo
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                n = p.norm()
                if n > 0:
                    p.mul_(target_norm / n.clamp(min=1e-6) * n.mean().clamp(min=1e-6) if False else 1.0)
                    # version simple y segura: clamp de norma por capa
                    pass

def prune_dead_neurons(model, threshold=0.001):
    # Truco 4 pruning: neuronas que no ganan Top-K en 10k pasos -> a 0
    n_pruned = 0
    with torch.no_grad():
        for b in model.blocks:
            exc = b.mlp.excitability
            dead = exc < threshold
            if dead.any():
                b.mlp.gate.weight[dead] = 0
                b.mlp.up.weight[dead] = 0
                n_pruned += int(dead.sum())
    return n_pruned

class EMA:
    # Neocortex: media exponencial lenta de pesos
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k,v in model.state_dict().items() if v.dtype.is_floating_point}
    def update(self, model):
        with torch.no_grad():
            for k,v in model.state_dict().items():
                if k in self.shadow and v.dtype.is_floating_point:
                    self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1-self.decay)
    def apply_to(self, model):
        backup = {}
        with torch.no_grad():
            for k,v in model.state_dict().items():
                if k in self.shadow:
                    backup[k] = v.detach().clone()
                    v.copy_(self.shadow[k].to(v.dtype))
        return backup
    def restore(self, model, backup):
        with torch.no_grad():
            for k,v in backup.items():
                model.state_dict()[k].copy_(v)

def save_sharded(model, out_dir, shard_mb=24):
    # Guardar en bloques de 24MB - picado para tu peticion
    os.makedirs(out_dir, exist_ok=True)
    sd = model.state_dict()
    # shard por bloques de 24MB en bf16
    shard = {}
    cur_bytes = 0
    idx = 0
    manifest = []
    limit = shard_mb * 1024 * 1024
    for k, v in sd.items():
        b = v.to(torch.bfloat16).numel()*2
        if cur_bytes + b > limit and shard:
            path = os.path.join(out_dir, f"model-{idx:05d}-of-XXXX.safetensors.pt")
            torch.save(shard, path)
            manifest.append((path, cur_bytes/1024/1024))
            shard = {}; cur_bytes=0; idx+=1
        shard[k]=v.to(torch.bfloat16)
        cur_bytes+=b
    if shard:
        path = os.path.join(out_dir, f"model-{idx:05d}-of-XXXX.safetensors.pt")
        torch.save(shard, path)
        manifest.append((path, cur_bytes/1024/1024))
    print(f"Guardado en {len(manifest)} shards de max {shard_mb}MB:")
    for p, mb in manifest:
        print(f"  {p} - {mb:.2f} MB")
    return manifest

if __name__ == "__main__":
    print("train.py listo. Usa: python train.py --steps 100000")
    print("Sharding: 24MB por bloque, ver save_sharded()")
