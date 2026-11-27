"""Paired 10-step architecture audit: pre-audit MAIN vs optimized MAIN vs sparse MLP."""
import hashlib, importlib.util, json, os, subprocess, time, types
from pathlib import Path
import torch
import torch.nn.functional as F

BASE_SHA = "d9479e7f8d69b3068aba6adefcac2928b86f76f2"
SEED, BATCH, SEQ = 20261127, 2, 128
WARMUP, MEASURE = 3, 10
THREADS, INTEROP = 9, 1
START_STEP = 25000

def load_model_module(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def sparse_mlp_forward(self, x):
    gate, up = self.gate_up(x).chunk(2, dim=-1)
    h = F.silu(gate) * up
    baseline = float(self.top_k) / float(self.d_ff)
    h = h + (self.excitability - baseline).to(h.dtype) * self.excitability_strength
    indices = torch.topk(h, k=self.top_k, dim=-1).indices
    selected = h.gather(-1, indices)
    flat_indices = indices.reshape(-1)
    rows = torch.arange(flat_indices.numel() // self.top_k, device=x.device).repeat_interleave(self.top_k)
    sparse_indices = torch.stack((rows, flat_indices), dim=0)
    sparse_values = selected.reshape(-1)
    sparse_h = torch.sparse_coo_tensor(
        sparse_indices, sparse_values, (rows[-1].item() + 1, self.d_ff),
        device=x.device, dtype=h.dtype,
    ).coalesce()
    out = torch.sparse.mm(sparse_h, self.down.weight.t()).reshape(*x.shape[:-1], self.down.weight.shape[0])
    if self.training:
        with torch.no_grad():
            counts = torch.bincount(flat_indices, minlength=self.d_ff).to(self.excitability.dtype)
            win_rate = counts / float(flat_indices.numel())
            self.excitability.mul_(self.excitability_decay).add_(win_rate, alpha=1.0-self.excitability_decay)
    return out

def make_model(module, state):
    torch.manual_seed(SEED)
    model = module.LumenV2(vocab=8192, ctx=512, loss_mode="experimental")
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model

def make_batches():
    gen = torch.Generator().manual_seed(SEED + 99)
    batches = []
    for _ in range(WARMUP + MEASURE):
        x = torch.randint(4, 8184, (BATCH, SEQ), generator=gen)
        y = torch.randint(4, 8184, (BATCH, SEQ), generator=gen)
        y[:, :int(SEQ * .6)] = -100
        batches.append((x, y))
    hasher = hashlib.sha256()\n    for x, y in batches:\n        hasher.update(bytes(x.contiguous().untyped_storage()))\n        hasher.update(bytes(y.contiguous().untyped_storage()))\n    digest = hasher.hexdigest()
    return batches, digest

def train_step(model, optimizer, batch, step):
    model.train()
    x, y = batch
    optimizer.zero_grad(set_to_none=True)
    _, loss = model(x, targets=y, active_layers=range(len(model.blocks)), return_logits=False)
    if loss is None or not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    loss.backward()
    if (step + 1) % 5 == 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("non-finite gradient norm")
    optimizer.step()
    return float(loss.detach().item())

def run_arm(label, model, optimizer, batches):
    timings, losses = [], []
    for i, batch in enumerate(batches):
        started = time.perf_counter()
        loss = train_step(model, optimizer, batch, START_STEP + i)
        timings.append(time.perf_counter() - started)
        losses.append(loss)
    return {"label":label,"elapsed_seconds":sum(timings),"steps_per_second":len(timings)/sum(timings),
            "mean_loss":sum(losses)/len(losses),"losses":losses}

def main():
    torch.set_num_threads(THREADS)
    try: torch.set_num_interop_threads(INTEROP)
    except RuntimeError: pass
    torch.backends.mkldnn.enabled = True
    baseline_source = subprocess.check_output(["git","show",f"{BASE_SHA}:Model.py"], text=True)
    Path("/tmp/Model_baseline.py").write_text(baseline_source, encoding="utf-8")
    baseline_mod = load_model_module("baseline_model", "/tmp/Model_baseline.py")
    current_mod = load_model_module("current_model", str(Path("Model.py").resolve()))
    torch.manual_seed(SEED)
    seed_model = current_mod.LumenV2(vocab=8192, ctx=512, loss_mode="experimental")
    state = {k:v.detach().clone() for k,v in seed_model.state_dict().items()}
    baseline = make_model(baseline_mod, state)
    current = make_model(current_mod, state)
    sparse = make_model(current_mod, state)
    for block in sparse.blocks:
        block.mlp.forward = types.MethodType(sparse_mlp_forward, block.mlp)
    # Verify sparse path numerically before timing it.
    sparse.eval(); current.eval()
    probe = torch.randint(4, 8184, (1, 16))
    with torch.no_grad():
        dense_out, _ = current(probe, active_layers=range(len(current.blocks)), return_logits=True)
        sparse_out, _ = sparse(probe, active_layers=range(len(sparse.blocks)), return_logits=True)
    torch.testing.assert_close(sparse_out, dense_out, rtol=2e-4, atol=2e-5)
    arms = {
        "pre_audit_main": (baseline, torch.optim.AdamW(baseline.parameters(), lr=2e-4, betas=(.9,.95), weight_decay=.01)),
        "optimized_main": (current, torch.optim.AdamW(current.parameters(), lr=2e-4, betas=(.9,.95), weight_decay=.01)),
        "optimized_main_sparse_mlp": (sparse, torch.optim.AdamW(sparse.parameters(), lr=2e-4, betas=(.9,.95), weight_decay=.01)),
    }
    batches, batch_sha = make_batches()
    # Warm each arm; alternate arm order to reduce ordering/thermal bias.
    for i in range(WARMUP):
        order = list(arms.items())
        if i % 2: order.reverse()
        for label, (model, optimizer) in order:
            train_step(model, optimizer, batches[i], START_STEP + i)
    results = {name: {"times": [], "losses": []} for name in arms}
    for i in range(MEASURE):
        order = list(arms.items())
        shift = i % len(order)
        order = order[shift:] + order[:shift]
        for label, (model, optimizer) in order:
            started = time.perf_counter()
            loss = train_step(model, optimizer, batches[WARMUP+i], START_STEP + WARMUP + i)
            results[label]["times"].append(time.perf_counter()-started)
            results[label]["losses"].append(loss)
    summary = {}
    for label, values in results.items():
        elapsed = sum(values["times"])
        summary[label] = {"elapsed_seconds":elapsed,"steps_per_second":MEASURE/elapsed,
                          "mean_loss":sum(values["losses"])/MEASURE,
                          "losses":values["losses"]}
    base_speed = summary["pre_audit_main"]["steps_per_second"]
    for label in ("optimized_main","optimized_main_sparse_mlp"):
        summary[label]["speed_gain_vs_pre_audit_percent"] = (summary[label]["steps_per_second"]/base_speed-1)*100
        summary[label]["loss_delta_vs_pre_audit_percent"] = (
            summary[label]["mean_loss"]/summary["pre_audit_main"]["mean_loss"]-1)*100
    result = {
        "benchmark":"paired architecture audit; 3 arms interleaved on one runner",
        "baseline_commit":BASE_SHA,"candidate_commit":os.environ.get("GITHUB_SHA","unknown"),
        "threads":{"intraop":torch.get_num_threads(),"interop":torch.get_num_interop_threads()},
        "warmup_steps_each":WARMUP,"measured_steps_each":MEASURE,
        "batch_size":BATCH,"sequence_length":SEQ,"seed":SEED,"batch_sha256":batch_sha,
        "sparse_forward_equivalence":"passed (rtol=2e-4, atol=2e-5)",
        "results":summary,
        "note":"10 measured updates per arm are a screening result, not a long-run quality validation."
    }
    Path("architecture-audit-result.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("ARCHITECTURE_AUDIT_RESULT "+json.dumps(result),flush=True)
    Path(os.getenv("GITHUB_STEP_SUMMARY","bench.md")).write_text(
        "# LUMEN architecture audit\n\n"+
        "\n\n".join(f"**{k}**: {v['steps_per_second']:.4f} steps/s; loss {v['mean_loss']:.6f}; "
        f"speed vs pre-audit {v.get('speed_gain_vs_pre_audit_percent',0):+.2f}%"
        for k,v in summary.items())+"\n",encoding="utf-8")

if __name__=="__main__": main()
