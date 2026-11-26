"""Export the frozen LUMEN checkpoint to a CPU TorchScript inference graph."""
import argparse
from pathlib import Path
import torch
from Model import LumenV2, active_layers_for_step

class InferenceGraph(torch.nn.Module):
    def __init__(self, model, active_layers):
        super().__init__()
        self.model = model
        self.active_layers = active_layers
    def forward(self, token_ids):
        logits, _ = self.model(token_ids, active_layers=self.active_layers, return_logits=True)
        return logits

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--step",type=int,required=True)
    p.add_argument("--threads",type=int,default=4)
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
    example=torch.zeros((1,16),dtype=torch.long)
    with torch.inference_mode():
        traced=torch.jit.trace(graph,example,strict=True)
        traced=torch.jit.freeze(traced)
        traced.save(a.out)
        expected=graph(example)
        actual=torch.jit.load(a.out,map_location="cpu")(example)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    print({"step":a.step,"active_layers":active,"output":a.out,"logits_shape":list(actual.shape),
           "checkpoint_sha256":__import__("hashlib").sha256(Path(a.checkpoint).read_bytes()).hexdigest()})
if __name__=="__main__": main()
