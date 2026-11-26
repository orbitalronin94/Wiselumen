import tempfile
import unittest
from pathlib import Path

import torch

from Model import LumenV2, seq_for_step, active_layers_for_step
from Train import EMA, get_optimizer, save_sharded, train_steps


class LumenV2Tests(unittest.TestCase):
    def make_model(self, loss_mode="experimental"):
        return LumenV2(
            vocab=64, d_model=32, n_layer=2, n_head=4,
            d_ff=48, top_k=8, ctx=16, rope_max_len=32,
            loss_mode=loss_mode,
        )

    def test_v21_curricula_and_layer_skipping(self):
        self.assertEqual([seq_for_step(s) for s in (0, 14999, 15000, 30000)], [128, 128, 256, 512])
        self.assertEqual(seq_for_step(99999, "original"), 256)
        self.assertEqual(active_layers_for_step(0), [0, 1, 6, 7])
        self.assertEqual(active_layers_for_step(10000), [0, 1, 2, 5, 6, 7])
        self.assertEqual(active_layers_for_step(25000), list(range(8)))
        model = self.make_model().eval()
        tokens = torch.randint(0, 64, (2, 12))
        with torch.no_grad():
            logits, loss = model(tokens, targets=tokens, active_layers=[0, 1])
        self.assertEqual(tuple(logits.shape), (2, 12, 64))
        self.assertTrue(torch.isfinite(loss))

    def test_forward_backward_and_weight_tying(self):
        model = self.make_model()
        tokens = torch.randint(0, 64, (2, 12))
        logits, loss = model(tokens, targets=tokens)
        self.assertEqual(tuple(logits.shape), (2, 12, 64))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIs(model.tok.weight, model.head.weight)
        self.assertGreater(model.tok.weight.grad.abs().sum().item(), 0)

    def test_causal_attention_does_not_read_future_tokens(self):
        torch.manual_seed(7)
        model = self.make_model().eval()
        first = torch.randint(0, 64, (1, 10))
        second = first.clone()
        second[:, 5:] = (second[:, 5:] + 13) % 64
        with torch.no_grad():
            logits_a, _ = model(first)
            logits_b, _ = model(second)
        torch.testing.assert_close(logits_a[:, :5], logits_b[:, :5])

    def test_excitability_is_bounded_frequency_ema(self):
        model = self.make_model().train()
        tokens = torch.randint(0, 64, (2, 12))
        model(tokens, targets=tokens)
        for block in model.blocks:
            self.assertTrue(torch.all(block.mlp.excitability >= 0))
            self.assertTrue(torch.all(block.mlp.excitability <= 1))

    def test_optimizer_has_no_duplicate_parameters(self):
        model = self.make_model()
        optimizer = get_optimizer(model)
        ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})

    def test_two_step_training_and_ema(self):
        model = self.make_model()
        ema = EMA(model, decay=0.9)
        batches = [
            (torch.randint(0, 64, (2, 12)), torch.randint(0, 64, (2, 12)))
            for _ in range(2)
        ]
        result = train_steps(
            model, batches, steps=2, warmup=1, total_steps=10,
            start_step=0, ema=ema, log_every=0,
        )
        self.assertEqual(result["steps"], 2)
        self.assertTrue(torch.isfinite(torch.tensor(result["last_loss"])))

    def test_sharded_save_writes_manifest(self):
        model = self.make_model()
        with tempfile.TemporaryDirectory() as directory:
            manifest = save_sharded(model, directory, shard_mb=1)
            self.assertGreaterEqual(manifest["shard_count"], 1)
            self.assertTrue((Path(directory) / "model-manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
