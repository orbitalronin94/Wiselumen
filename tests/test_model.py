import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from Model import LumenV2, seq_for_step, active_layers_for_step
from Train import EMA, get_optimizer, save_sharded, train_steps


class LumenV2Tests(unittest.TestCase):
    def make_model(self, loss_mode="experimental"):
        return LumenV2(
            vocab=64, d_model=32, n_layer=2, n_head=4,
            d_ff=48, top_k=8, ctx=16, rope_max_len=32,
            loss_mode=loss_mode,
        )

    def test_rope_half_width_tables_match_interleaved_reference(self):
        from Model import RoPE
        rope = RoPE(head_dim=8, max_len=16)
        self.assertEqual(tuple(rope.cos.shape), (16, 4))
        torch.manual_seed(19)
        q = torch.randn(2, 4, 7, 8)
        k = torch.randn(2, 4, 7, 8)
        q_rot, k_rot = rope.rotate(q, k)
        cos = torch.repeat_interleave(rope.cos[:7], 2, dim=-1)[None, None, :, :]
        sin = torch.repeat_interleave(rope.sin[:7], 2, dim=-1)[None, None, :, :]
        def reference(x):
            even, odd = x[..., 0::2], x[..., 1::2]
            return torch.stack((even*cos[..., 0::2] - odd*sin[..., 0::2],
                                even*sin[..., 0::2] + odd*cos[..., 0::2]), dim=-1).flatten(-2)
        torch.testing.assert_close(q_rot, reference(q), rtol=0, atol=0)
        torch.testing.assert_close(k_rot, reference(k), rtol=0, atol=0)

    def test_full_layer_range_forward_matches_list(self):
        torch.manual_seed(23)
        model = self.make_model().eval()
        tokens = torch.randint(0, 64, (2, 9))
        with torch.no_grad():
            list_logits, list_loss = model(tokens, targets=tokens, active_layers=[0, 1])
            range_logits, range_loss = model(tokens, targets=tokens, active_layers=range(2))
        torch.testing.assert_close(range_logits, list_logits, rtol=0, atol=0)
        torch.testing.assert_close(range_loss, list_loss, rtol=0, atol=0)

    def test_v21_curricula_and_layer_skipping(self):
        self.assertEqual([seq_for_step(s) for s in (0, 14999, 15000, 30000)], [128, 128, 256, 512])
        self.assertEqual(seq_for_step(99999, "original"), 256)
        self.assertEqual(active_layers_for_step(0), [0, 1, 6, 7])
        self.assertEqual(active_layers_for_step(10000), [0, 1, 2, 5, 6, 7])
        self.assertEqual(active_layers_for_step(25000), list(range(8)))
        model = LumenV2(
            vocab=64, d_model=32, n_layer=8, n_head=4,
            d_ff=48, top_k=8, ctx=16, rope_max_len=32,
        ).eval()
        tokens = torch.randint(0, 64, (2, 12))
        with torch.no_grad():
            logits, loss = model(tokens, targets=tokens, active_layers=[0, 1, 6, 7])
        self.assertEqual(tuple(logits.shape), (2, 12, 64))
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(model.blocks), 8)


    def test_experimental_loss_weights_match_softmax_reference(self):
        torch.manual_seed(29)
        logits = torch.randn(17, 64, dtype=torch.float32)
        targets = torch.randint(0, 64, (17,))
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            old_p_correct = probs.gather(1, targets[:, None]).squeeze(1)
            old_max_prob = probs.max(dim=-1).values
            old_weight = (1.0 - old_p_correct) + (
                (old_max_prob > 0.7) & (old_p_correct < 0.3)
            ).float() * 0.5
            log_norm = torch.logsumexp(logits, dim=-1)
            p_correct = torch.exp(
                logits.gather(1, targets[:, None]).squeeze(1) - log_norm
            )
            max_prob = torch.exp(logits.max(dim=-1).values - log_norm)
            new_weight = (1.0 - p_correct) + (
                (max_prob > 0.7) & (p_correct < 0.3)
            ).float() * 0.5
        torch.testing.assert_close(new_weight, old_weight, rtol=2e-6, atol=2e-7)

    def test_masked_targets_supervise_only_assistant_positions(self):
        model = LumenV2(
            vocab=64, d_model=32, n_layer=2, n_head=4,
            d_ff=48, top_k=8, ctx=16, rope_max_len=32,
            loss_mode="cross_entropy",
        )
        tokens = torch.randint(0, 64, (2, 8))
        targets = tokens.clone()
        targets[:, :5] = -100
        logits, loss = model(tokens, targets=targets)
        expected = torch.nn.functional.cross_entropy(
            logits[:, 5:, :].reshape(-1, 64),
            targets[:, 5:].reshape(-1),
            label_smoothing=0.05,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.allclose(loss, expected, atol=1e-6))
        model.zero_grad(set_to_none=True)
        loss.backward()
        self.assertIsNotNone(model.tok.weight.grad)

    def test_all_masked_targets_fail_clearly(self):
        model = self.make_model()
        tokens = torch.randint(0, 64, (2, 8))
        with self.assertRaisesRegex(ValueError, "at least one supervised token"):
            model(tokens, targets=torch.full_like(tokens, -100))

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
