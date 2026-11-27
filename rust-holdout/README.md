# LUMEN V2.2 — Rust inference runtime

The standard V2.2 evaluation/generation runner is **native Rust orchestration and decoding with LibTorch CPU kernels**. It no longer launches a Python inference process.

## Runtime responsibilities

- Rust loads the frozen tokenizer with `tokenizers`, assembles the system/user/assistant prompt, scores MC answer candidates and performs greedy freeform generation.
- Rust calls the exported TorchScript graph through `tch` / LibTorch. The model's tensor kernels are LibTorch CPU kernels; this is **not** a pure-Rust tensor backend.
- Freeform generation uses incremental KV caching: the prompt is prefetched once, then each new token updates the cache instead of recomputing the full prefix.
- The checkpoint is exported to TorchScript once with `scripts/export_v22_torchscript.py`. The exporter checks full-forward parity and incremental-cache logits before saving.
- The runtime records SHA-256 fingerprints, accuracy, gold-answer NLL, predictions, generated token IDs, freeform text and phase timings.

Training remains in Python. This migration changes the inference/evaluation runtime, not the training implementation.

## Reproduce the frozen V2.2 holdout (validated step 32,500)

First export the checkpoint once:

```bash
python scripts/export_v22_torchscript.py \
  --checkpoint path/to/lumen-v22-v5-fastpath-step-32500.pt \
  --out lumen-v22-step-32500.ts \
  --step 32500 --threads 2
```

Then run the Rust evaluator:

```bash
cargo run --release --manifest-path rust-holdout/Cargo.toml -- \
  --model lumen-v22-step-10000.ts \
  --tokenizer path/to/tokenizer-v22-derived.json \
  --system system_prompt_short.txt \
  --mc-holdout path/to/holdout_multiple_choice.jsonl \
  --freeform-holdout path/to/freeform_identity_general_100.jsonl \
  --out-dir results/v22-step32500-rust \
  --step 32500 --threads 2
```

Expected frozen inputs: 2,431 MC rows and 100 freeform prompts. Keep the checkpoint, tokenizer, system prompt, holdout files, decoding settings and thread configuration fixed when comparing runtimes.

## Parity and benchmark

The paired reference is Python evaluation on the same V2.2 step-10,000 checkpoint and frozen inputs. The first native Rust run measured:

- Python total wall time: 1,073.67 s
- Rust total wall time: 692.11 s
- Rust/Python speedup: 1.5513x (35.5% less wall time)
- MC accuracy: 829 / 2,431 (34.10%) in both
- All 2,431 serialized MC decision records match in order (2,256 unique IDs; 175 repeated-ID rows in the source holdout)
- Freeform generated token IDs and text: identical across all 100 prompts
- Maximum absolute MC candidate-score difference: 2.3842e-6

The original Actions comparison incorrectly compared complete prediction objects, including floating-point score arrays, with exact equality. The comparator now checks IDs/gold/prediction labels exactly and allows a strict 1e-5 maximum absolute tolerance for numerical scores. A fresh workflow run must pass this corrected comparator before the benchmark is treated as the merge gate.

The reported 35.5% is an end-to-end wall-time result for this particular CPU runner and frozen holdout. It is not a universal Rust speedup, and the gain must not be attributed solely to KV caching: MC candidate scoring dominates total runtime.

## GitHub Actions

- Native Rust inference and paired parity benchmark: `.github/workflows/v22-native-rust-holdout.yml`
- Rust compile/test CI: `.github/workflows/rust-holdout-ci.yml`



The Rust CI job installs CPU PyTorch only to provide the LibTorch headers and shared libraries required by `tch` at compile time; the inference executable itself does not launch a Python process.


## Single-question inference optimization (step 32,500)

The same-runner, same-prompt benchmark compares eager recomputation, TorchScript recomputation and TorchScript incremental KV-cache decoding at 1, 2 and 4 intra-op threads. All nine configurations produced identical token IDs. At 2 threads, the measured 48-token generation took 6.356 s with eager recomputation, 6.148 s with TorchScript recomputation and 0.399 s with TorchScript KV cache (15.92x vs eager). This is a generation-path measurement on one prompt, not a 15.92x claim for full holdout evaluation. The native Rust runtime already uses the exported `forward_cached` method; its default and the standard Actions workflow are now set to 2 intra-op threads. The paired 100-prompt + 2,431-question parity run at step 32,500 is the integration gate.
