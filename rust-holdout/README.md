# LUMEN legacy holdout — Rust runner

This is the Rust entry point for running the frozen LUMEN V2.2 legacy holdout.

## Current parity-first stage

The Rust binary:
- validates that the frozen JSONL inputs contain exactly 2,431 MC rows and 100 freeform prompts;
- rejects malformed/non-object JSONL rows before model loading;
- fingerprints the checkpoint, tokenizer, system prompt and both holdout files with SHA-256;
- launches the existing PyTorch evaluator as one persistent child process;
- records wall-clock time and process status in `rust_pipeline_metadata.json`.

**The model forward pass is still PyTorch in this first stage.** This deliberately preserves the current evaluator's numerical behaviour while we establish a reproducible Rust-controlled pipeline. It is not yet a claim of faster Rust model inference.

## Run locally

From the repository root, with the same Python dependencies used by the normal evaluator:

```bash
cargo run --release --manifest-path rust-holdout/Cargo.toml -- \
  --python python \
  --script scripts/evaluate_v22_legacy_holdout.py \
  --checkpoint path/to/lumen-v22-step-10000.pt \
  --tokenizer path/to/tokenizer-v22-derived.json \
  --system system_prompt_short.txt \
  --mc-holdout path/to/holdout_multiple_choice.jsonl \
  --freeform-holdout path/to/freeform_identity_general_100.jsonl \
  --out-dir results/v22-step10000-rust \
  --step 10000 --threads 4
```

The runner passes the evaluator arguments through unchanged and writes an additional
`rust_pipeline_metadata.json` in the output directory.

## GitHub Actions

- Manual parity run: `.github/workflows/v22-rust-legacy-holdout.yml`
- Compile/test CI: `.github/workflows/rust-holdout-ci.yml`

The next migration stage can move tokenization, prompt assembly, scoring and eventually
the LUMEN forward pass behind Rust interfaces. Each stage must be checked against the
Python reference on the same checkpoint and frozen inputs before it becomes authoritative.
