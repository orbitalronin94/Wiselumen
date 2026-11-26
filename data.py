"""Strict V4-to-chat formatting for LUMEN V2.2.

This module is opt-in. It refuses to train unless the tokenizer explicitly
contains the eight reserved control tokens at IDs 8184..8191.
"""
from pathlib import Path

import torch

VOCAB_SIZE = 8192
BASE_VOCAB_SIZE = 8184
PAD_ID = 0
EOS_TOKEN = "[EOS]"
ROLE_IDS = {
    "<|system|>": 8184,
    "<|end_system|>": 8185,
    "<|user|>": 8186,
    "<|end_user|>": 8187,
    "<|assistant|>": 8188,
    "<|end_assistant|>": 8189,
    "<|bos|>": 8190,
    "<|sep|>": 8191,
}


def validate_tokenizer(tokenizer):
    """Fail closed unless the frozen/derived tokenizer has the exact V2.2 layout."""
    size = tokenizer.get_vocab_size(with_added_tokens=True)
    if size != VOCAB_SIZE:
        raise ValueError(
            f"V2.2 requires tokenizer vocab size {VOCAB_SIZE}; got {size}. "
            "Derive a tokenizer with 8184 base IDs plus 8 control IDs."
        )
    if tokenizer.token_to_id("[PAD]") != PAD_ID:
        raise ValueError("Tokenizer must define [PAD] at ID 0.")
    if tokenizer.token_to_id(EOS_TOKEN) is None:
        raise ValueError("Tokenizer must define [EOS].")
    for token, expected_id in ROLE_IDS.items():
        actual_id = tokenizer.token_to_id(token)
        if actual_id != expected_id:
            raise ValueError(
                f"Control token {token} must map to ID {expected_id}; got {actual_id}."
            )
    return {
        "vocab_size": size,
        "pad_id": PAD_ID,
        "eos_id": tokenizer.token_to_id(EOS_TOKEN),
        "role_ids": dict(ROLE_IDS),
    }


def _encode_text(tokenizer, text):
    ids = list(tokenizer.encode(str(text), add_special_tokens=False).ids)
    if any(token_id < 0 or token_id >= BASE_VOCAB_SIZE for token_id in ids):
        raise ValueError(
            "Ordinary text produced an ID in the reserved control-token range "
            "[8184, 8191]; tokenizer/control-token layout is incompatible."
        )
    return ids


def encode_chat(tokenizer, system, user, assistant, max_len=512):
    """Return causal-LM inputs and labels; only assistant targets receive loss."""
    if max_len < 8:
        raise ValueError("max_len is too small for the chat control structure.")
    metadata = validate_tokenizer(tokenizer)
    system_ids = _encode_text(tokenizer, system)
    user_ids = _encode_text(tokenizer, user)
    assistant_ids = _encode_text(tokenizer, assistant)

    eos_id = metadata["eos_id"]
    sequence = (
        [ROLE_IDS["<|bos|>"], ROLE_IDS["<|system|>"]]
        + system_ids
        + [ROLE_IDS["<|end_system|>"], ROLE_IDS["<|user|>"]]
        + user_ids
        + [ROLE_IDS["<|end_user|>"], ROLE_IDS["<|assistant|>"]]
        + assistant_ids
        + [ROLE_IDS["<|end_assistant|>"], eos_id]
    )
    if len(sequence) > max_len:
        raise ValueError(
            f"Formatted conversation needs {len(sequence)} tokens, max_len={max_len}; "
            "truncate/filter it explicitly before batching."
        )

    supervised = (
        [False] * (2 + len(system_ids) + 1 + 1 + len(user_ids) + 1 + 1)
        + [True] * (len(assistant_ids) + 2)
    )
    # Next-token prediction: input[t] predicts sequence[t+1].
    input_ids = sequence[:-1]
    labels = [
        sequence[index + 1] if supervised[index + 1] else -100
        for index in range(len(sequence) - 1)
    ]
    if not any(label != -100 for label in labels):
        raise ValueError("Assistant completion must contain at least one token.")
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def format_v4_row(row, tokenizer, system_prompt, max_len=512):
    """Convert one verified Spanish V4 prompt/completion row into a chat sample."""
    if row.get("split") == "holdout" or row.get("source") == "holdout_500.jsonl":
        raise ValueError("Holdout row cannot be used for training.")
    if row.get("language") != "es":
        raise ValueError("V2.2 chat conversion requires a verified Spanish row.")
    if "prompt" not in row or "completion" not in row:
        raise ValueError("V4 row must contain prompt and completion.")
    return encode_chat(
        tokenizer, system_prompt, row["prompt"], row["completion"], max_len=max_len
    )


def collate_chat(samples, pad_id=PAD_ID):
    """Right-pad inputs; padded and non-assistant labels are always -100."""
    if not samples:
        raise ValueError("Cannot collate an empty chat batch.")
    width = max(int(inputs.numel()) for inputs, _ in samples)
    if width <= 0:
        raise ValueError("Chat batch contains an empty input.")
    inputs = torch.full((len(samples), width), pad_id, dtype=torch.long)
    labels = torch.full((len(samples), width), -100, dtype=torch.long)
    for index, (sample_inputs, sample_labels) in enumerate(samples):
        if sample_inputs.ndim != 1 or sample_labels.shape != sample_inputs.shape:
            raise ValueError("Each sample must contain matching 1D inputs and labels.")
        length = sample_inputs.numel()
        inputs[index, :length] = sample_inputs
        labels[index, :length] = sample_labels
    if not torch.any(labels.ne(-100)):
        raise ValueError("Batch contains no supervised assistant tokens.")
    return inputs, labels

def sample_mixed_rows(corpus, conflict_rows, batch_size, rng, conflict_ratio=0.12):
    """Sample V4 rows plus a controlled share of separate synthetic conflict rows."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not 0.0 <= conflict_ratio <= 1.0:
        raise ValueError("conflict_ratio must be between 0 and 1.")
    if len(corpus) <= 0:
        raise ValueError("V4 corpus is empty.")
    if conflict_ratio > 0 and not conflict_rows:
        raise ValueError("conflict_ratio is positive but no conflict rows were supplied.")
    n_conflict = int(round(batch_size * conflict_ratio))
    if conflict_ratio > 0 and conflict_rows:
        n_conflict = max(1, n_conflict)
    n_conflict = min(batch_size, n_conflict)
    rows = [corpus[rng.randrange(len(corpus))] for _ in range(batch_size - n_conflict)]
    rows.extend(rng.choice(conflict_rows) for _ in range(n_conflict))
    rng.shuffle(rows)
    return rows

