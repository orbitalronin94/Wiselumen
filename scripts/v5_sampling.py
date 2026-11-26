"""Exact-share sampler and formatter for LUMEN V2.2 Corpus V5.

For batch_size=4, every 10 optimizer steps contain exactly 27 V4, 3 lore,
and 10 synthetic-conflict examples: 67.5% / 7.5% / 25%.
"""
from __future__ import annotations
import random

CYCLE_STEPS = 10
BATCH_SIZE = 4
COUNTS = (("v4", 27), ("lore", 3), ("conflict", 10))

def sample_v5_batches(v4_rows, lore_rows, conflict_rows, steps, rng=None, batch_size=BATCH_SIZE):
    """Return batches of (source_kind, row), preserving the source label."""
    if batch_size != BATCH_SIZE:
        raise ValueError("V5 exact sampler currently requires batch_size=4")
    if steps <= 0 or steps % CYCLE_STEPS:
        raise ValueError("steps must be a positive multiple of 10 for exact 67.5/7.5/25 shares")
    if not v4_rows or not lore_rows or not conflict_rows:
        raise ValueError("V4, lore and conflict pools must all be non-empty")
    rng = rng or random.Random(20261126)
    pools = {"v4": v4_rows, "lore": lore_rows, "conflict": conflict_rows}
    batches = []
    for _ in range(steps // CYCLE_STEPS):
        tags = [kind for kind, count in COUNTS for _ in range(count)]
        rng.shuffle(tags)
        for offset in range(0, len(tags), batch_size):
            chosen = [(kind, rng.choice(pools[kind])) for kind in tags[offset:offset + batch_size]]
            rng.shuffle(chosen)
            batches.append(chosen)
    return batches

def format_lore_row(row, tokenizer, system_prompt, max_len=512):
    """Format one curated fact/lore/bridge row as a supervised chat sample."""
    from data import encode_chat
    kind = row.get("type")
    title = str(row.get("title", "")).strip()
    body = str(row.get("text", "")).strip()
    if kind not in {"fact", "lore", "bridge"} or not title or not body:
        raise ValueError(f"Malformed curated lore record: {row.get('id', '?')}")
    if kind == "fact":
        prompt = (
            f"Explica el contenido factual titulado «{title}». "
            "Respeta su alcance y señala las incertidumbres indicadas."
        )
    elif kind == "lore":
        prompt = (
            f"Presenta el documento ficticio «{title}» de La Concordia de las Nueve Cenizas. "
            "Conserva su voz, atribución y condición intradiegética."
        )
    else:
        prompt = (
            f"Explica el puente epistemológico «{title}»: separa el conocimiento externo "
            "de la interpretación del mundo ficticio."
        )
    return encode_chat(tokenizer, system_prompt, prompt, body, max_len=max_len)
