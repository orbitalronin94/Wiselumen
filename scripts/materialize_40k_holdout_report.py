#!/usr/bin/env python3
import json
from pathlib import Path

root = Path("artifact")
files = list(root.rglob("freeform_100_v22_step40000.json"))
if len(files) != 1:
    raise SystemExit(f"Expected exactly one freeform JSON, found {len(files)}")
free_path = files[0]
v5_path = free_path.with_name("v5_generative_holdout_20_step40000.json")
meta_path = free_path.with_name("generative_metadata.json")
free = json.loads(free_path.read_text(encoding="utf-8"))
v5 = json.loads(v5_path.read_text(encoding="utf-8"))
meta = json.loads(meta_path.read_text(encoding="utf-8"))
assert len(free["results"]) == 100, len(free["results"])
assert len(v5["results"]) == 20, len(v5["results"])

def clean(s):
    return str(s or "").strip().replace("\r\n", "\n")

out = [
    "# LUMEN V2.2 — Holdout generativo del checkpoint 40.000",
    "",
    "## Ficha de ejecución",
    "",
    "- Checkpoint: **40.000 pasos**",
    "- Muestra: **100 prompts generales + 20 probes V5 = 120 generaciones**",
    "- Inferencia: Rust + tch/LibTorch CPU, TorchScript y KV cache nativa",
    f"- Hilos: {meta.get('threads', 'no indicado')}",
    "- Decodificación general: greedy argmax, contexto 512, máximo 64 tokens nuevos",
    "- Decodificación V5: greedy argmax, contexto 512, máximo 96 tokens nuevos",
    "- Bloque MC: **no ejecutado** (holdout exclusivamente generativo)",
    "- Ejecución: completada; los 120 prompts se procesaron",
    "",
    "## 1. Muestra general — 100 preguntas y respuestas",
    "",
]
for i, item in enumerate(free["results"], 1):
    out += [
        f"### {i:03d}. {clean(item.get('id'))} — {clean(item.get('source'))}",
        "",
        "**Pregunta**",
        "",
        clean(item.get("prompt")),
        "",
        "**Respuesta del modelo**",
        "",
        "~~~text",
        clean(item.get("response")),
        "~~~",
        "",
    ]
out += ["## 2. Probes V5 — 20 preguntas y respuestas", ""]
for i, item in enumerate(v5["results"], 1):
    out += [
        f"### {i:02d}. {clean(item.get('id'))} — {clean(item.get('category'))}",
        "",
        "**Pregunta**",
        "",
        clean(item.get("prompt")),
        "",
        "**Respuesta del modelo**",
        "",
        "~~~text",
        clean(item.get("response")),
        "~~~",
        "",
    ]
wall = (free_path.parent / "wall_seconds.txt").read_text().strip() if (free_path.parent / "wall_seconds.txt").exists() else "no incluido"
out += [
    "## 3. Huellas de reproducibilidad",
    "",
    f"- Checkpoint SHA-256: {meta.get('checkpoint_sha256')}",
    f"- Tokenizer SHA-256: {meta.get('tokenizer_sha256')}",
    f"- System prompt SHA-256: {meta.get('system_prompt_sha256')}",
    f"- Prompts generales SHA-256: {meta.get('freeform_prompt_sha256')}",
    f"- Prompts V5 SHA-256: {meta.get('v5_prompt_sha256')}",
    "",
    "## 4. Alcance y cautela",
    "",
    "Este documento reproduce literalmente las preguntas y respuestas guardadas por el evaluador. Es una inspección generativa; no equivale a una puntuación automática de corrección factual. Las respuestas deben juzgarse por coherencia, seguimiento de instrucciones, consistencia del lore y calidad lingüística.",
    "",
    f"- Tiempo del proceso de inferencia (segundos): {wall}",
    "- Workflow: https://github.com/orbitalronin94/Wiselumen/actions/runs/45994377615",
    "- Artefacto: v22-rust-generative-holdout-step40000",
    "",
]
Path("HOLDOUT_LUMEN_V22_40K.md").write_text("\n".join(out), encoding="utf-8")
print("Created HOLDOUT_LUMEN_V22_40K.md with 100 + 20 prompt/response pairs")
