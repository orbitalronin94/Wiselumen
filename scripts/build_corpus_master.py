#!/usr/bin/env python3
"""Compile the complete Marper source, the Concordia seed and curated additions."""
from __future__ import annotations
import argparse, hashlib, json, re
from collections import Counter
from pathlib import Path

EXPECTED_BLOB_SHA = "b0eb703c5237d1d461321a30a89525e154f0219d"
HEADER = re.compile(r"^### Revelación del Códice de Marper\\s*#\\s*(\\d+)\\s*$", re.MULTILINE)

def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\\0" + data).hexdigest()

def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="Marper-Codice-Juntado (1).txt")
    ap.add_argument("--base", default="data/lore_seed_v0.3.jsonl")
    ap.add_argument("--additions", default="data/corpus_master_additions_v1.0.jsonl")
    ap.add_argument("--out-dir", default="build")
    args = ap.parse_args()
    source_path, base_path, additions_path = map(Path, (args.source, args.base, args.additions))
    raw = source_path.read_bytes()
    actual_blob = git_blob_sha(raw)
    if actual_blob != EXPECTED_BLOB_SHA:
        raise SystemExit(f"Source blob SHA mismatch: expected {EXPECTED_BLOB_SHA}, got {actual_blob}")
    source = raw.decode("utf-8")
    matches = list(HEADER.finditer(source))
    numbers = [int(m.group(1)) for m in matches]
    if len(matches) != 16848 or numbers != list(range(1, 16849)):
        raise SystemExit(f"Expected 16,848 sequential revelations; found {len(matches)}")
    records = read_jsonl(base_path) + read_jsonl(additions_path)
    preface_end = matches[0].start()
    records.append({
        "id":"PM00000","type":"lore","domain":"Palimpsesto de Marper",
        "title":"Prefacio y declaración inicial del Códice",
        "text":source[:preface_end],"in_world_date":None,
        "author":"Atribución no autenticada del archivo","source_form":"prefacio_archivistico",
        "characters":["Marper"],"places":[],"factual_claims":[],
        "uncertainty":["Las afirmaciones del prefacio son parte de la fuente y no hechos externos verificados."],
        "links":[],"canon_status":"provisional","source_id":"S001",
        "source_blob_sha":actual_blob,"source_record_number":0,
        "record_class":"palimpsest_preface",
        "truth_scope":"fictional_claims_and_unverified_assertions",
        "adaptation_status":"verbatim_transcription","review_status":"not_fact_checked"
    })
    topics = Counter()
    seen_text_hashes = set()
    duplicate_texts = 0
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i+1].start() if i+1 < len(matches) else len(source)
        block = source[start:end]
        paragraphs = [p.strip() for p in re.split(r"\\n\\s*\\n", block.strip()) if p.strip()]
        topic_paragraph = paragraphs[2] if len(paragraphs) > 2 else (paragraphs[1] if len(paragraphs) > 1 else "")
        topic_match = re.match(r"^([^:\\n]{1,140}):", topic_paragraph)
        topic = topic_match.group(1).strip() if topic_match else "Tema sin clasificar"
        topics[topic] += 1
        digest = hashlib.sha256(block.encode("utf-8")).hexdigest()
        if digest in seen_text_hashes:
            duplicate_texts += 1
        seen_text_hashes.add(digest)
        n = int(match.group(1))
        records.append({
            "id":f"PM{n:05d}","type":"lore",
            "domain":f"Palimpsesto: {topic}",
            "title":f"Revelación {n:05d} — {topic}",
            "text":block,"in_world_date":None,
            "author":"Monje glosador (atribución interna no autenticada)",
            "source_form":"revelacion_del_codice","characters":["Marper"],
            "places":[],"factual_claims":[],
            "uncertainty":["Las afirmaciones históricas, científicas y culturales del fragmento no han sido verificadas individualmente; la interpretación de Marper es ficción interna."],
            "links":[],"canon_status":"provisional","source_id":"S001",
            "source_blob_sha":actual_blob,"source_record_number":n,
            "record_class":"palimpsest_revelation",
            "truth_scope":"mixed_unverified_source_claims_and_fictional_marper_interpretation",
            "adaptation_status":"verbatim_transcription",
            "source_topic":topic,"review_status":"not_fact_checked"
        })
    ids = [r.get("id") for r in records]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate record IDs in master corpus")
    known = set(ids)
    unresolved = [(r["id"], target) for r in records for target in r.get("links", []) if target not in known]
    if unresolved:
        raise SystemExit(f"Unresolved links: {unresolved[:20]} (total {len(unresolved)})")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "corpus_master_v1.0.jsonl"
    counts = Counter(r["type"] for r in records)
    with output.open("w", encoding="utf-8", newline="\\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",",":")) + "\\n")
    output_bytes = output.stat().st_size
    output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "schema_version":"1.0.0","dataset":"Corpus Maestro",
        "canonical_names":{"original_tradition":"El Palimpsesto de Marper","fictional_universe":"La Concordia de las Nueve Cenizas","integrated_dataset":"Corpus Maestro"},
        "source":{"path":str(source_path),"git_blob_sha1":actual_blob,"sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw),"characters":len(source),"revelations":len(matches),"preface_included":True,"epilogue_included_in_final_revelation":True},
        "output":{"path":str(output),"sha256":output_sha256,"bytes":output_bytes,"decimal_MB":round(output_bytes/1_000_000,3),"MiB":round(output_bytes/1048576,3),"characters":sum(len(json.dumps(r,ensure_ascii=False,separators=(",",":")))+1 for r in records),"records":len(records),"counts":dict(counts),"palimpsest_revelations":len(matches),"palimpsest_preface":1,"topic_count":len(topics),"duplicate_verbatim_revelation_blocks":duplicate_texts,"unresolved_links":0},
        "additions":{"factual_records":sum(r.get("id","").startswith("F") for r in records)-23,"adaptation_dossiers":sum(r.get("record_class")=="palimpsest_adaptation" for r in records),"critical_bridges":sum(r.get("record_class")=="critical_bridge" for r in records)},
        "topics":[{"topic":k,"revelations":v} for k,v in sorted(topics.items())],
        "epistemic_policy":"All 16,848 source revelations and the preface are transcribed verbatim. Source claims are explicitly unverified and kept separate from factual records. Fictional interpretations are never promoted to external fact.",
        "status":"complete_for_declared_inputs; factual source audit remains separate"
    }
    manifest_path = out_dir / "corpus_master_manifest_v1.0.json"
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\\n",encoding="utf-8")
    print(json.dumps({"records":len(records),"counts":dict(counts),"source_bytes":len(raw),"master_bytes":output_bytes,"master_MB":round(output_bytes/1_000_000,3),"master_MiB":round(output_bytes/1048576,3),"topics":len(topics),"duplicate_source_blocks":duplicate_texts,"unresolved_links":len(unresolved),"sha256":output_sha256},ensure_ascii=False))

if __name__ == "__main__":
    main()
