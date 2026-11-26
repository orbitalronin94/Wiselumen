#!/usr/bin/env python3
"""Index every numbered revelation in the Palimpsest without duplicating its text."""
import json,re
from pathlib import Path
SOURCE=Path("Marper-Codice-Juntado (1).txt")
OUT=Path("data")
PATTERN=re.compile(r"^### Revelación del Códice de Marper\s*#\s*(\d+)\s*$",re.M)
SHA="b0eb703c5237d1d461321a30a89525e154f0219d"

def main():
    text=SOURCE.read_text(encoding="utf-8")
    matches=list(PATTERN.finditer(text))
    if len(matches)!=16848:
        raise SystemExit(f"Expected 16848 revelations, found {len(matches)}")
    rows=[]; line_starts=[]; cursor=0; line=1
    for i,m in enumerate(matches):
        while cursor<m.start():
            if text[cursor]=="\n": line+=1
            cursor+=1
        line_starts.append(line)
        end=matches[i+1].start() if i+1<len(matches) else len(text)
        block=text[m.end():end]
        paras=[p.strip() for p in re.split(r"\n\s*\n",block) if p.strip()]
        topic_para=next((p for p in paras if ":" in p and len(p)>80
                         and not p.startswith("El monje escribió")
                         and not re.match(r"^(En la celda|En el scriptorium|En la consola|En el convento|En la abadía)",p,re.I)),None)
        topic=topic_para.split(":",1)[0].strip() if topic_para else "Sin tema extraído"
        number=int(m.group(1))
        if number!=i+1: raise SystemExit(f"Nonsequential revelation number at {i+1}: {number}")
        rows.append({"id":f"PM{number:05d}","type":"source_record","source_id":"source:palimpsesto-de-marper",
          "source_blob_sha":SHA,"source_revelation_number":number,
          "title":f"Revelación del Códice de Marper #{number:05d}",
          "source_line_start":line,"source_line_end":0,"topic":topic,
          "topic_group":"mythology" if re.match(r"^(JAPONESA|NÓRDICA|EGIPCIA|GRIEGA)",topic,re.I) else "psychonautic_historical_culture",
          "record_class":"palimpsest_revelation","truth_scope":"mixed_claims_unreviewed"})
    for i,row in enumerate(rows):
        row["source_line_end"]=(line_starts[i+1]-1) if i+1<len(rows) else len(text.splitlines())
    OUT.mkdir(exist_ok=True)
    for part in range(4):
        chunk=rows[part*4212:(part+1)*4212]
        path=OUT/f"palimpsesto_index_v0.1_part{part+1}.jsonl"
        path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in chunk),encoding="utf-8")
    print(f"indexed={len(rows)} unique_ids={len({x['id'] for x in rows})} topics={len({x['topic'] for x in rows})}")

if __name__=="__main__": main()
