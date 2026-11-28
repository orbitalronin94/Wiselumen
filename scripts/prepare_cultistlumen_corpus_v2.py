import os, re, json, gzip, hashlib, sqlite3, unicodedata
from pathlib import Path
from collections import Counter
from datasets import load_dataset

OUT=Path("corpus_v2_candidate"); OUT.mkdir(exist_ok=True)
MAX_BYTES=int(float(os.getenv("MAX_OUTPUT_GB","7.5"))*1024**3)
MIN_CHARS=int(os.getenv("MIN_BDH_CHARS","800"))
DIGITIZED=re.compile(r"digitized\s+by\s+google|internet\s+archive|google\s+book\s+search",re.I)
PAGE=re.compile(r"^\s*(?:\d{1,5}|[ivxlcdm]{1,12})\s*$",re.I)
NOISE=re.compile(r"[|~`^_=<>]{3,}")
WORD=re.compile(r"\b[\wáéíóúüñÁÉÍÓÚÜÑ]+\b",re.I)

def clean(text):
    if not isinstance(text,str): return None,"not_text",{}
    text=unicodedata.normalize("NFC",text).replace("\r\n","\n").replace("\r","\n")
    lines=[re.sub(r"[ \t]+"," ",x).strip() for x in text.splitlines()]
    lines=[x for x in lines if x and not PAGE.match(x)]
    freq=Counter(x.casefold() for x in lines if len(x)<=140)
    lines=[x for x in lines if not(len(x)<=140 and freq[x.casefold()]>=4)]
    text="\n\n".join(lines)
    text=DIGITIZED.sub(" ",text)
    text=re.sub(r"\n{3,}","\n\n",text).strip()
    chars=len(text); words=WORD.findall(text)
    alpha=sum(c.isalpha() for c in text)/max(chars,1)
    if chars<MIN_CHARS:return None,"too_short",{"chars":chars}
    if alpha<0.62:return None,"low_alpha_ratio",{"chars":chars,"alpha_ratio":alpha}
    if len(words)<120:return None,"too_few_words",{"chars":chars,"words":len(words)}
    if len(NOISE.findall(text))>max(8,chars//1200):return None,"ocr_noise",{"chars":chars}
    return text,None,{"chars":chars,"words":len(words),"alpha_ratio":round(alpha,4)}

def writer(prefix):
    idx=0; raw=0; shard_raw=0; f=None; path=None; shards=[]
    def open_shard():
        nonlocal idx,raw,shard_raw,f,path
        idx+=1; path=OUT/f"{prefix}-{idx:05d}.jsonl.gz"
        f=gzip.open(path,"wt",encoding="utf-8",compresslevel=6); shard_raw=0
    open_shard()
    def put(obj):
        nonlocal raw,shard_raw,f,path
        line=json.dumps(obj,ensure_ascii=False,separators=(",",":"))+"\n"
        f.write(line); n=len(line.encode()); raw+=n; shard_raw+=n
        if shard_raw>=256*1024**2:
            f.close(); shards.append({"file":path.name,"compressed_bytes":path.stat().st_size,"raw_bytes":shard_raw}); open_shard()
    def close():
        nonlocal f
        if f:
            f.close()
            if path.exists() and path.stat().st_size: shards.append({"file":path.name,"compressed_bytes":path.stat().st_size,"raw_bytes":shard_raw})
            else: path.unlink(missing_ok=True)
            f=None
        return shards,raw
    return put,close

manifest={"sources":{},"warnings":["BDH OCR needs manual spot checks; filters are heuristic.","MentorES Hub metadata says CC BY-SA 4.0 while card prose says CC BY 4.0; license requires maintainer confirmation.","MentorES is stored separately as structured instruction data, not flattened into pretraining text."],"v1_modified":False}
put,close=writer("bdh_books_clean"); stats=Counter(); seen=sqlite3.connect(OUT/"bdh_seen.sqlite"); seen.execute("CREATE TABLE IF NOT EXISTS hashes(h TEXT PRIMARY KEY)")
ds=load_dataset("PleIAs/BDH-Books",split="train",streaming=True)
for i,row in enumerate(ds,1):
    raw=row.get("complete_text") or row.get("texto_OCR") or ""
    text,reason,q=clean(raw); stats["seen"]+=1
    if not text: stats["rejected_"+reason]+=1; continue
    h=hashlib.sha256(re.sub(r"\s+"," ",text.lower()).strip().encode()).hexdigest()
    try: seen.execute("INSERT INTO hashes VALUES (?)",(h,))
    except sqlite3.IntegrityError: stats["duplicate_exact"]+=1; continue
    keys=["file_id","idBNE","Autor Personas","Autor Entidades","Título","Editorial","Lugar de publicación","Fecha de publicación","Lengua de publicación","Lengua original","Tipo de documento","Tema","Género/Forma","Condiciones de uso","Procedencia"]
    meta={k:row.get(k) for k in keys if row.get(k) not in (None,"")}
    put({"source":"PleIAs/BDH-Books","license":"public_domain_as_declared_by_dataset_card","record_id":str(row.get("file_id") or row.get("idBNE") or i),"metadata":meta,"text":text,"quality":q,"text_sha256":h})
    stats["accepted"]+=1
    if i%1000==0: seen.commit(); print("BDH",i,dict(stats),flush=True)
    if i%100==0 and (OUT/"bdh_books_clean-00001.jsonl.gz").exists():
        size=sum(p.stat().st_size for p in OUT.glob("bdh_books_clean-*.jsonl.gz"))
        if size>MAX_BYTES: stats["stopped_output_cap"]+=1; break
seen.commit(); seen.close(); shards,raw=close()
manifest["sources"]["BDH-Books"]={"dataset":"PleIAs/BDH-Books","stats":dict(stats),"shards":shards,"raw_jsonl_bytes":raw}

put,close=writer("mentores"); ms=Counter()
md=load_dataset("projecte-aina/MentorES",streaming=True)
splits=md.items() if hasattr(md,"items") else [("train",md)]
for split,rows in splits:
    for row in rows:
        instruction=str(row.get("instruction") or "").strip()
        context=str(row.get("context") or "").strip()
        response=str(row.get("response") or row.get("answer") or "").strip()
        category=str(row.get("category") or "unknown").strip()
        if len(instruction)<8 or len(response)<3: ms["rejected_incomplete"]+=1; continue
        put({"source":"projecte-aina/MentorES","license":"license_requires_confirmation","split":split,"category":category,"instruction":instruction,"context":context,"response":response})
        ms["accepted"]+=1
shards,raw=close()
manifest["sources"]["MentorES"]={"dataset":"projecte-aina/MentorES","stats":dict(ms),"shards":shards,"raw_jsonl_bytes":raw}
(OUT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(manifest,ensure_ascii=False,indent=2))
