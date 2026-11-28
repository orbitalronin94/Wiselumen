#!/usr/bin/env python3
"""Generate a provenance-preserving Spanish draft corpus with Barcenas-LFM3."""
from __future__ import annotations
import concurrent.futures, hashlib, json, os, re, time, urllib.request
from pathlib import Path
BASE_URL=os.getenv("BASE_URL","http://127.0.0.1:8080/v1/chat/completions")
SEED=int(os.getenv("SEED","20261128")); EXAMPLES=int(os.getenv("EXAMPLES","128"))
MAX_TOKENS=int(os.getenv("MAX_TOKENS","256")); WORKERS=max(1,min(4,int(os.getenv("WORKERS","2"))))
TEMPERATURE=float(os.getenv("TEMPERATURE","0.35")); OUT=Path(os.getenv("OUT_DIR","artifacts/barcenas-corpus"))
CORPUS_SCHEMA="barcenas-spanish-draft-v1"
TOPICS=[
"la fotosíntesis","la respiración celular","la evolución por selección natural","la tectónica de placas","el ciclo del agua","la gravedad","la termodinámica","la electricidad","la genética mendeliana","el sistema inmunitario",
"la historia de al-Ándalus","la romanización de Hispania","la imprenta","la Revolución Industrial","la Constitución española","las Cortes Generales","la división de poderes","el derecho administrativo","la Unión Europea","la transición española",
"la oferta y la demanda","la inflación","el interés compuesto","la deuda pública","los impuestos progresivos","la productividad","la teoría de juegos","los bienes públicos","la contabilidad nacional","la economía cooperativa",
"las ecuaciones cuadráticas","la probabilidad condicionada","la inferencia estadística","los números primos","el cálculo diferencial","la geometría euclidiana","la lógica proposicional","la demostración por inducción","la teoría de grafos","la paradoja de Simpson",
"los algoritmos de ordenación","las tablas hash","la complejidad computacional","los sistemas operativos","las redes TCP/IP","el DNS","las bases de datos relacionales","el control de versiones","la criptografía de clave pública","el aprendizaje automático",
"la evaluación de modelos de lenguaje","la tokenización subword","la cuantización neuronal","la regularización","el sobreajuste","la reproducibilidad científica","el diseño experimental","la causalidad","la documentación técnica","la depuración de programas",
"la agricultura de secano","la gestión del agua","la energía solar","la energía eólica","la conservación del suelo","la biodiversidad","la pesca sostenible","la logística ferroviaria","la seguridad alimentaria","la salud pública"]
TASKS=[
"Explica {topic} con rigor y lenguaje claro. Incluye un ejemplo concreto y señala un límite o matiz importante.",
"Formula una pregunta de examen sobre {topic}, respóndela razonadamente y termina con una comprobación breve.",
"Compara {topic} con un concepto próximo. Expón tres diferencias y evita simplificaciones engañosas.",
"Presenta un caso práctico de {topic}, resuélvelo paso a paso y explicita los supuestos.",
"Corrige esta confusión habitual sobre {topic}: define los conceptos, explica el error y ofrece un contraejemplo.",
"Escribe una explicación breve de {topic} para un lector culto no especialista. Distingue hechos, hipótesis y opiniones.",
"Propón un pequeño ejercicio sobre {topic} con solución razonada. No inventes datos ni citas.",
"Expón los conceptos fundamentales de {topic}, su relación y una aplicación realista."]
def ask(prompt,idx):
    body={"model":"barcenas-lfm3-1.7b-q4_k_m","messages":[
      {"role":"system","content":"Eres un tutor y redactor didáctico en español. Responde en español peninsular natural, con precisión, sin inventar fuentes, cifras ni citas. Si una premisa es falsa, corrígela. Produce una respuesta autónoma, útil y bien estructurada; no repitas la pregunta."},
      {"role":"user","content":prompt}],
      "temperature":TEMPERATURE,"top_p":0.9,"max_tokens":MAX_TOKENS,"seed":SEED+idx,"stream":False}
    req=urllib.request.Request(BASE_URL,data=json.dumps(body,ensure_ascii=False).encode(),headers={"Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(req,timeout=900) as r: return json.loads(r.read().decode())
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    raw_path=OUT/"barcenas-draft.jsonl"; accepted_path=OUT/"barcenas-filtered.jsonl"; reject_path=OUT/"barcenas-rejected.jsonl"
    jobs=[(i+1,TASKS[(i//len(TOPICS))%len(TASKS)].format(topic=TOPICS[i%len(TOPICS)]),TOPICS[i%len(TOPICS)]) for i in range(EXAMPLES)]
    rows=[None]*EXAMPLES; start=time.perf_counter(); done=0
    def work(item):
        idx,prompt,topic=item; t0=time.perf_counter(); data=ask(prompt,idx); choice=(data.get("choices") or [{}])[0]
        answer=(choice.get("message",{}).get("content") or "").strip(); usage=data.get("usage") or {}
        return {"id":f"BAR-{idx:06d}","topic":topic,"prompt":prompt,"completion":answer,"teacher":"Danielbrdz/Barcenas-LFM3-1.7B",
          "quantization":"Q4_K_M","seed":SEED+idx,"temperature":TEMPERATURE,"max_tokens":MAX_TOKENS,
          "completion_tokens":usage.get("completion_tokens"),"finish_reason":choice.get("finish_reason"),
          "latency_seconds":round(time.perf_counter()-t0,3),
          "provenance":{"schema":CORPUS_SCHEMA,"status":"teacher_generated_unverified","generator":"scripts/generate_barcenas_corpus.py"}}
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures={pool.submit(work,item):item[0] for item in jobs}
        for fut in concurrent.futures.as_completed(futures):
            idx=futures[fut]
            try: rows[idx-1]=fut.result()
            except Exception as e: rows[idx-1]={"id":f"BAR-{idx:06d}","error":repr(e),"provenance":{"status":"generation_error"}}
            done+=1
            if done%10==0 or done==EXAMPLES: print(f"CORPUS_PROGRESS {done}/{EXAMPLES}",flush=True)
    seen=set(); accepted=[]; rejected=[]
    for row in rows:
        reason=None
        if "error" in row: reason="generation_error"
        else:
            text=row["completion"]; norm=re.sub(r"\s+"," ",text).strip().lower(); digest=hashlib.sha256(norm.encode()).hexdigest()
            if len(text)<100: reason="too_short"
            elif row.get("finish_reason")=="length": reason="truncated"
            elif digest in seen: reason="duplicate"
            elif len(norm.split())<18: reason="too_few_words"
            if reason is None: seen.add(digest)
        if reason: row["provenance"]["rejection_reason"]=reason; rejected.append(row)
        else: row["provenance"]["status"]="teacher_generated_heuristically_filtered"; accepted.append(row)
    for path,data in [(raw_path,rows),(accepted_path,accepted),(reject_path,rejected)]:
        path.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in data),encoding="utf-8")
    elapsed=time.perf_counter()-start
    summary={"schema":CORPUS_SCHEMA,"teacher":"Danielbrdz/Barcenas-LFM3-1.7B","quantization":"Q4_K_M","requested":EXAMPLES,
      "generated":sum("error" not in x for x in rows),"accepted_heuristically":len(accepted),"rejected":len(rejected),
      "rejection_reasons":{k:sum(1 for x in rejected if x.get("provenance",{}).get("rejection_reason")==k) for k in sorted({x.get("provenance",{}).get("rejection_reason","unknown") for x in rejected})},
      "workers":WORKERS,"max_tokens":MAX_TOKENS,"temperature":TEMPERATURE,"elapsed_seconds":round(elapsed,2),
      "examples_per_minute":round(EXAMPLES/(elapsed/60),2) if elapsed else 0,
      "important":"Heuristic filtering is not factual verification. Accepted rows remain draft data and require human/independent validation before training."}
    (OUT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("BARCENAS_CORPUS_RESULT "+json.dumps(summary,ensure_ascii=False),flush=True)
    if summary["generated"]<EXAMPLES: raise SystemExit("Some teacher requests failed; inspect rejected JSONL.")
if __name__=="__main__": main()
