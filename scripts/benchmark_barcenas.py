#!/usr/bin/env python3
"""Deterministic 200-prompt Spanish inference benchmark for Barcenas-LFM3-1.7B."""
from __future__ import annotations
import json, math, statistics, time, urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:8080/v1/chat/completions"
SEED = 20261128
TOPICS = [
"la diferencia entre energía y potencia", "la fotosíntesis y sus fases", "la estructura de una célula eucariota", "la selección natural y la evolución", "el ciclo del agua y el clima", "la gravedad y las órbitas", "la electricidad y la ley de Ohm", "la termodinámica y la entropía", "la diferencia entre masa y peso", "la tectónica de placas", "la historia de la imprenta", "la transición española", "la Revolución Industrial", "la organización territorial de España", "el funcionamiento de las Cortes Generales", "la diferencia entre ley y reglamento", "la inflación y los tipos de interés", "la oferta y la demanda", "el interés compuesto", "la diferencia entre déficit y deuda pública", "las proporciones y los porcentajes", "las ecuaciones de segundo grado", "la derivada y su interpretación geométrica", "la probabilidad condicionada", "la media, la mediana y la moda", "los números primos", "la complejidad algorítmica", "las tablas hash", "la recursividad en programación", "las diferencias entre TCP y UDP", "el funcionamiento de DNS", "la normalización de bases de datos", "el control de versiones con Git", "la diferencia entre proceso e hilo", "el aprendizaje supervisado", "el sobreajuste en modelos de aprendizaje automático", "la tokenización de texto", "la cuantización de modelos neuronales", "la diferencia entre correlación y causalidad", "la reproducibilidad de un experimento"
]
TEMPLATES = [
"Define con precisión {topic} y ofrece un ejemplo concreto.",
"Explica paso a paso {topic} para una persona sin conocimientos previos.",
"Compara {topic} con un concepto cercano y señala dos diferencias importantes.",
"Enumera tres ideas esenciales sobre {topic} y justifica brevemente cada una.",
"Propón un caso práctico sobre {topic}, resuélvelo y comprueba el resultado."
]

def request(prompt: str, max_tokens: int = 64) -> dict:
    payload = {"model":"barcenas-lfm3-1.7b-q4_k_m", "messages":[{"role":"user","content":prompt}], "temperature":0.1, "top_p":0.9, "max_tokens":max_tokens, "seed":SEED, "stream":False}
    req = urllib.request.Request(BASE_URL, data=json.dumps(payload,ensure_ascii=False).encode(), headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=600) as response:
        return json.loads(response.read().decode())

def percentile(values, p):
    values=sorted(values)
    if not values: return 0.0
    pos=(len(values)-1)*p; lo=math.floor(pos); hi=math.ceil(pos)
    return values[lo] if lo==hi else values[lo]*(hi-pos)+values[hi]*(pos-lo)

def main():
    out=Path("artifacts/barcenas-benchmark"); out.mkdir(parents=True,exist_ok=True)
    prompts=[{"id":f"ES-{i+1:03d}","category":TEMPLATES[j].split(" ")[0].lower(),"prompt":TEMPLATES[j].format(topic=topic)} for i,(topic,j) in enumerate((topic,j) for topic in TOPICS for j in range(len(TEMPLATES)))]
    assert len(prompts)==200
    print("BARCENAS_WARMUP_START",flush=True)
    for prompt in prompts[:2]: request(prompt["prompt"],max_tokens=8)
    results=[]; latencies=[]; generation_seconds=[]; token_total=0; prompt_total=0; start=time.perf_counter()
    for i,row in enumerate(prompts,1):
        t0=time.perf_counter(); data=request(row["prompt"]); wall=time.perf_counter()-t0
        choice=data.get("choices",[{}])[0]
        answer=choice.get("message",{}).get("content","") or ""
        usage=data.get("usage",{}) or {}; timings=data.get("timings",{}) or {}
        completion=int(usage.get("completion_tokens",timings.get("predicted_n",0)) or 0)
        prompt_tokens=int(usage.get("prompt_tokens",timings.get("prompt_n",0)) or 0)
        gen_s=float(timings.get("predicted_ms",0) or 0)/1000.0
        if gen_s<=0: gen_s=wall
        rate=completion/gen_s if gen_s>0 else 0.0
        row_result={**row,"response":answer,"completion_tokens":completion,"prompt_tokens":prompt_tokens,"wall_seconds":round(wall,4),"generation_seconds":round(gen_s,4),"tokens_per_second":round(rate,3),"finish_reason":choice.get("finish_reason")}
        results.append(row_result); latencies.append(wall); generation_seconds.append(gen_s); token_total+=completion; prompt_total+=prompt_tokens
        if i%10==0 or i==1 or i==len(prompts): print(f"BARCENAS {i}/200 latency={wall:.2f}s tokens={completion} tok_s={rate:.2f} chars={len(answer)}",flush=True)
    elapsed=time.perf_counter()-start
    total_gen=sum(generation_seconds)
    summary={"model":"Danielbrdz/Barcenas-LFM3-1.7B","gguf_repo":"bartowski/Danielbrdz_Barcenas-LFM3-1.7B-GGUF","quantization":"Q4_K_M","prompts":len(results),"seed":SEED,"temperature":0.1,"max_tokens":64,"completion_tokens":token_total,"prompt_tokens":prompt_total,"wall_seconds":round(elapsed,3),"requests_per_minute":round(len(results)/(elapsed/60),3),"aggregate_generation_tokens_per_second":round(token_total/total_gen,3) if total_gen else 0,"mean_request_latency_seconds":round(statistics.mean(latencies),3),"p50_request_latency_seconds":round(percentile(latencies,.5),3),"p95_request_latency_seconds":round(percentile(latencies,.95),3),"mean_response_tokens_per_second":round(statistics.mean([r["tokens_per_second"] for r in results]),3)}
    (out/"barcenas-200-results.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in results),encoding="utf-8")
    (out/"barcenas-200-summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# Bárcenas-LFM3-1.7B — benchmark español (200 prompts)","",f"- Cuantización: Q4_K_M",f"- Prompts: {len(results)}",f"- Tokens generados: {token_total}",f"- Throughput agregado: {summary['aggregate_generation_tokens_per_second']} tokens/s",f"- Latencia media: {summary['mean_request_latency_seconds']} s",f"- P50 / P95: {summary['p50_request_latency_seconds']} / {summary['p95_request_latency_seconds']} s",f"- Duración: {summary['wall_seconds']} s", "", "Las respuestas literales están en barcenas-200-results.jsonl. El throughput es una medición de esta máquina y esta cuantización, no una cifra universal."]
    (out/"BARCENAS_BENCHMARK_200.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("BARCENAS_BENCHMARK_RESULT "+json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=="__main__": main()
