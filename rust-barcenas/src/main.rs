use anyhow::{anyhow, Context, Result};
use cera::{BackendPreference, GenerateOpts, LoadConfig, Message, ModelLoader, ModelSource, SessionConfig};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{collections::HashSet, env, fs::{self, File}, io::{BufWriter, Read, Write}, path::Path, time::Instant};

const SYSTEM: &str = "Eres un tutor y redactor didáctico riguroso en español peninsular. Tu prioridad es la exactitud, no aparentar seguridad. Responde directamente y de forma autónoma; define los términos necesarios, razona paso a paso cuando el problema lo requiera y aporta ejemplos concretos. Distingue hechos establecidos, inferencias, hipótesis y opiniones. No inventes fuentes, citas, fechas, cifras, leyes ni resultados. Si no sabes algo o faltan datos, dilo y delimita qué puede concluirse. Detecta y corrige premisas falsas sin adoptar el error del usuario. Evita relleno, fórmulas vacías, repeticiones y conclusiones que no se sigan de la explicación. Usa una estructura proporcionada a la dificultad y termina con una conclusión útil.";
const TOPICS: &[&str] = &[
"la fotosíntesis","la respiración celular","la evolución por selección natural","la tectónica de placas","el ciclo del agua","la gravedad","la termodinámica","la electricidad","la genética mendeliana","el sistema inmunitario",
"la historia de al-Ándalus","la romanización de Hispania","la imprenta","la Revolución Industrial","la Constitución española","las Cortes Generales","la división de poderes","el derecho administrativo","la Unión Europea","la transición española",
"la oferta y la demanda","la inflación","el interés compuesto","la deuda pública","los impuestos progresivos","la productividad","la teoría de juegos","los bienes públicos","la contabilidad nacional","la economía cooperativa",
"las ecuaciones cuadráticas","la probabilidad condicionada","la inferencia estadística","los números primos","el cálculo diferencial","la geometría euclidiana","la lógica proposicional","la demostración por inducción","la teoría de grafos","la paradoja de Simpson",
"los algoritmos de ordenación","las tablas hash","la complejidad computacional","los sistemas operativos","las redes TCP/IP","el DNS","las bases de datos relacionales","el control de versiones","la criptografía de clave pública","el aprendizaje automático",
"la evaluación de modelos de lenguaje","la tokenización subword","la cuantización neuronal","la regularización","el sobreajuste","la reproducibilidad científica","el diseño experimental","la causalidad","la documentación técnica","la depuración de programas",
"la agricultura de secano","la gestión del agua","la energía solar","la energía eólica","la conservación del suelo","la biodiversidad","la pesca sostenible","la logística ferroviaria","la seguridad alimentaria","la salud pública"
];
const TASKS: &[&str] = &[
"Explica {topic} con rigor y lenguaje claro. Incluye un ejemplo concreto y señala un límite o matiz importante.",
"Formula una pregunta de examen sobre {topic}, respóndela razonadamente y termina con una comprobación breve.",
"Compara {topic} con un concepto próximo. Expón tres diferencias y evita simplificaciones engañosas.",
"Presenta un caso práctico de {topic}, resuélvelo paso a paso y explicita los supuestos.",
"Corrige una confusión habitual sobre {topic}: define los conceptos, explica el error y ofrece un contraejemplo.",
"Escribe una explicación breve de {topic} para un lector culto no especialista. Distingue hechos, hipótesis y opiniones.",
"Propón un pequeño ejercicio sobre {topic} con solución razonada. No inventes datos ni citas.",
"Expón los conceptos fundamentales de {topic}, su relación y una aplicación realista."
];

#[derive(Serialize)]
struct Row {
    id: String, topic: String, prompt: String, completion: String,
    teacher: &'static str, system_prompt: &'static str, system_prompt_sha256: String, format: &'static str, quantization: &'static str,
    seed: u64, temperature: f32, completion_tokens: u32,
    latency_seconds: f64, status: &'static str,
}
fn arg(name: &str, default: &str) -> String {
    let a: Vec<String> = env::args().collect();
    a.windows(2).find(|w| w[0] == name).map(|w| w[1].clone()).unwrap_or_else(|| default.to_string())
}
fn sha256(path: &str) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("opening {path}"))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 8 * 1024 * 1024];
    loop {
        let n = file.read(&mut buffer).with_context(|| format!("hashing {path}"))?;
        if n == 0 { break; }
        hasher.update(&buffer[..n]);
    }
    let digest = hasher.finalize();
    Ok(digest.iter().map(|byte| format!("{byte:02x}")).collect())
}
fn main() -> Result<()> {
    let model_path = arg("--model", "models/barcenas/Barcenas-LFM3-1.7B-Q4_K_M.gguf");
    let out_dir = arg("--out-dir", "artifacts/barcenas-rust-corpus");
    let examples: usize = arg("--examples", "128").parse()?;
    let max_tokens: u32 = arg("--max-tokens", "256").parse()?;
    let seed: u64 = arg("--seed", "20261128").parse()?;
    anyhow::ensure!(examples > 0 && examples <= 2000, "examples must be 1..=2000");
    anyhow::ensure!(max_tokens >= 32 && max_tokens <= 512, "max-tokens must be 32..=512");
    anyhow::ensure!(Path::new(&model_path).is_file(), "GGUF model missing: {model_path}");

    fs::create_dir_all(&out_dir)?;
    println!("BARCENAS_RUST_LOAD_START model={} sha256={}", model_path, sha256(&model_path)?);
    let load_start = Instant::now();
    let model = ModelLoader::new(ModelSource::path(&model_path))
        .config(LoadConfig { backend: BackendPreference::Cpu, ..LoadConfig::default() })
        .build_generative()
        .context("loading LFM3 GGUF through native Rust runtime")?;
    println!("BARCENAS_RUST_MODEL_READY load_seconds={:.2}", load_start.elapsed().as_secs_f64());

    let raw_file = File::create(format!("{out_dir}/barcenas-rust-raw.jsonl"))?;
    let accepted_file = File::create(format!("{out_dir}/barcenas-rust-accepted.jsonl"))?;
    let rejected_file = File::create(format!("{out_dir}/barcenas-rust-rejected.jsonl"))?;
    let mut raw = BufWriter::new(raw_file);
    let mut accepted = BufWriter::new(accepted_file);
    let mut rejected = BufWriter::new(rejected_file);
    let mut seen = HashSet::new();
    let mut accepted_n = 0usize;
    let mut rejected_n = 0usize;
    let mut total_tokens = 0u64;
    let started = Instant::now();

    for i in 0..examples {
        let topic = TOPICS[i % TOPICS.len()];
        let task = TASKS[(i / TOPICS.len()) % TASKS.len()];
        let prompt = task.replace("{topic}", topic);
        let item_seed = seed + i as u64 + 1;
        let session = model.create_session(SessionConfig { seed: Some(item_seed), ..SessionConfig::default() })?;
        let mut chat = session.into_chat().map_err(|(_, e)| anyhow!("LFM3 chat template/session rejected: {e:?}"))?;
        chat.ingest_messages(&[Message::system(SYSTEM), Message::user(prompt.clone())])?;
        let opts = GenerateOpts {
            max_tokens,
            seed: Some(item_seed),
            temperature: 0.35,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.1,
            ..GenerateOpts::default()
        };
        let t0 = Instant::now();
        let result = chat.complete(&opts).with_context(|| format!("generation failed for BAR-{:06}", i + 1))?;
        let latency = t0.elapsed().as_secs_f64();
        let completion = result.text.trim().to_string();
        let tokens = result.summary.tokens_generated;
        total_tokens += tokens as u64;
        let normalized = completion.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        let digest_bytes = Sha256::digest(normalized.as_bytes());
        let digest = digest_bytes.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
        let mut status = "teacher_generated_unverified";
        let reason: Option<&str> = if completion.chars().count() < 100 { Some("too_short") }
            else if tokens < 18 { Some("too_few_tokens") }
            else if !seen.insert(digest) { Some("duplicate") }
            else { None };
        if reason.is_none() { status = "teacher_generated_heuristically_filtered"; accepted_n += 1; }
        else { rejected_n += 1; }
        let row = Row {
            id: format!("BAR-{:06}", i + 1), topic: topic.to_string(), prompt,
            completion, teacher: "Danielbrdz/Barcenas-LFM3-1.7B", system_prompt: SYSTEM,
            system_prompt_sha256: Sha256::digest(SYSTEM.as_bytes()).iter().map(|byte| format!("{byte:02x}")).collect(),
            format: "GGUF", quantization: "Q4_K_M", seed: item_seed,
            temperature: 0.35, completion_tokens: tokens,
            latency_seconds: (latency * 1000.0).round() / 1000.0, status,
        };
        let line = serde_json::to_string(&row)?;
        writeln!(raw, "{line}")?;
        if reason.is_none() { writeln!(accepted, "{line}")?; }
        else {
            let mut value = serde_json::to_value(&row)?;
            value["rejection_reason"] = serde_json::Value::String(reason.unwrap().to_string());
            writeln!(rejected, "{}", serde_json::to_string(&value)?)?;
        }
        if (i + 1) % 16 == 0 || i + 1 == examples {
            raw.flush()?; accepted.flush()?; rejected.flush()?;
            let elapsed = started.elapsed().as_secs_f64();
            println!("BARCENAS_RUST_PROGRESS {}/{} accepted={} rejected={} tokens={} elapsed={:.1}s samples_per_min={:.2} tokens_per_sec={:.2}",
                i + 1, examples, accepted_n, rejected_n, total_tokens, elapsed,
                (i + 1) as f64 / (elapsed / 60.0), total_tokens as f64 / elapsed);
        }
    }
    raw.flush()?; accepted.flush()?; rejected.flush()?;
    let elapsed = started.elapsed().as_secs_f64();
    let summary = serde_json::json!({
        "runtime": "native Rust in-process; Cera 0.7.5; CPU backend; no llama.cpp; no HTTP server; no Python inference process",
        "model": "Danielbrdz/Barcenas-LFM3-1.7B",
        "model_path": model_path,
        "model_sha256": sha256(&model_path)?,
        "format": "GGUF", "quantization": "Q4_K_M",
        "requested": examples, "generated": examples,
        "accepted_heuristically": accepted_n, "rejected": rejected_n,
        "completion_tokens": total_tokens, "elapsed_seconds": elapsed,
        "examples_per_minute": examples as f64 / (elapsed / 60.0),
        "generated_tokens_per_second": total_tokens as f64 / elapsed,
        "max_tokens": max_tokens, "seed": seed,
        "validation": "Length and duplicate filters only; no factual verification. Accepted examples remain draft data."
    });
    fs::write(format!("{out_dir}/summary.json"), serde_json::to_vec_pretty(&summary)?)?;
    println!("BARCENAS_RUST_RESULT {}", summary);
    Ok(())
}
