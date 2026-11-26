use anyhow::{anyhow, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{env, fs, path::Path, time::Instant};
use tch::{CModule, Device, IValue, Kind, Tensor};
use tokenizers::Tokenizer;

const BOS: u32 = 8190;
const SYSTEM: u32 = 8184;
const END_SYSTEM: u32 = 8185;
const USER: u32 = 8186;
const END_USER: u32 = 8187;
const ASSISTANT: u32 = 8188;
const END_ASSISTANT: u32 = 8189;

fn arg(args: &[String], name: &str) -> Result<String> {
    let i = args.iter().position(|x| x == name).ok_or_else(|| anyhow!("Missing {name}"))?;
    args.get(i + 1).cloned().ok_or_else(|| anyhow!("Missing value for {name}"))
}
fn hash(path: &str) -> Result<String> {
    let bytes = fs::read(path).with_context(|| format!("read {path}"))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
fn rows(path: &str) -> Result<Vec<Value>> {
    let content = fs::read_to_string(path).with_context(|| format!("read {path}"))?;
    content.lines().filter(|x| !x.trim().is_empty())
        .map(|line| serde_json::from_str(line).context("invalid JSONL row"))
        .collect()
}
fn text(v: &Value) -> String {
    match v { Value::String(s) => s.clone(), Value::Null => String::new(), _ => v.to_string() }
}
#[derive(Default)]
struct Timings { tokenization_seconds: f64, mc_seconds: f64, freeform_prefill_seconds: f64, freeform_decode_seconds: f64, generation_forward_calls: usize }
fn encode(tok: &Tokenizer, s: &str, timing: &mut Timings) -> Result<Vec<u32>> {
    let t = Instant::now();
    let ids = tok.encode(s, false).map_err(|e| anyhow!("tokenizer encode: {e}"))?.get_ids().to_vec();
    timing.tokenization_seconds += t.elapsed().as_secs_f64();
    Ok(ids)
}
fn prefix(tok: &Tokenizer, system: &[u32], prompt: &str, timing: &mut Timings) -> Result<Vec<u32>> {
    let user = encode(tok, prompt.trim(), timing)?;
    if user.iter().any(|&x| x >= 8184) { return Err(anyhow!("ordinary user text collided with reserved role IDs")); }
    let mut ids = vec![BOS, SYSTEM];
    ids.extend_from_slice(system);
    ids.extend([END_SYSTEM, USER]);
    ids.extend(user);
    ids.extend([END_USER, ASSISTANT]);
    Ok(ids)
}
fn forward(model: &CModule, ids: &[u32], max_len: usize) -> Result<Tensor> {
    let context = if ids.len() > max_len { &ids[ids.len()-max_len..] } else { ids };
    let v: Vec<i64> = context.iter().map(|&x| x as i64).collect();
    let input = Tensor::from_slice(&v).to_device(Device::Cpu).view([1, v.len() as i64]);
    Ok(model.forward_ts(&[input])?)
}
fn candidate_nll(model: &CModule, tok: &Tokenizer, prefix: &[u32], choice: &Value, max_len: usize, timing: &mut Timings) -> Result<f64> {
    let mut suffix = encode(tok, &format!("\n{}", text(choice)), timing)?;
    if suffix.is_empty() { return Ok(f64::INFINITY); }
    if suffix.len() >= max_len { suffix.truncate(max_len - 1); }
    let context_len = max_len.saturating_sub(suffix.len()).max(1).min(prefix.len());
    let context = &prefix[prefix.len()-context_len..];
    let mut ids = context.to_vec();
    ids.extend_from_slice(&suffix);
    let logits = forward(model, &ids, max_len)?;
    let start = context.len() as i64 - 1;
    let logp = logits.get(0).log_softmax(-1, Kind::Float)
        .narrow(0, start, suffix.len() as i64);
    let targets: Vec<i64> = suffix.iter().map(|&x| x as i64).collect();
    let target = Tensor::from_slice(&targets).view([-1, 1]);
    let selected = logp.gather(1, &target, false);
    Ok(-selected.mean(Kind::Float).double_value(&[]))
}
fn cached_forward(model: &CModule, ids: &[u32], cache_k: Tensor, cache_v: Tensor, position: usize) -> Result<(Tensor, Tensor, Tensor)> {
    let v: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
    let input = Tensor::from_slice(&v).to_device(Device::Cpu).view([1, v.len() as i64]);
    let result = model.method_is("forward_cached", &[
        IValue::Tensor(input), IValue::Tensor(cache_k), IValue::Tensor(cache_v),
        IValue::Int(position as i64)
    ])?;
    match result {
        IValue::Tuple(mut values) if values.len() == 3 => {
            let logits = match values.remove(0) { IValue::Tensor(t) => t, _ => return Err(anyhow!("cached forward returned non-tensor logits")) };
            let ck = match values.remove(0) { IValue::Tensor(t) => t, _ => return Err(anyhow!("cached forward returned non-tensor K cache")) };
            let cv = match values.remove(0) { IValue::Tensor(t) => t, _ => return Err(anyhow!("cached forward returned non-tensor V cache")) };
            Ok((logits, ck, cv))
        }
        other => Err(anyhow!("unexpected cached forward return: {:?}", other)),
    }
}
fn generate(model: &CModule, tok: &Tokenizer, prefix: &[u32], max_len: usize, max_new: usize,
            eos: u32, pad: u32, timing: &mut Timings) -> Result<(String, Vec<u32>)> {
    let context = if prefix.len() > max_len { &prefix[prefix.len()-max_len..] } else { prefix };
    let mut cache_k = Tensor::zeros([8,1,8,max_len as i64,64], (Kind::Float,Device::Cpu));
    let mut cache_v = Tensor::zeros([8,1,8,max_len as i64,64], (Kind::Float,Device::Cpu));
    let t = Instant::now();
    let (mut logits, ck, cv) = cached_forward(model, context, cache_k, cache_v, 0)?;
    timing.freeform_prefill_seconds += t.elapsed().as_secs_f64();
    cache_k = ck; cache_v = cv;
    let mut generated = Vec::new();
    for step in 0..max_new {
        let last = logits.get(0).get(logits.size()[1]-1);
        let next = last.argmax(0, false).int64_value(&[]) as u32;
        if next == eos || next == pad || next == END_ASSISTANT { break; }
        generated.push(next);
        if step + 1 >= max_new || context.len() + generated.len() >= max_len { break; }
        let t = Instant::now();
        let (next_logits, ck, cv) = cached_forward(model, &[next], cache_k, cache_v, context.len()+generated.len()-1)?;
        timing.freeform_decode_seconds += t.elapsed().as_secs_f64();
        timing.generation_forward_calls += 1;
        logits = next_logits;
        cache_k = ck; cache_v = cv;
    }
    let answer = tok.decode(&generated, true).map_err(|e| anyhow!("tokenizer decode: {e}"))?.trim().to_string();
    Ok((answer, generated))
}
fn normalized_answer(answer: &Value, choices: &[Value]) -> Option<usize> {
    if let Some(i) = answer.as_u64() { return if (i as usize) < choices.len() { Some(i as usize) } else { None }; }
    if let Some(s) = answer.as_str() {
        let s = s.trim();
        if let Ok(i) = s.parse::<usize>() { if i < choices.len() { return Some(i); } }
        if s.len() == 1 {
            let b = s.as_bytes()[0].to_ascii_uppercase();
            if (b'A'..=b'Z').contains(&b) { let i = (b-b'A') as usize; if i < choices.len() { return Some(i); } }
        }
        for (i,c) in choices.iter().enumerate() { if s == text(c).trim() { return Some(i); } }
    }
    None
}
fn choices_of(v: &Value) -> Vec<Value> {
    if let Some(a) = v.as_array() { return a.clone(); }
    if let Some(m) = v.as_object() {
        let mut entries: Vec<_> = m.iter().collect();
        entries.sort_by(|a,b| a.0.cmp(b.0));
        return entries.into_iter().map(|(_,v)| v.clone()).collect();
    }
    Vec::new()
}
fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();
    let model_path = arg(&args, "--model")?;
    let tokenizer_path = arg(&args, "--tokenizer")?;
    let system_path = arg(&args, "--system")?;
    let mc_path = arg(&args, "--mc-holdout")?;
    let free_path = arg(&args, "--freeform-holdout")?;
    let out = arg(&args, "--out-dir")?;
    let step: usize = arg(&args, "--step")?.parse()?;
    let threads: i32 = arg(&args, "--threads")?.parse().unwrap_or(4);
    let max_len: usize = args.iter().position(|x| x=="--max-len").and_then(|i|args.get(i+1)).and_then(|x|x.parse().ok()).unwrap_or(512);
    let max_new: usize = args.iter().position(|x| x=="--max-new-tokens").and_then(|i|args.get(i+1)).and_then(|x|x.parse().ok()).unwrap_or(64);
    tch::set_num_threads(threads);
    tch::set_num_interop_threads(1);

    let mc = rows(&mc_path)?;
    let free = rows(&free_path)?;
    if mc.len()!=2431 { return Err(anyhow!("expected 2431 MC rows, got {}",mc.len())); }
    if free.len()!=100 { return Err(anyhow!("expected 100 freeform rows, got {}",free.len())); }
    let tok = Tokenizer::from_file(&tokenizer_path).map_err(|e|anyhow!("tokenizer load: {e}"))?;
    let eos = tok.token_to_id("[EOS]").ok_or_else(||anyhow!("tokenizer lacks [EOS]"))?;
    let pad = tok.token_to_id("[PAD]").ok_or_else(||anyhow!("tokenizer lacks [PAD]"))?;
    let system = fs::read_to_string(&system_path)?;
    let mut timing = Timings::default();
    let system_ids = encode(&tok, system.trim(), &mut timing)?;
    if system_ids.iter().any(|&x|x>=8184) { return Err(anyhow!("system prompt collides with reserved IDs")); }
    let mut model = CModule::load_on_device(&model_path, Device::Cpu)?;
    model.set_eval();
    fs::create_dir_all(&out)?;
    let started = Instant::now();

    let mc_started = Instant::now();\n    let mc_started = Instant::now();\n    let mc_started = Instant::now();
    let mut predictions = Vec::new();
    let mut correct = 0usize;
    let mut scored = 0usize;
    let mut nll_sum = 0.0f64;
    tch::no_grad(|| -> Result<()> {
        for (n,row) in mc.iter().enumerate() {
            let prompt = text(&row["prompt"]);
            let choices = choices_of(&row["choices"]);
            let gold = match normalized_answer(&row["answer"], &choices) { Some(x)=>x, None=>continue };
            if prompt.trim().is_empty() || choices.len()<2 { continue; }
            let p = prefix(&tok,&system_ids,&prompt,&mut timing)?;
            let mut scores = Vec::new();
            for c in &choices { scores.push(candidate_nll(&model,&tok,&p,c,max_len,&mut timing)?); }
            let pred = scores.iter().enumerate().min_by(|a,b|a.1.total_cmp(b.1)).map(|x|x.0).unwrap();
            scored += 1;
            if pred==gold { correct += 1; }
            nll_sum += scores[gold];
            predictions.push(json!({"id":row.get("id").cloned().unwrap_or(json!(n+1)),"gold":gold,"pred":pred,"scores":scores}));
            if (n+1)%100==0 { println!("rust_mc_progress {}/{}",n+1,mc.len()); }
        }
        Ok(())
    })?;
    timing.mc_seconds = mc_started.elapsed().as_secs_f64();\n    timing.mc_seconds = mc_started.elapsed().as_secs_f64();\n    timing.mc_seconds = mc_started.elapsed().as_secs_f64();
    let accuracy = correct as f64 / scored.max(1) as f64;
    let mc_report = json!({
        "checkpoint_step":step,"rows_in_source":mc.len(),"scored":scored,"correct":correct,
        "accuracy":accuracy,"mean_gold_answer_nll":nll_sum/scored.max(1) as f64,
        "holdout_sha256":hash(&mc_path)?,"tokenizer_sha256":hash(&tokenizer_path)?,
        "active_layers":[0,1,2,5,6,7],
        "method":"legacy frozen 2431-question MC set; conditional answer NLL; V2.2 system/user/assistant framing; greedy argmin"
    });
    let mut mc_full = mc_report.as_object().unwrap().clone();
    mc_full.insert("predictions".into(),Value::Array(predictions));
    fs::write(Path::new(&out).join(format!("official_holdout_2431_v22_step{step}.json")),serde_json::to_vec_pretty(&Value::Object(mc_full))?)?;

    let free_started = Instant::now();\n    let free_started = Instant::now();\n    let free_started = Instant::now();
    let mut free_results = Vec::new();
    tch::no_grad(|| -> Result<()> {
        for (n,row) in free.iter().enumerate() {
            let prompt = text(&row["prompt"]);
            let p = prefix(&tok,&system_ids,&prompt,&mut timing)?;
            let (response,token_ids)=generate(&model,&tok,&p,max_len,max_new,eos,pad,&mut timing)?;
            free_results.push(json!({"id":row.get("id").cloned().unwrap_or(json!(n+1)),"source":row.get("source").cloned().unwrap_or(Value::Null),"prompt":prompt,"response":response,"token_ids":token_ids}));
            if (n+1)%10==0 { println!("rust_freeform_progress {}/100",n+1); }
        }
        Ok(())
    })?;
    let freeform_seconds = free_started.elapsed().as_secs_f64();\n    let freeform_seconds = free_started.elapsed().as_secs_f64();\n    let freeform_seconds = free_started.elapsed().as_secs_f64();
    let free_report=json!({"checkpoint_step":step,"sample_size":free_results.len(),"active_layers":[0,1,2,5,6,7],
        "decoding":format!("greedy argmax; max_new_tokens={max_new}; context={max_len}"),
        "prompt_sha256":hash(&free_path)?,"results":free_results});
    fs::write(Path::new(&out).join(format!("freeform_100_v22_step{step}.json")),serde_json::to_vec_pretty(&free_report)?)?;
    let elapsed=started.elapsed().as_secs_f64();
    let meta=json!({"runtime":"Rust + tch/LibTorch CPU","inference_backend":"TorchScript CModule; no Python process during inference",
        "elapsed_seconds":elapsed,"threads":threads,
        "stage_timings":{"mc_total_seconds":timing.mc_seconds,"freeform_total_seconds":freeform_seconds,
            "tokenization_seconds":timing.tokenization_seconds,"generation_prefill_seconds":timing.freeform_prefill_seconds,
            "generation_decode_kv_seconds":timing.freeform_decode_seconds,"generation_forward_calls":timing.generation_forward_calls,
            "kv_cache_enabled":true},
        "checkpoint_sha256":hash(&model_path)?,"tokenizer_sha256":hash(&tokenizer_path)?,
        "system_prompt_sha256":hash(&system_path)?,"mc_holdout_sha256":hash(&mc_path)?,
        "freeform_holdout_sha256":hash(&free_path)?});
    fs::write(Path::new(&out).join("native_rust_metadata.json"),serde_json::to_vec_pretty(&meta)?)?;
    println!("NATIVE_RUST_RESULT {}",meta);
    println!("MC_RESULT {}",mc_report);
    Ok(())
}