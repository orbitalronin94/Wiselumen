use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

fn arg_value(args: &[String], name: &str) -> Result<String, String> {
    let i = args.iter().position(|x| x == name).ok_or_else(|| format!("Missing required argument {name}"))?;
    args.get(i + 1).cloned().ok_or_else(|| format!("Missing value for {name}"))
}

fn hash_file(path: &str) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("Cannot read {path}: {e}"))?;
    let mut h = Sha256::new();
    h.update(bytes);
    Ok(format!("{:x}", h.finalize()))
}

fn count_jsonl(path: &str) -> Result<usize, String> {
    let file = fs::File::open(path).map_err(|e| format!("Cannot open {path}: {e}"))?;
    let reader = BufReader::new(file);
    let mut count = 0usize;
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("Read error in {path}: {e}"))?;
        if line.trim().is_empty() { continue; }
        let value: Value = serde_json::from_str(&line)
            .map_err(|e| format!("Invalid JSON at {path}:{}: {e}", line_no + 1))?;
        if !value.is_object() {
            return Err(format!("Expected JSON object at {path}:{}", line_no + 1));
        }
        count += 1;
    }
    Ok(count)
}

fn main() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    let python = arg_value(&args, "--python")?;
    let script = arg_value(&args, "--script")?;
    let mc = arg_value(&args, "--mc-holdout")?;
    let freeform = arg_value(&args, "--freeform-holdout")?;
    let out_dir = PathBuf::from(arg_value(&args, "--out-dir")?);

    let mc_rows = count_jsonl(&mc)?;
    let freeform_rows = count_jsonl(&freeform)?;
    if mc_rows != 2431 {
        return Err(format!("Legacy MC holdout identity check failed: expected 2431 rows, found {mc_rows}"));
    }
    if freeform_rows != 100 {
        return Err(format!("Legacy freeform holdout identity check failed: expected 100 rows, found {freeform_rows}"));
    }

    fs::create_dir_all(&out_dir).map_err(|e| format!("Cannot create output directory: {e}"))?;
    let mut forwarded: Vec<String> = Vec::new();
    let mut skip_next = false;
    for (i, arg) in args.iter().enumerate() {
        if skip_next { skip_next = false; continue; }
        if arg == "--python" || arg == "--script" { skip_next = true; continue; }
        forwarded.push(arg.clone());
        if arg == "--mc-holdout" || arg == "--freeform-holdout" || arg == "--out-dir" { skip_next = true; }
    }

    let inputs = json!({
        "checkpoint_sha256": hash_file(&arg_value(&args, "--checkpoint")?)?,
        "tokenizer_sha256": hash_file(&arg_value(&args, "--tokenizer")?)?,
        "system_prompt_sha256": hash_file(&arg_value(&args, "--system")?)?,
        "mc_holdout_sha256": hash_file(&mc)?,
        "freeform_holdout_sha256": hash_file(&freeform)?,
    });

    println!("RUST_PREFLIGHT {{\"mc_rows\":{mc_rows},\"freeform_rows\":{freeform_rows}}}");
    let started = Instant::now();
    let status = Command::new(&python)
        .arg(&script)
        .args(&forwarded)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| format!("Could not launch Python inference backend: {e}"))?;
    let elapsed = started.elapsed().as_secs_f64();

    let metadata = json!({
        "runner": "lumen-holdout-rust",
        "runner_version": env!("CARGO_PKG_VERSION"),
        "inference_backend": "existing PyTorch LUMEN V2.2 evaluator (kept unchanged for parity)",
        "mc_rows_preflight": mc_rows,
        "freeform_rows_preflight": freeform_rows,
        "elapsed_seconds": elapsed,
        "inputs": inputs,
        "python_exit_code": status.code(),
        "success": status.success(),
        "note": "Rust validates frozen inputs, fingerprints all evaluation inputs, orchestrates the persistent evaluation process and records wall time. Model forward remains PyTorch in this parity-first stage."
    });
    let meta_path = out_dir.join("rust_pipeline_metadata.json");
    fs::write(&meta_path, serde_json::to_vec_pretty(&metadata).map_err(|e| e.to_string())?)
        .map_err(|e| format!("Cannot write {}: {e}", meta_path.display()))?;
    println!("RUST_PIPELINE_RESULT {}", metadata);
    if !status.success() {
        return Err(format!("Python evaluator failed with status {:?}", status.code()));
    }
    Ok(())
}
