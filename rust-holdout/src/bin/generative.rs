use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{env, fs, path::Path};
use tch::{CModule, Device, IValue, Kind, Tensor};
use tokenizers::Tokenizer;

const BOS:u32=8190; const SYSTEM:u32=8184; const END_SYSTEM:u32=8185;
const USER:u32=8186; const END_USER:u32=8187; const ASSISTANT:u32=8188; const END_ASSISTANT:u32=8189;
fn arg(a:&[String],n:&str)->Result<String>{let i=a.iter().position(|x|x==n).ok_or_else(||anyhow!("missing {n}"))?;Ok(a[i+1].clone())}
fn rows(p:&str)->Result<Vec<Value>>{Ok(fs::read_to_string(p)?.lines().filter(|x|!x.trim().is_empty()).map(serde_json::from_str).collect::<std::result::Result<_,_>>()?)}
fn hash(p:&str)->Result<String>{Ok(format!("{:x}",Sha256::digest(fs::read(p)?)))}
fn get(v:&Value,k:&str)->String{v.get(k).and_then(Value::as_str).unwrap_or("").to_string()}
fn cached(m:&CModule,ids:&[u32],ck:Tensor,cv:Tensor,pos:usize)->Result<(Tensor,Tensor,Tensor)>{
 let v:Vec<i64>=ids.iter().map(|&x|x as i64).collect();let input=Tensor::from_slice(&v).view([1,v.len() as i64]);
 let z=m.method_is("forward_cached",&[IValue::Tensor(input),IValue::Tensor(ck),IValue::Tensor(cv),IValue::Int(pos as i64)])?;
 if let IValue::Tuple(mut t)=z {if t.len()!=3{return Err(anyhow!("bad cached tuple"))}
 let l=match t.remove(0){IValue::Tensor(x)=>x,_=>return Err(anyhow!("bad logits"))};
 let k=match t.remove(0){IValue::Tensor(x)=>x,_=>return Err(anyhow!("bad K"))};
 let v=match t.remove(0){IValue::Tensor(x)=>x,_=>return Err(anyhow!("bad V"))};Ok((l,k,v))
 }else{Err(anyhow!("cached forward did not return tuple"))}
}
fn generate(m:&CModule,tok:&Tokenizer,sys:&[u32],prompt:&str,max_len:usize,max_new:usize,eos:u32,pad:u32)->Result<(String,Vec<u32>)>{
 let user=tok.encode(prompt.trim(),false).map_err(|e|anyhow!("{e}"))?.get_ids().to_vec();
 if user.iter().any(|&x|x>=8184){return Err(anyhow!("prompt collides with reserved IDs"))}
 let mut ids=vec![BOS,SYSTEM];ids.extend_from_slice(sys);ids.extend([END_SYSTEM,USER]);ids.extend(user);ids.extend([END_USER,ASSISTANT]);
 if ids.len()>max_len {ids=ids[ids.len()-max_len..].to_vec()}
 let shape=[8,1,8,max_len as i64,64];let ck=Tensor::zeros(shape,(Kind::Float,Device::Cpu));let cv=Tensor::zeros(shape,(Kind::Float,Device::Cpu));
 let (mut logits,mut ck,mut cv)=cached(m,&ids,ck,cv,0)?;let mut generated=Vec::new();
 for i in 0..max_new {let next=logits.get(0).get(logits.size()[1]-1).argmax(0,false).int64_value(&[]) as u32;
 if next==eos||next==pad||next==END_ASSISTANT{break} generated.push(next);
 if i+1>=max_new||ids.len()+generated.len()>=max_len{break}
 let (l,k,v)=cached(m,&[next],ck,cv,ids.len()+generated.len()-1)?;logits=l;ck=k;cv=v;
 }
 let answer=tok.decode(&generated,true).map_err(|e|anyhow!("{e}"))?.trim().to_string();Ok((answer,generated))
}
fn main()->Result<()>{
 let a:Vec<String>=env::args().skip(1).collect();let model_path=arg(&a,"--model")?;let tok_path=arg(&a,"--tokenizer")?;let sys_path=arg(&a,"--system")?;let free_path=arg(&a,"--freeform")?;let v5_path=arg(&a,"--v5")?;let out=arg(&a,"--out-dir")?;let step:usize=arg(&a,"--step")?.parse()?;let threads:i32=arg(&a,"--threads")?.parse()?;
 let tok=Tokenizer::from_file(&tok_path).map_err(|e|anyhow!("{e}"))?;let eos=tok.token_to_id("[EOS]").ok_or_else(||anyhow!("missing EOS"))?;let pad=tok.token_to_id("[PAD]").ok_or_else(||anyhow!("missing PAD"))?;
 let sys=fs::read_to_string(&sys_path)?;let sys_ids=tok.encode(sys.trim(),false).map_err(|e|anyhow!("{e}"))?.get_ids().to_vec();if sys_ids.iter().any(|&x|x>=8184){return Err(anyhow!("system prompt role collision"))}
 let free=rows(&free_path)?;let v5=rows(&v5_path)?;if free.len()!=100||v5.len()!=20{return Err(anyhow!("expected 100 freeform and 20 V5 rows; got {} and {}",free.len(),v5.len()))}
 tch::set_num_threads(threads);tch::set_num_interop_threads(1);let mut m=CModule::load_on_device(&model_path,Device::Cpu)?;m.set_eval();fs::create_dir_all(&out)?;
 let mut fresults=Vec::new();let mut vresults=Vec::new();
 tch::no_grad(||->Result<()>{
 for (i,r) in free.iter().enumerate(){let p=get(r,"prompt");let (response,token_ids)=generate(&m,&tok,&sys_ids,&p,512,64,eos,pad)?;fresults.push(json!({"id":r.get("id"),"source":r.get("source"),"prompt":p,"response":response,"token_ids":token_ids}));if (i+1)%10==0{println!("freeform {}/100",i+1)}}
 for (i,r) in v5.iter().enumerate(){let p=get(r,"prompt");let (response,token_ids)=generate(&m,&tok,&sys_ids,&p,512,96,eos,pad)?;vresults.push(json!({"id":r.get("id"),"category":r.get("category"),"prompt":p,"response":response,"token_ids":token_ids}));println!("v5 {}/20 id={} category={}",i+1,get(r,"id"),get(r,"category"))}
 Ok(())})?;
 let f=json!({"checkpoint_step":step,"sample_size":100,"decoding":"greedy argmax; max_new_tokens=64; context=512; native Rust KV cache","prompt_sha256":hash(&free_path)?,"results":fresults});
 let v=json!({"checkpoint_step":step,"sample_size":20,"decoding":"greedy argmax; max_new_tokens=96; context=512; native Rust KV cache","prompt_sha256":hash(&v5_path)?,"results":vresults});
 fs::write(Path::new(&out).join(format!("freeform_100_v22_step{step}.json")),serde_json::to_vec_pretty(&f)?)?;
 fs::write(Path::new(&out).join(format!("v5_generative_holdout_20_step{step}.json")),serde_json::to_vec_pretty(&v)?)?;
 let meta=json!({"checkpoint_step":step,"runtime":"Rust + tch/LibTorch CPU","kv_cache_enabled":true,"threads":threads,"freeform_prompts":100,"v5_generative_prompts":20,"checkpoint_sha256":hash(&model_path)?,"tokenizer_sha256":hash(&tok_path)?,"system_prompt_sha256":hash(&sys_path)?,"freeform_prompt_sha256":hash(&free_path)?,"v5_prompt_sha256":hash(&v5_path)?});
 fs::write(Path::new(&out).join("generative_metadata.json"),serde_json::to_vec_pretty(&meta)?)?;println!("GENERATIVE_RESULT {}",meta);Ok(())
}