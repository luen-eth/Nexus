//! Run explicitly on a CUDA machine: cargo run --release --features cuda --example compare_cuda -- model.gguf
#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use nexus::{
        backend::BackendType,
        engine::{InferenceConfig, InferenceEngine},
        quant::KvQuantType,
    };
    use std::time::Instant;
    let path = std::env::args().nth(1).ok_or("Provide a GGUF path")?;
    let config = InferenceConfig {
        max_seq_len: 256,
        kv_quant: KvQuantType::None,
        ..Default::default()
    };
    let mut cpu = InferenceEngine::new(Some(config.clone()));
    cpu.load_model(&path)?;
    let t = cpu.tokenizer.clone().ok_or("Tokenizer missing")?;
    let prompt = t
        .try_encode("The capital of France is")
        .map_err(std::io::Error::other)?;
    let start = Instant::now();
    let cpu_logits = cpu.prefill(&prompt).map_err(std::io::Error::other)?;
    let cpu_prefill = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let cpu_tokens = cpu.generate(&prompt, 32).map_err(std::io::Error::other)?;
    let cpu_generation = start.elapsed().as_secs_f64();
    let mut gpu = InferenceEngine::new(Some(InferenceConfig {
        backend: BackendType::Cuda,
        ..config
    }));
    gpu.load_model(&path)?;
    let start = Instant::now();
    let gpu_logits = gpu.prefill(&prompt).map_err(std::io::Error::other)?;
    let gpu_prefill = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let gpu_tokens = gpu.generate(&prompt, 32).map_err(std::io::Error::other)?;
    let gpu_generation = start.elapsed().as_secs_f64();
    assert_eq!(cpu_logits.len(), gpu_logits.len());
    let max_abs = cpu_logits
        .iter()
        .zip(&gpu_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let max_scaled = cpu_logits
        .iter()
        .zip(&gpu_logits)
        .map(|(a, b)| (a - b).abs() / a.abs().max(1.))
        .fold(0f32, f32::max);
    let top = |logits: &[f32]| {
        let mut ids: Vec<_> = (0..logits.len()).collect();
        ids.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        ids.truncate(10);
        ids
    };
    let cpu_top = top(&cpu_logits);
    let gpu_top = top(&gpu_logits);
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
         "model":path,"prompt_tokens":prompt,"vocabulary":cpu_logits.len(),"max_absolute_logit_error":max_abs,"max_scaled_logit_error":max_scaled,
         "top10_cpu":cpu_top,"top10_gpu":gpu_top,"greedy_tokens_identical":cpu_tokens==gpu_tokens,
         "completion":t.try_decode(&gpu_tokens[prompt.len()..]).map_err(std::io::Error::other)?,
         "cpu_prefill_seconds":cpu_prefill,"gpu_prefill_seconds":gpu_prefill,
         "cpu_generation_seconds":cpu_generation,"gpu_generation_seconds":gpu_generation,
         "gpu_kernel_launches":gpu.backend.kernel_launches(),"gpu_memory":gpu.backend.memory_info().used_memory
        }))?
    );
    assert!(gpu.backend.kernel_launches() > 0);
    assert!(max_scaled < 0.005, "GPU/CPU logit discrepancy");
    assert_eq!(
        cpu_tokens, gpu_tokens,
        "Greedy outputs differ; inspect logits and near ties"
    );
    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Enable --features cuda to run this CUDA comparison");
    std::process::exit(1);
}
