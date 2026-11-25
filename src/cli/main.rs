//! Nexus CLI tool - command-line interface for model inference.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use nexus::backend::{cpu::CpuBackend, BackendType};
use nexus::download;
use nexus::engine::{GenerationConfig, InferenceConfig, InferenceEngine};
use nexus::quant::{KvQuantType, QuantFormat};

#[derive(Parser, Debug)]
#[command(
    name = "nexus",
    version,
    about = "Nexus AI Inference Engine - High-performance LLM inference"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run inference on a GGUF model
    Run {
        /// Path to GGUF model file
        #[arg(short, long)]
        model: PathBuf,

        /// Prompt text for generation
        #[arg(short, long)]
        prompt: Option<String>,

        /// Number of tokens to generate
        #[arg(short = 'n', long, default_value = "128")]
        max_tokens: usize,

        /// Sampling temperature; 0 means greedy decoding
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,

        /// Nucleus sampling probability
        #[arg(long, default_value_t = 1.0)]
        top_p: f32,

        /// Keep only the top K logits before sampling; 0 disables top-k
        #[arg(long, default_value_t = 0)]
        top_k: usize,

        /// Repetition penalty applied to tokens already in the context
        #[arg(long, default_value_t = 1.0)]
        repeat_penalty: f32,

        /// Deterministic sampling seed
        #[arg(long)]
        seed: Option<u64>,

        /// Stop sequence; can be provided multiple times
        #[arg(long = "stop")]
        stop: Vec<String>,

        /// Context length
        #[arg(long, default_value_t = 4096)]
        ctx_size: usize,

        /// KV cache quantization type; none keeps the reference float quality
        #[arg(long, default_value = "none")]
        kv_quant: String,

        /// Runtime weight storage (FP32 only); use convert for file quantization.
        #[arg(long, default_value = "FP32")]
        weight_quant: String,

        /// Backend to use (cpu-simd, cuda); CUDA requires --features cuda
        #[arg(long, default_value = "auto")]
        backend: String,

        /// CPU matvec thread limit (default: 1). Small matrices stay serial.
        #[arg(long)]
        threads: Option<usize>,
        /// Allow a byte tokenizer only for synthetic test fixtures.
        #[arg(long)]
        allow_byte_tokenizer: bool,
    },

    /// Run a benchmark
    Bench {
        /// Path to GGUF model file
        #[arg(short, long)]
        model: PathBuf,

        /// Number of benchmark iterations
        #[arg(short, long, default_value = "3")]
        iterations: usize,

        /// Prompt length for benchmark
        #[arg(long, default_value_t = 512)]
        prompt_len: usize,

        /// Comma-separated prompt lengths for a benchmark matrix
        #[arg(long, value_delimiter = ',')]
        prompt_lens: Option<Vec<usize>>,

        /// Generated tokens per iteration
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,

        /// Warmup runs per prompt length, excluded from measured results.
        #[arg(long, default_value_t = 1)]
        warmup: usize,

        /// Backend to benchmark (cpu-simd or cuda).
        #[arg(long, default_value = "auto")]
        backend: String,

        /// CPU matvec thread limit (default: 1). Rejected for CUDA.
        #[arg(long)]
        threads: Option<usize>,

        /// KV format used for the benchmark; none keeps the reference float quality
        #[arg(long, default_value = "none")]
        kv_quant: String,

        /// Allow a byte tokenizer only for synthetic test fixtures.
        #[arg(long)]
        allow_byte_tokenizer: bool,

        /// Emit machine-readable JSON.
        #[arg(long)]
        json: bool,
    },

    /// Emit a golden logits report for regression testing
    Golden {
        /// Path to GGUF model file
        #[arg(short, long)]
        model: PathBuf,

        /// Prompt text used for prefill
        #[arg(short, long)]
        prompt: String,

        /// Number of top logits to include
        #[arg(long, default_value_t = 10)]
        top_k: usize,

        /// Optional JSON output path
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Show model information
    Info {
        /// Path to GGUF model file
        #[arg(short, long)]
        model: PathBuf,
    },

    /// Convert a model to GGUF format
    Convert {
        /// Source model path or HuggingFace repo ID
        #[arg(short, long)]
        input: PathBuf,

        /// Output GGUF path
        #[arg(short, long)]
        output: PathBuf,

        /// Quantization format; defaults to a faithful F32 copy, lossy codecs
        /// must be requested explicitly (same default as nexus-convert)
        #[arg(long, default_value = "f32")]
        quant: String,
    },

    /// Download a model from HuggingFace
    Download {
        /// Model identifier (owner/repository/path/to/file or index from list)
        #[arg(short, long)]
        model: Option<String>,

        /// Output directory for models
        #[arg(short, long, default_value = "models")]
        output: PathBuf,

        /// List available models instead of downloading
        #[arg(short, long)]
        list: bool,

        /// Download multiple models (comma-separated filters)
        #[arg(long, value_delimiter = ',')]
        all: Option<Vec<String>>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Restore the Unix pipe convention for the CLI: `nexus info | head`
    // should end quietly at the reader's close instead of panicking on
    // EPIPE with exit code 101. The server keeps SIGPIPE ignored on
    // purpose (a closed docker log stream must not kill it).
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }

    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "nexus=info,llama_cpp=warn".into()),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Some(Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            seed,
            stop,
            ctx_size,
            kv_quant,
            weight_quant,
            backend,
            threads,
            allow_byte_tokenizer,
        }) => {
            let generation = GenerationConfig {
                max_tokens,
                temperature,
                top_p,
                top_k,
                repetition_penalty: repeat_penalty,
                seed,
                stop_sequences: Vec::new(),
            };
            run_inference(
                model,
                prompt,
                generation,
                stop,
                ctx_size,
                &kv_quant,
                &weight_quant,
                &backend,
                threads,
                allow_byte_tokenizer,
            )
            .await?;
        }
        Some(Commands::Bench {
            model,
            iterations,
            prompt_len,
            prompt_lens,
            max_tokens,
            warmup,
            backend,
            threads,
            kv_quant,
            allow_byte_tokenizer,
            json,
        }) => {
            let prompt_lens = prompt_lens.unwrap_or_else(|| vec![prompt_len]);
            run_benchmark(
                model,
                iterations,
                prompt_lens,
                max_tokens,
                warmup,
                &backend,
                threads,
                &kv_quant,
                allow_byte_tokenizer,
                json,
            )?;
        }
        Some(Commands::Golden {
            model,
            prompt,
            top_k,
            output,
        }) => {
            emit_golden_logits(model, &prompt, top_k, output.as_deref())?;
        }
        Some(Commands::Info { model }) => {
            show_model_info(&model)?;
        }
        Some(Commands::Convert {
            input,
            output,
            quant,
        }) => {
            run_convert_command(&input, &output, &quant)?;
        }
        Some(Commands::Download {
            model,
            output,
            list,
            all,
        }) => {
            run_download(model.as_deref(), &output, list, all.as_deref()).await?;
        }
        None => {
            println!("Nexus AI Inference Engine v{}", env!("CARGO_PKG_VERSION"));
            println!("Use --help for usage information.");
            println!();
            println!("Examples:");
            println!("  nexus run -m model.gguf -p 'Hello, world!'");
            println!("  nexus bench -m model.gguf");
            println!("  nexus info -m model.gguf");
        }
    }

    Ok(())
}

fn run_convert_command(input: &Path, output: &Path, quant: &str) -> Result<()> {
    let quant = quant.to_ascii_lowercase().replace('_', "-");
    let sibling = std::env::current_exe()
        .ok()
        .map(|path| path.with_file_name("nexus-convert"));
    let converter = sibling
        .filter(|path| path.exists())
        .unwrap_or_else(|| PathBuf::from("nexus-convert"));

    let status = Command::new(&converter)
        .arg("--input")
        .arg(input)
        .arg("--output")
        .arg(output)
        .arg("--quant")
        .arg(quant)
        .status()
        .with_context(|| format!("Failed to launch converter: {}", converter.display()))?;

    if !status.success() {
        anyhow::bail!("Model conversion failed with status {}", status);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn run_inference(
    model_path: PathBuf,
    prompt: Option<String>,
    mut generation: GenerationConfig,
    stop: Vec<String>,
    ctx_size: usize,
    kv_quant: &str,
    weight_quant: &str,
    backend: &str,
    threads: Option<usize>,
    allow_byte_tokenizer: bool,
) -> Result<()> {
    // Parse configuration
    let kv_quant_type = KvQuantType::from_str(&kv_quant.to_ascii_lowercase().replace('-', "_"))
        .ok_or_else(|| anyhow::anyhow!("Unsupported KV quantization: {}", kv_quant))?;
    let weight_format = QuantFormat::from_str(&weight_quant.to_ascii_uppercase().replace('-', "_"))
        .ok_or_else(|| anyhow::anyhow!("Unsupported weight quantization: {}", weight_quant))?;
    if weight_format != QuantFormat::FP32 {
        anyhow::bail!("Runtime weights expand to F32; --weight-quant accepts FP32 only. Use the converter for file quantization");
    }

    let backend_type = match backend {
        "auto" | "cpu" | "cpu-simd" => BackendType::CpuSimd,
        "cuda" if cfg!(feature = "cuda") => BackendType::Cuda,
        "metal" | "cuda" | "vulkan" | "webgpu" => {
            anyhow::bail!(
                "Backend '{}' inference kernels are not implemented; use cpu-simd",
                backend
            );
        }
        other => anyhow::bail!("Unsupported backend: {}", other),
    };

    let config = InferenceConfig {
        max_seq_len: ctx_size,
        allow_byte_tokenizer,
        kv_quant: kv_quant_type,
        weight_format,
        backend: backend_type,
        ..Default::default()
    };

    // Create engine
    let mut engine = InferenceEngine::new(Some(config));
    let cpu_threads = configure_cpu_threads(&mut engine, threads)?;

    println!("Loading model: {}", model_path.display());
    println!("Backend: {}", engine.backend.name());
    if let Some((threads, kernel)) = cpu_threads {
        println!("CPU matvec thread limit: {threads} (small matrices stay serial)");
        println!("CPU matvec kernel: {kernel}");
    }
    println!("KV Quant: {}", kv_quant_type.name());
    println!("Weight Quant: {}", weight_format.name());
    println!();

    // Reject a prompt-less run before touching the model: loading a multi-GB
    // GGUF only to report a missing prompt wasted minutes and RAM. The
    // argument validations above stay ahead of this check on purpose.
    let Some(prompt) = prompt else {
        anyhow::bail!("no prompt provided: use -p 'text' (or --stop for stop sequences)");
    };

    // Load model
    match engine.load_model(&model_path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Model file not found: {}", model_path.display());
            eprintln!();
            eprintln!("Download a GGUF model first:");
            eprintln!("  huggingface-cli download <repo> <file> --local-dir .");
            std::process::exit(1);
        }
        Err(e) => return Err(e.into()),
    }

    // Show model info
    println!("{}", engine.info());
    println!(
        "Backend capabilities: {:?}",
        engine.backend.capabilities().features
    );
    println!();

    if let Some(model) = &engine.model {
        println!(
            "Model: {}",
            model.metadata.get_str("general.name").unwrap_or("Unknown")
        );
        println!("Architecture: {:?}", model.architecture());
        println!();
    }

    // Run inference
    println!("Prompt: {}", prompt);
    println!("---");

    let tokenizer = engine
        .tokenizer
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Tokenizer not loaded"))?;
    generation.stop_sequences.clear();
    let tokens = tokenizer.try_encode(&prompt).map_err(anyhow::Error::msg)?;
    let mut session = engine
        .start_session(&tokens, &generation)
        .map_err(anyhow::Error::msg)?;
    while !session.finished {
        engine
            .step_session(&mut session)
            .map_err(anyhow::Error::msg)?;
        let decoded = tokenizer
            .try_decode(session.visible_output())
            .map_err(anyhow::Error::msg)?;
        let (visible, matched) =
            nexus::tokenizer::filter_stop_text(&decoded, &stop, session.finished);
        if matched || session.finished {
            println!("{}", visible);
            break;
        }
    }

    // Print memory report
    if let Some(model) = &engine.model {
        model.memory.report();
    }

    Ok(())
}

fn configure_cpu_threads(
    engine: &mut InferenceEngine,
    threads: Option<usize>,
) -> Result<Option<(usize, &'static str)>> {
    if engine.config.backend != BackendType::CpuSimd {
        if threads.is_some() {
            anyhow::bail!("--threads is only supported for the CPU backend");
        }
        return Ok(None);
    }
    let count = threads.unwrap_or(1);
    let cpu = CpuBackend::with_threads(count).map_err(anyhow::Error::msg)?;
    let kernel = cpu.matvec_kernel();
    engine.backend = Box::new(cpu);
    Ok(Some((count, kernel)))
}

#[derive(Serialize)]
struct BenchmarkSample {
    iteration: usize,
    generated_tokens: usize,
    prefill_seconds: f64,
    decode_seconds: f64,
    elapsed_seconds: f64,
    first_token_seconds: Option<f64>,
    tokens_per_second: f64,
    decode_tokens_per_second: f64,
    finish_reason: String,
}

fn benchmark_sample(
    engine: &mut InferenceEngine,
    prompt: &[u32],
    max_tokens: usize,
    iteration: usize,
) -> Result<BenchmarkSample> {
    let start = Instant::now();
    let mut session = engine
        .start_session(
            prompt,
            &GenerationConfig {
                max_tokens,
                temperature: 0.0,
                seed: Some(42),
                ..Default::default()
            },
        )
        .map_err(anyhow::Error::msg)?;
    // start_session allocates private caches; step_session advances one prefill
    // token per call. Include session setup in the prefill phase.
    for _ in 0..prompt.len() {
        if engine
            .step_session(&mut session)
            .map_err(anyhow::Error::msg)?
            .is_some()
            || session.finished
        {
            anyhow::bail!("Generation session did not preserve the prefill boundary");
        }
    }
    let prefill_seconds = start.elapsed().as_secs_f64();
    let decode_start = Instant::now();
    let mut first_token_seconds = None;
    while !session.finished {
        if engine
            .step_session(&mut session)
            .map_err(anyhow::Error::msg)?
            .is_some()
            && first_token_seconds.is_none()
        {
            first_token_seconds = Some(start.elapsed().as_secs_f64());
        }
    }
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    let elapsed_seconds = start.elapsed().as_secs_f64();
    let generated_tokens = session.output.len().saturating_sub(prompt.len());
    Ok(BenchmarkSample {
        iteration,
        generated_tokens,
        prefill_seconds,
        decode_seconds,
        elapsed_seconds,
        first_token_seconds,
        tokens_per_second: rate(generated_tokens, elapsed_seconds),
        decode_tokens_per_second: rate(generated_tokens, decode_seconds),
        finish_reason: session.finish_reason.into(),
    })
}

fn rate(count: usize, seconds: f64) -> f64 {
    if seconds > 0.0 {
        count as f64 / seconds
    } else {
        0.0
    }
}

fn percentiles(mut values: Vec<f64>) -> serde_json::Value {
    values.sort_by(f64::total_cmp);
    // Nearest-rank percentiles. Called only after at least one measured run.
    let percentile = |p: f64| values[((p * values.len() as f64).ceil() as usize).saturating_sub(1)];
    serde_json::json!({"p50":percentile(0.5),"p95":percentile(0.95)})
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

#[allow(clippy::too_many_arguments)]
fn run_benchmark(
    model_path: PathBuf,
    iterations: usize,
    prompt_lens: Vec<usize>,
    max_tokens: usize,
    warmup: usize,
    backend: &str,
    threads: Option<usize>,
    kv_quant: &str,
    allow_byte_tokenizer: bool,
    json: bool,
) -> Result<()> {
    if iterations == 0 || max_tokens == 0 || prompt_lens.is_empty() || prompt_lens.contains(&0) {
        anyhow::bail!("Benchmark iterations, max_tokens and prompt lengths must be positive");
    }
    let backend = match backend {
        "auto" | "cpu" | "cpu-simd" => BackendType::CpuSimd,
        "cuda" if cfg!(feature = "cuda") => BackendType::Cuda,
        other => anyhow::bail!("Unsupported benchmark backend: {other}"),
    };
    let kv_quant = KvQuantType::from_str(&kv_quant.to_ascii_lowercase().replace('-', "_"))
        .ok_or_else(|| anyhow::anyhow!("Unsupported KV quantization: {kv_quant}"))?;
    let mut engine = InferenceEngine::new(Some(InferenceConfig {
        backend,
        kv_quant,
        allow_byte_tokenizer,
        ..Default::default()
    }));
    let cpu_threads = configure_cpu_threads(&mut engine, threads)?;
    let model_sha256 = sha256_file(&model_path)?;
    let loading = Instant::now();
    engine.load_model(&model_path)?;
    let model_load_seconds = loading.elapsed().as_secs_f64();
    for &len in &prompt_lens {
        if len
            .checked_add(max_tokens)
            .is_none_or(|n| n > engine.config.max_seq_len)
        {
            anyhow::bail!(
                "Benchmark prompt length {len} plus max_tokens exceeds context length {}",
                engine.config.max_seq_len
            );
        }
    }
    let tokenizer = engine
        .tokenizer
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("Tokenizer not loaded"))?;
    let needed = *prompt_lens.iter().max().unwrap();
    let valid_ids: Vec<u32> = (0..engine.vocab_size)
        .filter_map(|id| u32::try_from(id).ok())
        .filter(|&id| tokenizer.valid_id(id))
        .take(needed)
        .collect();
    if valid_ids.is_empty() {
        anyhow::bail!("Benchmark model has no valid vocabulary IDs");
    }
    if !json {
        println!("{}\nWarmup runs per row: {warmup}", engine.info());
    }
    let mut matrix = Vec::new();
    for prompt_len in prompt_lens {
        // Deterministic synthetic IDs, repeated within the model's valid vocabulary.
        let prompt: Vec<u32> = valid_ids.iter().copied().cycle().take(prompt_len).collect();
        for _ in 0..warmup {
            benchmark_sample(&mut engine, &prompt, max_tokens, 0)?;
        }
        let mut results = Vec::with_capacity(iterations);
        for iteration in 1..=iterations {
            let sample = benchmark_sample(&mut engine, &prompt, max_tokens, iteration)?;
            if !json {
                println!("Prompt {prompt_len}, iteration {iteration}: {} tokens, prefill {:.6}s, decode {:.6}s, {:.2} end-to-end tok/s",sample.generated_tokens,sample.prefill_seconds,sample.decode_seconds,sample.tokens_per_second);
            }
            results.push(sample);
        }
        let statistics = serde_json::json!({
            "elapsed_seconds":percentiles(results.iter().map(|r|r.elapsed_seconds).collect()),
            "prefill_seconds":percentiles(results.iter().map(|r|r.prefill_seconds).collect()),
            "decode_seconds":percentiles(results.iter().map(|r|r.decode_seconds).collect()),
            "tokens_per_second":percentiles(results.iter().map(|r|r.tokens_per_second).collect()),
            "decode_tokens_per_second":percentiles(results.iter().map(|r|r.decode_tokens_per_second).collect()),
        });
        let avg = results.iter().map(|r| r.tokens_per_second).sum::<f64>() / iterations as f64;
        if !json {
            println!("  latency p50/p95: {}", statistics["elapsed_seconds"]);
        }
        matrix.push(serde_json::json!({"prompt_len":prompt_len,"max_tokens":max_tokens,"average_tokens_per_second":avg,"statistics":statistics,"results":results}));
    }
    if json {
        let mut tensor_types = std::collections::BTreeMap::<String, usize>::new();
        if let Some(model) = &engine.model {
            for tensor in &model.metadata.tensors {
                *tensor_types
                    .entry(tensor.data_type.to_string())
                    .or_default() += 1;
            }
        }
        println!(
            "{}",
            serde_json::json!({
                "benchmark_schema_version":2,
                "model":model_path,"model_sha256":model_sha256,"model_load_seconds":model_load_seconds,
                "iterations":iterations,"warmup_iterations":warmup,"max_tokens":max_tokens,
                "backend":engine.backend.name(),"cpu_matvec_thread_limit":cpu_threads.map(|c|c.0),
                "cpu_matvec_kernel":cpu_threads.map(|c|c.1),
                "backend_features":engine.backend.capabilities().features,
                "kv_quant":engine.config.kv_quant.name(),
                "runtime_weight_storage":"f32","gguf_tensor_types":tensor_types,
                "sampling":{"temperature":0.0,"seed":42,"top_p":1.0,"top_k":0},
                "kernel_launches":engine.backend.kernel_launches(),"resident_weight_bytes":engine.backend.resident_weight_bytes(),
                "context_length":engine.config.max_seq_len,
                "build_profile":if cfg!(debug_assertions){"debug"}else{"release"},
                "input_kind":"synthetic-valid-token-ids","percentile_method":"nearest-rank",
                "timing_semantics":"prefill includes session allocation; decode includes sampling, EOS checks and generation forwards; end-to-end excludes model loading; model_load is measured after hashing so filesystem cache may be warm; first_token is the first visible step return",
                "matrix":matrix
            })
        );
    }
    Ok(())
}

fn emit_golden_logits(
    model_path: PathBuf,
    prompt: &str,
    top_k: usize,
    output: Option<&Path>,
) -> Result<()> {
    let mut engine = InferenceEngine::new(Some(InferenceConfig::default()));
    engine.load_model(&model_path)?;
    let tokenizer = engine
        .tokenizer
        .clone()
        .ok_or_else(|| anyhow::anyhow!("Tokenizer not loaded"))?;
    let tokens = tokenizer.try_encode(prompt).map_err(anyhow::Error::msg)?;
    let logits = engine
        .prefill(&tokens)
        .map_err(|err| anyhow::anyhow!("Prefill failed: {}", err))?;
    let mut top: Vec<(usize, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .collect();
    top.sort_by(|a, b| b.1.total_cmp(&a.1));
    top.truncate(top_k);

    let report = serde_json::json!({
        "model": model_path,
        "prompt": prompt,
        "prompt_tokens": tokens,
        "top_logits": top
            .into_iter()
            .map(|(token_id, logit)| serde_json::json!({
                "token_id": token_id,
                "logit": logit,
            }))
            .collect::<Vec<_>>(),
    });
    let report = serde_json::to_string_pretty(&report)?;
    if let Some(path) = output {
        fs::write(path, report)?;
    } else {
        println!("{}", report);
    }
    Ok(())
}

fn show_model_info(model_path: &std::path::Path) -> Result<()> {
    if model_path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
        return show_mlx_info(model_path);
    }

    let metadata = nexus::gguf::GgufReader::parse_file(model_path)?;

    println!("=== Model Information ===");
    println!("File: {}", model_path.display());
    println!("GGUF Version: {}", metadata.version);
    println!("Tensors: {}", metadata.tensor_count);
    println!();

    if let Some(name) = metadata.get_str("general.name") {
        println!("Name: {}", name);
    }
    if let Some(arch) = metadata.architecture() {
        println!("Architecture: {}", arch);
    }
    if let Some(ctx_len) = metadata.context_length() {
        println!("Context length: {}", ctx_len);
    }
    if let Some(n_layers) = metadata.num_layers() {
        println!("Layers: {}", n_layers);
    }
    if let Some(n_heads) = metadata.num_attention_heads() {
        println!("Attention heads: {}", n_heads);
    }
    if let Some(kv_heads) = metadata.num_key_value_heads() {
        println!("KV heads: {}", kv_heads);
    }
    if let Some(hidden) = metadata.hidden_size() {
        println!("Hidden size: {}", hidden);
    }
    if let Some(ff_dim) = metadata.feed_forward_size() {
        println!("FFN dimension: {}", ff_dim);
    }
    if let Some(vocab) = metadata.vocab_size() {
        println!("Vocabulary size: {}", vocab);
    }
    if let Some(freq_base) = metadata
        .metadata
        .get("llama.rope.freq_base")
        .or_else(|| metadata.metadata.get("mistral.rope.freq_base"))
    {
        if let Some(f) = freq_base.as_f32() {
            println!("Rope freq base: {}", f);
        }
    }

    println!();
    println!("=== Tensors ({}) ===", metadata.tensors.len());
    for tensor in metadata.tensors.iter().take(20) {
        let _shape: String = tensor
            .shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(" x ");
        println!(
            "  {:<50} {} [{}]",
            tensor.name,
            format_size(tensor.data_size()),
            tensor.data_type
        );
    }
    if metadata.tensors.len() > 20 {
        println!("  ... and {} more tensors", metadata.tensors.len() - 20);
    }

    Ok(())
}

fn show_mlx_info(model_path: &std::path::Path) -> Result<()> {
    let tensors = nexus::mlx::list_tensors(model_path)?;
    println!("=== MLX/Safetensors Model Information ===");
    println!("File: {}", model_path.display());
    println!("Tensors: {}", tensors.len());
    println!();
    println!("=== Tensors ({}) ===", tensors.len());
    for (name, shape, dtype) in tensors.iter().take(20) {
        let shape = shape
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(" x ");
        println!("  {:<50} {} [{}]", name, shape, dtype);
    }
    if tensors.len() > 20 {
        println!("  ... and {} more tensors", tensors.len() - 20);
    }
    Ok(())
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

async fn run_download(
    model: Option<&str>,
    output_dir: &Path,
    list: bool,
    all: Option<&[String]>,
) -> Result<()> {
    if list {
        download::list_models();
        return Ok(());
    }

    let output_path = output_dir.to_path_buf();
    fs::create_dir_all(&output_path).context("Failed to create models directory")?;

    if let Some(model_id) = model {
        // Try to parse as index first
        let models = download::available_models();
        if let Ok(idx) = model_id.parse::<usize>() {
            if idx > 0 && idx <= models.len() {
                let m = &models[idx - 1];
                println!("Downloading: {}", m.description);
                download::download_model(&m.repo_id, &m.filename, &output_path).await?;
                return Ok(());
            }
        }

        // Preserve the owner/repository pair and any nested filename path.
        if model_id.contains('/') {
            let parts: Vec<&str> = model_id.splitn(3, '/').collect();
            anyhow::ensure!(
                parts.len() == 3 && parts.iter().all(|part| !part.is_empty()),
                "Model identifier must be owner/repository/filename"
            );
            let repo_id = format!("{}/{}", parts[0], parts[1]);
            let path = download::download_model(&repo_id, parts[2], &output_path).await?;
            println!("Downloaded to: {}", path.display());
            return Ok(());
        }

        // Try as filename only - search in known repos
        let models = download::available_models();
        for m in &models {
            if m.filename == model_id || m.filename.contains(model_id) {
                println!("Found: {} -> {}", model_id, m.description);
                download::download_model(&m.repo_id, &m.filename, &output_path).await?;
                return Ok(());
            }
        }

        eprintln!(
            "Model '{}' not found. Use --list to see available models.",
            model_id
        );
        std::process::exit(1);
    } else if let Some(filters) = all {
        let filter_refs: Vec<&str> = filters.iter().map(|s| s.as_str()).collect();
        let paths = download::download_models(&output_path, &filter_refs).await?;
        println!(
            "\nDownloaded {} models to: {}",
            paths.len(),
            output_path.display()
        );
    } else {
        eprintln!("Specify a model with --model or use --list to see available models.");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  nexus download --list");
        eprintln!("  nexus download --model 1");
        eprintln!("  nexus download --model Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf");
        eprintln!("  nexus download --all q4_k,q5_k");
        std::process::exit(1);
    }

    Ok(())
}
