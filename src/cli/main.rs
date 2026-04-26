//! Nexus CLI tool - command-line interface for model inference.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use nexus::backend::BackendType;
use nexus::download;
use nexus::engine::{GenerationConfig, InferenceConfig, InferenceEngine};
use nexus::quant::{KvQuantType, QuantFormat};
use nexus::tokenizer::{RuntimeTokenizer, Tokenizer};

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
        #[arg(short, long, default_value = "128")]
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

        /// KV cache quantization type
        #[arg(long, default_value = "tq_3b2b")]
        kv_quant: String,

        /// Weight quantization format
        #[arg(long, default_value = "Q4_K")]
        weight_quant: String,

        /// Backend to use (cpu-simd, metal)
        #[arg(long, default_value = "auto")]
        backend: String,

        /// Number of threads for CPU backend
        #[arg(long)]
        threads: Option<usize>,
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

        /// Emit machine-readable JSON
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

        /// Quantization format
        #[arg(long, default_value = "q4-0")]
        quant: String,
    },

    /// Download a model from HuggingFace
    Download {
        /// Model identifier (repo_id/filename or index from list)
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
    tracing_subscriber::fmt()
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
            )
            .await?;
        }
        Some(Commands::Bench {
            model,
            iterations,
            prompt_len,
            prompt_lens,
            max_tokens,
            json,
        }) => {
            let prompt_lens = prompt_lens.unwrap_or_else(|| vec![prompt_len]);
            run_benchmark(model, iterations, prompt_lens, max_tokens, json).await?;
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
    _threads: Option<usize>,
) -> Result<()> {
    // Parse configuration
    let kv_quant_type = KvQuantType::from_str(kv_quant).unwrap_or(KvQuantType::TurboQuant3b2b);
    let weight_format = QuantFormat::from_str(weight_quant).unwrap_or(QuantFormat::Q4_K);

    let backend_type = match backend {
        "cpu" | "cpu-simd" => BackendType::CpuSimd,
        "metal" => BackendType::Metal,
        "cuda" | "vulkan" | "webgpu" => {
            anyhow::bail!(
                "Backend '{}' is listed in capabilities but not implemented yet; use cpu-simd or metal",
                backend
            );
        }
        _ => BackendType::default(),
    };

    let config = InferenceConfig {
        max_seq_len: ctx_size,
        kv_quant: kv_quant_type,
        weight_format,
        backend: backend_type,
        ..Default::default()
    };

    // Create engine
    let mut engine = InferenceEngine::new(Some(config));

    println!("Loading model: {}", model_path.display());
    println!("Backend: {}", engine.backend.name());
    println!("KV Quant: {}", kv_quant_type.name());
    println!("Weight Quant: {}", weight_format.name());
    println!();

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
    if let Some(prompt_text) = prompt {
        println!("Prompt: {}", prompt_text);
        println!("---");

        let tokenizer =
            RuntimeTokenizer::from_metadata(engine.model.as_ref().map(|model| model.metadata()));
        generation.stop_sequences = stop
            .iter()
            .map(|sequence| tokenizer.encode(sequence))
            .collect();
        let tokens = tokenizer.encode(&prompt_text);

        match engine.generate_with_config(&tokens, &generation) {
            Ok(output) => {
                let generated = tokenizer.decode(&output[tokens.len()..]);
                println!("{}", generated);
            }
            Err(e) => eprintln!("Error during generation: {}", e),
        }
    } else {
        println!("No prompt provided. Use -p 'text' to generate text.");
    }

    // Print memory report
    if let Some(model) = &engine.model {
        model.memory.report();
    }

    Ok(())
}

async fn run_benchmark(
    model_path: PathBuf,
    iterations: usize,
    prompt_lens: Vec<usize>,
    max_tokens: usize,
    json: bool,
) -> Result<()> {
    if !json {
        println!("Running benchmark with {} iterations...", iterations);
        println!("Model: {}", model_path.display());
        println!();
    }

    let config = InferenceConfig::default();
    let mut engine = InferenceEngine::new(Some(config));

    match engine.load_model(&model_path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Model file not found: {}", model_path.display());
            std::process::exit(1);
        }
        Err(e) => return Err(e.into()),
    }

    if !json {
        println!("{}", engine.info());
        println!();
    }

    let mut matrix = Vec::new();
    for prompt_len in prompt_lens {
        let test_tokens: Vec<u32> = (0..prompt_len).map(|i| i as u32).collect();
        let mut results = Vec::new();

        if !json {
            println!("Prompt length: {}", prompt_len);
        }
        for i in 0..iterations {
            let start = std::time::Instant::now();
            match engine.generate(&test_tokens, max_tokens) {
                Ok(_) => {
                    let elapsed = start.elapsed();
                    let tokens_per_sec = max_tokens as f64 / elapsed.as_secs_f64();
                    results.push(serde_json::json!({
                        "iteration": i + 1,
                        "elapsed_seconds": elapsed.as_secs_f64(),
                        "tokens_per_second": tokens_per_sec,
                    }));
                    if !json {
                        println!(
                            "  Iteration {}: {:.2} tok/s ({:.3}s)",
                            i + 1,
                            tokens_per_sec,
                            elapsed.as_secs_f64()
                        );
                    }
                }
                Err(e) => {
                    if json {
                        results.push(serde_json::json!({
                            "iteration": i + 1,
                            "error": e,
                        }));
                    } else {
                        eprintln!("  Iteration {} failed: {}", i + 1, e);
                    }
                }
            }
        }
        let successful: Vec<f64> = results
            .iter()
            .filter_map(|result| result.get("tokens_per_second"))
            .filter_map(serde_json::Value::as_f64)
            .collect();
        let average_tokens_per_second = if successful.is_empty() {
            0.0
        } else {
            successful.iter().sum::<f64>() / successful.len() as f64
        };
        matrix.push(serde_json::json!({
            "prompt_len": prompt_len,
            "max_tokens": max_tokens,
            "average_tokens_per_second": average_tokens_per_second,
            "results": results,
        }));
    }

    if json {
        println!(
            "{}",
            serde_json::json!({
                "model": model_path,
                "iterations": iterations,
                "max_tokens": max_tokens,
                "backend": engine.backend.name(),
                "matrix": matrix,
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
    let tokenizer =
        RuntimeTokenizer::from_metadata(engine.model.as_ref().map(|model| model.metadata()));
    let tokens = tokenizer.encode(prompt);
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
    if let Some(freq_base) = metadata.metadata.get("llama.rope.freq_base") {
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

    let output_path = download::models_dir(output_dir);
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

        // Try as repo_id/filename format
        if model_id.contains('/') {
            let parts: Vec<&str> = model_id.splitn(2, '/').collect();
            if parts.len() == 2 {
                let path = download::download_model(parts[0], parts[1], &output_path).await?;
                println!("Downloaded to: {}", path.display());
                return Ok(());
            }
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
