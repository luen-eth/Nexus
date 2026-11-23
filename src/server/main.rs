//! Nexus API Server - OpenAI-compatible HTTP server.

use anyhow::{anyhow, Result};
use clap::Parser;
use std::path::PathBuf;

use nexus::engine::{InferenceConfig, InferenceEngine};
use nexus::quant::{KvQuantType, QuantFormat};
use nexus::server::{parse_backend, ServerConfig};

#[derive(Parser, Debug)]
#[command(
    name = "nexus-server",
    version,
    about = "Nexus API Server - OpenAI-compatible LLM server"
)]
struct Cli {
    /// Path to GGUF model file
    #[arg(short, long)]
    model: Option<PathBuf>,

    /// Host to bind to; use --host 0.0.0.0 to expose the API on other interfaces.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port to listen on
    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    /// Context length
    #[arg(long, default_value_t = 4096)]
    ctx_size: usize,

    /// KV cache quantization type; none keeps the reference float quality
    #[arg(long, default_value = "none")]
    kv_quant: String,

    /// Runtime weight storage (f32 only). GGUF file codecs are detected automatically.
    #[arg(long, default_value = "f32")]
    weight_quant: String,

    /// Backend: cpu, or cuda with --features cuda. "auto" selects cpu.
    #[arg(long, default_value = "auto")]
    backend: String,

    /// Maximum sequences decoded concurrently (token-interleaved on the CPU).
    /// Each active sequence owns its own KV cache. Excess requests queue up to
    /// 4x this value; beyond that the server answers 503.
    #[arg(long, default_value_t = 4)]
    batch_size: usize,
    /// Allow byte tokenization only for synthetic test fixtures.
    #[arg(long)]
    allow_byte_tokenizer: bool,

    /// Maximum tensor elements to eagerly load; 0 means no cap.
    #[arg(long)]
    max_loaded_tensor_elements: Option<usize>,

    /// Optional API key. Also configurable through NEXUS_API_KEY.
    #[arg(long, env = "NEXUS_API_KEY")]
    api_key: Option<String>,

    /// CORS allow-origin value. Also configurable through NEXUS_CORS_ORIGIN.
    #[arg(long, env = "NEXUS_CORS_ORIGIN")]
    cors_origin: Option<String>,

    /// Per-minute request limit per client IP (TCP peer address; forwarding
    /// headers are ignored). Also configurable through NEXUS_RATE_LIMIT_PER_MINUTE.
    #[arg(long, env = "NEXUS_RATE_LIMIT_PER_MINUTE")]
    rate_limit_per_minute: Option<usize>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "nexus=info".into()),
        )
        .init();

    let mut cli = Cli::parse();

    // Parse configuration
    let kv_quant_type = KvQuantType::from_str(&cli.kv_quant.to_ascii_lowercase().replace('-', "_"))
        .ok_or_else(|| anyhow!("Unsupported KV quantization: {}", cli.kv_quant))?;
    if kv_quant_type != KvQuantType::None {
        // Every recorded validation and benchmark ran with unquantized KV.
        // Make the quality trade-off visible instead of silent.
        eprintln!(
            "Warning: KV cache quantization '{}' is active; generated text differs from the unquantized reference.",
            cli.kv_quant
        );
    }
    let weight_name = cli.weight_quant.to_ascii_uppercase().replace('-', "_");
    let weight_format = QuantFormat::from_str(if weight_name == "F32" {
        "FP32"
    } else {
        &weight_name
    })
    .ok_or_else(|| anyhow!("Unsupported runtime weight storage: {}", cli.weight_quant))?;
    if weight_format != QuantFormat::FP32 {
        return Err(anyhow!(
            "Runtime weight storage supports only f32; GGUF file codecs are detected automatically"
        ));
    }
    let backend_type = parse_backend(&cli.backend).map_err(|err| anyhow!("{err}"))?;
    if cli.batch_size == 0 {
        return Err(anyhow!("--batch-size must be at least 1"));
    }
    // An empty --api-key usually means an unset environment variable in a
    // script; treating it as "authentication enabled with an empty secret"
    // would authorize any client that sends an empty key header. The same
    // holds for whitespace-only keys (pasted with a stray space), so reject
    // everything that is empty after trimming.
    if cli
        .api_key
        .as_ref()
        .is_some_and(|key| key.trim().is_empty())
    {
        return Err(anyhow!(
            "--api-key must not be empty or whitespace; unset it to serve without authentication"
        ));
    }
    // Store the trimmed secret: clients send the exact key, and a
    // trailing-space paste must not silently change what clients must send.
    cli.api_key = cli.api_key.map(|key| key.trim().to_string());

    let config = InferenceConfig {
        max_seq_len: cli.ctx_size,
        allow_byte_tokenizer: cli.allow_byte_tokenizer,
        batch_size: cli.batch_size,
        kv_quant: kv_quant_type,
        weight_format,
        backend: backend_type,
        max_loaded_tensor_elements: match cli.max_loaded_tensor_elements {
            Some(0) => None,
            Some(value) => Some(value),
            None => InferenceConfig::default().max_loaded_tensor_elements,
        },
        ..Default::default()
    };

    // Create engine
    let mut engine = InferenceEngine::new(Some(config.clone()));

    // Load model if specified (synchronous file I/O; keep it off the reactor).
    if let Some(model_path) = &cli.model {
        println!("Loading model: {}", model_path.display());
        let path = model_path.clone();
        let (loaded, result) = tokio::task::spawn_blocking(move || {
            let result = engine.load_model(&path);
            (engine, result)
        })
        .await?;
        match result {
            Ok(()) => {
                engine = loaded;
                println!("Model loaded successfully.");
                println!("{}", engine.info());
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                engine = InferenceEngine::new(Some(config.clone()));
                eprintln!("Model file not found: {}", model_path.display());
                eprintln!();
                eprintln!("Starting server without model. Load a model with --model <path>");
            }
            Err(e) => return Err(e.into()),
        }
    } else {
        println!("No model specified. Starting server without model.");
        println!("Load a model with --model <path>");
    }

    let server_config = ServerConfig {
        api_key: cli.api_key,
        cors_origin: cli.cors_origin,
        rate_limit_per_minute: cli.rate_limit_per_minute,
    };

    // Start server
    nexus::server::run_server_with_requested_config(
        engine,
        &cli.host,
        cli.port,
        server_config,
        cli.model,
        config,
    )
    .await?;

    Ok(())
}
