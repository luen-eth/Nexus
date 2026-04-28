//! Nexus API Server - OpenAI-compatible HTTP server.

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

use nexus::backend::BackendType;
use nexus::engine::{InferenceConfig, InferenceEngine};
use nexus::quant::{KvQuantType, QuantFormat};
use nexus::server::ServerConfig;

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

    /// Host to bind to
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to listen on
    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    /// Context length
    #[arg(long, default_value_t = 4096)]
    ctx_size: usize,

    /// KV cache quantization type
    #[arg(long, default_value = "tq_3b2b")]
    kv_quant: String,

    /// Weight quantization format
    #[arg(long, default_value = "Q4_K")]
    weight_quant: String,

    /// Backend to use
    #[arg(long, default_value = "auto")]
    backend: String,

    /// Maximum tensor elements to eagerly load; 0 means no cap.
    #[arg(long)]
    max_loaded_tensor_elements: Option<usize>,

    /// Optional API key. Also configurable through NEXUS_API_KEY.
    #[arg(long, env = "NEXUS_API_KEY")]
    api_key: Option<String>,

    /// CORS allow-origin value. Also configurable through NEXUS_CORS_ORIGIN.
    #[arg(long, env = "NEXUS_CORS_ORIGIN")]
    cors_origin: Option<String>,

    /// Per-minute request limit per forwarded IP. Also configurable through NEXUS_RATE_LIMIT_PER_MINUTE.
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

    let cli = Cli::parse();

    // Parse configuration
    let kv_quant_type = KvQuantType::from_str(&cli.kv_quant).unwrap_or(KvQuantType::TurboQuant3b2b);
    let weight_format = QuantFormat::from_str(&cli.weight_quant).unwrap_or(QuantFormat::Q4_K);

    let backend_type = match cli.backend.as_str() {
        "cpu" | "cpu-simd" => BackendType::CpuSimd,
        "metal" => BackendType::Metal,
        _ => BackendType::default(),
    };

    let config = InferenceConfig {
        max_seq_len: cli.ctx_size,
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
    let mut engine = InferenceEngine::new(Some(config));

    // Load model if specified
    if let Some(model_path) = &cli.model {
        println!("Loading model: {}", model_path.display());
        match engine.load_model(model_path) {
            Ok(()) => {
                println!("Model loaded successfully.");
                println!("{}", engine.info());
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
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
    nexus::server::run_server_with_config(engine, &cli.host, cli.port, server_config, cli.model)
        .await?;

    Ok(())
}
