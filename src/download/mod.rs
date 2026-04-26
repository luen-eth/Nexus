//! Model downloader - fetches models from HuggingFace Hub.
//! Supports GGUF and MLX model formats.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use futures_util::TryStreamExt;

/// Available model repositories for download
pub struct ModelRepo {
    pub repo_id: String,
    pub filename: String,
    pub description: String,
}

impl ModelRepo {
    pub fn new(repo_id: &str, filename: &str, description: &str) -> Self {
        ModelRepo {
            repo_id: repo_id.to_string(),
            filename: filename.to_string(),
            description: description.to_string(),
        }
    }
}

/// Predefined small models (~1B parameters) for testing
pub fn available_models() -> Vec<ModelRepo> {
    vec![
        ModelRepo::new(
            "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            "Qwen 2.5 1.5B - Q4_K_M quantization",
        ),
        ModelRepo::new(
            "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            "qwen2.5-1.5b-instruct-q5_k_m.gguf",
            "Qwen 2.5 1.5B - Q5_K_M quantization",
        ),
        ModelRepo::new(
            "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            "qwen2.5-1.5b-instruct-q6_k.gguf",
            "Qwen 2.5 1.5B - Q6_K quantization",
        ),
        ModelRepo::new(
            "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            "qwen2.5-1.5b-instruct-q8_0.gguf",
            "Qwen 2.5 1.5B - Q8_0 quantization",
        ),
        ModelRepo::new(
            "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            "qwen2.5-1.5b-instruct-q2_k.gguf",
            "Qwen 2.5 1.5B - Q2_K quantization (smallest)",
        ),
        ModelRepo::new(
            "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            "qwen2.5-1.5b-instruct-q3_k_m.gguf",
            "Qwen 2.5 1.5B - Q3_K_M quantization",
        ),
        ModelRepo::new(
            "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            "qwen2.5-1.5b-instruct-q3_k_s.gguf",
            "Qwen 2.5 1.5B - Q3_K_S quantization",
        ),
        ModelRepo::new(
            "bartowski/Qwen2.5-1.5B-Instruct-GGUF",
            "qwen2.5-1.5b-instruct-iq2_xs.gguf",
            "Qwen 2.5 1.5B - IQ2_XS (extreme quantization)",
        ),
        ModelRepo::new(
            "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            "model.safetensors",
            "Qwen 2.5 1.5B - MLX 4-bit quantized (safetensors)",
        ),
        ModelRepo::new(
            "mlx-community/Qwen2.5-1.5B-Instruct-8bit",
            "model.safetensors",
            "Qwen 2.5 1.5B - MLX 8-bit quantized (safetensors)",
        ),
    ]
}

/// Download a model file from HuggingFace
pub async fn download_model(repo_id: &str, filename: &str, output_dir: &Path) -> Result<PathBuf> {
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo_id, filename
    );

    println!("Downloading: {}", filename);
    println!("From: {}/{}", repo_id, filename);
    println!();

    // Create output directory if it doesn't exist
    fs::create_dir_all(output_dir).context("Failed to create output directory")?;

    let output_path = output_dir.join(filename);

    // Check if file already exists
    if output_path.exists() {
        let size = fs::metadata(&output_path)?.len();
        println!(
            "Already exists: {} ({:.2} MB)",
            filename,
            size as f64 / 1_048_576.0
        );
        return Ok(output_path);
    }

    // Download the file
    let client = reqwest::Client::builder()
        .user_agent("nexus-inference/0.1.0")
        .timeout(std::time::Duration::from_secs(3600))
        .build()
        .context("Failed to create HTTP client")?;

    let response = client
        .get(&url)
        .send()
        .await
        .context(format!("Failed to download from {}", url))?;

    if !response.status().is_success() {
        anyhow::bail!(
            "Download failed: HTTP {} for {}/{}",
            response.status(),
            repo_id,
            filename
        );
    }

    let total_size = response.content_length().unwrap_or(0);
    let mut file = fs::File::create(&output_path).context("Failed to create output file")?;
    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();

    while let Ok(Some(chunk)) = stream.try_next().await {
        let bytes = chunk;
        file.write_all(&bytes)
            .context("Failed to write download chunk")?;
        downloaded += bytes.len() as u64;

        if total_size > 0 {
            let percent = (downloaded as f64 / total_size as f64) * 100.0;
            let downloaded_mb = downloaded as f64 / 1_048_576.0;
            let total_mb = total_size as f64 / 1_048_576.0;
            print!(
                "\r  Progress: {:.1}% ({:.2} MB / {:.2} MB)",
                percent, downloaded_mb, total_mb
            );
            io::stdout().flush()?;
        } else {
            let downloaded_mb = downloaded as f64 / 1_048_576.0;
            print!("\r  Downloaded: {:.2} MB", downloaded_mb);
            io::stdout().flush()?;
        }
    }

    println!();
    let final_size = output_path.metadata()?.len();
    println!(
        "Downloaded: {} ({:.2} MB)",
        filename,
        final_size as f64 / 1_048_576.0
    );

    Ok(output_path)
}

/// Download all available models (or a subset)
pub async fn download_models(output_dir: &Path, filters: &[&str]) -> Result<Vec<PathBuf>> {
    let models = available_models();
    let mut downloaded = Vec::new();

    for model in &models {
        // Filter if specified
        if !filters.is_empty() {
            let matches = filters
                .iter()
                .any(|f| model.filename.contains(f) || model.repo_id.contains(f));
            if !matches {
                continue;
            }
        }

        println!("\n=== {} ===", model.description);
        match download_model(&model.repo_id, &model.filename, output_dir).await {
            Ok(path) => downloaded.push(path),
            Err(e) => {
                eprintln!("  Failed: {}", e);
            }
        }
    }

    println!("\n=== Download Summary ===");
    println!("Successfully downloaded: {} models", downloaded.len());
    for path in &downloaded {
        let size = fs::metadata(path)?.len();
        println!("  {} ({:.2} MB)", path.display(), size as f64 / 1_048_576.0);
    }

    Ok(downloaded)
}

/// List available models
pub fn list_models() {
    let models = available_models();
    println!("Available models for download:\n");

    for (i, model) in models.iter().enumerate() {
        println!("  {}. {}", i + 1, model.description);
        println!("     {} -> {}", model.repo_id, model.filename);
        println!();
    }
}

/// Get the default models directory
pub fn models_dir(base_dir: &Path) -> PathBuf {
    base_dir.join("models")
}
