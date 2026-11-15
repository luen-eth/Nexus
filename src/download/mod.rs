//! Model downloader - fetches models from HuggingFace Hub.
//! Supports GGUF and MLX model formats.

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, Result};
use futures_util::TryStreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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

/// Download a model file from HuggingFace.
/// Only fully written files with a matching local integrity receipt are cached.
pub async fn download_model(repo_id: &str, filename: &str, output_dir: &Path) -> Result<PathBuf> {
    validate_relative_path(filename)?;
    anyhow::ensure!(
        repo_id.split('/').count() == 2
            && repo_id
                .split('/')
                .all(|part| !part.is_empty() && part != "." && part != ".."),
        "Repository ID must be owner/repository"
    );
    let mut url = reqwest::Url::parse("https://huggingface.co/")?;
    url.path_segments_mut()
        .map_err(|_| anyhow::anyhow!("Invalid download base URL"))?
        .extend(repo_id.split('/'))
        .extend(["resolve", "main"])
        .extend(filename.split('/'));
    let client = reqwest::Client::builder()
        .user_agent("nexus-inference/0.1.0")
        .timeout(std::time::Duration::from_secs(3600))
        .build()
        .context("Failed to create HTTP client")?;
    println!("Downloading: {repo_id}/{filename}");
    download_from_url(&client, url.as_str(), filename, output_dir).await
}

fn validate_relative_path(filename: &str) -> Result<()> {
    anyhow::ensure!(
        !filename.is_empty()
            && Path::new(filename)
                .components()
                .all(|component| matches!(component, Component::Normal(_)))
            && !filename.contains('\\'),
        "Download filename must be a relative path without parent components"
    );
    Ok(())
}

#[derive(Serialize, Deserialize)]
struct DownloadReceipt {
    version: u32,
    url: String,
    bytes: u64,
    sha256: String,
}

fn receipt_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".nexus-download.json");
    PathBuf::from(name)
}

fn cache_is_valid(path: &Path, url: &str) -> bool {
    let check = || -> Result<bool> {
        // A receipt is small. Do not read an arbitrarily large local file into memory.
        let receipt_file = fs::File::open(receipt_path(path))?;
        anyhow::ensure!(
            receipt_file.metadata()?.len() <= 16 * 1024,
            "Oversized receipt"
        );
        let receipt: DownloadReceipt = serde_json::from_reader(receipt_file)?;
        if receipt.version != 1 || receipt.url != url || receipt.bytes == 0 {
            return Ok(false);
        }
        let mut file = fs::File::open(path)?;
        if !file.metadata()?.is_file() || file.metadata()?.len() != receipt.bytes {
            return Ok(false);
        }
        let mut hash = Sha256::new();
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let len = file.read(&mut buffer)?;
            if len == 0 {
                break;
            }
            hash.update(&buffer[..len]);
        }
        Ok(format!("{:x}", hash.finalize()) == receipt.sha256)
    };
    check().unwrap_or(false)
}

/// Removes unfinished files on HTTP/write errors and task cancellation.
struct PendingFile {
    path: PathBuf,
    file: Option<fs::File>,
}

impl PendingFile {
    fn new(destination: &Path) -> Result<Self> {
        let parent = destination.parent().context("Missing download directory")?;
        let path = parent.join(format!(".nexus-{}.part", uuid::Uuid::new_v4()));
        let file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .context("Failed to create temporary download")?;
        Ok(Self {
            path,
            file: Some(file),
        })
    }

    fn write_all(&mut self, data: &[u8]) -> Result<()> {
        self.file
            .as_mut()
            .context("Temporary file is closed")?
            .write_all(data)?;
        Ok(())
    }

    fn persist(mut self, destination: &Path) -> Result<()> {
        if let Some(file) = self.file.take() {
            file.sync_all().context("Failed to sync download")?;
            drop(file);
        }
        fs::rename(&self.path, destination).context("Failed to publish download atomically")?;
        Ok(())
    }
}

impl Drop for PendingFile {
    fn drop(&mut self) {
        self.file.take();
        let _ = fs::remove_file(&self.path);
    }
}

async fn download_from_url(
    client: &reqwest::Client,
    url: &str,
    filename: &str,
    output_dir: &Path,
) -> Result<PathBuf> {
    validate_relative_path(filename)?;
    let output_path = output_dir.join(filename);
    let cache_path = output_path.clone();
    let cache_url = url.to_owned();
    if tokio::task::spawn_blocking(move || cache_is_valid(&cache_path, &cache_url)).await? {
        println!("Verified cached file: {}", output_path.display());
        return Ok(output_path);
    }
    fs::create_dir_all(output_path.parent().context("Missing output directory")?)?;
    let response = client
        .get(url)
        .send()
        .await
        .context("Model download request failed")?;
    anyhow::ensure!(
        response.status() == reqwest::StatusCode::OK,
        "Download failed: HTTP {}",
        response.status()
    );
    let total_size = response.content_length();
    let mut file = PendingFile::new(&output_path)?;
    let mut hash = Sha256::new();
    let mut downloaded = 0u64;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream
        .try_next()
        .await
        .context("Model download stream interrupted")?
    {
        file.write_all(&chunk)
            .context("Failed to write download chunk")?;
        hash.update(&chunk);
        downloaded = downloaded
            .checked_add(chunk.len() as u64)
            .context("Download size overflow")?;
        if let Some(total) = total_size.filter(|total| *total > 0) {
            print!(
                "\r  Progress: {:.1}% ({:.2} MB / {:.2} MB)",
                downloaded as f64 / total as f64 * 100.0,
                downloaded as f64 / 1_048_576.0,
                total as f64 / 1_048_576.0
            );
            io::stdout().flush()?;
        }
    }
    anyhow::ensure!(downloaded > 0, "Downloaded model is empty");
    if let Some(expected) = total_size {
        anyhow::ensure!(
            downloaded == expected,
            "Incomplete model download: expected {expected} bytes, received {downloaded}"
        );
    }
    let receipt = DownloadReceipt {
        version: 1,
        url: url.to_owned(),
        bytes: downloaded,
        sha256: format!("{:x}", hash.finalize()),
    };
    let metadata_path = receipt_path(&output_path);
    let mut metadata = PendingFile::new(&metadata_path)?;
    metadata.write_all(&serde_json::to_vec(&receipt)?)?;
    file.persist(&output_path)?;
    metadata.persist(&metadata_path)?;
    println!(
        "\nDownloaded: {filename} ({:.2} MB)",
        downloaded as f64 / 1_048_576.0
    );
    Ok(output_path)
}

/// Download all available models (or a subset)
pub async fn download_models(output_dir: &Path, filters: &[&str]) -> Result<Vec<PathBuf>> {
    let models = available_models();
    let mut downloaded = Vec::new();
    let mut failures = Vec::new();

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
                failures.push(format!("{}/{}: {e:#}", model.repo_id, model.filename));
            }
        }
    }

    println!("\n=== Download Summary ===");
    println!("Successfully downloaded: {} models", downloaded.len());
    for path in &downloaded {
        let size = fs::metadata(path)?.len();
        println!("  {} ({:.2} MB)", path.display(), size as f64 / 1_048_576.0);
    }

    anyhow::ensure!(
        failures.is_empty(),
        "{} model download(s) failed: {}",
        failures.len(),
        failures.join("; ")
    );
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn http_fixture(responses: Vec<Vec<u8>>) -> (String, std::thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/model", listener.local_addr().unwrap());
        let task = std::thread::spawn(move || {
            for response in responses {
                let (mut socket, _) = listener.accept().unwrap();
                socket
                    .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                    .unwrap();
                let mut header = Vec::new();
                let mut byte = [0u8; 1];
                while !header.ends_with(b"\r\n\r\n") {
                    socket.read_exact(&mut byte).unwrap();
                    header.push(byte[0]);
                }
                socket.write_all(&response).unwrap();
            }
        });
        (url, task)
    }

    fn response(body: &[u8]) -> Vec<u8> {
        let mut data = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        data.extend_from_slice(body);
        data
    }

    fn assert_no_parts(dir: &Path) {
        assert!(fs::read_dir(dir).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".part")
        }));
    }

    #[tokio::test]
    async fn complete_download_has_verified_offline_cache() {
        let dir = tempfile::tempdir().unwrap();
        let (url, server) = http_fixture(vec![response(b"model bytes")]);
        let client = reqwest::Client::new();
        let path = download_from_url(&client, &url, "nested/model.gguf", dir.path())
            .await
            .unwrap();
        server.join().unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"model bytes");
        assert!(cache_is_valid(&path, &url));
        // The listener is closed: a second success proves no HTTP request is needed.
        assert_eq!(
            download_from_url(&client, &url, "nested/model.gguf", dir.path())
                .await
                .unwrap(),
            path
        );
        assert!(!cache_is_valid(&path, "http://different-source/model"));
        assert_no_parts(path.parent().unwrap());
    }

    #[tokio::test]
    async fn truncated_download_errors_and_never_publishes_partial_file() {
        let dir = tempfile::tempdir().unwrap();
        let (url, server) = http_fixture(vec![
            b"HTTP/1.1 200 OK\r\nContent-Length: 99\r\nConnection: close\r\n\r\nshort".to_vec(),
        ]);
        let result =
            download_from_url(&reqwest::Client::new(), &url, "model.gguf", dir.path()).await;
        server.join().unwrap();
        assert!(result.is_err());
        assert!(!dir.path().join("model.gguf").exists());
        assert!(!receipt_path(&dir.path().join("model.gguf")).exists());
        assert_no_parts(dir.path());
    }

    #[tokio::test]
    async fn failed_replacement_preserves_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        fs::write(&path, b"existing unchecked model").unwrap();
        let (url, server) = http_fixture(vec![
            b"HTTP/1.1 200 OK\r\nContent-Length: 99\r\nConnection: close\r\n\r\nshort".to_vec(),
        ]);
        assert!(
            download_from_url(&reqwest::Client::new(), &url, "model.gguf", dir.path())
                .await
                .is_err()
        );
        server.join().unwrap();
        assert_eq!(fs::read(path).unwrap(), b"existing unchecked model");
        assert_no_parts(dir.path());
    }

    #[tokio::test]
    async fn same_size_corruption_and_legacy_cache_are_downloaded_again() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        fs::write(&path, b"legacy data").unwrap();
        let (url, server) = http_fixture(vec![response(b"valid model"), response(b"valid model")]);
        let client = reqwest::Client::new();
        download_from_url(&client, &url, "model.gguf", dir.path())
            .await
            .unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"valid model");
        fs::write(&path, b"badxx model").unwrap();
        assert!(!cache_is_valid(&path, &url));
        download_from_url(&client, &url, "model.gguf", dir.path())
            .await
            .unwrap();
        server.join().unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"valid model");
        assert_no_parts(dir.path());
    }

    #[tokio::test]
    async fn chunked_without_length_is_verified_and_empty_body_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let (url, server) = http_fixture(vec![b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n5\r\nmodel\r\n0\r\n\r\n".to_vec(), response(b"")]);
        let client = reqwest::Client::new();
        let path = download_from_url(&client, &url, "model.gguf", dir.path())
            .await
            .unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"model");
        assert!(cache_is_valid(&path, &url));
        assert!(download_from_url(&client, &url, "empty.gguf", dir.path())
            .await
            .is_err());
        server.join().unwrap();
        assert!(!dir.path().join("empty.gguf").exists());
        assert_no_parts(dir.path());
    }

    #[tokio::test]
    async fn cancelled_download_cleans_up_temporary_file() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let dir = tempfile::tempdir().unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/model", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut header = Vec::new();
            let mut byte = [0u8; 1];
            while !header.ends_with(b"\r\n\r\n") {
                socket.read_exact(&mut byte).await.unwrap();
                header.push(byte[0]);
            }
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 999\r\n\r\npartial")
                .await
                .unwrap();
            std::future::pending::<()>().await;
        });
        let directory = dir.path().to_owned();
        let download = tokio::spawn(async move {
            download_from_url(&reqwest::Client::new(), &url, "model.gguf", &directory).await
        });
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            loop {
                if fs::read_dir(dir.path()).unwrap().any(|entry| {
                    entry
                        .unwrap()
                        .file_name()
                        .to_string_lossy()
                        .ends_with(".part")
                }) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        download.abort();
        assert!(download.await.unwrap_err().is_cancelled());
        server.abort();
        assert!(!dir.path().join("model.gguf").exists());
        assert_no_parts(dir.path());
    }

    #[test]
    fn rejects_absolute_and_traversing_paths() {
        for path in ["", "../model", "/model", "x/../../model", "x\\model"] {
            assert!(validate_relative_path(path).is_err(), "{path}");
        }
        assert!(validate_relative_path("nested/model.gguf").is_ok());
    }
}
