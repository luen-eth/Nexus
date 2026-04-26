//! MLX model loader - supports MLX safetensors format.
//! MLX models use the safetensors format for weight storage.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use serde::Deserialize;
use serde_json::Value;

// ============================================================================
// Safetensors Header Format
// ============================================================================

/// Safetensors file header (JSON + tensor metadata)
#[derive(Debug, Clone)]
pub struct SafeTensorsHeader {
    /// Tensor names in order
    pub tensor_names: Vec<String>,
    /// Tensor metadata: name -> (shape, dtype, data_offset, data_size)
    pub tensors: HashMap<String, TensorMeta>,
    /// Total header size in bytes
    pub header_size: u64,
}

/// Metadata for a single tensor in safetensors format
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offset: u64,
    pub data_size: usize,
}

impl TensorMeta {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn elem_size(&self) -> usize {
        match self.dtype.as_str() {
            "F32" => 4,
            "F16" | "BF16" => 2,
            "I64" => 8,
            "I32" => 4,
            "I16" => 2,
            "I8" => 1,
            "U8" => 1,
            _ => 4, // default to F32 size
        }
    }

    pub fn data_size(&self) -> usize {
        self.num_elements() * self.elem_size()
    }
}

// ============================================================================
// MLX Model Info
// ============================================================================

/// MLX model metadata extracted from config.json
#[derive(Debug, Clone, Deserialize)]
pub struct MlxModelConfig {
    pub model_type: Option<String>,
    pub hidden_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub vocab_size: Option<usize>,
    pub max_position_embeddings: Option<usize>,
    pub rms_norm_eps: Option<f32>,
    pub rope_theta: Option<f32>,
}

impl MlxModelConfig {
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).context("Failed to parse model config")
    }
}

// ============================================================================
// Safetensors Parser
// ============================================================================

/// Parse a safetensors file header
pub fn parse_safetensors_header(path: &Path) -> Result<SafeTensorsHeader> {
    let file = File::open(path).context("Failed to open safetensors file")?;
    let mut reader = io::BufReader::new(file);

    // Read the header size (8 bytes, little-endian u64)
    let mut size_buf = [0u8; 8];
    reader
        .read_exact(&mut size_buf)
        .context("Failed to read header size")?;
    let header_size = u64::from_le_bytes(size_buf);

    if header_size == 0 || header_size > 100_000_000 {
        anyhow::bail!(
            "Invalid header size: {} (possible corrupted file)",
            header_size
        );
    }

    // Read the header JSON
    let mut header_bytes = vec![0u8; header_size as usize];
    reader
        .read_exact(&mut header_bytes)
        .context("Failed to read header bytes")?;

    // Parse JSON - skip the __metadata__ entry if present
    let json_str = std::str::from_utf8(&header_bytes).context("Header is not valid UTF-8")?;

    // Parse the tensor metadata. Safetensors stores each tensor as:
    // { "dtype": "BF16", "shape": [..], "data_offsets": [start, end] }.
    let tensors: HashMap<String, Value> =
        serde_json::from_str(json_str).context("Failed to parse safetensors JSON")?;

    // Extract tensor names and metadata
    let mut tensor_names = Vec::new();
    let mut tensor_map = HashMap::new();
    let data_base = 8u64 + header_size;

    // Sort tensor names for consistent ordering
    let mut sorted_keys: Vec<String> = tensors.keys().cloned().collect();
    sorted_keys.sort();

    for name in &sorted_keys {
        if name == "__metadata__" {
            continue;
        }
        let entry = &tensors[name];
        tensor_names.push(name.clone());

        let dtype = entry
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("Missing dtype for tensor {}", name))?
            .to_string();
        let shape: Vec<usize> = entry
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("Missing shape for tensor {}", name))?
            .iter()
            .map(|v| v.as_u64().map(|n| n as usize))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| anyhow::anyhow!("Invalid shape for tensor {}", name))?;
        let offsets = entry
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow::anyhow!("Missing data_offsets for tensor {}", name))?;
        if offsets.len() != 2 {
            anyhow::bail!("Invalid data_offsets for tensor {}", name);
        }
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("Invalid start offset for tensor {}", name))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("Invalid end offset for tensor {}", name))?;
        let data_size = (end - start) as usize;

        tensor_map.insert(
            name.clone(),
            TensorMeta {
                dtype,
                shape,
                data_offset: data_base + start,
                data_size,
            },
        );
    }

    Ok(SafeTensorsHeader {
        tensor_names,
        tensors: tensor_map,
        header_size: 8 + header_size,
    })
}

/// Load tensor data from a safetensors file
pub fn load_tensor(path: &Path, tensor_name: &str) -> Result<Vec<f32>> {
    let header = parse_safetensors_header(path)?;
    let meta = header.tensors.get(tensor_name).ok_or_else(|| {
        anyhow::anyhow!("Tensor '{}' not found in {}", tensor_name, path.display())
    })?;

    let file = File::open(path).context("Failed to open file")?;
    let mut reader = io::BufReader::new(file);

    // Seek to tensor data offset
    reader
        .seek(SeekFrom::Start(meta.data_offset))
        .context("Failed to seek to tensor data")?;

    // Read tensor data
    let mut data = vec![0u8; meta.data_size];
    reader
        .read_exact(&mut data)
        .context("Failed to read tensor data")?;

    // Convert to f32 based on dtype
    match meta.dtype.as_str() {
        "F32" => {
            let f32_data: &[f32] = bytemuck::cast_slice(&data);
            Ok(f32_data.to_vec())
        }
        "F16" => {
            // Convert FP16 to FP32
            let f16_data: &[u16] = bytemuck::cast_slice(&data);
            let f32_data: Vec<f32> = f16_data
                .iter()
                .map(|&bits| half::f16::from_bits(bits).to_f32())
                .collect();
            Ok(f32_data)
        }
        "BF16" => {
            // BF16 has same bit representation as top 16 bits of FP32
            let bf16_data: &[u16] = bytemuck::cast_slice(&data);
            let f32_data: Vec<f32> = bf16_data
                .iter()
                .map(|&bits| f32::from_bits((bits as u32) << 16))
                .collect();
            Ok(f32_data)
        }
        "I32" => {
            let i32_data: &[i32] = bytemuck::cast_slice(&data);
            Ok(i32_data.iter().map(|&x| x as f32).collect())
        }
        "I64" => {
            let i64_data: &[i64] = bytemuck::cast_slice(&data);
            Ok(i64_data.iter().map(|&x| x as f32).collect())
        }
        _ => {
            anyhow::bail!(
                "Unsupported dtype: {} for tensor {}",
                meta.dtype,
                tensor_name
            );
        }
    }
}

/// List all tensors in a safetensors file
pub fn list_tensors(path: &Path) -> Result<Vec<(String, Vec<usize>, String)>> {
    let header = parse_safetensors_header(path)?;
    let mut tensors = Vec::new();
    for name in &header.tensor_names {
        if let Some(meta) = header.tensors.get(name) {
            tensors.push((name.clone(), meta.shape.clone(), meta.dtype.clone()));
        }
    }
    Ok(tensors)
}

// ============================================================================
// MLX Model Loader
// ============================================================================

/// MLX model loader - loads weights from safetensors files
pub struct MlxModelLoader {
    pub config: Option<MlxModelConfig>,
    pub header: SafeTensorsHeader,
    pub weight_path: PathBuf,
    tensor_files: HashMap<String, PathBuf>,
}

impl MlxModelLoader {
    /// Create a new MLX model loader
    pub fn new(model_dir: &Path) -> Result<Self> {
        let weight_paths = Self::find_safetensors(model_dir)?;
        let weight_path = weight_paths[0].clone();

        let mut tensor_names = Vec::new();
        let mut tensors = HashMap::new();
        let mut tensor_files = HashMap::new();
        let mut header_size = 0u64;
        for path in &weight_paths {
            let header = parse_safetensors_header(path)?;
            header_size += header.header_size;
            for name in header.tensor_names {
                if tensors.contains_key(&name) {
                    anyhow::bail!("Duplicate tensor {} across safetensors shards", name);
                }
                let meta = header
                    .tensors
                    .get(&name)
                    .ok_or_else(|| anyhow::anyhow!("Missing metadata for tensor {}", name))?
                    .clone();
                tensor_files.insert(name.clone(), path.clone());
                tensors.insert(name.clone(), meta);
                tensor_names.push(name);
            }
        }
        tensor_names.sort();
        let header = SafeTensorsHeader {
            tensor_names,
            tensors,
            header_size,
        };

        // Try to load config
        let config = if let Ok(config_str) = std::fs::read_to_string(model_dir.join("config.json"))
        {
            MlxModelConfig::from_json(&config_str).ok()
        } else {
            None
        };

        Ok(MlxModelLoader {
            config,
            header,
            weight_path,
            tensor_files,
        })
    }

    /// Find safetensors files in directory
    fn find_safetensors(dir: &Path) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "safetensors" {
                    files.push(path);
                }
            }
        }
        files.sort();
        if files.is_empty() {
            anyhow::bail!("No safetensors file found in {}", dir.display());
        }
        Ok(files)
    }

    /// Load a specific tensor
    pub fn load_tensor(&self, name: &str) -> Result<Vec<f32>> {
        let path = self
            .tensor_files
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Tensor '{}' not found", name))?;
        load_tensor(path, name)
    }

    /// Get all tensor names
    pub fn tensor_names(&self) -> &[String] {
        &self.header.tensor_names
    }

    /// Get tensor metadata
    pub fn tensor_meta(&self, name: &str) -> Option<&TensorMeta> {
        self.header.tensors.get(name)
    }

    /// Get model info summary
    pub fn info(&self) -> String {
        let total_tensors = self.header.tensor_names.len();
        let total_size: usize = self.header.tensors.values().map(|t| t.data_size).sum();

        format!(
            "MLX Model: {} tensors, {}\nPath: {}",
            total_tensors,
            format_size(total_size),
            self.weight_path.display()
        )
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_f32_safetensors(path: &Path, name: &str, shape: &[usize], values: &[f32]) {
        let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = serde_json::json!({
            name: {
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [0, data.len()]
            }
        })
        .to_string();
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        file.write_all(&data).unwrap();
    }

    #[test]
    fn test_tensor_meta_calculation() {
        let meta = TensorMeta {
            dtype: "F32".to_string(),
            shape: vec![100, 200],
            data_offset: 1024,
            data_size: 80000,
        };
        assert_eq!(meta.num_elements(), 20000);
        assert_eq!(meta.elem_size(), 4);
        assert_eq!(meta.data_size(), 80000);
    }

    #[test]
    fn test_tensor_meta_f16() {
        let meta = TensorMeta {
            dtype: "F16".to_string(),
            shape: vec![100, 200],
            data_offset: 0,
            data_size: 40000,
        };
        assert_eq!(meta.elem_size(), 2);
        assert_eq!(meta.data_size(), 40000);
    }

    #[test]
    fn test_format_size() {
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1024 * 5), "5.00 KB");
        assert_eq!(format_size(1024 * 1024 * 10), "10.00 MB");
        assert_eq!(format_size(1024 * 1024 * 1024 * 2), "2.00 GB");
    }

    #[test]
    fn test_loader_reads_multiple_safetensors_shards() {
        let dir = tempfile::tempdir().unwrap();
        write_f32_safetensors(
            &dir.path().join("a.safetensors"),
            "a.weight",
            &[2],
            &[1.0, 2.0],
        );
        write_f32_safetensors(
            &dir.path().join("b.safetensors"),
            "b.weight",
            &[2],
            &[3.0, 4.0],
        );

        let loader = MlxModelLoader::new(dir.path()).unwrap();
        assert_eq!(loader.tensor_names().len(), 2);
        assert_eq!(loader.load_tensor("a.weight").unwrap(), vec![1.0, 2.0]);
        assert_eq!(loader.load_tensor("b.weight").unwrap(), vec![3.0, 4.0]);
    }
}
