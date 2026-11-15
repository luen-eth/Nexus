//! MLX model loader - supports MLX safetensors format.
//! MLX models use the safetensors format for weight storage.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use serde::de::{self, MapAccess, Visitor};
use serde::{Deserialize, Deserializer};

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
        self.shape.iter().copied().fold(1, usize::saturating_mul)
    }

    pub fn elem_size(&self) -> usize {
        match self.dtype.as_str() {
            "F32" => 4,
            "F16" | "BF16" => 2,
            "I64" => 8,
            "I32" => 4,
            "I16" => 2,
            "F64" | "U64" => 8,
            "U32" => 4,
            "U16" => 2,
            "I8" | "U8" | "BOOL" => 1,
            _ => 0, // Unsupported dtypes must never be guessed as F32.
        }
    }

    pub fn data_size(&self) -> usize {
        self.num_elements().saturating_mul(self.elem_size())
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

// Matches the safetensors format header limit. Check the actual file boundary
// before reserving this memory; a forged length in an eight-byte file allocates nothing.
const MAX_HEADER_BYTES: u64 = 100_000_000;
const MAX_CONFIG_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTensorMeta {
    dtype: String,
    shape: Vec<u64>,
    data_offsets: [u64; 2],
}

// Deserialize entries directly rather than via Value/HashMap, which silently
// accept duplicate tensor keys (or duplicate dtype/shape/offset fields).
struct RawHeader(HashMap<String, RawTensorMeta>);
impl<'de> Deserialize<'de> for RawHeader {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct HeaderVisitor;
        impl<'de> Visitor<'de> for HeaderVisitor {
            type Value = RawHeader;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a safetensors header object")
            }
            fn visit_map<M: MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                let mut tensors = HashMap::new();
                let mut seen_metadata = false;
                while let Some(name) = map.next_key::<String>()? {
                    if name == "__metadata__" {
                        if seen_metadata {
                            return Err(de::Error::custom("Duplicate __metadata__ entry"));
                        }
                        let _: HashMap<String, String> = map.next_value()?;
                        seen_metadata = true;
                    } else {
                        if tensors.contains_key(&name) {
                            return Err(de::Error::custom(format!("Duplicate tensor: {name}")));
                        }
                        tensors.insert(name, map.next_value::<RawTensorMeta>()?);
                    }
                }
                Ok(RawHeader(tensors))
            }
        }
        deserializer.deserialize_map(HeaderVisitor)
    }
}

/// Parse and validate all tensor ranges before any tensor payload is allocated.
pub fn parse_safetensors_header(path: &Path) -> Result<SafeTensorsHeader> {
    let file = File::open(path).context("Failed to open safetensors file")?;
    let file_len = file.metadata()?.len();
    let mut reader = io::BufReader::new(file);
    let mut size_buf = [0u8; 8];
    reader
        .read_exact(&mut size_buf)
        .context("Failed to read header size")?;
    let header_size = u64::from_le_bytes(size_buf);
    let data_base = header_size
        .checked_add(8)
        .context("Safetensors header offset overflow")?;
    if header_size == 0 || header_size > MAX_HEADER_BYTES || data_base > file_len {
        anyhow::bail!("Invalid header size: {header_size} (file bytes: {file_len}, budget: {MAX_HEADER_BYTES})");
    }
    let header_len = usize::try_from(header_size).context("Header size exceeds address space")?;
    let mut header_bytes = Vec::new();
    header_bytes
        .try_reserve_exact(header_len)
        .context("Unable to allocate safetensors header")?;
    header_bytes.resize(header_len, 0);
    reader
        .read_exact(&mut header_bytes)
        .context("Failed to read header bytes")?;
    if header_bytes.first() != Some(&b'{') {
        anyhow::bail!("Safetensors header must start with a JSON object");
    }
    let RawHeader(raw) =
        serde_json::from_slice(&header_bytes).context("Failed to parse safetensors JSON")?;
    let payload_len = file_len - data_base;
    let mut tensor_names: Vec<_> = raw.keys().cloned().collect();
    tensor_names.sort();
    let mut tensors = HashMap::new();
    let mut ranges = Vec::new();
    for (name, raw) in raw {
        let shape = raw
            .shape
            .into_iter()
            .map(usize::try_from)
            .collect::<std::result::Result<Vec<_>, _>>()
            .with_context(|| format!("Shape exceeds address space for tensor {name}"))?;
        let [start, end] = raw.data_offsets;
        let span = end
            .checked_sub(start)
            .with_context(|| format!("Reversed data offsets for tensor {name}"))?;
        if end > payload_len {
            anyhow::bail!("Tensor {name} extends beyond safetensors payload");
        }
        let data_size = usize::try_from(span).context("Tensor size exceeds address space")?;
        let meta = TensorMeta {
            dtype: raw.dtype,
            shape,
            data_offset: data_base
                .checked_add(start)
                .context("Tensor offset overflow")?,
            data_size,
        };
        let elem_size = meta.elem_size();
        if elem_size == 0 {
            anyhow::bail!("Unsupported dtype: {} for tensor {name}", meta.dtype);
        }
        let elements = checked_elements(&meta.shape)
            .with_context(|| format!("Tensor {name} shape overflow"))?;
        let expected = elements
            .checked_mul(elem_size)
            .with_context(|| format!("Tensor {name} byte size overflow"))?;
        if expected != data_size {
            anyhow::bail!(
                "Tensor {name} shape/dtype requires {expected} bytes, offsets describe {data_size}"
            );
        }
        ranges.push((start, end, name.clone()));
        tensors.insert(name, meta);
    }
    // The format requires an entirely indexed, non-overlapping data buffer.
    // Empty tensors are legal; sort them before a nonempty tensor at the same start.
    ranges.sort_unstable();
    let mut cursor = 0;
    for (start, end, name) in ranges {
        if start != cursor {
            anyhow::bail!("Tensor {name} overlaps another tensor or leaves a payload gap");
        }
        cursor = end;
    }
    if cursor != payload_len {
        anyhow::bail!("Unindexed trailing safetensors payload bytes");
    }
    Ok(SafeTensorsHeader {
        tensor_names,
        tensors,
        header_size: data_base,
    })
}

fn checked_elements(shape: &[usize]) -> Result<usize> {
    // Zero-sized dimensions describe an empty tensor, even if other dimensions
    // would overflow when multiplied before encountering the zero.
    if shape.contains(&0) {
        return Ok(0);
    }
    shape.iter().try_fold(1usize, |n, &dim| {
        n.checked_mul(dim).context("Tensor element count overflow")
    })
}

/// Load tensor data from a safetensors file using endian-explicit, aligned-independent reads.
pub fn load_tensor(path: &Path, tensor_name: &str) -> Result<Vec<f32>> {
    let header = parse_safetensors_header(path)?;
    let meta = header
        .tensors
        .get(tensor_name)
        .ok_or_else(|| anyhow::anyhow!("Tensor '{tensor_name}' not found in {}", path.display()))?;
    load_tensor_data(path, tensor_name, meta)
}

fn load_tensor_data(path: &Path, name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
    let elem_size = meta.elem_size();
    if elem_size == 0 {
        anyhow::bail!("Unsupported dtype: {} for tensor {name}", meta.dtype);
    }
    let elements = checked_elements(&meta.shape)?;
    if elements.checked_mul(elem_size) != Some(meta.data_size) {
        anyhow::bail!("Invalid tensor byte length: {name}");
    }
    if !matches!(meta.dtype.as_str(), "F32" | "F16" | "BF16" | "I32" | "I64") {
        anyhow::bail!(
            "Unsupported dtype for f32 loading: {} for tensor {name}",
            meta.dtype
        );
    }
    let file = File::open(path).context("Failed to open tensor data file")?;
    let file_len = file.metadata()?.len();
    if meta
        .data_offset
        .checked_add(meta.data_size as u64)
        .is_none_or(|end| end > file_len)
    {
        anyhow::bail!("Tensor {name} extends beyond safetensors file");
    }
    let mut reader = io::BufReader::new(file);
    reader
        .seek(SeekFrom::Start(meta.data_offset))
        .context("Failed to seek to tensor data")?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .context("Unable to allocate tensor output")?;
    // Avoid retaining a second full-size raw payload alongside the f32 output.
    let mut bytes = [0u8; 64 * 1024];
    let mut remaining = meta.data_size;
    while remaining > 0 {
        let count = remaining.min(bytes.len());
        reader
            .read_exact(&mut bytes[..count])
            .with_context(|| format!("Failed to read tensor {name}"))?;
        for chunk in bytes[..count].chunks_exact(elem_size) {
            let value = match meta.dtype.as_str() {
                "F32" => f32::from_le_bytes(chunk.try_into()?),
                "F16" => half::f16::from_bits(u16::from_le_bytes(chunk.try_into()?)).to_f32(),
                "BF16" => f32::from_bits((u16::from_le_bytes(chunk.try_into()?) as u32) << 16),
                "I64" => i64::from_le_bytes(chunk.try_into()?) as f32,
                "I32" => i32::from_le_bytes(chunk.try_into()?) as f32,
                _ => unreachable!("dtype validated above"),
            };
            output.push(value);
        }
        remaining -= count;
    }
    Ok(output)
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
            header_size = header_size
                .checked_add(header.header_size)
                .context("Shard header size overflow")?;
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

        // Absence is allowed for low-level tensor inspection; present-but-invalid
        // configuration must not silently turn conversion into an anonymous model.
        let config_path = model_dir.join("config.json");
        let config =
            match File::open(&config_path) {
                Ok(file) => {
                    if file.metadata()?.len() > MAX_CONFIG_BYTES {
                        anyhow::bail!("Model config exceeds {MAX_CONFIG_BYTES} byte budget");
                    }
                    let mut bytes = Vec::new();
                    file.take(MAX_CONFIG_BYTES + 1)
                        .read_to_end(&mut bytes)
                        .context("Failed to read model config")?;
                    if bytes.len() as u64 > MAX_CONFIG_BYTES {
                        anyhow::bail!("Model config exceeds byte budget");
                    }
                    let json = std::str::from_utf8(&bytes).context("Model config is not UTF-8")?;
                    Some(MlxModelConfig::from_json(json).with_context(|| {
                        format!("Invalid model config: {}", config_path.display())
                    })?)
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => None,
                Err(error) => return Err(error).context("Failed to open model config"),
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
        let meta = self
            .header
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Missing tensor metadata: {name}"))?;
        load_tensor_data(path, name, meta)
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
