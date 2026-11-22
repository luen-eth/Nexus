//! Model conversion tool - convert safetensors/MLX models to GGUF format.
//! Supports adding Nexus TurboQuant metadata for KV cache compression.

#![allow(non_camel_case_types)]

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use nexus::gguf::{GgufDataType, GgufReader};
use nexus::mlx::MlxModelLoader;
use nexus::quant::{KvQuantType, QuantEngine, QuantFormat as NexusQuantFormat};

const GGUF_VERSION: u32 = 3;
const ALIGNMENT: usize = 32;

#[derive(Parser, Debug)]
#[command(
    name = "nexus-convert",
    about = "Convert models to GGUF format for Nexus"
)]
struct Cli {
    /// Source model directory containing config.json and safetensors files
    #[arg(short, long)]
    input: PathBuf,

    /// Output GGUF file path
    #[arg(short, long)]
    output: PathBuf,

    /// Quantization format. The default is a faithful F32 copy; lossy codecs
    /// must be requested explicitly, as in llama.cpp converters.
    #[arg(long, default_value = "f32")]
    quant: QuantFormat,

    /// Add TurboQuant KV cache metadata
    #[arg(long)]
    turboquant: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum QuantFormat {
    F32,
    F16,
    Q8_0,
    Q4_K,
    Q4_0,
    Q2_K,
}

#[derive(Debug)]
struct ConvertedTensor {
    name: String,
    shape: Vec<usize>,
    gguf_type: u32,
    data: Vec<u8>,
    offset: u64,
}

#[derive(Debug, Default)]
struct QuantizationSummary {
    tensors: usize,
    values: usize,
    total_abs_error: f64,
    max_abs_error: f32,
}

impl QuantizationSummary {
    fn record(&mut self, original: &[f32], reconstructed: &[f32]) {
        self.tensors += 1;
        for (expected, actual) in original.iter().zip(reconstructed) {
            let error = (expected - actual).abs();
            self.total_abs_error += error as f64;
            self.max_abs_error = self.max_abs_error.max(error);
            self.values += 1;
        }
    }

    fn mean_abs_error(&self) -> f64 {
        if self.values == 0 {
            0.0
        } else {
            self.total_abs_error / self.values as f64
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    println!("Converting model:");
    println!("  Input:  {}", cli.input.display());
    println!("  Output: {}", cli.output.display());
    println!("  Quant:  {:?}", cli.quant);
    println!(
        "  TurboQuant: {}",
        if cli.turboquant { "yes" } else { "no" }
    );
    println!();

    if !cli.input.exists() {
        anyhow::bail!("Input path does not exist: {}", cli.input.display());
    }
    if cli.input.is_file() {
        anyhow::bail!(
            "Input must be a model directory containing config.json and .safetensors files"
        );
    }

    let loader = MlxModelLoader::new(&cli.input).with_context(|| {
        format!(
            "Failed to load safetensors model from {}",
            cli.input.display()
        )
    })?;
    validate_model_config(&loader)?;
    let requested_format = cli.quant.to_nexus_quant()?;

    let mut tensors = Vec::new();
    let mut names = HashSet::new();
    let mut quantization = QuantizationSummary::default();
    let mut offset = 0u64;
    for name in loader.tensor_names() {
        let meta = loader
            .tensor_meta(name)
            .with_context(|| format!("Missing metadata for tensor {}", name))?;
        let values = loader
            .load_tensor(name)
            .with_context(|| format!("Failed to load tensor {}", name))?;
        if values.len() != meta.num_elements() {
            anyhow::bail!(
                "Tensor {} shape/data mismatch: metadata says {} elements, loaded {}",
                name,
                meta.num_elements(),
                values.len()
            );
        }

        let values = permute_llama_qk(name, &meta.shape, values, loader.config.as_ref())?;
        let canonical_name = canonical_tensor_name(name);
        if !names.insert(canonical_name.clone()) {
            anyhow::bail!(
                "Tensor name collision after canonicalization: {} -> {}",
                name,
                canonical_name
            );
        }

        let format = tensor_quant_format(&meta.shape, requested_format);
        let quant_engine = QuantEngine::new(format, KvQuantType::None);
        let gguf_type = weight_data_type(format)?.id();
        let data = quant_engine.quantize_weights(&values)?;
        let reconstructed = quant_engine.dequantize(&data, values.len())?;
        if reconstructed.len() != values.len() {
            anyhow::bail!(
                "Tensor {} quantization roundtrip length mismatch: got {}, expected {}",
                name,
                reconstructed.len(),
                values.len()
            );
        }
        quantization.record(&values, &reconstructed);
        let aligned_size = align(data.len(), ALIGNMENT);

        tensors.push(ConvertedTensor {
            name: canonical_name,
            shape: meta.shape.clone(),
            gguf_type,
            data,
            offset,
        });
        offset = offset
            .checked_add(aligned_size as u64)
            .context("Converted tensor offsets overflow")?;
    }

    let metadata = build_metadata(&loader, &cli.input, cli.turboquant)?;
    write_gguf(&cli.output, &metadata, &tensors)?;
    validate_written_gguf(&cli.output, tensors.len())?;
    println!("Wrote GGUF: {}", cli.output.display());
    println!("Tensors: {}", tensors.len());
    println!("Validation: ok (GGUF parsed, tensor table matches conversion output)");
    println!(
        "Quantization error: mean_abs={:.6}, max_abs={:.6} over {} values",
        quantization.mean_abs_error(),
        quantization.max_abs_error,
        quantization.values
    );

    Ok(())
}

/// The current runtime and HF transforms are implemented for dense LLaMA only.
fn validate_model_config(loader: &MlxModelLoader) -> Result<()> {
    let config = loader
        .config
        .as_ref()
        .context("A valid config.json is required for model conversion")?;
    // Dense Mistral-7B graphs are identical to LLaMA, so both convert through
    // the same path. Anything else stays an explicit rejection.
    let accepted_architectures = ["llama", "mistral"];
    let model_type = config
        .model_type
        .as_deref()
        .context("config.json is missing model_type")?;
    if !accepted_architectures.contains(&model_type) {
        anyhow::bail!(
            "Unsupported model architecture {model_type:?}; only dense llama and mistral conversion are implemented"
        );
    }
    // The engine implements unscaled RoPE only. Dropping a `rope_scaling`
    // declaration silently would produce a file that loads fine and then
    // generates wrong long-context output, so refuse the conversion instead.
    if let Some(scaling) = &config.rope_scaling {
        let declares_none = ["rope_type", "type"].iter().any(|key| {
            scaling
                .get(key)
                .and_then(serde_json::Value::as_str)
                .is_some_and(|value| value.eq_ignore_ascii_case("none"))
        });
        if !declares_none {
            anyhow::bail!(
                "config.json declares rope_scaling ({scaling}); scaled-RoPE conversion is not implemented. Convert an unscaled checkpoint"
            );
        }
    }
    for (name, value) in [
        ("hidden_size", config.hidden_size),
        ("num_hidden_layers", config.num_hidden_layers),
        ("num_attention_heads", config.num_attention_heads),
        ("num_key_value_heads", config.num_key_value_heads),
        ("intermediate_size", config.intermediate_size),
        ("vocab_size", config.vocab_size),
        ("max_position_embeddings", config.max_position_embeddings),
    ] {
        if value.is_some_and(|n| n > u32::MAX as usize) {
            anyhow::bail!("{name} exceeds GGUF U32 metadata range");
        }
    }
    let hidden = config
        .hidden_size
        .filter(|n| *n > 0)
        .context("Missing/zero hidden_size")?;
    let heads = config
        .num_attention_heads
        .filter(|n| *n > 0)
        .context("Missing/zero num_attention_heads")?;
    let kv = config.num_key_value_heads.unwrap_or(heads);
    config
        .num_hidden_layers
        .context("Missing num_hidden_layers")?;
    config
        .intermediate_size
        .filter(|n| *n > 0)
        .context("Missing/zero intermediate_size")?;
    config
        .vocab_size
        .filter(|n| *n > 0)
        .context("Missing/zero vocab_size")?;
    config
        .max_position_embeddings
        .filter(|n| *n > 0)
        .context("Missing/zero max_position_embeddings")?;
    if kv == 0
        || !hidden.is_multiple_of(heads)
        || !heads.is_multiple_of(kv)
        || !(hidden / heads).is_multiple_of(2)
    {
        anyhow::bail!("Invalid LLaMA attention dimensions");
    }
    if config.rope_theta.is_some_and(|v| !v.is_finite() || v <= 0.)
        || config
            .rms_norm_eps
            .is_some_and(|v| !v.is_finite() || v <= 0.)
    {
        anyhow::bail!("Invalid LLaMA rope_theta/rms_norm_eps");
    }
    Ok(())
}

fn weight_data_type(format: NexusQuantFormat) -> Result<GgufDataType> {
    Ok(match format {
        NexusQuantFormat::FP32 => GgufDataType::F32,
        NexusQuantFormat::FP16 => GgufDataType::F16,
        NexusQuantFormat::Q8_0 => GgufDataType::Q8_0,
        NexusQuantFormat::Q4_0 => GgufDataType::Q4_0,
        NexusQuantFormat::Q4_K => GgufDataType::Q4_K,
        NexusQuantFormat::Q2_K => GgufDataType::Q2_K,
        other => anyhow::bail!("Unsupported standard weight codec {other}"),
    })
}

/// Quantize matrices only when each row fits complete codec blocks; preserve norms/biases.
fn tensor_quant_format(shape: &[usize], requested: NexusQuantFormat) -> NexusQuantFormat {
    let ty = match weight_data_type(requested) {
        Ok(ty) => ty,
        Err(_) => return requested,
    };
    let (block, _) = ty.block_layout();
    if block > 1
        && (shape.len() < 2
            || shape
                .last()
                .is_none_or(|n| *n == 0 || !n.is_multiple_of(block)))
    {
        NexusQuantFormat::FP32
    } else {
        requested
    }
}

struct PendingOutput(PathBuf);
impl Drop for PendingOutput {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Convert HF split-half Q/K row order to LLaMA GGUF adjacent-pair RoPE order.
fn permute_llama_qk(
    name: &str,
    shape: &[usize],
    values: Vec<f32>,
    config: Option<&nexus::mlx::MlxModelConfig>,
) -> Result<Vec<f32>> {
    let is_q = name.ends_with(".q_proj.weight");
    let is_k = name.ends_with(".k_proj.weight");
    if !is_q && !is_k {
        return Ok(values);
    }
    let config = config.context("Q/K conversion requires model config")?;
    // HF Mistral uses the same split-half RoPE pairing as LLaMA
    // (`apply_rotary_pos_emb` + `rotate_half`), so both labels need the
    // identical permute; skipping it would silently rotate wrong pairs.
    if !matches!(config.model_type.as_deref(), Some("llama" | "mistral")) {
        return Ok(values);
    }
    let heads = config
        .num_attention_heads
        .context("Missing num_attention_heads")?;
    let hidden = config.hidden_size.context("Missing hidden_size")?;
    let kv_heads = config.num_key_value_heads.unwrap_or(heads);
    if heads == 0
        || hidden == 0
        || !hidden.is_multiple_of(heads)
        || kv_heads == 0
        || heads % kv_heads != 0
    {
        anyhow::bail!("Invalid llama/mistral head dimensions");
    }
    let dim = hidden / heads;
    let rows_heads = if is_q { heads } else { kv_heads };
    if dim % 2 != 0 || shape.len() != 2 || shape[0] != rows_heads * dim || shape[1] != hidden {
        anyhow::bail!("Invalid llama/mistral Q/K shape for {name}: {shape:?}");
    }
    let cols = shape[1];
    let mut output = vec![0.; values.len()];
    for head in 0..rows_heads {
        for pair in 0..dim / 2 {
            for half in 0..2 {
                let src = (head * dim + half * (dim / 2) + pair) * cols;
                let dst = (head * dim + pair * 2 + half) * cols;
                output[dst..dst + cols].copy_from_slice(&values[src..src + cols]);
            }
        }
    }
    Ok(output)
}

fn canonical_tensor_name(name: &str) -> String {
    if name == "model.embed_tokens.weight" {
        return "token_embd.weight".to_string();
    }
    if name == "model.norm.weight" {
        return "output_norm.weight".to_string();
    }
    if name == "lm_head.weight" {
        return "output.weight".to_string();
    }

    if let Some(rest) = name.strip_prefix("model.layers.") {
        let mut parts = rest.splitn(2, '.');
        if let (Some(layer), Some(suffix)) = (parts.next(), parts.next()) {
            let suffix = match suffix {
                "input_layernorm.weight" => "attn_norm.weight",
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                other => other,
            };
            return format!("blk.{}.{}", layer, suffix);
        }
    }

    name.to_string()
}

impl QuantFormat {
    fn to_nexus_quant(self) -> Result<NexusQuantFormat> {
        match self {
            QuantFormat::F32 => Ok(NexusQuantFormat::FP32),
            QuantFormat::F16 => Ok(NexusQuantFormat::FP16),
            QuantFormat::Q8_0 => Ok(NexusQuantFormat::Q8_0),
            QuantFormat::Q4_0 => Ok(NexusQuantFormat::Q4_0),
            QuantFormat::Q4_K => Ok(NexusQuantFormat::Q4_K),
            QuantFormat::Q2_K => Ok(NexusQuantFormat::Q2_K),
        }
    }

    #[cfg(test)]
    fn gguf_type(self) -> Result<u32> {
        Ok(weight_data_type(self.to_nexus_quant()?)?.id())
    }
}

fn build_metadata(
    loader: &MlxModelLoader,
    input: &Path,
    turboquant: bool,
) -> Result<HashMap<String, GgufValue>> {
    validate_model_config(loader)?;
    let mut metadata = HashMap::new();
    metadata.insert(
        "general.alignment".to_string(),
        GgufValue::U32(ALIGNMENT as u32),
    );
    metadata.insert(
        "general.name".to_string(),
        GgufValue::String(
            input
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("nexus-model")
                .to_string(),
        ),
    );

    if let Some(config) = &loader.config {
        if let Some(model_type) = &config.model_type {
            metadata.insert(
                "general.architecture".to_string(),
                GgufValue::String(model_type.clone()),
            );
        }
        // Accepted architectures are validated as llama or mistral; keep the
        // metadata keys consistent with the declared architecture so the
        // engine resolves dimensions through the matching prefix.
        let prefix = if config.model_type.as_deref() == Some("mistral") {
            "mistral"
        } else {
            "llama"
        };
        insert_usize(
            &mut metadata,
            &format!("{prefix}.embedding_length"),
            config.hidden_size,
        );
        insert_usize(
            &mut metadata,
            &format!("{prefix}.block_count"),
            config.num_hidden_layers,
        );
        insert_usize(
            &mut metadata,
            &format!("{prefix}.attention.head_count"),
            config.num_attention_heads,
        );
        insert_usize(
            &mut metadata,
            &format!("{prefix}.attention.head_count_kv"),
            config.num_key_value_heads,
        );
        insert_usize(
            &mut metadata,
            &format!("{prefix}.feed_forward_length"),
            config.intermediate_size,
        );
        insert_usize(
            &mut metadata,
            &format!("{prefix}.vocab_size"),
            config.vocab_size,
        );
        insert_usize(
            &mut metadata,
            &format!("{prefix}.context_length"),
            config.max_position_embeddings,
        );
        if let Some(eps) = config.rms_norm_eps {
            metadata.insert(
                format!("{prefix}.attention.layer_norm_rms_epsilon"),
                GgufValue::F32(eps),
            );
        }
        if let Some(theta) = config.rope_theta {
            metadata.insert(format!("{prefix}.rope.freq_base"), GgufValue::F32(theta));
        }
    }

    if !metadata.contains_key("general.architecture") {
        metadata.insert(
            "general.architecture".to_string(),
            GgufValue::String("llama".to_string()),
        );
    }

    if turboquant {
        metadata.insert(
            "nexus.kv_quant_type".to_string(),
            GgufValue::String("tq_3b2b".to_string()),
        );
        metadata.insert("nexus.turboquant".to_string(), GgufValue::Bool(true));
    }

    insert_tokenizer_metadata(&mut metadata, input)?;

    Ok(metadata)
}

/// Token-id ceiling for tokenizer.json imports. Real vocabularies stay far
/// below this; without a bound, `tokens.resize(id + 1, …)` with a hostile or
/// corrupt id panics on overflow (debug) or aborts the process on allocation
/// failure (release) instead of returning a conversion error.
const MAX_TOKEN_ID: u64 = 1 << 24;

fn checked_token_id(id: u64) -> Result<usize> {
    if id >= MAX_TOKEN_ID {
        anyhow::bail!(
            "token id {id} exceeds the conversion limit {MAX_TOKEN_ID}; tokenizer.json is malformed"
        );
    }
    usize::try_from(id).context("token id does not fit the platform")
}

fn insert_usize(metadata: &mut HashMap<String, GgufValue>, key: &str, value: Option<usize>) {
    if let Some(value) = value {
        metadata.insert(key.to_string(), GgufValue::U32(value as u32));
    }
}

#[derive(Debug)]
enum GgufValue {
    Bool(bool),
    U32(u32),
    I32(i32),
    F32(f32),
    String(String),
    StringVec(Vec<String>),
}

fn write_gguf(
    path: &Path,
    metadata: &HashMap<String, GgufValue>,
    tensors: &[ConvertedTensor],
) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .context("GGUF output needs a filename")?
        .to_string_lossy();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    let pending = PendingOutput(parent.join(format!(".{name}.{}.{nonce}.tmp", std::process::id())));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&pending.0)
        .with_context(|| format!("Failed to create staged GGUF {}", pending.0.display()))?;
    let mut writer = BufWriter::new(file);

    writer.write_all(b"GGUF")?;
    write_u32(&mut writer, GGUF_VERSION)?;
    write_u64(&mut writer, tensors.len() as u64)?;
    write_u64(&mut writer, metadata.len() as u64)?;

    let mut metadata_entries: Vec<_> = metadata.iter().collect();
    metadata_entries.sort_by(|a, b| a.0.cmp(b.0));
    for (key, value) in metadata_entries {
        write_string(&mut writer, key)?;
        write_value(&mut writer, value)?;
    }

    for tensor in tensors {
        write_string(&mut writer, &tensor.name)?;
        write_u32(&mut writer, tensor.shape.len() as u32)?;
        // Safetensors dimensions are [rows, columns]; GGUF stores columns first.
        for dim in tensor.shape.iter().rev() {
            write_u64(&mut writer, *dim as u64)?;
        }
        write_u32(&mut writer, tensor.gguf_type)?;
        write_u64(&mut writer, tensor.offset)?;
    }

    pad_to_alignment(&mut writer, ALIGNMENT)?;
    for tensor in tensors {
        writer.write_all(&tensor.data)?;
        pad_to_alignment(&mut writer, ALIGNMENT)?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    drop(writer);
    validate_written_gguf(&pending.0, tensors.len())?;
    std::fs::rename(&pending.0, path)
        .with_context(|| format!("Failed to publish {}", path.display()))?;
    Ok(())
}

fn validate_written_gguf(path: &Path, expected_tensors: usize) -> Result<()> {
    let metadata = GgufReader::parse_file(path)
        .with_context(|| format!("Failed to validate written GGUF {}", path.display()))?;
    if metadata.version != GGUF_VERSION {
        anyhow::bail!(
            "Written GGUF version mismatch: got {}, expected {}",
            metadata.version,
            GGUF_VERSION
        );
    }
    if metadata.tensors.len() != expected_tensors
        || metadata.tensor_count as usize != expected_tensors
    {
        anyhow::bail!(
            "Written GGUF tensor count mismatch: table={}, header={}, expected={}",
            metadata.tensors.len(),
            metadata.tensor_count,
            expected_tensors
        );
    }
    for tensor in &metadata.tensors {
        if tensor.shape.is_empty() || tensor.num_elements() == 0 {
            anyhow::bail!("Written GGUF contains empty tensor shape: {}", tensor.name);
        }
    }
    Ok(())
}

fn write_value(writer: &mut dyn Write, value: &GgufValue) -> Result<()> {
    match value {
        GgufValue::Bool(value) => {
            write_u32(writer, 7)?;
            writer.write_all(&[*value as u8])?;
        }
        GgufValue::U32(value) => {
            write_u32(writer, 4)?;
            write_u32(writer, *value)?;
        }
        GgufValue::I32(value) => {
            write_u32(writer, 5)?;
            writer.write_all(&value.to_le_bytes())?;
        }
        GgufValue::F32(value) => {
            write_u32(writer, 6)?;
            writer.write_all(&value.to_le_bytes())?;
        }
        GgufValue::String(value) => {
            write_u32(writer, 8)?;
            write_string(writer, value)?;
        }
        GgufValue::StringVec(values) => {
            write_u32(writer, 9)?;
            write_u32(writer, 8)?;
            write_u64(writer, values.len() as u64)?;
            for value in values {
                write_string(writer, value)?;
            }
        }
    }
    Ok(())
}

fn insert_tokenizer_metadata(
    metadata: &mut HashMap<String, GgufValue>,
    input: &Path,
) -> Result<()> {
    let tokenizer_path = input.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Ok(());
    }

    let tokenizer: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&tokenizer_path)
            .with_context(|| format!("Failed to read {}", tokenizer_path.display()))?,
    )
    .with_context(|| format!("Failed to parse {}", tokenizer_path.display()))?;
    metadata.insert(
        "nexus.tokenizer.json".into(),
        GgufValue::String(std::fs::read_to_string(&tokenizer_path)?),
    );
    for (filename, key) in [
        ("tokenizer_config.json", "nexus.tokenizer.config"),
        ("generation_config.json", "nexus.generation.config"),
        ("config.json", "nexus.model.config"),
    ] {
        let path = input.join(filename);
        if path.exists() {
            metadata.insert(
                key.into(),
                GgufValue::String(std::fs::read_to_string(path)?),
            );
        }
    }
    let model = tokenizer
        .get("model")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("tokenizer.json is missing model object"))?;

    if let Some(model_type) = model.get("type").and_then(serde_json::Value::as_str) {
        metadata.insert(
            "nexus.tokenizer.model".to_string(),
            GgufValue::String(model_type.to_ascii_lowercase()),
        );
    }

    let mut tokens = Vec::new();
    let mut token_to_id = HashMap::new();
    if let Some(vocab) = model.get("vocab").and_then(serde_json::Value::as_object) {
        let max_id = vocab
            .values()
            .filter_map(serde_json::Value::as_u64)
            .max()
            .unwrap_or(0);
        tokens.resize(checked_token_id(max_id)? + 1, String::new());
        for (token, id) in vocab {
            if let Some(id) = id.as_u64() {
                let id = checked_token_id(id)?;
                if id >= tokens.len() {
                    tokens.resize(id + 1, String::new());
                }
                tokens[id] = token.clone();
                token_to_id.insert(token.clone(), id as u32);
            }
        }
    }

    if let Some(added) = tokenizer
        .get("added_tokens")
        .and_then(serde_json::Value::as_array)
    {
        for token in added {
            let Some(id) = token.get("id").and_then(serde_json::Value::as_u64) else {
                continue;
            };
            let Some(content) = token.get("content").and_then(serde_json::Value::as_str) else {
                continue;
            };
            let id = checked_token_id(id)?;
            if id >= tokens.len() {
                tokens.resize(id + 1, String::new());
            }
            tokens[id] = content.to_string();
            token_to_id.insert(content.to_string(), id as u32);
        }
    }

    if !tokens.is_empty() {
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::StringVec(tokens),
        );
    }

    if let Some(merges) = model.get("merges").and_then(serde_json::Value::as_array) {
        let merges = merges
            .iter()
            .filter_map(|merge| {
                if let Some(merge) = merge.as_str() {
                    Some(merge.to_string())
                } else {
                    let pair = merge.as_array()?;
                    if pair.len() == 2 {
                        Some(format!("{} {}", pair[0].as_str()?, pair[1].as_str()?))
                    } else {
                        None
                    }
                }
            })
            .collect::<Vec<_>>();
        if !merges.is_empty() {
            metadata.insert(
                "tokenizer.ggml.merges".to_string(),
                GgufValue::StringVec(merges),
            );
        }
    }

    if let Some(unk) = model
        .get("unk_token")
        .and_then(serde_json::Value::as_str)
        .and_then(|token| token_to_id.get(token))
    {
        metadata.insert(
            "tokenizer.ggml.unknown_token_id".to_string(),
            GgufValue::U32(*unk),
        );
    }

    insert_tokenizer_behavior_metadata(metadata, &tokenizer);
    insert_tokenizer_config_metadata(metadata, input, &token_to_id)?;

    Ok(())
}

fn insert_tokenizer_behavior_metadata(
    metadata: &mut HashMap<String, GgufValue>,
    tokenizer: &serde_json::Value,
) {
    if tokenizer_component_has_type(tokenizer.get("normalizer"), "Lowercase") {
        metadata.insert(
            "nexus.tokenizer.normalizer.lowercase".to_string(),
            GgufValue::Bool(true),
        );
    }
    if tokenizer_component_has_type(tokenizer.get("normalizer"), "StripAccents") {
        metadata.insert(
            "nexus.tokenizer.normalizer.strip_accents".to_string(),
            GgufValue::Bool(true),
        );
    }
    if tokenizer_component_has_type(tokenizer.get("normalizer"), "BertNormalizer") {
        if tokenizer
            .pointer("/normalizer/lowercase")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            metadata.insert(
                "nexus.tokenizer.normalizer.lowercase".to_string(),
                GgufValue::Bool(true),
            );
        }
        if tokenizer
            .pointer("/normalizer/strip_accents")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            metadata.insert(
                "nexus.tokenizer.normalizer.strip_accents".to_string(),
                GgufValue::Bool(true),
            );
        }
    }

    if tokenizer_component_has_type(tokenizer.get("pre_tokenizer"), "ByteLevel") {
        metadata.insert(
            "nexus.tokenizer.pre_tokenizer.byte_level".to_string(),
            GgufValue::Bool(true),
        );
        if tokenizer
            .pointer("/pre_tokenizer/add_prefix_space")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        {
            metadata.insert(
                "nexus.tokenizer.pre_tokenizer.add_prefix_space".to_string(),
                GgufValue::Bool(true),
            );
        }
        metadata.insert(
            "nexus.tokenizer.space_marker".to_string(),
            GgufValue::String("Ġ".to_string()),
        );
    }

    if tokenizer_component_has_type(tokenizer.get("decoder"), "ByteLevel") {
        metadata.insert(
            "nexus.tokenizer.decoder.byte_level".to_string(),
            GgufValue::Bool(true),
        );
    }
}

fn tokenizer_component_has_type(component: Option<&serde_json::Value>, expected: &str) -> bool {
    let Some(component) = component else {
        return false;
    };
    if component
        .get("type")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|kind| kind == expected)
    {
        return true;
    }
    component
        .get("normalizers")
        .or_else(|| component.get("pretokenizers"))
        .or_else(|| component.get("decoders"))
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .any(|item| tokenizer_component_has_type(Some(item), expected))
        })
        .unwrap_or(false)
}

fn insert_tokenizer_config_metadata(
    metadata: &mut HashMap<String, GgufValue>,
    input: &Path,
    token_to_id: &HashMap<String, u32>,
) -> Result<()> {
    let config_path = input.join("tokenizer_config.json");
    let config: serde_json::Value = if config_path.exists() {
        serde_json::from_slice(&std::fs::read(&config_path)?)
            .with_context(|| format!("Failed to parse {}", config_path.display()))?
    } else {
        serde_json::Value::Null
    };
    let template = if input.join("chat_template.jinja").exists() {
        Some(std::fs::read_to_string(input.join("chat_template.jinja"))?)
    } else if input.join("chat_template.json").exists() {
        let value = serde_json::from_slice(&std::fs::read(input.join("chat_template.json"))?)?;
        nexus::tokenizer::exact::select_chat_template(&value).map_err(anyhow::Error::msg)?
    } else {
        nexus::tokenizer::exact::select_chat_template(
            config
                .get("chat_template")
                .unwrap_or(&serde_json::Value::Null),
        )
        .map_err(anyhow::Error::msg)?
    };
    if let Some(template) = template {
        metadata.insert(
            "tokenizer.chat_template".to_string(),
            GgufValue::String(template.to_string()),
        );
    }
    if config
        .get("add_prefix_space")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
    {
        metadata.insert(
            "nexus.tokenizer.pre_tokenizer.add_prefix_space".to_string(),
            GgufValue::Bool(true),
        );
    }

    for (json_key, gguf_key) in [
        ("bos_token", "tokenizer.ggml.bos_token_id"),
        ("eos_token", "tokenizer.ggml.eos_token_id"),
        ("unk_token", "tokenizer.ggml.unknown_token_id"),
        ("sep_token", "tokenizer.ggml.separator_token_id"),
        ("pad_token", "tokenizer.ggml.padding_token_id"),
    ] {
        if metadata.contains_key(gguf_key) {
            continue;
        }
        if let Some(token) = tokenizer_config_token(&config, json_key) {
            if let Some(id) = token_to_id.get(token) {
                metadata.insert(gguf_key.to_string(), GgufValue::U32(*id));
            }
        }
    }

    for (json_key, gguf_key) in [
        ("bos_token_id", "tokenizer.ggml.bos_token_id"),
        ("eos_token_id", "tokenizer.ggml.eos_token_id"),
        ("unk_token_id", "tokenizer.ggml.unknown_token_id"),
        ("sep_token_id", "tokenizer.ggml.separator_token_id"),
        ("pad_token_id", "tokenizer.ggml.padding_token_id"),
    ] {
        if metadata.contains_key(gguf_key) {
            continue;
        }
        if let Some(id) = config.get(json_key).and_then(serde_json::Value::as_i64) {
            metadata.insert(gguf_key.to_string(), GgufValue::I32(id as i32));
        }
    }

    Ok(())
}

fn tokenizer_config_token<'a>(config: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    let value = config.get(key)?;
    if let Some(token) = value.as_str() {
        return Some(token);
    }
    value
        .as_object()
        .and_then(|object| object.get("content"))
        .and_then(serde_json::Value::as_str)
}

fn write_string(writer: &mut dyn Write, value: &str) -> Result<()> {
    write_u64(writer, value.len() as u64)?;
    writer.write_all(value.as_bytes())?;
    Ok(())
}

fn write_u32(writer: &mut dyn Write, value: u32) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn write_u64(writer: &mut dyn Write, value: u64) -> Result<()> {
    writer.write_all(&value.to_le_bytes())?;
    Ok(())
}

fn pad_to_alignment(writer: &mut BufWriter<File>, alignment: usize) -> Result<()> {
    let pos = writer.stream_position()? as usize;
    let padding = (alignment - pos % alignment) % alignment;
    if padding > 0 {
        writer.write_all(&vec![0u8; padding])?;
    }
    Ok(())
}

fn align(value: usize, alignment: usize) -> usize {
    value.div_ceil(alignment) * alignment
}

#[cfg(test)]
mod tests {
    use super::*;
    use nexus::gguf::GgufReader;
    use std::fs;

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
    fn test_canonical_tensor_name() {
        assert_eq!(
            canonical_tensor_name("model.embed_tokens.weight"),
            "token_embd.weight"
        );
        assert_eq!(
            canonical_tensor_name("model.layers.0.self_attn.q_proj.weight"),
            "blk.0.attn_q.weight"
        );
        assert_eq!(
            canonical_tensor_name("model.layers.1.mlp.down_proj.weight"),
            "blk.1.ffn_down.weight"
        );
    }

    #[test]
    fn test_write_converted_gguf_is_readable() {
        let dir = std::env::temp_dir().join(format!("nexus-convert-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"llama","hidden_size":4,"num_hidden_layers":0,"num_attention_heads":1,"num_key_value_heads":1,"intermediate_size":4,"vocab_size":2,"max_position_embeddings":16}"#,
        )
        .unwrap();
        write_f32_safetensors(
            &dir.join("model.safetensors"),
            "model.embed_tokens.weight",
            &[2, 4],
            &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        );

        let loader = MlxModelLoader::new(&dir).unwrap();
        let engine = QuantEngine::new(NexusQuantFormat::FP32, KvQuantType::None);
        let name = loader.tensor_names()[0].clone();
        let meta = loader.tensor_meta(&name).unwrap();
        let values = loader.load_tensor(&name).unwrap();
        let tensors = vec![ConvertedTensor {
            name: canonical_tensor_name(&name),
            shape: meta.shape.clone(),
            gguf_type: QuantFormat::F32.gguf_type().unwrap(),
            data: engine.quantize_weights(&values).unwrap(),
            offset: 0,
        }];
        let metadata = build_metadata(&loader, &dir, false).unwrap();
        let output = dir.join("out.gguf");
        write_gguf(&output, &metadata, &tensors).unwrap();

        let parsed = GgufReader::parse_file(&output).unwrap();
        assert_eq!(
            parsed
                .get_tensor("token_embd.weight")
                .unwrap()
                .data_type
                .name(),
            "F32"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn chat_template_files_and_named_defaults_survive_conversion() {
        let dir = std::env::temp_dir().join(format!("nexus-chat-template-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let mut metadata = HashMap::new();
        for value in [
            r#"{"chat_template":"json-template"}"#,
            r#"{"default":"json-template","tool_use":"wrong"}"#,
        ] {
            fs::write(dir.join("chat_template.json"), value).unwrap();
            insert_tokenizer_config_metadata(&mut metadata, &dir, &HashMap::new()).unwrap();
            assert!(
                matches!(metadata.get("tokenizer.chat_template"),Some(GgufValue::String(s)) if s=="json-template")
            );
        }
        fs::write(dir.join("chat_template.jinja"), "jinja-template").unwrap();
        insert_tokenizer_config_metadata(&mut metadata, &dir, &HashMap::new()).unwrap();
        assert!(
            matches!(metadata.get("tokenizer.chat_template"),Some(GgufValue::String(s)) if s=="jinja-template")
        );
        fs::remove_file(dir.join("chat_template.jinja")).unwrap();
        fs::remove_file(dir.join("chat_template.json")).unwrap();
        fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template":[{"name":"default","template":"config-template"}]}"#,
        )
        .unwrap();
        insert_tokenizer_config_metadata(&mut metadata, &dir, &HashMap::new()).unwrap();
        assert!(
            matches!(metadata.get("tokenizer.chat_template"),Some(GgufValue::String(s)) if s=="config-template")
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_tokenizer_json_metadata_is_written() {
        let dir = std::env::temp_dir().join(format!(
            "nexus-tokenizer-convert-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"llama","hidden_size":4,"num_hidden_layers":0,"num_attention_heads":1,"num_key_value_heads":1,"intermediate_size":4,"vocab_size":4,"max_position_embeddings":16}"#,
        )
        .unwrap();
        fs::write(
            dir.join("tokenizer.json"),
            r#"{"normalizer":{"type":"Sequence","normalizers":[{"type":"Lowercase"},{"type":"StripAccents"}]},"pre_tokenizer":{"type":"ByteLevel","add_prefix_space":true},"decoder":{"type":"ByteLevel"},"model":{"type":"BPE","unk_token":"<unk>","vocab":{"<unk>":0,"h":1,"e":2,"he":3},"merges":["h e"]}}"#,
        )
        .unwrap();
        fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"chat_template":"<|im_start|>{{ role }}","unk_token":"<unk>"}"#,
        )
        .unwrap();
        write_f32_safetensors(
            &dir.join("model.safetensors"),
            "model.embed_tokens.weight",
            &[1, 4],
            &[0.1, 0.2, 0.3, 0.4],
        );

        let loader = MlxModelLoader::new(&dir).unwrap();
        let metadata = build_metadata(&loader, &dir, false).unwrap();
        let tokens = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|value| match value {
                GgufValue::StringVec(values) => Some(values),
                _ => None,
            })
            .unwrap();
        assert_eq!(tokens[3], "he");
        assert!(matches!(
            metadata.get("tokenizer.ggml.unknown_token_id"),
            Some(GgufValue::U32(0))
        ));
        assert!(metadata.contains_key("tokenizer.chat_template"));
        assert!(matches!(
            metadata.get("nexus.tokenizer.normalizer.lowercase"),
            Some(GgufValue::Bool(true))
        ));
        assert!(matches!(
            metadata.get("nexus.tokenizer.normalizer.strip_accents"),
            Some(GgufValue::Bool(true))
        ));
        assert!(matches!(
            metadata.get("nexus.tokenizer.pre_tokenizer.byte_level"),
            Some(GgufValue::Bool(true))
        ));
        assert!(matches!(
            metadata.get("nexus.tokenizer.pre_tokenizer.add_prefix_space"),
            Some(GgufValue::Bool(true))
        ));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_standard_k_quant_writing_is_mapped_to_gguf_types() {
        assert_eq!(
            QuantFormat::Q4_K.to_nexus_quant().unwrap(),
            NexusQuantFormat::Q4_K
        );
        assert_eq!(QuantFormat::Q4_K.gguf_type().unwrap(), 12);
        assert_eq!(
            QuantFormat::Q2_K.to_nexus_quant().unwrap(),
            NexusQuantFormat::Q2_K
        );
        assert_eq!(QuantFormat::Q2_K.gguf_type().unwrap(), 10);
    }
    #[test]
    fn llama_qk_permutation_interleaves_hf_split_halves_per_head() {
        // GQA geometry: 4 attention heads and 2 KV heads with head_dim 4, so
        // the K side is exercised with more than one KV head (a K-only bug
        // that only differs at kv_heads > 1 previously passed every test).
        // Mistral uses the same split-half RoPE pairing as LLaMA, so both
        // labels must produce the identical permutation.
        let configs = [
            r#"{"model_type":"llama","hidden_size":16,"num_attention_heads":4,"num_key_value_heads":2}"#,
            r#"{"model_type":"mistral","hidden_size":16,"num_attention_heads":4,"num_key_value_heads":2}"#,
        ];
        for (index, config_json) in configs.iter().enumerate() {
            let config = nexus::mlx::MlxModelConfig::from_json(config_json).unwrap();
            let q: Vec<f32> = (0..16).flat_map(|row| vec![row as f32; 16]).collect();
            let k: Vec<f32> = (0..8).flat_map(|row| vec![row as f32; 16]).collect();
            let q = permute_llama_qk(
                "model.layers.0.self_attn.q_proj.weight",
                &[16, 16],
                q,
                Some(&config),
            )
            .unwrap();
            let k = permute_llama_qk(
                "model.layers.0.self_attn.k_proj.weight",
                &[8, 16],
                k,
                Some(&config),
            )
            .unwrap();
            assert_eq!(
                q.as_chunks::<16>()
                    .0
                    .iter()
                    .map(|row| row[0])
                    .collect::<Vec<_>>(),
                vec![0., 2., 1., 3., 4., 6., 5., 7., 8., 10., 9., 11., 12., 14., 13., 15.],
                "Q permutation for config {index}"
            );
            assert_eq!(
                k.as_chunks::<16>()
                    .0
                    .iter()
                    .map(|row| row[0])
                    .collect::<Vec<_>>(),
                vec![0., 2., 1., 3., 4., 6., 5., 7.],
                "K permutation for config {index} must permute each KV head block"
            );
        }
    }

    #[test]
    fn mistral_conversion_matches_llama_conversion_end_to_end() {
        // Same weights under both accepted labels: the converted GGUFs must be
        // indistinguishable to the engine (identical logits). This is the
        // regression gate for the Q/K permutation gate: before it covered
        // mistral, the mistral file was written with unpermuted rows and
        // silently produced wrong attention.
        //
        // This test compares two outputs of the same converter; the permuted
        // layout itself is anchored externally by
        // `gguf_head_dim_four_logits_match_llama_cpp_cpu_f32`
        // (llama.cpp C++ reference) — the gate chain only holds because both
        // tests exist. GQA geometry (kv_heads = 2) keeps a K-only permutation
        // bug from passing.
        let llama_dir = tiny_llm_fixture_dir("mistral-e2e-llama", "llama", false);
        let mistral_dir = tiny_llm_fixture_dir("mistral-e2e-mistral", "mistral", false);
        // Real Mistral-7B ships no lm_head.weight (tied embeddings): the tied
        // resolution must yield the same logits as the explicit-head file.
        let tied_dir = tiny_llm_fixture_dir("mistral-e2e-tied", "mistral", true);
        let llama_file = convert_fixture(&llama_dir);
        let mistral_file = convert_fixture(&mistral_dir);
        let tied_file = convert_fixture(&tied_dir);

        let mut llama = engine_for_gguf(&llama_file);
        let mut mistral = engine_for_gguf(&mistral_file);
        let mut tied = engine_for_gguf(&tied_file);
        for prompt in [[0u32, 1, 2], [3, 0, 1]] {
            assert_eq!(
                llama.prefill(&prompt).unwrap(),
                mistral.prefill(&prompt).unwrap(),
                "converted output must not depend on the model_type label"
            );
            assert_eq!(
                mistral.prefill(&prompt).unwrap(),
                tied.prefill(&prompt).unwrap(),
                "tied embeddings must resolve to the same output head"
            );
        }
        for dir in [&llama_dir, &mistral_dir, &tied_dir] {
            fs::remove_dir_all(dir).unwrap();
        }
    }

    fn tiny_llm_fixture_dir(tag: &str, model_type: &str, tied_head: bool) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nexus-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("config.json"),
            format!(
                r#"{{"model_type":"{model_type}","hidden_size":16,"num_hidden_layers":1,"num_attention_heads":4,"num_key_value_heads":2,"intermediate_size":32,"vocab_size":4,"max_position_embeddings":32}}"#
            ),
        )
        .unwrap();
        // Position-dependent values: constant rows would hide a Q/K
        // permutation defect. head_dim = 4 so the split-half and adjacent-pair
        // layouts genuinely differ (they coincide at head_dim = 2), and
        // kv_heads = 2 so the K permutation exercises multiple KV heads.
        let value = |len: usize| -> Vec<f32> {
            (0..len)
                .map(|i| ((i * 13 + 7) % 17) as f32 / 17.0 - 0.4)
                .collect()
        };
        // (HF name, HF shape [rows, cols]) for a one-layer GQA model.
        let mut tensors: Vec<(&str, Vec<usize>)> = vec![
            ("model.embed_tokens.weight", vec![4, 16]),
            ("model.norm.weight", vec![16]),
            ("model.layers.0.input_layernorm.weight", vec![16]),
            ("model.layers.0.post_attention_layernorm.weight", vec![16]),
            ("model.layers.0.self_attn.q_proj.weight", vec![16, 16]),
            ("model.layers.0.self_attn.k_proj.weight", vec![8, 16]),
            ("model.layers.0.self_attn.v_proj.weight", vec![8, 16]),
            ("model.layers.0.self_attn.o_proj.weight", vec![16, 16]),
            ("model.layers.0.mlp.gate_proj.weight", vec![32, 16]),
            ("model.layers.0.mlp.up_proj.weight", vec![32, 16]),
            ("model.layers.0.mlp.down_proj.weight", vec![16, 32]),
        ];
        if !tied_head {
            // Byte-identical to the embedding (same generator and length), so
            // the tied-vs-explicit comparison below is maximally sensitive to
            // how the engine resolves the output head.
            tensors.push(("lm_head.weight", vec![4, 16]));
        }
        for (index, (name, shape)) in tensors.into_iter().enumerate() {
            write_f32_safetensors(
                &dir.join(format!("shard-{index:02}.safetensors")),
                name,
                &shape,
                &value(shape.iter().product()),
            );
        }
        dir
    }

    /// Run the real conversion pipeline over a fixture directory.
    fn convert_fixture(dir: &Path) -> std::path::PathBuf {
        let loader = MlxModelLoader::new(dir).unwrap();
        validate_model_config(&loader).unwrap();
        let mut tensors = Vec::new();
        let mut names = HashSet::new();
        let mut offset = 0u64;
        for name in loader.tensor_names() {
            let meta = loader.tensor_meta(name.as_str()).unwrap();
            let values = loader.load_tensor(name.as_str()).unwrap();
            let values =
                permute_llama_qk(name.as_str(), &meta.shape, values, loader.config.as_ref())
                    .unwrap();
            let canonical_name = canonical_tensor_name(name.as_str());
            assert!(names.insert(canonical_name.clone()));
            let format = tensor_quant_format(&meta.shape, NexusQuantFormat::FP32);
            let quant_engine = QuantEngine::new(format, KvQuantType::None);
            let data = quant_engine.quantize_weights(&values).unwrap();
            let aligned = align(data.len(), ALIGNMENT);
            tensors.push(ConvertedTensor {
                name: canonical_name,
                shape: meta.shape.clone(),
                gguf_type: weight_data_type(format).unwrap().id(),
                data,
                offset,
            });
            offset += aligned as u64;
        }
        let metadata = build_metadata(&loader, dir, false).unwrap();
        let output = dir.join("model.gguf");
        write_gguf(&output, &metadata, &tensors).unwrap();
        validate_written_gguf(&output, tensors.len()).unwrap();
        output
    }

    fn engine_for_gguf(path: &Path) -> nexus::engine::InferenceEngine {
        let mut engine =
            nexus::engine::InferenceEngine::new(Some(nexus::engine::InferenceConfig {
                allow_byte_tokenizer: true,
                max_seq_len: 32,
                ..Default::default()
            }));
        engine.load_model(path).unwrap();
        engine
    }

    #[test]
    fn gguf_tensor_dimensions_reverse_safetensors_order() {
        let file = tempfile_for_dimension_test();
        let tensor = ConvertedTensor {
            name: "token_embd.weight".into(),
            shape: vec![3, 8],
            gguf_type: 0,
            data: vec![0; 3 * 8 * 4],
            offset: 0,
        };
        write_gguf(&file, &HashMap::new(), &[tensor]).unwrap();
        let metadata = GgufReader::parse_file(&file).unwrap();
        assert_eq!(metadata.tensors[0].shape, vec![8, 3]);
        std::fs::remove_file(file).unwrap();
    }

    fn tempfile_for_dimension_test() -> PathBuf {
        std::env::temp_dir().join(format!("nexus-dim-test-{}.gguf", std::process::id()))
    }
    #[test]
    fn quantization_preserves_vectors_and_incomplete_rows() {
        for (shape, requested, expected) in [
            (vec![256], NexusQuantFormat::Q4_K, NexusQuantFormat::FP32),
            (vec![2, 16], NexusQuantFormat::Q4_0, NexusQuantFormat::FP32),
            (vec![2, 33], NexusQuantFormat::Q8_0, NexusQuantFormat::FP32),
            (vec![2, 32], NexusQuantFormat::Q4_0, NexusQuantFormat::Q4_0),
            (vec![2, 256], NexusQuantFormat::Q4_K, NexusQuantFormat::Q4_K),
            (vec![2, 256], NexusQuantFormat::Q2_K, NexusQuantFormat::Q2_K),
        ] {
            assert_eq!(tensor_quant_format(&shape, requested), expected);
        }
    }

    #[test]
    fn invalid_conversion_does_not_replace_existing_output() {
        let dir = std::env::temp_dir().join(format!(
            "nexus-atomic-format-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        let output = dir.join("model.gguf");
        fs::write(&output, b"previous valid output").unwrap();
        let invalid = ConvertedTensor {
            name: "w".into(),
            shape: vec![31],
            gguf_type: 2,
            data: vec![0; 18],
            offset: 0,
        };
        assert!(write_gguf(&output, &HashMap::new(), &[invalid]).is_err());
        assert_eq!(fs::read(&output).unwrap(), b"previous valid output");
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
        let valid = ConvertedTensor {
            name: "w".into(),
            shape: vec![2, 3],
            gguf_type: 0,
            data: (0..6).flat_map(|i| (i as f32).to_le_bytes()).collect(),
            offset: 0,
        };
        write_gguf(&output, &HashMap::new(), &[valid]).unwrap();
        assert_eq!(
            GgufReader::parse_file(&output).unwrap().tensors[0].shape,
            [3, 2]
        );
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 1);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn invalid_or_unimplemented_architecture_is_rejected() {
        let dir = std::env::temp_dir().join(format!("nexus-format-config-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        write_f32_safetensors(
            &dir.join("model.safetensors"),
            "model.embed_tokens.weight",
            &[1, 2],
            &[1., 2.],
        );
        for config in [
            r#"{}"#,
            r#"{"model_type":"qwen2","hidden_size":2}"#,
            r#"{"model_type":"llama","hidden_size":0}"#,
        ] {
            fs::write(dir.join("config.json"), config).unwrap();
            let loader = MlxModelLoader::new(&dir).unwrap();
            assert!(validate_model_config(&loader).is_err());
            assert!(build_metadata(&loader, &dir, false).is_err());
        }
        fs::remove_dir_all(dir).unwrap();
    }

    fn config_fixture_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "nexus-{tag}-config-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&dir).unwrap();
        write_f32_safetensors(
            &dir.join("model.safetensors"),
            "model.embed_tokens.weight",
            &[1, 2],
            &[1., 2.],
        );
        dir
    }

    #[test]
    fn mistral_conversion_writes_mistral_prefixed_metadata() {
        let dir = config_fixture_dir("mistral");
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"mistral","hidden_size":4,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"intermediate_size":4,"vocab_size":2,"max_position_embeddings":16}"#,
        )
        .unwrap();
        let loader = MlxModelLoader::new(&dir).unwrap();
        validate_model_config(&loader).unwrap();
        let metadata = build_metadata(&loader, &dir, false).unwrap();
        assert!(matches!(
            metadata.get("general.architecture"),
            Some(GgufValue::String(value)) if value == "mistral"
        ));
        assert!(metadata.contains_key("mistral.embedding_length"));
        assert!(metadata.contains_key("mistral.block_count"));
        assert!(metadata.contains_key("mistral.context_length"));
        assert!(
            !metadata.contains_key("llama.embedding_length"),
            "keys must follow the declared architecture, not a hard-coded prefix"
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rope_scaling_declarations_are_rejected_unless_they_declare_none() {
        let base = r#"{"model_type":"llama","hidden_size":4,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"intermediate_size":4,"vocab_size":2,"max_position_embeddings":16"#;
        for (extra, expect_ok) in [
            (
                Some(r#""rope_scaling":{"rope_type":"llama3","factor":8.0}"#),
                false,
            ),
            (
                Some(r#""rope_scaling":{"type":"linear","factor":32.0}"#),
                false,
            ),
            (Some(r#""rope_scaling":{"rope_type":"none"}"#), true),
            (None, true),
        ] {
            let mut config = base.to_string();
            if let Some(extra) = extra {
                config.push(',');
                config.push_str(extra);
            }
            config.push('}');
            let dir = config_fixture_dir("rope-scaling");
            fs::write(dir.join("config.json"), &config).unwrap();
            let loader = MlxModelLoader::new(&dir).unwrap();
            let result = validate_model_config(&loader);
            assert_eq!(result.is_ok(), expect_ok, "config: {config}");
            if expect_ok {
                assert!(build_metadata(&loader, &dir, false).is_ok());
            } else {
                let message = result.unwrap_err().to_string();
                assert!(
                    message.contains("rope_scaling"),
                    "the rejection must name rope_scaling: {message}"
                );
            }
            fs::remove_dir_all(&dir).unwrap();
        }
    }

    #[test]
    fn hostile_tokenizer_token_ids_fail_conversion_instead_of_aborting() {
        let dir = config_fixture_dir("token-ids");
        fs::write(
            dir.join("config.json"),
            r#"{"model_type":"llama","hidden_size":4,"num_hidden_layers":1,"num_attention_heads":1,"num_key_value_heads":1,"intermediate_size":4,"vocab_size":2,"max_position_embeddings":16}"#,
        )
        .unwrap();
        // The `model` object is required; without it the failure would come
        // from tokenizer validation instead of the token-id guard.
        for hostile in [
            r#"{"model":{"type":"bpe","vocab":{"a":0}},"added_tokens":[{"id":18446744073709551615,"content":"x"}]}"#,
            r#"{"model":{"type":"bpe","vocab":{"a":10000000000}}}"#,
        ] {
            fs::write(dir.join("tokenizer.json"), hostile).unwrap();
            let loader = MlxModelLoader::new(&dir).unwrap();
            let result = build_metadata(&loader, &dir, false);
            let message = result
                .as_ref()
                .map(|_| String::new())
                .unwrap_or_else(|err| err.to_string());
            assert!(result.is_err(), "hostile ids must be rejected: {hostile}");
            assert!(
                message.contains("token id"),
                "the rejection must come from the token-id guard: {message}"
            );
        }
        // In-bounds ids still convert.
        fs::write(
            dir.join("tokenizer.json"),
            r#"{"model":{"type":"bpe","vocab":{"a":0,"b":1}},"added_tokens":[{"id":5,"content":"<pad>"}]}"#,
        )
        .unwrap();
        let loader = MlxModelLoader::new(&dir).unwrap();
        assert!(build_metadata(&loader, &dir, false).is_ok());
        fs::remove_dir_all(&dir).unwrap();
    }
}
