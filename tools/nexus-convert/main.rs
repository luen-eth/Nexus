//! Model conversion tool - convert safetensors/MLX models to GGUF format.
//! Supports adding Nexus TurboQuant metadata for KV cache compression.

#![allow(non_camel_case_types)]

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use nexus::gguf::GgufReader;
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

    /// Quantization format
    #[arg(long, default_value = "q4-0")]
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
    let quant_format = cli.quant.to_nexus_quant()?;
    let gguf_type = cli.quant.gguf_type()?;
    let quant_engine = QuantEngine::new(quant_format, KvQuantType::None);

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

        let canonical_name = canonical_tensor_name(name);
        if !names.insert(canonical_name.clone()) {
            anyhow::bail!(
                "Tensor name collision after canonicalization: {} -> {}",
                name,
                canonical_name
            );
        }

        let data = quant_engine.quantize_weights(&values);
        let reconstructed = quant_engine.dequantize(&data, values.len());
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
        offset += aligned_size as u64;
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

    fn gguf_type(self) -> Result<u32> {
        match self {
            QuantFormat::F32 => Ok(0),
            QuantFormat::F16 => Ok(1),
            QuantFormat::Q4_0 => Ok(2),
            QuantFormat::Q8_0 => Ok(8),
            QuantFormat::Q2_K => Ok(10),
            QuantFormat::Q4_K => Ok(12),
        }
    }
}

fn build_metadata(
    loader: &MlxModelLoader,
    input: &Path,
    turboquant: bool,
) -> Result<HashMap<String, GgufValue>> {
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
        insert_usize(&mut metadata, "llama.embedding_length", config.hidden_size);
        insert_usize(&mut metadata, "llama.block_count", config.num_hidden_layers);
        insert_usize(
            &mut metadata,
            "llama.attention.head_count",
            config.num_attention_heads,
        );
        insert_usize(
            &mut metadata,
            "llama.attention.head_count_kv",
            config.num_key_value_heads,
        );
        insert_usize(
            &mut metadata,
            "llama.feed_forward_length",
            config.intermediate_size,
        );
        insert_usize(&mut metadata, "llama.vocab_size", config.vocab_size);
        insert_usize(
            &mut metadata,
            "llama.context_length",
            config.max_position_embeddings,
        );
        if let Some(eps) = config.rms_norm_eps {
            metadata.insert(
                "llama.attention.layer_norm_rms_epsilon".to_string(),
                GgufValue::F32(eps),
            );
        }
        if let Some(theta) = config.rope_theta {
            metadata.insert("llama.rope.freq_base".to_string(), GgufValue::F32(theta));
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
    let file =
        File::create(path).with_context(|| format!("Failed to create {}", path.display()))?;
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
        for dim in &tensor.shape {
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
    let model = tokenizer
        .get("model")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("tokenizer.json is missing model object"))?;

    if let Some(model_type) = model.get("type").and_then(serde_json::Value::as_str) {
        metadata.insert(
            "tokenizer.ggml.model".to_string(),
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
            .unwrap_or(0) as usize;
        tokens.resize(max_id + 1, String::new());
        for (token, id) in vocab {
            if let Some(id) = id.as_u64() {
                let id = id as usize;
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
            let id = id as usize;
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
    if !config_path.exists() {
        return Ok(());
    }

    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?,
    )
    .with_context(|| format!("Failed to parse {}", config_path.display()))?;

    if let Some(template) = config
        .get("chat_template")
        .and_then(serde_json::Value::as_str)
    {
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
        let engine = QuantEngine::new(NexusQuantFormat::Q8_0, KvQuantType::None);
        let name = loader.tensor_names()[0].clone();
        let meta = loader.tensor_meta(&name).unwrap();
        let values = loader.load_tensor(&name).unwrap();
        let tensors = vec![ConvertedTensor {
            name: canonical_tensor_name(&name),
            shape: meta.shape.clone(),
            gguf_type: QuantFormat::Q8_0.gguf_type().unwrap(),
            data: engine.quantize_weights(&values),
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
            "Q8_0"
        );
        let _ = fs::remove_dir_all(&dir);
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
}
