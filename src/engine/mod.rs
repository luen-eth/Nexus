//! Inference engine - loads GGUF models and runs inference.
//! Combines weight dequantization, KV cache management (with TurboQuant),
//! and transformer computation.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io;
use std::path::Path;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::backend::{Backend, BackendFactory, BackendType};
use crate::gguf::{GgufDataType, GgufReader, ModelMetadata, TensorInfo};
use crate::kernel;
use crate::memory::UnifiedMemory;
use crate::quant::{KvQuantType, QuantEngine, QuantFormat};

/// Model architecture types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelArch {
    Llama,
    Mistral,
    Mixtral,
    Gemma,
    Phi,
    Qwen2,
    Deepseek,
    Unknown(String),
}

impl ModelArch {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "llama" => ModelArch::Llama,
            "mistral" => ModelArch::Mistral,
            "mixtral" => ModelArch::Mixtral,
            "gemma" => ModelArch::Gemma,
            "phi" => ModelArch::Phi,
            "qwen2" => ModelArch::Qwen2,
            "deepseek" => ModelArch::Deepseek,
            other => ModelArch::Unknown(other.to_string()),
        }
    }
}

impl fmt::Display for ModelArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelArch::Llama => write!(f, "llama"),
            ModelArch::Mistral => write!(f, "mistral"),
            ModelArch::Mixtral => write!(f, "mixtral"),
            ModelArch::Gemma => write!(f, "gemma"),
            ModelArch::Phi => write!(f, "phi"),
            ModelArch::Qwen2 => write!(f, "qwen2"),
            ModelArch::Deepseek => write!(f, "deepseek"),
            ModelArch::Unknown(s) => write!(f, "{}", s),
        }
    }
}

/// Inference configuration
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Number of parallel sequences (batch size)
    pub batch_size: usize,
    /// KV cache quantization type
    pub kv_quant: KvQuantType,
    /// Weight quantization format
    pub weight_format: QuantFormat,
    /// Backend to use
    pub backend: BackendType,
    /// Rope frequency base
    pub rope_freq_base: f32,
    /// RMS norm epsilon
    pub rms_norm_eps: f32,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        InferenceConfig {
            max_seq_len: 4096,
            batch_size: 1,
            kv_quant: KvQuantType::TurboQuant3b2b,
            weight_format: QuantFormat::Q4_K,
            backend: BackendType::default(),
            rope_freq_base: 1000000.0,
            rms_norm_eps: 1e-5,
        }
    }
}

/// Text generation/sampling configuration.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
    pub stop_sequences: Vec<Vec<u32>>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        GenerationConfig {
            max_tokens: 128,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repetition_penalty: 1.0,
            seed: None,
            stop_sequences: Vec::new(),
        }
    }
}

/// Verification statistics for speculative decoding.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeStats {
    pub accepted: usize,
    pub rejected: usize,
}

/// Top-k MoE router output for one token.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeRoute {
    pub expert: usize,
    pub weight: f32,
}

/// Borrowed MoE expert weights for routed FFN dispatch.
pub struct MoeExpertWeights<'a> {
    pub gate: &'a [Vec<f32>],
    pub up: &'a [Vec<f32>],
    pub down: &'a [Vec<f32>],
}

/// MoE dispatch dimensions and routing options.
#[derive(Debug, Clone, Copy)]
pub struct MoeDispatchConfig {
    pub top_k: usize,
    pub hidden_size: usize,
    pub ff_dim: usize,
}

/// Loaded model weights (dequantized on demand)
pub struct ModelWeights {
    pub metadata: ModelMetadata,
    pub arch: ModelArch,
    pub tensors: HashMap<String, Vec<f32>>,
    pub quant_engine: QuantEngine,
    pub memory: UnifiedMemory,
}

impl ModelWeights {
    /// Get a tensor by name, dequantizing if necessary
    pub fn get_tensor(&self, name: &str) -> Option<&Vec<f32>> {
        self.tensors.get(name)
    }

    /// Get tensor info from metadata
    pub fn get_tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.metadata.get_tensor(name)
    }

    /// Get model architecture
    pub fn architecture(&self) -> &ModelArch {
        &self.arch
    }

    /// Get model metadata
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }
}

/// Packed low-bit storage for a KV cache stream.
pub struct PackedKvBuffer {
    data: Vec<u8>,
    mins: Vec<f32>,
    scales: Vec<f32>,
    bits: u32,
    head_dim: usize,
    bytes_per_token: usize,
}

impl PackedKvBuffer {
    pub fn new(bits: u32, head_dim: usize, max_seq_len: usize) -> Self {
        let bytes_per_token = (head_dim * bits as usize).div_ceil(8);
        PackedKvBuffer {
            data: vec![0; bytes_per_token * max_seq_len],
            mins: vec![0.0; max_seq_len],
            scales: vec![1.0; max_seq_len],
            bits,
            head_dim,
            bytes_per_token,
        }
    }

    pub fn add(&mut self, position: usize, values: &[f32]) {
        assert_eq!(values.len(), self.head_dim, "KV vector dimension mismatch");

        let levels = (1u32 << self.bits) - 1;
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = if (max - min).abs() > 1e-8 {
            (max - min) / levels as f32
        } else {
            1.0
        };

        self.mins[position] = min;
        self.scales[position] = scale;

        let start = position * self.bytes_per_token;
        let end = start + self.bytes_per_token;
        self.data[start..end].fill(0);

        for (i, &value) in values.iter().enumerate() {
            let quantized = ((value - min) / scale).round().clamp(0.0, levels as f32) as u32;
            self.write_bits(start, i, quantized);
        }
    }

    pub fn get_all(&self, seq_len: usize) -> Vec<f32> {
        let mut output = vec![0.0; seq_len * self.head_dim];
        let mask = (1u32 << self.bits) - 1;

        for position in 0..seq_len {
            let start = position * self.bytes_per_token;
            let min = self.mins[position];
            let scale = self.scales[position];

            for dim in 0..self.head_dim {
                let q = self.read_bits(start, dim) & mask;
                output[position * self.head_dim + dim] = min + q as f32 * scale;
            }
        }

        output
    }

    pub fn value_at(&self, position: usize, dim: usize) -> f32 {
        let start = position * self.bytes_per_token;
        let q = self.read_bits(start, dim) & ((1u32 << self.bits) - 1);
        self.mins[position] + q as f32 * self.scales[position]
    }

    pub fn shift_left(&mut self, seq_len: usize) {
        if seq_len <= 1 {
            return;
        }

        let bytes = (seq_len - 1) * self.bytes_per_token;
        self.data
            .copy_within(self.bytes_per_token..self.bytes_per_token + bytes, 0);
        let tail_start = bytes;
        let tail_end = tail_start + self.bytes_per_token;
        self.data[tail_start..tail_end].fill(0);

        self.mins.copy_within(1..seq_len, 0);
        self.scales.copy_within(1..seq_len, 0);
        self.mins[seq_len - 1] = 0.0;
        self.scales[seq_len - 1] = 1.0;
    }

    pub fn clear(&mut self) {
        self.data.fill(0);
        self.mins.fill(0.0);
        self.scales.fill(1.0);
    }

    pub fn memory_usage(&self) -> usize {
        self.data.len()
            + self.mins.len() * std::mem::size_of::<f32>()
            + self.scales.len() * std::mem::size_of::<f32>()
    }

    fn write_bits(&mut self, token_start: usize, index: usize, value: u32) {
        let bit_offset = index * self.bits as usize;
        for bit in 0..self.bits as usize {
            if ((value >> bit) & 1) == 0 {
                continue;
            }
            let absolute = token_start * 8 + bit_offset + bit;
            let byte = absolute / 8;
            let bit_in_byte = absolute % 8;
            self.data[byte] |= 1 << bit_in_byte;
        }
    }

    fn read_bits(&self, token_start: usize, index: usize) -> u32 {
        let bit_offset = index * self.bits as usize;
        let mut value = 0u32;
        for bit in 0..self.bits as usize {
            let absolute = token_start * 8 + bit_offset + bit;
            let byte = absolute / 8;
            let bit_in_byte = absolute % 8;
            let bit_value = (self.data[byte] >> bit_in_byte) & 1;
            value |= (bit_value as u32) << bit;
        }
        value
    }
}

/// KV cache buffer for one key or value stream.
pub enum KvCacheBuffer {
    Float(Vec<f32>),
    Quantized(PackedKvBuffer),
}

impl KvCacheBuffer {
    fn new(bits: u32, head_dim: usize, max_seq_len: usize, quantized: bool) -> Self {
        if quantized {
            KvCacheBuffer::Quantized(PackedKvBuffer::new(bits, head_dim, max_seq_len))
        } else {
            KvCacheBuffer::Float(vec![0.0; max_seq_len * head_dim])
        }
    }

    fn add(&mut self, position: usize, head_dim: usize, values: &[f32]) {
        match self {
            KvCacheBuffer::Float(data) => {
                let offset = position * head_dim;
                data[offset..offset + head_dim].copy_from_slice(values);
            }
            KvCacheBuffer::Quantized(buffer) => buffer.add(position, values),
        }
    }

    fn get_all(&self, seq_len: usize, head_dim: usize) -> Vec<f32> {
        match self {
            KvCacheBuffer::Float(data) => data[..seq_len * head_dim].to_vec(),
            KvCacheBuffer::Quantized(buffer) => buffer.get_all(seq_len),
        }
    }

    fn value_at(&self, position: usize, dim: usize, head_dim: usize) -> f32 {
        match self {
            KvCacheBuffer::Float(data) => data[position * head_dim + dim],
            KvCacheBuffer::Quantized(buffer) => buffer.value_at(position, dim),
        }
    }

    fn shift_left(&mut self, seq_len: usize, head_dim: usize) {
        match self {
            KvCacheBuffer::Float(data) => {
                let total = seq_len * head_dim;
                let skip = head_dim;
                let src_len = total - skip;
                data.copy_within(skip..total, 0);
                data[src_len..total].fill(0.0);
            }
            KvCacheBuffer::Quantized(buffer) => buffer.shift_left(seq_len),
        }
    }

    fn clear(&mut self) {
        match self {
            KvCacheBuffer::Float(data) => data.fill(0.0),
            KvCacheBuffer::Quantized(buffer) => buffer.clear(),
        }
    }

    fn memory_usage(&self) -> usize {
        match self {
            KvCacheBuffer::Float(data) => data.len() * std::mem::size_of::<f32>(),
            KvCacheBuffer::Quantized(buffer) => buffer.memory_usage(),
        }
    }

    fn is_quantized(&self) -> bool {
        matches!(self, KvCacheBuffer::Quantized(_))
    }
}

/// KV cache for a single layer
pub struct KvCache {
    pub keys: KvCacheBuffer,
    pub values: KvCacheBuffer,
    pub seq_len: usize,
    pub max_seq_len: usize,
    pub head_dim: usize,
    pub quant_type: KvQuantType,
}

impl KvCache {
    pub fn new(head_dim: usize, max_seq_len: usize, quant_type: KvQuantType) -> Self {
        let quantized = quant_type != KvQuantType::None;
        KvCache {
            keys: KvCacheBuffer::new(quant_type.key_bits(), head_dim, max_seq_len, quantized),
            values: KvCacheBuffer::new(quant_type.val_bits(), head_dim, max_seq_len, quantized),
            seq_len: 0,
            max_seq_len,
            head_dim,
            quant_type,
        }
    }

    /// Add a token to the cache
    pub fn add(&mut self, key: &[f32], value: &[f32]) {
        if self.seq_len >= self.max_seq_len {
            // Shift cache (simple implementation)
            self.shift();
        }

        self.keys.add(self.seq_len, self.head_dim, key);
        self.values.add(self.seq_len, self.head_dim, value);
        self.seq_len += 1;
    }

    /// Shift cache by one position (for when cache is full)
    fn shift(&mut self) {
        self.keys.shift_left(self.seq_len, self.head_dim);
        self.values.shift_left(self.seq_len, self.head_dim);
        self.seq_len -= 1;
    }

    /// Get the full cache content
    pub fn get_all(&self) -> (Vec<f32>, Vec<f32>) {
        (
            self.keys.get_all(self.seq_len, self.head_dim),
            self.values.get_all(self.seq_len, self.head_dim),
        )
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.seq_len = 0;
    }

    /// Memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.keys.memory_usage() + self.values.memory_usage()
    }

    pub fn is_quantized(&self) -> bool {
        self.keys.is_quantized() || self.values.is_quantized()
    }

    pub fn attention_decode(
        &self,
        q: &[f32],
        query_head_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
    ) -> Vec<f32> {
        let group = (num_q_heads / num_kv_heads).max(1);
        let scale = 1.0 / (query_head_dim as f32).sqrt();
        let mut output = vec![0.0f32; num_q_heads * query_head_dim];

        for qh in 0..num_q_heads {
            let kvh = (qh / group).min(num_kv_heads - 1);
            let mut scores = vec![0.0f32; self.seq_len];

            for (token, score_slot) in scores.iter_mut().enumerate().take(self.seq_len) {
                let mut score = 0.0f32;
                for d in 0..query_head_dim {
                    let dim = kvh * query_head_dim + d;
                    score +=
                        q[qh * query_head_dim + d] * self.keys.value_at(token, dim, self.head_dim);
                }
                *score_slot = score * scale;
            }

            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for score in &mut scores {
                *score = (*score - max_score).exp();
                sum_exp += *score;
            }
            if sum_exp <= 0.0 {
                continue;
            }

            for d in 0..query_head_dim {
                let mut weighted_sum = 0.0f32;
                let dim = kvh * query_head_dim + d;
                for (token, score) in scores.iter().enumerate().take(self.seq_len) {
                    weighted_sum +=
                        (*score / sum_exp) * self.values.value_at(token, dim, self.head_dim);
                }
                output[qh * query_head_dim + d] = weighted_sum;
            }
        }

        output
    }
}

/// Main inference engine
pub struct InferenceEngine {
    pub model: Option<ModelWeights>,
    pub kv_caches: Vec<KvCache>,
    pub config: InferenceConfig,
    pub backend: Box<dyn Backend>,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub ff_dim: usize,
}

impl InferenceEngine {
    /// Create a new inference engine with the given config
    pub fn new(config: Option<InferenceConfig>) -> Self {
        let config = config.unwrap_or_default();
        let backend = BackendFactory::create(Some(config.backend));

        InferenceEngine {
            model: None,
            kv_caches: Vec::new(),
            config,
            backend,
            vocab_size: 0,
            num_layers: 0,
            num_heads: 0,
            num_kv_heads: 0,
            head_dim: 0,
            hidden_size: 0,
            ff_dim: 0,
        }
    }

    /// Load a model from a GGUF file
    pub fn load_model<P: AsRef<Path>>(&mut self, path: P) -> io::Result<()> {
        let model_path = path.as_ref();

        // Parse GGUF header and tensor metadata with mmap-backed parser.
        let metadata = GgufReader::parse_file(model_path)?;

        // Determine architecture
        let arch_str: String = metadata.architecture().unwrap_or("unknown").to_string();
        let arch = ModelArch::from_str(&arch_str);

        // Extract model parameters
        let num_layers = metadata.num_layers().unwrap_or(32);
        let hidden_size = metadata.hidden_size().unwrap_or(4096);
        let ff_dim = metadata.feed_forward_size().unwrap_or(11008);
        let num_heads = metadata.num_attention_heads().unwrap_or(32);
        let num_kv_heads = metadata.num_key_value_heads().unwrap_or(num_heads);
        let vocab_size = metadata.vocab_size().unwrap_or(32000);
        let rope_freq_base = metadata.rope_freq_base();
        let head_dim = hidden_size / num_heads;

        // Update config from metadata
        let mut config = self.config.clone();
        config.max_seq_len = metadata.context_length().unwrap_or(4096);
        config.rope_freq_base = rope_freq_base;

        // Create quant engine
        let weight_format = self
            .detect_weight_quant(&metadata)
            .unwrap_or(QuantFormat::Q4_K);
        config.weight_format = weight_format;
        let mut quant_engine = QuantEngine::new(weight_format, config.kv_quant);

        // Generate rotation matrices for TurboQuant
        quant_engine.generate_rotations(head_dim, num_layers);

        // Create memory pool
        let memory = UnifiedMemory::new();

        // Allocate paged KV cache accounting in the unified memory pool.
        let kv_bytes_per_dim = if config.kv_quant == KvQuantType::None {
            std::mem::size_of::<f32>() * 2
        } else {
            ((config.kv_quant.key_bits() + config.kv_quant.val_bits()) as usize).div_ceil(8)
                + std::mem::size_of::<f32>() * 2
        };

        let mut kv_caches = Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let _ = memory.allocate_kv_cache(
                layer,
                config.max_seq_len,
                num_kv_heads * head_dim,
                kv_bytes_per_dim,
            );
            kv_caches.push(KvCache::new(
                num_kv_heads * head_dim,
                config.max_seq_len,
                config.kv_quant,
            ));
        }

        // Read supported tensor data from GGUF (we store dequantized weights).
        let mut tensors = HashMap::new();
        let mut skipped_tensors = 0usize;
        for tensor_info in &metadata.tensors {
            let elements = tensor_info.num_elements();
            if elements == 0 || elements > 10_000_000 {
                skipped_tensors += 1;
                continue;
            }

            let supported = matches!(
                tensor_info.data_type,
                GgufDataType::F32
                    | GgufDataType::F16
                    | GgufDataType::BF16
                    | GgufDataType::Q8_0
                    | GgufDataType::Q4_0
                    | GgufDataType::Q4_K
                    | GgufDataType::Q2_K
            );
            if !supported {
                skipped_tensors += 1;
                tracing::debug!(
                    "Skipping tensor with unsupported dequantization: {} ({})",
                    tensor_info.name,
                    tensor_info.data_type
                );
                continue;
            }

            tracing::info!(
                "Loading tensor: {} ({} elements, {}, offset={})",
                tensor_info.name,
                elements,
                tensor_info.data_type,
                metadata.absolute_tensor_offset(tensor_info)
            );

            match GgufReader::read_tensor_f32(model_path, &metadata, tensor_info) {
                Ok(data) => {
                    tensors.insert(tensor_info.name.clone(), data);
                }
                Err(e) => {
                    skipped_tensors += 1;
                    tracing::warn!(
                        "Failed to load tensor {} ({}): {}",
                        tensor_info.name,
                        tensor_info.data_type,
                        e
                    );
                }
            }
        }
        if skipped_tensors > 0 {
            tracing::info!(
                "Skipped {} tensors that are too large or not yet supported",
                skipped_tensors
            );
        }

        // Store model
        self.model = Some(ModelWeights {
            metadata,
            arch,
            tensors,
            quant_engine,
            memory,
        });

        self.kv_caches = kv_caches;
        self.config = config;
        self.vocab_size = vocab_size;
        self.num_layers = num_layers;
        self.num_heads = num_heads;
        self.num_kv_heads = num_kv_heads;
        self.head_dim = head_dim;
        self.hidden_size = hidden_size;
        self.ff_dim = ff_dim;

        tracing::info!(
            "Model loaded: {} ({} layers, {} heads, {} dim)",
            arch_str,
            num_layers,
            num_heads,
            hidden_size
        );

        Ok(())
    }

    /// Detect weight quantization format from metadata
    fn detect_weight_quant(&self, metadata: &ModelMetadata) -> Option<QuantFormat> {
        // Try to detect from tensor data types
        for tensor in &metadata.tensors {
            match tensor.data_type {
                GgufDataType::F32 => return Some(QuantFormat::FP32),
                GgufDataType::F16 | GgufDataType::BF16 => return Some(QuantFormat::FP16),
                GgufDataType::Q4_0 => return Some(QuantFormat::Q4_0),
                GgufDataType::Q4_1 => return Some(QuantFormat::Q4_1),
                GgufDataType::Q4_K => return Some(QuantFormat::Q4_K),
                GgufDataType::Q5_0 => return Some(QuantFormat::Q5_0),
                GgufDataType::Q5_1 => return Some(QuantFormat::Q5_1),
                GgufDataType::Q5_K => return Some(QuantFormat::Q5_K),
                GgufDataType::Q6_K => return Some(QuantFormat::Q6_K),
                GgufDataType::Q8_0 => return Some(QuantFormat::Q8_0),
                GgufDataType::Q8_K => return Some(QuantFormat::Q8_K),
                GgufDataType::Q2_K => return Some(QuantFormat::Q2_K),
                GgufDataType::Q3_K => return Some(QuantFormat::Q3_K),
                _ => {}
            }
        }
        None
    }

    fn tensor<'a>(model: &'a ModelWeights, names: &[String]) -> Result<&'a Vec<f32>, String> {
        for name in names {
            if let Some(tensor) = model.get_tensor(name) {
                return Ok(tensor);
            }
        }
        Err(format!("Missing tensor: {}", names.join(" or ")))
    }

    fn layer_tensor_names(layer: usize, role: &str) -> Vec<String> {
        let hf_role = match role {
            "attn_norm" => "input_layernorm",
            "attn_q" => "self_attn.q_proj",
            "attn_k" => "self_attn.k_proj",
            "attn_v" => "self_attn.v_proj",
            "attn_output" => "self_attn.o_proj",
            "ffn_norm" => "post_attention_layernorm",
            "ffn_gate" => "mlp.gate_proj",
            "ffn_up" => "mlp.up_proj",
            "ffn_down" => "mlp.down_proj",
            _ => role,
        };

        vec![
            format!("blk.{}.{}.weight", layer, role),
            format!("model.layers.{}.{}.weight", layer, hf_role),
            format!("layers.{}.{}.weight", layer, hf_role),
        ]
    }

    fn embedding_names() -> Vec<String> {
        vec![
            "token_embd.weight".to_string(),
            "token_embd".to_string(),
            "model.embed_tokens.weight".to_string(),
            "transformer.wte.weight".to_string(),
        ]
    }

    fn output_norm_names() -> Vec<String> {
        vec![
            "output_norm.weight".to_string(),
            "norm.weight".to_string(),
            "model.norm.weight".to_string(),
            "transformer.ln_f.weight".to_string(),
        ]
    }

    fn output_head_names() -> Vec<String> {
        vec![
            "output.weight".to_string(),
            "lm_head.weight".to_string(),
            "model.embed_tokens.weight".to_string(),
            "token_embd.weight".to_string(),
        ]
    }

    fn matvec(weight: &[f32], rows: usize, cols: usize, input: &[f32]) -> Result<Vec<f32>, String> {
        if input.len() != cols {
            return Err(format!(
                "Matvec input shape mismatch: got {}, expected {}",
                input.len(),
                cols
            ));
        }
        if weight.len() < rows * cols {
            return Err(format!(
                "Matvec weight too small: got {}, expected at least {}",
                weight.len(),
                rows * cols
            ));
        }

        let mut output = vec![0.0f32; rows];
        for (row, out_slot) in output.iter_mut().enumerate().take(rows) {
            let mut sum = 0.0f32;
            let offset = row * cols;
            for col in 0..cols {
                sum += weight[offset + col] * input[col];
            }
            *out_slot = sum;
        }
        Ok(output)
    }

    fn add_inplace(dst: &mut [f32], src: &[f32]) {
        for (d, s) in dst.iter_mut().zip(src) {
            *d += *s;
        }
    }

    fn greedy_token(logits: &[f32]) -> u32 {
        logits
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }

    fn sample_token(
        logits: &[f32],
        output: &[u32],
        config: &GenerationConfig,
        rng: &mut StdRng,
    ) -> u32 {
        if logits.is_empty() {
            return 0;
        }

        if config.temperature <= 0.0 {
            return Self::greedy_token(logits);
        }

        let seen: HashSet<u32> = output.iter().copied().collect();
        let penalty = config.repetition_penalty.max(1.0);
        let temperature = config.temperature.max(1e-5);
        let mut candidates: Vec<(usize, f32)> = logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .map(|(idx, mut logit)| {
                if penalty > 1.0 && seen.contains(&(idx as u32)) {
                    if logit >= 0.0 {
                        logit /= penalty;
                    } else {
                        logit *= penalty;
                    }
                }
                (idx, logit / temperature)
            })
            .collect();

        if candidates.is_empty() {
            return 0;
        }

        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        if config.top_k > 0 && candidates.len() > config.top_k {
            candidates.truncate(config.top_k);
        }

        let max_logit = candidates[0].1;
        let mut probs: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|(idx, logit)| (idx, (logit - max_logit).exp()))
            .collect();
        let sum: f32 = probs.iter().map(|(_, prob)| *prob).sum();
        if sum <= 0.0 {
            return probs.first().map(|(idx, _)| *idx as u32).unwrap_or(0);
        }
        for (_, prob) in &mut probs {
            *prob /= sum;
        }

        if config.top_p < 1.0 {
            let top_p = config.top_p.clamp(0.0, 1.0);
            let mut cumulative = 0.0;
            let mut keep = 0usize;
            for (_, prob) in &probs {
                cumulative += *prob;
                keep += 1;
                if cumulative >= top_p {
                    break;
                }
            }
            probs.truncate(keep.max(1));
            let renorm: f32 = probs.iter().map(|(_, prob)| *prob).sum();
            if renorm > 0.0 {
                for (_, prob) in &mut probs {
                    *prob /= renorm;
                }
            }
        }

        let mut draw = rng.gen::<f32>();
        for (idx, prob) in probs {
            if draw <= prob {
                return idx as u32;
            }
            draw -= prob;
        }
        Self::greedy_token(logits)
    }

    fn matched_stop_len(
        output: &[u32],
        prompt_len: usize,
        stop_sequences: &[Vec<u32>],
    ) -> Option<usize> {
        stop_sequences
            .iter()
            .filter(|seq| !seq.is_empty() && output.len() >= prompt_len + seq.len())
            .find(|seq| output.ends_with(seq))
            .map(Vec::len)
    }

    /// Select top-k experts per token from router logits.
    pub fn route_moe_experts(
        router_logits: &[f32],
        num_tokens: usize,
        num_experts: usize,
        top_k: usize,
    ) -> Vec<Vec<MoeRoute>> {
        if num_tokens == 0 || num_experts == 0 || top_k == 0 {
            return Vec::new();
        }

        let mut routes = Vec::with_capacity(num_tokens);
        for token in 0..num_tokens {
            let start = token * num_experts;
            let end = start + num_experts;
            if end > router_logits.len() {
                break;
            }

            let mut experts: Vec<(usize, f32)> = router_logits[start..end]
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, logit)| logit.is_finite())
                .collect();
            experts.sort_by(|a, b| b.1.total_cmp(&a.1));
            experts.truncate(top_k.min(num_experts));

            let max_logit = experts
                .iter()
                .map(|(_, logit)| *logit)
                .fold(f32::NEG_INFINITY, f32::max);
            let mut weights: Vec<MoeRoute> = experts
                .into_iter()
                .map(|(expert, logit)| MoeRoute {
                    expert,
                    weight: (logit - max_logit).exp(),
                })
                .collect();
            let sum: f32 = weights.iter().map(|route| route.weight).sum();
            if sum > 0.0 {
                for route in &mut weights {
                    route.weight /= sum;
                }
            }
            routes.push(weights);
        }

        routes
    }

    /// Dispatch one hidden state through top-k routed SwiGLU experts.
    pub fn dispatch_moe_ffn(
        hidden: &[f32],
        router_logits: &[f32],
        experts: MoeExpertWeights<'_>,
        config: MoeDispatchConfig,
    ) -> Result<Vec<f32>, String> {
        let num_experts = experts.gate.len();
        if num_experts == 0 || experts.up.len() != num_experts || experts.down.len() != num_experts
        {
            return Err("MoE expert weight lists must be non-empty and aligned".to_string());
        }
        if hidden.len() != config.hidden_size {
            return Err(format!(
                "MoE hidden shape mismatch: got {}, expected {}",
                hidden.len(),
                config.hidden_size
            ));
        }

        let routes = Self::route_moe_experts(router_logits, 1, num_experts, config.top_k);
        let routes = routes
            .first()
            .ok_or_else(|| "MoE router produced no routes".to_string())?;
        let mut output = vec![0.0f32; config.hidden_size];

        for route in routes {
            let expert = route.expert;
            let gate = Self::matvec(
                &experts.gate[expert],
                config.ff_dim,
                config.hidden_size,
                hidden,
            )?;
            let up = Self::matvec(
                &experts.up[expert],
                config.ff_dim,
                config.hidden_size,
                hidden,
            )?;
            let mut gated = vec![0.0f32; config.ff_dim];
            for i in 0..config.ff_dim {
                gated[i] = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
            }
            let expert_out = Self::matvec(
                &experts.down[expert],
                config.hidden_size,
                config.ff_dim,
                &gated,
            )?;
            for (dst, value) in output.iter_mut().zip(expert_out) {
                *dst += route.weight * value;
            }
        }

        Ok(output)
    }

    fn apply_rope_heads(values: &mut [f32], position: usize, head_dim: usize, freq_base: f32) {
        for head in values.chunks_exact_mut(head_dim) {
            kernel::rope_single(head, position, head_dim, freq_base);
        }
    }

    fn transformer_logits(&self, model: &ModelWeights, hidden: &[f32]) -> Result<Vec<f32>, String> {
        let default_weight = vec![1.0f32; self.hidden_size];
        let norm_weight =
            Self::tensor(model, &Self::output_norm_names()).unwrap_or(&default_weight);
        let hidden = kernel::rms_norm(hidden, norm_weight, self.config.rms_norm_eps);

        if let Ok(output) = Self::tensor(model, &Self::output_head_names()) {
            let rows = output.len() / self.hidden_size;
            return Self::matvec(output, rows, self.hidden_size, &hidden);
        }

        Ok(hidden)
    }

    /// Run a single stateless forward pass. This is kept for tests and callers
    /// that only need a smoke-test decode without mutating KV cache.
    pub fn decode(&self, token: u32, position: usize) -> Result<Vec<f32>, String> {
        let model = self.model.as_ref().ok_or("Model not loaded")?;

        // Get embedding for this token
        let mut hidden = if let Ok(embedding) = Self::tensor(model, &Self::embedding_names()) {
            let rows = (embedding.len() / self.hidden_size).max(1);
            let start = (token as usize % rows) * self.hidden_size;
            let end = start + self.hidden_size;
            embedding[start..end.min(embedding.len())].to_vec()
        } else {
            // Large GGUF embeddings are not materialized in the smoke-test loader;
            // synthesize a deterministic embedding so the pipeline can be tested.
            (0..self.hidden_size)
                .map(|i| ((token as usize + i) % 1024) as f32 / 1024.0 - 0.5)
                .collect()
        };

        if hidden.len() < self.hidden_size {
            hidden.resize(self.hidden_size, 0.0);
        }

        // Apply RMS norm (simplified - in real model this is per-layer)
        let default_weight = vec![1.0f32; self.hidden_size];
        let rms_weight = model.get_tensor("norm.weight").unwrap_or(&default_weight);

        hidden = kernel::rms_norm(&hidden, rms_weight, self.config.rms_norm_eps);

        // Apply RoPE
        kernel::rope_single(
            &mut hidden,
            position,
            self.head_dim,
            self.config.rope_freq_base,
        );

        Ok(hidden)
    }

    /// Run a single autoregressive decode step through loaded transformer layers.
    pub fn decode_with_cache(&mut self, token: u32, position: usize) -> Result<Vec<f32>, String> {
        let model = self.model.as_ref().ok_or("Model not loaded")?;

        let embedding = Self::tensor(model, &Self::embedding_names())?;
        let vocab_rows = (embedding.len() / self.hidden_size).max(1);
        let start = (token as usize % vocab_rows) * self.hidden_size;
        let end = start + self.hidden_size;
        if end > embedding.len() {
            return Err("Token embedding tensor has an invalid shape".to_string());
        }
        let mut hidden = embedding[start..end].to_vec();

        let kv_dim = self.num_kv_heads * self.head_dim;
        for layer in 0..self.num_layers {
            let attn_norm = Self::tensor(model, &Self::layer_tensor_names(layer, "attn_norm"))?;
            let q_weight = Self::tensor(model, &Self::layer_tensor_names(layer, "attn_q"))?;
            let k_weight = Self::tensor(model, &Self::layer_tensor_names(layer, "attn_k"))?;
            let v_weight = Self::tensor(model, &Self::layer_tensor_names(layer, "attn_v"))?;
            let o_weight = Self::tensor(model, &Self::layer_tensor_names(layer, "attn_output"))?;

            let normed = kernel::rms_norm(&hidden, attn_norm, self.config.rms_norm_eps);
            let mut q = Self::matvec(
                q_weight,
                self.num_heads * self.head_dim,
                self.hidden_size,
                &normed,
            )?;
            let mut k = Self::matvec(k_weight, kv_dim, self.hidden_size, &normed)?;
            let v = Self::matvec(v_weight, kv_dim, self.hidden_size, &normed)?;

            Self::apply_rope_heads(&mut q, position, self.head_dim, self.config.rope_freq_base);
            Self::apply_rope_heads(&mut k, position, self.head_dim, self.config.rope_freq_base);

            let cache = self
                .kv_caches
                .get_mut(layer)
                .ok_or_else(|| format!("Missing KV cache for layer {}", layer))?;
            cache.add(&k, &v);
            let attn = cache.attention_decode(&q, self.head_dim, self.num_heads, self.num_kv_heads);
            let attn_out = Self::matvec(o_weight, self.hidden_size, self.hidden_size, &attn)?;
            Self::add_inplace(&mut hidden, &attn_out);

            let ffn_norm = Self::tensor(model, &Self::layer_tensor_names(layer, "ffn_norm"))?;
            let gate_weight = Self::tensor(model, &Self::layer_tensor_names(layer, "ffn_gate"))?;
            let up_weight = Self::tensor(model, &Self::layer_tensor_names(layer, "ffn_up"))?;
            let down_weight = Self::tensor(model, &Self::layer_tensor_names(layer, "ffn_down"))?;

            let normed = kernel::rms_norm(&hidden, ffn_norm, self.config.rms_norm_eps);
            let gate = Self::matvec(gate_weight, self.ff_dim, self.hidden_size, &normed)?;
            let up = Self::matvec(up_weight, self.ff_dim, self.hidden_size, &normed)?;
            let mut gated = vec![0.0f32; self.ff_dim];
            for i in 0..self.ff_dim {
                gated[i] = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
            }
            let ffn_out = Self::matvec(down_weight, self.hidden_size, self.ff_dim, &gated)?;
            Self::add_inplace(&mut hidden, &ffn_out);
        }

        self.transformer_logits(model, &hidden)
    }

    /// Run a batch of tokens through the model (prefill)
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Ok(vec![]);
        }

        for cache in &mut self.kv_caches {
            cache.clear();
        }

        let mut logits = Vec::new();
        for (position, &token) in tokens.iter().enumerate() {
            logits = match self.decode_with_cache(token, position) {
                Ok(logits) => logits,
                Err(err) => {
                    tracing::debug!("Falling back to stateless decode during prefill: {}", err);
                    self.decode(token, position)?
                }
            };
        }

        Ok(logits)
    }

    /// Generate text by autoregressive decoding
    pub fn generate(&mut self, prompt: &[u32], max_tokens: usize) -> Result<Vec<u32>, String> {
        self.generate_with_config(
            prompt,
            &GenerationConfig {
                max_tokens,
                ..Default::default()
            },
        )
    }

    /// Generate text by autoregressive decoding with sampling controls.
    pub fn generate_with_config(
        &mut self,
        prompt: &[u32],
        generation: &GenerationConfig,
    ) -> Result<Vec<u32>, String> {
        let mut output = prompt.to_vec();

        let mut logits = self.prefill(prompt)?;
        let mut rng = StdRng::seed_from_u64(generation.seed.unwrap_or(0x5eed_5eed));

        // Autoregressive decode
        for _ in 0..generation.max_tokens {
            let next_token = Self::sample_token(&logits, &output, generation, &mut rng);

            output.push(next_token);
            if let Some(stop_len) =
                Self::matched_stop_len(&output, prompt.len(), &generation.stop_sequences)
            {
                let new_len = output.len().saturating_sub(stop_len);
                output.truncate(new_len);
                break;
            }

            let position = output.len() - 1;
            logits = match self.decode_with_cache(next_token, position) {
                Ok(logits) => logits,
                Err(err) => {
                    tracing::debug!(
                        "Falling back to stateless decode during generation: {}",
                        err
                    );
                    self.decode(next_token, position)?
                }
            };
        }

        Ok(output)
    }

    /// Generate text with external draft tokens verified against the target model.
    ///
    /// Accepted draft tokens are appended directly; rejected draft tokens are
    /// replaced with the target model's sampled token for that position.
    pub fn generate_with_draft_tokens(
        &mut self,
        prompt: &[u32],
        generation: &GenerationConfig,
        draft_tokens: &[u32],
    ) -> Result<(Vec<u32>, SpeculativeStats), String> {
        let mut output = prompt.to_vec();
        let mut logits = self.prefill(prompt)?;
        let mut rng = StdRng::seed_from_u64(generation.seed.unwrap_or(0x5eed_5eed));
        let mut draft_index = 0usize;
        let mut stats = SpeculativeStats::default();

        for _ in 0..generation.max_tokens {
            let verified = Self::sample_token(&logits, &output, generation, &mut rng);
            let next_token = if let Some(&drafted) = draft_tokens.get(draft_index) {
                draft_index += 1;
                if drafted == verified {
                    stats.accepted += 1;
                    drafted
                } else {
                    stats.rejected += 1;
                    verified
                }
            } else {
                verified
            };

            output.push(next_token);
            if let Some(stop_len) =
                Self::matched_stop_len(&output, prompt.len(), &generation.stop_sequences)
            {
                let new_len = output.len().saturating_sub(stop_len);
                output.truncate(new_len);
                break;
            }

            let position = output.len() - 1;
            logits = match self.decode_with_cache(next_token, position) {
                Ok(logits) => logits,
                Err(err) => {
                    tracing::debug!(
                        "Falling back to stateless decode during speculative generation: {}",
                        err
                    );
                    self.decode(next_token, position)?
                }
            };
        }

        Ok((output, stats))
    }

    /// Run speculative decoding with a separate draft engine.
    pub fn generate_with_draft_engine(
        &mut self,
        draft: &mut InferenceEngine,
        prompt: &[u32],
        generation: &GenerationConfig,
        draft_window: usize,
    ) -> Result<(Vec<u32>, SpeculativeStats), String> {
        let mut output = prompt.to_vec();
        let mut stats = SpeculativeStats::default();
        let window = draft_window.max(1);

        while output.len().saturating_sub(prompt.len()) < generation.max_tokens {
            let remaining = generation
                .max_tokens
                .saturating_sub(output.len().saturating_sub(prompt.len()));
            let draft_generation = GenerationConfig {
                max_tokens: remaining.min(window),
                ..generation.clone()
            };
            let proposed = draft.generate_with_config(&output, &draft_generation)?;
            let draft_tokens = proposed[output.len().min(proposed.len())..].to_vec();

            let verify_generation = GenerationConfig {
                max_tokens: draft_tokens.len().max(1).min(remaining),
                ..generation.clone()
            };
            let before_len = output.len();
            let (verified, step_stats) =
                self.generate_with_draft_tokens(&output, &verify_generation, &draft_tokens)?;
            output = verified;
            stats.accepted += step_stats.accepted;
            stats.rejected += step_stats.rejected;

            if output.len() == before_len {
                break;
            }
            if Self::matched_stop_len(&output, prompt.len(), &generation.stop_sequences).is_some() {
                break;
            }
        }

        Ok((output, stats))
    }

    /// Get model info as a summary string
    pub fn info(&self) -> String {
        let model = self.model.as_ref().map(|m| {
            format!(
                "Architecture: {}\nLayers: {}\nHeads: {} (Q: {}) (KV: {})\nHidden: {}\nFFN: {}\nVocab: {}\nRope base: {}",
                m.architecture(),
                self.num_layers,
                self.num_heads,
                self.num_heads,
                self.num_kv_heads,
                self.hidden_size,
                self.ff_dim,
                self.vocab_size,
                self.config.rope_freq_base,
            )
        }).unwrap_or_else(|| "No model loaded".to_string());

        let kv_mem = self
            .kv_caches
            .iter()
            .map(|kv| kv.memory_usage())
            .sum::<usize>();

        format!(
            "{}\n\nBackend: {}\nKV Quant: {}\nWeight Quant: {}\nMax seq len: {}\nKV cache memory: {}",
            model,
            self.backend.name(),
            self.config.kv_quant.name(),
            self.config.weight_format.name(),
            self.config.max_seq_len,
            crate::memory::format_size(kv_mem),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn identity(rows: usize, cols: usize) -> Vec<f32> {
        let mut weight = vec![0.0; rows * cols];
        for i in 0..rows.min(cols) {
            weight[i * cols + i] = 1.0;
        }
        weight
    }

    fn tiny_stateless_engine() -> InferenceEngine {
        let mut tensors = HashMap::new();
        tensors.insert(
            "token_embd.weight".to_string(),
            vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5],
        );
        tensors.insert("output_norm.weight".to_string(), vec![1.0; 2]);
        tensors.insert(
            "output.weight".to_string(),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        );

        let model = ModelWeights {
            metadata: ModelMetadata {
                version: 3,
                tensor_count: tensors.len() as u64,
                kv_count: 0,
                byte_order: crate::gguf::GgufByteOrder::LittleEndian,
                alignment: 32,
                tensor_data_offset: 0,
                metadata: HashMap::new(),
                tensors: Vec::new(),
            },
            arch: ModelArch::Llama,
            tensors,
            quant_engine: QuantEngine::new(QuantFormat::FP32, KvQuantType::None),
            memory: UnifiedMemory::new(),
        };

        let mut engine = InferenceEngine::new(Some(InferenceConfig {
            kv_quant: KvQuantType::None,
            weight_format: QuantFormat::FP32,
            ..Default::default()
        }));
        engine.model = Some(model);
        engine.vocab_size = 3;
        engine.num_layers = 0;
        engine.num_heads = 1;
        engine.num_kv_heads = 1;
        engine.head_dim = 2;
        engine.hidden_size = 2;
        engine.ff_dim = 2;
        engine
    }

    #[test]
    fn test_engine_creation() {
        let engine = InferenceEngine::new(None);
        assert!(engine.model.is_none());
        assert_eq!(engine.config.max_seq_len, 4096);
    }

    #[test]
    fn test_kv_cache() {
        let mut cache = KvCache::new(8, 4, KvQuantType::None);

        let key = vec![1.0f32; 8];
        let value = vec![2.0f32; 8];

        cache.add(&key, &value);
        assert_eq!(cache.seq_len, 1);

        cache.add(&key, &value);
        assert_eq!(cache.seq_len, 2);

        let (keys, values) = cache.get_all();
        assert_eq!(keys.len(), 16); // 2 * 8
        assert_eq!(values.len(), 16);
    }

    #[test]
    fn test_kv_cache_clear() {
        let mut cache = KvCache::new(8, 4, KvQuantType::None);
        cache.add(&[1.0; 8], &[2.0; 8]);
        cache.clear();
        assert_eq!(cache.seq_len, 0);
    }

    #[test]
    fn test_kv_cache_memory() {
        let cache_fp16 = KvCache::new(64, 1024, KvQuantType::None);
        let cache_tq = KvCache::new(64, 1024, KvQuantType::TurboQuant3b2b);

        assert!(!cache_fp16.is_quantized());
        assert!(cache_tq.is_quantized());
        assert!(cache_fp16.memory_usage() > cache_tq.memory_usage());
    }

    #[test]
    fn test_quantized_kv_cache_roundtrip() {
        let mut cache = KvCache::new(8, 4, KvQuantType::TurboQuant4b4b);
        let key = vec![-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0];
        let value = vec![1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -1.0];

        cache.add(&key, &value);
        let (keys, values) = cache.get_all();

        assert_eq!(keys.len(), 8);
        assert_eq!(values.len(), 8);
        for (actual, expected) in keys.iter().zip(key.iter()) {
            assert!((actual - expected).abs() <= 0.15);
        }
        for (actual, expected) in values.iter().zip(value.iter()) {
            assert!((actual - expected).abs() <= 0.15);
        }
    }

    #[test]
    fn test_quantized_kv_attention_direct_path() {
        let mut cache = KvCache::new(2, 4, KvQuantType::TurboQuant4b4b);
        cache.add(&[1.0, 0.0], &[0.25, 0.75]);
        cache.add(&[0.0, 1.0], &[0.5, 0.5]);

        let output = cache.attention_decode(&[1.0, 0.0], 2, 1, 1);
        assert_eq!(output.len(), 2);
        assert!(output[0].is_finite());
        assert!(output[1].is_finite());
    }

    #[test]
    fn test_model_arch() {
        assert_eq!(ModelArch::from_str("llama"), ModelArch::Llama);
        assert_eq!(ModelArch::from_str("mistral"), ModelArch::Mistral);
        assert_eq!(
            ModelArch::from_str("unknown_model"),
            ModelArch::Unknown("unknown_model".to_string())
        );
    }

    #[test]
    fn test_inference_config_defaults() {
        let config = InferenceConfig::default();
        assert_eq!(config.max_seq_len, 4096);
        assert_eq!(config.batch_size, 1);
        assert_eq!(config.kv_quant, KvQuantType::TurboQuant3b2b);
    }

    #[test]
    fn test_kv_cache_shift() {
        let mut cache = KvCache::new(4, 3, KvQuantType::None);

        // Add 3 tokens (fill cache)
        cache.add(&[1.0; 4], &[2.0; 4]);
        cache.add(&[3.0; 4], &[4.0; 4]);
        cache.add(&[5.0; 4], &[6.0; 4]);

        assert_eq!(cache.seq_len, 3);

        // Add one more (should shift)
        cache.add(&[7.0; 4], &[8.0; 4]);
        assert_eq!(cache.seq_len, 3);
    }

    #[test]
    fn test_decode_with_cache_runs_transformer_layer() {
        let mut tensors = HashMap::new();
        tensors.insert(
            "token_embd.weight".to_string(),
            vec![
                0.1, 0.2, 0.3, 0.4, 0.2, 0.1, 0.4, 0.3, 0.3, 0.4, 0.1, 0.2, 0.4, 0.3, 0.2, 0.1,
                0.5, 0.1, 0.1, 0.5, 0.1, 0.5, 0.5, 0.1,
            ],
        );
        tensors.insert("blk.0.attn_norm.weight".to_string(), vec![1.0; 4]);
        tensors.insert("blk.0.attn_q.weight".to_string(), identity(4, 4));
        tensors.insert("blk.0.attn_k.weight".to_string(), identity(2, 4));
        tensors.insert("blk.0.attn_v.weight".to_string(), identity(2, 4));
        tensors.insert("blk.0.attn_output.weight".to_string(), identity(4, 4));
        tensors.insert("blk.0.ffn_norm.weight".to_string(), vec![1.0; 4]);
        tensors.insert("blk.0.ffn_gate.weight".to_string(), vec![0.05; 8 * 4]);
        tensors.insert("blk.0.ffn_up.weight".to_string(), vec![0.04; 8 * 4]);
        tensors.insert("blk.0.ffn_down.weight".to_string(), vec![0.03; 4 * 8]);
        tensors.insert("output_norm.weight".to_string(), vec![1.0; 4]);
        tensors.insert("output.weight".to_string(), vec![0.02; 6 * 4]);

        let model = ModelWeights {
            metadata: ModelMetadata {
                version: 3,
                tensor_count: tensors.len() as u64,
                kv_count: 0,
                byte_order: crate::gguf::GgufByteOrder::LittleEndian,
                alignment: 32,
                tensor_data_offset: 0,
                metadata: HashMap::new(),
                tensors: Vec::new(),
            },
            arch: ModelArch::Llama,
            tensors,
            quant_engine: QuantEngine::new(QuantFormat::FP32, KvQuantType::None),
            memory: UnifiedMemory::new(),
        };

        let mut engine = InferenceEngine::new(Some(InferenceConfig {
            kv_quant: KvQuantType::None,
            weight_format: QuantFormat::FP32,
            ..Default::default()
        }));
        engine.model = Some(model);
        engine.kv_caches = vec![KvCache::new(2, 8, KvQuantType::None)];
        engine.vocab_size = 6;
        engine.num_layers = 1;
        engine.num_heads = 2;
        engine.num_kv_heads = 1;
        engine.head_dim = 2;
        engine.hidden_size = 4;
        engine.ff_dim = 8;

        let logits = engine.decode_with_cache(0, 0).unwrap();
        assert_eq!(logits.len(), 6);
        assert_eq!(engine.kv_caches[0].seq_len, 1);
    }

    #[test]
    fn test_greedy_sampling_picks_largest_logit() {
        let mut rng = StdRng::seed_from_u64(7);
        let config = GenerationConfig {
            temperature: 0.0,
            ..Default::default()
        };
        let token = InferenceEngine::sample_token(&[0.1, 3.0, 1.0], &[], &config, &mut rng);
        assert_eq!(token, 1);
    }

    #[test]
    fn test_stop_sequence_match() {
        let output = vec![1, 2, 3, 4];
        let stop = vec![vec![3, 4]];
        assert_eq!(
            InferenceEngine::matched_stop_len(&output, 1, &stop),
            Some(2)
        );
    }

    #[test]
    fn test_route_moe_experts_selects_top_k() {
        let routes = InferenceEngine::route_moe_experts(&[0.1, 3.0, 1.0, 2.0, 0.5, 0.2], 2, 3, 2);
        assert_eq!(routes.len(), 2);
        assert_eq!(routes[0][0].expert, 1);
        assert_eq!(routes[0][1].expert, 2);
        let sum: f32 = routes[0].iter().map(|route| route.weight).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dispatch_moe_ffn_runs_top_k_experts() {
        let hidden = vec![1.0, 2.0];
        let expert_gate = vec![identity(2, 2), vec![0.5; 4]];
        let expert_up = vec![identity(2, 2), vec![0.25; 4]];
        let expert_down = vec![identity(2, 2), vec![0.1; 4]];
        let output = InferenceEngine::dispatch_moe_ffn(
            &hidden,
            &[0.0, 4.0],
            MoeExpertWeights {
                gate: &expert_gate,
                up: &expert_up,
                down: &expert_down,
            },
            MoeDispatchConfig {
                top_k: 1,
                hidden_size: 2,
                ff_dim: 2,
            },
        )
        .unwrap();
        assert_eq!(output.len(), 2);
        assert!(output[0].is_finite());
        assert!(output[1].is_finite());
    }

    #[test]
    fn test_speculative_generation_verifies_draft_tokens() {
        let mut engine = tiny_stateless_engine();

        let config = GenerationConfig {
            max_tokens: 2,
            temperature: 0.0,
            ..Default::default()
        };
        let (output, stats) = engine
            .generate_with_draft_tokens(&[0], &config, &[1, 1])
            .unwrap();

        assert_eq!(output, vec![0, 1, 2]);
        assert_eq!(
            stats,
            SpeculativeStats {
                accepted: 1,
                rejected: 1,
            }
        );
    }

    #[test]
    fn test_speculative_generation_uses_draft_engine() {
        let mut target = tiny_stateless_engine();
        let mut draft = tiny_stateless_engine();
        let config = GenerationConfig {
            max_tokens: 2,
            temperature: 0.0,
            ..Default::default()
        };

        let (output, stats) = target
            .generate_with_draft_engine(&mut draft, &[0], &config, 2)
            .unwrap();

        assert_eq!(output, vec![0, 1, 2]);
        assert_eq!(stats.accepted, 2);
        assert_eq!(stats.rejected, 0);
    }

    #[test]
    fn test_tiny_model_prefill_logits_are_stable() {
        let mut tensors = HashMap::new();
        tensors.insert("token_embd.weight".to_string(), vec![1.0, 0.0, 0.0, 1.0]);
        tensors.insert("output_norm.weight".to_string(), vec![1.0; 2]);
        tensors.insert("output.weight".to_string(), vec![0.0, 0.0, 1.0, 0.0]);

        let model = ModelWeights {
            metadata: ModelMetadata {
                version: 3,
                tensor_count: tensors.len() as u64,
                kv_count: 0,
                byte_order: crate::gguf::GgufByteOrder::LittleEndian,
                alignment: 32,
                tensor_data_offset: 0,
                metadata: HashMap::new(),
                tensors: Vec::new(),
            },
            arch: ModelArch::Llama,
            tensors,
            quant_engine: QuantEngine::new(QuantFormat::FP32, KvQuantType::None),
            memory: UnifiedMemory::new(),
        };

        let mut engine = InferenceEngine::new(Some(InferenceConfig {
            kv_quant: KvQuantType::None,
            weight_format: QuantFormat::FP32,
            rms_norm_eps: 0.0,
            ..Default::default()
        }));
        engine.model = Some(model);
        engine.vocab_size = 2;
        engine.num_layers = 0;
        engine.num_heads = 1;
        engine.num_kv_heads = 1;
        engine.head_dim = 2;
        engine.hidden_size = 2;
        engine.ff_dim = 2;

        let logits = engine.prefill(&[0]).unwrap();
        assert_eq!(logits.len(), 2);
        assert!((logits[0] - 0.0).abs() < 1e-6);
        assert!((logits[1] - std::f32::consts::SQRT_2).abs() < 1e-6);
    }
}
