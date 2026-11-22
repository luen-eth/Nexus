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
    /// Runtime weight storage. Only FP32 is implemented; source GGUF codecs are independent.
    pub weight_format: QuantFormat,
    /// Backend to use
    pub backend: BackendType,
    /// Rope frequency base
    pub rope_freq_base: f32,
    /// RMS norm epsilon
    pub rms_norm_eps: f32,
    /// Optional cap for eagerly materialized tensor elements.
    /// Set to None to attempt loading all supported tensors.
    pub max_loaded_tensor_elements: Option<usize>,
    /// Explicitly allow incomplete byte/vocabulary fixtures, never production defaults.
    pub allow_byte_tokenizer: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        InferenceConfig {
            max_seq_len: 4096,
            batch_size: 1,
            // Unquantized by default: TurboQuant 3b/2b rescales every token
            // with a single min/max pair and visibly degrades quality. It is
            // an explicit opt-in, not a silent default.
            kv_quant: KvQuantType::None,
            weight_format: QuantFormat::FP32,
            backend: BackendType::default(),
            rope_freq_base: 10000.0,
            rms_norm_eps: 1e-5,
            max_loaded_tensor_elements: None,
            allow_byte_tokenizer: false,
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

/// A single active generation with private KV state and sampling RNG.
pub struct GenerationSession {
    pub output: Vec<u32>,
    pub prompt_len: usize,
    pub finished: bool,
    pub finish_reason: &'static str,
    caches: Vec<KvCache>,
    logits: Vec<f32>,
    generation: GenerationConfig,
    rng: StdRng,
    eos: Option<u32>,
    prefill_position: usize,
}

impl GenerationSession {
    /// Bytes in this session's allocated KV payload, including packed scales/minima.
    /// Does not include engine caches, weights, allocator overhead or GPU mirrors.
    pub fn kv_cache_memory_usage(&self) -> usize {
        self.caches.iter().map(KvCache::memory_usage).sum()
    }

    /// Completion tokens that are safe to send without leaking a partial stop string.
    pub fn visible_output(&self) -> &[u32] {
        let generated = &self.output[self.prompt_len..];
        let held = if self.finished {
            0
        } else {
            self.generation
                .stop_sequences
                .iter()
                .map(|stop| {
                    (1..stop.len().min(generated.len() + 1))
                        .rev()
                        .find(|&len| generated.ends_with(&stop[..len]))
                        .unwrap_or(0)
                })
                .max()
                .unwrap_or(0)
        };
        &generated[..generated.len() - held]
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
    /// Construct a packed physical-slot buffer. Supported widths are 1..=8 bits.
    pub fn new(bits: u32, head_dim: usize, max_seq_len: usize) -> Self {
        Self::try_new(bits, head_dim, max_seq_len).expect("Invalid packed KV buffer configuration")
    }

    pub fn try_new(bits: u32, head_dim: usize, max_seq_len: usize) -> Result<Self, String> {
        if !(1..=8).contains(&bits) || head_dim == 0 || max_seq_len == 0 {
            return Err("Packed KV requires 1..=8 bits, a positive dimension and capacity".into());
        }
        let bytes_per_token = head_dim
            .checked_mul(bits as usize)
            .ok_or("Packed KV dimension overflow")?
            .div_ceil(8);
        let byte_len = bytes_per_token
            .checked_mul(max_seq_len)
            .ok_or("Packed KV capacity overflow")?;
        Ok(Self {
            data: zeroed_kv_bytes(byte_len)?,
            mins: zeroed_kv_floats(max_seq_len)?,
            scales: unit_kv_scales(max_seq_len)?,
            bits,
            head_dim,
            bytes_per_token,
        })
    }

    pub fn add(&mut self, position: usize, values: &[f32]) {
        assert_eq!(values.len(), self.head_dim, "KV vector dimension mismatch");
        assert!(
            position < self.mins.len(),
            "KV physical position exceeds capacity"
        );

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
        assert!(seq_len <= self.mins.len(), "KV length exceeds capacity");
        let mut output = vec![0.0; seq_len * self.head_dim];
        for (position, row) in output.chunks_exact_mut(self.head_dim).enumerate() {
            self.decode_range(position, 0, row);
        }
        output
    }

    pub fn value_at(&self, position: usize, dim: usize) -> f32 {
        assert!(
            position < self.mins.len() && dim < self.head_dim,
            "KV index exceeds dimensions"
        );
        let start = position * self.bytes_per_token;
        let q = self.read_bits(start, dim);
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

    fn decode_range(&self, position: usize, first_dim: usize, output: &mut [f32]) {
        let start = position * self.bytes_per_token;
        let min = self.mins[position];
        let scale = self.scales[position];
        for (dim, slot) in output.iter_mut().enumerate() {
            *slot = min + self.read_bits(start, first_dim + dim) as f32 * scale;
        }
    }

    // Every field is <=8 bits, so even a byte-straddling field touches at most
    // two bytes. Rows are cleared by add before OR writes; neighbouring fields
    // and the final row's unused padding bits are never read as payload.
    #[inline]
    fn write_bits(&mut self, token_start: usize, index: usize, value: u32) {
        let offset = index * self.bits as usize;
        let byte = token_start + offset / 8;
        let shift = offset % 8;
        let packed = (value as u16) << shift;
        self.data[byte] |= packed as u8;
        if shift + self.bits as usize > 8 {
            self.data[byte + 1] |= (packed >> 8) as u8;
        }
    }

    #[inline]
    fn read_bits(&self, token_start: usize, index: usize) -> u32 {
        let offset = index * self.bits as usize;
        let byte = token_start + offset / 8;
        let shift = offset % 8;
        let mut packed = u16::from(self.data[byte]);
        if shift + self.bits as usize > 8 {
            packed |= u16::from(self.data[byte + 1]) << 8;
        }
        u32::from((packed >> shift) & ((1u16 << self.bits) - 1))
    }
}

// Using the global allocator's zeroed allocation preserves demand-zero pages
// on supporting allocators/operating systems. Reserving then filling a Vec
// eagerly writes the entire maximum-context cache, even for a short session.
// Keep these helpers concrete: all-zero bits are valid u8 and positive f32
// zero, but would not be a valid initialization for an arbitrary Rust type.
fn zeroed_kv_bytes(length: usize) -> Result<Vec<u8>, String> {
    if length == 0 {
        return Ok(Vec::new());
    }
    let layout = std::alloc::Layout::array::<u8>(length)
        .map_err(|error| format!("Invalid KV byte buffer layout: {error}"))?;
    // SAFETY: length is nonzero, and Layout::array checked the size bound.
    // alloc_zeroed uses the same global allocator that Vec uses on drop.
    let pointer = unsafe { std::alloc::alloc_zeroed(layout) };
    if pointer.is_null() {
        return Err("Cannot allocate zeroed KV byte buffer".into());
    }
    // SAFETY: the allocation has exactly the u8 alignment and length capacity;
    // every element is initialized to a valid zero, and ownership moves to Vec.
    Ok(unsafe { Vec::from_raw_parts(pointer, length, length) })
}

fn zeroed_kv_floats(length: usize) -> Result<Vec<f32>, String> {
    if length == 0 {
        return Ok(Vec::new());
    }
    let layout = std::alloc::Layout::array::<f32>(length)
        .map_err(|error| format!("Invalid KV float buffer layout: {error}"))?;
    // SAFETY: length is nonzero, and Layout::array checked both byte-size and
    // isize bounds. The global allocation is aligned for f32 and Vec-compatible.
    let pointer = unsafe { std::alloc::alloc_zeroed(layout) }.cast::<f32>();
    if pointer.is_null() {
        return Err("Cannot allocate zeroed KV float buffer".into());
    }
    // SAFETY: all-zero bits represent initialized +0.0 f32 elements. Layout,
    // length and capacity match exactly, and Vec takes sole allocation ownership.
    Ok(unsafe { Vec::from_raw_parts(pointer, length, length) })
}

fn unit_kv_scales(length: usize) -> Result<Vec<f32>, String> {
    let mut data = Vec::new();
    data.try_reserve_exact(length)
        .map_err(|error| format!("Cannot allocate KV buffer: {error}"))?;
    data.resize(length, 1.0);
    Ok(data)
}

/// KV cache buffer for one key or value stream.
pub enum KvCacheBuffer {
    Float(Vec<f32>),
    Quantized(PackedKvBuffer),
}

impl KvCacheBuffer {
    fn try_new(
        bits: u32,
        head_dim: usize,
        max_seq_len: usize,
        quantized: bool,
    ) -> Result<Self, String> {
        if quantized {
            Ok(Self::Quantized(PackedKvBuffer::try_new(
                bits,
                head_dim,
                max_seq_len,
            )?))
        } else {
            let length = head_dim
                .checked_mul(max_seq_len)
                .ok_or("KV float capacity overflow")?;
            Ok(Self::Float(zeroed_kv_floats(length)?))
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

    // Float caches borrow their row directly; packed caches dequantize only a
    // single KV head into a scratch vector reused by every corresponding Q head.
    fn head_slice<'a>(
        &'a self,
        position: usize,
        first_dim: usize,
        token_dim: usize,
        scratch: &'a mut [f32],
    ) -> &'a [f32] {
        match self {
            Self::Float(data) => {
                let start = position * token_dim + first_dim;
                &data[start..start + scratch.len()]
            }
            Self::Quantized(buffer) => {
                buffer.decode_range(position, first_dim, scratch);
                scratch
            }
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
    // Physical slot for logical token zero. Insertion only overwrites one slot.
    start_slot: usize,
}

impl KvCache {
    pub fn new(head_dim: usize, max_seq_len: usize, quant_type: KvQuantType) -> Self {
        Self::try_new(head_dim, max_seq_len, quant_type).expect("Invalid KV cache configuration")
    }

    /// Fallible constructor for public callers handling untrusted dimensions.
    pub fn try_new(
        head_dim: usize,
        max_seq_len: usize,
        quant_type: KvQuantType,
    ) -> Result<Self, String> {
        if head_dim == 0 || max_seq_len == 0 {
            return Err("KV cache requires a positive dimension and capacity".into());
        }
        let quantized = quant_type != KvQuantType::None;
        Ok(Self {
            keys: KvCacheBuffer::try_new(quant_type.key_bits(), head_dim, max_seq_len, quantized)?,
            values: KvCacheBuffer::try_new(
                quant_type.val_bits(),
                head_dim,
                max_seq_len,
                quantized,
            )?,
            seq_len: 0,
            max_seq_len,
            head_dim,
            quant_type,
            start_slot: 0,
        })
    }

    /// Append a physical slot; full caches overwrite the oldest token in O(head_dim).
    /// This storage primitive does not alter the engine's context or RoPE policy.
    pub fn add(&mut self, key: &[f32], value: &[f32]) {
        assert_eq!(key.len(), self.head_dim, "KV key dimension mismatch");
        assert_eq!(value.len(), self.head_dim, "KV value dimension mismatch");
        assert!(self.seq_len <= self.max_seq_len, "Invalid KV cache length");
        let full = self.seq_len == self.max_seq_len;
        let position = if full {
            self.start_slot
        } else {
            self.physical_slot(self.seq_len)
        };
        self.keys.add(position, self.head_dim, key);
        self.values.add(position, self.head_dim, value);
        if full {
            self.start_slot = if position + 1 == self.max_seq_len {
                0
            } else {
                position + 1
            };
        } else {
            self.seq_len += 1;
        }
    }

    #[inline]
    fn physical_slot(&self, logical: usize) -> usize {
        let tail = self.max_seq_len - self.start_slot;
        if logical < tail {
            self.start_slot + logical
        } else {
            logical - tail
        }
    }

    /// Snapshot in oldest-to-newest logical order, independent of physical wraps.
    pub fn get_all(&self) -> (Vec<f32>, Vec<f32>) {
        let mut keys = vec![0.0; self.seq_len * self.head_dim];
        let mut values = vec![0.0; self.seq_len * self.head_dim];
        let mut scratch = vec![0.0; self.head_dim];
        for logical in 0..self.seq_len {
            let position = self.physical_slot(logical);
            let start = logical * self.head_dim;
            keys[start..start + self.head_dim].copy_from_slice(self.keys.head_slice(
                position,
                0,
                self.head_dim,
                &mut scratch,
            ));
            values[start..start + self.head_dim].copy_from_slice(self.values.head_slice(
                position,
                0,
                self.head_dim,
                &mut scratch,
            ));
        }
        (keys, values)
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.seq_len = 0;
        self.start_slot = 0;
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
        assert!(
            query_head_dim > 0 && num_kv_heads > 0 && num_q_heads > 0,
            "Attention dimensions must be positive"
        );
        assert_eq!(
            num_q_heads % num_kv_heads,
            0,
            "Q heads must be divisible by KV heads"
        );
        assert_eq!(
            num_kv_heads.checked_mul(query_head_dim),
            Some(self.head_dim),
            "Attention/cache dimensions disagree"
        );
        assert_eq!(
            num_q_heads.checked_mul(query_head_dim),
            Some(q.len()),
            "Invalid query shape"
        );
        let group = num_q_heads / num_kv_heads;
        let scale = 1.0 / (query_head_dim as f32).sqrt();
        let mut output = vec![0.0f32; q.len()];
        if self.seq_len == 0 {
            return output;
        }
        let score_len = group
            .checked_mul(self.seq_len)
            .expect("Attention scratch size overflow");
        let mut scores = vec![0.0f32; score_len];
        let mut sums = vec![0.0f32; group];
        let mut scratch = vec![0.0f32; query_head_dim];

        for kvh in 0..num_kv_heads {
            let first_dim = kvh * query_head_dim;
            let first_qh = kvh * group;
            // Decode each key element once per KV head, not once per query head.
            for token in 0..self.seq_len {
                let key = self.keys.head_slice(
                    self.physical_slot(token),
                    first_dim,
                    self.head_dim,
                    &mut scratch,
                );
                for qh in 0..group {
                    let query =
                        &q[(first_qh + qh) * query_head_dim..(first_qh + qh + 1) * query_head_dim];
                    let mut score = 0.0;
                    for d in 0..query_head_dim {
                        score += query[d] * key[d];
                    }
                    scores[qh * self.seq_len + token] = score * scale;
                }
            }
            for (qh, sum_slot) in sums.iter_mut().enumerate() {
                let row = &mut scores[qh * self.seq_len..(qh + 1) * self.seq_len];
                let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for value in row {
                    *value = (*value - max).exp();
                    sum += *value;
                }
                *sum_slot = sum;
            }
            // Token-major value reads retain ascending-token accumulation for each
            // output coordinate, preserving the previous F32 summation order.
            for token in 0..self.seq_len {
                let value = self.values.head_slice(
                    self.physical_slot(token),
                    first_dim,
                    self.head_dim,
                    &mut scratch,
                );
                for (qh, &sum) in sums.iter().enumerate() {
                    if sum <= 0.0 {
                        continue;
                    }
                    let weight = scores[qh * self.seq_len + token] / sum;
                    let row = &mut output
                        [(first_qh + qh) * query_head_dim..(first_qh + qh + 1) * query_head_dim];
                    for d in 0..query_head_dim {
                        row[d] += weight * value[d];
                    }
                }
            }
        }

        output
    }
}

/// Main inference engine
pub struct InferenceEngine {
    pub model: Option<ModelWeights>,
    pub tokenizer: Option<crate::tokenizer::RuntimeTokenizer>,
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
            tokenizer: None,
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

        if self.config.weight_format != QuantFormat::FP32 {
            return Err(io::Error::new(io::ErrorKind::Unsupported,
                "Only F32 runtime weight storage is implemented; GGUF source quantization is detected separately"));
        }
        if let Some(reason) = self.backend.failure_reason() {
            return Err(io::Error::new(io::ErrorKind::Unsupported, reason));
        }
        // Parse GGUF header and tensor metadata with mmap-backed parser.
        let metadata = GgufReader::parse_file(model_path)?;

        let invalid = |message: String| io::Error::new(io::ErrorKind::InvalidData, message);
        let arch_str = metadata.architecture().unwrap_or("unknown").to_string();
        let arch = ModelArch::from_str(&arch_str);
        // Dense Mistral-7B graphs are identical to LLaMA (no sliding window,
        // no MoE, standard RoPE), so they share the same execution path. The
        // metadata keys of such files carry the `mistral.` prefix instead of
        // the hard-coded `llama.` one.
        if !matches!(arch, ModelArch::Llama | ModelArch::Mistral) {
            return Err(invalid(format!(
                "Unsupported inference architecture: {arch_str}; only llama and mistral are implemented"
            )));
        }
        let arch_prefix = match arch {
            ModelArch::Mistral => "mistral",
            _ => "llama",
        };
        if !matches!(
            self.config.backend,
            BackendType::CpuSimd | BackendType::Cuda
        ) || !self.backend.is_available()
        {
            return Err(invalid(format!(
                "{} inference kernels are not implemented; use cpu-simd",
                self.config.backend.name()
            )));
        }
        let required = |key: String| {
            metadata
                .get_usize(&key)
                .ok_or_else(|| invalid(format!("Missing model dimension: {key}")))
        };
        let num_layers = required(format!("{arch_prefix}.block_count"))?;
        let hidden_size = required(format!("{arch_prefix}.embedding_length"))?;
        let ff_dim = required(format!("{arch_prefix}.feed_forward_length"))?;
        let num_heads = required(format!("{arch_prefix}.attention.head_count"))?;
        let num_kv_heads = metadata
            .get_usize(&format!("{arch_prefix}.attention.head_count_kv"))
            .unwrap_or(num_heads);
        let model_context = required(format!("{arch_prefix}.context_length"))?;
        if hidden_size == 0
            || ff_dim == 0
            || num_heads == 0
            || num_kv_heads == 0
            || model_context == 0
            || self.config.max_seq_len == 0
            || hidden_size % num_heads != 0
            || num_heads % num_kv_heads != 0
        {
            return Err(invalid("Invalid model dimensions or context length".into()));
        }
        if metadata
            .get_usize(&format!("{arch_prefix}.expert_count"))
            .is_some_and(|n| n > 0)
            || metadata
                .get_usize(&format!("{arch_prefix}.expert_used_count"))
                .is_some_and(|n| n > 0)
            || metadata
                .get_usize(&format!("{arch_prefix}.attention.sliding_window"))
                .is_some_and(|n| n > 0)
            || metadata
                .tensors
                .iter()
                .any(|tensor| tensor.name.ends_with(".bias"))
        {
            return Err(invalid(
                "MoE, sliding-window and bias-bearing llama/mistral graphs are not implemented"
                    .into(),
            ));
        }
        let head_dim = hidden_size / num_heads;
        for key in [
            format!("{arch_prefix}.attention.key_length"),
            format!("{arch_prefix}.attention.value_length"),
            format!("{arch_prefix}.rope.dimension_count"),
        ] {
            if metadata.metadata.contains_key(&key) && metadata.get_usize(&key) != Some(head_dim) {
                return Err(invalid(format!(
                    "Unsupported or invalid head dimension: {key}"
                )));
            }
        }
        if head_dim % 2 != 0
            || metadata
                .get_usize(&format!("{arch_prefix}.rope.dimension_count"))
                .is_some_and(|dim| dim != head_dim)
        {
            return Err(invalid(
                "Only full, even-sized RoPE head dimensions are supported".into(),
            ));
        }
        if metadata
            .tensors
            .iter()
            .any(|tensor| tensor.name == "rope_freqs.weight")
            || metadata.metadata.iter().any(|(key, value)| {
                key == &format!("{arch_prefix}.rope.scale_linear")
                    || key == &format!("{arch_prefix}.rope.scale")
                    || (key.starts_with(&format!("{arch_prefix}.rope.scaling."))
                        && !(key == &format!("{arch_prefix}.rope.scaling.type")
                            && value.as_str() == Some("none")))
            })
        {
            return Err(invalid(
                "Scaled RoPE and rope_freqs.weight are not implemented".into(),
            ));
        }
        let mut config = self.config.clone();
        config.max_seq_len = config.max_seq_len.min(model_context);
        let parameter = |key: &str, default: f32| -> io::Result<f32> {
            match metadata.metadata.get(key) {
                None => Ok(default),
                Some(value) => value
                    .as_f32()
                    .ok_or_else(|| invalid(format!("Invalid floating model parameter: {key}"))),
            }
        };
        config.rope_freq_base = parameter(&format!("{arch_prefix}.rope.freq_base"), 10000.0)?;
        config.rms_norm_eps = parameter(
            &format!("{arch_prefix}.attention.layer_norm_rms_epsilon"),
            config.rms_norm_eps,
        )?;
        if !config.rope_freq_base.is_finite()
            || config.rope_freq_base <= 0.0
            || !config.rms_norm_eps.is_finite()
            || config.rms_norm_eps <= 0.0
        {
            return Err(invalid("Invalid RoPE frequency or RMSNorm epsilon".into()));
        }
        // Validate dimensions and all required weights before allocating any KV caches.
        let mut tensors = HashMap::new();
        for tensor_info in &metadata.tensors {
            let elements = tensor_info
                .shape
                .iter()
                .try_fold(1usize, |total, &dim| {
                    usize::try_from(dim)
                        .ok()
                        .and_then(|dim| total.checked_mul(dim))
                })
                .ok_or_else(|| {
                    invalid(format!("Tensor dimension overflow: {}", tensor_info.name))
                })?;
            if elements == 0
                || config
                    .max_loaded_tensor_elements
                    .is_some_and(|limit| elements > limit)
            {
                continue;
            }
            if !matches!(
                tensor_info.data_type,
                GgufDataType::F32
                    | GgufDataType::F16
                    | GgufDataType::BF16
                    | GgufDataType::Q8_0
                    | GgufDataType::Q4_0
                    | GgufDataType::Q4_K
                    | GgufDataType::Q2_K
            ) {
                continue;
            }
            let data = GgufReader::read_tensor_f32(model_path, &metadata, tensor_info)?;
            if data.len() != elements || data.iter().any(|value| !value.is_finite()) {
                return Err(invalid(format!(
                    "Invalid tensor data: {}",
                    tensor_info.name
                )));
            }
            if tensors.insert(tensor_info.name.clone(), data).is_some() {
                return Err(invalid(format!("Duplicate tensor: {}", tensor_info.name)));
            }
        }
        // Resolve aliases against the FILE, not only successfully loaded tensors.
        // A present unsupported output head must never silently become tied embeddings.
        let find = |names: &[String]| -> io::Result<&Vec<f32>> {
            let name = names
                .iter()
                .find(|name| metadata.get_tensor(name).is_some())
                .ok_or_else(|| {
                    invalid(format!("Missing required tensor: {}", names.join(" or ")))
                })?;
            tensors.get(name).ok_or_else(|| {
                invalid(format!(
                    "Unsupported, capped or empty required tensor: {name}"
                ))
            })
        };
        let embedding_names = Self::embedding_names();
        let embedding = find(&embedding_names)
            .map_err(|err| invalid(format!("Invalid token embedding: {err}")))?;
        if embedding.len() % hidden_size != 0 || embedding.is_empty() {
            return Err(invalid("Invalid token embedding shape".into()));
        }
        let vocab_size = embedding.len() / hidden_size;
        if vocab_size > u32::MAX as usize
            || metadata.vocab_size().is_some_and(|size| size != vocab_size)
        {
            return Err(invalid(
                "Vocabulary and embedding dimensions disagree".into(),
            ));
        }
        let check = |names: Vec<String>, rows: usize, cols: usize| -> io::Result<()> {
            let expected = rows
                .checked_mul(cols)
                .ok_or_else(|| invalid("Tensor shape overflow".into()))?;
            let tensor = find(&names)?;
            let info = names
                .iter()
                .find_map(|name| metadata.get_tensor(name))
                .ok_or_else(|| invalid("Required tensor metadata is missing".into()))?;
            let expected_shape = if rows == 1 && info.shape.len() == 1 {
                vec![cols as u64]
            } else {
                vec![cols as u64, rows as u64]
            };
            if tensor.len() != expected || info.shape != expected_shape {
                return Err(invalid(format!(
                    "Invalid shape for {}: expected {expected} elements, got {}",
                    names[0],
                    tensor.len()
                )));
            }
            Ok(())
        };
        check(embedding_names, vocab_size, hidden_size)?;
        check(Self::output_norm_names(), 1, hidden_size)?;
        check(Self::output_head_names(), vocab_size, hidden_size)?;
        for layer in 0..num_layers {
            for role in ["attn_norm", "ffn_norm"] {
                check(Self::layer_tensor_names(layer, role), 1, hidden_size)?;
            }
            for role in ["attn_q", "attn_output"] {
                check(
                    Self::layer_tensor_names(layer, role),
                    hidden_size,
                    hidden_size,
                )?;
            }
            for role in ["attn_k", "attn_v"] {
                check(
                    Self::layer_tensor_names(layer, role),
                    num_kv_heads * head_dim,
                    hidden_size,
                )?;
            }
            for role in ["ffn_gate", "ffn_up"] {
                check(Self::layer_tensor_names(layer, role), ff_dim, hidden_size)?;
            }
            check(
                Self::layer_tensor_names(layer, "ffn_down"),
                hidden_size,
                ff_dim,
            )?;
        }
        // Source tensor codec is a file property; execution materializes every
        // supported tensor as F32. Do not turn it into a fake runtime override.
        config.weight_format = QuantFormat::FP32;
        let quant_engine = QuantEngine::new(QuantFormat::FP32, config.kv_quant);
        let memory = UnifiedMemory::new();
        let kv_dim = num_kv_heads * head_dim;
        kv_dim
            .checked_mul(config.max_seq_len)
            .and_then(|n| n.checked_mul(8))
            .and_then(|n| n.checked_mul(num_layers))
            .ok_or_else(|| invalid("KV cache size overflow".into()))?;
        let kv_caches = (0..num_layers)
            .map(|_| KvCache::new(kv_dim, config.max_seq_len, config.kv_quant))
            .collect();

        let tokenizer = crate::tokenizer::RuntimeTokenizer::from_model(
            model_path,
            &metadata,
            vocab_size,
            config.allow_byte_tokenizer,
        )
        .map_err(io::Error::other)?;
        self.backend
            .prepare_weights(&tensors)
            .map_err(io::Error::other)?;
        // Store model
        self.model = Some(ModelWeights {
            metadata,
            arch,
            tensors,
            quant_engine,
            memory,
        });

        self.tokenizer = Some(tokenizer);
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

    fn tensor<'a>(model: &'a ModelWeights, names: &[String]) -> Result<&'a Vec<f32>, String> {
        if let Some(name) = names
            .iter()
            .find(|name| model.metadata.get_tensor(name).is_some())
        {
            return model
                .get_tensor(name)
                .ok_or_else(|| format!("Missing tensor: {name}"));
        }
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

        let penalty = config.repetition_penalty;
        if config.temperature <= 0.0 && penalty == 1.0 {
            return Self::greedy_token(logits);
        }
        let seen: HashSet<u32> = if penalty == 1.0 {
            HashSet::new()
        } else {
            output.iter().copied().collect()
        };
        let temperature = config.temperature.max(1e-5);
        let mut candidates: Vec<(usize, f32)> = logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .map(|(idx, mut logit)| {
                if penalty != 1.0 && seen.contains(&(idx as u32)) {
                    if logit >= 0.0 {
                        logit /= penalty;
                    } else {
                        logit *= penalty;
                    }
                }
                (idx, logit)
            })
            .collect();

        if candidates.is_empty() {
            return 0;
        }

        if config.temperature <= 0.0 {
            return candidates
                .iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(id, _)| *id as u32)
                .unwrap_or(0);
        }
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        if config.top_k > 0 && candidates.len() > config.top_k {
            candidates.truncate(config.top_k);
        }

        let max_logit = candidates[0].1;
        let mut probs: Vec<(usize, f32)> = candidates
            .into_iter()
            .map(|(idx, logit)| (idx, ((logit - max_logit) / temperature).exp()))
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

        let fallback = probs[0].0 as u32;
        let mut draw = rng.gen::<f32>();
        for (idx, prob) in probs {
            if draw <= prob {
                return idx as u32;
            }
            draw -= prob;
        }
        fallback
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
        let norm_weight = Self::tensor(model, &Self::output_norm_names())?;
        let hidden = kernel::rms_norm(hidden, norm_weight, self.config.rms_norm_eps);
        let output = Self::tensor(model, &Self::output_head_names())?;
        self.backend
            .matvec(output, self.vocab_size, self.hidden_size, &hidden)
    }

    /// Stateless forward pass through the same validated transformer as generation.
    pub fn decode(&self, token: u32, position: usize) -> Result<Vec<f32>, String> {
        let mut caches = self.new_kv_caches();
        self.decode_into(token, position, &mut caches)
    }

    /// Run a single autoregressive decode step through loaded transformer layers.
    pub fn decode_with_cache(&mut self, token: u32, position: usize) -> Result<Vec<f32>, String> {
        let mut caches = std::mem::take(&mut self.kv_caches);
        let result = self.decode_into(token, position, &mut caches);
        self.kv_caches = caches;
        result
    }

    fn new_kv_caches(&self) -> Vec<KvCache> {
        (0..self.num_layers)
            .map(|_| {
                KvCache::new(
                    self.num_kv_heads * self.head_dim,
                    self.config.max_seq_len,
                    self.config.kv_quant,
                )
            })
            .collect()
    }

    fn decode_into(
        &self,
        token: u32,
        position: usize,
        caches: &mut [KvCache],
    ) -> Result<Vec<f32>, String> {
        let model = self.model.as_ref().ok_or("Model not loaded")?;
        if self.hidden_size == 0 || token as usize >= self.vocab_size {
            return Err(format!("Token {token} is outside the model vocabulary"));
        }
        if position >= self.config.max_seq_len {
            return Err("Context length exceeded".into());
        }
        let embedding = Self::tensor(model, &Self::embedding_names())?;
        let start = token as usize * self.hidden_size;
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
            let mut q = self.backend.matvec(
                q_weight,
                self.num_heads * self.head_dim,
                self.hidden_size,
                &normed,
            )?;
            let mut k = self
                .backend
                .matvec(k_weight, kv_dim, self.hidden_size, &normed)?;
            let v = self
                .backend
                .matvec(v_weight, kv_dim, self.hidden_size, &normed)?;

            Self::apply_rope_heads(&mut q, position, self.head_dim, self.config.rope_freq_base);
            Self::apply_rope_heads(&mut k, position, self.head_dim, self.config.rope_freq_base);

            let cache = caches
                .get_mut(layer)
                .ok_or_else(|| format!("Missing KV cache for layer {}", layer))?;
            cache.add(&k, &v);
            let attn = cache.attention_decode(&q, self.head_dim, self.num_heads, self.num_kv_heads);
            let attn_out =
                self.backend
                    .matvec(o_weight, self.hidden_size, self.hidden_size, &attn)?;
            Self::add_inplace(&mut hidden, &attn_out);

            let ffn_norm = Self::tensor(model, &Self::layer_tensor_names(layer, "ffn_norm"))?;
            let gate_weight = Self::tensor(model, &Self::layer_tensor_names(layer, "ffn_gate"))?;
            let up_weight = Self::tensor(model, &Self::layer_tensor_names(layer, "ffn_up"))?;
            let down_weight = Self::tensor(model, &Self::layer_tensor_names(layer, "ffn_down"))?;

            let normed = kernel::rms_norm(&hidden, ffn_norm, self.config.rms_norm_eps);
            let gate = self
                .backend
                .matvec(gate_weight, self.ff_dim, self.hidden_size, &normed)?;
            let up = self
                .backend
                .matvec(up_weight, self.ff_dim, self.hidden_size, &normed)?;
            let mut gated = vec![0.0f32; self.ff_dim];
            for i in 0..self.ff_dim {
                gated[i] = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
            }
            let ffn_out =
                self.backend
                    .matvec(down_weight, self.hidden_size, self.ff_dim, &gated)?;
            Self::add_inplace(&mut hidden, &ffn_out);
        }

        self.transformer_logits(model, &hidden)
    }

    /// Run a batch of tokens through the model (prefill)
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("Prompt must contain at least one token".into());
        }
        if tokens.len() > self.config.max_seq_len {
            return Err("Prompt exceeds context length".into());
        }
        for cache in &mut self.kv_caches {
            cache.clear();
        }

        let mut logits = Vec::new();
        for (position, &token) in tokens.iter().enumerate() {
            logits = self.decode_with_cache(token, position)?;
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
        self.generate_with_config_streaming(prompt, generation, |_, _| Ok(()))
    }

    /// Generate text and call `on_token` whenever a generated token is accepted.
    ///
    /// The callback runs synchronously on the generation thread. Returning an
    /// error stops generation and propagates that error to the caller.
    pub fn generate_with_config_streaming<F>(
        &mut self,
        prompt: &[u32],
        generation: &GenerationConfig,
        mut on_token: F,
    ) -> Result<Vec<u32>, String>
    where
        F: FnMut(u32, &[u32]) -> Result<(), String>,
    {
        let mut session = self.start_session(prompt, generation)?;
        let mut emitted = 0;
        while !session.finished {
            self.step_session(&mut session)?;
            let visible_len = session.visible_output().len();
            for i in emitted..visible_len {
                let end = session.prompt_len + i + 1;
                on_token(session.output[end - 1], &session.output[..end])?;
            }
            emitted = visible_len;
        }
        Ok(session.output)
    }

    /// Prepare an independent sequence, sharing weights but owning its KV cache and RNG.
    pub fn start_session(
        &mut self,
        prompt: &[u32],
        generation: &GenerationConfig,
    ) -> Result<GenerationSession, String> {
        if self.model.is_none() {
            return Err("Model not loaded".into());
        }
        if prompt.is_empty() {
            return Err("Prompt must contain at least one token".into());
        }
        if prompt
            .len()
            .checked_add(generation.max_tokens)
            .is_none_or(|n| n > self.config.max_seq_len)
        {
            return Err("Prompt and max_tokens exceed context length".into());
        }
        if !generation.temperature.is_finite()
            || generation.temperature < 0.0
            || !generation.top_p.is_finite()
            || generation.top_p <= 0.0
            || generation.top_p > 1.0
            || !generation.repetition_penalty.is_finite()
            || generation.repetition_penalty <= 0.0
        {
            return Err("Invalid sampling parameters".into());
        }
        if let Some(token) = prompt.iter().find(|&&token| {
            token as usize >= self.vocab_size
                || self.tokenizer.as_ref().is_some_and(|t| !t.valid_id(token))
        }) {
            return Err(format!("Token {token} is outside the model vocabulary"));
        }
        // A zero-token request returns its prompt without allocating a full KV cache.
        let caches = if generation.max_tokens == 0 {
            Vec::new()
        } else {
            self.new_kv_caches()
        };
        let logits = Vec::new();
        let eos = self
            .model
            .as_ref()
            .and_then(|m| m.metadata.get_usize("tokenizer.ggml.eos_token_id"))
            .and_then(|id| u32::try_from(id).ok());
        Ok(GenerationSession {
            output: prompt.to_vec(),
            prompt_len: prompt.len(),
            finished: generation.max_tokens == 0,
            finish_reason: "length",
            caches,
            logits,
            generation: generation.clone(),
            eos,
            prefill_position: 0,
            rng: generation
                .seed
                .map(StdRng::seed_from_u64)
                .unwrap_or_else(StdRng::from_entropy),
        })
    }

    /// Advance one prefill or generation token. `None` with `finished == false`
    /// means that prefill advanced; callers should continue scheduling the session.
    pub fn step_session(&mut self, session: &mut GenerationSession) -> Result<Option<u32>, String> {
        if session.finished {
            return Ok(None);
        }
        if session.prefill_position < session.prompt_len {
            let position = session.prefill_position;
            session.logits =
                self.decode_into(session.output[position], position, &mut session.caches)?;
            session.prefill_position += 1;
            return Ok(None);
        }
        if let Some(tokenizer) = &self.tokenizer {
            tokenizer.mask_logits(&mut session.logits);
        }
        if session.logits.len() != self.vocab_size || !session.logits.iter().any(|v| v.is_finite())
        {
            return Err("Model returned invalid logits".into());
        }
        let token = Self::sample_token(
            &session.logits,
            &session.output,
            &session.generation,
            &mut session.rng,
        );
        if Some(token) == session.eos || self.tokenizer.as_ref().is_some_and(|t| t.is_eog(token)) {
            session.finished = true;
            session.finish_reason = "stop";
            return Ok(None);
        }
        session.output.push(token);
        if let Some(len) = Self::matched_stop_len(
            &session.output,
            session.prompt_len,
            &session.generation.stop_sequences,
        ) {
            session.output.truncate(session.output.len() - len);
            session.finished = true;
            session.finish_reason = "stop";
            return Ok(None);
        }
        if session.output.len() - session.prompt_len >= session.generation.max_tokens {
            session.finished = true;
            session.finish_reason = "length";
        } else {
            session.logits =
                self.decode_into(token, session.output.len() - 1, &mut session.caches)?;
        }
        Ok(Some(token))
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
        let mut session = self.start_session(prompt, generation)?;
        let mut stats = SpeculativeStats::default();
        let mut index = 0;
        while !session.finished {
            if let Some(token) = self.step_session(&mut session)? {
                if let Some(&draft) = draft_tokens.get(index) {
                    if token == draft {
                        stats.accepted += 1;
                    } else {
                        stats.rejected += 1;
                    }
                }
                index += 1;
            }
        }
        Ok((session.output, stats))
    }

    /// Run speculative decoding with a separate draft engine.
    pub fn generate_with_draft_engine(
        &mut self,
        draft: &mut InferenceEngine,
        prompt: &[u32],
        generation: &GenerationConfig,
        draft_window: usize,
    ) -> Result<(Vec<u32>, SpeculativeStats), String> {
        let mut target = self.start_session(prompt, generation)?;
        while !target.finished && target.prefill_position < target.prompt_len {
            self.step_session(&mut target)?;
        }
        let mut stats = SpeculativeStats::default();
        while !target.finished {
            let remaining = generation
                .max_tokens
                .saturating_sub(target.output.len() - target.prompt_len);
            let proposed = draft.generate_with_config(
                &target.output,
                &GenerationConfig {
                    max_tokens: remaining.min(draft_window.max(1)),
                    ..generation.clone()
                },
            )?;
            let drafts = &proposed[target.output.len()..];
            for index in 0..drafts.len().max(1) {
                if target.finished {
                    break;
                }
                if let Some(token) = self.step_session(&mut target)? {
                    if let Some(&candidate) = drafts.get(index) {
                        if token == candidate {
                            stats.accepted += 1;
                        } else {
                            stats.rejected += 1;
                        }
                    }
                }
            }
        }
        Ok((target.output, stats))
    }

    /// Actual materialized F32 weight payload, irrespective of source quantization.
    pub fn weight_memory_usage(&self) -> usize {
        self.model
            .as_ref()
            .map(|model| {
                model
                    .tensors
                    .values()
                    .map(|tensor| tensor.len() * std::mem::size_of::<f32>())
                    .sum()
            })
            .unwrap_or(0)
    }

    /// Payload owned by the engine's standalone decode caches, excluding sessions.
    pub fn kv_cache_memory_usage(&self) -> usize {
        self.kv_caches.iter().map(KvCache::memory_usage).sum()
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

        let kv_mem = self.kv_cache_memory_usage();

        let source_formats = self
            .model
            .as_ref()
            .map(|model| {
                model
                    .metadata
                    .tensors
                    .iter()
                    .map(|tensor| tensor.data_type.to_string())
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_else(|| "none".into());
        format!(
            "{}\n\nBackend: {}\nKV Quant: {}\nSource tensor encodings: {}\nRuntime weight storage: F32\nWeight payload memory: {}\nMax seq len: {}\nKV cache memory: {}",
            model,
            self.backend.name(),
            self.config.kv_quant.name(),
            source_formats,
            crate::memory::format_size(self.weight_memory_usage()),
            self.config.max_seq_len,
            crate::memory::format_size(kv_mem),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn zeroed_kv_allocations_initialize_all_elements_and_allow_vec_growth() {
        for length in [0, 1, 31, 4097] {
            let mut bytes = zeroed_kv_bytes(length).unwrap();
            assert_eq!(bytes.len(), length);
            assert!(bytes.iter().all(|&value| value == 0));
            bytes.fill(0xa5);
            bytes.push(0x3c);
            assert!(bytes[..length].iter().all(|&value| value == 0xa5));
            assert_eq!(bytes[length], 0x3c);

            let mut floats = zeroed_kv_floats(length).unwrap();
            assert_eq!(floats.len(), length);
            assert!(floats.iter().all(|value| value.to_bits() == 0));
            floats.fill(1.25);
            floats.push(-2.5);
            assert!(floats[..length].iter().all(|&value| value == 1.25));
            assert_eq!(floats[length], -2.5);
        }
    }

    #[test]
    fn zeroed_kv_allocations_reject_invalid_layouts_before_allocating() {
        assert!(zeroed_kv_bytes(isize::MAX as usize + 1).is_err());
        assert!(zeroed_kv_floats(isize::MAX as usize / size_of::<f32>() + 1).is_err());
        assert!(zeroed_kv_floats(usize::MAX).is_err());
    }

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
        // The unquantized cache is the default: TurboQuant 3b/2b visibly
        // degrades output quality and every recorded reference ran unquantized.
        assert_eq!(config.kv_quant, KvQuantType::None);
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
