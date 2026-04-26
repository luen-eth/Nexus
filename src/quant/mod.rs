//! Quantization engine supporting weight quantization (Q2-Q8, MXFP4)
//! and KV cache quantization (TurboQuant: 3-bit keys, 2/4-bit values).
//!
//! Combines llama.cpp's weight quantization with TurboQuant's KV cache compression.

#![allow(non_camel_case_types)]

use std::fmt;

// ============================================================================
// Weight Quantization Types (llama.cpp style)
// ============================================================================

/// Weight quantization formats matching llama.cpp
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantFormat {
    Q2_K,
    Q3_K,
    Q4_0,
    Q4_1,
    Q4_K,
    Q5_0,
    Q5_1,
    Q5_K,
    Q6_K,
    Q8_0,
    Q8_K,
    MXFP4,
    FP16,
    FP32,
}

impl QuantFormat {
    pub fn name(&self) -> &'static str {
        match self {
            QuantFormat::Q2_K => "Q2_K",
            QuantFormat::Q3_K => "Q3_K",
            QuantFormat::Q4_0 => "Q4_0",
            QuantFormat::Q4_1 => "Q4_1",
            QuantFormat::Q4_K => "Q4_K",
            QuantFormat::Q5_0 => "Q5_0",
            QuantFormat::Q5_1 => "Q5_1",
            QuantFormat::Q5_K => "Q5_K",
            QuantFormat::Q6_K => "Q6_K",
            QuantFormat::Q8_0 => "Q8_0",
            QuantFormat::Q8_K => "Q8_K",
            QuantFormat::MXFP4 => "MXFP4",
            QuantFormat::FP16 => "FP16",
            QuantFormat::FP32 => "FP32",
        }
    }

    pub fn bits(&self) -> u32 {
        match self {
            QuantFormat::Q2_K => 2,
            QuantFormat::Q3_K => 3,
            QuantFormat::Q4_0 | QuantFormat::Q4_1 | QuantFormat::Q4_K => 4,
            QuantFormat::Q5_0 | QuantFormat::Q5_1 | QuantFormat::Q5_K => 5,
            QuantFormat::Q6_K => 6,
            QuantFormat::Q8_0 | QuantFormat::Q8_K => 8,
            QuantFormat::MXFP4 => 4,
            QuantFormat::FP16 => 16,
            QuantFormat::FP32 => 32,
        }
    }

    pub fn compression_ratio(&self) -> f32 {
        match self {
            QuantFormat::Q2_K => 0.125,
            QuantFormat::Q3_K => 0.1875,
            QuantFormat::Q4_0 | QuantFormat::Q4_1 | QuantFormat::Q4_K => 0.25,
            QuantFormat::Q5_0 | QuantFormat::Q5_1 | QuantFormat::Q5_K => 0.3125,
            QuantFormat::Q6_K => 0.375,
            QuantFormat::Q8_0 | QuantFormat::Q8_K => 0.5,
            QuantFormat::MXFP4 => 0.25,
            QuantFormat::FP16 => 0.5,
            QuantFormat::FP32 => 1.0,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<QuantFormat> {
        match s.to_uppercase().as_str() {
            "Q2_K" => Some(QuantFormat::Q2_K),
            "Q3_K" => Some(QuantFormat::Q3_K),
            "Q4_0" => Some(QuantFormat::Q4_0),
            "Q4_1" => Some(QuantFormat::Q4_1),
            "Q4_K" => Some(QuantFormat::Q4_K),
            "Q5_0" => Some(QuantFormat::Q5_0),
            "Q5_1" => Some(QuantFormat::Q5_1),
            "Q5_K" => Some(QuantFormat::Q5_K),
            "Q6_K" => Some(QuantFormat::Q6_K),
            "Q8_0" => Some(QuantFormat::Q8_0),
            "Q8_K" => Some(QuantFormat::Q8_K),
            "MXFP4" | "MXFP4_4X4" => Some(QuantFormat::MXFP4),
            "FP16" => Some(QuantFormat::FP16),
            "FP32" => Some(QuantFormat::FP32),
            _ => None,
        }
    }
}

impl fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// QuantType (alias for compatibility)
// ============================================================================

pub use QuantFormat as QuantType;

// ============================================================================
// KV Cache Quantization Types (TurboQuant style)
// ============================================================================

/// KV cache quantization types inspired by TurboQuant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KvQuantType {
    /// No KV quantization (bf16/fp16)
    None,
    /// TurboQuant: 3-bit keys, 2-bit values
    TurboQuant3b2b,
    /// TurboQuant: 3-bit keys, 4-bit values
    TurboQuant3b4b,
    /// TurboQuant: 4-bit keys, 4-bit values
    TurboQuant4b4b,
    /// Simple block quantization for KV
    BlockQ4,
    /// Simple block quantization for KV
    BlockQ6,
}

impl KvQuantType {
    pub fn name(&self) -> &'static str {
        match self {
            KvQuantType::None => "none",
            KvQuantType::TurboQuant3b2b => "tq_3b2b",
            KvQuantType::TurboQuant3b4b => "tq_3b4b",
            KvQuantType::TurboQuant4b4b => "tq_4b4b",
            KvQuantType::BlockQ4 => "block_q4",
            KvQuantType::BlockQ6 => "block_q6",
        }
    }

    pub fn key_bits(&self) -> u32 {
        match self {
            KvQuantType::None => 16,
            KvQuantType::TurboQuant3b2b | KvQuantType::TurboQuant3b4b => 3,
            KvQuantType::TurboQuant4b4b => 4,
            KvQuantType::BlockQ4 => 4,
            KvQuantType::BlockQ6 => 6,
        }
    }

    pub fn val_bits(&self) -> u32 {
        match self {
            KvQuantType::None => 16,
            KvQuantType::TurboQuant3b2b => 2,
            KvQuantType::TurboQuant3b4b | KvQuantType::TurboQuant4b4b => 4,
            KvQuantType::BlockQ4 => 4,
            KvQuantType::BlockQ6 => 6,
        }
    }

    pub fn bits(&self) -> u32 {
        match self {
            KvQuantType::None => 16,
            _ => self.key_bits() + self.val_bits(),
        }
    }

    /// Compression ratio relative to fp16 KV cache
    pub fn compression_ratio(&self) -> f32 {
        match self {
            KvQuantType::None => 1.0,
            KvQuantType::TurboQuant3b2b => 4.41,
            KvQuantType::TurboQuant3b4b => 2.75,
            KvQuantType::TurboQuant4b4b => 2.0,
            KvQuantType::BlockQ4 => 4.0,
            KvQuantType::BlockQ6 => 2.67,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<KvQuantType> {
        match s {
            "none" | "bf16" | "fp16" => Some(KvQuantType::None),
            "tq_3b2b" | "turboquant_3b2b" => Some(KvQuantType::TurboQuant3b2b),
            "tq_3b4b" | "turboquant_3b4b" => Some(KvQuantType::TurboQuant3b4b),
            "tq_4b4b" | "turboquant_4b4b" => Some(KvQuantType::TurboQuant4b4b),
            "block_q4" => Some(KvQuantType::BlockQ4),
            "block_q6" => Some(KvQuantType::BlockQ6),
            _ => None,
        }
    }
}

impl fmt::Display for KvQuantType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ============================================================================
// Quantization Parameters (per-group scales and zeros)
// ============================================================================

/// Group quantization parameters for a single group of values
#[derive(Debug, Clone)]
pub struct QuantParams {
    /// Scale factor for this group
    pub scale: f32,
    /// Delta (used in K-quants for secondary scaling)
    pub delta: f32,
}

impl QuantParams {
    pub fn new(scale: f32, delta: f32) -> Self {
        QuantParams { scale, delta }
    }

    pub fn identity() -> Self {
        QuantParams {
            scale: 1.0,
            delta: 0.0,
        }
    }
}

/// Group size for quantization (affects compression quality)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupSize {
    /// 32 elements per group
    N32,
    /// 64 elements per group
    N64,
    /// 128 elements per group
    N128,
    /// 256 elements per group
    N256,
}

impl GroupSize {
    pub fn value(&self) -> usize {
        match self {
            GroupSize::N32 => 32,
            GroupSize::N64 => 64,
            GroupSize::N128 => 128,
            GroupSize::N256 => 256,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<GroupSize> {
        match s {
            "32" => Some(GroupSize::N32),
            "64" => Some(GroupSize::N64),
            "128" => Some(GroupSize::N128),
            "256" => Some(GroupSize::N256),
            _ => None,
        }
    }
}

// ============================================================================
// TurboQuant-specific structures
// ============================================================================

/// Random orthogonal rotation matrix metadata (for TurboQuant)
#[derive(Debug, Clone)]
pub struct RotationMatrix {
    /// Dimensions of the rotation matrix (d x d)
    pub dim: usize,
    /// Matrix stored as flat array (row-major)
    pub data: Vec<f32>,
}

impl RotationMatrix {
    pub fn new(dim: usize) -> Self {
        RotationMatrix {
            dim,
            data: vec![0.0; dim * dim],
        }
    }

    /// Generate a random orthogonal rotation matrix using QR decomposition
    pub fn generate_random(dim: usize) -> Self {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(42); // deterministic seed

        // Generate random matrix and QR-decompose for orthogonality
        let mut data = vec![0.0f32; dim * dim];
        for i in 0..dim {
            for j in 0..dim {
                data[i * dim + j] = rng.gen_range(-1.0..1.0);
            }
        }

        // Simple Gram-Schmidt orthogonalization
        for i in 0..dim {
            // Normalize row i
            let mut norm = 0.0f32;
            for j in 0..dim {
                norm += data[i * dim + j] * data[i * dim + j];
            }
            norm = norm.sqrt();
            if norm > 1e-8 {
                for j in 0..dim {
                    data[i * dim + j] /= norm;
                }
            }

            // Orthogonalize against previous rows
            for k in 0..i {
                let mut dot = 0.0f32;
                for j in 0..dim {
                    dot += data[i * dim + j] * data[k * dim + j];
                }
                for j in 0..dim {
                    data[i * dim + j] -= dot * data[k * dim + j];
                }
            }

            // Re-normalize after orthogonalization
            let mut norm = 0.0f32;
            for j in 0..dim {
                norm += data[i * dim + j] * data[i * dim + j];
            }
            norm = norm.sqrt();
            if norm > 1e-8 {
                for j in 0..dim {
                    data[i * dim + j] /= norm;
                }
            }
        }

        RotationMatrix { dim, data }
    }

    /// Apply rotation to a vector: y = R @ x
    pub fn apply(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.dim, "Vector dimension mismatch");
        let mut y = vec![0.0f32; self.dim];
        for (i, yi) in y.iter_mut().enumerate().take(self.dim) {
            for (j, &xj) in x.iter().enumerate().take(self.dim) {
                *yi += self.data[i * self.dim + j] * xj;
            }
        }
        y
    }

    /// Apply inverse rotation: x = R^T @ y (since R is orthogonal, R^-1 = R^T)
    pub fn apply_inverse(&self, y: &[f32]) -> Vec<f32> {
        assert_eq!(y.len(), self.dim, "Vector dimension mismatch");
        let mut x = vec![0.0f32; self.dim];
        for (i, xi) in x.iter_mut().enumerate().take(self.dim) {
            for (j, &yj) in y.iter().enumerate().take(self.dim) {
                *xi += self.data[j * self.dim + i] * yj;
            }
        }
        x
    }
}

/// Lloyd-Max quantizer codebook for Beta distribution
#[derive(Debug, Clone)]
pub struct Codebook {
    /// Number of quantization levels (2^bits)
    pub bits: u32,
    /// Number of dimensions this codebook covers
    pub dim: usize,
    /// Reconstruction levels (centroids)
    pub levels: Vec<f32>,
}

impl Codebook {
    pub fn new(bits: u32, dim: usize) -> Self {
        let n_levels = 1u32 << bits;
        let n_levels = n_levels as usize;

        // Generate Lloyd-Max quantization levels for Beta(2, 5) distribution
        // This is the distribution that rotated KV cache values follow
        let mut levels = Vec::with_capacity(n_levels);

        for i in 0..n_levels {
            // Approximate Lloyd-Max centroids using uniform sampling of CDF
            let t = (i as f32 + 0.5) / n_levels as f32;
            // Simple approximation: map to [-1, 1] range
            let level = 2.0 * t - 1.0;
            levels.push(level);
        }

        Codebook { bits, dim, levels }
    }

    /// Quantize a single value using this codebook
    pub fn quantize(&self, value: f32) -> f32 {
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;

        for (i, &level) in self.levels.iter().enumerate() {
            let dist = (level - value).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        self.levels[best_idx]
    }

    /// Quantize a batch of values
    pub fn quantize_batch(&self, values: &[f32]) -> Vec<f32> {
        values.iter().map(|&v| self.quantize(v)).collect()
    }

    /// Encode quantized values to bits (u8)
    pub fn encode(&self, values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::new();

        match self.bits {
            2 => {
                // Pack 4 values per byte (2 bits each)
                for chunk in values.chunks(4) {
                    let mut byte: u8 = 0;
                    for (i, &val) in chunk.iter().enumerate() {
                        let best_idx = self.find_closest(val);
                        byte |= (best_idx as u8 & 0x3) << (i * 2);
                    }
                    bytes.push(byte);
                }
            }
            3 => {
                // Pack values into a dense 3-bit stream.
                bytes.resize((values.len() * 3).div_ceil(8), 0);
                for (i, &val) in values.iter().enumerate() {
                    let best_idx = self.find_closest(val);
                    write_packed_bits(&mut bytes, i, 3, (best_idx & 0x7) as u32);
                }
            }
            4 => {
                // Pack 2 values per byte (4 bits each)
                for chunk in values.chunks(2) {
                    let mut byte: u8 = 0;
                    for (i, &val) in chunk.iter().enumerate() {
                        let best_idx = self.find_closest(val);
                        byte |= (best_idx as u8 & 0xF) << (i * 4);
                    }
                    bytes.push(byte);
                }
            }
            _ => {
                for &val in values {
                    let best_idx = self.find_closest(val);
                    bytes.push(best_idx as u8);
                }
            }
        }
        bytes
    }

    fn find_closest(&self, value: f32) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f32::MAX;
        for (i, &level) in self.levels.iter().enumerate() {
            let dist = (level - value).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }
        best_idx
    }

    /// Decode bits back to values
    pub fn decode(&self, bits: &[u8]) -> Vec<f32> {
        let output_len = match self.bits {
            2 => bits.len() * 4,
            3 => (bits.len() * 8) / 3,
            4 => bits.len() * 2,
            _ => bits.len(),
        };
        self.decode_exact(bits, output_len)
    }

    /// Decode exactly `output_len` values from packed codebook bits.
    pub fn decode_exact(&self, bits: &[u8], output_len: usize) -> Vec<f32> {
        let mut values = Vec::new();

        match self.bits {
            2 => {
                // Unpack 4 values per byte
                for &byte in bits {
                    for i in 0..4 {
                        let idx = ((byte >> (i * 2)) & 0x3) as usize;
                        if idx < self.levels.len() {
                            values.push(self.levels[idx]);
                        } else {
                            values.push(0.0);
                        }
                        if values.len() == output_len {
                            return values;
                        }
                    }
                }
            }
            3 => {
                for i in 0..output_len {
                    let idx = read_packed_bits(bits, i, 3) as usize;
                    if idx < self.levels.len() {
                        values.push(self.levels[idx]);
                    } else {
                        values.push(0.0);
                    }
                }
            }
            4 => {
                // Unpack 2 values per byte
                for &byte in bits {
                    for i in 0..2 {
                        let idx = ((byte >> (i * 4)) & 0xF) as usize;
                        if idx < self.levels.len() {
                            values.push(self.levels[idx]);
                        } else {
                            values.push(0.0);
                        }
                        if values.len() == output_len {
                            return values;
                        }
                    }
                }
            }
            _ => {
                for &byte in bits {
                    let idx = byte as usize;
                    if idx < self.levels.len() {
                        values.push(self.levels[idx]);
                    } else {
                        values.push(0.0);
                    }
                    if values.len() == output_len {
                        return values;
                    }
                }
            }
        }
        values
    }
}

// ============================================================================
// Quantization Engine
// ============================================================================

/// Quantization engine that handles weight and KV cache quantization
pub struct QuantEngine {
    /// Weight quantization format
    pub weight_format: QuantFormat,
    /// KV cache quantization type
    pub kv_quant: KvQuantType,
    /// Group size for quantization
    pub group_size: GroupSize,
    /// TurboQuant rotation matrices (one per layer)
    pub rotations: Vec<RotationMatrix>,
    /// Lloyd-Max codebooks for each bit depth
    pub codebooks: std::collections::HashMap<(u32, usize), Codebook>,
}

impl QuantEngine {
    pub fn new(weight_format: QuantFormat, kv_quant: KvQuantType) -> Self {
        let mut engine = QuantEngine {
            weight_format,
            kv_quant,
            group_size: GroupSize::N256,
            rotations: Vec::new(),
            codebooks: std::collections::HashMap::new(),
        };

        // Pre-generate codebooks for common configurations
        for bits in [2u32, 3, 4] {
            for dim in [128usize, 256] {
                engine
                    .codebooks
                    .insert((bits, dim), Codebook::new(bits, dim));
            }
        }

        engine
    }

    /// Add a rotation matrix for a specific layer
    pub fn add_rotation(&mut self, layer: usize, rotation: RotationMatrix) {
        if layer >= self.rotations.len() {
            self.rotations
                .resize(layer + 1, RotationMatrix::new(rotation.dim));
        }
        self.rotations[layer] = rotation;
    }

    /// Generate default rotation matrices for a given dimension and layer count
    pub fn generate_rotations(&mut self, dim: usize, layer_count: usize) {
        self.rotations.clear();
        for _ in 0..layer_count {
            self.rotations.push(RotationMatrix::generate_random(dim));
        }
    }

    /// Get or create a codebook for given bit depth and dimension
    pub fn get_codebook(&mut self, bits: u32, dim: usize) -> &Codebook {
        let key = (bits, dim);
        self.codebooks
            .entry(key)
            .or_insert_with(|| Codebook::new(bits, dim));
        self.codebooks.get(&key).unwrap()
    }

    /// Quantize a batch of values using the configured weight format
    pub fn quantize_weights(&self, data: &[f32]) -> Vec<u8> {
        match self.weight_format {
            QuantFormat::FP32 => data.iter().flat_map(|&v| v.to_le_bytes()).collect(),
            QuantFormat::FP16 => data
                .iter()
                .flat_map(|&v| half::f16::from_f32(v).to_bits().to_le_bytes())
                .collect(),
            QuantFormat::Q8_0 | QuantFormat::Q8_K => Self::quantize_q8_0(data),
            QuantFormat::Q4_0 | QuantFormat::Q4_1 | QuantFormat::MXFP4 => Self::quantize_q4_0(data),
            QuantFormat::Q4_K => Self::quantize_q4_k(data),
            QuantFormat::Q2_K => Self::quantize_q2_k(data),
            QuantFormat::Q3_K => Self::quantize_affine_nbit(data, 3),
            QuantFormat::Q5_0 | QuantFormat::Q5_1 | QuantFormat::Q5_K => {
                Self::quantize_affine_nbit(data, 5)
            }
            QuantFormat::Q6_K => Self::quantize_affine_nbit(data, 6),
        }
    }

    /// Dequantize a batch of values back to f32
    pub fn dequantize(&self, data: &[u8], output_len: usize) -> Vec<f32> {
        match self.weight_format {
            QuantFormat::FP32 => data
                .chunks_exact(4)
                .take(output_len)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            QuantFormat::FP16 => data
                .chunks_exact(2)
                .take(output_len)
                .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            QuantFormat::Q8_0 | QuantFormat::Q8_K => Self::dequantize_q8_0(data, output_len),
            QuantFormat::Q4_0 | QuantFormat::Q4_1 | QuantFormat::MXFP4 => {
                Self::dequantize_q4_0(data, output_len)
            }
            QuantFormat::Q4_K => Self::dequantize_q4_k(data, output_len),
            QuantFormat::Q2_K => Self::dequantize_q2_k(data, output_len),
            QuantFormat::Q3_K => Self::dequantize_affine_nbit(data, output_len, 3),
            QuantFormat::Q5_0 | QuantFormat::Q5_1 | QuantFormat::Q5_K => {
                Self::dequantize_affine_nbit(data, output_len, 5)
            }
            QuantFormat::Q6_K => Self::dequantize_affine_nbit(data, output_len, 6),
        }
    }

    fn quantize_q8_0(data: &[f32]) -> Vec<u8> {
        const QK: usize = 32;
        let mut out = Vec::with_capacity(data.len().div_ceil(QK) * (2 + QK));
        for chunk in data.chunks(QK) {
            let max_abs = chunk.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
            out.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
            for i in 0..QK {
                let value = chunk.get(i).copied().unwrap_or(0.0);
                let q = if scale > 0.0 {
                    (value / scale).round().clamp(-128.0, 127.0) as i8
                } else {
                    0
                };
                out.push(q as u8);
            }
        }
        out
    }

    fn dequantize_q8_0(data: &[u8], output_len: usize) -> Vec<f32> {
        const QK: usize = 32;
        const BLOCK: usize = 2 + QK;
        let mut out = Vec::with_capacity(output_len);
        for block in data.chunks_exact(BLOCK) {
            let scale = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            for &q in &block[2..] {
                out.push((q as i8) as f32 * scale);
                if out.len() == output_len {
                    return out;
                }
            }
        }
        out
    }

    fn quantize_q4_0(data: &[f32]) -> Vec<u8> {
        const QK: usize = 32;
        const QB: usize = QK / 2;
        let mut out = Vec::with_capacity(data.len().div_ceil(QK) * (2 + QB));
        for chunk in data.chunks(QK) {
            let max_abs = chunk.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 0.0 };
            out.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
            for i in 0..QB {
                let lo = Self::quantize_symmetric_4bit(chunk.get(i).copied().unwrap_or(0.0), scale);
                let hi =
                    Self::quantize_symmetric_4bit(chunk.get(i + QB).copied().unwrap_or(0.0), scale);
                out.push((lo & 0x0f) | ((hi & 0x0f) << 4));
            }
        }
        out
    }

    fn dequantize_q4_0(data: &[u8], output_len: usize) -> Vec<f32> {
        const QK: usize = 32;
        const QB: usize = QK / 2;
        const BLOCK: usize = 2 + QB;
        let mut out = Vec::with_capacity(output_len);
        for block in data.chunks_exact(BLOCK) {
            let scale = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let qs = &block[2..];
            for &packed in qs {
                out.push(((packed & 0x0f) as i8 - 8) as f32 * scale);
                if out.len() == output_len {
                    return out;
                }
            }
            for &packed in qs {
                out.push(((packed >> 4) as i8 - 8) as f32 * scale);
                if out.len() == output_len {
                    return out;
                }
            }
        }
        out
    }

    fn quantize_symmetric_4bit(value: f32, scale: f32) -> u8 {
        if scale > 0.0 {
            let q = (value / scale).round().clamp(-8.0, 7.0) as i32;
            (q + 8) as u8
        } else {
            8
        }
    }

    fn quantize_q4_k(data: &[f32]) -> Vec<u8> {
        const QK: usize = 256;
        const SUB: usize = 32;
        const NSUB: usize = QK / SUB;
        let mut out = Vec::with_capacity(data.len().div_ceil(QK) * 144);

        for chunk in data.chunks(QK) {
            let mut block = [0.0f32; QK];
            block[..chunk.len()].copy_from_slice(chunk);

            let mut sub_scales = [0.0f32; NSUB];
            let mut sub_offsets = [0.0f32; NSUB];
            for sub in 0..NSUB {
                let values = &block[sub * SUB..(sub + 1) * SUB];
                let min = values.iter().copied().fold(f32::INFINITY, f32::min);
                let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let offset = if min < 0.0 { -min } else { 0.0 };
                let scale = ((max + offset) / 15.0).max(0.0);
                sub_scales[sub] = scale;
                sub_offsets[sub] = offset;
            }

            let max_scale = sub_scales.iter().copied().fold(0.0f32, f32::max);
            let max_offset = sub_offsets.iter().copied().fold(0.0f32, f32::max);
            let d = if max_scale > 0.0 {
                max_scale / 63.0
            } else {
                0.0
            };
            let dmin = if max_offset > 0.0 {
                max_offset / 63.0
            } else {
                0.0
            };

            let mut scales = [0u8; NSUB];
            let mut mins = [0u8; NSUB];
            for sub in 0..NSUB {
                scales[sub] = quantize_u6(sub_scales[sub], d);
                mins[sub] = quantize_u6(sub_offsets[sub], dmin);
            }

            out.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            out.extend_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
            out.extend_from_slice(&pack_q4_k_scales(&scales, &mins));

            let qs_start = out.len();
            out.resize(qs_start + QK / 2, 0);
            for i in 0..QK / 2 {
                let lo = quantize_q4_k_value(block[i], d, dmin, scales[i / SUB], mins[i / SUB]);
                let hi = quantize_q4_k_value(
                    block[i + QK / 2],
                    d,
                    dmin,
                    scales[(i + QK / 2) / SUB],
                    mins[(i + QK / 2) / SUB],
                );
                out[qs_start + i] = (lo & 0x0f) | ((hi & 0x0f) << 4);
            }
        }

        out
    }

    fn dequantize_q4_k(data: &[u8], output_len: usize) -> Vec<f32> {
        const QK: usize = 256;
        const BLOCK: usize = 144;
        let mut out = Vec::with_capacity(output_len);

        for block in data.chunks_exact(BLOCK) {
            let d = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let dmin = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
            let mut scales = [0u8; 8];
            let mut mins = [0u8; 8];
            unpack_q4_k_scales(&block[4..16], &mut scales, &mut mins);
            let qs = &block[16..];

            for i in 0..QK {
                let packed = qs[i % (QK / 2)];
                let q = if i < QK / 2 {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                let sub = i / 32;
                out.push(d * scales[sub] as f32 * q as f32 - dmin * mins[sub] as f32);
                if out.len() == output_len {
                    return out;
                }
            }
        }

        out
    }

    fn quantize_q2_k(data: &[f32]) -> Vec<u8> {
        const QK: usize = 256;
        const SUB: usize = 16;
        const NSUB: usize = QK / SUB;
        let mut out = Vec::with_capacity(data.len().div_ceil(QK) * 84);

        for chunk in data.chunks(QK) {
            let mut block = [0.0f32; QK];
            block[..chunk.len()].copy_from_slice(chunk);

            let mut sub_scales = [0.0f32; NSUB];
            let mut sub_offsets = [0.0f32; NSUB];
            for sub in 0..NSUB {
                let values = &block[sub * SUB..(sub + 1) * SUB];
                let min = values.iter().copied().fold(f32::INFINITY, f32::min);
                let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let offset = if min < 0.0 { -min } else { 0.0 };
                let scale = ((max + offset) / 3.0).max(0.0);
                sub_scales[sub] = scale;
                sub_offsets[sub] = offset;
            }

            let max_scale = sub_scales.iter().copied().fold(0.0f32, f32::max);
            let max_offset = sub_offsets.iter().copied().fold(0.0f32, f32::max);
            let d = if max_scale > 0.0 {
                max_scale / 15.0
            } else {
                0.0
            };
            let dmin = if max_offset > 0.0 {
                max_offset / 15.0
            } else {
                0.0
            };

            let scales_start = out.len();
            out.resize(scales_start + NSUB, 0);
            let qs_start = out.len();
            out.resize(qs_start + QK / 4, 0);

            for sub in 0..NSUB {
                let scale_q = quantize_u4(sub_scales[sub], d);
                let min_q = quantize_u4(sub_offsets[sub], dmin);
                out[scales_start + sub] = (scale_q & 0x0f) | ((min_q & 0x0f) << 4);
            }

            for i in 0..QK {
                let sub = i / SUB;
                let packed = out[scales_start + sub];
                let scale_q = packed & 0x0f;
                let min_q = packed >> 4;
                let q = quantize_q2_k_value(block[i], d, dmin, scale_q, min_q);
                out[qs_start + i / 4] |= (q & 0x03) << ((i % 4) * 2);
            }

            out.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            out.extend_from_slice(&half::f16::from_f32(dmin).to_bits().to_le_bytes());
        }

        out
    }

    fn dequantize_q2_k(data: &[u8], output_len: usize) -> Vec<f32> {
        const QK: usize = 256;
        const BLOCK: usize = 84;
        let mut out = Vec::with_capacity(output_len);

        for block in data.chunks_exact(BLOCK) {
            let scales = &block[..16];
            let qs = &block[16..80];
            let d = half::f16::from_bits(u16::from_le_bytes([block[80], block[81]])).to_f32();
            let dmin = half::f16::from_bits(u16::from_le_bytes([block[82], block[83]])).to_f32();

            for i in 0..QK {
                let q = (qs[i / 4] >> ((i % 4) * 2)) & 0x03;
                let packed = scales[i / 16];
                let scale = (packed & 0x0f) as f32;
                let min = (packed >> 4) as f32;
                out.push(d * scale * q as f32 - dmin * min);
                if out.len() == output_len {
                    return out;
                }
            }
        }

        out
    }

    fn quantize_affine_nbit(data: &[f32], bits: u32) -> Vec<u8> {
        const QK: usize = 32;
        let packed_bytes = (QK * bits as usize).div_ceil(8);
        let mut out = Vec::with_capacity(data.len().div_ceil(QK) * (4 + packed_bytes));
        let levels = (1u32 << bits) - 1;

        for chunk in data.chunks(QK) {
            let min = chunk.iter().copied().fold(f32::INFINITY, f32::min);
            let max = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = if (max - min).abs() > 1e-8 {
                (max - min) / levels as f32
            } else {
                1.0
            };
            out.extend_from_slice(&half::f16::from_f32(min).to_bits().to_le_bytes());
            out.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
            let packed_start = out.len();
            out.resize(packed_start + packed_bytes, 0);

            for i in 0..QK {
                let value = chunk.get(i).copied().unwrap_or(min);
                let q = ((value - min) / scale).round().clamp(0.0, levels as f32) as u32;
                Self::write_bits(
                    &mut out[packed_start..packed_start + packed_bytes],
                    i,
                    bits,
                    q,
                );
            }
        }
        out
    }

    fn dequantize_affine_nbit(data: &[u8], output_len: usize, bits: u32) -> Vec<f32> {
        const QK: usize = 32;
        let packed_bytes = (QK * bits as usize).div_ceil(8);
        let block_bytes = 4 + packed_bytes;
        let mut out = Vec::with_capacity(output_len);
        let mask = (1u32 << bits) - 1;

        for block in data.chunks_exact(block_bytes) {
            let min = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
            let scale = half::f16::from_bits(u16::from_le_bytes([block[2], block[3]])).to_f32();
            let packed = &block[4..];
            for i in 0..QK {
                let q = Self::read_bits(packed, i, bits) & mask;
                out.push(min + q as f32 * scale);
                if out.len() == output_len {
                    return out;
                }
            }
        }
        out
    }

    fn write_bits(data: &mut [u8], index: usize, bits: u32, value: u32) {
        let bit_offset = index * bits as usize;
        for bit in 0..bits as usize {
            if ((value >> bit) & 1) == 0 {
                continue;
            }
            let absolute = bit_offset + bit;
            data[absolute / 8] |= 1 << (absolute % 8);
        }
    }

    fn read_bits(data: &[u8], index: usize, bits: u32) -> u32 {
        let bit_offset = index * bits as usize;
        let mut value = 0u32;
        for bit in 0..bits as usize {
            let absolute = bit_offset + bit;
            let bit_value = (data[absolute / 8] >> (absolute % 8)) & 1;
            value |= (bit_value as u32) << bit;
        }
        value
    }

    /// TurboQuant: quantize KV cache entries
    /// Returns compressed (key_bits + val_bits) per element
    pub fn turboquant_kv(&mut self, keys: &[f32], values: &[f32]) -> (Vec<u8>, Vec<u8>) {
        let key_dim = keys.len();
        let val_dim = values.len();
        let key_bits = self.kv_quant.key_bits();
        let val_bits = self.kv_quant.val_bits();

        // Get codebooks (ensure they exist)
        {
            self.get_codebook(key_bits, key_dim);
            self.get_codebook(val_bits, val_dim);
        }

        // Apply rotation if available
        let rotated_keys = if let Some(rotation) = self.rotations.first() {
            rotation.apply(keys)
        } else {
            keys.to_vec()
        };

        // Get codebooks for actual use
        let key_codebook = self.codebooks.get(&(key_bits, key_dim)).unwrap();
        let val_codebook = self.codebooks.get(&(val_bits, val_dim)).unwrap();

        // Quantize
        let quantized_keys = key_codebook.encode(&rotated_keys);
        let quantized_values = val_codebook.encode(values);

        (quantized_keys, quantized_values)
    }

    /// TurboQuant: dequantize KV cache entries
    pub fn turboquant_dequant_kv(
        &mut self,
        quant_keys: &[u8],
        quant_values: &[u8],
    ) -> (Vec<f32>, Vec<f32>) {
        let key_dim = self.rotations.first().map(|r| r.dim).unwrap_or(256);
        let val_dim = key_dim;
        let key_bits = self.kv_quant.key_bits();
        let val_bits = self.kv_quant.val_bits();

        // Get codebooks for actual use
        let key_codebook = self.codebooks.get(&(key_bits, key_dim)).unwrap();
        let val_codebook = self.codebooks.get(&(val_bits, val_dim)).unwrap();

        let dequant_keys = key_codebook.decode_exact(quant_keys, key_dim);
        let dequant_values = val_codebook.decode_exact(quant_values, val_dim);

        // Apply inverse rotation if available
        let final_keys = if let Some(rotation) = self.rotations.first() {
            rotation.apply_inverse(&dequant_keys)
        } else {
            dequant_keys
        };

        (final_keys, dequant_values)
    }

    /// Calculate KV cache memory usage for a given configuration
    pub fn kv_cache_memory(&self, num_layers: usize, head_dim: usize, seq_len: usize) -> usize {
        match self.kv_quant {
            KvQuantType::None => {
                // FP16: 2 bytes per element per key/value
                num_layers * 2 * head_dim * seq_len * 2
            }
            _ => {
                let bits = num_layers
                    * head_dim
                    * seq_len
                    * (self.kv_quant.key_bits() as usize + self.kv_quant.val_bits() as usize);
                bits.div_ceil(8)
            }
        }
    }
}

fn write_packed_bits(data: &mut [u8], index: usize, bits: u32, value: u32) {
    let bit_offset = index * bits as usize;
    for bit in 0..bits as usize {
        if ((value >> bit) & 1) == 0 {
            continue;
        }
        let absolute = bit_offset + bit;
        data[absolute / 8] |= 1 << (absolute % 8);
    }
}

fn read_packed_bits(data: &[u8], index: usize, bits: u32) -> u32 {
    let bit_offset = index * bits as usize;
    let mut value = 0u32;
    for bit in 0..bits as usize {
        let absolute = bit_offset + bit;
        let byte = data.get(absolute / 8).copied().unwrap_or(0);
        let bit_value = (byte >> (absolute % 8)) & 1;
        value |= (bit_value as u32) << bit;
    }
    value
}

fn quantize_u6(value: f32, scale: f32) -> u8 {
    if scale > 0.0 && value > 0.0 {
        (value / scale).round().clamp(1.0, 63.0) as u8
    } else {
        0
    }
}

fn quantize_u4(value: f32, scale: f32) -> u8 {
    if scale > 0.0 && value > 0.0 {
        (value / scale).round().clamp(1.0, 15.0) as u8
    } else {
        0
    }
}

fn quantize_q4_k_value(value: f32, d: f32, dmin: f32, scale_q: u8, min_q: u8) -> u8 {
    let scale = d * scale_q as f32;
    let offset = dmin * min_q as f32;
    if scale > 0.0 {
        ((value + offset) / scale).round().clamp(0.0, 15.0) as u8
    } else {
        0
    }
}

fn quantize_q2_k_value(value: f32, d: f32, dmin: f32, scale_q: u8, min_q: u8) -> u8 {
    let scale = d * scale_q as f32;
    let offset = dmin * min_q as f32;
    if scale > 0.0 {
        ((value + offset) / scale).round().clamp(0.0, 3.0) as u8
    } else {
        0
    }
}

fn pack_q4_k_scales(scales: &[u8; 8], mins: &[u8; 8]) -> [u8; 12] {
    let mut out = [0u8; 12];
    for i in 0..4 {
        out[i] = (scales[i] & 0x3f) | ((scales[i + 4] >> 4) << 6);
        out[i + 4] = (mins[i] & 0x3f) | ((mins[i + 4] >> 4) << 6);
        out[i + 8] = (scales[i + 4] & 0x0f) | ((mins[i + 4] & 0x0f) << 4);
    }
    out
}

fn unpack_q4_k_scales(packed: &[u8], scales: &mut [u8; 8], mins: &mut [u8; 8]) {
    for i in 0..4 {
        scales[i] = packed[i] & 0x3f;
        mins[i] = packed[i + 4] & 0x3f;
        scales[i + 4] = (packed[i + 8] & 0x0f) | ((packed[i] >> 6) << 4);
        mins[i + 4] = (packed[i + 8] >> 4) | ((packed[i + 4] >> 6) << 4);
    }
}

impl Default for QuantEngine {
    fn default() -> Self {
        QuantEngine::new(QuantFormat::Q4_K, KvQuantType::TurboQuant3b2b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quant_format_names() {
        assert_eq!(QuantFormat::Q2_K.name(), "Q2_K");
        assert_eq!(QuantFormat::Q4_K.name(), "Q4_K");
        assert_eq!(QuantFormat::MXFP4.name(), "MXFP4");
    }

    #[test]
    fn test_quant_format_bits() {
        assert_eq!(QuantFormat::Q2_K.bits(), 2);
        assert_eq!(QuantFormat::Q4_0.bits(), 4);
        assert_eq!(QuantFormat::Q6_K.bits(), 6);
        assert_eq!(QuantFormat::Q8_0.bits(), 8);
    }

    #[test]
    fn test_quant_format_compression() {
        assert_eq!(QuantFormat::Q2_K.compression_ratio(), 0.125);
        assert_eq!(QuantFormat::Q4_K.compression_ratio(), 0.25);
        assert_eq!(QuantFormat::Q8_0.compression_ratio(), 0.5);
    }

    #[test]
    fn test_quant_format_from_str() {
        assert_eq!(QuantFormat::from_str("q4_0"), Some(QuantFormat::Q4_0));
        assert_eq!(QuantFormat::from_str("Q8_K"), Some(QuantFormat::Q8_K));
        assert_eq!(QuantFormat::from_str("invalid"), None);
    }

    #[test]
    fn test_kv_quant_types() {
        assert_eq!(KvQuantType::None.bits(), 16);
        assert_eq!(KvQuantType::TurboQuant3b2b.key_bits(), 3);
        assert_eq!(KvQuantType::TurboQuant3b2b.val_bits(), 2);
        assert_eq!(KvQuantType::TurboQuant4b4b.compression_ratio(), 2.0);
    }

    #[test]
    fn test_rotation_matrix() {
        let rot = RotationMatrix::generate_random(64);
        assert_eq!(rot.dim, 64);
        assert_eq!(rot.data.len(), 64 * 64);

        // Test apply and inverse
        let x: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let y = rot.apply(&x);
        let x_back = rot.apply_inverse(&y);

        // Check reconstruction error
        let max_err: f32 = x
            .iter()
            .zip(x_back.iter())
            .map(|(a, b)| (a - b).abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        assert!(
            max_err < 1e-4,
            "Rotation reconstruction error too large: {}",
            max_err
        );
    }

    #[test]
    fn test_codebook() {
        let cb = Codebook::new(4, 128);
        assert_eq!(cb.bits, 4);
        assert_eq!(cb.levels.len(), 16);

        // Test quantize
        let q = cb.quantize(0.5);
        assert!((-1.0..=1.0).contains(&q));

        // Test encode/decode roundtrip
        let test_vals: Vec<f32> = (0..16).map(|i| i as f32 * 0.1 - 0.75).collect();
        let encoded = cb.encode(&test_vals);
        let decoded = cb.decode(&encoded);
        assert_eq!(decoded.len(), test_vals.len());
    }

    #[test]
    fn test_codebook_3bit_is_bit_packed() {
        let cb = Codebook::new(3, 16);
        let values: Vec<f32> = (0..16).map(|i| i as f32 / 8.0 - 1.0).collect();
        let encoded = cb.encode(&values);
        assert_eq!(encoded.len(), 6);
        let decoded = cb.decode_exact(&encoded, values.len());
        assert_eq!(decoded.len(), values.len());
    }

    #[test]
    fn test_quant_engine() {
        let engine = QuantEngine::new(QuantFormat::Q4_K, KvQuantType::TurboQuant3b2b);
        assert_eq!(engine.weight_format, QuantFormat::Q4_K);
        assert_eq!(engine.kv_quant, KvQuantType::TurboQuant3b2b);

        // Test weight quantization
        let data: Vec<f32> = (0..64).map(|i| i as f32 * 0.1 - 3.2).collect();
        let quantized = engine.quantize_weights(&data);
        assert!(!quantized.is_empty());

        // Test KV cache memory calculation
        let mem = engine.kv_cache_memory(4, 64, 1024);
        assert_eq!(mem, (4usize * 64 * 1024 * (3 + 2)).div_ceil(8));
    }

    #[test]
    fn test_q8_weight_quant_roundtrip() {
        let engine = QuantEngine::new(QuantFormat::Q8_0, KvQuantType::None);
        let data: Vec<f32> = (0..64).map(|i| i as f32 / 16.0 - 2.0).collect();
        let quantized = engine.quantize_weights(&data);
        let decoded = engine.dequantize(&quantized, data.len());

        assert_eq!(decoded.len(), data.len());
        for (actual, expected) in decoded.iter().zip(data.iter()) {
            assert!((actual - expected).abs() < 0.03);
        }
    }

    #[test]
    fn test_q4_weight_quant_roundtrip() {
        let engine = QuantEngine::new(QuantFormat::Q4_0, KvQuantType::None);
        let data: Vec<f32> = (0..64).map(|i| i as f32 / 16.0 - 2.0).collect();
        let quantized = engine.quantize_weights(&data);
        let decoded = engine.dequantize(&quantized, data.len());

        assert_eq!(decoded.len(), data.len());
        for (actual, expected) in decoded.iter().zip(data.iter()) {
            assert!((actual - expected).abs() < 0.3);
        }
    }

    #[test]
    fn test_q4_k_weight_quant_roundtrip() {
        let engine = QuantEngine::new(QuantFormat::Q4_K, KvQuantType::None);
        let data: Vec<f32> = (0..300).map(|i| i as f32 / 31.0 - 4.0).collect();
        let quantized = engine.quantize_weights(&data);
        assert_eq!(quantized.len(), data.len().div_ceil(256) * 144);
        let decoded = engine.dequantize(&quantized, data.len());

        assert_eq!(decoded.len(), data.len());
        for (actual, expected) in decoded.iter().zip(data.iter()) {
            assert!((actual - expected).abs() < 0.4);
        }
    }

    #[test]
    fn test_q2_k_weight_quant_roundtrip() {
        let engine = QuantEngine::new(QuantFormat::Q2_K, KvQuantType::None);
        let data: Vec<f32> = (0..300).map(|i| i as f32 / 31.0 - 4.0).collect();
        let quantized = engine.quantize_weights(&data);
        assert_eq!(quantized.len(), data.len().div_ceil(256) * 84);
        let decoded = engine.dequantize(&quantized, data.len());

        assert_eq!(decoded.len(), data.len());
        for (actual, expected) in decoded.iter().zip(data.iter()) {
            assert!((actual - expected).abs() < 1.6);
        }
    }

    #[test]
    fn test_affine_nbit_weight_quant_roundtrip() {
        let engine = QuantEngine::new(QuantFormat::Q3_K, KvQuantType::None);
        let data: Vec<f32> = (0..64).map(|i| i as f32 / 16.0 - 2.0).collect();
        let quantized = engine.quantize_weights(&data);
        let decoded = engine.dequantize(&quantized, data.len());

        assert_eq!(decoded.len(), data.len());
        for (actual, expected) in decoded.iter().zip(data.iter()) {
            assert!((actual - expected).abs() < 0.6);
        }
    }

    #[test]
    fn test_turboquant_roundtrip() {
        let mut engine = QuantEngine::new(QuantFormat::FP16, KvQuantType::TurboQuant3b2b);
        engine.generate_rotations(64, 1);

        let keys: Vec<f32> = (0..64).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect();
        let values: Vec<f32> = (0..64).map(|_| rand::random::<f32>() * 2.0 - 1.0).collect();

        let (qk, qv) = engine.turboquant_kv(&keys, &values);
        let (dk, dv) = engine.turboquant_dequant_kv(&qk, &qv);

        assert_eq!(dk.len(), keys.len());
        assert_eq!(dv.len(), values.len());

        // Check cosine similarity
        let dot: f32 = keys.iter().zip(dk.iter()).map(|(a, b)| a * b).sum();
        let norm_k: f32 = keys.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_d: f32 = dk.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_k > 1e-8 && norm_d > 1e-8 {
            let cos_sim = dot / (norm_k * norm_d);
            assert!(cos_sim > 0.9, "Key cosine similarity too low: {}", cos_sim);
        }
    }

    #[test]
    fn test_group_size() {
        assert_eq!(GroupSize::N32.value(), 32);
        assert_eq!(GroupSize::N64.value(), 64);
        assert_eq!(GroupSize::from_str("128"), Some(GroupSize::N128));
    }

    #[test]
    fn test_quant_params() {
        let p = QuantParams::new(0.5, -0.1);
        assert_eq!(p.scale, 0.5);
        assert_eq!(p.delta, -0.1);

        let identity = QuantParams::identity();
        assert_eq!(identity.scale, 1.0);
        assert_eq!(identity.delta, 0.0);
    }
}
