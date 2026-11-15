//! CPU projections with runtime AVX2/FMA dispatch and a portable scalar fallback.
//! Default execution is single-threaded. Explicitly configured private thread
//! pools are used only above a conservative work threshold.
use rayon::prelude::*;

use super::{Backend, BackendCapabilities, BackendType, KernelOp, KernelOutput, MemoryInfo};
use crate::quant::{KvQuantType, QuantEngine, QuantFormat};

const PARALLEL_MIN_ELEMENTS: usize = 1 << 20;
type Dot = fn(&[f32], &[f32]) -> f32;

pub struct CpuBackend {
    dot: Dot,
    kernel: &'static str,
    pool: Option<rayon::ThreadPool>,
    threads: usize,
}

impl CpuBackend {
    pub fn new() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return Self {
                dot: dot_avx2_fma,
                kernel: "avx2-fma",
                pool: None,
                threads: 1,
            };
        }
        Self::scalar()
    }

    /// Ordered scalar baseline, also used automatically on non-AVX2/FMA CPUs.
    pub fn scalar() -> Self {
        Self {
            dot: dot_scalar,
            kernel: "scalar",
            pool: None,
            threads: 1,
        }
    }

    /// Explicit per-backend limit. No global Rayon settings are changed.
    pub fn with_threads(threads: usize) -> Result<Self, String> {
        let maximum = num_cpus::get().max(1);
        if threads == 0 || threads > maximum {
            return Err(format!("CPU threads must be between 1 and {maximum}"));
        }
        let mut backend = Self::new();
        if threads > 1 {
            backend.pool = Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|index| format!("nexus-cpu-{index}"))
                    .build()
                    .map_err(|e| format!("CPU thread pool: {e}"))?,
            );
        }
        backend.threads = threads;
        Ok(backend)
    }

    pub fn thread_count(&self) -> usize {
        self.threads
    }
    /// Selected strategy; vectors shorter than 32 elements use the scalar path.
    pub fn matvec_kernel(&self) -> &'static str {
        self.kernel
    }

    fn parallel_matvec(&self, rows: usize, cols: usize) -> bool {
        self.pool.is_some()
            && rows >= 128
            && cols >= 128
            && rows.saturating_mul(cols) >= PARALLEL_MIN_ELEMENTS
    }

    fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<f32>, String> {
        if m == 0
            || n == 0
            || k == 0
            || m.checked_mul(k) != Some(a.len())
            || k.checked_mul(n) != Some(b.len())
            || m.checked_mul(n).is_none()
        {
            return Err("Invalid matmul dimensions".into());
        }
        finite(a)?;
        finite(b)?;
        let mut c = vec![0.0; m * n];
        let row_kernel = |(i, row): (usize, &mut [f32])| {
            for kk in 0..k {
                let x = a[i * k + kk];
                for (out, &w) in row.iter_mut().zip(&b[kk * n..(kk + 1) * n]) {
                    *out += x * w;
                }
            }
        };
        if let Some(pool) = &self.pool {
            if m.saturating_mul(n).saturating_mul(k) >= PARALLEL_MIN_ELEMENTS && m >= 128 {
                pool.install(|| c.par_chunks_mut(n).enumerate().for_each(row_kernel));
            } else {
                c.chunks_mut(n).enumerate().for_each(row_kernel);
            }
        } else {
            c.chunks_mut(n).enumerate().for_each(row_kernel);
        }
        Ok(c)
    }

    fn attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        scale: f32,
        mask: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        let dim = q.len();
        if dim == 0
            || k.is_empty()
            || !k.len().is_multiple_of(dim)
            || k.len() != v.len()
            || !scale.is_finite()
            || scale <= 0.0
        {
            return Err("Invalid attention dimensions or scale".into());
        }
        let tokens = k.len() / dim;
        if mask.is_some_and(|m| {
            m.len() != tokens || m.iter().any(|x| x.is_nan() || *x == f32::INFINITY)
        }) {
            return Err(
                "Attention mask must contain one finite or -infinity additive value per KV token"
                    .into(),
            );
        }
        finite(q)?;
        finite(k)?;
        finite(v)?;
        let mut scores: Vec<f32> = k
            .chunks_exact(dim)
            .enumerate()
            .map(|(i, key)| {
                q.iter().zip(key).map(|(a, b)| a * b).sum::<f32>() * scale
                    + mask.map_or(0.0, |m| m[i])
            })
            .collect();
        if scores.iter().any(|x| x.is_nan() || *x == f32::INFINITY) {
            return Err("Attention score overflow".into());
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut out = vec![0.0; dim];
        if max == f32::NEG_INFINITY {
            return Ok(out);
        }
        let mut total = 0.0;
        for score in &mut scores {
            *score = (*score - max).exp();
            total += *score;
        }
        for (score, value) in scores.into_iter().zip(v.chunks_exact(dim)) {
            let probability = score / total;
            for (out, &value) in out.iter_mut().zip(value) {
                *out += probability * value;
            }
        }
        Ok(out)
    }
}

fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Only installed by new() after detecting both CPU features. Every load is
// unaligned and wholly inside equal-length slices; the remainder is scalar.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn dot_avx2_fma(a: &[f32], b: &[f32]) -> f32 {
    if a.len() < 32 {
        return dot_scalar(a, b);
    }
    // SAFETY: this function pointer is only selected after runtime AVX2+FMA
    // detection. matvec validates both slice lengths before entering it.
    unsafe { dot_avx2_fma_inner(a, b) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2_fma_inner(a: &[f32], b: &[f32]) -> f32 {
    let mut s0 = _mm256_setzero_ps();
    let mut s1 = _mm256_setzero_ps();
    let mut s2 = _mm256_setzero_ps();
    let mut s3 = _mm256_setzero_ps();
    let mut i = 0;
    while i + 32 <= a.len() {
        // SAFETY: loop bounds prove all four 8-element loads fit both slices.
        unsafe {
            s0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
                s0,
            );
            s1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i + 8)),
                _mm256_loadu_ps(b.as_ptr().add(i + 8)),
                s1,
            );
            s2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i + 16)),
                _mm256_loadu_ps(b.as_ptr().add(i + 16)),
                s2,
            );
            s3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i + 24)),
                _mm256_loadu_ps(b.as_ptr().add(i + 24)),
                s3,
            );
        }
        i += 32;
    }
    while i + 8 <= a.len() {
        // SAFETY: same checked slice bounds for the last complete vector.
        unsafe {
            s0 = _mm256_fmadd_ps(
                _mm256_loadu_ps(a.as_ptr().add(i)),
                _mm256_loadu_ps(b.as_ptr().add(i)),
                s0,
            );
        }
        i += 8;
    }
    let sum = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    let mut lanes = [0.0f32; 8];
    // SAFETY: lanes owns eight writable F32 values; storeu has no alignment demand.
    unsafe {
        _mm256_storeu_ps(lanes.as_mut_ptr(), sum);
    }
    let mut result: f32 = lanes.into_iter().sum();
    for (&a, &b) in a[i..].iter().zip(&b[i..]) {
        result += a * b;
    }
    result
}

fn finite(values: &[f32]) -> Result<(), String> {
    if values.iter().any(|x| !x.is_finite()) {
        Err("Kernel inputs must be finite".into())
    } else {
        Ok(())
    }
}

fn norm_shape(input: &[f32], weight: &[f32], bias: Option<&[f32]>, eps: f32) -> Result<(), String> {
    if input.is_empty()
        || input.len() != weight.len()
        || bias.is_some_and(|b| b.len() != input.len())
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err("Invalid normalization dimensions or epsilon".into());
    }
    finite(input)?;
    finite(weight)?;
    if let Some(b) = bias {
        finite(b)?;
    }
    Ok(())
}

// QuantEngine exposes additional experimental aliases. Do not advertise those
// as interoperable GGUF formats through this public backend contract.
fn quant_size(format: QuantFormat, len: usize) -> Result<usize, String> {
    if len == 0 {
        return Err("Quantization length must be positive".into());
    }
    let (block, bytes) = match format {
        QuantFormat::FP32 => (1, 4),
        QuantFormat::FP16 => (1, 2),
        QuantFormat::Q8_0 => (32, 34),
        QuantFormat::Q4_0 => (32, 18),
        QuantFormat::Q4_K => (256, 144),
        QuantFormat::Q2_K => (256, 84),
        _ => {
            return Err(format!(
                "Unsupported backend quantization format: {}",
                format.name()
            ))
        }
    };
    len.div_ceil(block)
        .checked_mul(bytes)
        .ok_or_else(|| "Quantization size overflow".into())
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu-simd"
    }
    fn backend_type(&self) -> BackendType {
        BackendType::CpuSimd
    }
    fn is_available(&self) -> bool {
        true
    }
    fn memory_info(&self) -> MemoryInfo {
        // No OS memory measurement is implemented. Zero means unknown; avoid
        // fabricating a fixed 8 GiB machine or using it for admission decisions.
        MemoryInfo {
            total_memory: 0,
            free_memory: 0,
            used_memory: 0,
        }
    }
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            backend: BackendType::CpuSimd,
            available: true,
            features: vec![
                "portable-f32-fallback",
                self.kernel,
                if self.pool.is_some() {
                    "private-pool-thresholded"
                } else {
                    "single-thread"
                },
                "memory-info-unknown",
            ],
            supported_ops: vec![
                "matvec",
                "matmul",
                "attention",
                "rms_norm",
                "layer_norm",
                "add",
                "mul",
                "softmax",
                "quantize",
                "dequantize",
            ],
        }
    }
    fn matvec(
        &self,
        weight: &[f32],
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if rows == 0
            || cols == 0
            || input.len() != cols
            || rows.checked_mul(cols) != Some(weight.len())
        {
            return Err("Invalid matvec dimensions".into());
        }
        finite(input)?;
        let mut output = vec![0.0; rows];
        let dot = self.dot;
        if self.parallel_matvec(rows, cols) {
            self.pool.as_ref().unwrap().install(|| {
                output
                    .par_iter_mut()
                    .zip(weight.par_chunks_exact(cols))
                    .for_each(|(out, row)| *out = dot(row, input));
            });
        } else {
            for (out, row) in output.iter_mut().zip(weight.chunks_exact(cols)) {
                *out = dot(row, input);
            }
        }
        // No separate weights scan/doubled memory bandwidth: NaN/Inf weights
        // necessarily propagate into their row result. Reject numerical overflow.
        finite(&output)?;
        Ok(output)
    }
    fn execute(&self, op: &KernelOp) -> Result<KernelOutput, String> {
        let out = match op {
            KernelOp::MatMul { a, b, m, n, k } => self.matmul(a, b, *m, *n, *k)?,
            KernelOp::Attention {
                q,
                k,
                v,
                scale,
                mask,
            } => self.attention(q, k, v, *scale, mask.as_deref())?,
            KernelOp::RmsNorm { input, weight, eps } => {
                norm_shape(input, weight, None, *eps)?;
                crate::kernel::rms_norm(input, weight, *eps)
            }
            KernelOp::LayerNorm {
                input,
                weight,
                bias,
                eps,
            } => {
                norm_shape(input, weight, bias.as_deref(), *eps)?;
                crate::kernel::layer_norm(input, weight, bias.as_deref(), *eps)
            }
            KernelOp::Add { a, b } | KernelOp::Mul { a, b } => {
                if a.is_empty() || a.len() != b.len() {
                    return Err("Elementwise shapes must match and be nonempty".into());
                }
                finite(a)?;
                finite(b)?;
                a.iter()
                    .zip(b.iter())
                    .map(|(a, b)| {
                        if matches!(op, KernelOp::Add { .. }) {
                            a + b
                        } else {
                            a * b
                        }
                    })
                    .collect()
            }
            KernelOp::Softmax { input, dim } => {
                if *dim == 0 || input.is_empty() || !input.len().is_multiple_of(*dim) {
                    return Err("Softmax row width must divide input length".into());
                }
                finite(input)?;
                input
                    .chunks_exact(*dim)
                    .flat_map(|row| crate::kernel::softmax(row, *dim))
                    .collect()
            }
            KernelOp::Quantize { input, format } => {
                quant_size(*format, input.len())?;
                finite(input)?;
                return Ok(KernelOutput::U8(
                    QuantEngine::new(*format, KvQuantType::None)
                        .quantize_weights(input)
                        .map_err(|e| e.to_string())?,
                ));
            }
            KernelOp::Dequantize {
                input,
                format,
                output_len,
            } => {
                if input.len() != quant_size(*format, *output_len)? {
                    return Err("Invalid quantized buffer length".into());
                }
                let values = QuantEngine::new(*format, KvQuantType::None)
                    .dequantize(input, *output_len)
                    .map_err(|e| e.to_string())?;
                if values.len() != *output_len {
                    return Err("Dequantization returned an invalid shape".into());
                }
                values
            }
        };
        finite(&out)?;
        Ok(KernelOutput::F32(out))
    }
    fn synchronize(&self) {}
}
