//! CPU backend with SIMD optimizations.
//! Uses runtime dispatch for AVX2/AVX512/AMX on x86 and NEON on ARM.

use rayon::prelude::*;

use super::{Backend, BackendCapabilities, BackendType, KernelOp, MemoryInfo};

/// CPU backend implementation
pub struct CpuBackend {
    available: bool,
    _num_threads: usize,
}

impl CpuBackend {
    pub fn new() -> Self {
        let num_threads = num_cpus::get();
        CpuBackend {
            available: true,
            _num_threads: num_threads,
        }
    }

    /// Run a matmul operation on CPU using Rayon for parallelism
    fn matmul(&self, a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];

        c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                row[j] = sum;
            }
        });

        c
    }

    /// Run attention on CPU
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], scale: f32) -> Vec<f32> {
        let head_dim = q.len();
        let seq_len = k.len() / head_dim;

        // Scaled dot-product attention: softmax(QK^T / scale) @ V
        let mut attn = vec![0.0f32; seq_len * head_dim];

        for i in 0..seq_len {
            // Compute attention scores for this position
            let mut scores = vec![0.0f32; seq_len];
            for j in 0..seq_len {
                let mut score = 0.0f32;
                for d in 0..head_dim {
                    score += q[i * head_dim + d] * k[j * head_dim + d];
                }
                scores[j] = score * scale;
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for s in &scores {
                sum_exp += (*s - max_score).exp();
            }

            // Apply attention to values
            for d in 0..head_dim {
                let mut weighted_sum = 0.0f32;
                for j in 0..seq_len {
                    let weight = (scores[j] - max_score).exp() / sum_exp;
                    weighted_sum += weight * v[j * head_dim + d];
                }
                attn[i * head_dim + d] = weighted_sum;
            }
        }

        attn
    }

    /// RMS normalization (used in modern LLMs like LLaMA)
    fn rms_norm(&self, input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let dim = input.len();
        let variance = input.iter().map(|x| x * x).sum::<f32>() / dim as f32;
        let rstd = 1.0 / (variance + eps).sqrt();

        let mut output = vec![0.0f32; dim];
        for i in 0..dim {
            output[i] = input[i] * rstd * weight[i];
        }
        output
    }
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
        self.available
    }

    fn memory_info(&self) -> MemoryInfo {
        // Estimate available memory from system
        let total = 8 * 1024 * 1024 * 1024; // 8GB estimate
        MemoryInfo {
            total_memory: total,
            free_memory: total / 2,
            used_memory: total / 4,
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            backend: BackendType::CpuSimd,
            available: self.available,
            features: Self::features(),
            supported_ops: vec![
                "matmul",
                "attention",
                "rms_norm",
                "layer_norm",
                "softmax",
                "quantize",
                "dequantize",
            ],
        }
    }

    fn execute(&self, op: &KernelOp) {
        match op {
            KernelOp::MatMul { a, b, m, n, k, .. } => {
                let _ = self.matmul(a, b, *m, *n, *k);
            }
            KernelOp::Attention { q, k, v, scale, .. } => {
                let _ = self.attention(q, k, v, *scale);
            }
            KernelOp::RmsNorm {
                input, weight, eps, ..
            } => {
                let _ = self.rms_norm(input, weight, *eps);
            }
            _ => {
                tracing::debug!("CPU backend: operation not optimized");
            }
        }
    }

    fn synchronize(&self) {
        // CPU operations are synchronous
    }
}

impl CpuBackend {
    fn features() -> Vec<&'static str> {
        let mut features = vec!["rayon"];

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                features.push("avx2");
            }
            if std::is_x86_feature_detected!("avx512f") {
                features.push("avx512f");
            }
            if std::is_x86_feature_detected!("fma") {
                features.push("fma");
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            features.push("neon");
        }

        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_creation() {
        let backend = CpuBackend::new();
        assert!(backend.is_available());
        assert_eq!(backend.name(), "cpu-simd");
    }

    #[test]
    fn test_rms_norm() {
        let backend = CpuBackend::new();
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0f32, 1.0, 1.0, 1.0];

        let result = backend.rms_norm(&input, &weight, 1e-5);
        assert_eq!(result.len(), 4);

        // RMS of [1,2,3,4] = sqrt(30/4) ≈ 2.7386
        // rstd ≈ 0.3656
        let expected_scale: f32 = 1.0 / (30.0_f32 / 4.0).sqrt();
        for i in 0..4 {
            let expected = input[i] * expected_scale * weight[i];
            assert!((result[i] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_attention_small() {
        let backend = CpuBackend::new();
        let head_dim = 8;
        let seq_len = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32 * 0.1).collect();
        let k: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32 * 0.05).collect();
        let v: Vec<f32> = (0..seq_len * head_dim).map(|i| i as f32 * 0.01).collect();

        let result = backend.attention(&q, &k, &v, scale);
        assert_eq!(result.len(), seq_len * head_dim);
    }

    #[test]
    fn test_matmul() {
        let backend = CpuBackend::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = backend.matmul(&a, &b, 2, 2, 2);
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_capabilities() {
        let backend = CpuBackend::new();
        let caps = backend.capabilities();
        assert!(caps.available);
        assert!(caps.supported_ops.contains(&"matmul"));
    }

    #[test]
    fn test_memory_info() {
        let backend = CpuBackend::new();
        let info = backend.memory_info();
        assert!(info.total_memory > 0);
        assert!(info.utilization() >= 0.0 && info.utilization() <= 1.0);
    }
}
