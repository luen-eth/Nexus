//! Backend abstraction for different hardware targets.
//! Supports CPU SIMD, Metal (Apple Silicon), and CUDA backends.

use std::sync::Arc;

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod metal;

/// Supported backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendType {
    /// CPU with SIMD optimizations (AVX2/AVX512/AMX on x86, NEON on ARM)
    CpuSimd,
    /// Apple Metal GPU
    Metal,
    /// NVIDIA CUDA GPU
    Cuda,
    /// Vulkan compute backend
    Vulkan,
    /// Browser/WebGPU backend
    WebGpu,
}

/// Backend capability summary for runtime selection and diagnostics.
#[derive(Debug, Clone)]
pub struct BackendCapabilities {
    pub backend: BackendType,
    pub available: bool,
    pub features: Vec<&'static str>,
    pub supported_ops: Vec<&'static str>,
}

impl BackendType {
    pub fn name(&self) -> &'static str {
        match self {
            BackendType::CpuSimd => "cpu-simd",
            BackendType::Metal => "metal",
            BackendType::Cuda => "cuda",
            BackendType::Vulkan => "vulkan",
            BackendType::WebGpu => "webgpu",
        }
    }

    pub fn is_gpu(&self) -> bool {
        match self {
            BackendType::CpuSimd => false,
            BackendType::Metal | BackendType::Cuda | BackendType::Vulkan | BackendType::WebGpu => {
                true
            }
        }
    }
}

impl Default for BackendType {
    fn default() -> Self {
        Self::detect()
    }
}

impl BackendType {
    /// Auto-detect the best available backend
    pub fn detect() -> Self {
        // GPU device probing does not imply an implemented inference backend.
        BackendType::CpuSimd
    }
}

/// Backend trait - all hardware backends implement this
pub trait Backend: Send + Sync {
    /// Backend name
    fn name(&self) -> &str;

    /// Backend type
    fn backend_type(&self) -> BackendType;

    /// Check if the backend is available
    fn is_available(&self) -> bool;

    /// Get device memory info
    fn memory_info(&self) -> MemoryInfo;

    /// Get backend capabilities.
    fn capabilities(&self) -> BackendCapabilities;

    fn prepare_weights(
        &self,
        _tensors: &std::collections::HashMap<String, Vec<f32>>,
    ) -> Result<(), String> {
        Ok(())
    }

    /// Dense F32 projection. Backends must return errors rather than discard results.
    fn matvec(
        &self,
        weight: &[f32],
        rows: usize,
        cols: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if cols == 0
            || rows == 0
            || input.len() != cols
            || rows.checked_mul(cols) != Some(weight.len())
        {
            return Err("Invalid matvec dimensions".into());
        }
        Ok(weight
            .chunks_exact(cols)
            .map(|row| row.iter().zip(input).map(|(w, x)| w * x).sum())
            .take(rows)
            .collect())
    }
    fn kernel_launches(&self) -> u64 {
        0
    }
    fn resident_weight_bytes(&self) -> u64 {
        0
    }
    fn failure_reason(&self) -> Option<String> {
        None
    }

    /// Execute an operation and return its owned result. Unsupported operations
    /// must return an error; backend selection never implies a CPU fallback.
    fn execute(&self, op: &KernelOp) -> Result<KernelOutput, String>;

    /// Synchronize the backend (wait for all operations to complete)
    fn synchronize(&self);
}

/// GPU/CPU memory information. A zero total means the measurement is unknown.
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    pub total_memory: usize,
    pub free_memory: usize,
    pub used_memory: usize,
}

impl MemoryInfo {
    pub fn utilization(&self) -> f32 {
        if self.total_memory == 0 {
            0.0
        } else {
            self.used_memory as f32 / self.total_memory as f32
        }
    }
}

/// Owned result of a backend operation.
#[derive(Debug, Clone, PartialEq)]
pub enum KernelOutput {
    F32(Vec<f32>),
    U8(Vec<u8>),
}

/// Kernel operation descriptor
#[derive(Debug, Clone)]
pub enum KernelOp {
    /// Row-major matrix multiplication: A[m,k] @ B[k,n] -> C[m,n]
    MatMul {
        a: Arc<[f32]>,
        b: Arc<[f32]>,
        m: usize,
        n: usize,
        k: usize,
    },
    /// Single-query scaled dot-product attention. Q is one vector of length D;
    /// K/V are token-major [tokens,D]. Mask is an optional additive value per
    /// KV token; negative infinity excludes a token. All-masked output is zero.
    Attention {
        q: Arc<[f32]>,
        k: Arc<[f32]>,
        v: Arc<[f32]>,
        scale: f32,
        mask: Option<Arc<[f32]>>,
    },
    /// Layer normalization
    LayerNorm {
        input: Arc<[f32]>,
        weight: Arc<[f32]>,
        bias: Option<Arc<[f32]>>,
        eps: f32,
    },
    /// RMS normalization (used in modern LLMs)
    RmsNorm {
        input: Arc<[f32]>,
        weight: Arc<[f32]>,
        eps: f32,
    },
    /// Element-wise add
    Add { a: Arc<[f32]>, b: Arc<[f32]> },
    /// Element-wise multiply
    Mul { a: Arc<[f32]>, b: Arc<[f32]> },
    /// Softmax over each row of the last dimension (row width = dim).
    Softmax { input: Arc<[f32]>, dim: usize },
    /// Quantize data
    Quantize {
        input: Arc<[f32]>,
        format: crate::quant::QuantFormat,
    },
    /// Dequantize data
    Dequantize {
        input: Arc<[u8]>,
        format: crate::quant::QuantFormat,
        output_len: usize,
    },
}

struct UnavailableBackend {
    kind: BackendType,
    reason: String,
}
impl Backend for UnavailableBackend {
    fn name(&self) -> &str {
        self.kind.name()
    }
    fn backend_type(&self) -> BackendType {
        self.kind
    }
    fn is_available(&self) -> bool {
        false
    }
    fn memory_info(&self) -> MemoryInfo {
        MemoryInfo {
            total_memory: 0,
            free_memory: 0,
            used_memory: 0,
        }
    }
    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            backend: self.kind,
            available: false,
            features: vec![],
            supported_ops: vec![],
        }
    }
    fn failure_reason(&self) -> Option<String> {
        Some(self.reason.clone())
    }
    fn prepare_weights(
        &self,
        _: &std::collections::HashMap<String, Vec<f32>>,
    ) -> Result<(), String> {
        Err(self.reason.clone())
    }
    fn matvec(&self, _: &[f32], _: usize, _: usize, _: &[f32]) -> Result<Vec<f32>, String> {
        Err(self.reason.clone())
    }
    fn execute(&self, _: &KernelOp) -> Result<KernelOutput, String> {
        Err(self.reason.clone())
    }
    fn synchronize(&self) {}
}

/// Backend factory
pub struct BackendFactory;

impl BackendFactory {
    /// Create a backend based on the requested type or auto-detect
    pub fn create(backend_type: Option<BackendType>) -> Box<dyn Backend> {
        let bt = backend_type.unwrap_or_default();

        match bt {
            BackendType::CpuSimd => Box::new(cpu::CpuBackend::new()),
            BackendType::Metal => Box::new(UnavailableBackend {
                kind: bt,
                reason: "Metal inference kernels are not implemented".into(),
            }),
            BackendType::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    match cuda::CudaBackend::new() {
                        Ok(backend) => Box::new(backend),
                        Err(reason) => Box::new(UnavailableBackend { kind: bt, reason }),
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Box::new(UnavailableBackend {
                        kind: bt,
                        reason: "CUDA feature not compiled; enable --features cuda".into(),
                    })
                }
            }
            BackendType::Vulkan | BackendType::WebGpu => Box::new(UnavailableBackend {
                kind: bt,
                reason: "Inference kernels are not implemented".into(),
            }),
        }
    }

    /// Auto-detect and create the best available backend
    pub fn auto() -> Box<dyn Backend> {
        Self::create(None)
    }

    /// Probe available backends without selecting one.
    pub fn available_backends() -> Vec<BackendCapabilities> {
        vec![
            cpu::CpuBackend::new().capabilities(),
            metal::MetalBackend::new().capabilities(),
            Self::create(Some(BackendType::Cuda)).capabilities(),
            Self::unavailable_capability(BackendType::Vulkan),
            Self::unavailable_capability(BackendType::WebGpu),
        ]
    }

    fn unavailable_capability(backend: BackendType) -> BackendCapabilities {
        BackendCapabilities {
            backend,
            available: false,
            features: Vec::new(),
            supported_ops: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_type_names() {
        assert_eq!(BackendType::CpuSimd.name(), "cpu-simd");
        assert_eq!(BackendType::Metal.name(), "metal");
        assert_eq!(BackendType::Cuda.name(), "cuda");
        assert!(BackendType::WebGpu.is_gpu());
    }

    #[test]
    fn test_backend_detection() {
        let detected = BackendType::detect();
        assert!(matches!(
            detected,
            BackendType::CpuSimd | BackendType::Metal
        ));
    }

    #[test]
    fn test_memory_info() {
        let info = MemoryInfo {
            total_memory: 8 * 1024 * 1024 * 1024,
            free_memory: 6 * 1024 * 1024 * 1024,
            used_memory: 2 * 1024 * 1024 * 1024,
        };
        assert_eq!(info.utilization(), 0.25);
    }

    #[test]
    fn test_backend_capabilities() {
        let capabilities = BackendFactory::available_backends();
        assert!(capabilities
            .iter()
            .any(|cap| cap.backend == BackendType::CpuSimd));
        assert!(capabilities
            .iter()
            .any(|cap| cap.backend == BackendType::Cuda
                && (cfg!(feature = "cuda") || !cap.available)));
        assert!(capabilities
            .iter()
            .any(|cap| cap.supported_ops.contains(&"rms_norm")));
    }
}
