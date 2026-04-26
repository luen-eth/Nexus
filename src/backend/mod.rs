//! Backend abstraction for different hardware targets.
//! Supports CPU SIMD, Metal (Apple Silicon), and CUDA backends.

use std::sync::Arc;

pub mod cpu;
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
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        {
            BackendType::Metal
        }
        #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
        {
            BackendType::CpuSimd
        }
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

    /// Execute a kernel operation
    fn execute(&self, op: &KernelOp);

    /// Synchronize the backend (wait for all operations to complete)
    fn synchronize(&self);
}

/// GPU/CPU memory information
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

/// Kernel operation descriptor
#[derive(Debug, Clone)]
pub enum KernelOp {
    /// Matrix multiplication: C = A @ B^T
    MatMul {
        a: Arc<[f32]>,
        b: Arc<[f32]>,
        m: usize,
        n: usize,
        k: usize,
        output: Arc<Vec<f32>>,
    },
    /// Attention: scaled dot-product attention
    Attention {
        q: Arc<[f32]>,
        k: Arc<[f32]>,
        v: Arc<[f32]>,
        scale: f32,
        mask: Option<Arc<[f32]>>,
        output: Arc<Vec<f32>>,
    },
    /// Layer normalization
    LayerNorm {
        input: Arc<[f32]>,
        weight: Arc<[f32]>,
        bias: Option<Arc<[f32]>>,
        eps: f32,
        output: Arc<Vec<f32>>,
    },
    /// RMS normalization (used in modern LLMs)
    RmsNorm {
        input: Arc<[f32]>,
        weight: Arc<[f32]>,
        eps: f32,
        output: Arc<Vec<f32>>,
    },
    /// Element-wise add
    Add {
        a: Arc<[f32]>,
        b: Arc<[f32]>,
        output: Arc<Vec<f32>>,
    },
    /// Element-wise multiply
    Mul {
        a: Arc<[f32]>,
        b: Arc<[f32]>,
        output: Arc<Vec<f32>>,
    },
    /// Softmax
    Softmax {
        input: Arc<[f32]>,
        dim: usize,
        output: Arc<Vec<f32>>,
    },
    /// Quantize data
    Quantize {
        input: Arc<[f32]>,
        format: crate::quant::QuantFormat,
        output: Arc<Vec<u8>>,
    },
    /// Dequantize data
    Dequantize {
        input: Arc<[u8]>,
        format: crate::quant::QuantFormat,
        output_len: usize,
        output: Arc<Vec<f32>>,
    },
}

/// Backend factory
pub struct BackendFactory;

impl BackendFactory {
    /// Create a backend based on the requested type or auto-detect
    pub fn create(backend_type: Option<BackendType>) -> Box<dyn Backend> {
        let bt = backend_type.unwrap_or_default();

        match bt {
            BackendType::CpuSimd => Box::new(cpu::CpuBackend::new()),
            BackendType::Metal => {
                #[cfg(feature = "metal")]
                {
                    Box::new(metal::MetalBackend::new())
                }
                #[cfg(not(feature = "metal"))]
                {
                    eprintln!("Metal backend not compiled in. Use --features metal");
                    Box::new(cpu::CpuBackend::new())
                }
            }
            BackendType::Cuda | BackendType::Vulkan | BackendType::WebGpu => {
                eprintln!(
                    "{} backend is not implemented in this build; falling back to CPU",
                    bt.name()
                );
                Box::new(cpu::CpuBackend::new())
            }
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
            Self::unavailable_capability(BackendType::Cuda),
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
            .any(|cap| cap.backend == BackendType::Cuda && !cap.available));
        assert!(capabilities
            .iter()
            .any(|cap| cap.supported_ops.contains(&"rms_norm")));
    }
}
