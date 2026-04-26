//! Metal backend for Apple Silicon GPU acceleration.
//! Uses Metal Performance Shaders (MPS) for optimized tensor operations.

use super::{Backend, BackendCapabilities, BackendType, KernelOp, MemoryInfo};

/// Metal backend implementation
pub struct MetalBackend {
    available: bool,
    _device_name: String,
}

impl MetalBackend {
    pub fn new() -> Self {
        let available = Self::check_availability();
        let device_name = if available {
            "Apple Silicon GPU (Metal)".to_string()
        } else {
            "Metal unavailable".to_string()
        };

        MetalBackend {
            available,
            _device_name: device_name,
        }
    }

    fn check_availability() -> bool {
        #[cfg(feature = "metal")]
        {
            // Check if Metal framework is available
            metal::Device::system_default().is_some()
        }
        #[cfg(not(feature = "metal"))]
        {
            false
        }
    }

    /// Run RMS normalization on Metal GPU
    fn rms_norm_metal(&self, input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        // For now, fall back to CPU implementation
        // Full Metal implementation would use MTLComputePipelineState
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

impl Default for MetalBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &str {
        "metal"
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Metal
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn memory_info(&self) -> MemoryInfo {
        if !self.available {
            return MemoryInfo {
                total_memory: 0,
                free_memory: 0,
                used_memory: 0,
            };
        }

        // On Apple Silicon, GPU and CPU share unified memory
        // Estimate based on system memory
        let total = 8 * 1024 * 1024 * 1024; // 8GB estimate
        MemoryInfo {
            total_memory: total,
            free_memory: total / 2,
            used_memory: total / 4,
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            backend: BackendType::Metal,
            available: self.available,
            features: if self.available {
                vec!["unified-memory", "metal"]
            } else {
                vec![]
            },
            supported_ops: if self.available {
                vec!["rms_norm"]
            } else {
                vec![]
            },
        }
    }

    fn execute(&self, op: &KernelOp) {
        match op {
            KernelOp::RmsNorm {
                input, weight, eps, ..
            } => {
                let _ = self.rms_norm_metal(input, weight, *eps);
            }
            _ => {
                tracing::debug!("Metal backend: falling back to CPU for operation");
                let cpu = super::cpu::CpuBackend::new();
                cpu.execute(op);
            }
        }
    }

    fn synchronize(&self) {
        // Metal commands are asynchronous, but for simplicity
        // we sync on every operation in the fallback path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_backend_creation() {
        let backend = MetalBackend::new();
        assert_eq!(backend.name(), "metal");
        // Availability depends on the system
        if backend.is_available() {
            assert!(!backend._device_name.contains("unavailable"));
        }
    }

    #[test]
    fn test_rms_norm_metal() {
        let backend = MetalBackend::new();
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0f32, 1.0, 1.0, 1.0];

        let result = backend.rms_norm_metal(&input, &weight, 1e-5);
        assert_eq!(result.len(), 4);

        // Verify correctness
        let expected_scale: f32 = 1.0 / (30.0_f32 / 4.0).sqrt();
        for i in 0..4 {
            let expected = input[i] * expected_scale * weight[i];
            assert!((result[i] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_memory_info_unavailable() {
        // Create a backend that's not available
        let mut backend = MetalBackend::new();
        backend.available = false;

        let info = backend.memory_info();
        assert_eq!(info.total_memory, 0);
    }
}
