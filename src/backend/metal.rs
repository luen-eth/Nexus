//! Optional Metal device probe. Inference kernels are not implemented.
use super::{Backend, BackendCapabilities, BackendType, KernelOp, KernelOutput, MemoryInfo};

pub struct MetalBackend {
    device_detected: bool,
}
impl MetalBackend {
    pub fn new() -> Self {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        let device_detected = metal::Device::system_default().is_some();
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let device_detected = false;
        Self { device_detected }
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
            backend: BackendType::Metal,
            available: false,
            features: if self.device_detected {
                vec!["device-detected", "inference-not-implemented"]
            } else {
                vec![]
            },
            supported_ops: vec![],
        }
    }
    fn failure_reason(&self) -> Option<String> {
        Some("Metal inference kernels are not implemented".into())
    }
    fn prepare_weights(
        &self,
        _: &std::collections::HashMap<String, Vec<f32>>,
    ) -> Result<(), String> {
        Err(self.failure_reason().unwrap())
    }
    fn matvec(&self, _: &[f32], _: usize, _: usize, _: &[f32]) -> Result<Vec<f32>, String> {
        Err(self.failure_reason().unwrap())
    }
    fn execute(&self, _: &KernelOp) -> Result<KernelOutput, String> {
        Err(self.failure_reason().unwrap())
    }
    fn synchronize(&self) {}
}
