//! Nexus - High-performance AI inference engine
//! Combining TurboQuant KV cache compression, llama.cpp weight quantization,
//! and MLX unified memory architecture.

pub mod backend;
pub mod download;
pub mod engine;
pub mod gguf;
pub mod kernel;
pub mod memory;
pub mod mlx;
pub mod quant;
pub mod scheduler;
pub mod server;
pub mod tokenizer;

pub use engine::InferenceEngine;
pub use gguf::{GgufReader, ModelMetadata};
pub use memory::UnifiedMemory;
pub use quant::{QuantFormat, QuantType};

#[cfg(test)]
mod tests {
    use crate::quant::{QuantFormat, QuantType};

    #[test]
    fn test_quant_format_names() {
        assert_eq!(QuantFormat::Q2_K.name(), "Q2_K");
        assert_eq!(QuantFormat::Q4_0.name(), "Q4_0");
        assert_eq!(QuantFormat::Q4_K.name(), "Q4_K");
        assert_eq!(QuantFormat::Q6_K.name(), "Q6_K");
        assert_eq!(QuantFormat::Q8_0.name(), "Q8_0");
        assert_eq!(QuantFormat::MXFP4.name(), "MXFP4");
    }

    #[test]
    fn test_quant_type_bits() {
        assert_eq!(QuantType::Q2_K.bits(), 2);
        assert_eq!(QuantType::Q4_0.bits(), 4);
        assert_eq!(QuantType::Q4_K.bits(), 4);
        assert_eq!(QuantType::Q6_K.bits(), 6);
        assert_eq!(QuantType::Q8_0.bits(), 8);
        assert_eq!(QuantType::FP16.bits(), 16);
        assert_eq!(QuantType::FP32.bits(), 32);
    }

    #[test]
    fn test_kv_quant_type() {
        use crate::quant::KvQuantType;
        assert_eq!(KvQuantType::None.bits(), 16);
        assert_eq!(KvQuantType::TurboQuant3b2b.key_bits(), 3);
        assert_eq!(KvQuantType::TurboQuant3b2b.val_bits(), 2);
        assert_eq!(KvQuantType::TurboQuant4b4b.key_bits(), 4);
        assert_eq!(KvQuantType::TurboQuant4b4b.val_bits(), 4);
    }
}
