//! GGUF (GPT-Generated Unified Format) parser and writer
//! Supports all GGUF versions and quantization types.

#![allow(non_camel_case_types)]

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

// GGUF magic number and versions
pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"
pub const GGUF_VERSION_1: u32 = 1;
pub const GGUF_VERSION_2: u32 = 2;
pub const GGUF_VERSION_3: u32 = 3;

// GGUF byte order
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GgufByteOrder {
    LittleEndian,
}

impl GgufByteOrder {
    pub fn validate(&self, reader: &mut dyn Read) -> io::Result<()> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != *b"GGUF" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid GGUF magic: {:02x?}", magic),
            ));
        }
        Ok(())
    }

    pub fn read_u32(&self, reader: &mut dyn Read) -> io::Result<u32> {
        let mut buf = [0u8; 4];
        reader.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    pub fn read_u64(&self, reader: &mut dyn Read) -> io::Result<u64> {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    pub fn write_u32(&self, writer: &mut dyn Write, val: u32) -> io::Result<()> {
        writer.write_all(&val.to_le_bytes())
    }

    pub fn write_u64(&self, writer: &mut dyn Write, val: u64) -> io::Result<()> {
        writer.write_all(&val.to_le_bytes())
    }
}

/// GGUF tensor data types (matches llama.cpp ggml_type)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufDataType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    I8,
    I16,
    I32,
    I64,
    F64,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q8_K,
    IQ2_XXS,
    IQ2_XS,
    Q2_S,
    IQ1_S,
    IQ1_N,
    IQ4_NL,
    IQ4_XS,
    IQ3_XXS,
    IQ3_S,
    IQ2_S,
    Q9_K,
    I8x4,
    I8x8,
    F4E2M1,
    F4E2M1x2,
    F8E4M3,
    F8E4M3x2,
    F8E5M2,
    F8E5M2x2,
    BF16,
    Q4_0_4x4,
    Q4_0_8x8,
    TF32,
}

impl GgufDataType {
    pub fn id(&self) -> u32 {
        match self {
            GgufDataType::F32 => 0,
            GgufDataType::F16 => 1,
            GgufDataType::Q4_0 => 2,
            GgufDataType::Q4_1 => 3,
            GgufDataType::Q5_0 => 6,
            GgufDataType::Q5_1 => 7,
            GgufDataType::Q8_0 => 8,
            GgufDataType::Q8_1 => 9,
            GgufDataType::Q2_K => 10,
            GgufDataType::Q3_K => 11,
            GgufDataType::Q4_K => 12,
            GgufDataType::Q5_K => 13,
            GgufDataType::Q6_K => 14,
            GgufDataType::Q8_K => 15,
            GgufDataType::I8 => 16,
            GgufDataType::I16 => 17,
            GgufDataType::I32 => 18,
            GgufDataType::I64 => 19,
            GgufDataType::F64 => 20,
            GgufDataType::IQ2_XXS => 21,
            GgufDataType::IQ2_XS => 22,
            GgufDataType::Q2_S => 23,
            GgufDataType::IQ3_XXS => 24,
            GgufDataType::IQ1_S => 25,
            GgufDataType::IQ4_NL => 26,
            GgufDataType::IQ3_S => 27,
            GgufDataType::BF16 => 28,
            GgufDataType::IQ4_XS => 30,
            GgufDataType::IQ1_N => 31,
            GgufDataType::IQ2_S => 34,
            GgufDataType::Q4_0_4x4 => 36,
            GgufDataType::Q4_0_8x8 => 37,
            GgufDataType::Q9_K => 38,
            GgufDataType::I8x4 => 39,
            GgufDataType::I8x8 => 40,
            GgufDataType::F4E2M1 => 41,
            GgufDataType::F4E2M1x2 => 42,
            GgufDataType::F8E4M3 => 43,
            GgufDataType::F8E4M3x2 => 44,
            GgufDataType::F8E5M2 => 45,
            GgufDataType::F8E5M2x2 => 46,
            GgufDataType::TF32 => 47,
        }
    }

    pub fn from_id(id: u32) -> Option<GgufDataType> {
        match id {
            0 => Some(GgufDataType::F32),
            1 => Some(GgufDataType::F16),
            2 => Some(GgufDataType::Q4_0),
            3 => Some(GgufDataType::Q4_1),
            6 => Some(GgufDataType::Q5_0),
            7 => Some(GgufDataType::Q5_1),
            8 => Some(GgufDataType::Q8_0),
            9 => Some(GgufDataType::Q8_1),
            10 => Some(GgufDataType::Q2_K),
            11 => Some(GgufDataType::Q3_K),
            12 => Some(GgufDataType::Q4_K),
            13 => Some(GgufDataType::Q5_K),
            14 => Some(GgufDataType::Q6_K),
            15 => Some(GgufDataType::Q8_K),
            16 => Some(GgufDataType::I8),
            17 => Some(GgufDataType::I16),
            18 => Some(GgufDataType::I32),
            19 => Some(GgufDataType::I64),
            20 => Some(GgufDataType::F64),
            21 => Some(GgufDataType::IQ2_XXS),
            22 => Some(GgufDataType::IQ2_XS),
            23 => Some(GgufDataType::Q2_S),
            24 => Some(GgufDataType::IQ3_XXS),
            25 => Some(GgufDataType::IQ1_S),
            26 => Some(GgufDataType::IQ4_NL),
            27 => Some(GgufDataType::IQ3_S),
            28 => Some(GgufDataType::BF16),
            30 => Some(GgufDataType::IQ4_XS),
            31 => Some(GgufDataType::IQ1_N),
            34 => Some(GgufDataType::IQ2_S),
            36 => Some(GgufDataType::Q4_0_4x4),
            37 => Some(GgufDataType::Q4_0_8x8),
            38 => Some(GgufDataType::Q9_K),
            39 => Some(GgufDataType::I8x4),
            40 => Some(GgufDataType::I8x8),
            41 => Some(GgufDataType::F4E2M1),
            42 => Some(GgufDataType::F4E2M1x2),
            43 => Some(GgufDataType::F8E4M3),
            44 => Some(GgufDataType::F8E4M3x2),
            45 => Some(GgufDataType::F8E5M2),
            46 => Some(GgufDataType::F8E5M2x2),
            47 => Some(GgufDataType::TF32),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            GgufDataType::F32 => "F32",
            GgufDataType::F16 => "F16",
            GgufDataType::Q4_0 => "Q4_0",
            GgufDataType::Q4_1 => "Q4_1",
            GgufDataType::Q5_0 => "Q5_0",
            GgufDataType::Q5_1 => "Q5_1",
            GgufDataType::Q8_0 => "Q8_0",
            GgufDataType::Q8_1 => "Q8_1",
            GgufDataType::I8 => "I8",
            GgufDataType::I16 => "I16",
            GgufDataType::I32 => "I32",
            GgufDataType::I64 => "I64",
            GgufDataType::F64 => "F64",
            GgufDataType::Q9_K => "Q9_K",
            GgufDataType::Q2_K => "Q2_K",
            GgufDataType::Q3_K => "Q3_K",
            GgufDataType::Q4_K => "Q4_K",
            GgufDataType::Q5_K => "Q5_K",
            GgufDataType::Q6_K => "Q6_K",
            GgufDataType::IQ2_XXS => "IQ2_XXS",
            GgufDataType::IQ2_XS => "IQ2_XS",
            GgufDataType::Q2_S => "Q2_S",
            GgufDataType::I8x4 => "I8x4",
            GgufDataType::I8x8 => "I8x8",
            GgufDataType::F4E2M1 => "F4E2M1",
            GgufDataType::BF16 => "BF16",
            GgufDataType::Q4_0_4x4 => "Q4_0_4x4",
            GgufDataType::Q4_0_8x8 => "Q4_0_8x8",
            GgufDataType::TF32 => "TF32",
            GgufDataType::Q8_K => "Q8_K",
            GgufDataType::IQ1_S => "IQ1_S",
            GgufDataType::IQ1_N => "IQ1_N",
            GgufDataType::IQ3_XXS => "IQ3_XXS",
            GgufDataType::IQ3_S => "IQ3_S",
            GgufDataType::IQ2_S => "IQ2_S",
            GgufDataType::IQ4_NL => "IQ4_NL",
            GgufDataType::IQ4_XS => "IQ4_XS",
            GgufDataType::F8E4M3 => "F8E4M3",
            GgufDataType::F8E4M3x2 => "F8E4M3x2",
            GgufDataType::F8E5M2 => "F8E5M2",
            GgufDataType::F8E5M2x2 => "F8E5M2x2",
            GgufDataType::F4E2M1x2 => "F4E2M1x2",
        }
    }

    /// Returns bytes per element for this data type (approximate for quantized types)
    pub fn bytes_per_element(&self, elements: usize) -> usize {
        fn blocks(elements: usize, block: usize, type_size: usize) -> usize {
            elements.div_ceil(block) * type_size
        }

        match self {
            GgufDataType::F32 | GgufDataType::TF32 => elements * 4,
            GgufDataType::F16 | GgufDataType::BF16 => elements * 2,
            GgufDataType::I8 => elements,
            GgufDataType::I16 => elements * 2,
            GgufDataType::I32 => elements * 4,
            GgufDataType::I64 => elements * 8,
            GgufDataType::F64 => elements * 8,
            GgufDataType::Q4_0 | GgufDataType::Q4_0_4x4 | GgufDataType::Q4_0_8x8 => {
                blocks(elements, 32, 18)
            }
            GgufDataType::Q4_1 => blocks(elements, 32, 20),
            GgufDataType::Q5_0 => blocks(elements, 32, 22),
            GgufDataType::Q5_1 => blocks(elements, 32, 24),
            GgufDataType::Q8_0 => blocks(elements, 32, 34),
            GgufDataType::Q8_1 => blocks(elements, 32, 36),
            GgufDataType::Q2_K => blocks(elements, 256, 84),
            GgufDataType::Q3_K => blocks(elements, 256, 110),
            GgufDataType::Q4_K => blocks(elements, 256, 144),
            GgufDataType::Q5_K => blocks(elements, 256, 176),
            GgufDataType::Q6_K => blocks(elements, 256, 210),
            GgufDataType::Q8_K => blocks(elements, 256, 292),
            GgufDataType::Q9_K => blocks(elements, 256, 324),
            GgufDataType::IQ1_S | GgufDataType::IQ1_N => blocks(elements, 256, 32),
            GgufDataType::IQ2_XXS
            | GgufDataType::IQ2_XS
            | GgufDataType::IQ2_S
            | GgufDataType::Q2_S => blocks(elements, 256, 64),
            GgufDataType::IQ3_XXS | GgufDataType::IQ3_S => blocks(elements, 256, 96),
            GgufDataType::IQ4_NL
            | GgufDataType::IQ4_XS
            | GgufDataType::F4E2M1
            | GgufDataType::F4E2M1x2 => elements.div_ceil(2),
            GgufDataType::F8E4M3
            | GgufDataType::F8E4M3x2
            | GgufDataType::F8E5M2
            | GgufDataType::F8E5M2x2 => elements,
            GgufDataType::I8x4 | GgufDataType::I8x8 => elements,
        }
    }

    pub fn elem_size(&self) -> usize {
        match self {
            GgufDataType::F32 | GgufDataType::TF32 => 4,
            GgufDataType::F16 | GgufDataType::BF16 => 2,
            _ => 0, // quantized types use block-based sizing
        }
    }
}

impl fmt::Display for GgufDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// GGUF metadata value types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufValueType {
    Bool,
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
    String,
    Array,
}

impl GgufValueType {
    pub fn from_id(id: u32) -> Option<GgufValueType> {
        match id {
            0 => Some(GgufValueType::U8),
            1 => Some(GgufValueType::I8),
            2 => Some(GgufValueType::U16),
            3 => Some(GgufValueType::I16),
            4 => Some(GgufValueType::U32),
            5 => Some(GgufValueType::I32),
            6 => Some(GgufValueType::F32),
            7 => Some(GgufValueType::Bool),
            8 => Some(GgufValueType::String),
            9 => Some(GgufValueType::Array),
            10 => Some(GgufValueType::U64),
            11 => Some(GgufValueType::I64),
            12 => Some(GgufValueType::F64),
            _ => None,
        }
    }

    pub fn id(&self) -> u32 {
        match self {
            GgufValueType::U8 => 0,
            GgufValueType::I8 => 1,
            GgufValueType::U16 => 2,
            GgufValueType::I16 => 3,
            GgufValueType::U32 => 4,
            GgufValueType::I32 => 5,
            GgufValueType::F32 => 6,
            GgufValueType::Bool => 7,
            GgufValueType::String => 8,
            GgufValueType::Array => 9,
            GgufValueType::U64 => 10,
            GgufValueType::I64 => 11,
            GgufValueType::F64 => 12,
        }
    }
}

/// GGUF metadata value (holds any supported type)
#[derive(Debug, Clone)]
pub enum GgufValue {
    Bool(bool),
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    String(String),
    U8Vec(Vec<u8>),
    I32Vec(Vec<i32>),
    U64Vec(Vec<u64>),
    F32Vec(Vec<f32>),
    StringVec(Vec<String>),
}

impl GgufValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            GgufValue::F32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GgufValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<i32> {
        match self {
            GgufValue::I32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<u32> {
        match self {
            GgufValue::U32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            GgufValue::I64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_string_vec(&self) -> Option<&Vec<String>> {
        match self {
            GgufValue::StringVec(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_i32_vec(&self) -> Option<&Vec<i32>> {
        match self {
            GgufValue::I32Vec(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f32_vec(&self) -> Option<&Vec<f32>> {
        match self {
            GgufValue::F32Vec(v) => Some(v),
            _ => None,
        }
    }
}

/// GGUF tensor info
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub shape: Vec<u64>,
    pub data_type: GgufDataType,
    pub offset: u64,
    pub stride: Vec<usize>,
}

impl TensorInfo {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().map(|s| *s as usize).product()
    }

    pub fn data_size(&self) -> usize {
        self.data_type.bytes_per_element(self.num_elements())
    }
}

/// Model metadata extracted from GGUF
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub byte_order: GgufByteOrder,
    pub alignment: usize,
    pub tensor_data_offset: u64,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorInfo>,
}

impl ModelMetadata {
    /// Get a string metadata value by key
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }

    /// Get an f32 metadata value by key
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(|v| v.as_f32())
    }

    /// Get an i32 metadata value by key
    pub fn get_i32(&self, key: &str) -> Option<i32> {
        self.metadata.get(key).and_then(|v| v.as_i32())
    }

    /// Get an unsigned metadata value as usize.
    pub fn get_usize(&self, key: &str) -> Option<usize> {
        self.metadata.get(key).and_then(|v| {
            v.as_u64()
                .map(|v| v as usize)
                .or_else(|| v.as_u32().map(|v| v as usize))
                .or_else(|| v.as_i64().and_then(|v| usize::try_from(v).ok()))
                .or_else(|| v.as_i32().and_then(|v| usize::try_from(v).ok()))
        })
    }

    /// Get model architecture
    pub fn architecture(&self) -> Option<&str> {
        self.get_str("general.architecture")
    }

    /// Get model name
    pub fn name(&self) -> Option<&str> {
        self.get_str("general.name")
            .or_else(|| self.get_str("llama.vocab_size"))
    }

    /// Get context length
    pub fn context_length(&self) -> Option<usize> {
        self.get_usize("llama.context_length")
    }

    /// Get number of attention heads
    pub fn num_attention_heads(&self) -> Option<usize> {
        self.get_usize("llama.attention.head_count")
    }

    /// Get number of key heads (for GQA)
    pub fn num_key_value_heads(&self) -> Option<usize> {
        self.get_usize("llama.attention.head_count_kv")
    }

    /// Get number of layers
    pub fn num_layers(&self) -> Option<usize> {
        self.get_usize("llama.block_count")
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> Option<usize> {
        self.get_usize("llama.embedding_length")
    }

    /// Get feed forward size
    pub fn feed_forward_size(&self) -> Option<usize> {
        self.get_usize("llama.feed_forward_length")
    }

    /// Get rope dimensionality
    pub fn rope_dim(&self) -> Option<usize> {
        self.get_usize("llama.rope.dimension_count")
    }

    /// Get rope frequency base
    pub fn rope_freq_base(&self) -> f32 {
        self.get_f32("llama.rope.freq_base").unwrap_or(1000000.0)
    }

    /// Get quantization format of weights
    pub fn quant_format(&self) -> Option<&str> {
        self.get_str("general.quantization_version")
            .or_else(|| self.get_str("llama.quantization_version"))
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> Option<usize> {
        self.get_usize("llama.vocab_size")
    }

    /// Get tensor by name
    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Absolute byte offset of a tensor's data in the GGUF file.
    pub fn absolute_tensor_offset(&self, tensor: &TensorInfo) -> u64 {
        self.tensor_data_offset + tensor.offset
    }
}

/// GGUF reader for parsing model files
pub struct GgufReader {
    byte_order: GgufByteOrder,
    version: u32,
    metadata: HashMap<String, GgufValue>,
    tensors: Vec<TensorInfo>,
    tensor_count: u64,
    kv_count: u64,
}

impl GgufReader {
    pub fn new() -> Self {
        GgufReader {
            byte_order: GgufByteOrder::LittleEndian,
            version: 0,
            metadata: HashMap::new(),
            tensors: Vec::new(),
            tensor_count: 0,
            kv_count: 0,
        }
    }

    /// Read GGUF header from a reader
    pub fn read_header(&mut self, reader: &mut dyn Read) -> io::Result<()> {
        // Validate magic
        self.byte_order.validate(reader)?;

        // Read version
        self.version = self.byte_order.read_u32(reader)?;

        if self.version != GGUF_VERSION_2 && self.version != GGUF_VERSION_3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Unsupported GGUF version: {}. Only v2 and v3 supported.",
                    self.version
                ),
            ));
        }

        // Read tensor count (v3) or tensor count (v2)
        self.tensor_count = self.byte_order.read_u64(reader)?;

        // Read KV count
        self.kv_count = self.byte_order.read_u64(reader)?;

        Ok(())
    }

    /// Read metadata key-value pairs
    pub fn read_metadata(&mut self, reader: &mut dyn Read) -> io::Result<()> {
        for _ in 0..self.kv_count {
            let key_len = self.byte_order.read_u64(reader)?;
            let mut key_bytes = vec![0u8; key_len as usize];
            reader.read_exact(&mut key_bytes)?;
            let key = String::from_utf8(key_bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            let type_id = self.byte_order.read_u32(reader)?;
            let value_type = GgufValueType::from_id(type_id).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Unknown value type: {}", type_id),
                )
            })?;

            let value = Self::read_value(reader, value_type, &self.byte_order)?;
            self.metadata.insert(key, value);
        }

        Ok(())
    }

    /// Read tensor info
    pub fn read_tensors(&mut self, reader: &mut dyn Read) -> io::Result<()> {
        for _ in 0..self.tensor_count {
            // Read name length and name
            let name_len = self.byte_order.read_u64(reader)?;
            let mut name_bytes = vec![0u8; name_len as usize];
            reader.read_exact(&mut name_bytes)?;
            let name = String::from_utf8(name_bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            // Read number of dimensions
            let n_dims = self.byte_order.read_u32(reader)?;

            // Read shape
            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                let dim = self.byte_order.read_u64(reader)?;
                shape.push(dim);
            }

            // Read data type
            let data_type_id = self.byte_order.read_u32(reader)?;
            let data_type = GgufDataType::from_id(data_type_id).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Unknown data type: {}", data_type_id),
                )
            })?;

            let tensor_offset = self.byte_order.read_u64(reader)?;

            self.tensors.push(TensorInfo {
                name,
                n_dims,
                shape,
                data_type,
                offset: tensor_offset,
                stride: vec![1; n_dims as usize],
            });
        }

        Ok(())
    }

    /// Read a GGUF value based on its type
    fn read_value(
        reader: &mut dyn Read,
        value_type: GgufValueType,
        bo: &GgufByteOrder,
    ) -> io::Result<GgufValue> {
        match value_type {
            GgufValueType::Bool => {
                let mut byte = [0u8; 1];
                reader.read_exact(&mut byte)?;
                Ok(GgufValue::Bool(byte[0] != 0))
            }
            GgufValueType::U8 => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::U8(buf[0]))
            }
            GgufValueType::I8 => {
                let mut buf = [0u8; 1];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::I8(buf[0] as i8))
            }
            GgufValueType::U16 => Ok(GgufValue::U16(bo.read_u32(reader)? as u16)),
            GgufValueType::I16 => Ok(GgufValue::I16(bo.read_u32(reader)? as i16)),
            GgufValueType::U32 => Ok(GgufValue::U32(bo.read_u32(reader)?)),
            GgufValueType::I32 => Ok(GgufValue::I32(bo.read_u32(reader)? as i32)),
            GgufValueType::F32 => {
                let mut buf = [0u8; 4];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::F32(f32::from_le_bytes(buf)))
            }
            GgufValueType::U64 => Ok(GgufValue::U64(bo.read_u64(reader)?)),
            GgufValueType::I64 => Ok(GgufValue::I64(bo.read_u64(reader)? as i64)),
            GgufValueType::F64 => {
                let mut buf = [0u8; 8];
                reader.read_exact(&mut buf)?;
                Ok(GgufValue::F64(f64::from_le_bytes(buf)))
            }
            GgufValueType::String => {
                let str_len = bo.read_u64(reader)?;
                let mut str_bytes = vec![0u8; str_len as usize];
                reader.read_exact(&mut str_bytes)?;
                let s = String::from_utf8(str_bytes)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                Ok(GgufValue::String(s))
            }
            GgufValueType::Array => {
                let arr_type_id = bo.read_u32(reader)?;
                let arr_type = GgufValueType::from_id(arr_type_id).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "Unknown array element type")
                })?;
                let arr_len = bo.read_u64(reader)?;

                match arr_type {
                    GgufValueType::Bool => {
                        let mut vec = Vec::with_capacity(arr_len as usize);
                        for _ in 0..arr_len {
                            let mut b = [0u8; 1];
                            reader.read_exact(&mut b)?;
                            vec.push(b[0] != 0);
                        }
                        Ok(GgufValue::U8Vec(vec.iter().map(|&b| b as u8).collect()))
                    }
                    GgufValueType::I32 => {
                        let mut vec = Vec::with_capacity(arr_len as usize);
                        for _ in 0..arr_len {
                            let val = bo.read_u32(reader)?;
                            vec.push(val as i32);
                        }
                        Ok(GgufValue::I32Vec(vec))
                    }
                    GgufValueType::U64 => {
                        let mut vec = Vec::with_capacity(arr_len as usize);
                        for _ in 0..arr_len {
                            vec.push(bo.read_u64(reader)?);
                        }
                        Ok(GgufValue::U64Vec(vec))
                    }
                    GgufValueType::F32 => {
                        let mut vec = Vec::with_capacity(arr_len as usize);
                        for _ in 0..arr_len {
                            let mut buf = [0u8; 4];
                            reader.read_exact(&mut buf)?;
                            vec.push(f32::from_le_bytes(buf));
                        }
                        Ok(GgufValue::F32Vec(vec))
                    }
                    GgufValueType::String => {
                        let mut vec = Vec::with_capacity(arr_len as usize);
                        for _ in 0..arr_len {
                            let str_len = bo.read_u64(reader)?;
                            let mut str_bytes = vec![0u8; str_len as usize];
                            reader.read_exact(&mut str_bytes)?;
                            let s = String::from_utf8(str_bytes)
                                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                            vec.push(s);
                        }
                        Ok(GgufValue::StringVec(vec))
                    }
                    _ => Err(io::Error::new(
                        io::ErrorKind::Unsupported,
                        "Array type not fully supported",
                    )),
                }
            }
        }
    }

    /// Full parse: reads header, metadata, and tensor info
    pub fn parse(&mut self, reader: &mut dyn Read) -> io::Result<ModelMetadata> {
        self.read_header(reader)?;
        self.read_metadata(reader)?;
        self.read_tensors(reader)?;

        Ok(ModelMetadata {
            version: self.version,
            tensor_count: self.tensor_count,
            kv_count: self.kv_count,
            byte_order: self.byte_order,
            alignment: self
                .metadata
                .get("general.alignment")
                .and_then(|v| {
                    v.as_u32()
                        .map(|v| v as usize)
                        .or_else(|| v.as_u64().map(|v| v as usize))
                })
                .unwrap_or(32),
            tensor_data_offset: 0,
            metadata: self.metadata.clone(),
            tensors: self.tensors.clone(),
        })
    }

    /// Get tensor data offset from file
    pub fn tensor_data_offset(&self) -> u64 {
        // The offset field in TensorInfo already accounts for alignment
        if let Some(last_tensor) = self.tensors.last() {
            last_tensor.offset + last_tensor.data_size() as u64
        } else {
            0
        }
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub fn metadata(&self) -> &HashMap<String, GgufValue> {
        &self.metadata
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Parse a GGUF file from a path using memory mapping
    pub fn parse_file<P: AsRef<Path>>(path: P) -> io::Result<ModelMetadata> {
        let file = File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(io::Error::other)?;

        let mut reader = GgufReader::new();
        reader.parse_from_bytes(&mmap)
    }

    /// Read raw tensor bytes from a GGUF file using parsed metadata.
    pub fn read_tensor_bytes<P: AsRef<Path>>(
        path: P,
        metadata: &ModelMetadata,
        tensor: &TensorInfo,
    ) -> io::Result<Vec<u8>> {
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(metadata.absolute_tensor_offset(tensor)))?;

        let mut data = vec![0u8; tensor.data_size()];
        file.read_exact(&mut data)?;
        Ok(data)
    }

    /// Read a tensor and dequantize supported GGUF types to f32.
    pub fn read_tensor_f32<P: AsRef<Path>>(
        path: P,
        metadata: &ModelMetadata,
        tensor: &TensorInfo,
    ) -> io::Result<Vec<f32>> {
        let data = Self::read_tensor_bytes(path, metadata, tensor)?;
        Self::tensor_bytes_to_f32(tensor.data_type, tensor.num_elements(), &data)
    }

    /// Convert supported GGUF tensor payloads to f32.
    pub fn tensor_bytes_to_f32(
        data_type: GgufDataType,
        output_len: usize,
        data: &[u8],
    ) -> io::Result<Vec<f32>> {
        match data_type {
            GgufDataType::F32 => {
                if data.len() < output_len * 4 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "F32 tensor payload is truncated",
                    ));
                }
                Ok(data
                    .chunks_exact(4)
                    .take(output_len)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect())
            }
            GgufDataType::F16 => {
                if data.len() < output_len * 2 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "F16 tensor payload is truncated",
                    ));
                }
                Ok(data
                    .chunks_exact(2)
                    .take(output_len)
                    .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                    .collect())
            }
            GgufDataType::BF16 => {
                if data.len() < output_len * 2 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "BF16 tensor payload is truncated",
                    ));
                }
                Ok(data
                    .chunks_exact(2)
                    .take(output_len)
                    .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                    .collect())
            }
            GgufDataType::Q8_0 => Self::dequantize_q8_0(data, output_len),
            GgufDataType::Q4_0 => Self::dequantize_q4_0(data, output_len),
            GgufDataType::Q4_K => Self::dequantize_q4_k(data, output_len),
            GgufDataType::Q2_K => Self::dequantize_q2_k(data, output_len),
            other => Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("Dequantization for {} is not implemented", other),
            )),
        }
    }

    fn decode_f16_le(bytes: &[u8]) -> f32 {
        half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32()
    }

    fn dequantize_q8_0(data: &[u8], output_len: usize) -> io::Result<Vec<f32>> {
        const QK: usize = 32;
        const BLOCK: usize = 2 + QK;
        let needed = output_len.div_ceil(QK) * BLOCK;
        if data.len() < needed {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Q8_0 tensor payload is truncated",
            ));
        }

        let mut output = Vec::with_capacity(output_len);
        for block in data.chunks_exact(BLOCK) {
            let d = Self::decode_f16_le(&block[..2]);
            for &q in &block[2..] {
                output.push((q as i8) as f32 * d);
                if output.len() == output_len {
                    return Ok(output);
                }
            }
        }
        Ok(output)
    }

    fn dequantize_q4_0(data: &[u8], output_len: usize) -> io::Result<Vec<f32>> {
        const QK: usize = 32;
        const QB: usize = QK / 2;
        const BLOCK: usize = 2 + QB;
        let needed = output_len.div_ceil(QK) * BLOCK;
        if data.len() < needed {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Q4_0 tensor payload is truncated",
            ));
        }

        let mut output = Vec::with_capacity(output_len);
        for block in data.chunks_exact(BLOCK) {
            let d = Self::decode_f16_le(&block[..2]);
            let qs = &block[2..];

            for &packed in qs {
                let q = (packed & 0x0f) as i8 - 8;
                output.push(q as f32 * d);
                if output.len() == output_len {
                    return Ok(output);
                }
            }
            for &packed in qs {
                let q = (packed >> 4) as i8 - 8;
                output.push(q as f32 * d);
                if output.len() == output_len {
                    return Ok(output);
                }
            }
        }
        Ok(output)
    }

    fn dequantize_q4_k(data: &[u8], output_len: usize) -> io::Result<Vec<f32>> {
        const QK: usize = 256;
        const BLOCK: usize = 144;
        let needed = output_len.div_ceil(QK) * BLOCK;
        if data.len() < needed {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Q4_K tensor payload is truncated",
            ));
        }

        let mut output = Vec::with_capacity(output_len);
        for block in data.chunks_exact(BLOCK) {
            let d = Self::decode_f16_le(&block[..2]);
            let dmin = Self::decode_f16_le(&block[2..4]);
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
                output.push(d * scales[sub] as f32 * q as f32 - dmin * mins[sub] as f32);
                if output.len() == output_len {
                    return Ok(output);
                }
            }
        }
        Ok(output)
    }

    fn dequantize_q2_k(data: &[u8], output_len: usize) -> io::Result<Vec<f32>> {
        const QK: usize = 256;
        const BLOCK: usize = 84;
        let needed = output_len.div_ceil(QK) * BLOCK;
        if data.len() < needed {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Q2_K tensor payload is truncated",
            ));
        }

        let mut output = Vec::with_capacity(output_len);
        for block in data.chunks_exact(BLOCK) {
            let scales = &block[..16];
            let qs = &block[16..80];
            let d = Self::decode_f16_le(&block[80..82]);
            let dmin = Self::decode_f16_le(&block[82..84]);

            for i in 0..QK {
                let q = (qs[i / 4] >> ((i % 4) * 2)) & 0x03;
                let packed = scales[i / 16];
                let scale = (packed & 0x0f) as f32;
                let min = (packed >> 4) as f32;
                output.push(d * scale * q as f32 - dmin * min);
                if output.len() == output_len {
                    return Ok(output);
                }
            }
        }
        Ok(output)
    }

    /// Parse GGUF from a byte slice (memory-mapped data)
    fn parse_from_bytes(&mut self, data: &[u8]) -> io::Result<ModelMetadata> {
        let mut pos = 0usize;

        // Validate magic
        if data.len() < 8 || &data[0..4] != b"GGUF" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Invalid GGUF magic",
            ));
        }
        pos += 4;

        // Read version
        if data.len() < pos + 4 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Unexpected EOF reading version",
            ));
        }
        let version = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
        pos += 4;

        if version != GGUF_VERSION_2 && version != GGUF_VERSION_3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported GGUF version: {}", version),
            ));
        }

        // Read tensor count
        if data.len() < pos + 8 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Unexpected EOF reading tensor count",
            ));
        }
        let tensor_count = u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]);
        pos += 8;

        // Read KV count
        if data.len() < pos + 8 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Unexpected EOF reading kv count",
            ));
        }
        let kv_count = u64::from_le_bytes([
            data[pos],
            data[pos + 1],
            data[pos + 2],
            data[pos + 3],
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]);
        pos += 8;

        self.tensor_count = tensor_count;
        self.kv_count = kv_count;
        self.version = version;

        // Read metadata
        for _ in 0..kv_count {
            if pos + 8 > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Unexpected EOF reading metadata key length",
                ));
            }
            // GGUF: key length is always u64 (v2 and v3 both use u64 for keys)
            let name_len: usize;
            if version == GGUF_VERSION_3 {
                // v3 uses u32 for tensor names but u64 for KV keys
                name_len = u64::from_le_bytes([
                    data[pos],
                    data[pos + 1],
                    data[pos + 2],
                    data[pos + 3],
                    data[pos + 4],
                    data[pos + 5],
                    data[pos + 6],
                    data[pos + 7],
                ]) as usize;
                pos += 8;
            } else {
                name_len = u64::from_le_bytes([
                    data[pos],
                    data[pos + 1],
                    data[pos + 2],
                    data[pos + 3],
                    data[pos + 4],
                    data[pos + 5],
                    data[pos + 6],
                    data[pos + 7],
                ]) as usize;
                pos += 8;
            };

            if pos + name_len > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Key string exceeds file bounds",
                ));
            }
            let key = std::str::from_utf8(&data[pos..pos + name_len])
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            pos += name_len;

            if pos + 4 > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Unexpected EOF reading value type",
                ));
            }
            let type_id =
                u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            pos += 4;
            let value_type = GgufValueType::from_id(type_id).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Unknown value type: {}", type_id),
                )
            })?;

            let (value, read_len) = Self::read_value_from_bytes(data, pos, value_type)?;
            pos += read_len;
            self.metadata.insert(key.to_string(), value);
        }

        // Read tensor info
        for _ in 0..tensor_count {
            if pos + 8 > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Unexpected EOF reading tensor name length",
                ));
            }
            let name_len = u64::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]) as usize;
            pos += 8;

            if pos + name_len > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Tensor name exceeds file bounds",
                ));
            }
            let name = std::str::from_utf8(&data[pos..pos + name_len])
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            pos += name_len;

            if pos + 4 > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Unexpected EOF reading dims",
                ));
            }
            let n_dims =
                u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            pos += 4;

            let mut shape = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                if pos + 8 > data.len() {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "Unexpected EOF reading shape",
                    ));
                }
                let dim = u64::from_le_bytes([
                    data[pos],
                    data[pos + 1],
                    data[pos + 2],
                    data[pos + 3],
                    data[pos + 4],
                    data[pos + 5],
                    data[pos + 6],
                    data[pos + 7],
                ]);
                pos += 8;
                shape.push(dim);
            }

            if pos + 4 > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Unexpected EOF reading data type",
                ));
            }
            let data_type_id =
                u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            pos += 4;
            let data_type = GgufDataType::from_id(data_type_id).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Unknown data type: {}", data_type_id),
                )
            })?;

            if pos + 8 > data.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Unexpected EOF reading tensor offset",
                ));
            }
            let tensor_offset = u64::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
                data[pos + 4],
                data[pos + 5],
                data[pos + 6],
                data[pos + 7],
            ]);
            pos += 8;

            self.tensors.push(TensorInfo {
                name: name.to_string(),
                n_dims,
                shape,
                data_type,
                offset: tensor_offset,
                stride: vec![1; n_dims as usize],
            });
        }

        let alignment = self
            .metadata
            .get("general.alignment")
            .and_then(|v| {
                v.as_u32()
                    .map(|v| v as usize)
                    .or_else(|| v.as_u64().map(|v| v as usize))
            })
            .unwrap_or(32)
            .max(1);
        let tensor_data_offset = pos.div_ceil(alignment) * alignment;

        Ok(ModelMetadata {
            version,
            tensor_count: self.tensor_count,
            kv_count: self.kv_count,
            byte_order: GgufByteOrder::LittleEndian,
            alignment,
            tensor_data_offset: tensor_data_offset as u64,
            metadata: self.metadata.clone(),
            tensors: self.tensors.clone(),
        })
    }

    fn read_value_from_bytes(
        data: &[u8],
        pos: usize,
        value_type: GgufValueType,
    ) -> io::Result<(GgufValue, usize)> {
        let mut pos = pos;
        match value_type {
            GgufValueType::Bool => {
                let val = data[pos] != 0;
                Ok((GgufValue::Bool(val), 1))
            }
            GgufValueType::U8 => {
                let val = data[pos];
                Ok((GgufValue::U8(val), 1))
            }
            GgufValueType::I8 => {
                let val = data[pos] as i8;
                Ok((GgufValue::I8(val), 1))
            }
            GgufValueType::U16 => {
                let val = u16::from_le_bytes([data[pos], data[pos + 1]]);
                Ok((GgufValue::U16(val), 2))
            }
            GgufValueType::I16 => {
                let val = i16::from_le_bytes([data[pos], data[pos + 1]]);
                Ok((GgufValue::I16(val), 2))
            }
            GgufValueType::U32 => {
                let val =
                    u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
                Ok((GgufValue::U32(val), 4))
            }
            GgufValueType::I32 => {
                let val =
                    i32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
                Ok((GgufValue::I32(val), 4))
            }
            GgufValueType::F32 => {
                let val =
                    f32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
                Ok((GgufValue::F32(val), 4))
            }
            GgufValueType::U64 => {
                let val = u64::from_le_bytes([
                    data[pos],
                    data[pos + 1],
                    data[pos + 2],
                    data[pos + 3],
                    data[pos + 4],
                    data[pos + 5],
                    data[pos + 6],
                    data[pos + 7],
                ]);
                Ok((GgufValue::U64(val), 8))
            }
            GgufValueType::I64 => {
                let val = i64::from_le_bytes([
                    data[pos],
                    data[pos + 1],
                    data[pos + 2],
                    data[pos + 3],
                    data[pos + 4],
                    data[pos + 5],
                    data[pos + 6],
                    data[pos + 7],
                ]);
                Ok((GgufValue::I64(val), 8))
            }
            GgufValueType::F64 => {
                let val = f64::from_le_bytes([
                    data[pos],
                    data[pos + 1],
                    data[pos + 2],
                    data[pos + 3],
                    data[pos + 4],
                    data[pos + 5],
                    data[pos + 6],
                    data[pos + 7],
                ]);
                Ok((GgufValue::F64(val), 8))
            }
            GgufValueType::String => {
                let str_len = u64::from_le_bytes([
                    data[pos],
                    data[pos + 1],
                    data[pos + 2],
                    data[pos + 3],
                    data[pos + 4],
                    data[pos + 5],
                    data[pos + 6],
                    data[pos + 7],
                ]);
                pos += 8;
                let s = std::str::from_utf8(&data[pos..pos + str_len as usize])
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                Ok((GgufValue::String(s.to_string()), 8 + str_len as usize))
            }
            GgufValueType::Array => {
                let arr_type_id =
                    u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
                pos += 4;
                let arr_type = GgufValueType::from_id(arr_type_id).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "Unknown array type")
                })?;
                let arr_len = u64::from_le_bytes([
                    data[pos],
                    data[pos + 1],
                    data[pos + 2],
                    data[pos + 3],
                    data[pos + 4],
                    data[pos + 5],
                    data[pos + 6],
                    data[pos + 7],
                ]);
                pos += 8;

                let mut read = 12usize;
                match arr_type {
                    GgufValueType::I32 => {
                        let mut vec = Vec::with_capacity(arr_len as usize);
                        for _ in 0..arr_len {
                            let val = i32::from_le_bytes([
                                data[pos],
                                data[pos + 1],
                                data[pos + 2],
                                data[pos + 3],
                            ]);
                            vec.push(val);
                            pos += 4;
                            read += 4;
                        }
                        Ok((GgufValue::I32Vec(vec), read))
                    }
                    GgufValueType::F32 => {
                        let mut vec = Vec::with_capacity(arr_len as usize);
                        for _ in 0..arr_len {
                            let val = f32::from_le_bytes([
                                data[pos],
                                data[pos + 1],
                                data[pos + 2],
                                data[pos + 3],
                            ]);
                            vec.push(val);
                            pos += 4;
                            read += 4;
                        }
                        Ok((GgufValue::F32Vec(vec), read))
                    }
                    GgufValueType::String => {
                        let mut vec = Vec::with_capacity(arr_len as usize);
                        for _ in 0..arr_len {
                            let str_len = u64::from_le_bytes([
                                data[pos],
                                data[pos + 1],
                                data[pos + 2],
                                data[pos + 3],
                                data[pos + 4],
                                data[pos + 5],
                                data[pos + 6],
                                data[pos + 7],
                            ]);
                            pos += 8;
                            let s = std::str::from_utf8(&data[pos..pos + str_len as usize])
                                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                            vec.push(s.to_string());
                            pos += str_len as usize;
                            read += 8 + str_len as usize;
                        }
                        Ok((GgufValue::StringVec(vec), read))
                    }
                    _ => {
                        // Skip unknown array types
                        let skip = 12 + (arr_len as usize) * 4;
                        Ok((GgufValue::I32Vec(vec![]), skip))
                    }
                }
            }
        }
    }
}

impl Default for GgufReader {
    fn default() -> Self {
        Self::new()
    }
}

fn unpack_q4_k_scales(packed: &[u8], scales: &mut [u8; 8], mins: &mut [u8; 8]) {
    for i in 0..4 {
        scales[i] = packed[i] & 0x3f;
        mins[i] = packed[i + 4] & 0x3f;
        scales[i + 4] = (packed[i + 8] & 0x0f) | ((packed[i] >> 6) << 4);
        mins[i + 4] = (packed[i + 8] >> 4) | ((packed[i + 4] >> 6) << 4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{KvQuantType, QuantEngine, QuantFormat};
    use std::io::Write;

    fn push_u32(out: &mut Vec<u8>, value: u32) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(out: &mut Vec<u8>, value: u64) {
        out.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string(out: &mut Vec<u8>, value: &str) {
        push_u64(out, value.len() as u64);
        out.extend_from_slice(value.as_bytes());
    }

    fn push_kv_u32(out: &mut Vec<u8>, key: &str, value: u32) {
        push_string(out, key);
        push_u32(out, GgufValueType::U32.id());
        push_u32(out, value);
    }

    fn push_kv_string(out: &mut Vec<u8>, key: &str, value: &str) {
        push_string(out, key);
        push_u32(out, GgufValueType::String.id());
        push_string(out, value);
    }

    #[test]
    fn test_gguf_data_type_ids() {
        assert_eq!(GgufDataType::F32.id(), 0);
        assert_eq!(GgufDataType::F16.id(), 1);
        assert_eq!(GgufDataType::Q4_0.id(), 2);
        assert_eq!(GgufDataType::Q8_0.id(), 8);
    }

    #[test]
    fn test_gguf_data_type_from_id() {
        assert_eq!(GgufDataType::from_id(0), Some(GgufDataType::F32));
        assert_eq!(GgufDataType::from_id(1), Some(GgufDataType::F16));
        assert_eq!(GgufDataType::from_id(2), Some(GgufDataType::Q4_0));
        assert_eq!(GgufDataType::from_id(99), None);
    }

    #[test]
    fn test_gguf_value_type_ids() {
        assert_eq!(GgufValueType::U8.id(), 0);
        assert_eq!(GgufValueType::F32.id(), 6);
        assert_eq!(GgufValueType::Bool.id(), 7);
        assert_eq!(GgufValueType::String.id(), 8);
        assert_eq!(GgufValueType::Array.id(), 9);
        assert_eq!(GgufValueType::U64.id(), 10);
    }

    #[test]
    fn test_data_type_bytes_per_element() {
        assert_eq!(GgufDataType::F32.bytes_per_element(10), 40);
        assert_eq!(GgufDataType::F16.bytes_per_element(10), 20);
        assert!(GgufDataType::Q4_0.bytes_per_element(256) > 0);
        assert!(GgufDataType::Q8_0.bytes_per_element(256) > 0);
    }

    #[test]
    fn test_tensor_info() {
        let info = TensorInfo {
            name: "test".to_string(),
            n_dims: 2,
            shape: vec![128, 256],
            data_type: GgufDataType::F32,
            offset: 0,
            stride: vec![1, 128],
        };
        assert_eq!(info.num_elements(), 32768);
        assert_eq!(info.data_size(), 131072); // 32768 * 4 bytes
    }

    #[test]
    fn test_parse_file_reads_real_tensor_data() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        push_u32(&mut bytes, GGUF_VERSION_3);
        push_u64(&mut bytes, 1); // tensor count
        push_u64(&mut bytes, 4); // metadata count

        push_kv_u32(&mut bytes, "general.alignment", 32);
        push_kv_string(&mut bytes, "general.architecture", "llama");
        push_kv_u32(&mut bytes, "llama.embedding_length", 4);
        push_kv_u32(&mut bytes, "llama.block_count", 1);

        push_string(&mut bytes, "token_embd.weight");
        push_u32(&mut bytes, 1); // dims
        push_u64(&mut bytes, 4);
        push_u32(&mut bytes, GgufDataType::F32.id());
        push_u64(&mut bytes, 0); // tensor offset relative to data section

        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        let expected = [1.0f32, -2.0, 3.5, 4.25];
        for value in expected {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let file = tempfile::NamedTempFile::new().unwrap();
        let mut handle = file.reopen().unwrap();
        handle.write_all(&bytes).unwrap();
        handle.flush().unwrap();

        let metadata = GgufReader::parse_file(file.path()).unwrap();
        assert_eq!(metadata.alignment, 32);
        assert_eq!(
            metadata.tensor_data_offset as usize,
            bytes.len() - expected.len() * 4
        );
        assert_eq!(metadata.hidden_size(), Some(4));
        assert_eq!(metadata.num_layers(), Some(1));

        let tensor = metadata.get_tensor("token_embd.weight").unwrap();
        let values = GgufReader::read_tensor_f32(file.path(), &metadata, tensor).unwrap();
        assert_eq!(values, expected);
    }

    #[test]
    fn test_q8_0_dequantization() {
        let scale = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        let mut block = Vec::new();
        block.extend_from_slice(&scale);
        for i in 0..32 {
            block.push((i as i8 - 16) as u8);
        }

        let values = GgufReader::tensor_bytes_to_f32(GgufDataType::Q8_0, 32, &block).unwrap();
        assert_eq!(values.len(), 32);
        assert_eq!(values[0], -8.0);
        assert_eq!(values[16], 0.0);
        assert_eq!(values[31], 7.5);
    }

    #[test]
    fn test_k_quant_dequantization() {
        let data: Vec<f32> = (0..300).map(|i| i as f32 / 31.0 - 4.0).collect();

        let q4_engine = QuantEngine::new(QuantFormat::Q4_K, KvQuantType::None);
        let q4 = q4_engine.quantize_weights(&data);
        let q4_values =
            GgufReader::tensor_bytes_to_f32(GgufDataType::Q4_K, data.len(), &q4).unwrap();
        assert_eq!(q4.len(), data.len().div_ceil(256) * 144);
        assert_eq!(q4_values.len(), data.len());

        let q2_engine = QuantEngine::new(QuantFormat::Q2_K, KvQuantType::None);
        let q2 = q2_engine.quantize_weights(&data);
        let q2_values =
            GgufReader::tensor_bytes_to_f32(GgufDataType::Q2_K, data.len(), &q2).unwrap();
        assert_eq!(q2.len(), data.len().div_ceil(256) * 84);
        assert_eq!(q2_values.len(), data.len());
    }

    #[test]
    fn test_truncated_tensor_table_returns_error() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        push_u32(&mut bytes, GGUF_VERSION_3);
        push_u64(&mut bytes, 1); // tensor count
        push_u64(&mut bytes, 0); // metadata count

        let file = tempfile::NamedTempFile::new().unwrap();
        let mut handle = file.reopen().unwrap();
        handle.write_all(&bytes).unwrap();
        handle.flush().unwrap();

        let result = std::panic::catch_unwind(|| GgufReader::parse_file(file.path()));
        assert!(result.is_ok());
        assert!(result.unwrap().is_err());
    }

    #[test]
    fn test_gguf_magic() {
        assert_eq!(GGUF_MAGIC, 0x46554747);
        assert_eq!(&GGUF_MAGIC.to_le_bytes(), b"GGUF");
    }
}
