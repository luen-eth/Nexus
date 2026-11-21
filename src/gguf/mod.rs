//! GGUF (GPT-Generated Unified Format) parser and writer
//! GGUF v2/v3 little-endian metadata, checked tensor layout and selected weight codecs.

#![allow(non_camel_case_types)]

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn zeroed_bytes(len: usize) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| invalid("GGUF allocation exceeds available memory"))?;
    bytes.resize(len, 0);
    Ok(bytes)
}

/// Bounds metadata parsing independently of the model payload size.
const MAX_HEADER_BYTES: u64 = 128 * 1024 * 1024;
struct CountingReader<'a> {
    inner: io::Take<&'a mut dyn Read>,
    position: u64,
}
impl Read for CountingReader<'_> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.position += n as u64;
        Ok(n)
    }
}

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

/// Stable GGML IDs and exact block geometry (ggml fc2b0053, April 2026).
/// Recognition does not imply an implemented decoder. Removed/reserved IDs are rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GgufDataType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,
}

impl GgufDataType {
    pub fn id(&self) -> u32 {
        *self as u32
    }
    pub fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::F32),
            1 => Some(Self::F16),
            2 => Some(Self::Q4_0),
            3 => Some(Self::Q4_1),
            6 => Some(Self::Q5_0),
            7 => Some(Self::Q5_1),
            8 => Some(Self::Q8_0),
            9 => Some(Self::Q8_1),
            10 => Some(Self::Q2_K),
            11 => Some(Self::Q3_K),
            12 => Some(Self::Q4_K),
            13 => Some(Self::Q5_K),
            14 => Some(Self::Q6_K),
            15 => Some(Self::Q8_K),
            16 => Some(Self::IQ2_XXS),
            17 => Some(Self::IQ2_XS),
            18 => Some(Self::IQ3_XXS),
            19 => Some(Self::IQ1_S),
            20 => Some(Self::IQ4_NL),
            21 => Some(Self::IQ3_S),
            22 => Some(Self::IQ2_S),
            23 => Some(Self::IQ4_XS),
            24 => Some(Self::I8),
            25 => Some(Self::I16),
            26 => Some(Self::I32),
            27 => Some(Self::I64),
            28 => Some(Self::F64),
            29 => Some(Self::IQ1_M),
            30 => Some(Self::BF16),
            34 => Some(Self::TQ1_0),
            35 => Some(Self::TQ2_0),
            39 => Some(Self::MXFP4),
            40 => Some(Self::NVFP4),
            41 => Some(Self::Q1_0),
            _ => None,
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2_K => "Q2_K",
            Self::Q3_K => "Q3_K",
            Self::Q4_K => "Q4_K",
            Self::Q5_K => "Q5_K",
            Self::Q6_K => "Q6_K",
            Self::Q8_K => "Q8_K",
            Self::IQ2_XXS => "IQ2_XXS",
            Self::IQ2_XS => "IQ2_XS",
            Self::IQ3_XXS => "IQ3_XXS",
            Self::IQ1_S => "IQ1_S",
            Self::IQ4_NL => "IQ4_NL",
            Self::IQ3_S => "IQ3_S",
            Self::IQ2_S => "IQ2_S",
            Self::IQ4_XS => "IQ4_XS",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::IQ1_M => "IQ1_M",
            Self::BF16 => "BF16",
            Self::TQ1_0 => "TQ1_0",
            Self::TQ2_0 => "TQ2_0",
            Self::MXFP4 => "MXFP4",
            Self::NVFP4 => "NVFP4",
            Self::Q1_0 => "Q1_0",
        }
    }
    /// Number of elements and bytes in one on-disk block.
    pub fn block_layout(&self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::Q2_K => (256, 84),
            Self::Q3_K => (256, 110),
            Self::Q4_K => (256, 144),
            Self::Q5_K => (256, 176),
            Self::Q6_K => (256, 210),
            Self::Q8_K => (256, 292),
            Self::IQ2_XXS => (256, 66),
            Self::IQ2_XS => (256, 74),
            Self::IQ3_XXS => (256, 98),
            Self::IQ1_S => (256, 50),
            Self::IQ4_NL => (32, 18),
            Self::IQ3_S => (256, 110),
            Self::IQ2_S => (256, 82),
            Self::IQ4_XS => (256, 136),
            Self::I8 => (1, 1),
            Self::I16 => (1, 2),
            Self::I32 => (1, 4),
            Self::I64 => (1, 8),
            Self::F64 => (1, 8),
            Self::IQ1_M => (256, 56),
            Self::BF16 => (1, 2),
            Self::TQ1_0 => (256, 54),
            Self::TQ2_0 => (256, 66),
            Self::MXFP4 => (32, 17),
            Self::NVFP4 => (64, 36),
            Self::Q1_0 => (128, 18),
        }
    }
    pub fn decode_supported(&self) -> bool {
        matches!(
            self,
            Self::F32 | Self::F16 | Self::BF16 | Self::Q8_0 | Self::Q4_0 | Self::Q4_K | Self::Q2_K
        )
    }
    /// Size of padded blocks; tensor rows must additionally satisfy block divisibility.
    pub fn checked_data_size(&self, elements: usize) -> io::Result<usize> {
        let (block, size) = self.block_layout();
        elements
            .div_ceil(block)
            .checked_mul(size)
            .ok_or_else(|| invalid("GGUF tensor byte size overflow"))
    }
    /// Compatibility helper: invalid sizes saturate; use checked_data_size for untrusted input.
    pub fn bytes_per_element(&self, elements: usize) -> usize {
        self.checked_data_size(elements).unwrap_or(usize::MAX)
    }
    pub fn elem_size(&self) -> usize {
        let (block, size) = self.block_layout();
        if block == 1 {
            size
        } else {
            0
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
#[derive(Debug, Clone, PartialEq)]
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
    I8Vec(Vec<i8>),
    U16Vec(Vec<u16>),
    I16Vec(Vec<i16>),
    I64Vec(Vec<i64>),
    F64Vec(Vec<f64>),
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
    pub fn checked_num_elements(&self) -> io::Result<usize> {
        if self.shape.is_empty() || self.shape.len() > 4 || self.n_dims as usize != self.shape.len()
        {
            return Err(invalid("GGUF tensor dimensions must be 1..=4"));
        }
        self.shape.iter().try_fold(1usize, |n, &d| {
            let d = usize::try_from(d).map_err(|_| invalid("Tensor dimension overflow"))?;
            if d == 0 {
                return Err(invalid("Zero tensor dimension"));
            }
            n.checked_mul(d)
                .filter(|n| *n <= isize::MAX as usize)
                .ok_or_else(|| invalid("Tensor element count overflow"))
        })
    }
    pub fn checked_data_size(&self) -> io::Result<usize> {
        let n = self.checked_num_elements()?;
        let (block, _) = self.data_type.block_layout();
        if !self.shape[0].is_multiple_of(block as u64) {
            return Err(invalid(
                "GGUF tensor row is not divisible by codec block size",
            ));
        }
        self.data_type.checked_data_size(n)
    }
    pub fn num_elements(&self) -> usize {
        self.checked_num_elements().unwrap_or(usize::MAX)
    }
    pub fn data_size(&self) -> usize {
        self.checked_data_size().unwrap_or(usize::MAX)
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

    /// Resolve an architecture-scoped metadata key: try the file's own
    /// architecture prefix first, then the `llama.` fallback so that pre-arch
    /// tooling and `llama`-labelled files keep working.
    fn arch_usize(&self, key: &str) -> Option<usize> {
        if let Some(arch) = self.architecture() {
            if let Some(value) = self.get_usize(&format!("{arch}.{key}")) {
                return Some(value);
            }
        }
        self.get_usize(&format!("llama.{key}"))
    }

    /// Get context length
    pub fn context_length(&self) -> Option<usize> {
        self.arch_usize("context_length")
    }

    /// Get number of attention heads
    pub fn num_attention_heads(&self) -> Option<usize> {
        self.arch_usize("attention.head_count")
    }

    /// Get number of key heads (for GQA)
    pub fn num_key_value_heads(&self) -> Option<usize> {
        self.arch_usize("attention.head_count_kv")
    }

    /// Get number of layers
    pub fn num_layers(&self) -> Option<usize> {
        self.arch_usize("block_count")
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> Option<usize> {
        self.arch_usize("embedding_length")
    }

    /// Get feed forward size
    pub fn feed_forward_size(&self) -> Option<usize> {
        self.arch_usize("feed_forward_length")
    }

    /// Get rope dimensionality
    pub fn rope_dim(&self) -> Option<usize> {
        self.arch_usize("rope.dimension_count")
    }

    /// Get rope frequency base
    pub fn rope_freq_base(&self) -> f32 {
        if let Some(arch) = self.architecture() {
            if let Some(value) = self.get_f32(&format!("{arch}.rope.freq_base")) {
                return value;
            }
        }
        self.get_f32("llama.rope.freq_base").unwrap_or(10000.0)
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

    /// Validate before allocation/decoding, including calls supplied with hand-built metadata.
    pub fn validate_tensor_layout(&self, file_len: u64) -> io::Result<()> {
        if self.alignment == 0 || !self.alignment.is_power_of_two() || self.alignment > 1024 * 1024
        {
            return Err(invalid("Invalid GGUF alignment"));
        }
        if self.tensor_count != self.tensors.len() as u64 {
            return Err(invalid("Tensor count mismatch"));
        }
        let mut names = HashSet::new();
        let mut ranges = Vec::with_capacity(self.tensors.len());
        for tensor in &self.tensors {
            if tensor.name.is_empty() || !names.insert(tensor.name.as_str()) {
                return Err(invalid("Duplicate/empty GGUF tensor name"));
            }
            if !tensor.offset.is_multiple_of(self.alignment as u64) {
                return Err(invalid("Unaligned GGUF tensor offset"));
            }
            let size = tensor.checked_data_size()? as u64;
            let start = self
                .tensor_data_offset
                .checked_add(tensor.offset)
                .ok_or_else(|| invalid("Tensor offset overflow"))?;
            let end = start
                .checked_add(size)
                .ok_or_else(|| invalid("Tensor range overflow"))?;
            if end > file_len {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Tensor payload exceeds file",
                ));
            }
            ranges.push((start, end));
        }
        ranges.sort_unstable();
        if ranges.windows(2).any(|w| w[0].1 > w[1].0) {
            return Err(invalid("Overlapping GGUF tensor payloads"));
        }
        Ok(())
    }

    /// Absolute byte offset of a tensor's data in the GGUF file.
    pub fn absolute_tensor_offset(&self, tensor: &TensorInfo) -> u64 {
        self.tensor_data_offset.saturating_add(tensor.offset)
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
    data_offset: u64,
}

impl GgufReader {
    pub fn new() -> Self {
        GgufReader {
            data_offset: 0,
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
        if self.tensor_count > 1_000_000 || self.kv_count > 100_000 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "GGUF table count exceeds limits",
            ));
        }
        Ok(())
    }

    /// Read metadata key-value pairs
    pub fn read_metadata(&mut self, reader: &mut dyn Read) -> io::Result<()> {
        for _ in 0..self.kv_count {
            let key_len = self.byte_order.read_u64(reader)?;
            if key_len == 0 || key_len > 65535 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "GGUF key exceeds limit",
                ));
            }
            let mut key_bytes = zeroed_bytes(key_len as usize)?;
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
            if self.metadata.insert(key, value).is_some() {
                return Err(invalid("Duplicate GGUF metadata key"));
            }
        }

        Ok(())
    }

    /// Read tensor info
    pub fn read_tensors(&mut self, reader: &mut dyn Read) -> io::Result<()> {
        for _ in 0..self.tensor_count {
            // Read name length and name
            let name_len = self.byte_order.read_u64(reader)?;
            if name_len == 0 || name_len > 64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "GGUF tensor name exceeds limit",
                ));
            }
            let mut name_bytes = zeroed_bytes(name_len as usize)?;
            reader.read_exact(&mut name_bytes)?;
            let name = String::from_utf8(name_bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

            // Read number of dimensions
            let n_dims = self.byte_order.read_u32(reader)?;

            // Read shape
            if n_dims == 0 || n_dims > 4 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "GGUF tensor dimensions must be 1..=4",
                ));
            }
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
                if byte[0] > 1 {
                    return Err(invalid("Invalid GGUF boolean"));
                }
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
            GgufValueType::U16 => {
                let mut b = [0u8; 2];
                reader.read_exact(&mut b)?;
                Ok(GgufValue::U16(u16::from_le_bytes(b)))
            }
            GgufValueType::I16 => {
                let mut b = [0u8; 2];
                reader.read_exact(&mut b)?;
                Ok(GgufValue::I16(i16::from_le_bytes(b)))
            }
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
                if str_len > 64 * 1024 * 1024 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "GGUF metadata string exceeds 64 MiB limit",
                    ));
                }
                let mut str_bytes = zeroed_bytes(str_len as usize)?;
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
                if arr_len > 1_000_000 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "GGUF metadata array exceeds item limit",
                    ));
                }
                macro_rules! primitive_vec {
                    ($scalar:ident, $vector:ident) => {{
                        let mut values = Vec::with_capacity(arr_len as usize);
                        for _ in 0..arr_len {
                            if let GgufValue::$scalar(v) = Self::read_value(reader, arr_type, bo)? {
                                values.push(v);
                            }
                        }
                        Ok(GgufValue::$vector(values))
                    }};
                }
                match arr_type {
                    GgufValueType::U8 => primitive_vec!(U8, U8Vec),
                    GgufValueType::I8 => primitive_vec!(I8, I8Vec),
                    GgufValueType::U16 => primitive_vec!(U16, U16Vec),
                    GgufValueType::I16 => primitive_vec!(I16, I16Vec),
                    GgufValueType::I64 => primitive_vec!(I64, I64Vec),
                    GgufValueType::F64 => primitive_vec!(F64, F64Vec),
                    GgufValueType::Bool => {
                        let mut vec = Vec::with_capacity(arr_len as usize);
                        for _ in 0..arr_len {
                            let mut b = [0u8; 1];
                            reader.read_exact(&mut b)?;
                            if b[0] > 1 {
                                return Err(invalid("Invalid GGUF boolean"));
                            }
                            vec.push(b[0]);
                        }
                        Ok(GgufValue::U8Vec(vec))
                    }
                    GgufValueType::I32 => {
                        let mut vec = Vec::with_capacity(arr_len as usize);
                        for _ in 0..arr_len {
                            let val = bo.read_u32(reader)?;
                            vec.push(val as i32);
                        }
                        Ok(GgufValue::I32Vec(vec))
                    }
                    GgufValueType::U32 => {
                        let mut vec = Vec::with_capacity(arr_len as usize);
                        for _ in 0..arr_len {
                            vec.push(bo.read_u32(reader)? as u64);
                        }
                        Ok(GgufValue::U64Vec(vec))
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
                            if str_len > 64 * 1024 * 1024 {
                                return Err(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "GGUF metadata string exceeds 64 MiB limit",
                                ));
                            }
                            let mut str_bytes = zeroed_bytes(str_len as usize)?;
                            reader.read_exact(&mut str_bytes)?;
                            let s = String::from_utf8(str_bytes)
                                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                            vec.push(s);
                        }
                        Ok(GgufValue::StringVec(vec))
                    }
                    GgufValueType::Array => Err(invalid("Nested GGUF arrays are not allowed")),
                }
            }
        }
    }

    /// Full parse: reads header, metadata, and tensor info
    pub fn parse(&mut self, reader: &mut dyn Read) -> io::Result<ModelMetadata> {
        *self = Self::new();
        let mut reader = CountingReader {
            inner: reader.take(MAX_HEADER_BYTES),
            position: 0,
        };
        self.read_header(&mut reader)?;
        self.read_metadata(&mut reader)?;
        self.read_tensors(&mut reader)?;
        let alignment = match self.metadata.get("general.alignment") {
            None => 32,
            Some(GgufValue::U32(n)) => *n as usize,
            _ => return Err(invalid("GGUF alignment must be U32")),
        };
        if alignment == 0 || !alignment.is_power_of_two() || alignment > 1024 * 1024 {
            return Err(invalid("Invalid GGUF alignment"));
        }
        let tensor_data_offset = reader
            .position
            .checked_add(alignment as u64 - 1)
            .map(|p| p / alignment as u64 * alignment as u64)
            .ok_or_else(|| invalid("Header alignment overflow"))?;
        let metadata = ModelMetadata {
            version: self.version,
            tensor_count: self.tensor_count,
            kv_count: self.kv_count,
            byte_order: self.byte_order,
            alignment,
            tensor_data_offset,
            metadata: self.metadata.clone(),
            tensors: self.tensors.clone(),
        };
        self.data_offset = tensor_data_offset;
        // Read-only streams have no file length. Still validate all descriptor arithmetic and overlap.
        metadata.validate_tensor_layout(u64::MAX)?;
        Ok(metadata)
    }

    /// Get tensor data offset from file
    pub fn tensor_data_offset(&self) -> u64 {
        self.data_offset
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
        let offset = metadata
            .tensor_data_offset
            .checked_add(tensor.offset)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Tensor offset overflow"))?;
        let size = tensor.checked_data_size()?;
        if offset
            .checked_add(size as u64)
            .is_none_or(|end| end > file.metadata().map(|m| m.len()).unwrap_or(0))
        {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Tensor payload exceeds file",
            ));
        }
        file.seek(SeekFrom::Start(offset))?;
        let mut data = zeroed_bytes(size)?;
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
        Self::tensor_bytes_to_f32(tensor.data_type, tensor.checked_num_elements()?, &data)
    }

    /// Convert supported GGUF tensor payloads to f32.
    pub fn tensor_bytes_to_f32(
        data_type: GgufDataType,
        output_len: usize,
        data: &[u8],
    ) -> io::Result<Vec<f32>> {
        if !data_type.decode_supported() {
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                format!("Dequantization for {data_type} is not implemented"),
            ));
        }
        let needed = data_type.checked_data_size(output_len)?;
        if data.len() < needed {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "Tensor payload is truncated",
            ));
        }
        if output_len == 0 {
            return Ok(Vec::new());
        }
        match data_type {
            GgufDataType::F32 => {
                if data.len() < output_len * 4 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "F32 tensor payload is truncated",
                    ));
                }
                Ok(data
                    .as_chunks::<4>()
                    .0
                    .iter()
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
                    .as_chunks::<2>()
                    .0
                    .iter()
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
                    .as_chunks::<2>()
                    .0
                    .iter()
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
        for block in data.as_chunks::<BLOCK>().0 {
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
        for block in data.as_chunks::<BLOCK>().0 {
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
        for block in data.as_chunks::<BLOCK>().0 {
            let d = Self::decode_f16_le(&block[..2]);
            let dmin = Self::decode_f16_le(&block[2..4]);
            let mut scales = [0u8; 8];
            let mut mins = [0u8; 8];
            unpack_q4_k_scales(&block[4..16], &mut scales, &mut mins);
            let qs = &block[16..];

            for i in 0..QK {
                let packed = qs[(i / 64) * 32 + i % 32];
                let q = (packed >> (4 * ((i / 32) % 2))) & 0x0f;
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
        for block in data.as_chunks::<BLOCK>().0 {
            let scales = &block[..16];
            let qs = &block[16..80];
            let d = Self::decode_f16_le(&block[80..82]);
            let dmin = Self::decode_f16_le(&block[82..84]);

            for i in 0..QK {
                let q = (qs[(i / 128) * 32 + i % 32] >> (2 * ((i % 128) / 32))) & 0x03;
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
        let metadata = self.parse(&mut io::Cursor::new(data))?;
        metadata.validate_tensor_layout(data.len() as u64)?;
        Ok(metadata)
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
        let q4 = q4_engine.quantize_weights(&data).unwrap();
        let q4_values =
            GgufReader::tensor_bytes_to_f32(GgufDataType::Q4_K, data.len(), &q4).unwrap();
        assert_eq!(q4.len(), data.len().div_ceil(256) * 144);
        assert_eq!(q4_values.len(), data.len());

        let q2_engine = QuantEngine::new(QuantFormat::Q2_K, KvQuantType::None);
        let q2 = q2_engine.quantize_weights(&data).unwrap();
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
    #[test]
    fn metadata_lengths_are_bounded_and_scalar_reads_are_exact() {
        for is_array in [false, true] {
            let mut bytes = b"GGUF".to_vec();
            push_u32(&mut bytes, 3);
            push_u64(&mut bytes, 0);
            push_u64(&mut bytes, 1);
            push_string(&mut bytes, "nexus.tokenizer.json");
            push_u32(&mut bytes, if is_array { 9 } else { 8 });
            if is_array {
                push_u32(&mut bytes, 8);
            }
            push_u64(&mut bytes, u64::MAX);
            let mut file = tempfile::NamedTempFile::new().unwrap();
            file.write_all(&bytes).unwrap();
            let result = std::panic::catch_unwind(|| GgufReader::parse_file(file.path()));
            assert!(result.is_ok());
            assert!(result.unwrap().is_err());
        }
        let mut bytes = b"GGUF".to_vec();
        push_u32(&mut bytes, 3);
        push_u64(&mut bytes, 0);
        push_u64(&mut bytes, 2);
        push_string(&mut bytes, "test.short");
        push_u32(&mut bytes, 2);
        bytes.extend_from_slice(&0x1234u16.to_le_bytes());
        push_kv_string(&mut bytes, "general.name", "cursor stays aligned");
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        let metadata = GgufReader::parse_file(file.path()).unwrap();
        assert!(matches!(
            metadata.metadata["test.short"],
            GgufValue::U16(0x1234)
        ));
        assert_eq!(
            metadata.get_str("general.name"),
            Some("cursor stays aligned")
        );
    }
}
