#![allow(dead_code)]
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

pub struct TinyModel {
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<(String, Vec<u64>, Vec<f32>)>,
    pub types: BTreeMap<String, u32>,
}

#[derive(Clone)]
pub enum Value {
    Text(String),
    Uint(u32),
    Float(f32),
    Bool(bool),
    Strings(Vec<String>),
    Floats(Vec<f32>),
    Ints(Vec<i32>),
}

impl TinyModel {
    /// Complete one-layer GQA transformer with deterministic, nontrivial weights.
    ///
    /// head_dim 4 keeps the adjacent-pair and split-half RoPE layouts
    /// distinct (they coincide at head_dim 2, hiding permutation defects);
    /// every real model family uses head_dim >= 32.
    pub fn new() -> Self {
        Self::with_head_dim(4)
    }

    pub fn with_head_dim(head_dim: usize) -> Self {
        let hidden = 2 * head_dim;
        let ff = 2 * hidden;
        let mut metadata = BTreeMap::new();
        metadata.insert("general.architecture".into(), Value::Text("llama".into()));
        metadata.insert("general.name".into(), Value::Text("tiny-regression".into()));
        for (key, value) in [
            ("block_count", 1),
            ("embedding_length", hidden as u32),
            ("feed_forward_length", ff as u32),
            ("attention.head_count", 2),
            ("attention.head_count_kv", 1),
            ("context_length", 32),
            ("vocab_size", 8),
        ] {
            metadata.insert(format!("llama.{key}"), Value::Uint(value));
        }
        metadata.insert("llama.rope.freq_base".into(), Value::Float(10000.0));
        metadata.insert(
            "llama.attention.layer_norm_rms_epsilon".into(),
            Value::Float(1e-5),
        );
        let mut model = Self {
            metadata,
            tensors: Vec::new(),
            types: BTreeMap::new(),
        };
        for (name, rows, cols) in [
            ("token_embd.weight", 8, hidden),
            ("output.weight", 8, hidden),
            ("blk.0.attn_q.weight", hidden, hidden),
            ("blk.0.attn_k.weight", head_dim, hidden),
            ("blk.0.attn_v.weight", head_dim, hidden),
            ("blk.0.attn_output.weight", hidden, hidden),
            ("blk.0.ffn_gate.weight", ff, hidden),
            ("blk.0.ffn_up.weight", ff, hidden),
            ("blk.0.ffn_down.weight", hidden, ff),
        ] {
            let salt = model.tensors.len() + 1;
            let data = (0..rows * cols)
                .map(|i| ((i * 17 + salt * 7) % 31) as f32 / 31.0 - 0.45)
                .collect();
            model
                .tensors
                .push((name.into(), vec![cols as u64, rows as u64], data));
        }
        for name in [
            "output_norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.ffn_norm.weight",
        ] {
            model
                .tensors
                .push((name.into(), vec![hidden as u64], vec![1.; hidden]));
        }
        model
    }

    pub fn resize_vocab(&mut self, size: usize) {
        self.metadata
            .insert("llama.vocab_size".into(), Value::Uint(size as u32));
        for (name, shape, data) in &mut self.tensors {
            if name == "token_embd.weight" || name == "output.weight" {
                let cols = shape[0] as usize;
                shape[1] = size as u64;
                data.resize(size * cols, 0.01);
            }
        }
    }

    pub fn write(&self, path: &Path) {
        fn string(out: &mut Vec<u8>, text: &str) {
            out.extend_from_slice(&(text.len() as u64).to_le_bytes());
            out.extend_from_slice(text.as_bytes());
        }
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(self.metadata.len() as u64).to_le_bytes());
        for (key, value) in &self.metadata {
            string(&mut bytes, key);
            match value {
                Value::Bool(v) => {
                    bytes.extend_from_slice(&7u32.to_le_bytes());
                    bytes.push(*v as u8);
                }
                Value::Strings(values) => {
                    bytes.extend_from_slice(&9u32.to_le_bytes());
                    bytes.extend_from_slice(&8u32.to_le_bytes());
                    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
                    for v in values {
                        string(&mut bytes, v);
                    }
                }
                Value::Floats(values) => {
                    bytes.extend_from_slice(&9u32.to_le_bytes());
                    bytes.extend_from_slice(&6u32.to_le_bytes());
                    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
                    for v in values {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }
                Value::Ints(values) => {
                    bytes.extend_from_slice(&9u32.to_le_bytes());
                    bytes.extend_from_slice(&5u32.to_le_bytes());
                    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
                    for v in values {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }
                Value::Text(value) => {
                    bytes.extend_from_slice(&8u32.to_le_bytes());
                    string(&mut bytes, value);
                }
                Value::Uint(value) => {
                    bytes.extend_from_slice(&4u32.to_le_bytes());
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
                Value::Float(value) => {
                    bytes.extend_from_slice(&6u32.to_le_bytes());
                    bytes.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        let mut payload = Vec::new();
        for (name, shape, values) in &self.tensors {
            string(&mut bytes, name);
            bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for dim in shape {
                bytes.extend_from_slice(&dim.to_le_bytes());
            }
            bytes.extend_from_slice(&self.types.get(name).copied().unwrap_or(0).to_le_bytes());
            bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            for value in values {
                payload.extend_from_slice(&value.to_le_bytes());
            }
            payload.resize(payload.len().div_ceil(32) * 32, 0);
        }
        bytes.resize(bytes.len().div_ceil(32) * 32, 0);
        bytes.extend(payload);
        std::fs::File::create(path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
    }
}
