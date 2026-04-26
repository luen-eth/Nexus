//! Tokenization interface.
//!
//! Nexus uses GGUF tokenizer metadata when present and falls back to a byte-level
//! tokenizer for smoke tests and incomplete model metadata.

use std::collections::HashMap;

use crate::gguf::ModelMetadata;

/// Minimal tokenizer contract used by runtime entrypoints.
pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode(&self, tokens: &[u32]) -> String;
}

/// Byte-level tokenizer: every UTF-8 byte maps to one token ID.
#[derive(Debug, Default, Clone, Copy)]
pub struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.bytes().map(u32::from).collect()
    }

    fn decode(&self, tokens: &[u32]) -> String {
        let bytes: Vec<u8> = tokens
            .iter()
            .map(|&token| u8::try_from(token).unwrap_or(b'?'))
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Well-known special token IDs from GGUF metadata.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SpecialTokenIds {
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub unk: Option<u32>,
    pub sep: Option<u32>,
    pub pad: Option<u32>,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct TokenizerBehavior {
    pub lowercase: bool,
    pub strip_accents: bool,
    pub byte_level: bool,
    pub add_prefix_space: bool,
    pub collapse_whitespace: bool,
    pub space_marker: Option<String>,
}

/// GGUF vocabulary tokenizer using `tokenizer.ggml.tokens` metadata.
#[derive(Debug, Clone)]
pub struct GgufTokenizer {
    tokens: Vec<String>,
    token_to_id: HashMap<String, u32>,
    sorted_tokens: Vec<String>,
    merge_ranks: HashMap<(String, String), usize>,
    special_tokens: SpecialTokenIds,
    chat_template: Option<String>,
    behavior: TokenizerBehavior,
}

impl GgufTokenizer {
    pub fn from_metadata(metadata: &ModelMetadata) -> Option<Self> {
        let tokens = metadata
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|value| value.as_string_vec())?
            .clone();
        if tokens.is_empty() {
            return None;
        }

        let token_to_id: HashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(idx, token)| (token.clone(), idx as u32))
            .collect();
        let mut sorted_tokens = tokens.clone();
        sorted_tokens.sort_by_key(|token| std::cmp::Reverse(token.len()));
        let merge_ranks = metadata
            .metadata
            .get("tokenizer.ggml.merges")
            .and_then(|value| value.as_string_vec())
            .map(|merges| {
                merges
                    .iter()
                    .enumerate()
                    .filter_map(|(rank, merge)| {
                        let (left, right) = merge.split_once(' ')?;
                        Some(((left.to_string(), right.to_string()), rank))
                    })
                    .collect()
            })
            .unwrap_or_default();
        let special_tokens = SpecialTokenIds {
            bos: metadata_token_id(metadata, "tokenizer.ggml.bos_token_id"),
            eos: metadata_token_id(metadata, "tokenizer.ggml.eos_token_id"),
            unk: metadata_token_id(metadata, "tokenizer.ggml.unknown_token_id"),
            sep: metadata_token_id(metadata, "tokenizer.ggml.seperator_token_id")
                .or_else(|| metadata_token_id(metadata, "tokenizer.ggml.separator_token_id")),
            pad: metadata_token_id(metadata, "tokenizer.ggml.padding_token_id"),
        };
        let chat_template = metadata
            .metadata
            .get("tokenizer.chat_template")
            .or_else(|| metadata.metadata.get("tokenizer.ggml.chat_template"))
            .and_then(|value| value.as_str())
            .map(ToString::to_string);
        let behavior = TokenizerBehavior {
            lowercase: metadata_bool(metadata, "nexus.tokenizer.normalizer.lowercase"),
            strip_accents: metadata_bool(metadata, "nexus.tokenizer.normalizer.strip_accents"),
            byte_level: metadata_bool(metadata, "nexus.tokenizer.pre_tokenizer.byte_level"),
            add_prefix_space: metadata_bool(
                metadata,
                "nexus.tokenizer.pre_tokenizer.add_prefix_space",
            ),
            collapse_whitespace: metadata_bool(
                metadata,
                "nexus.tokenizer.normalizer.collapse_whitespace",
            ),
            space_marker: metadata
                .metadata
                .get("nexus.tokenizer.space_marker")
                .and_then(|value| value.as_str())
                .map(ToString::to_string)
                .or_else(|| detect_space_marker(&tokens)),
        };

        Some(GgufTokenizer {
            tokens,
            token_to_id,
            sorted_tokens,
            merge_ranks,
            special_tokens,
            chat_template,
            behavior,
        })
    }

    pub fn special_tokens(&self) -> SpecialTokenIds {
        self.special_tokens
    }

    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    pub fn behavior(&self) -> &TokenizerBehavior {
        &self.behavior
    }

    fn prepare_text(&self, text: &str) -> String {
        let mut normalized = text.to_string();
        if self.behavior.lowercase {
            normalized = normalized.to_lowercase();
        }
        if self.behavior.strip_accents {
            normalized = strip_common_accents(&normalized);
        }
        if self.behavior.collapse_whitespace {
            normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");
        }
        if self.behavior.byte_level
            && self.behavior.add_prefix_space
            && !normalized.is_empty()
            && !normalized.starts_with(char::is_whitespace)
        {
            normalized.insert(0, ' ');
        }
        if let Some(marker) = &self.behavior.space_marker {
            normalized = normalized.replace(' ', marker);
        }
        normalized
    }

    fn encode_greedy(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            if let Some(token) = self
                .sorted_tokens
                .iter()
                .find(|token| !token.is_empty() && remaining.starts_with(token.as_str()))
            {
                if let Some(id) = self.token_to_id.get(token) {
                    tokens.push(*id);
                    remaining = &remaining[token.len()..];
                    continue;
                }
            }

            let ch = remaining.chars().next().unwrap();
            if let Some(id) = self.token_to_id.get(&ch.to_string()) {
                tokens.push(*id);
            } else if let Some(unk) = self.special_tokens.unk {
                tokens.push(unk);
            } else {
                tokens.extend(ch.to_string().bytes().map(u32::from));
            }
            remaining = &remaining[ch.len_utf8()..];
        }

        tokens
    }

    fn encode_bpe(&self, text: &str) -> Vec<u32> {
        let mut pieces: Vec<String> = text.chars().map(|ch| ch.to_string()).collect();
        if pieces.is_empty() {
            return Vec::new();
        }

        loop {
            let best = pieces
                .windows(2)
                .enumerate()
                .filter_map(|(idx, pair)| {
                    let rank = self
                        .merge_ranks
                        .get(&(pair[0].clone(), pair[1].clone()))
                        .copied()?;
                    let merged = format!("{}{}", pair[0], pair[1]);
                    self.token_to_id.get(&merged).map(|_| (idx, rank, merged))
                })
                .min_by_key(|(_, rank, _)| *rank);

            if let Some((idx, _, merged)) = best {
                pieces.splice(idx..idx + 2, [merged]);
            } else {
                break;
            }
        }

        let mut output = Vec::new();
        for piece in pieces {
            if let Some(id) = self.token_to_id.get(&piece) {
                output.push(*id);
            } else {
                output.extend(self.encode_greedy(&piece));
            }
        }
        output
    }
}

impl Tokenizer for GgufTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let prepared = self.prepare_text(text);
        if self.merge_ranks.is_empty() {
            self.encode_greedy(&prepared)
        } else {
            self.encode_bpe(&prepared)
        }
    }

    fn decode(&self, tokens: &[u32]) -> String {
        let mut output = String::new();
        let fallback = ByteTokenizer;
        for &token in tokens {
            if let Some(piece) = self.tokens.get(token as usize) {
                output.push_str(piece);
            } else {
                output.push_str(&fallback.decode(&[token]));
            }
        }
        if let Some(marker) = &self.behavior.space_marker {
            output = output.replace(marker, " ");
        }
        output
    }
}

/// Runtime tokenizer selected from model metadata, with byte fallback.
#[derive(Debug, Clone)]
pub enum RuntimeTokenizer {
    Byte(ByteTokenizer),
    Gguf(Box<GgufTokenizer>),
}

impl RuntimeTokenizer {
    pub fn from_metadata(metadata: Option<&ModelMetadata>) -> Self {
        metadata
            .and_then(GgufTokenizer::from_metadata)
            .map(Box::new)
            .map(RuntimeTokenizer::Gguf)
            .unwrap_or(RuntimeTokenizer::Byte(ByteTokenizer))
    }

    pub fn render_chat_messages<'a, I>(&self, messages: I, add_generation_prompt: bool) -> String
    where
        I: IntoIterator<Item = (&'a str, &'a str)>,
    {
        let messages: Vec<(&str, &str)> = messages.into_iter().collect();
        let template = match self {
            RuntimeTokenizer::Byte(_) => None,
            RuntimeTokenizer::Gguf(tokenizer) => tokenizer.chat_template(),
        };

        if let Some(template) = template {
            if template.contains("<|im_start|>") {
                let mut rendered = String::new();
                for (role, content) in &messages {
                    rendered.push_str("<|im_start|>");
                    rendered.push_str(role);
                    rendered.push('\n');
                    rendered.push_str(content);
                    rendered.push_str("<|im_end|>\n");
                }
                if add_generation_prompt {
                    rendered.push_str("<|im_start|>assistant\n");
                }
                return rendered;
            }
            if template.contains("[/INST]") {
                return render_llama_inst_chat(&messages, add_generation_prompt);
            }
        }

        render_plain_chat(&messages, add_generation_prompt)
    }
}

impl Default for RuntimeTokenizer {
    fn default() -> Self {
        RuntimeTokenizer::Byte(ByteTokenizer)
    }
}

impl Tokenizer for RuntimeTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            RuntimeTokenizer::Byte(tokenizer) => tokenizer.encode(text),
            RuntimeTokenizer::Gguf(tokenizer) => tokenizer.encode(text),
        }
    }

    fn decode(&self, tokens: &[u32]) -> String {
        match self {
            RuntimeTokenizer::Byte(tokenizer) => tokenizer.decode(tokens),
            RuntimeTokenizer::Gguf(tokenizer) => tokenizer.decode(tokens),
        }
    }
}

fn metadata_token_id(metadata: &ModelMetadata, key: &str) -> Option<u32> {
    metadata.metadata.get(key).and_then(|value| match value {
        crate::gguf::GgufValue::U32(v) => Some(*v),
        crate::gguf::GgufValue::I32(v) => u32::try_from(*v).ok(),
        crate::gguf::GgufValue::U64(v) => u32::try_from(*v).ok(),
        crate::gguf::GgufValue::I64(v) => u32::try_from(*v).ok(),
        _ => None,
    })
}

fn metadata_bool(metadata: &ModelMetadata, key: &str) -> bool {
    metadata
        .metadata
        .get(key)
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}

fn detect_space_marker(tokens: &[String]) -> Option<String> {
    if tokens.iter().any(|token| token.contains('Ġ')) {
        Some("Ġ".to_string())
    } else if tokens.iter().any(|token| token.contains('▁')) {
        Some("▁".to_string())
    } else {
        None
    }
}

fn strip_common_accents(input: &str) -> String {
    input
        .chars()
        .map(|ch| match ch {
            'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' => 'a',
            'ç' | 'ć' | 'ĉ' | 'ċ' | 'č' => 'c',
            'ď' | 'đ' => 'd',
            'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ĕ' | 'ė' | 'ę' | 'ě' => 'e',
            'ĝ' | 'ğ' | 'ġ' | 'ģ' => 'g',
            'ĥ' | 'ħ' => 'h',
            'ì' | 'í' | 'î' | 'ï' | 'ĩ' | 'ī' | 'ĭ' | 'į' | 'ı' => 'i',
            'ĵ' => 'j',
            'ķ' => 'k',
            'ĺ' | 'ļ' | 'ľ' | 'ŀ' | 'ł' => 'l',
            'ñ' | 'ń' | 'ņ' | 'ň' => 'n',
            'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ŏ' | 'ő' => 'o',
            'ŕ' | 'ŗ' | 'ř' => 'r',
            'ś' | 'ŝ' | 'ş' | 'š' => 's',
            'ţ' | 'ť' | 'ŧ' => 't',
            'ù' | 'ú' | 'û' | 'ü' | 'ũ' | 'ū' | 'ŭ' | 'ů' | 'ű' | 'ų' => 'u',
            'ŵ' => 'w',
            'ý' | 'ÿ' | 'ŷ' => 'y',
            'ź' | 'ż' | 'ž' => 'z',
            'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' | 'Ā' | 'Ă' | 'Ą' => 'A',
            'Ç' | 'Ć' | 'Ĉ' | 'Ċ' | 'Č' => 'C',
            'Ď' | 'Đ' => 'D',
            'È' | 'É' | 'Ê' | 'Ë' | 'Ē' | 'Ĕ' | 'Ė' | 'Ę' | 'Ě' => 'E',
            'Ì' | 'Í' | 'Î' | 'Ï' | 'Ĩ' | 'Ī' | 'Ĭ' | 'Į' => 'I',
            'Ñ' | 'Ń' | 'Ņ' | 'Ň' => 'N',
            'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' | 'Ø' | 'Ō' | 'Ŏ' | 'Ő' => 'O',
            'Ù' | 'Ú' | 'Û' | 'Ü' | 'Ũ' | 'Ū' | 'Ŭ' | 'Ů' | 'Ű' | 'Ų' => 'U',
            other => other,
        })
        .collect()
}

fn render_plain_chat(messages: &[(&str, &str)], add_generation_prompt: bool) -> String {
    let mut rendered = String::new();
    for (role, content) in messages {
        rendered.push_str(role);
        rendered.push_str(": ");
        rendered.push_str(content);
        rendered.push('\n');
    }
    if add_generation_prompt {
        rendered.push_str("assistant: ");
    }
    rendered
}

fn render_llama_inst_chat(messages: &[(&str, &str)], add_generation_prompt: bool) -> String {
    let mut rendered = String::new();
    let mut system = None;
    for (role, content) in messages {
        match *role {
            "system" => system = Some(*content),
            "user" => {
                rendered.push_str("<s>[INST] ");
                if let Some(system) = system.take() {
                    rendered.push_str("<<SYS>>\n");
                    rendered.push_str(system);
                    rendered.push_str("\n<</SYS>>\n\n");
                }
                rendered.push_str(content);
                rendered.push_str(" [/INST]");
            }
            "assistant" => {
                rendered.push(' ');
                rendered.push_str(content);
                rendered.push_str(" </s>");
            }
            _ => {
                rendered.push(' ');
                rendered.push_str(content);
            }
        }
    }
    if add_generation_prompt && !rendered.ends_with("[/INST]") {
        rendered.push_str(" [INST] [/INST]");
    }
    rendered
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgufByteOrder, GgufValue};

    #[test]
    fn test_byte_tokenizer_roundtrip_ascii() {
        let tokenizer = ByteTokenizer;
        let tokens = tokenizer.encode("hello");
        assert_eq!(tokens, vec![104, 101, 108, 108, 111]);
        assert_eq!(tokenizer.decode(&tokens), "hello");
    }

    #[test]
    fn test_byte_tokenizer_replaces_out_of_range_ids() {
        let tokenizer = ByteTokenizer;
        assert_eq!(tokenizer.decode(&[65, 300]), "A?");
    }

    #[test]
    fn test_gguf_tokenizer_uses_vocab_metadata() {
        let mut metadata = ModelMetadata {
            version: 3,
            tensor_count: 0,
            kv_count: 0,
            byte_order: GgufByteOrder::LittleEndian,
            alignment: 32,
            tensor_data_offset: 0,
            metadata: HashMap::new(),
            tensors: Vec::new(),
        };
        metadata.metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::StringVec(vec!["h".to_string(), "he".to_string(), "llo".to_string()]),
        );

        let tokenizer = RuntimeTokenizer::from_metadata(Some(&metadata));
        assert_eq!(tokenizer.encode("hello"), vec![1, 2]);
        assert_eq!(tokenizer.decode(&[1, 2]), "hello");
    }

    #[test]
    fn test_gguf_tokenizer_uses_bpe_merges_and_special_ids() {
        let mut metadata = ModelMetadata {
            version: 3,
            tensor_count: 0,
            kv_count: 0,
            byte_order: GgufByteOrder::LittleEndian,
            alignment: 32,
            tensor_data_offset: 0,
            metadata: HashMap::new(),
            tensors: Vec::new(),
        };
        metadata.metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::StringVec(vec![
                "<unk>".to_string(),
                "h".to_string(),
                "e".to_string(),
                "he".to_string(),
                "l".to_string(),
                "o".to_string(),
                "ll".to_string(),
                "llo".to_string(),
                "hello".to_string(),
            ]),
        );
        metadata.metadata.insert(
            "tokenizer.ggml.merges".to_string(),
            GgufValue::StringVec(vec![
                "h e".to_string(),
                "l l".to_string(),
                "ll o".to_string(),
                "he llo".to_string(),
            ]),
        );
        metadata.metadata.insert(
            "tokenizer.ggml.unknown_token_id".to_string(),
            GgufValue::U32(0),
        );

        let tokenizer = RuntimeTokenizer::from_metadata(Some(&metadata));
        assert_eq!(tokenizer.encode("hello"), vec![8]);
        assert_eq!(tokenizer.encode("!"), vec![0]);
    }

    #[test]
    fn test_runtime_tokenizer_renders_chatml_template() {
        let mut metadata = ModelMetadata {
            version: 3,
            tensor_count: 0,
            kv_count: 0,
            byte_order: GgufByteOrder::LittleEndian,
            alignment: 32,
            tensor_data_offset: 0,
            metadata: HashMap::new(),
            tensors: Vec::new(),
        };
        metadata.metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::StringVec(vec!["<|im_start|>".to_string()]),
        );
        metadata.metadata.insert(
            "tokenizer.chat_template".to_string(),
            GgufValue::String("<|im_start|>{{ role }}".to_string()),
        );

        let tokenizer = RuntimeTokenizer::from_metadata(Some(&metadata));
        let rendered = tokenizer.render_chat_messages([("user", "hello")], true);
        assert_eq!(
            rendered,
            "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn test_tokenizer_applies_normalizer_and_byte_level_prefix_space() {
        let mut metadata = ModelMetadata {
            version: 3,
            tensor_count: 0,
            kv_count: 0,
            byte_order: GgufByteOrder::LittleEndian,
            alignment: 32,
            tensor_data_offset: 0,
            metadata: HashMap::new(),
            tensors: Vec::new(),
        };
        metadata.metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            GgufValue::StringVec(vec![
                "<unk>".to_string(),
                "Ġcafe".to_string(),
                "Ġ".to_string(),
                "c".to_string(),
                "a".to_string(),
                "f".to_string(),
                "e".to_string(),
            ]),
        );
        metadata.metadata.insert(
            "tokenizer.ggml.unknown_token_id".to_string(),
            GgufValue::U32(0),
        );
        metadata.metadata.insert(
            "nexus.tokenizer.normalizer.lowercase".to_string(),
            GgufValue::Bool(true),
        );
        metadata.metadata.insert(
            "nexus.tokenizer.normalizer.strip_accents".to_string(),
            GgufValue::Bool(true),
        );
        metadata.metadata.insert(
            "nexus.tokenizer.pre_tokenizer.byte_level".to_string(),
            GgufValue::Bool(true),
        );
        metadata.metadata.insert(
            "nexus.tokenizer.pre_tokenizer.add_prefix_space".to_string(),
            GgufValue::Bool(true),
        );

        let tokenizer = RuntimeTokenizer::from_metadata(Some(&metadata));
        assert_eq!(tokenizer.encode("Café"), vec![1]);
        assert_eq!(tokenizer.decode(&[1]), " cafe");
    }
}
