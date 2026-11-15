//! Standards-based tokenizer execution for loaded model files.
use crate::gguf::ModelMetadata;
use llama_cpp_2::{
    llama_backend::LlamaBackend,
    model::{params::LlamaModelParams, LlamaModel},
    token::LlamaToken,
};
use std::path::Path;
use std::sync::{Arc, OnceLock};

static LLAMA_BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();
#[derive(Debug)]
enum Implementation {
    Hf(Box<tokenizers::Tokenizer>),
    Gguf(Arc<LlamaModel>),
}
#[derive(Debug)]
pub struct ExactTokenizer {
    implementation: Implementation,
    template: Option<String>,
    context: serde_json::Map<String, serde_json::Value>,
    eos_ids: Vec<u32>,
    valid_ids: Vec<bool>,
    prefix_stable: bool,
}
impl ExactTokenizer {
    pub fn load(path: &Path, metadata: &ModelMetadata, vocab_size: usize) -> Result<Self, String> {
        for key in [
            "bos_token_id",
            "eos_token_id",
            "unknown_token_id",
            "padding_token_id",
            "separator_token_id",
            "eot_token_id",
            "eom_token_id",
        ] {
            let key = format!("tokenizer.ggml.{key}");
            if metadata.metadata.contains_key(&key)
                && metadata.get_usize(&key).is_none_or(|id| id >= vocab_size)
            {
                return Err(format!("Invalid or out-of-range special token: {key}"));
            }
        }
        if metadata.get_str("nexus.tokenizer.json").is_none() {
            let token_count = metadata
                .metadata
                .get("tokenizer.ggml.tokens")
                .and_then(|v| v.as_string_vec())
                .map(|v| v.len())
                .ok_or("GGUF tokenizer vocabulary is missing")?;
            if token_count == 0 || token_count > vocab_size {
                return Err("GGUF tokenizer vocabulary disagrees with embedding rows".into());
            }
            for (key, valid) in [
                ("tokenizer.ggml.scores", metadata.metadata.get("tokenizer.ggml.scores").is_none_or(|v| matches!(v, crate::gguf::GgufValue::F32Vec(values) if values.len() == token_count))),
                ("tokenizer.ggml.token_type", metadata.metadata.get("tokenizer.ggml.token_type").is_none_or(|v| matches!(v, crate::gguf::GgufValue::I32Vec(values) if values.len() == token_count))),
            ] {
                if !valid { return Err(format!("Invalid GGUF tokenizer array: {key}")); }
            }
            for key in [
                "bos_token_id",
                "eos_token_id",
                "unknown_token_id",
                "padding_token_id",
                "separator_token_id",
                "eot_token_id",
                "eom_token_id",
            ] {
                if metadata
                    .get_usize(&format!("tokenizer.ggml.{key}"))
                    .is_some_and(|id| id >= token_count)
                {
                    return Err(format!(
                        "GGUF special token {key} exceeds tokenizer vocabulary"
                    ));
                }
            }
        }
        let implementation = if let Some(json) = metadata.get_str("nexus.tokenizer.json") {
            let mut tokenizer = tokenizers::Tokenizer::from_bytes(json.as_bytes())
                .map_err(|e| format!("Invalid embedded HF tokenizer: {e}"))?;
            // Inference owns the context budget; serialized training/batch settings
            // must never silently truncate prompts or append padding.
            tokenizer.with_truncation(None).map_err(|e| e.to_string())?;
            tokenizer.with_padding(None);
            if tokenizer.get_vocab_size(true) == 0 {
                return Err("Tokenizer vocabulary is empty".into());
            }
            if tokenizer
                .get_vocab(true)
                .values()
                .any(|&id| id as usize >= vocab_size)
            {
                return Err("Tokenizer ID exceeds embedding vocabulary".into());
            }
            Implementation::Hf(Box::new(tokenizer))
        } else {
            if metadata.get_str("tokenizer.ggml.model").is_none() {
                return Err(
                    "Model has no complete tokenizer; provide tokenizer.json when converting"
                        .into(),
                );
            }
            let backend = LLAMA_BACKEND
                .get_or_init(|| LlamaBackend::init().map_err(|e| e.to_string()))
                .as_ref()
                .map_err(Clone::clone)?;
            let model = LlamaModel::load_from_file(
                backend,
                path,
                &LlamaModelParams::default()
                    .with_vocab_only(true)
                    .with_n_gpu_layers(0)
                    .with_use_mmap(false),
            )
            .map_err(|e| format!("GGUF tokenizer load failed: {e}"))?;
            if model.n_vocab() < 0 || model.n_vocab() as usize > vocab_size {
                return Err("GGUF vocabulary exceeds embedding rows".into());
            }
            Implementation::Gguf(Arc::new(model))
        };
        let valid_ids = match &implementation {
            Implementation::Hf(t) => {
                let mut valid = vec![false; vocab_size];
                for id in t.get_vocab(true).values() {
                    valid[*id as usize] = true;
                }
                valid
            }
            Implementation::Gguf(m) => (0..vocab_size)
                .map(|id| id < m.n_vocab() as usize)
                .collect(),
        };
        let prefix_stable = metadata
            .get_str("nexus.tokenizer.json")
            .map(|json| {
                let value: serde_json::Value = serde_json::from_str(json).unwrap_or_default();
                matches!(
                    value.pointer("/decoder/type").and_then(|v| v.as_str()),
                    Some("ByteLevel")
                )
            })
            .unwrap_or_else(|| native_prefix_stable(metadata));
        let config: serde_json::Value = metadata
            .get_str("nexus.tokenizer.config")
            .map(serde_json::from_str)
            .transpose()
            .map_err(|e| format!("Invalid tokenizer config: {e}"))?
            .unwrap_or_default();
        if !config.is_null() && !config.is_object() {
            return Err("Tokenizer config must be an object".into());
        }
        let mut context = config.as_object().cloned().unwrap_or_default();
        for (key, id_key) in [
            ("bos_token", "tokenizer.ggml.bos_token_id"),
            ("eos_token", "tokenizer.ggml.eos_token_id"),
            ("unk_token", "tokenizer.ggml.unknown_token_id"),
            ("pad_token", "tokenizer.ggml.padding_token_id"),
        ] {
            let configured = context
                .get(key)
                .and_then(|v| {
                    v.as_str()
                        .or_else(|| v.get("content").and_then(|v| v.as_str()))
                })
                .map(str::to_owned);
            let piece = configured
                .or_else(|| {
                    metadata.get_usize(id_key).and_then(|id| {
                        metadata
                            .metadata
                            .get("tokenizer.ggml.tokens")?
                            .as_string_vec()?
                            .get(id)
                            .cloned()
                    })
                })
                .unwrap_or_default();
            context.insert(key.into(), serde_json::Value::String(piece));
        }
        let mut eos_ids = Vec::new();
        for key in [
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.eot_token_id",
            "tokenizer.ggml.eom_token_id",
        ] {
            if let Some(id) = metadata.get_usize(key).and_then(|v| u32::try_from(v).ok()) {
                eos_ids.push(id);
            }
        }
        for key in ["nexus.generation.config", "nexus.model.config"] {
            if let Some(json) = metadata.get_str(key) {
                let cfg: serde_json::Value = serde_json::from_str(json)
                    .map_err(|e| format!("Invalid generation config: {e}"))?;
                let config_object = cfg
                    .as_object()
                    .ok_or("Generation/model config must be an object")?;
                let parse_id = |id: &serde_json::Value| {
                    id.as_u64()
                        .and_then(|id| u32::try_from(id).ok())
                        .ok_or("EOS IDs must be nonnegative 32-bit integers")
                };
                match config_object.get("eos_token_id") {
                    None | Some(serde_json::Value::Null) => {}
                    Some(serde_json::Value::Array(ids)) => {
                        for id in ids {
                            eos_ids.push(parse_id(id)?);
                        }
                    }
                    Some(id) => eos_ids.push(parse_id(id)?),
                }
            }
        }
        if eos_ids
            .iter()
            .any(|&id| !valid_ids.get(id as usize).copied().unwrap_or(false))
        {
            return Err("EOS ID exceeds embedding vocabulary".into());
        }
        let template = match metadata.get_str("tokenizer.chat_template") {
            Some(template) => Some(template.to_owned()),
            None => select_chat_template(
                config
                    .get("chat_template")
                    .unwrap_or(&serde_json::Value::Null),
            )?,
        };
        if let Some(template) = &template {
            let mut env = template_environment();
            env.add_template("chat", template)
                .map_err(|e| format!("Invalid chat template: {e}"))?;
        }
        Ok(Self {
            implementation,
            template,
            context,
            eos_ids,
            valid_ids,
            prefix_stable,
        })
    }
    pub fn valid_id(&self, id: u32) -> bool {
        self.valid_ids.get(id as usize).copied().unwrap_or(false)
    }
    pub fn mask_logits(&self, logits: &mut [f32]) {
        for (id, value) in logits.iter_mut().enumerate() {
            if !self.valid_ids.get(id).copied().unwrap_or(false) {
                *value = f32::NEG_INFINITY;
            }
        }
    }
    pub fn prefix_stable(&self) -> bool {
        self.prefix_stable
    }
    pub fn kind(&self) -> &'static str {
        match self.implementation {
            Implementation::Hf(_) => "huggingface-json",
            Implementation::Gguf(_) => "llama.cpp-gguf",
        }
    }
    pub fn encode(&self, text: &str, add_special: bool) -> Result<Vec<u32>, String> {
        match &self.implementation {
            Implementation::Hf(tokenizer) => tokenizer
                .encode(text, add_special)
                .map(|encoding| encoding.get_ids().to_vec())
                .map_err(|e| format!("Tokenization failed: {e}")),
            Implementation::Gguf(model) => Ok(model
                .vocab()
                .tokenize(text.as_bytes(), add_special, true)
                .iter()
                .map(|t| t.0 as u32)
                .collect()),
        }
    }
    pub fn decode(&self, tokens: &[u32]) -> Result<String, String> {
        if tokens.is_empty() {
            return Ok(String::new());
        }
        if tokens.iter().any(|&id| !self.valid_id(id)) {
            return Err("Token ID is outside tokenizer vocabulary".into());
        }
        match &self.implementation {
            Implementation::Hf(tokenizer) => tokenizer
                .decode(tokens, true)
                .map_err(|e| format!("Detokenization failed: {e}")),
            Implementation::Gguf(model) => {
                if tokens.iter().any(|&t| t >= model.n_vocab() as u32) {
                    return Err("Invalid token ID for GGUF decoding".into());
                }
                let tokens: Vec<_> = tokens.iter().map(|&t| LlamaToken(t as i32)).collect();
                Ok(
                    String::from_utf8_lossy(&model.vocab().detokenize(&tokens, true, false))
                        .into_owned(),
                )
            }
        }
    }
    pub fn is_eog(&self, token: u32) -> bool {
        self.eos_ids.contains(&token)
            || match &self.implementation {
                Implementation::Gguf(model) if token < (model.n_vocab() as u32) => {
                    model.vocab().is_eog(LlamaToken(token as i32))
                }
                _ => false,
            }
    }
    pub fn render(
        &self,
        messages: &[(&str, &str)],
        add_generation_prompt: bool,
    ) -> Result<String, String> {
        let template=self.template.as_ref().ok_or("Model has no chat template; use text completions or provide a template when converting")?;
        let mut env = template_environment();
        env.add_template("chat", template)
            .map_err(|e| format!("Chat template: {e}"))?;
        let mut context = self.context.clone();
        context.insert(
            "messages".into(),
            serde_json::Value::Array(
                messages
                    .iter()
                    .map(|(role, content)| serde_json::json!({"role":role,"content":content}))
                    .collect(),
            ),
        );
        context.insert("add_generation_prompt".into(), add_generation_prompt.into());
        env.get_template("chat")
            .map_err(|e| e.to_string())?
            .render(minijinja::value::Serde(context))
            .map_err(|e| format!("Chat template: {e}"))
    }
}
fn template_environment() -> minijinja::Environment<'static> {
    let mut env = minijinja::Environment::new();
    env.set_syntax(
        minijinja::syntax::SyntaxConfig::builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .build()
            .expect("Static Jinja syntax"),
    );
    env.set_fuel(Some(1_000_000));
    env.add_filter("tojson", super::template_json::tojson);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |message: String| -> Result<String, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                message,
            ))
        },
    );
    env.add_function(
        "strftime_now",
        |format: String| -> Result<String, minijinja::Error> {
            if chrono::format::StrftimeItems::new(&format)
                .any(|item| matches!(item, chrono::format::Item::Error))
            {
                return Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    "Invalid date format",
                ));
            }
            Ok(chrono::Utc::now().format(&format).to_string())
        },
    );
    env
}

// Native BPE cleanup can revise punctuation/quotes already decoded. Only
// explicitly verified profiles may stream incrementally; others buffer safely.
fn native_prefix_stable(metadata: &ModelMetadata) -> bool {
    match metadata.get_str("tokenizer.ggml.model") {
        Some("llama") => true,
        Some("gpt2") => matches!(
            metadata.get_str("tokenizer.ggml.pre"),
            Some("qwen2" | "deepseek-r1-qwen")
        ),
        _ => false,
    }
}

/// Select the ordinary chat template. Named tool templates are not chosen
/// implicitly: ambiguous collections require a `default` entry.
pub fn select_chat_template(value: &serde_json::Value) -> Result<Option<String>, String> {
    use serde_json::Value;
    match value {
        Value::Null => Ok(None),
        Value::String(template) => Ok(Some(template.clone())),
        Value::Array(entries) => {
            let default = entries
                .iter()
                .find(|entry| entry["name"] == "default")
                .and_then(|entry| entry.get("template"));
            match default {
                Some(Value::String(template)) => Ok(Some(template.clone())),
                _ => Err("Named chat templates require a default template".into()),
            }
        }
        Value::Object(entries) => {
            if let Some(template) = entries.get("chat_template") {
                return select_chat_template(template);
            }
            match entries.get("default") {
                Some(Value::String(template)) => Ok(Some(template.clone())),
                _ => Err("Named chat templates require a default template".into()),
            }
        }
        _ => Err("Unsupported chat template format".into()),
    }
}
