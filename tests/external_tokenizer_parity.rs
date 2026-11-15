//! Explicitly run with --ignored after preparing pinned external reference assets.
use nexus::{
    gguf::{GgufByteOrder, GgufReader, GgufValue, ModelMetadata},
    tokenizer::RuntimeTokenizer,
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
fn base() -> PathBuf {
    std::env::var_os("NEXUS_REFERENCE_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap()).join(".cache/nexus-reference-models")
        })
}
fn empty() -> ModelMetadata {
    ModelMetadata {
        version: 3,
        tensor_count: 0,
        kv_count: 0,
        byte_order: GgufByteOrder::LittleEndian,
        alignment: 32,
        tensor_data_offset: 0,
        metadata: HashMap::new(),
        tensors: vec![],
    }
}
#[test]
#[ignore = "requires downloaded pinned model/tokenizer fixtures; see tests/reference"]
fn real_hf_tokenizers_and_chat_templates_match_python_goldens() {
    for name in ["smollm", "qwen", "llama3"] {
        let root = base().join(name);
        let mut metadata = empty();
        for (file, key) in [
            ("tokenizer.json", "nexus.tokenizer.json"),
            ("tokenizer_config.json", "nexus.tokenizer.config"),
            ("config.json", "nexus.model.config"),
            ("generation_config.json", "nexus.generation.config"),
        ] {
            metadata.metadata.insert(
                key.into(),
                GgufValue::String(std::fs::read_to_string(root.join(file)).unwrap()),
            );
        }
        let config: serde_json::Value =
            serde_json::from_str(metadata.get_str("nexus.tokenizer.config").unwrap()).unwrap();
        metadata.metadata.insert(
            "tokenizer.chat_template".into(),
            GgufValue::String(config["chat_template"].as_str().unwrap().into()),
        );
        let model_config: serde_json::Value =
            serde_json::from_str(metadata.get_str("nexus.model.config").unwrap()).unwrap();
        let t = RuntimeTokenizer::from_model(
            Path::new("unused"),
            &metadata,
            model_config["vocab_size"].as_u64().unwrap() as usize,
            false,
        )
        .unwrap();
        let golden: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(root.join("golden.json")).unwrap())
                .unwrap();
        for case in golden["cases"].as_array().unwrap() {
            let text = case["text"].as_str().unwrap();
            let expected: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
            assert_eq!(t.try_encode(text).unwrap(), expected, "{name} {text:?}");
            assert_eq!(
                t.try_decode(&expected).unwrap(),
                case["decoded"].as_str().unwrap(),
                "{name}"
            );
            let fragments: Vec<u32> = serde_json::from_value(case["fragment_ids"].clone()).unwrap();
            assert_eq!(t.encode_fragment(text).unwrap(), fragments);
        }
        let turns = golden["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| (m["role"].as_str().unwrap(), m["content"].as_str().unwrap()));
        let rendered = t.try_render_chat_messages(turns, true).unwrap();
        assert_eq!(rendered, golden["rendered"].as_str().unwrap(), "{name}");
        let ids: Vec<u32> = serde_json::from_value(golden["chat_ids"].clone()).unwrap();
        assert_eq!(t.encode_fragment(&rendered).unwrap(), ids, "{name}");
    }
}
#[test]
#[ignore = "requires upstream llama.cpp vocab fixtures and recorded C++ tokenizer outputs"]
fn original_gguf_vocabularies_match_llama_cpp_token_ids_and_bytes() {
    let models = std::env::var_os("NEXUS_LLAMA_VOCABS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/nexus-llama-reference/models"));
    for family in ["llama-spm", "llama-bpe", "qwen2", "bert-bge"] {
        let path = models.join(format!("ggml-vocab-{family}.gguf"));
        let metadata = GgufReader::parse_file(&path).unwrap();
        let vocab = metadata.metadata["tokenizer.ggml.tokens"]
            .as_string_vec()
            .unwrap()
            .len();
        let t = RuntimeTokenizer::from_model(&path, &metadata, vocab, false).unwrap();
        assert_eq!(t.prefix_stable(), matches!(family, "llama-spm" | "qwen2"));
        let golden: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(base().join(format!("gguf-{family}-golden.json"))).unwrap(),
        )
        .unwrap();
        for case in golden.as_array().unwrap() {
            let text = case["text"].as_str().unwrap();
            let ids: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
            assert_eq!(t.try_encode(text).unwrap(), ids, "{family} {text:?}");
            assert_eq!(
                t.try_decode(&ids).unwrap(),
                case["decoded"].as_str().unwrap(),
                "{family} {text:?}"
            );
        }
    }
}
