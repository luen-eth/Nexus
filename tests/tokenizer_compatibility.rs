mod common;
use common::{TinyModel, Value};
use nexus::engine::{InferenceConfig, InferenceEngine};
use serde_json::Value as Json;

fn loaded_hf(family: &Json) -> (InferenceEngine, tempfile::NamedTempFile) {
    let name = family["name"].as_str().unwrap();
    let mut model = TinyModel::new();
    model.resize_vocab(family["vocab_size"].as_u64().unwrap() as usize + 16);
    let json = family
        .get("tokenizer_json")
        .and_then(Json::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| {
            std::fs::read_to_string(format!(
                "{}/tests/fixtures/tokenizers/{name}.json",
                env!("CARGO_MANIFEST_DIR")
            ))
            .unwrap()
        });
    model
        .metadata
        .insert("nexus.tokenizer.json".into(), Value::Text(json));
    model.metadata.insert(
        "nexus.tokenizer.config".into(),
        Value::Text(family["config"].to_string()),
    );
    if let Some(template) = family["config"]["chat_template"].as_str() {
        model.metadata.insert(
            "tokenizer.chat_template".into(),
            Value::Text(template.into()),
        );
    }
    let file = tempfile::NamedTempFile::new().unwrap();
    model.write(file.path());
    let mut engine = InferenceEngine::new(None);
    engine.load_model(file.path()).unwrap();
    (engine, file)
}
fn families() -> Json {
    serde_json::from_str(include_str!("fixtures/tokenizers/golden.json")).unwrap()
}
#[test]
fn imported_hf_pipelines_match_python_token_ids_and_unicode_decoding() {
    for family in families()["families"].as_array().unwrap() {
        let (engine, _file) = loaded_hf(family);
        let tokenizer = engine.tokenizer.as_ref().unwrap();
        assert_eq!(tokenizer.kind(), "huggingface-json");
        for case in family["cases"].as_array().unwrap() {
            let text = case["text"].as_str().unwrap();
            let expected: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
            let fragments: Vec<u32> = serde_json::from_value(case["fragment_ids"].clone()).unwrap();
            assert_eq!(
                tokenizer.try_encode(text).unwrap(),
                expected,
                "{} {text:?}",
                family["name"]
            );
            assert_eq!(tokenizer.encode_fragment(text).unwrap(), fragments);
            assert_eq!(
                tokenizer.try_decode(&expected).unwrap(),
                case["decoded"].as_str().unwrap()
            );
        }
    }
}
#[test]
fn chat_templates_match_python_jinja_and_do_not_add_a_second_bos() {
    for family in families()["families"].as_array().unwrap() {
        let (engine, _file) = loaded_hf(family);
        let tokenizer = engine.tokenizer.as_ref().unwrap();
        let messages = family["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| (m["role"].as_str().unwrap(), m["content"].as_str().unwrap()));
        let rendered = tokenizer.try_render_chat_messages(messages, true).unwrap();
        assert_eq!(rendered, family["rendered"].as_str().unwrap());
        let expected: Vec<u32> = serde_json::from_value(family["chat_ids"].clone()).unwrap();
        assert_eq!(tokenizer.encode_fragment(&rendered).unwrap(), expected);
    }
}
#[test]
fn absent_or_invalid_tokenizers_fail_atomically_and_padded_ids_are_masked() {
    let family = &families()["families"][0];
    let (mut engine, _file) = loaded_hf(family);
    let original = engine.prefill(&[3]).unwrap();
    for invalid in [None, Some("not-json")] {
        let mut model = TinyModel::new();
        if let Some(json) = invalid {
            model
                .metadata
                .insert("nexus.tokenizer.json".into(), Value::Text(json.into()));
        }
        let file = tempfile::NamedTempFile::new().unwrap();
        model.write(file.path());
        assert!(engine.load_model(file.path()).is_err());
        assert_eq!(engine.prefill(&[3]).unwrap(), original);
    }
    let tokenizer = engine.tokenizer.as_ref().unwrap();
    let mut logits = vec![1.; engine.vocab_size];
    tokenizer.mask_logits(&mut logits);
    let invalid_id = engine.vocab_size - 1;
    assert!(!tokenizer.valid_id(invalid_id as u32));
    assert_eq!(logits[invalid_id], f32::NEG_INFINITY);
    assert!(engine.generate(&[invalid_id as u32], 1).is_err());
}
#[test]
fn template_errors_are_reported_and_all_configured_eos_ids_are_recognized() {
    let family = &families()["families"][1];
    let (engine, file) = loaded_hf(family);
    let mut model = TinyModel::new();
    model.resize_vocab(engine.vocab_size);
    let json = std::fs::read_to_string(format!(
        "{}/tests/fixtures/tokenizers/byte-bpe.json",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    model
        .metadata
        .insert("nexus.tokenizer.json".into(), Value::Text(json));
    model.metadata.insert(
        "nexus.generation.config".into(),
        Value::Text(r#"{"eos_token_id":[1,2]}"#.into()),
    );
    model.metadata.insert(
        "tokenizer.chat_template".into(),
        Value::Text("{{ raise_exception('roles must alternate') }}".into()),
    );
    model.write(file.path());
    let mut engine = InferenceEngine::new(None);
    engine.load_model(file.path()).unwrap();
    let t = engine.tokenizer.as_ref().unwrap();
    assert!(t.is_eog(1));
    assert!(t.is_eog(2));
    assert!(t
        .try_render_chat_messages([("user", "hello")], true)
        .unwrap_err()
        .contains("roles must alternate"));
}
#[test]
fn native_gguf_sentencepiece_accumulates_byte_fallback_and_controls() {
    let mut model = TinyModel::new();
    let mut tokens = vec![
        "<unk>".into(),
        "<s>".into(),
        "</s>".into(),
        "▁".into(),
        "a".into(),
        "b".into(),
        "c".into(),
        "hello".into(),
    ];
    tokens.extend((0..256).map(|b| format!("<0x{b:02X}>")));
    let mut types = vec![2, 3, 3, 1, 1, 1, 1, 1];
    types.extend(vec![6; 256]);
    model.resize_vocab(tokens.len());
    model
        .metadata
        .insert("tokenizer.ggml.model".into(), Value::Text("llama".into()));
    model
        .metadata
        .insert("tokenizer.ggml.tokens".into(), Value::Strings(tokens));
    model.metadata.insert(
        "tokenizer.ggml.scores".into(),
        Value::Floats(vec![-1.; 264]),
    );
    model
        .metadata
        .insert("tokenizer.ggml.token_type".into(), Value::Ints(types));
    for (key, id) in [
        ("bos_token_id", 1),
        ("eos_token_id", 2),
        ("unknown_token_id", 0),
    ] {
        model
            .metadata
            .insert(format!("tokenizer.ggml.{key}"), Value::Uint(id));
    }
    model
        .metadata
        .insert("tokenizer.ggml.add_bos_token".into(), Value::Bool(true));
    model
        .metadata
        .insert("tokenizer.ggml.add_space_prefix".into(), Value::Bool(true));
    model.metadata.insert(
        "tokenizer.chat_template".into(),
        Value::Text("{{ bos_token }}{{ messages[0]['content'] }}".into()),
    );
    let file = tempfile::NamedTempFile::new().unwrap();
    model.write(file.path());
    let mut engine = InferenceEngine::new(Some(InferenceConfig::default()));
    engine.load_model(file.path()).unwrap();
    let tokenizer = engine.tokenizer.as_ref().unwrap();
    assert_eq!(tokenizer.kind(), "llama.cpp-gguf");
    let ids = tokenizer.try_encode("hello 🙂世界").unwrap();
    assert_eq!(ids[0], 1);
    let decoded = tokenizer.try_decode(&ids).unwrap();
    // llama_detokenize preserves the inserted space after removing an explicit BOS.
    assert_eq!(decoded, " hello 🙂世界");
    assert!(tokenizer.try_decode(&[]).unwrap().is_empty());
    let rendered = tokenizer
        .try_render_chat_messages([("user", "hello")], true)
        .unwrap();
    let ids = tokenizer.encode_fragment(&rendered).unwrap();
    assert_eq!(ids.iter().filter(|&&t| t == 1).count(), 1);
}

#[test]
fn serialized_padding_and_truncation_do_not_change_inference_prompts() {
    let mut family = families()["families"][0].clone();
    let raw = include_str!("fixtures/tokenizers/wordpiece.json");
    let mut json: Json = serde_json::from_str(raw).unwrap();
    json["truncation"] = serde_json::json!({"direction":"Right","max_length":4,"strategy":"LongestFirst","stride":0});
    json["padding"] = serde_json::json!({"strategy":{"Fixed":16},"direction":"Right","pad_to_multiple_of":null,"pad_id":0,"pad_type_id":0,"pad_token":"[UNK]"});
    family["tokenizer_json"] = json.to_string().into();
    let (engine, _file) = loaded_hf(&family);
    let tokenizer = engine.tokenizer.as_ref().unwrap();
    assert_eq!(
        tokenizer
            .try_encode("hello world hello world hello world")
            .unwrap(),
        vec![1, 3, 4, 3, 4, 3, 4, 2]
    );
    assert_eq!(tokenizer.try_encode("hello").unwrap(), vec![1, 3, 2]);
}
#[test]
fn named_chat_templates_select_default_and_reject_ambiguity() {
    let mut family = families()["families"][0].clone();
    for templates in [
        serde_json::json!([{"name":"tool_use","template":"wrong"},{"name":"default","template":"{{ messages[0].content }}"}]),
        serde_json::json!({"tool_use":"wrong","default":"{{ messages[0].content }}"}),
    ] {
        family["config"]["chat_template"] = templates;
        let (engine, _file) = loaded_hf(&family);
        assert_eq!(
            engine
                .tokenizer
                .as_ref()
                .unwrap()
                .try_render_chat_messages([("user", "hello")], true)
                .unwrap(),
            "hello"
        );
    }
    assert!(nexus::tokenizer::exact::select_chat_template(
        &serde_json::json!({"tool_use":"wrong"})
    )
    .is_err());
}
#[test]
fn template_json_matches_python_without_html_escaping() {
    let golden: Json = serde_json::from_str(include_str!(
        "fixtures/tokenizers/json-template-golden.json"
    ))
    .unwrap();
    for case in golden.as_array().unwrap() {
        let mut family = families()["families"][0].clone();
        family["config"]["chat_template"] = case["template"].clone();
        let (engine, _file) = loaded_hf(&family);
        let actual = engine
            .tokenizer
            .as_ref()
            .unwrap()
            .try_render_chat_messages([("user", "<>&' Türkçe 🙂")], true)
            .unwrap();
        assert_eq!(actual, case["rendered"].as_str().unwrap());
    }
}

#[test]
fn malformed_eos_and_tokenizer_configs_fail_without_replacing_a_valid_model() {
    let family = &families()["families"][0];
    let (mut engine, file) = loaded_hf(family);
    let original = engine.prefill(&[3]).unwrap();
    let tokenizer_json = std::fs::read_to_string(format!(
        "{}/tests/fixtures/tokenizers/wordpiece.json",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap();
    for (key, value) in [
        ("tokenizer.ggml.eos_token_id", Value::Text("wrong".into())),
        (
            "nexus.generation.config",
            Value::Text(r#"{"eos_token_id":[1,"wrong"]}"#.into()),
        ),
        (
            "nexus.generation.config",
            Value::Text(r#"{"eos_token_id":-1}"#.into()),
        ),
        (
            "nexus.generation.config",
            Value::Text(r#"{"eos_token_id":4294967296}"#.into()),
        ),
        ("nexus.generation.config", Value::Text("[]".into())),
        ("nexus.tokenizer.config", Value::Text("[]".into())),
    ] {
        let mut invalid = TinyModel::new();
        invalid.resize_vocab(engine.vocab_size);
        invalid.metadata.insert(
            "nexus.tokenizer.json".into(),
            Value::Text(tokenizer_json.clone()),
        );
        invalid.metadata.insert(key.into(), value);
        invalid.write(file.path());
        assert!(engine.load_model(file.path()).is_err(), "accepted {key}");
        assert_eq!(engine.prefill(&[3]).unwrap(), original);
    }
}

#[test]
fn inconsistent_native_tokenizer_arrays_are_rejected_before_native_loading() {
    let mut model = TinyModel::new();
    model
        .metadata
        .insert("tokenizer.ggml.model".into(), Value::Text("llama".into()));
    model.metadata.insert(
        "tokenizer.ggml.tokens".into(),
        Value::Strings(vec!["<unk>".into(); 8]),
    );
    model
        .metadata
        .insert("tokenizer.ggml.scores".into(), Value::Floats(vec![0.; 7]));
    let file = tempfile::NamedTempFile::new().unwrap();
    model.write(file.path());
    let mut engine = InferenceEngine::new(None);
    assert!(engine
        .load_model(file.path())
        .unwrap_err()
        .to_string()
        .contains("tokenizer array"));
    assert!(engine.model.is_none());
}
