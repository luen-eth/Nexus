mod common;
use common::{TinyModel, Value};
use nexus::backend::BackendType;
use nexus::engine::{GenerationConfig, InferenceConfig, InferenceEngine};
use nexus::quant::KvQuantType;

fn engine(model: &TinyModel, mut config: InferenceConfig) -> InferenceEngine {
    config.allow_byte_tokenizer = true;
    let file = tempfile::NamedTempFile::new().unwrap();
    model.write(file.path());
    let mut engine = InferenceEngine::new(Some(config));
    engine.load_model(file.path()).unwrap();
    engine
}

#[test]
fn context_override_is_preserved_and_clamped_to_model_limit() {
    for (requested, expected) in [(8, 8), (64, 32)] {
        let engine = engine(
            &TinyModel::new(),
            InferenceConfig {
                max_seq_len: requested,
                ..Default::default()
            },
        );
        assert_eq!(engine.config.max_seq_len, expected);
        assert!(engine
            .kv_caches
            .iter()
            .all(|cache| cache.max_seq_len == expected));
    }
}

#[test]
fn mistral_architecture_is_accepted_with_identical_outputs() {
    // Dense Mistral graphs are identical to LLaMA; the same weights under
    // `mistral.*` metadata keys must load and produce the same logits.
    let mut llama = engine(&TinyModel::new(), InferenceConfig::default());

    let mut mistral_model = TinyModel::new();
    mistral_model.metadata = mistral_model
        .metadata
        .iter()
        .map(|(key, value)| {
            let renamed = key
                .strip_prefix("llama.")
                .map(|rest| format!("mistral.{rest}"))
                .unwrap_or_else(|| key.clone());
            (renamed, value.clone())
        })
        .collect();
    mistral_model
        .metadata
        .insert("general.architecture".into(), Value::Text("mistral".into()));
    let mut mistral = engine(&mistral_model, InferenceConfig::default());

    for prompt in [[0u32, 1, 2], [3, 0, 1]] {
        assert_eq!(
            llama.prefill(&prompt).unwrap(),
            mistral.prefill(&prompt).unwrap(),
            "logits must not depend on the metadata prefix"
        );
    }

    // The embedding-vs-vocab cross-check must read the architecture's own key:
    // a mistral file whose declared vocab disagrees with the embedding rows
    // has to fail exactly like a llama file (previously the check silently
    // skipped because it read only `llama.vocab_size`).
    mistral_model
        .metadata
        .insert("mistral.vocab_size".into(), Value::Uint(99));
    let file = tempfile::NamedTempFile::new().unwrap();
    mistral_model.write(file.path());
    let mut engine = InferenceEngine::new(Some(InferenceConfig::default()));
    let error = engine.load_model(file.path()).unwrap_err().to_string();
    assert!(
        error.contains("Vocabulary and embedding dimensions disagree"),
        "unexpected error: {error}"
    );
}

#[test]
fn invalid_loads_fail_without_replacing_the_working_model() {
    let mut engine = engine(&TinyModel::new(), InferenceConfig::default());
    let before = engine.prefill(&[0, 1]).unwrap();
    for defect in 0..6 {
        let mut model = TinyModel::new();
        match defect {
            0 => model.tensors.clear(),
            1 => model
                .tensors
                .retain(|(name, _, _)| name != "blk.0.attn_q.weight"),
            2 => {
                model
                    .metadata
                    .insert("llama.attention.head_count".into(), Value::Uint(0));
            }
            3 => {
                model
                    .metadata
                    .insert("general.architecture".into(), Value::Text("qwen2".into()));
            }
            4 => {
                // Target tensors by name: indexing the list makes the defect
                // silently migrate to another tensor if the fixture order
                // ever changes.
                let tensor = model
                    .tensors
                    .iter_mut()
                    .find(|(name, _, _)| name == "token_embd.weight")
                    .expect("fixture token_embd");
                tensor.2[0] = f32::NAN;
            }
            _ => {
                // Reverse a deliberately non-square tensor's dims: the
                // embedding is square at head_dim 4, while attn_k is
                // [hidden, kv_dim] and must fail the loader's shape check.
                let tensor = model
                    .tensors
                    .iter_mut()
                    .find(|(name, _, _)| name == "blk.0.attn_k.weight")
                    .expect("fixture attn_k");
                tensor.1.reverse();
            }
        }
        let file = tempfile::NamedTempFile::new().unwrap();
        model.write(file.path());
        assert!(engine.load_model(file.path()).is_err(), "defect {defect}");
        assert_eq!(engine.prefill(&[0, 1]).unwrap(), before);
    }
}

#[test]
fn required_weights_cannot_be_silently_capped() {
    let file = tempfile::NamedTempFile::new().unwrap();
    TinyModel::new().write(file.path());
    let mut engine = InferenceEngine::new(Some(InferenceConfig {
        max_loaded_tensor_elements: Some(4),
        ..Default::default()
    }));
    assert!(engine
        .load_model(file.path())
        .unwrap_err()
        .to_string()
        .contains("embedding"));
    assert!(engine.model.is_none());
}

#[test]
fn generation_never_synthesizes_missing_weights_or_wraps_token_ids() {
    let mut engine = engine(&TinyModel::new(), InferenceConfig::default());
    assert!(engine.generate(&[8], 1).unwrap_err().contains("vocabulary"));
    engine
        .model
        .as_mut()
        .unwrap()
        .tensors
        .remove("blk.0.attn_q.weight");
    assert!(engine
        .generate(&[0], 1)
        .unwrap_err()
        .contains("Missing tensor"));
}

#[test]
fn generation_validates_context_and_sampling_before_running() {
    let mut engine = engine(
        &TinyModel::new(),
        InferenceConfig {
            max_seq_len: 4,
            ..Default::default()
        },
    );
    assert!(engine.generate(&[], 1).is_err());
    assert!(engine.generate(&[0, 1], 3).is_err());
    assert!(engine.generate(&[0], usize::MAX).is_err());
    assert!(engine
        .generate_with_config(
            &[0],
            &GenerationConfig {
                max_tokens: 1,
                top_p: 0.,
                ..Default::default()
            }
        )
        .is_err());
    assert_eq!(engine.generate(&[0], 0).unwrap(), vec![0]);
}

#[test]
fn interleaved_sessions_match_isolated_generation_with_real_layer_caches() {
    for kv_quant in [KvQuantType::None, KvQuantType::TurboQuant3b2b] {
        let config = InferenceConfig {
            kv_quant,
            ..Default::default()
        };
        let mut engine = engine(&TinyModel::new(), config);
        let generation = GenerationConfig {
            max_tokens: 6,
            temperature: 0.7,
            seed: Some(42),
            ..Default::default()
        };
        let expected_a = engine.generate_with_config(&[0, 1], &generation).unwrap();
        let expected_b = engine
            .generate_with_config(&[2, 3, 4], &generation)
            .unwrap();
        let mut a = engine.start_session(&[0, 1], &generation).unwrap();
        let mut b = engine.start_session(&[2, 3, 4], &generation).unwrap();
        while !a.finished || !b.finished {
            engine.step_session(&mut a).unwrap();
            engine.step_session(&mut b).unwrap();
        }
        assert_eq!(a.output, expected_a);
        assert_eq!(b.output, expected_b);
    }
}

#[test]
fn eos_finishes_without_emitting_special_token() {
    let model = TinyModel::new();
    let mut engine = engine(&model, InferenceConfig::default());
    let first = engine.generate(&[0], 1).unwrap()[1];
    engine.model.as_mut().unwrap().metadata.metadata.insert(
        "tokenizer.ggml.eos_token_id".into(),
        nexus::gguf::GgufValue::U32(first),
    );
    let mut streamed = Vec::new();
    let output = engine
        .generate_with_config_streaming(
            &[0],
            &GenerationConfig {
                max_tokens: 4,
                ..Default::default()
            },
            |t, _| {
                streamed.push(t);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(output, vec![0]);
    assert!(streamed.is_empty());
}

#[test]
fn multi_token_stop_is_withheld_from_stream_and_final_output() {
    let mut engine = engine(&TinyModel::new(), InferenceConfig::default());
    let expected = engine.generate(&[0], 4).unwrap();
    let stop = expected[1..3].to_vec();
    let mut streamed = Vec::new();
    let output = engine
        .generate_with_config_streaming(
            &[0],
            &GenerationConfig {
                max_tokens: 4,
                stop_sequences: vec![stop],
                ..Default::default()
            },
            |t, _| {
                streamed.push(t);
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(output, vec![0]);
    assert!(streamed.is_empty());
}

#[test]
fn gpu_probe_is_not_advertised_as_an_inference_backend() {
    use nexus::backend::BackendFactory;
    assert_eq!(BackendType::detect(), BackendType::CpuSimd);
    assert!(BackendFactory::available_backends()
        .iter()
        .all(|cap| cap.backend == BackendType::CpuSimd
            || (cfg!(feature = "cuda") && cap.backend == BackendType::Cuda)
            || !cap.available));
    let file = tempfile::NamedTempFile::new().unwrap();
    TinyModel::new().write(file.path());
    let mut engine = InferenceEngine::new(Some(InferenceConfig {
        backend: BackendType::Metal,
        ..Default::default()
    }));
    assert!(engine
        .load_model(file.path())
        .unwrap_err()
        .to_string()
        .contains("not implemented"));
}

#[test]
fn one_layer_gqa_logits_match_an_independent_scalar_reference() {
    // Reference: separate Python float64 RMSNorm -> Q/K/V -> split-half RoPE ->
    // GQA softmax attention -> residual -> SwiGLU -> residual -> output norm/head.
    // Uses the deterministic F32 fixture with prompt [0, 1, 2] and unquantized KV.
    //
    // Pinned to head_dim 2 on purpose: the expected constants were produced by
    // that external script for this exact fixture, and at head_dim 2 the
    // split-half and adjacent-pair RoPE layouts coincide, so this test gates
    // the whole forward pass numerically. The RoPE *layout* itself is anchored
    // at head_dim 4 against llama.cpp in
    // `gguf_head_dim_four_logits_match_llama_cpp_cpu_f32`.
    let mut engine = engine(
        &TinyModel::with_head_dim(2),
        InferenceConfig {
            kv_quant: KvQuantType::None,
            ..Default::default()
        },
    );
    let actual = engine.prefill(&[0, 1, 2]).unwrap();
    let expected = [
        0.5067232,
        0.5525978,
        0.5984724,
        -0.53670055,
        -0.49082592,
        -1.0943773,
        0.54495203,
        0.59082663,
    ];
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 2e-6, "{actual} != {expected}");
    }
}

#[test]
fn unsupported_explicit_output_head_cannot_fall_back_to_tied_embeddings() {
    let mut model = TinyModel::new();
    model
        .types
        .insert("output.weight".into(), nexus::gguf::GgufDataType::Q5_0.id());
    let file = tempfile::NamedTempFile::new().unwrap();
    model.write(file.path());
    let mut engine = InferenceEngine::new(Some(InferenceConfig {
        allow_byte_tokenizer: true,
        ..Default::default()
    }));
    let error = engine.load_model(file.path()).unwrap_err().to_string();
    // A strict GGUF parser may reject unsupported block geometry before the
    // engine resolves required tensor names. Either way no fallback may activate.
    assert!(
        engine.model.is_none(),
        "unexpected model after load error: {error}"
    );
    // Actual tied embeddings (no output head in the file) remain supported.
    model.tensors.retain(|(name, _, _)| name != "output.weight");
    model.write(file.path());
    engine.load_model(file.path()).unwrap();
    assert!(engine.generate(&[0], 1).is_ok());
}

#[test]
fn unsupported_rope_modes_are_rejected_and_default_is_standard_llama() {
    let mut model = TinyModel::new();
    model.metadata.remove("llama.rope.freq_base");
    assert_eq!(
        engine(&model, InferenceConfig::default())
            .config
            .rope_freq_base,
        10000.
    );
    let file = tempfile::NamedTempFile::new().unwrap();
    model.metadata.insert(
        "llama.rope.scaling.type".into(),
        Value::Text("linear".into()),
    );
    model.write(file.path());
    assert!(InferenceEngine::new(None)
        .load_model(file.path())
        .unwrap_err()
        .to_string()
        .contains("RoPE"));
    model.metadata.remove("llama.rope.scaling.type");
    model
        .tensors
        .push(("rope_freqs.weight".into(), vec![2], vec![1.; 2]));
    model.write(file.path());
    assert!(InferenceEngine::new(None)
        .load_model(file.path())
        .unwrap_err()
        .to_string()
        .contains("RoPE"));
}

#[test]
fn a_short_session_can_finish_while_another_session_is_still_prefilling() {
    let mut engine = engine(&TinyModel::new(), InferenceConfig::default());
    let mut long = engine
        .start_session(
            &[0; 20],
            &GenerationConfig {
                max_tokens: 3,
                ..Default::default()
            },
        )
        .unwrap();
    let mut short = engine
        .start_session(
            &[1],
            &GenerationConfig {
                max_tokens: 1,
                ..Default::default()
            },
        )
        .unwrap();
    for _ in 0..3 {
        engine.step_session(&mut long).unwrap();
        engine.step_session(&mut short).unwrap();
    }
    assert!(short.finished);
    assert!(!long.finished);
    assert_eq!(long.output.len(), 20); // still prefilling, no generation token yet
    assert_eq!(short.output.len(), 2);
}

#[test]
fn gguf_head_dim_four_logits_match_llama_cpp_cpu_f32() {
    // Reference commit: tests/reference/llama-cpp-commit.txt.
    // C++ runner: tests/reference/llama_logits.cpp. Same one-layer GGUF weights,
    // prompt IDs [3,4,5], F32 KV. Four dimensions distinguish adjacent vs split RoPE.
    let mut engine = engine(
        &TinyModel::with_head_dim(4),
        InferenceConfig {
            kv_quant: KvQuantType::None,
            ..Default::default()
        },
    );
    let logits = engine.prefill(&[3, 4, 5]).unwrap();
    let expected = [
        -0.38200113,
        -0.05194059,
        -0.08273226,
        -0.38249704,
        0.87248015,
        -0.10852437,
        0.7971229,
        -0.082533926,
    ];
    for (actual, expected) in logits.iter().zip(expected) {
        assert!((actual - expected).abs() < 3e-6, "{actual} != {expected}");
    }
}

#[test]
fn speculative_windows_preserve_target_rng_and_stop_state() {
    let target_model = TinyModel::new();
    let mut draft_model = TinyModel::new();
    for (name, _, values) in &mut draft_model.tensors {
        if name == "output.weight" {
            for value in values {
                *value = -*value;
            }
        }
    }
    for temperature in [0., 0.8] {
        let generation = GenerationConfig {
            max_tokens: 8,
            temperature,
            seed: Some(42),
            ..Default::default()
        };
        let mut reference = engine(&target_model, InferenceConfig::default());
        let expected = reference
            .generate_with_config(&[0, 1], &generation)
            .unwrap();
        for window in [1, 2, 5] {
            let mut target = engine(&target_model, InferenceConfig::default());
            let mut draft = engine(&draft_model, InferenceConfig::default());
            let (actual, stats) = target
                .generate_with_draft_engine(&mut draft, &[0, 1], &generation, window)
                .unwrap();
            assert_eq!(
                actual, expected,
                "temperature={temperature}, window={window}"
            );
            assert!(stats.rejected > 0);
            // A multi-token stop crossing the first window must be removed exactly
            // as in ordinary generation. The target RNG must not restart per window.
            let stopped = GenerationConfig {
                stop_sequences: vec![expected[2..4].to_vec()],
                ..generation.clone()
            };
            let expected_stop = reference.generate_with_config(&[0, 1], &stopped).unwrap();
            let (actual, _) = target
                .generate_with_draft_engine(&mut draft, &[0, 1], &stopped, window)
                .unwrap();
            assert_eq!(actual, expected_stop, "stop at window={window}");
            assert_eq!(actual, vec![0, 1]);
        }
    }
}

fn constant_logits_model() -> TinyModel {
    let mut model = TinyModel::new();
    for (name, shape, values) in &mut model.tensors {
        if name == "token_embd.weight" {
            let hidden = shape[0] as usize;
            values.fill(0.);
            for row in values.chunks_exact_mut(hidden) {
                row[0] = 1.;
            }
        } else if name == "output.weight" {
            values.fill(0.);
            values[0] = 5.;
            values[shape[0] as usize] = 4.;
        } else if name.ends_with("norm.weight") {
            values.fill(1.);
        } else {
            values.fill(0.);
        }
    }
    model
}

#[test]
fn greedy_sampling_applies_penalties_and_rewards_before_selection() {
    let mut engine = engine(&constant_logits_model(), InferenceConfig::default());
    for (prompt, penalty, expected) in [(0, 2., 1), (1, 0.5, 1), (0, 1., 0)] {
        let generation = GenerationConfig {
            max_tokens: 1,
            repetition_penalty: penalty,
            ..Default::default()
        };
        let greedy = engine.generate_with_config(&[prompt], &generation).unwrap();
        assert_eq!(greedy, vec![prompt, expected]);
        let top_one = engine
            .generate_with_config(
                &[prompt],
                &GenerationConfig {
                    temperature: 1.,
                    top_k: 1,
                    seed: Some(9),
                    ..generation
                },
            )
            .unwrap();
        assert_eq!(greedy, top_one);
    }
}

#[test]
fn epsilon_is_applied_and_invalid_or_unsupported_parameters_fail_atomically() {
    let mut model = TinyModel::with_head_dim(4);
    model.metadata.insert(
        "llama.attention.layer_norm_rms_epsilon".into(),
        Value::Float(0.01),
    );
    let mut engine = engine(&model, InferenceConfig::default());
    assert_eq!(engine.config.rms_norm_eps, 0.01);
    let before = engine.prefill(&[1, 2]).unwrap();
    let file = tempfile::NamedTempFile::new().unwrap();
    for (key, value) in [
        (
            "llama.attention.layer_norm_rms_epsilon",
            Value::Text("wrong".into()),
        ),
        (
            "llama.attention.layer_norm_rms_epsilon",
            Value::Float(f32::NAN),
        ),
        ("llama.rope.freq_base", Value::Uint(10000)),
        ("llama.rope.scale_linear", Value::Float(2.)),
        ("llama.rope.dimension_count", Value::Uint(2)),
        ("llama.attention.key_length", Value::Uint(2)),
        ("llama.expert_count", Value::Uint(8)),
    ] {
        let mut invalid = TinyModel::with_head_dim(4);
        invalid.metadata.insert(key.into(), value);
        invalid.write(file.path());
        assert!(engine.load_model(file.path()).is_err(), "accepted {key}");
        assert_eq!(engine.config.rms_norm_eps, 0.01);
        assert_eq!(engine.prefill(&[1, 2]).unwrap(), before);
    }
    let mut bias_model = TinyModel::with_head_dim(4);
    bias_model
        .tensors
        .push(("blk.0.attn_q.bias".into(), vec![8], vec![0.; 8]));
    bias_model.write(file.path());
    assert!(engine
        .load_model(file.path())
        .unwrap_err()
        .to_string()
        .contains("bias"));
}

#[test]
fn payload_counters_include_actual_f32_weights_and_private_session_caches() {
    let model = TinyModel::new();
    let weight_bytes = model
        .tensors
        .iter()
        .map(|(_, _, v)| v.len() * 4)
        .sum::<usize>();
    for quant in [KvQuantType::None, KvQuantType::TurboQuant3b2b] {
        let mut engine = engine(
            &model,
            InferenceConfig {
                max_seq_len: 8,
                kv_quant: quant,
                ..Default::default()
            },
        );
        assert_eq!(engine.weight_memory_usage(), weight_bytes);
        let kv_dim = engine.num_kv_heads * engine.head_dim;
        let bytes_per_token = if quant == KvQuantType::None {
            kv_dim * 4 * 2
        } else {
            (kv_dim * quant.key_bits() as usize).div_ceil(8)
                + (kv_dim * quant.val_bits() as usize).div_ceil(8)
                + 16 // min + scale for K and V
        };
        let expected = bytes_per_token * engine.config.max_seq_len * engine.num_layers;
        assert_eq!(engine.kv_cache_memory_usage(), expected);
        let session = engine
            .start_session(
                &[0],
                &GenerationConfig {
                    max_tokens: 1,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(session.kv_cache_memory_usage(), expected);
        let mut zero = engine
            .start_session(
                &[0],
                &GenerationConfig {
                    max_tokens: 0,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(zero.finished);
        assert_eq!(zero.kv_cache_memory_usage(), 0);
        assert_eq!(engine.step_session(&mut zero).unwrap(), None);
        assert_eq!(zero.output, vec![0]);
    }
}

#[test]
fn eos_and_stop_finish_reasons_survive_speculative_window_boundaries() {
    let model = constant_logits_model();
    let mut target = engine(&model, InferenceConfig::default());
    let mut draft = engine(&model, InferenceConfig::default());
    target.model.as_mut().unwrap().metadata.metadata.insert(
        "tokenizer.ggml.eos_token_id".into(),
        nexus::gguf::GgufValue::U32(0),
    );
    let generation = GenerationConfig {
        max_tokens: 5,
        ..Default::default()
    };
    let (out, _) = target
        .generate_with_draft_engine(&mut draft, &[1], &generation, 1)
        .unwrap();
    assert_eq!(out, vec![1]);
    let mut session = target.start_session(&[1], &generation).unwrap();
    while !session.finished {
        target.step_session(&mut session).unwrap();
    }
    assert_eq!(session.finish_reason, "stop");
    assert!(session.visible_output().is_empty());
}

#[test]
fn runtime_weight_storage_is_explicit_f32_and_other_overrides_are_rejected() {
    use nexus::quant::QuantFormat;
    assert_eq!(InferenceConfig::default().weight_format, QuantFormat::FP32);
    let file = tempfile::NamedTempFile::new().unwrap();
    TinyModel::new().write(file.path());
    let mut rejected = InferenceEngine::new(Some(InferenceConfig {
        weight_format: QuantFormat::Q4_K,
        allow_byte_tokenizer: true,
        ..Default::default()
    }));
    let error = rejected.load_model(file.path()).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::Unsupported);
    assert!(rejected.model.is_none());
    let loaded = engine(&TinyModel::new(), InferenceConfig::default());
    assert_eq!(loaded.config.weight_format, QuantFormat::FP32);
    assert!(loaded.info().contains("Runtime weight storage: F32"));
    assert!(loaded.info().contains("Source tensor encodings: F32"));
}

#[test]
fn quantized_gguf_source_can_reload_with_f32_runtime_configuration() {
    use nexus::gguf::{GgufDataType, GgufReader};
    use nexus::quant::{QuantEngine, QuantFormat};
    let model = TinyModel::with_head_dim(16); // hidden 32, legal Q8_0 row geometry
    let file = tempfile::NamedTempFile::new().unwrap();
    model.write(file.path());
    let metadata = GgufReader::parse_file(file.path()).unwrap();
    let embedding = metadata.get_tensor("token_embd.weight").unwrap();
    let source = &model.tensors[0].2;
    let packed = QuantEngine::new(QuantFormat::Q8_0, KvQuantType::None)
        .quantize_weights(source)
        .unwrap();
    let mut bytes = std::fs::read(file.path()).unwrap();
    let name = b"token_embd.weight";
    let table_name = bytes.windows(name.len()).position(|b| b == name).unwrap();
    let dtype_offset = table_name + name.len() + 4 + 8 * embedding.shape.len();
    bytes[dtype_offset..dtype_offset + 4].copy_from_slice(&GgufDataType::Q8_0.id().to_le_bytes());
    let payload = metadata.absolute_tensor_offset(embedding) as usize;
    bytes[payload..payload + packed.len()].copy_from_slice(&packed);
    // Keep subsequent aligned tensor offsets unchanged. GGUF permits padding.
    bytes[payload + packed.len()..payload + source.len() * 4].fill(0);
    std::fs::write(file.path(), bytes).unwrap();
    let mut loaded = InferenceEngine::new(Some(InferenceConfig {
        allow_byte_tokenizer: true,
        kv_quant: KvQuantType::None,
        ..Default::default()
    }));
    loaded.load_model(file.path()).unwrap();
    let expected = loaded.prefill(&[1, 2, 3]).unwrap();
    assert_eq!(loaded.config.weight_format, QuantFormat::FP32);
    assert!(loaded.info().contains("Q8_0"));
    loaded.load_model(file.path()).unwrap();
    assert_eq!(loaded.config.weight_format, QuantFormat::FP32);
    assert_eq!(loaded.prefill(&[1, 2, 3]).unwrap(), expected);
}

#[test]
fn ring_storage_does_not_expand_engine_context_or_change_absolute_rope_positions() {
    let mut model = TinyModel::with_head_dim(4);
    model
        .metadata
        .insert("llama.context_length".into(), Value::Uint(4));
    let mut engine = engine(
        &model,
        InferenceConfig {
            max_seq_len: 4,
            kv_quant: KvQuantType::None,
            ..Default::default()
        },
    );
    let expected = engine.prefill(&[0, 1, 2, 3]).unwrap();
    assert_eq!(engine.kv_caches[0].seq_len, 4);
    assert!(engine
        .decode_with_cache(4, 4)
        .unwrap_err()
        .contains("Context"));
    assert!(engine.generate(&[0, 1, 2, 3], 1).is_err());
    assert_eq!(engine.prefill(&[0, 1, 2, 3]).unwrap(), expected);
}
