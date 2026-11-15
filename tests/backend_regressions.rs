use nexus::backend::{
    cpu::CpuBackend, metal::MetalBackend, Backend, BackendFactory, BackendType, KernelOp,
    KernelOutput,
};
use nexus::quant::QuantFormat;
use std::sync::Arc;

fn f32_result(result: Result<KernelOutput, String>) -> Vec<f32> {
    match result.unwrap() {
        KernelOutput::F32(v) => v,
        other => panic!("Unexpected output: {other:?}"),
    }
}
fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < 1e-5, "{actual:?} != {expected:?}");
    }
}

#[test]
fn public_backend_returns_rectangular_matmul_and_normalization_results() {
    let cpu = CpuBackend::new();
    let out = f32_result(cpu.execute(&KernelOp::MatMul {
        a: Arc::from([1., 2., 3., 4., 5., 6.]),
        b: Arc::from([1., 2., 3., 4., 5., 6.]),
        m: 2,
        n: 2,
        k: 3,
    }));
    close(&out, &[22., 28., 49., 64.]);
    let input: Arc<[f32]> = Arc::from([1., 3.]);
    let out = f32_result(cpu.execute(&KernelOp::RmsNorm {
        input: input.clone(),
        weight: Arc::from([1., 2.]),
        eps: 1e-5,
    }));
    close(&out, &[1. / (5.00001f32).sqrt(), 6. / (5.00001f32).sqrt()]);
    let out = f32_result(cpu.execute(&KernelOp::LayerNorm {
        input,
        weight: Arc::from([1., 2.]),
        bias: Some(Arc::from([0., 1.])),
        eps: 1e-5,
    }));
    close(
        &out,
        &[-1. / (1.00001f32).sqrt(), 2. / (1.00001f32).sqrt() + 1.],
    );
}

#[test]
fn public_attention_handles_multiple_kv_tokens_and_additive_masks() {
    let cpu = CpuBackend::new();
    let q: Arc<[f32]> = Arc::from([1., 0.]);
    let k: Arc<[f32]> = Arc::from([1., 0., 0., 1.]);
    let v: Arc<[f32]> = Arc::from([2., 0., 0., 4.]);
    let op = |mask| KernelOp::Attention {
        q: q.clone(),
        k: k.clone(),
        v: v.clone(),
        scale: 1.,
        mask,
    };
    let p = std::f32::consts::E / (std::f32::consts::E + 1.);
    close(
        &f32_result(cpu.execute(&op(None))),
        &[2. * p, 4. * (1. - p)],
    );
    close(
        &f32_result(cpu.execute(&op(Some(Arc::from([f32::NEG_INFINITY, 0.]))))),
        &[0., 4.],
    );
    close(
        &f32_result(cpu.execute(&op(Some(Arc::from([f32::NEG_INFINITY, f32::NEG_INFINITY]))))),
        &[0., 0.],
    );
    assert!(cpu.execute(&op(Some(Arc::from([0.])))).is_err());
    assert!(cpu.execute(&op(Some(Arc::from([f32::NAN, 0.])))).is_err());
}

#[test]
fn backend_shape_errors_are_results_instead_of_panics_or_silent_success() {
    let cpu = CpuBackend::new();
    let ops = [
        KernelOp::MatMul {
            a: Arc::from([1.]),
            b: Arc::from([2.]),
            m: usize::MAX,
            n: 2,
            k: 2,
        },
        KernelOp::Attention {
            q: Arc::from([]),
            k: Arc::from([1.]),
            v: Arc::from([2.]),
            scale: 1.,
            mask: None,
        },
        KernelOp::Attention {
            q: Arc::from([1., 0.]),
            k: Arc::from([1., 0., 1.]),
            v: Arc::from([2., 0., 2.]),
            scale: 1.,
            mask: None,
        },
        KernelOp::RmsNorm {
            input: Arc::from([1., 2.]),
            weight: Arc::from([1.]),
            eps: 1e-5,
        },
        KernelOp::Add {
            a: Arc::from([1.]),
            b: Arc::from([1., 2.]),
        },
        KernelOp::Softmax {
            input: Arc::from([1., 2., 3.]),
            dim: 2,
        },
        KernelOp::Dequantize {
            input: Arc::from([0u8]),
            format: QuantFormat::FP32,
            output_len: 1,
        },
    ];
    for op in ops {
        assert!(cpu.execute(&op).is_err(), "{op:?}");
    }
}

#[test]
fn softmax_is_per_last_dimension_and_elementwise_results_are_returned() {
    let cpu = CpuBackend::new();
    close(
        &f32_result(cpu.execute(&KernelOp::Softmax {
            input: Arc::from([0., 0., 10., 10.]),
            dim: 2,
        })),
        &[0.5; 4],
    );
    close(
        &f32_result(cpu.execute(&KernelOp::Add {
            a: Arc::from([1., 2.]),
            b: Arc::from([3., 4.]),
        })),
        &[4., 6.],
    );
    close(
        &f32_result(cpu.execute(&KernelOp::Mul {
            a: Arc::from([1., 2.]),
            b: Arc::from([3., 4.]),
        })),
        &[3., 8.],
    );
}

#[test]
fn quantized_outputs_are_typed_and_truncated_buffers_rejected() {
    let cpu = CpuBackend::new();
    let input: Arc<[f32]> = Arc::from([0.25, -0.5, 0.75]);
    for format in [QuantFormat::FP32, QuantFormat::FP16, QuantFormat::Q8_0] {
        let KernelOutput::U8(bytes) = cpu
            .execute(&KernelOp::Quantize {
                input: input.clone(),
                format,
            })
            .unwrap()
        else {
            panic!("Expected bytes")
        };
        let out = f32_result(cpu.execute(&KernelOp::Dequantize {
            input: bytes.clone().into(),
            format,
            output_len: 3,
        }));
        for (a, b) in out.iter().zip(input.iter()) {
            assert!((a - b).abs() < 0.01);
        }
        assert!(cpu
            .execute(&KernelOp::Dequantize {
                input: bytes[..bytes.len() - 1].into(),
                format,
                output_len: 3
            })
            .is_err());
    }
    assert!(cpu
        .execute(&KernelOp::Quantize {
            input,
            format: QuantFormat::Q8_K
        })
        .is_err());
}

#[test]
fn unavailable_backends_return_errors_without_cpu_fallback() {
    let op = KernelOp::RmsNorm {
        input: Arc::from([1.]),
        weight: Arc::from([1.]),
        eps: 1e-5,
    };
    let metal = MetalBackend::new();
    assert!(!metal.is_available());
    assert!(metal.execute(&op).is_err());
    assert!(metal.matvec(&[1.], 1, 1, &[1.]).is_err());
    for kind in [BackendType::Metal, BackendType::Vulkan, BackendType::WebGpu] {
        let backend = BackendFactory::create(Some(kind));
        assert!(!backend.is_available());
        assert!(backend.execute(&op).is_err());
    }
    #[cfg(feature = "cuda")]
    if let Ok(cuda) = nexus::backend::cuda::CudaBackend::new() {
        assert!(cuda.execute(&op).is_err());
    }
}

#[test]
fn production_cpu_matvec_matches_f64_for_ragged_unaligned_inputs() {
    let dispatched = CpuBackend::new();
    let scalar = CpuBackend::scalar();
    assert_eq!(dispatched.thread_count(), 1);
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        assert_eq!(dispatched.matvec_kernel(), "avx2-fma");
    }
    for cols in [
        1, 7, 8, 15, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257, 576, 1025, 1536,
    ] {
        let rows = 7;
        let weights: Vec<f32> = (0..rows * cols + 1)
            .map(|i| ((i * 17 % 71) as f32 - 35.) / 71.)
            .collect();
        let input: Vec<f32> = (0..cols + 1)
            .map(|i| ((i * 13 % 37) as f32 - 18.) / 37.)
            .collect();
        // Deliberately offset both slices to exercise unaligned vector loads.
        for cpu in [&dispatched, &scalar] {
            let actual = cpu.matvec(&weights[1..], rows, cols, &input[1..]).unwrap();
            for (row, &actual) in actual.iter().enumerate() {
                let expected: f64 = weights[1 + row * cols..1 + (row + 1) * cols]
                    .iter()
                    .zip(&input[1..])
                    .map(|(&w, &x)| w as f64 * x as f64)
                    .sum();
                assert!(
                    (actual as f64 - expected).abs() < 1e-5 + expected.abs() * 1e-5,
                    "{} cols={cols} row={row}: {actual} != {expected}",
                    cpu.matvec_kernel()
                );
            }
        }
    }
}

#[test]
fn cpu_matvec_shape_overflow_and_nonfinite_values_return_errors() {
    for cpu in [CpuBackend::new(), CpuBackend::scalar()] {
        for (weight, rows, cols, input) in [
            (&[][..], 0, 0, &[][..]),
            (&[1.][..], usize::MAX, 2, &[1., 1.][..]),
            (&[1.][..], 1, 2, &[1., 1.][..]),
            (&[1.][..], 1, 1, &[][..]),
            (&[f32::NAN][..], 1, 1, &[1.][..]),
            (&[f32::INFINITY][..], 1, 1, &[0.][..]),
            (&[1.][..], 1, 1, &[f32::NAN][..]),
            (&[f32::MAX][..], 1, 1, &[f32::MAX][..]),
        ] {
            assert!(cpu.matvec(weight, rows, cols, input).is_err());
        }
        // Exercise nonfinite detection inside the vectorized body and its tail.
        for bad in [0, 31, 32] {
            let mut weights = vec![1.; 33];
            weights[bad] = f32::NAN;
            assert!(cpu.matvec(&weights, 1, 33, &[1.; 33]).is_err());
        }
    }
}

#[test]
fn explicit_private_pool_preserves_rows_and_rejects_invalid_limits() {
    assert!(CpuBackend::with_threads(0).is_err());
    assert!(CpuBackend::with_threads(num_cpus::get().saturating_add(1)).is_err());
    let count = num_cpus::get().clamp(1, 2);
    let parallel = CpuBackend::with_threads(count).unwrap();
    assert_eq!(parallel.thread_count(), count);
    let serial = CpuBackend::new();
    for (rows, cols) in [(3, 7), (128, 129), (1024, 1025)] {
        let weight: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7 % 41) as f32 - 20.) / 41.)
            .collect();
        let input: Vec<f32> = (0..cols)
            .map(|i| ((i * 5 % 29) as f32 - 14.) / 29.)
            .collect();
        let actual = parallel.matvec(&weight, rows, cols, &input).unwrap();
        assert_eq!(actual, serial.matvec(&weight, rows, cols, &input).unwrap());
        for (row, &actual) in actual.iter().enumerate() {
            let expected: f64 = weight[row * cols..(row + 1) * cols]
                .iter()
                .zip(&input)
                .map(|(&w, &x)| w as f64 * x as f64)
                .sum();
            assert!((actual as f64 - expected).abs() < 1e-5 + expected.abs() * 1e-5);
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn vectorized_cpu_engine_matches_scalar_and_cuda_on_ragged_hidden_sizes() {
    use nexus::engine::{InferenceConfig, InferenceEngine};
    use nexus::quant::KvQuantType;
    #[path = "common/mod.rs"]
    mod common;
    let probe = BackendFactory::create(Some(BackendType::Cuda));
    if !probe.is_available() {
        assert!(
            std::env::var_os("NEXUS_REQUIRE_CUDA").is_none(),
            "CUDA required: {:?}",
            probe.failure_reason()
        );
        return;
    }
    drop(probe);
    for head_dim in [32, 34] {
        let file = tempfile::NamedTempFile::new().unwrap();
        common::TinyModel::with_head_dim(head_dim).write(file.path());
        let cfg = InferenceConfig {
            allow_byte_tokenizer: true,
            kv_quant: KvQuantType::None,
            ..Default::default()
        };
        let mut cpu = InferenceEngine::new(Some(cfg.clone()));
        cpu.load_model(file.path()).unwrap();
        let mut scalar = InferenceEngine::new(Some(cfg.clone()));
        scalar.backend = Box::new(CpuBackend::scalar());
        scalar.load_model(file.path()).unwrap();
        let mut gpu = InferenceEngine::new(Some(InferenceConfig {
            backend: BackendType::Cuda,
            ..cfg
        }));
        gpu.load_model(file.path()).unwrap();
        let actual = cpu.prefill(&[3, 4, 5]).unwrap();
        for other in [
            scalar.prefill(&[3, 4, 5]).unwrap(),
            gpu.prefill(&[3, 4, 5]).unwrap(),
        ] {
            for (&a, &b) in actual.iter().zip(&other) {
                assert!(
                    (a - b).abs() < 1e-5 + 1e-5 * b.abs(),
                    "head_dim={head_dim}: {a} != {b}"
                );
            }
        }
        let tokens = cpu.generate(&[3, 4, 5], 8).unwrap();
        assert_eq!(tokens, scalar.generate(&[3, 4, 5], 8).unwrap());
        assert_eq!(tokens, gpu.generate(&[3, 4, 5], 8).unwrap());
    }
}
