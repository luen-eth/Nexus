#![cfg(feature = "cuda")]
mod common;
use nexus::{
    backend::{cuda::CudaBackend, Backend, BackendType},
    engine::{InferenceConfig, InferenceEngine},
    quant::KvQuantType,
};
use std::collections::HashMap;
fn device() -> Option<CudaBackend> {
    match CudaBackend::new() {
        Ok(d) => Some(d),
        Err(e) => {
            assert!(
                std::env::var_os("NEXUS_REQUIRE_CUDA").is_none(),
                "CUDA required: {e}"
            );
            eprintln!("CUDA unavailable: {e}");
            None
        }
    }
}
#[test]
fn cuda_matvec_matches_f64_reference_for_ragged_rectangular_shapes() {
    let Some(device) = device() else { return };
    for (rows, cols) in [(1, 1), (2, 3), (19, 31), (7, 257), (65, 1025), (65536, 32)] {
        let weight: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 17 % 71) as f32 - 35.) / 71.)
            .collect();
        let input: Vec<f32> = (0..cols)
            .map(|i| ((i * 13 % 37) as f32 - 18.) / 37.)
            .collect();
        let tensors = HashMap::from([("matrix".into(), weight)]);
        device.prepare_weights(&tensors).unwrap();
        let weights = &tensors["matrix"];
        let actual = device.matvec(weights, rows, cols, &input).unwrap();
        for (row, actual) in actual.iter().enumerate() {
            let expected: f64 = weights[row * cols..(row + 1) * cols]
                .iter()
                .zip(&input)
                .map(|(&w, &x)| w as f64 * x as f64)
                .sum();
            assert!(
                (*actual as f64 - expected).abs() < 1e-5 + expected.abs() * 1e-5,
                "{rows}x{cols}: {actual} != {expected}"
            );
        }
    }
    assert_eq!(device.kernel_launches(), 6);
    let unprepared = [1f32, 2., 3.];
    assert!(device.matvec(&unprepared, 1, 3, &unprepared).is_err());
    assert_eq!(
        device.kernel_launches(),
        6,
        "An error must not execute a fallback"
    );
}
#[test]
fn cuda_engine_logits_generation_and_thread_transfer_match_cpu() {
    if device().is_none() {
        return;
    }
    let file = tempfile::NamedTempFile::new().unwrap();
    common::TinyModel::with_head_dim(4).write(file.path());
    let config = InferenceConfig {
        kv_quant: KvQuantType::None,
        allow_byte_tokenizer: true,
        ..Default::default()
    };
    let mut cpu = InferenceEngine::new(Some(config.clone()));
    cpu.load_model(file.path()).unwrap();
    let expected = cpu.prefill(&[3, 4, 5]).unwrap();
    let output = cpu.generate(&[3, 4, 5], 8).unwrap();
    let mut gpu = InferenceEngine::new(Some(InferenceConfig {
        backend: BackendType::Cuda,
        ..config
    }));
    gpu.load_model(file.path()).unwrap();
    let actual = gpu.prefill(&[3, 4, 5]).unwrap();
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < 3e-6, "{a} != {b}");
    }
    assert_eq!(gpu.backend.kernel_launches(), 3 * 8);
    // Server loads on a blocking pool and executes on a dedicated worker.
    let gpu = std::thread::spawn(move || {
        assert_eq!(gpu.generate(&[3, 4, 5], 8).unwrap(), output);
        assert_eq!(gpu.backend.kernel_launches(), (3 + 3 + 7) * 8);
        gpu
    })
    .join()
    .unwrap();
    assert_eq!(gpu.backend.backend_type(), BackendType::Cuda);
    assert!(gpu.backend.is_available());
}

fn assert_reference(device: &CudaBackend, weight: &[f32], rows: usize, cols: usize, seed: usize) {
    let input: Vec<f32> = (0..cols)
        .map(|i| {
            if seed.is_multiple_of(3) {
                0.
            } else {
                ((i * 13 + seed * 7) % 37) as f32 / 37. - 0.5
            }
        })
        .collect();
    let actual = device.matvec(weight, rows, cols, &input).unwrap();
    assert_eq!(
        actual.len(),
        rows,
        "Logical result must not include capacity tail"
    );
    for (row, &actual) in actual.iter().enumerate() {
        let expected: f64 = weight[row * cols..(row + 1) * cols]
            .iter()
            .zip(&input)
            .map(|(&a, &b)| a as f64 * b as f64)
            .sum();
        assert!(
            (actual as f64 - expected).abs() < 1e-5 + expected.abs() * 1e-5,
            "{rows}x{cols} seed={seed}: {actual} != {expected}"
        );
    }
}

#[test]
fn cuda_workspace_reuses_capacity_for_growing_shrinking_and_ragged_shapes() {
    let Some(device) = device() else { return };
    let shapes = [(1, 1), (2, 3), (19, 31), (7, 257), (65, 1025), (3, 17)];
    let tensors: HashMap<String, Vec<f32>> = shapes
        .iter()
        .enumerate()
        .map(|(j, &(rows, cols))| {
            (
                j.to_string(),
                (0..rows * cols)
                    .map(|i| ((i * 17 % 71) as f32 - 35.) / 71.)
                    .collect(),
            )
        })
        .collect();
    device.prepare_weights(&tensors).unwrap();
    // Growth includes input-only and output-only steps. Repeat below these maxima
    // with zero/nonzero inputs so stale tails or output reuse cannot mask errors.
    for (j, &(rows, cols)) in shapes.iter().enumerate() {
        assert_reference(&device, &tensors[&j.to_string()], rows, cols, j + 1);
    }
    let warmed = device.workspace_stats().unwrap();
    for repeat in 0..5 {
        for (j, &(rows, cols)) in shapes.iter().enumerate().rev() {
            assert_reference(&device, &tensors[&j.to_string()], rows, cols, repeat + j);
        }
    }
    let final_stats = device.workspace_stats().unwrap();
    assert_eq!(final_stats.input_allocations, warmed.input_allocations);
    assert_eq!(final_stats.output_allocations, warmed.output_allocations);
    assert_eq!(final_stats.retained_bytes, warmed.retained_bytes);
    assert!(final_stats.retained_bytes <= final_stats.limit_bytes);
    assert!(final_stats.peak_live_bytes <= 2 * final_stats.limit_bytes);
    let columns: usize = shapes.iter().map(|s| s.1).sum();
    let rows: usize = shapes.iter().map(|s| s.0).sum();
    assert_eq!(final_stats.input_upload_bytes, (6 * columns * 4) as u64);
    assert_eq!(final_stats.output_download_bytes, (6 * rows * 4) as u64);
    assert_eq!(device.kernel_launches(), 36);
}

#[test]
fn cuda_workspace_budget_and_invalid_requests_leave_cached_state_usable() {
    if device().is_none() {
        return;
    }
    let device = CudaBackend::new_with_workspace_limit_bytes(40).unwrap();
    let tensors = HashMap::from([
        ("a".into(), vec![1.; 3 * 5]),
        ("b".into(), vec![2.; 4 * 5]),
        ("over".into(), vec![3.; 4 * 7]),
    ]);
    device.prepare_weights(&tensors).unwrap();
    assert_reference(&device, &tensors["a"], 3, 5, 1);
    let stats = device.workspace_stats().unwrap();
    // Geometric growth would require 48 bytes; exact capacities fit 32.
    assert_eq!(stats.input_capacity_elements, 5);
    assert_eq!(stats.output_capacity_elements, 3);
    assert_eq!(stats.retained_bytes, 32);
    assert_reference(&device, &tensors["b"], 4, 5, 2);
    let before = device.workspace_stats().unwrap();
    assert_eq!(before.input_allocations, 1);
    assert_eq!(before.output_allocations, 2);
    assert_eq!(before.retained_bytes, 36);
    let launches = device.kernel_launches();
    assert!(device
        .matvec(&tensors["over"], 4, 7, &[1.; 7])
        .unwrap_err()
        .contains("workspace limit"));
    assert!(device.matvec(&tensors["a"], 3, 5, &[1.; 4]).is_err());
    assert!(device.matvec(&[0.], 1, 1, &[1.]).is_err());
    assert_eq!(device.kernel_launches(), launches);
    assert_eq!(device.workspace_stats().unwrap(), before);
    assert!(device.is_available());
    assert_reference(&device, &tensors["a"], 3, 5, 3);
    assert_eq!(device.workspace_stats().unwrap().retained_bytes, 36);
    assert!(CudaBackend::new_with_workspace_limit_bytes(7).is_err());
}

#[test]
fn cuda_workspace_serializes_concurrent_calls_and_thread_handoffs() {
    use std::sync::Arc;
    let Some(device) = device() else { return };
    let device = Arc::new(device);
    let shapes = [(65, 1025), (3, 17), (19, 31), (7, 257)];
    let tensors: HashMap<String, Vec<f32>> = shapes
        .iter()
        .enumerate()
        .map(|(j, &(rows, cols))| {
            (
                j.to_string(),
                (0..rows * cols)
                    .map(|i| ((i * 11 % 59) as f32 - 29.) / 59.)
                    .collect(),
            )
        })
        .collect();
    let tensors = Arc::new(tensors);
    device.prepare_weights(&tensors).unwrap();
    assert_reference(&device, &tensors["0"], 65, 1025, 1);
    let warmed = device.workspace_stats().unwrap();
    let threads: Vec<_> = (0..4)
        .map(|thread| {
            let device = device.clone();
            let tensors = tensors.clone();
            std::thread::spawn(move || {
                for iteration in 0..8 {
                    let index = (iteration + thread) % shapes.len();
                    let (rows, cols) = shapes[index];
                    assert_reference(
                        &device,
                        &tensors[&index.to_string()],
                        rows,
                        cols,
                        iteration + thread,
                    );
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
    assert_eq!(device.kernel_launches(), 33);
    let stats = device.workspace_stats().unwrap();
    assert_eq!(stats.input_allocations, warmed.input_allocations);
    assert_eq!(stats.output_allocations, warmed.output_allocations);
    assert!(device.is_available());
}

#[test]
fn cuda_weight_replacement_reuses_workspace_and_never_returns_stale_weights() {
    use std::sync::{Arc, Barrier};
    let Some(device) = device() else { return };
    let device = Arc::new(device);
    let a = Arc::new(HashMap::from([("a".into(), vec![1.; 5 * 7])]));
    let b = HashMap::from([("b".into(), vec![3.; 5 * 7])]);
    device.prepare_weights(&a).unwrap();
    assert_eq!(device.matvec(&a["a"], 5, 7, &[1.; 7]).unwrap(), vec![7.; 5]);
    let warmed = device.workspace_stats().unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let worker = {
        let device = device.clone();
        let a = a.clone();
        let barrier = barrier.clone();
        std::thread::spawn(move || {
            barrier.wait();
            for _ in 0..16 {
                match device.matvec(&a["a"], 5, 7, &[1.; 7]) {
                    Ok(output) => assert_eq!(output, vec![7.; 5]),
                    Err(error) => assert!(error.contains("not prepared"), "{error}"),
                }
            }
        })
    };
    barrier.wait();
    device.prepare_weights(&b).unwrap();
    worker.join().unwrap();
    assert!(device.matvec(&a["a"], 5, 7, &[1.; 7]).is_err());
    assert_eq!(
        device.matvec(&b["b"], 5, 7, &[-2.; 7]).unwrap(),
        vec![-42.; 5]
    );
    assert_eq!(device.resident_weight_bytes(), (5 * 7 * 4) as u64);
    device.prepare_weights(&HashMap::new()).unwrap();
    assert_eq!(device.resident_weight_bytes(), 0);
    assert!(device.matvec(&b["b"], 5, 7, &[1.; 7]).is_err());
    device.prepare_weights(&a).unwrap();
    assert_eq!(device.matvec(&a["a"], 5, 7, &[0.; 7]).unwrap(), vec![0.; 5]);
    let stats = device.workspace_stats().unwrap();
    assert_eq!(stats.input_allocations, warmed.input_allocations);
    assert_eq!(stats.output_allocations, warmed.output_allocations);
    assert_eq!(stats.weight_upload_bytes, (3 * 5 * 7 * 4) as u64);
    assert!(device.is_available());
}
