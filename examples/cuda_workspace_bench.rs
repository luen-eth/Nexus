//! Focused device-allocation benchmark; excludes model upload and host fixture setup.
//! cargo run --release --features cuda --example cuda_workspace_bench -- --iterations 2000
//! Development-time timings are preliminary; use controlled runs for before/after claims.
#[cfg(feature = "cuda")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    use nexus::backend::{cuda::CudaBackend, Backend};
    use std::collections::HashMap;
    use std::time::Instant;

    #[derive(Parser)]
    struct Args {
        #[arg(long, default_value_t = 2000)]
        iterations: usize,
        #[arg(long, default_value_t = 16)]
        warmup: usize,
        #[arg(long, default_value_t = 513)]
        rows: usize,
        #[arg(long, default_value_t = 1025)]
        cols: usize,
    }
    let args = Args::parse();
    if args.rows == 0
        || args.cols == 0
        || args.iterations == 0
        || args
            .rows
            .checked_mul(args.cols)
            .is_none_or(|n| n > 16 * 1024 * 1024)
    {
        return Err("Use positive rows/cols/iterations and at most 16M matrix elements".into());
    }
    let shapes = [
        (args.rows, args.cols),
        (args.rows.min(127), args.cols.min(257)),
        (args.rows.min(17), args.cols.min(31)),
        (1, 1),
    ];
    let tensors: HashMap<String, Vec<f32>> = shapes
        .iter()
        .enumerate()
        .map(|(index, &(rows, cols))| {
            (
                index.to_string(),
                (0..rows * cols)
                    .map(|i| ((i * 17 % 71) as f32 - 35.) / 71.)
                    .collect(),
            )
        })
        .collect();
    let inputs: Vec<Vec<f32>> = shapes
        .iter()
        .map(|&(_, cols)| {
            (0..cols)
                .map(|i| ((i * 13 % 37) as f32 - 18.) / 37.)
                .collect()
        })
        .collect();
    let weights: Vec<_> = (0..shapes.len())
        .map(|index| &tensors[&index.to_string()])
        .collect();
    let gpu = CudaBackend::new()?;
    gpu.prepare_weights(&tensors)?;
    // Warm all capacities and independently verify first/last rows before timing.
    for (index, &(rows, cols)) in shapes.iter().enumerate() {
        let output = gpu.matvec(weights[index], rows, cols, &inputs[index])?;
        assert_eq!(output.len(), rows);
        for row in [0, rows - 1] {
            let expected: f64 = weights[index][row * cols..(row + 1) * cols]
                .iter()
                .zip(&inputs[index])
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum();
            assert!((output[row] as f64 - expected).abs() < 1e-5 + expected.abs() * 1e-5);
        }
    }
    for i in 0..args.warmup {
        let index = i % shapes.len();
        let (rows, cols) = shapes[index];
        std::hint::black_box(gpu.matvec(weights[index], rows, cols, &inputs[index])?);
    }
    let before = gpu.workspace_stats()?;
    let launches_before = gpu.kernel_launches();
    let start = Instant::now();
    let mut checksum = 0f64;
    for i in 0..args.iterations {
        let index = i % shapes.len();
        let (rows, cols) = shapes[index];
        let output = gpu.matvec(weights[index], rows, cols, &inputs[index])?;
        checksum += output[0] as f64;
        std::hint::black_box(output);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let after = gpu.workspace_stats()?;
    assert_eq!(before.input_allocations, after.input_allocations);
    assert_eq!(before.output_allocations, after.output_allocations);
    assert_eq!(
        gpu.kernel_launches() - launches_before,
        args.iterations as u64
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "benchmark":"cuda_workspace_matvec",
            "timing_scope":"hot matvec with logical host copies and stream synchronization; upload/setup excluded",
            "debug_build":cfg!(debug_assertions),
            "shapes":shapes,
            "warmup":args.warmup,
            "iterations":args.iterations,
            "elapsed_seconds":elapsed,
            "microseconds_per_matvec":elapsed*1e6/args.iterations as f64,
            "hot_input_device_allocations":after.input_allocations-before.input_allocations,
            "hot_output_device_allocations":after.output_allocations-before.output_allocations,
            "hot_input_upload_bytes":after.input_upload_bytes-before.input_upload_bytes,
            "hot_output_download_bytes":after.output_download_bytes-before.output_download_bytes,
            "resident_weight_bytes":gpu.resident_weight_bytes(),
            "workspace_before":before,
            "workspace_after":after,
            "checksum":checksum
        }))?
    );
    Ok(())
}
#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("Enable --features cuda to run the CUDA workspace benchmark");
    std::process::exit(1);
}
