//! Preliminary CPU microbench, not a model-level acceptance benchmark.
//! cargo run --release --locked --example cpu_matvec_bench -- --threads 1,2,4
use clap::Parser;
use nexus::backend::{cpu::CpuBackend, Backend};
use std::{hint::black_box, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "4x4,64x64,576x576,1536x576,1024x1025,49152x576"
    )]
    shapes: Vec<String>,
    #[arg(long, value_delimiter = ',', default_value = "1")]
    threads: Vec<usize>,
    #[arg(long, default_value_t = 15)]
    iterations: usize,
}

// Exact ordered F32 projection from the phase-1 trait default. Keep shape checks
// and allocation inside the timed operation; this is not a model correctness oracle.
fn phase1_scalar(
    weight: &[f32],
    rows: usize,
    cols: usize,
    input: &[f32],
) -> Result<Vec<f32>, String> {
    if rows == 0 || cols == 0 || input.len() != cols || rows.checked_mul(cols) != Some(weight.len())
    {
        return Err("Invalid baseline shape".into());
    }
    Ok(weight
        .chunks_exact(cols)
        .map(|row| row.iter().zip(input).map(|(w, x)| w * x).sum())
        .take(rows)
        .collect())
}

fn measure(
    mut op: impl FnMut() -> Result<Vec<f32>, String>,
    iterations: usize,
    batch: usize,
) -> Result<serde_json::Value, String> {
    for _ in 0..3 {
        black_box(op()?);
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        for _ in 0..batch {
            black_box(op()?);
        }
        samples.push(start.elapsed().as_secs_f64() / batch as f64);
    }
    samples.sort_by(f64::total_cmp);
    let rank = |p: f64| samples[((samples.len() as f64 * p).ceil() as usize).saturating_sub(1)];
    Ok(
        serde_json::json!({"p50_seconds":rank(0.5),"p95_seconds":rank(0.95),"batch":batch,"samples":samples}),
    )
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.iterations == 0 || args.shapes.is_empty() || args.threads.is_empty() {
        return Err("Provide positive iterations and nonempty shapes/thread limits".into());
    }
    let backends: Vec<_> = args
        .threads
        .iter()
        .map(|&n| CpuBackend::with_threads(n).map_err(std::io::Error::other))
        .collect::<Result<_, _>>()?;
    let portable = CpuBackend::scalar();
    let mut report = Vec::new();
    for shape in &args.shapes {
        let (r, c) = shape.split_once('x').ok_or("Expected ROWSxCOLS")?;
        let rows: usize = r.parse()?;
        let cols: usize = c.parse()?;
        let elements = rows
            .checked_mul(cols)
            .filter(|&n| n > 0)
            .ok_or("Invalid or overflowing shape")?;
        let weight_bytes = elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or("Weight byte count overflow")?;
        let weights: Vec<f32> = (0..elements)
            .map(|i| ((i.wrapping_mul(17) % 71) as f32 - 35.) / 71.)
            .collect();
        let input: Vec<f32> = (0..cols)
            .map(|i| ((i.wrapping_mul(13) % 37) as f32 - 18.) / 37.)
            .collect();
        let expected =
            phase1_scalar(&weights, rows, cols, &input).map_err(std::io::Error::other)?;
        let batch = if elements < 8192 {
            100
        } else if elements < 1 << 20 {
            10
        } else {
            1
        };
        let baseline = measure(
            || phase1_scalar(black_box(&weights), rows, cols, black_box(&input)),
            args.iterations,
            batch,
        )?;
        let scalar = measure(
            || portable.matvec(black_box(&weights), rows, cols, black_box(&input)),
            args.iterations,
            batch,
        )?;
        let mut variants = Vec::new();
        for backend in &backends {
            let out = backend.matvec(&weights, rows, cols, &input)?;
            let max_diff = out
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f32::max);
            let timing = measure(
                || backend.matvec(black_box(&weights), rows, cols, black_box(&input)),
                args.iterations,
                batch,
            )?;
            variants.push(serde_json::json!({"threads":backend.thread_count(),"kernel":backend.matvec_kernel(),"max_abs_vs_phase1":max_diff,"timing":timing}));
        }
        report.push(serde_json::json!({"rows":rows,"cols":cols,"weight_bytes":weight_bytes,"phase1_scalar":baseline,"current_scalar_fallback":scalar,"variants":variants}));
    }
    println!(
        "{}",
        serde_json::json!({"build_profile":if cfg!(debug_assertions){"debug"}else{"release"},"preliminary":true,"measurement":"matvec only, allocation included, three warmups, batched small shapes; shared machine; no overall model-speed claim","iterations":args.iterations,"results":report})
    );
    Ok(())
}
