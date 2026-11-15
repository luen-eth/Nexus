//! Bounded deterministic mutation corpus. This is not coverage-guided fuzzing.
use clap::Parser;
use nexus::gguf::GgufReader;
use nexus::mlx::{load_tensor, parse_safetensors_header};
use serde_json::json;
use std::io::Cursor;
use std::path::{Path, PathBuf};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 20_000)]
    cases: usize,
    #[arg(long, default_value_t = 0x4e455855535f3236u64)]
    seed: u64,
    #[arg(long)]
    crash_dir: PathBuf,
}

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn string(out: &mut Vec<u8>, text: &str) {
    out.extend_from_slice(&(text.len() as u64).to_le_bytes());
    out.extend_from_slice(text.as_bytes());
}

fn gguf_seed(kind: u32, elements: u64, payload: usize) -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    string(&mut bytes, "general.architecture");
    bytes.extend_from_slice(&8u32.to_le_bytes());
    string(&mut bytes, "llama");
    string(&mut bytes, "weight");
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&elements.to_le_bytes());
    bytes.extend_from_slice(&kind.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.resize(bytes.len().div_ceil(32) * 32 + payload, 0);
    bytes
}

fn safetensors_seed(dtype: &str, bytes_per_element: usize) -> Vec<u8> {
    let header = json!({"weight":{"dtype":dtype,"shape":[4],"data_offsets":[0,4*bytes_per_element]},"__metadata__":{"format":"pt"}}).to_string();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.resize(bytes.len() + 4 * bytes_per_element, 0);
    bytes
}

fn mutate(seed: &[u8], rng: &mut u64, mode: usize) -> Vec<u8> {
    let mut bytes = seed.to_vec();
    let at = (next(rng) as usize) % bytes.len();
    match mode % 8 {
        0 => bytes[at] ^= 1 << (next(rng) % 8),
        1 => bytes[at] = next(rng) as u8,
        2 => bytes.truncate(at),
        3 => {
            let extra = (next(rng) as usize) % 16 + 1;
            bytes.splice(at..at, (0..extra).map(|_| next(rng) as u8));
        }
        4 => {
            let end = (at + (next(rng) as usize) % 16 + 1).min(bytes.len());
            bytes.drain(at..end);
        }
        5 => {
            let value = [0, 1, u32::MAX as u64, u64::MAX, 1 << 32, 1 << 63][next(rng) as usize % 6];
            for (target, source) in bytes[at..].iter_mut().zip(value.to_le_bytes()) {
                *target = source;
            }
        }
        6 => bytes.extend((0..(next(rng) as usize % 16)).map(|_| next(rng) as u8)),
        _ => {
            bytes = (0..next(rng) as usize % 1024)
                .map(|_| next(rng) as u8)
                .collect();
        }
    }
    bytes
}

fn audit_gguf(path: &Path, bytes: &[u8]) -> [usize; 2] {
    let stream = GgufReader::new().parse(&mut Cursor::new(bytes));
    let mapped = GgufReader::parse_file(path);
    if let Ok(ref model) = mapped {
        model.validate_tensor_layout(bytes.len() as u64).unwrap();
        let stream_model = stream
            .as_ref()
            .expect("Valid mmap file must also parse as a stream");
        assert_eq!(model.tensor_data_offset, stream_model.tensor_data_offset);
        assert_eq!(model.tensor_count, stream_model.tensor_count);
        assert_eq!(model.kv_count, stream_model.kv_count);
        for (a, b) in model.tensors.iter().zip(&stream_model.tensors) {
            assert_eq!(a.name, b.name);
            assert_eq!(a.shape, b.shape);
            assert_eq!(a.offset, b.offset);
            assert_eq!(a.data_type, b.data_type);
        }
    }
    [usize::from(stream.is_ok()), usize::from(mapped.is_ok())]
}

fn audit_safetensors(path: &Path) -> bool {
    let Ok(header) = parse_safetensors_header(path) else {
        return false;
    };
    for (name, meta) in &header.tensors {
        let loaded = load_tensor(path, name);
        if ["F32", "F16", "BF16", "I64", "I32", "I16", "I8", "U8"].contains(&meta.dtype.as_str()) {
            assert_eq!(loaded.unwrap().len(), meta.num_elements());
        }
    }
    true
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.cases == 0 || args.cases > 1_000_000 || args.seed == 0 {
        return Err("cases must be 1..=1000000 and seed must be nonzero".into());
    }
    std::fs::create_dir_all(&args.crash_dir)?;
    let seeds = [
        gguf_seed(0, 4, 16),
        gguf_seed(1, 4, 8),
        gguf_seed(30, 4, 8),
        gguf_seed(2, 32, 18),
        gguf_seed(8, 32, 34),
        gguf_seed(12, 256, 144),
        gguf_seed(10, 256, 84),
        safetensors_seed("F32", 4),
        safetensors_seed("F16", 2),
        safetensors_seed("BF16", 2),
        safetensors_seed("I32", 4),
    ];
    let temp = tempfile::NamedTempFile::new()?;
    let mut rng = args.seed;
    let mut accepted = [0usize; 3];
    let mut attempted = [0usize; 3];
    let started = std::time::Instant::now();
    for case in 0..args.cases + seeds.len() {
        let seed_index = if case < seeds.len() {
            case
        } else {
            next(&mut rng) as usize % seeds.len()
        };
        let bytes = if case < seeds.len() {
            seeds[seed_index].clone()
        } else {
            mutate(&seeds[seed_index], &mut rng, case)
        };
        std::fs::write(temp.path(), &bytes)?;
        let result = std::panic::catch_unwind(|| {
            if seed_index < 7 {
                let counts = audit_gguf(temp.path(), &bytes);
                [counts[0], counts[1], 0]
            } else {
                [0, 0, usize::from(audit_safetensors(temp.path()))]
            }
        });
        let counts = match result {
            Ok(counts) => counts,
            Err(_) => {
                let crash = args
                    .crash_dir
                    .join(format!("seed-{}-case-{case}.case", args.seed));
                std::fs::write(&crash, &bytes)?;
                return Err(
                    format!("Parser panic/invariant failure; saved {}", crash.display()).into(),
                );
            }
        };
        if case < seeds.len() {
            assert_eq!(counts, if seed_index < 7 { [1, 1, 0] } else { [0, 0, 1] });
        }
        for i in 0..3 {
            accepted[i] += counts[i];
            attempted[i] += usize::from(if i < 2 {
                seed_index < 7
            } else {
                seed_index >= 7
            });
        }
    }
    println!(
        "{}",
        json!({"seed":args.seed,"mutations":args.cases,"positive_controls":seeds.len(),"attempted":{"gguf_stream":attempted[0],"gguf_mmap":attempted[1],"safetensors":attempted[2]},"accepted":{"gguf_stream":accepted[0],"gguf_mmap":accepted[1],"safetensors":accepted[2]},"seconds":started.elapsed().as_secs_f64(),"panics":0,"method":"bounded deterministic mutation; not coverage-guided fuzzing"})
    );
    Ok(())
}
