mod common;
use sha2::{Digest, Sha256};
use std::process::Command;

fn offline_download_command() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_nexus"));
    command.arg("download");
    // A cache regression must fail locally instead of requesting a remote model.
    for variable in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"] {
        command.env(variable, "http://127.0.0.1:9");
    }
    command.env("NO_PROXY", "");
    command
}

#[test]
fn cli_download_accepts_repository_and_nested_filename_from_verified_cache() {
    let dir = tempfile::tempdir().unwrap();
    let content = b"offline model fixture";
    for filename in ["model.gguf", "quantized/q4/model.gguf"] {
        let path = dir.path().join(filename);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, content).unwrap();
        let receipt = serde_json::json!({
            "version": 1,
            "url": format!("https://huggingface.co/test-owner/test-repo/resolve/main/{filename}"),
            "bytes": content.len(),
            "sha256": format!("{:x}", Sha256::digest(content)),
        });
        let mut receipt_path = path.as_os_str().to_os_string();
        receipt_path.push(".nexus-download.json");
        std::fs::write(receipt_path, serde_json::to_vec(&receipt).unwrap()).unwrap();

        let output = offline_download_command()
            .arg("--model")
            .arg(format!("test-owner/test-repo/{filename}"))
            .arg("--output")
            .arg(dir.path())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{filename}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("Verified cached file:"));
        assert_eq!(std::fs::read(path).unwrap(), content);
    }
}

#[test]
fn cli_download_rejects_malformed_identifiers_and_parent_paths() {
    let dir = tempfile::tempdir().unwrap();
    for model in [
        "owner/repo",
        "owner//model.gguf",
        "/repo/model.gguf",
        "owner/repo/",
        "owner/repo/../model.gguf",
        "owner/repo/nested/../../model.gguf",
        "owner/repo//model.gguf",
        "owner/repo/\\model.gguf",
    ] {
        let output = offline_download_command()
            .args(["--model", model, "--output"])
            .arg(dir.path())
            .output()
            .unwrap();
        assert!(!output.status.success(), "{model}");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("Model identifier must be owner/repository/filename")
                || stderr.contains("Download filename must be a relative path"),
            "{model}: {stderr}"
        );
    }
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
}

#[test]
fn cli_returns_failure_when_inference_fails() {
    let file = tempfile::NamedTempFile::new().unwrap();
    common::TinyModel::new().write(file.path());
    // The fixture vocabulary has eight entries; ASCII 'h' is outside it.
    let output = Command::new(env!("CARGO_BIN_EXE_nexus"))
        .args(["run", "--model"])
        .arg(file.path())
        .args([
            "--prompt",
            "hello",
            "--max-tokens",
            "1",
            "--allow-byte-tokenizer",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("vocabulary"));
}

#[test]
fn run_without_a_prompt_fails_before_loading_the_model() {
    // NEW-4 gate: a prompt-less run previously loaded the whole multi-GB
    // model, printed a notice and exited 0. It must fail fast, after the
    // argument validations but before any model file is touched.
    let file = tempfile::NamedTempFile::new().unwrap();
    common::TinyModel::new().write(file.path());
    let output = Command::new(env!("CARGO_BIN_EXE_nexus"))
        .args(["run", "--model"])
        .arg(file.path())
        .args(["--allow-byte-tokenizer"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("no prompt provided"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    // The model was never loaded: no post-load engine banner was printed.
    assert!(!String::from_utf8_lossy(&output.stdout).contains("Layers:"));
}

#[cfg(unix)]
#[test]
fn piped_cli_output_never_panics_on_epipe() {
    // NEW-5 gate: `nexus info --model X | head` previously panicked with
    // "failed printing to stdout: Broken pipe" and exited 101. With the
    // restored SIGPIPE default the CLI either finishes before the reader
    // closes (0) or ends with the conventional SIGPIPE (141) — never a
    // Rust panic.
    use std::os::unix::process::ExitStatusExt;
    let file = tempfile::NamedTempFile::new().unwrap();
    common::TinyModel::new().write(file.path());
    let script = format!(
        "{:?} info --model {:?} | head -1",
        env!("CARGO_BIN_EXE_nexus"),
        file.path()
    );
    let output = Command::new("sh").arg("-c").arg(script).output().unwrap();
    let code = output.status.code().unwrap_or(-1);
    let signal = output.status.signal();
    assert!(
        code == 0 || signal == Some(13),
        "unexpected exit: code={code:?} signal={signal:?}, stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_ne!(code, 101, "the CLI must not panic on EPIPE");
}

#[test]
fn cli_rejects_unavailable_and_unknown_backends() {
    for backend in ["metal", "vulkan", "webgpu", "typo"] {
        let output = Command::new(env!("CARGO_BIN_EXE_nexus"))
            .args(["run", "--model", "unused.gguf", "--backend", backend])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{backend}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("backend")
                || String::from_utf8_lossy(&output.stderr).contains("Backend")
        );
    }
}

#[test]
fn every_cli_subcommand_can_parse_help_without_panicking() {
    for command in ["run", "bench", "golden", "info", "convert", "download"] {
        let output = Command::new(env!("CARGO_BIN_EXE_nexus"))
            .args([command, "--help"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{command}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn cli_rejects_zero_threads_before_loading_a_model() {
    let output = Command::new(env!("CARGO_BIN_EXE_nexus"))
        .args(["run", "--model", "unused.gguf", "--threads", "0"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("CPU threads must be between"));
}

#[test]
fn benchmark_rejects_empty_work_before_loading_a_model() {
    for args in [
        ["--iterations", "0"],
        ["--max-tokens", "0"],
        ["--prompt-len", "0"],
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_nexus"))
            .args(["bench", "--model", "unused.gguf"])
            .args(args)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("must be positive"));
    }
}

fn benchmark_fixture(eos: bool) -> tempfile::NamedTempFile {
    let file = tempfile::NamedTempFile::new().unwrap();
    let mut model = common::TinyModel::new();
    if eos {
        model
            .metadata
            .insert("tokenizer.ggml.eos_token_id".into(), common::Value::Uint(7));
        for (name, shape, data) in &mut model.tensors {
            if name == "token_embd.weight" || name.ends_with("norm.weight") {
                data.fill(1.);
            } else {
                data.fill(0.);
            }
            if name == "output.weight" {
                let width = shape[0] as usize;
                data[7 * width..8 * width].fill(1.);
            }
        }
    }
    model.write(file.path());
    file
}

#[test]
fn benchmark_has_warmup_valid_ids_and_split_latency_statistics() {
    let file = benchmark_fixture(false);
    let output = Command::new(env!("CARGO_BIN_EXE_nexus"))
        .args(["bench", "--model"])
        .arg(file.path())
        .args([
            "--prompt-len",
            "12",
            "--max-tokens",
            "3",
            "--iterations",
            "3",
            "--warmup",
            "2",
            "--allow-byte-tokenizer",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["warmup_iterations"], 2);
    assert_eq!(report["sampling"]["temperature"], 0.0);
    assert_eq!(report["sampling"]["seed"], 42);
    assert_eq!(report["runtime_weight_storage"], "f32");
    assert!(report.get("weight_quant").is_none());
    assert!(report["gguf_tensor_types"]["F32"].as_u64().unwrap() > 0);
    assert_eq!(report["input_kind"], "synthetic-valid-token-ids");
    assert_eq!(report["model_sha256"].as_str().unwrap().len(), 64);
    let row = &report["matrix"][0];
    assert_eq!(row["results"].as_array().unwrap().len(), 3);
    for sample in row["results"].as_array().unwrap() {
        let n = sample["generated_tokens"].as_u64().unwrap();
        assert!(n <= 3);
        for key in [
            "elapsed_seconds",
            "prefill_seconds",
            "decode_seconds",
            "tokens_per_second",
            "decode_tokens_per_second",
        ] {
            assert!(sample[key].as_f64().unwrap().is_finite());
        }
        assert!(
            sample["elapsed_seconds"].as_f64().unwrap()
                >= sample["prefill_seconds"].as_f64().unwrap()
                    + sample["decode_seconds"].as_f64().unwrap()
        );
    }
    let mut elapsed: Vec<f64> = row["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["elapsed_seconds"].as_f64().unwrap())
        .collect();
    elapsed.sort_by(f64::total_cmp);
    assert_eq!(
        row["statistics"]["elapsed_seconds"]["p50"]
            .as_f64()
            .unwrap(),
        elapsed[1]
    );
    assert_eq!(
        row["statistics"]["elapsed_seconds"]["p95"]
            .as_f64()
            .unwrap(),
        elapsed[2]
    );
}

#[test]
fn benchmark_counts_actual_tokens_on_immediate_eos() {
    let file = benchmark_fixture(true);
    let output = Command::new(env!("CARGO_BIN_EXE_nexus"))
        .args(["bench", "--model"])
        .arg(file.path())
        .args([
            "--prompt-len",
            "2",
            "--max-tokens",
            "4",
            "--iterations",
            "1",
            "--allow-byte-tokenizer",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    let sample = &report["matrix"][0]["results"][0];
    assert_eq!(sample["generated_tokens"], 0);
    assert_eq!(sample["tokens_per_second"], 0.);
    assert_eq!(sample["finish_reason"], "stop");
}

#[test]
fn benchmark_rejects_context_overflow() {
    let file = benchmark_fixture(false);
    let output = Command::new(env!("CARGO_BIN_EXE_nexus"))
        .args(["bench", "--model"])
        .arg(file.path())
        .args([
            "--prompt-len",
            "32",
            "--max-tokens",
            "1",
            "--allow-byte-tokenizer",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("exceeds context"));
}

#[test]
fn cli_rejects_weight_formats_that_do_not_control_runtime_storage() {
    for format in ["Q4_K", "FP16"] {
        let output = Command::new(env!("CARGO_BIN_EXE_nexus"))
            .args(["run", "--model", "unused.gguf", "--weight-quant", format])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("Runtime weights expand to F32"));
    }
}

#[test]
fn explicit_cpu_thread_limit_is_applied_to_benchmark_and_run() {
    let file = benchmark_fixture(false);
    let output = Command::new(env!("CARGO_BIN_EXE_nexus"))
        .args(["bench", "--model"])
        .arg(file.path())
        .args([
            "--prompt-len",
            "2",
            "--max-tokens",
            "2",
            "--iterations",
            "1",
            "--threads",
            "1",
            "--allow-byte-tokenizer",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["cpu_matvec_thread_limit"], 1);
    assert!(matches!(
        report["cpu_matvec_kernel"].as_str(),
        Some("scalar" | "avx2-fma")
    ));
    assert!(report["backend_features"]
        .as_array()
        .unwrap()
        .iter()
        .any(|s| s == "single-thread"));
    let output = Command::new(env!("CARGO_BIN_EXE_nexus"))
        .args(["run", "--model"])
        .arg(file.path())
        // A prompt is required before the model is loaded now; the fixture's
        // byte vocabulary only admits control-byte tokens.
        .args(["--prompt", "\u{1}\u{2}", "--max-tokens", "1"])
        .args(["--threads", "1", "--allow-byte-tokenizer"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("CPU matvec thread limit: 1"));
}

#[cfg(feature = "cuda")]
#[test]
fn cli_rejects_cpu_thread_knob_for_cuda_before_loading_model() {
    let output = Command::new(env!("CARGO_BIN_EXE_nexus"))
        .args([
            "run",
            "--model",
            "unused.gguf",
            "--backend",
            "cuda",
            "--threads",
            "1",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr)
        .contains("--threads is only supported for the CPU backend"));
}
