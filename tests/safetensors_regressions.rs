use nexus::mlx::{load_tensor, parse_safetensors_header, MlxModelLoader};
use serde_json::{json, Value};
use std::io::Write;
use std::path::Path;

fn write_raw(path: &Path, header: &str, payload: &[u8]) {
    let mut file = std::fs::File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(header.as_bytes()).unwrap();
    file.write_all(payload).unwrap();
}
fn write(path: &Path, header: Value, payload: &[u8]) {
    write_raw(path, &header.to_string(), payload);
}

#[test]
fn malformed_ranges_shapes_and_dtypes_return_errors_without_panicking() {
    let cases = [
        (
            "reversed",
            json!({"x":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}),
            vec![0; 4],
        ),
        (
            "offset-overflow",
            json!({"x":{"dtype":"F32","shape":[1],"data_offsets":[u64::MAX-4,u64::MAX]}}),
            vec![0; 4],
        ),
        (
            "out-of-file",
            json!({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}),
            vec![0; 4],
        ),
        (
            "nonmultiple",
            json!({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,3]}}),
            vec![0; 3],
        ),
        (
            "shape-mismatch",
            json!({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}),
            vec![0; 4],
        ),
        (
            "shape-overflow",
            json!({"x":{"dtype":"F32","shape":[u64::MAX,2],"data_offsets":[0,0]}}),
            vec![],
        ),
        (
            "byte-size-overflow",
            json!({"x":{"dtype":"F32","shape":[u64::MAX/2],"data_offsets":[0,0]}}),
            vec![],
        ),
        (
            "negative-shape",
            json!({"x":{"dtype":"F32","shape":[-1],"data_offsets":[0,0]}}),
            vec![],
        ),
        (
            "unknown-dtype",
            json!({"x":{"dtype":"FAKE","shape":[1],"data_offsets":[0,4]}}),
            vec![0; 4],
        ),
        (
            "overlap",
            json!({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}),
            vec![0; 4],
        ),
        (
            "gap",
            json!({"x":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}),
            vec![0; 8],
        ),
        (
            "trailing",
            json!({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}),
            vec![0; 8],
        ),
        ("metadata-values", json!({"__metadata__":{"bad":1}}), vec![]),
    ];
    let file = tempfile::NamedTempFile::new().unwrap();
    for (name, header, bytes) in cases {
        write(file.path(), header, &bytes);
        let parsed = std::panic::catch_unwind(|| parse_safetensors_header(file.path()));
        assert!(parsed.is_ok(), "parser panicked: {name}");
        assert!(parsed.unwrap().is_err(), "parser accepted: {name}");
        let loaded = std::panic::catch_unwind(|| load_tensor(file.path(), "x"));
        assert!(loaded.is_ok(), "loader panicked: {name}");
        assert!(loaded.unwrap().is_err(), "loader accepted: {name}");
    }
}

#[test]
fn duplicate_tensor_and_descriptor_keys_are_rejected() {
    let file = tempfile::NamedTempFile::new().unwrap();
    for header in [
        r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"x":{"dtype":"F32","dtype":"I32","shape":[1],"data_offsets":[0,4]}}"#,
        r#"{"__metadata__":{},"__metadata__":{}}"#,
    ] {
        write_raw(file.path(), header, &[0; 4]);
        assert!(parse_safetensors_header(file.path()).is_err());
    }
}

#[test]
fn advertised_header_lengths_are_checked_against_budget_and_file_before_allocation() {
    let file = tempfile::NamedTempFile::new().unwrap();
    for size in [0, 100_000_000, 100_000_001, u64::MAX] {
        std::fs::write(file.path(), size.to_le_bytes()).unwrap();
        assert!(parse_safetensors_header(file.path()).is_err());
    }
    write_raw(file.path(), "{}", &[]);
    assert!(parse_safetensors_header(file.path()).is_ok());
}

#[test]
fn scalar_empty_and_endian_explicit_float_tensors_load_correctly() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let mut payload = vec![];
    payload.extend(half::f16::from_f32(1.5).to_bits().to_le_bytes());
    payload.extend(half::bf16::from_f32(-2.5).to_bits().to_le_bytes());
    payload.extend(3.25f32.to_le_bytes());
    payload.extend((-17i32).to_le_bytes());
    payload.extend(123456i64.to_le_bytes());
    write(
        file.path(),
        json!({
            "empty":{"dtype":"F32","shape":[2,0,3],"data_offsets":[0,0]},
            "scalar":{"dtype":"F16","shape":[],"data_offsets":[0,2]},
            "bf":{"dtype":"BF16","shape":[1],"data_offsets":[2,4]},
            "f32":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},
            "i32":{"dtype":"I32","shape":[1],"data_offsets":[8,12]},
            "i64":{"dtype":"I64","shape":[1],"data_offsets":[12,20]},
            "__metadata__":{"format":"pt"}
        }),
        &payload,
    );
    for (name, values) in [
        ("empty", vec![]),
        ("scalar", vec![1.5]),
        ("bf", vec![-2.5]),
        ("f32", vec![3.25]),
        ("i32", vec![-17.]),
        ("i64", vec![123456.]),
    ] {
        assert_eq!(load_tensor(file.path(), name).unwrap(), values, "{name}");
    }
    assert!(load_tensor(file.path(), "missing").is_err());
}

#[test]
fn valid_but_unsupported_packed_dtype_is_not_misread_as_float_weights() {
    let file = tempfile::NamedTempFile::new().unwrap();
    write(
        file.path(),
        json!({"packed":{"dtype":"U32","shape":[1],"data_offsets":[0,4]}}),
        &0x12345678u32.to_le_bytes(),
    );
    assert!(parse_safetensors_header(file.path()).is_ok());
    assert!(load_tensor(file.path(), "packed")
        .unwrap_err()
        .to_string()
        .contains("Unsupported dtype"));
}

#[test]
fn tensor_loading_detects_a_truncated_payload_after_loader_construction() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    write(
        &path,
        json!({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}),
        &42f32.to_le_bytes(),
    );
    let loader = MlxModelLoader::new(dir.path()).unwrap();
    assert_eq!(loader.load_tensor("x").unwrap(), vec![42.]);
    let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    file.set_len(std::fs::metadata(&path).unwrap().len() - 1)
        .unwrap();
    assert!(loader.load_tensor("x").is_err());
}

#[test]
fn present_but_invalid_model_config_is_never_silently_ignored() {
    let dir = tempfile::tempdir().unwrap();
    write(
        &dir.path().join("model.safetensors"),
        json!({"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}),
        &42f32.to_le_bytes(),
    );
    assert!(MlxModelLoader::new(dir.path()).unwrap().config.is_none());
    let path = dir.path().join("config.json");
    for bad in ["{ broken", r#"{"hidden_size":"wrong"}"#, "null"] {
        std::fs::write(&path, bad).unwrap();
        let error = MlxModelLoader::new(dir.path())
            .err()
            .expect("invalid config must fail");
        assert!(format!("{error:#}").contains("config"));
    }
    std::fs::write(&path, r#"{"model_type":"llama","hidden_size":4}"#).unwrap();
    assert_eq!(
        MlxModelLoader::new(dir.path())
            .unwrap()
            .config
            .unwrap()
            .hidden_size,
        Some(4)
    );
    std::fs::File::create(&path)
        .unwrap()
        .set_len(16 * 1024 * 1024 + 1)
        .unwrap();
    let error = MlxModelLoader::new(dir.path()).err().unwrap();
    assert!(error.to_string().contains("budget"));
}
