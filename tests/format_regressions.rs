use nexus::gguf::{GgufDataType as T, GgufReader};
use nexus::quant::{KvQuantType, QuantEngine, QuantFormat};
use std::io::{Cursor, ErrorKind, Write};

fn string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}
fn file(bytes: &[u8]) -> tempfile::NamedTempFile {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(bytes).unwrap();
    file
}
fn header(tensors: u64, kv: u64) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend_from_slice(&3u32.to_le_bytes());
    b.extend_from_slice(&tensors.to_le_bytes());
    b.extend_from_slice(&kv.to_le_bytes());
    b
}
fn tensor(b: &mut Vec<u8>, name: &str, shape: &[u64], ty: u32, offset: u64) {
    string(b, name);
    b.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for d in shape {
        b.extend_from_slice(&d.to_le_bytes());
    }
    b.extend_from_slice(&ty.to_le_bytes());
    b.extend_from_slice(&offset.to_le_bytes());
}
fn pad(b: &mut Vec<u8>) {
    b.resize(b.len().div_ceil(32) * 32, 0);
}
fn floats(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

#[test]
fn official_type_ids_and_exact_block_sizes() {
    // Independent hard-coded ggml.h/ggml-common.h ABI, including reserved holes.
    let expected = [
        (0, "F32", 1, 4),
        (1, "F16", 1, 2),
        (2, "Q4_0", 32, 18),
        (3, "Q4_1", 32, 20),
        (6, "Q5_0", 32, 22),
        (7, "Q5_1", 32, 24),
        (8, "Q8_0", 32, 34),
        (9, "Q8_1", 32, 36),
        (10, "Q2_K", 256, 84),
        (11, "Q3_K", 256, 110),
        (12, "Q4_K", 256, 144),
        (13, "Q5_K", 256, 176),
        (14, "Q6_K", 256, 210),
        (15, "Q8_K", 256, 292),
        (16, "IQ2_XXS", 256, 66),
        (17, "IQ2_XS", 256, 74),
        (18, "IQ3_XXS", 256, 98),
        (19, "IQ1_S", 256, 50),
        (20, "IQ4_NL", 32, 18),
        (21, "IQ3_S", 256, 110),
        (22, "IQ2_S", 256, 82),
        (23, "IQ4_XS", 256, 136),
        (24, "I8", 1, 1),
        (25, "I16", 1, 2),
        (26, "I32", 1, 4),
        (27, "I64", 1, 8),
        (28, "F64", 1, 8),
        (29, "IQ1_M", 256, 56),
        (30, "BF16", 1, 2),
        (34, "TQ1_0", 256, 54),
        (35, "TQ2_0", 256, 66),
        (39, "MXFP4", 32, 17),
        (40, "NVFP4", 64, 36),
        (41, "Q1_0", 128, 18),
    ];
    for (id, name, block, bytes) in expected {
        let ty = T::from_id(id).unwrap();
        assert_eq!(
            (ty.id(), ty.name(), ty.block_layout()),
            (id, name, (block, bytes))
        );
        assert_eq!(ty.checked_data_size(block).unwrap(), bytes);
    }
    for id in [4, 5, 31, 32, 33, 36, 37, 38, 42, u32::MAX] {
        assert!(T::from_id(id).is_none());
    }
    assert_eq!(
        GgufReader::tensor_bytes_to_f32(T::F64, 1, &1f64.to_le_bytes())
            .unwrap_err()
            .kind(),
        ErrorKind::Unsupported
    );
    assert_eq!(
        GgufReader::tensor_bytes_to_f32(T::BF16, 1, &0x3f80u16.to_le_bytes()).unwrap(),
        [1.]
    );
}

#[test]
fn upstream_k_quant_golden_blocks_decode_exactly() {
    for (ty, bytes, expected) in [
        (
            T::Q4_K,
            include_bytes!("fixtures/format/q4-k.bin").as_slice(),
            include_bytes!("fixtures/format/q4-k.expected.f32").as_slice(),
        ),
        (
            T::Q2_K,
            include_bytes!("fixtures/format/q2-k.bin").as_slice(),
            include_bytes!("fixtures/format/q2-k.expected.f32").as_slice(),
        ),
    ] {
        let expected = floats(expected);
        assert_eq!(
            GgufReader::tensor_bytes_to_f32(ty, expected.len(), bytes).unwrap(),
            expected
        );
        assert!(
            GgufReader::tensor_bytes_to_f32(ty, expected.len(), &bytes[..bytes.len() - 1]).is_err()
        );
        assert!(GgufReader::tensor_bytes_to_f32(ty, 0, bytes)
            .unwrap()
            .is_empty());
    }
}

// Independent layout oracle ported from upstream C; never calls a Nexus decoder.
fn reference_decode(bytes: &[u8], q4: bool) -> Vec<f32> {
    fn half(b: &[u8]) -> f32 {
        half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()
    }
    let mut y = Vec::new();
    for b in bytes.chunks_exact(if q4 { 144 } else { 84 }) {
        if q4 {
            let d = half(b);
            let m = half(&b[2..]);
            let scales = &b[4..16];
            for group in 0..4 {
                for lane in 0..2 {
                    let j = 2 * group + lane;
                    let (sc, mn) = if j < 4 {
                        (scales[j] & 63, scales[j + 4] & 63)
                    } else {
                        (
                            (scales[j + 4] & 15) | ((scales[j - 4] >> 6) << 4),
                            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
                        )
                    };
                    for v in &b[16 + 32 * group..16 + 32 * (group + 1)] {
                        y.push(d * sc as f32 * ((v >> (4 * lane)) & 15) as f32 - m * mn as f32);
                    }
                }
            }
        } else {
            let d = half(&b[80..]);
            let m = half(&b[82..]);
            let mut sc_index = 0;
            for group in 0..2 {
                for shift in [0, 2, 4, 6] {
                    for lane in 0..2 {
                        let sc = b[sc_index];
                        sc_index += 1;
                        for v in &b[16 + group * 32 + lane * 16..16 + group * 32 + (lane + 1) * 16]
                        {
                            y.push(
                                d * (sc & 15) as f32 * ((v >> shift) & 3) as f32
                                    - m * (sc >> 4) as f32,
                            );
                        }
                    }
                }
            }
        }
    }
    y
}

#[test]
fn encoded_k_quant_is_correct_in_an_independent_decoder() {
    let x: Vec<f32> = (0..512)
        .map(|i| (i as f32 * 0.17).sin() * 5. + i as f32 * 0.002)
        .collect();
    for (format, q4, bound) in [
        (QuantFormat::Q4_K, true, 0.25),
        (QuantFormat::Q2_K, false, 0.8),
    ] {
        let bytes = QuantEngine::new(format, KvQuantType::None)
            .quantize_weights(&x)
            .unwrap();
        let decoded = reference_decode(&bytes, q4);
        assert_eq!(decoded.len(), x.len());
        let mae: f32 = decoded
            .iter()
            .zip(&x)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / x.len() as f32;
        assert!(mae < bound, "{format}: MAE {mae}");
    }
}

#[test]
fn unsupported_weight_formats_and_invalid_input_are_errors() {
    for format in [
        QuantFormat::Q3_K,
        QuantFormat::Q4_1,
        QuantFormat::Q5_0,
        QuantFormat::Q5_1,
        QuantFormat::Q5_K,
        QuantFormat::Q6_K,
        QuantFormat::Q8_K,
        QuantFormat::MXFP4,
    ] {
        let q = QuantEngine::new(format, KvQuantType::None);
        assert_eq!(
            q.quantize_weights(&[1.; 256]).unwrap_err().kind(),
            ErrorKind::Unsupported
        );
        assert_eq!(
            q.dequantize(&[0; 256], 256).unwrap_err().kind(),
            ErrorKind::Unsupported
        );
    }
    for format in [QuantFormat::FP32, QuantFormat::FP16, QuantFormat::Q4_K] {
        assert!(QuantEngine::new(format, KvQuantType::None)
            .quantize_weights(&[f32::NAN])
            .is_err());
    }
    let q = QuantEngine::new(QuantFormat::FP32, KvQuantType::None);
    assert!(q.dequantize(&[0; 3], 1).is_err());
    assert!(GgufReader::tensor_bytes_to_f32(T::F32, usize::MAX, &[]).is_err());
}

#[test]
fn stream_and_mmap_use_identical_metadata_widths_and_tensor_offset() {
    let cases = [
        (0, vec![7]),
        (1, vec![255]),
        (2, 7u16.to_le_bytes().to_vec()),
        (3, (-7i16).to_le_bytes().to_vec()),
        (4, 7u32.to_le_bytes().to_vec()),
        (5, (-7i32).to_le_bytes().to_vec()),
        (6, 7f32.to_le_bytes().to_vec()),
        (7, vec![1]),
        (8, {
            let mut b = Vec::new();
            string(&mut b, "hello");
            b
        }),
        (10, 7u64.to_le_bytes().to_vec()),
        (11, (-7i64).to_le_bytes().to_vec()),
        (12, 7f64.to_le_bytes().to_vec()),
    ];
    for (ty, value) in cases {
        for array in [false, true] {
            let mut b = header(1, 2);
            string(&mut b, "a");
            if array {
                b.extend_from_slice(&9u32.to_le_bytes());
                b.extend_from_slice(&(ty as u32).to_le_bytes());
                b.extend_from_slice(&1u64.to_le_bytes());
            } else {
                b.extend_from_slice(&(ty as u32).to_le_bytes());
            }
            b.extend_from_slice(&value);
            string(&mut b, "sentinel");
            b.extend_from_slice(&4u32.to_le_bytes());
            b.extend_from_slice(&9u32.to_le_bytes());
            tensor(&mut b, "w", &[1], 0, 0);
            pad(&mut b);
            let offset = b.len() as u64;
            b.extend_from_slice(&42f32.to_le_bytes());
            let f = file(&b);
            let mapped = GgufReader::parse_file(f.path()).unwrap();
            let mut reader = GgufReader::new();
            let streamed = reader.parse(&mut Cursor::new(&b)).unwrap();
            assert_eq!(
                mapped.metadata, streamed.metadata,
                "type={ty} array={array}"
            );
            assert_eq!(streamed.get_usize("sentinel"), Some(9));
            assert_eq!(streamed.tensor_data_offset, offset);
            assert_eq!(reader.tensor_data_offset(), offset);
            assert_eq!(
                GgufReader::read_tensor_f32(f.path(), &streamed, &streamed.tensors[0]).unwrap(),
                [42.]
            );
        }
    }
}

#[test]
fn malformed_headers_and_layouts_never_panic() {
    let mut b = header(1, 0);
    tensor(&mut b, "w", &[1], 0, 0);
    pad(&mut b);
    b.extend_from_slice(&42f32.to_le_bytes());
    for end in 0..b.len() {
        let f = file(&b[..end]);
        let r = std::panic::catch_unwind(|| GgufReader::parse_file(f.path()));
        assert!(r.is_ok());
        assert!(r.unwrap().is_err(), "prefix {end}");
    }
    for (shape, ty, offset) in [
        (vec![0], 0, 0),
        (vec![1; 5], 0, 0),
        (vec![u64::MAX, 2], 0, 0),
        (vec![1], 0, 4096),
        (vec![1], 0, u64::MAX),
        (vec![1], 0, 1),
        (vec![31], 2, 0),
    ] {
        let mut b = header(1, 0);
        tensor(&mut b, "w", &shape, ty, offset);
        pad(&mut b);
        b.extend_from_slice(&[0; 128]);
        let f = file(&b);
        assert!(GgufReader::parse_file(f.path()).is_err());
    }
    for name in ["a", "b"] {
        let mut b = header(2, 0);
        tensor(&mut b, "a", &[1], 0, 0);
        tensor(&mut b, name, &[1], 0, 0);
        pad(&mut b);
        b.extend_from_slice(&[0; 4]);
        assert!(GgufReader::parse_file(file(&b).path()).is_err());
    }
    for (ty, value) in [
        (8, u64::MAX.to_le_bytes().to_vec()),
        (7, vec![2]),
        (9, {
            let mut b = 10u32.to_le_bytes().to_vec();
            b.extend_from_slice(&u64::MAX.to_le_bytes());
            b
        }),
        (9, {
            let mut b = 9u32.to_le_bytes().to_vec();
            b.extend_from_slice(&0u64.to_le_bytes());
            b
        }),
    ] {
        let mut b = header(0, 1);
        string(&mut b, "a");
        b.extend_from_slice(&(ty as u32).to_le_bytes());
        b.extend_from_slice(&value);
        let f = file(&b);
        let r = std::panic::catch_unwind(|| GgufReader::parse_file(f.path()));
        assert!(r.is_ok());
        assert!(r.unwrap().is_err());
    }
    for alignment in [0, 3, u32::MAX] {
        let mut b = header(0, 1);
        string(&mut b, "general.alignment");
        b.extend_from_slice(&4u32.to_le_bytes());
        b.extend_from_slice(&alignment.to_le_bytes());
        assert!(GgufReader::new().parse(&mut Cursor::new(b)).is_err());
    }
}

#[test]
fn parser_reuse_does_not_keep_previous_model_metadata() {
    let mut b = header(0, 1);
    string(&mut b, "old");
    b.extend_from_slice(&4u32.to_le_bytes());
    b.extend_from_slice(&1u32.to_le_bytes());
    let mut reader = GgufReader::new();
    assert!(reader
        .parse(&mut Cursor::new(b))
        .unwrap()
        .metadata
        .contains_key("old"));
    assert!(reader
        .parse(&mut Cursor::new(header(0, 0)))
        .unwrap()
        .metadata
        .is_empty());
}
