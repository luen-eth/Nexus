use nexus::engine::{KvCache, PackedKvBuffer};
use nexus::quant::KvQuantType;
use std::collections::VecDeque;

const FORMATS: [KvQuantType; 6] = [
    KvQuantType::None,
    KvQuantType::TurboQuant3b2b,
    KvQuantType::TurboQuant3b4b,
    KvQuantType::TurboQuant4b4b,
    KvQuantType::BlockQ4,
    KvQuantType::BlockQ6,
];

fn values(token: usize, dim: usize, salt: usize) -> Vec<f32> {
    if token.is_multiple_of(7) {
        return vec![token as f32 * 0.25 - 1.; dim];
    }
    (0..dim)
        .map(|d| ((token * 19 + d * 13 + salt) % 43) as f32 * 0.125 - 2.5)
        .collect()
}
fn reference_quantized(input: &[f32], bits: Option<u32>) -> Vec<f32> {
    let Some(bits) = bits else {
        return input.to_vec();
    };
    let min = input.iter().copied().fold(f32::INFINITY, f32::min);
    let max = input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let levels = ((1u32 << bits) - 1) as f32;
    let scale = if (max - min).abs() > 1e-8 {
        (max - min) / levels
    } else {
        1.
    };
    input
        .iter()
        .map(|v| min + ((v - min) / scale).round().clamp(0., levels) * scale)
        .collect()
}
fn attention_reference(
    q: &[f32],
    keys: &[Vec<f32>],
    values: &[Vec<f32>],
    head_dim: usize,
    kv_heads: usize,
) -> Vec<f32> {
    let q_heads = q.len() / head_dim;
    let group = q_heads / kv_heads;
    let scale = 1. / (head_dim as f32).sqrt();
    let mut out = vec![0.; q.len()];
    for qh in 0..q_heads {
        let kvh = qh / group;
        let mut scores = keys
            .iter()
            .map(|key| {
                let mut dot = 0.;
                for d in 0..head_dim {
                    dot += q[qh * head_dim + d] * key[kvh * head_dim + d];
                }
                dot * scale
            })
            .collect::<Vec<_>>();
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.;
        for s in &mut scores {
            *s = (*s - max).exp();
            sum += *s;
        }
        for d in 0..head_dim {
            let mut weighted = 0.;
            for (token, score) in scores.iter().enumerate() {
                weighted += (*score / sum) * values[token][kvh * head_dim + d];
            }
            out[qh * head_dim + d] = weighted;
        }
    }
    out
}

#[test]
fn ring_snapshots_preserve_logical_order_for_every_format_after_multiple_wraps() {
    for format in FORMATS {
        for dim in [1, 3, 5, 7, 9, 17, 33] {
            for capacity in [1, 2, 5] {
                let mut cache = KvCache::new(dim, capacity, format);
                let footprint = cache.memory_usage();
                let mut expected_k = VecDeque::new();
                let mut expected_v = VecDeque::new();
                for token in 0..capacity * 4 + 3 {
                    let k = values(token, dim, 0);
                    let v = values(token, dim, 7);
                    cache.add(&k, &v);
                    expected_k.push_back(reference_quantized(
                        &k,
                        (format != KvQuantType::None).then(|| format.key_bits()),
                    ));
                    expected_v.push_back(reference_quantized(
                        &v,
                        (format != KvQuantType::None).then(|| format.val_bits()),
                    ));
                    if expected_k.len() > capacity {
                        expected_k.pop_front();
                        expected_v.pop_front();
                    }
                    let (actual_k, actual_v) = cache.get_all();
                    assert_eq!(
                        actual_k,
                        expected_k.iter().flatten().copied().collect::<Vec<_>>(),
                        "keys {format:?}/{dim}/{capacity}/{token}"
                    );
                    assert_eq!(
                        actual_v,
                        expected_v.iter().flatten().copied().collect::<Vec<_>>(),
                        "values {format:?}/{dim}/{capacity}/{token}"
                    );
                    assert_eq!(cache.seq_len, expected_k.len());
                    assert_eq!(cache.memory_usage(), footprint);
                }
                cache.clear();
                assert_eq!(cache.seq_len, 0);
                let (k, v) = cache.get_all();
                assert!(k.is_empty() && v.is_empty());
                let k = values(123, dim, 1);
                let v = values(17, dim, 9);
                cache.add(&k, &v);
                assert_eq!(
                    cache.get_all(),
                    (
                        reference_quantized(
                            &k,
                            (format != KvQuantType::None).then(|| format.key_bits())
                        ),
                        reference_quantized(
                            &v,
                            (format != KvQuantType::None).then(|| format.val_bits())
                        )
                    )
                );
            }
        }
    }
}

#[test]
fn packed_extraction_matches_independent_affine_reference_at_every_byte_boundary() {
    // All supported field widths, including straddling six-bit and ragged three-bit rows.
    for bits in 1..=8 {
        for dim in [1, 2, 3, 5, 7, 8, 9, 15, 17, 31, 33] {
            let mut packed = PackedKvBuffer::new(bits, dim, 3);
            for token in 0..12 {
                let position = token % 3;
                let input = values(token, dim, 11);
                let expected = reference_quantized(&input, Some(bits));
                packed.add(position, &input);
                for (d, expected) in expected.iter().enumerate() {
                    assert_eq!(
                        packed.value_at(position, d),
                        *expected,
                        "bits={bits}/dim={dim}/token={token}/d={d}"
                    );
                }
                let all = packed.get_all(3);
                assert_eq!(
                    &all[position * dim..(position + 1) * dim],
                    expected.as_slice()
                );
            }
            packed.clear();
            assert_eq!(packed.get_all(3), vec![0.; dim * 3]);
        }
    }
}

#[test]
fn grouped_attention_matches_previous_scalar_order_through_wraps_and_clear() {
    for format in FORMATS {
        for (kv_heads, q_heads) in [(1, 1), (1, 8), (2, 8), (3, 6), (3, 9)] {
            for head_dim in [1, 3, 7, 16] {
                for capacity in [1, 3, 7] {
                    let dim = kv_heads * head_dim;
                    let mut cache = KvCache::new(dim, capacity, format);
                    let mut expected_k = VecDeque::new();
                    let mut expected_v = VecDeque::new();
                    for token in 0..capacity * 3 + 2 {
                        let k = values(token, dim, 2);
                        let v = values(token, dim, 5);
                        cache.add(&k, &v);
                        expected_k.push_back(reference_quantized(
                            &k,
                            (format != KvQuantType::None).then(|| format.key_bits()),
                        ));
                        expected_v.push_back(reference_quantized(
                            &v,
                            (format != KvQuantType::None).then(|| format.val_bits()),
                        ));
                        if expected_k.len() > capacity {
                            expected_k.pop_front();
                            expected_v.pop_front();
                        }
                        let q = values(token + 3, q_heads * head_dim, 9);
                        let expected = attention_reference(
                            &q,
                            &expected_k.iter().cloned().collect::<Vec<_>>(),
                            &expected_v.iter().cloned().collect::<Vec<_>>(),
                            head_dim,
                            kv_heads,
                        );
                        let actual = cache.attention_decode(&q, head_dim, q_heads, kv_heads);
                        // Same token/dimension accumulation order is preserved, so the
                        // F32 and dequantized packed reference are bitwise identical.
                        assert_eq!(actual.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),expected.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),"{format:?}/kv={kv_heads}/q={q_heads}/dim={head_dim}/cap={capacity}/t={token}");
                    }
                    cache.clear();
                    assert_eq!(
                        cache.attention_decode(
                            &vec![1.; q_heads * head_dim],
                            head_dim,
                            q_heads,
                            kv_heads
                        ),
                        vec![0.; q_heads * head_dim]
                    );
                }
            }
        }
    }
}

#[test]
fn invalid_cache_dimensions_have_a_fallible_rejection_path() {
    for format in FORMATS {
        assert!(KvCache::try_new(0, 1, format).is_err());
        assert!(KvCache::try_new(1, 0, format).is_err());
        assert!(KvCache::try_new(usize::MAX, 2, format).is_err());
    }
    for bits in [0, 9, 16, 32] {
        assert!(PackedKvBuffer::try_new(bits, 4, 2).is_err());
    }
    assert!(PackedKvBuffer::try_new(3, 0, 2).is_err());
    assert!(PackedKvBuffer::try_new(3, 4, 0).is_err());
    assert!(PackedKvBuffer::try_new(8, usize::MAX, 2).is_err());
    let cache = KvCache::new(6, 3, KvQuantType::None);
    assert!(std::panic::catch_unwind(|| cache.attention_decode(&[1.; 6], 0, 2, 1)).is_err());
    assert!(std::panic::catch_unwind(|| cache.attention_decode(&[1.; 9], 3, 3, 2)).is_err());
    assert!(std::panic::catch_unwind(|| cache.attention_decode(&[1.; 8], 4, 2, 1)).is_err());
}
