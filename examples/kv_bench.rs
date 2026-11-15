//! Synthetic, same-process KV microbenchmark. It is not a model/GPU throughput benchmark.
//! Legacy module below freezes the stage-1 shift/bit-loop implementation for comparison.
use clap::Parser;
use nexus::{engine::KvCache, quant::KvQuantType};
use std::{hint::black_box, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long, default_value_t = 256)]
    context: usize,
    #[arg(long, default_value_t = 64)]
    head_dim: usize,
    #[arg(long, default_value_t = 2)]
    kv_heads: usize,
    #[arg(long, default_value_t = 8)]
    q_heads: usize,
    #[arg(long, default_value_t = 25)]
    iterations: usize,
    #[arg(long, default_value_t = 5000)]
    writes: usize,
    #[arg(long, default_value_t = 3)]
    samples: usize,
    #[arg(long, default_value = "all")]
    format: String,
}
fn row(i: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|d| ((i * 13 + d * 7) % 53) as f32 * 0.0625 - 1.25)
        .collect()
}
fn median_ms(mut work: impl FnMut(), samples: usize) -> f64 {
    work(); // allocation/code/data warmup, excluded from timed samples
    let mut times = Vec::new();
    for _ in 0..samples {
        let now = Instant::now();
        work();
        times.push(now.elapsed().as_secs_f64() * 1000.);
    }
    times.sort_by(f64::total_cmp);
    times[times.len() / 2]
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.context == 0
        || args.head_dim == 0
        || args.kv_heads == 0
        || args.q_heads == 0
        || !args.q_heads.is_multiple_of(args.kv_heads)
        || args.samples == 0
        || args.iterations == 0
        || args.writes == 0
    {
        return Err("positive sizes/counts and Q heads divisible by KV heads are required".into());
    }
    let dim = args
        .head_dim
        .checked_mul(args.kv_heads)
        .ok_or("dimension overflow")?;
    let qdim = args
        .head_dim
        .checked_mul(args.q_heads)
        .ok_or("query dimension overflow")?;
    let formats = if args.format == "all" {
        vec![
            KvQuantType::None,
            KvQuantType::TurboQuant3b2b,
            KvQuantType::TurboQuant3b4b,
            KvQuantType::TurboQuant4b4b,
            KvQuantType::BlockQ4,
            KvQuantType::BlockQ6,
        ]
    } else {
        vec![KvQuantType::from_str(&args.format).ok_or("unknown KV format")?]
    };
    let rows = (0..64).map(|i| row(i, dim)).collect::<Vec<_>>();
    let queries = (0..8).map(|i| row(i, qdim)).collect::<Vec<_>>();
    let mut results = Vec::new();
    for format in formats {
        let mut current = KvCache::try_new(dim, args.context, format)?;
        let mut old = legacy::KvCache::new(dim, args.context, format);
        for token in 0..args.context + args.context / 2 {
            let k = &rows[token % rows.len()];
            let v = &rows[(token + 1) % rows.len()];
            current.add(k, v);
            old.add(k, v);
        }
        // Check equality after a partial wrap, including all grouped heads.
        let expected =
            old.attention_decode(&queries[0], args.head_dim, args.q_heads, args.kv_heads);
        let actual =
            current.attention_decode(&queries[0], args.head_dim, args.q_heads, args.kv_heads);
        let max_error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        if max_error > 2e-6 {
            return Err(format!("KV parity failed for {format:?}: {max_error}").into());
        }
        assert_eq!(current.get_all(), old.get_all());
        let legacy_append_ms = median_ms(
            || {
                for i in 0..args.writes {
                    old.add(
                        black_box(&rows[i % rows.len()]),
                        black_box(&rows[(i + 1) % rows.len()]),
                    );
                }
                black_box(&old);
            },
            args.samples,
        );
        let ring_append_ms = median_ms(
            || {
                for i in 0..args.writes {
                    current.add(
                        black_box(&rows[i % rows.len()]),
                        black_box(&rows[(i + 1) % rows.len()]),
                    );
                }
                black_box(&current);
            },
            args.samples,
        );
        assert_eq!(current.get_all(), old.get_all());
        // Refill identically after the append benchmark before comparing attention.
        old.clear();
        current.clear();
        for token in 0..args.context + args.context / 2 {
            old.add(&rows[token % rows.len()], &rows[(token + 1) % rows.len()]);
            current.add(&rows[token % rows.len()], &rows[(token + 1) % rows.len()]);
        }
        let legacy_attention_ms = median_ms(
            || {
                for i in 0..args.iterations {
                    black_box(old.attention_decode(
                        black_box(&queries[i % queries.len()]),
                        args.head_dim,
                        args.q_heads,
                        args.kv_heads,
                    ));
                }
            },
            args.samples,
        );
        let grouped_attention_ms = median_ms(
            || {
                for i in 0..args.iterations {
                    black_box(current.attention_decode(
                        black_box(&queries[i % queries.len()]),
                        args.head_dim,
                        args.q_heads,
                        args.kv_heads,
                    ));
                }
            },
            args.samples,
        );
        results.push(serde_json::json!({
            "format":format.name(),"context":args.context,"head_dim":args.head_dim,"kv_heads":args.kv_heads,"q_heads":args.q_heads,
            "writes_per_sample":args.writes,"attention_calls_per_sample":args.iterations,"samples":args.samples,
            "cache_payload_bytes":current.memory_usage(),"max_abs_attention_error":max_error,
            "append":{"legacy_shift_ms":legacy_append_ms,"ring_ms":ring_append_ms,"ratio":legacy_append_ms/ring_append_ms},
            "attention":{"legacy_bit_loop_ms":legacy_attention_ms,"grouped_extract_ms":grouped_attention_ms,"ratio":legacy_attention_ms/grouped_attention_ms}
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"scope":"synthetic_local_preliminary_microbenchmark","results":results})
        )?
    );
    Ok(())
}

#[allow(dead_code)]
mod legacy {
    use nexus::quant::KvQuantType;
    /// Packed low-bit storage for a KV cache stream.
    pub struct PackedKvBuffer {
        data: Vec<u8>,
        mins: Vec<f32>,
        scales: Vec<f32>,
        bits: u32,
        head_dim: usize,
        bytes_per_token: usize,
    }

    impl PackedKvBuffer {
        pub fn new(bits: u32, head_dim: usize, max_seq_len: usize) -> Self {
            let bytes_per_token = (head_dim * bits as usize).div_ceil(8);
            PackedKvBuffer {
                data: vec![0; bytes_per_token * max_seq_len],
                mins: vec![0.0; max_seq_len],
                scales: vec![1.0; max_seq_len],
                bits,
                head_dim,
                bytes_per_token,
            }
        }

        pub fn add(&mut self, position: usize, values: &[f32]) {
            assert_eq!(values.len(), self.head_dim, "KV vector dimension mismatch");

            let levels = (1u32 << self.bits) - 1;
            let min = values.iter().copied().fold(f32::INFINITY, f32::min);
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = if (max - min).abs() > 1e-8 {
                (max - min) / levels as f32
            } else {
                1.0
            };

            self.mins[position] = min;
            self.scales[position] = scale;

            let start = position * self.bytes_per_token;
            let end = start + self.bytes_per_token;
            self.data[start..end].fill(0);

            for (i, &value) in values.iter().enumerate() {
                let quantized = ((value - min) / scale).round().clamp(0.0, levels as f32) as u32;
                self.write_bits(start, i, quantized);
            }
        }

        pub fn get_all(&self, seq_len: usize) -> Vec<f32> {
            let mut output = vec![0.0; seq_len * self.head_dim];
            let mask = (1u32 << self.bits) - 1;

            for position in 0..seq_len {
                let start = position * self.bytes_per_token;
                let min = self.mins[position];
                let scale = self.scales[position];

                for dim in 0..self.head_dim {
                    let q = self.read_bits(start, dim) & mask;
                    output[position * self.head_dim + dim] = min + q as f32 * scale;
                }
            }

            output
        }

        pub fn value_at(&self, position: usize, dim: usize) -> f32 {
            let start = position * self.bytes_per_token;
            let q = self.read_bits(start, dim) & ((1u32 << self.bits) - 1);
            self.mins[position] + q as f32 * self.scales[position]
        }

        pub fn shift_left(&mut self, seq_len: usize) {
            if seq_len <= 1 {
                return;
            }

            let bytes = (seq_len - 1) * self.bytes_per_token;
            self.data
                .copy_within(self.bytes_per_token..self.bytes_per_token + bytes, 0);
            let tail_start = bytes;
            let tail_end = tail_start + self.bytes_per_token;
            self.data[tail_start..tail_end].fill(0);

            self.mins.copy_within(1..seq_len, 0);
            self.scales.copy_within(1..seq_len, 0);
            self.mins[seq_len - 1] = 0.0;
            self.scales[seq_len - 1] = 1.0;
        }

        pub fn clear(&mut self) {
            self.data.fill(0);
            self.mins.fill(0.0);
            self.scales.fill(1.0);
        }

        pub fn memory_usage(&self) -> usize {
            self.data.len()
                + self.mins.len() * std::mem::size_of::<f32>()
                + self.scales.len() * std::mem::size_of::<f32>()
        }

        fn write_bits(&mut self, token_start: usize, index: usize, value: u32) {
            let bit_offset = index * self.bits as usize;
            for bit in 0..self.bits as usize {
                if ((value >> bit) & 1) == 0 {
                    continue;
                }
                let absolute = token_start * 8 + bit_offset + bit;
                let byte = absolute / 8;
                let bit_in_byte = absolute % 8;
                self.data[byte] |= 1 << bit_in_byte;
            }
        }

        fn read_bits(&self, token_start: usize, index: usize) -> u32 {
            let bit_offset = index * self.bits as usize;
            let mut value = 0u32;
            for bit in 0..self.bits as usize {
                let absolute = token_start * 8 + bit_offset + bit;
                let byte = absolute / 8;
                let bit_in_byte = absolute % 8;
                let bit_value = (self.data[byte] >> bit_in_byte) & 1;
                value |= (bit_value as u32) << bit;
            }
            value
        }
    }

    /// KV cache buffer for one key or value stream.
    pub enum KvCacheBuffer {
        Float(Vec<f32>),
        Quantized(PackedKvBuffer),
    }

    impl KvCacheBuffer {
        fn new(bits: u32, head_dim: usize, max_seq_len: usize, quantized: bool) -> Self {
            if quantized {
                KvCacheBuffer::Quantized(PackedKvBuffer::new(bits, head_dim, max_seq_len))
            } else {
                KvCacheBuffer::Float(vec![0.0; max_seq_len * head_dim])
            }
        }

        fn add(&mut self, position: usize, head_dim: usize, values: &[f32]) {
            match self {
                KvCacheBuffer::Float(data) => {
                    let offset = position * head_dim;
                    data[offset..offset + head_dim].copy_from_slice(values);
                }
                KvCacheBuffer::Quantized(buffer) => buffer.add(position, values),
            }
        }

        fn get_all(&self, seq_len: usize, head_dim: usize) -> Vec<f32> {
            match self {
                KvCacheBuffer::Float(data) => data[..seq_len * head_dim].to_vec(),
                KvCacheBuffer::Quantized(buffer) => buffer.get_all(seq_len),
            }
        }

        fn value_at(&self, position: usize, dim: usize, head_dim: usize) -> f32 {
            match self {
                KvCacheBuffer::Float(data) => data[position * head_dim + dim],
                KvCacheBuffer::Quantized(buffer) => buffer.value_at(position, dim),
            }
        }

        fn shift_left(&mut self, seq_len: usize, head_dim: usize) {
            match self {
                KvCacheBuffer::Float(data) => {
                    let total = seq_len * head_dim;
                    let skip = head_dim;
                    let src_len = total - skip;
                    data.copy_within(skip..total, 0);
                    data[src_len..total].fill(0.0);
                }
                KvCacheBuffer::Quantized(buffer) => buffer.shift_left(seq_len),
            }
        }

        fn clear(&mut self) {
            match self {
                KvCacheBuffer::Float(data) => data.fill(0.0),
                KvCacheBuffer::Quantized(buffer) => buffer.clear(),
            }
        }

        fn memory_usage(&self) -> usize {
            match self {
                KvCacheBuffer::Float(data) => data.len() * std::mem::size_of::<f32>(),
                KvCacheBuffer::Quantized(buffer) => buffer.memory_usage(),
            }
        }

        fn is_quantized(&self) -> bool {
            matches!(self, KvCacheBuffer::Quantized(_))
        }
    }

    /// KV cache for a single layer
    pub struct KvCache {
        pub keys: KvCacheBuffer,
        pub values: KvCacheBuffer,
        pub seq_len: usize,
        pub max_seq_len: usize,
        pub head_dim: usize,
        pub quant_type: KvQuantType,
    }

    impl KvCache {
        pub fn new(head_dim: usize, max_seq_len: usize, quant_type: KvQuantType) -> Self {
            let quantized = quant_type != KvQuantType::None;
            KvCache {
                keys: KvCacheBuffer::new(quant_type.key_bits(), head_dim, max_seq_len, quantized),
                values: KvCacheBuffer::new(quant_type.val_bits(), head_dim, max_seq_len, quantized),
                seq_len: 0,
                max_seq_len,
                head_dim,
                quant_type,
            }
        }

        /// Add a token to the cache
        pub fn add(&mut self, key: &[f32], value: &[f32]) {
            if self.seq_len >= self.max_seq_len {
                // Shift cache (simple implementation)
                self.shift();
            }

            self.keys.add(self.seq_len, self.head_dim, key);
            self.values.add(self.seq_len, self.head_dim, value);
            self.seq_len += 1;
        }

        /// Shift cache by one position (for when cache is full)
        fn shift(&mut self) {
            self.keys.shift_left(self.seq_len, self.head_dim);
            self.values.shift_left(self.seq_len, self.head_dim);
            self.seq_len -= 1;
        }

        /// Get the full cache content
        pub fn get_all(&self) -> (Vec<f32>, Vec<f32>) {
            (
                self.keys.get_all(self.seq_len, self.head_dim),
                self.values.get_all(self.seq_len, self.head_dim),
            )
        }

        /// Clear the cache
        pub fn clear(&mut self) {
            self.keys.clear();
            self.values.clear();
            self.seq_len = 0;
        }

        /// Memory usage in bytes
        pub fn memory_usage(&self) -> usize {
            self.keys.memory_usage() + self.values.memory_usage()
        }

        pub fn is_quantized(&self) -> bool {
            self.keys.is_quantized() || self.values.is_quantized()
        }

        pub fn attention_decode(
            &self,
            q: &[f32],
            query_head_dim: usize,
            num_q_heads: usize,
            num_kv_heads: usize,
        ) -> Vec<f32> {
            let group = (num_q_heads / num_kv_heads).max(1);
            let scale = 1.0 / (query_head_dim as f32).sqrt();
            let mut output = vec![0.0f32; num_q_heads * query_head_dim];

            for qh in 0..num_q_heads {
                let kvh = (qh / group).min(num_kv_heads - 1);
                let mut scores = vec![0.0f32; self.seq_len];

                for (token, score_slot) in scores.iter_mut().enumerate().take(self.seq_len) {
                    let mut score = 0.0f32;
                    for d in 0..query_head_dim {
                        let dim = kvh * query_head_dim + d;
                        score += q[qh * query_head_dim + d]
                            * self.keys.value_at(token, dim, self.head_dim);
                    }
                    *score_slot = score * scale;
                }

                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for score in &mut scores {
                    *score = (*score - max_score).exp();
                    sum_exp += *score;
                }
                if sum_exp <= 0.0 {
                    continue;
                }

                for d in 0..query_head_dim {
                    let mut weighted_sum = 0.0f32;
                    let dim = kvh * query_head_dim + d;
                    for (token, score) in scores.iter().enumerate().take(self.seq_len) {
                        weighted_sum +=
                            (*score / sum_exp) * self.values.value_at(token, dim, self.head_dim);
                    }
                    output[qh * query_head_dim + d] = weighted_sum;
                }
            }

            output
        }
    }
}
