//! Transformer kernel implementations.
//! Contains attention, feed-forward, and normalization kernels.

/// Compute RoPE (Rotary Position Embeddings) for a given sequence position.
/// This is the standard RoPE implementation used in LLaMA and derivatives.
pub fn apply_rope(
    tokens: &[f32],
    positions: &[usize],
    head_dim: usize,
    num_heads: usize,
    freq_base: f32,
) -> Vec<f32> {
    let seq_len = positions.len();
    let mut output = vec![0.0f32; seq_len * head_dim];

    for (seq_idx, &pos) in positions.iter().enumerate() {
        for head in 0..num_heads {
            let head_start = head * head_dim / 2;
            for d in 0..head_dim / 2 {
                let dim_idx = head_start + d;
                let inv_freq = (pos as f32) / freq_base.powf(2.0 * (d as f32) / (head_dim as f32));

                let cos = inv_freq.cos();
                let sin = inv_freq.sin();

                let x0 = tokens[seq_idx * head_dim + dim_idx];
                let x1 = tokens[seq_idx * head_dim + dim_idx + head_dim / 2];

                output[seq_idx * head_dim + dim_idx] = x0 * cos - x1 * sin;
                output[seq_idx * head_dim + dim_idx + head_dim / 2] = x0 * sin + x1 * cos;
            }
        }
    }

    output
}

/// Compute GQA (Grouped Query Attention) attention scores.
/// Optimized for batched inference with KV cache.
pub fn grouped_query_attention(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    kv_seq_len: usize,
    head_dim: usize,
    num_kv_heads: usize,
    scale: f32,
) -> Vec<f32> {
    let num_q_heads = q.len() / head_dim;

    // Output: (num_q_heads, kv_seq_len)
    let mut scores = vec![0.0f32; num_q_heads * kv_seq_len];

    for qh in 0..num_q_heads {
        let kvh = qh / (num_q_heads / num_kv_heads); // Map Q head to KV head

        for k in 0..kv_seq_len {
            let mut score = 0.0f32;
            for d in 0..head_dim {
                score +=
                    q[qh * head_dim + d] * k_cache[kvh * kv_seq_len * head_dim + k * head_dim + d];
            }
            scores[qh * kv_seq_len + k] = score * scale;
        }
    }

    // Softmax and apply to values
    let mut output = vec![0.0f32; num_q_heads * head_dim];

    for qh in 0..num_q_heads {
        let kvh = qh / (num_q_heads / num_kv_heads);

        // Find max for numerical stability
        let start = qh * kv_seq_len;
        let end = start + kv_seq_len;
        let max_score = scores[start..end]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        // Compute sum of exp
        let mut sum_exp = 0.0f32;
        for s in &scores[start..end] {
            sum_exp += (*s - max_score).exp();
        }

        // Apply attention to values
        for d in 0..head_dim {
            let mut weighted_sum = 0.0f32;
            for k in 0..kv_seq_len {
                let weight = (scores[qh * kv_seq_len + k] - max_score).exp() / sum_exp;
                weighted_sum += weight * v_cache[kvh * kv_seq_len * head_dim + k * head_dim + d];
            }
            output[qh * head_dim + d] = weighted_sum;
        }
    }

    output
}

/// SwiGLU feed-forward network (used in modern LLMs).
/// FFN(x) = W3 * silu(GW(x)) @ W1 + b
/// where silu(x) = x * sigmoid(x)
pub fn swiglu_ffn(
    input: &[f32],
    w1: &[f32],
    w3: &[f32],
    w2: &[f32],
    hidden_size: usize,
    ff_dim: usize,
) -> Vec<f32> {
    let batch_size = input.len() / hidden_size;

    // Compute gate and up projections in parallel
    let mut gate_values = vec![0.0f32; batch_size * ff_dim];
    let mut up_values = vec![0.0f32; batch_size * ff_dim];

    for b in 0..batch_size {
        for j in 0..ff_dim {
            // gate = W1 @ input
            let mut gate = 0.0f32;
            for i in 0..hidden_size {
                gate += w1[j * hidden_size + i] * input[b * hidden_size + i];
            }
            gate_values[b * ff_dim + j] = gate;

            // up = W3 @ input
            let mut up = 0.0f32;
            for i in 0..hidden_size {
                up += w3[j * hidden_size + i] * input[b * hidden_size + i];
            }
            up_values[b * ff_dim + j] = up;
        }
    }

    // SwiGLU: gate * silu(up)
    let mut gated = vec![0.0f32; batch_size * ff_dim];
    for i in 0..gate_values.len() {
        let gate = gate_values[i];
        let up = up_values[i];
        gated[i] = gate * (up / (1.0 + (-up).exp())); // silu
    }

    // Output projection: W2 @ gated
    let mut output = vec![0.0f32; batch_size * hidden_size];
    for b in 0..batch_size {
        for i in 0..hidden_size {
            let mut sum = 0.0f32;
            for j in 0..ff_dim {
                sum += w2[i * ff_dim + j] * gated[b * ff_dim + j];
            }
            output[b * hidden_size + i] = sum;
        }
    }

    output
}

/// SiGLU feed-forward network (simpler variant).
pub fn sigmoid_glu_ffn(
    input: &[f32],
    w1: &[f32],
    w2: &[f32],
    hidden_size: usize,
    ff_dim: usize,
) -> Vec<f32> {
    let batch_size = input.len() / hidden_size;

    // gate and up projections
    let mut gate_values = vec![0.0f32; batch_size * ff_dim];
    let mut up_values = vec![0.0f32; batch_size * ff_dim];

    for b in 0..batch_size {
        for j in 0..ff_dim {
            let mut gate = 0.0f32;
            for i in 0..hidden_size {
                gate += w1[j * hidden_size + i] * input[b * hidden_size + i];
            }
            gate_values[b * ff_dim + j] = gate;

            let mut up = 0.0f32;
            for i in 0..hidden_size {
                up += w2[j * hidden_size + i] * input[b * hidden_size + i];
            }
            up_values[b * ff_dim + j] = up;
        }
    }

    // GLU: gate * sigmoid(up)
    let mut output = vec![0.0f32; batch_size * ff_dim];
    for i in 0..gate_values.len() {
        output[i] = gate_values[i] * (1.0 / (1.0 + (-up_values[i]).exp()));
    }

    output
}

/// Compute RMS normalization.
pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let dim = input.len();
    let variance = input.iter().map(|x| x * x).sum::<f32>() / dim as f32;
    let rstd = 1.0 / (variance + eps).sqrt();

    let mut output = vec![0.0f32; dim];
    for i in 0..dim {
        output[i] = input[i] * rstd * weight[i];
    }
    output
}

/// Compute Layer normalization.
pub fn layer_norm(input: &[f32], weight: &[f32], bias: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let dim = input.len();

    // Compute mean
    let mean: f32 = input.iter().sum::<f32>() / dim as f32;

    // Compute variance
    let variance: f32 = input.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / dim as f32;

    // Normalize and apply scale/bias
    let mut output = vec![0.0f32; dim];
    let rstd = 1.0 / (variance + eps).sqrt();

    for i in 0..dim {
        let normalized = (input[i] - mean) * rstd;
        output[i] = normalized * weight[i] + bias.map(|b| b[i]).unwrap_or(0.0);
    }

    output
}

/// Compute softmax along the last dimension.
pub fn softmax(input: &[f32], _dim: usize) -> Vec<f32> {
    let mut output = input.to_vec();

    // Find max for numerical stability
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp and sum
    let mut sum = 0.0f32;
    for val in &mut output {
        *val = (*val - max_val).exp();
        sum += *val;
    }

    // Normalize
    if sum > 0.0 {
        for val in &mut output {
            *val /= sum;
        }
    }

    output
}

/// Rotary embedding for a single token (optimized for decode).
pub fn rope_single(tokens: &mut [f32], position: usize, head_dim: usize, freq_base: f32) {
    for d in 0..head_dim / 2 {
        let inv_freq = (position as f32) / freq_base.powf(2.0 * (d as f32) / (head_dim as f32));

        let cos = inv_freq.cos();
        let sin = inv_freq.sin();

        let x0 = tokens[d];
        let x1 = tokens[d + head_dim / 2];

        tokens[d] = x0 * cos - x1 * sin;
        tokens[d + head_dim / 2] = x0 * sin + x1 * cos;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope() {
        let mut tokens = vec![1.0f32, 2.0, 3.0, 4.0];
        rope_single(&mut tokens, 0, 4, 10000.0);

        // At position 0, inv_freq = 0, cos(0) = 1, sin(0) = 1
        // So tokens should be [1, 2, 3, 4] (identity at pos=0 for this freq)
        assert!((tokens[0] - 1.0).abs() < 1e-5);
        assert!((tokens[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_rms_norm() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0f32, 1.0, 1.0, 1.0];

        let result = rms_norm(&input, &weight, 1e-5);
        assert_eq!(result.len(), 4);

        // Verify RMS calculation
        let expected_scale: f32 = 1.0 / (30.0_f32 / 4.0).sqrt();
        for i in 0..4 {
            assert!((result[i] - input[i] * expected_scale).abs() < 1e-5);
        }
    }

    #[test]
    fn test_layer_norm() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let weight = vec![1.0f32, 1.0, 1.0, 1.0];

        let result = layer_norm(&input, &weight, None, 1e-5);
        assert_eq!(result.len(), 4);

        // Mean = 2.5, variance = 1.25
        let mean: f32 = 2.5;
        let var: f32 = 1.25;
        let rstd: f32 = 1.0 / (var + 1e-5).sqrt();

        for i in 0..4 {
            let expected = (input[i] - mean) * rstd * weight[i];
            assert!((result[i] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_softmax() {
        let input = vec![1.0f32, 2.0, 3.0];
        let result = softmax(&input, 3);

        // Sum should be 1.0
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Largest value should have largest output
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_swiglu_ffn() {
        let input = vec![1.0f32, 2.0];
        let hidden_size = 2;
        let ff_dim = 4;

        let w1: Vec<f32> = (0..ff_dim * hidden_size).map(|i| i as f32 * 0.1).collect();
        let w3: Vec<f32> = (0..ff_dim * hidden_size).map(|i| i as f32 * 0.05).collect();
        let w2: Vec<f32> = (0..hidden_size * ff_dim).map(|i| i as f32 * 0.02).collect();

        let result = swiglu_ffn(&input, &w1, &w3, &w2, hidden_size, ff_dim);
        assert_eq!(result.len(), hidden_size);
    }

    #[test]
    fn test_gqa_attention() {
        let head_dim = 8;
        let kv_seq_len = 4;
        let num_kv_heads = 2;
        let num_q_heads = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q: Vec<f32> = vec![0.1f32; num_q_heads * head_dim];
        let k: Vec<f32> = vec![0.1f32; num_kv_heads * kv_seq_len * head_dim];
        let v: Vec<f32> = vec![0.1f32; num_kv_heads * kv_seq_len * head_dim];

        let result = grouped_query_attention(&q, &k, &v, kv_seq_len, head_dim, num_kv_heads, scale);
        assert_eq!(result.len(), num_q_heads * head_dim);
    }
}
