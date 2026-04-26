# Nexus

Nexus is a Rust inference engine prototype that combines GGUF loading, MLX-style safetensors ingestion, weight/KV quantization, and an OpenAI-compatible API surface.

![Nexus architecture](docs/assets/nexus-architecture.svg)

## Architecture

```mermaid
flowchart TB
    subgraph "API and Tools"
        CLI["CLI"]
        Server["OpenAI-compatible server"]
        Bench["Benchmark and golden logits"]
    end

    subgraph "Runtime Engine"
        Scheduler["Paged request scheduler"]
        Decode["LLaMA-style decode"]
        Spec["Draft-engine speculative verify"]
        Moe["MoE top-k routed FFN helper"]
    end

    subgraph "Model and Quantization"
        GGUF["GGUF v2/v3 parser"]
        Convert["Safetensors/MLX to GGUF converter"]
        WeightQ["F32/F16/BF16/Q8_0/Q4_0/Q4_K/Q2_K"]
        KVQ["Packed low-bit KV cache"]
    end

    subgraph "Memory and Backends"
        Pages["Per-request KV page assignment"]
        CPU["CPU SIMD path"]
        Metal["Metal capability probe"]
        Planned["CUDA/Vulkan/WebGPU capability placeholders"]
    end

    CLI --> Decode
    Server --> Scheduler
    Bench --> Decode
    Scheduler --> Pages
    Scheduler --> Decode
    Spec --> Decode
    Moe --> Decode
    Convert --> GGUF
    GGUF --> WeightQ
    Decode --> KVQ
    KVQ --> Pages
    Decode --> CPU
    Decode --> Metal
    Decode -. unavailable .-> Planned
```

## Request Flow

```mermaid
sequenceDiagram
    participant Client
    participant API as Server/CLI
    participant Tok as RuntimeTokenizer
    participant S as Scheduler
    participant E as InferenceEngine
    participant KV as Paged KV Cache

    Client->>API: prompt or chat messages
    API->>Tok: render chat template and encode
    API->>S: add request
    S->>KV: assign request KV pages
    S->>E: decode batch
    E->>KV: read/write packed keys and values
    E-->>API: generated token IDs
    API->>Tok: decode completion
    API-->>Client: text or SSE chunks
    S->>KV: release pages on completion
```

## Quantization Flow

```mermaid
flowchart LR
    ST["HF safetensors shards"] --> Loader["MLX/safetensors loader"]
    Loader --> Converter["nexus-convert"]
    Converter --> Meta["GGUF metadata and tokenizer fields"]
    Converter --> Weights["Weight quantization"]
    Weights --> Q80["Q8_0"]
    Weights --> Q40["Q4_0"]
    Weights --> Q4K["Q4_K"]
    Weights --> Q2K["Q2_K"]
    Meta --> GGUF["GGUF v3 file"]
    Q80 --> GGUF
    Q40 --> GGUF
    Q4K --> GGUF
    Q2K --> GGUF
    GGUF --> Runtime["Runtime dequant/load"]
    Runtime --> KV["Packed KV direct attention"]
```

## Current Support

| Area | Status |
| --- | --- |
| GGUF parsing | GGUF v2/v3 metadata and tensor table parsing |
| Tensor loading | F32, F16, BF16, Q8_0, Q4_0, Q4_K, Q2_K |
| Runtime | LLaMA-style decode path with RMSNorm, RoPE, GQA attention, SwiGLU, KV cache, external draft-engine speculative verification |
| MoE | Top-k router selection and routed SwiGLU expert dispatch helper |
| KV cache | Float and packed low-bit storage, direct attention reads from quantized cache |
| Tokenizer | GGUF tokens, BPE merges, special token IDs, basic chat-template rendering, lowercase/accent/byte-level metadata, byte fallback |
| API | `/v1/chat/completions`, `/v1/models`, prompt rendering from full message list, streaming SSE chunks |
| Sampling | Greedy, temperature, top-p, top-k, repetition penalty, seed, stop sequences |
| Scheduler | Pending/running/completed lifecycle, decode batches, generated-token append, per-request KV page assignment |
| Converter | safetensors/MLX shards to GGUF v3 for F32/F16/Q8_0/Q4_0/Q4_K/Q2_K, tokenizer metadata import |
| Backends | CPU execution path, Metal capability probe, explicit unavailable CUDA/Vulkan/WebGPU capabilities |
| Benchmarks | CLI throughput matrix with text/JSON output and golden logits reports |

## Known Limits

- CUDA/Vulkan/WebGPU backends are exposed as unavailable capabilities but do not have kernels yet.
- Metal compute kernels are capability-probed but not production fused kernels.
- GGUF BPE and common tokenizer normalizer/pre-tokenizer metadata are supported; full Hugging Face tokenizer parity still needs exhaustive component coverage.
- Scheduler supports continuous lifecycle primitives; production multi-request paged attention kernels still need backend integration.

## Commands

```bash
cargo test --workspace
cargo run --bin nexus -- info --model path/to/model.gguf
cargo run --bin nexus -- run --model path/to/model.gguf --prompt "Hello" --max-tokens 32
cargo run --bin nexus -- bench --model path/to/model.gguf --prompt-lens 128,512,2048 --iterations 3 --max-tokens 64 --json
cargo run --bin nexus -- golden --model path/to/model.gguf --prompt "Hello" --top-k 10 --output golden.json
cargo run -p nexus-convert -- --input path/to/hf-model-dir --output model.gguf --quant q4-k
```

## Development Checks

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```
