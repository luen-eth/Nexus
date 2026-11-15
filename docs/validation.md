# Validation and compatibility

This document describes the November 2026 correctness and optimization work. Nexus is still an inference prototype. The tests below cover the stated formats and LLaMA execution path; passing them does not establish support for every GGUF model, production workload or GPU.

## Correctness changes

| Area | Behavior now required by regression tests |
| --- | --- |
| GGUF and quantization | Official GGML type IDs and block geometry; independent Q2_K/Q4_K reference blocks; checked dimensions, offsets, metadata limits, EOF and tensor overlap |
| Safetensors | Checked header size, dtype/shape byte length and data ranges; duplicate/gapped/overlapping tensors rejected; explicit little-endian reads |
| Conversion | Supported LLaMA architecture only; correct Q/K RoPE permutation and axes; atomic output replacement; incompatible block rows retained as floating point |
| Model loading | Required tensors, supported RoPE and tokenizer validated; failed replacement preserves the previous model; no synthetic fallback |
| Tokenization | Embedded HF pipeline or native GGUF vocabulary; special tokens, chat templates, EOS/EOT, stop text and UTF-8 stream handling |
| Sessions | Independent KV cache and RNG; context and token budgets checked; no unused final forward pass |
| HTTP runtime | Bounded worker/stream queues, disconnect cleanup, atomic model lifecycle, model identity checked at submission and admission |
| Scheduling | Requests that fit can bypass an oversized pending request, with an eight-bypass aging limit to prevent indefinite starvation |
| Downloads | Temporary output, advertised-length verification, propagated errors and URL/size/SHA-256 cache receipts; cancellation cleanup; CLI preserves owner/repository, nested filenames and the requested output directory |
| Memory accounting | Live/peak reservation accounting separated from host weight and session KV payload bytes |

The format fixtures use synthetic values, not redistributed trained weights. See [fixture provenance](../tests/fixtures/format/README.md) and [reference generation instructions](../tests/reference/README.txt).

## Optimizations and their limits

CPU matvec selects AVX2/FMA at runtime when both are available and otherwise uses the scalar implementation. The default is one thread. `nexus run` and `nexus bench --threads N` create a private bounded pool; sufficiently small matrices remain serial. The server uses the one-thread default. `RAYON_NUM_THREADS` does not configure the private matvec pool. SIMD changes floating-point accumulation order, so numeric tests use explicit error tolerances and greedy-token comparisons.

KV storage uses a ring instead of shifting the whole context on each full-cache insertion. Packed fields use bounded byte extraction. Grouped-query attention decodes a KV head once for its associated query heads. Regression matrices cover every KV format, ragged dimensions, repeated wrapping and clear/reuse. Ring storage does not add sliding-window model support or extend the model's context/RoPE limit.

CUDA keeps immutable F32 weights and reuses input/output device buffers under one execution lock. Retained workspace is limited to 32 MiB; replacement growth can temporarily hold up to 64 MiB of live buffer objects. Logical slices prevent stale capacity tails from being returned. Each call still copies activation input/results and synchronizes. Attention, RoPE, normalization, sampling and KV storage remain on the CPU. These changes are not GPU attention, packed-weight inference or fused multi-request batching.

## Reproduce the acceptance gates

The recorded local run on Arch Linux x86_64 passed 230 CPU tests and 240
all-features tests with a CUDA device required, plus two separately enabled
external tokenizer tests. These largely overlap and must not be added into a
unique-test total. Format, no-default-feature checking, all-target/all-feature
Clippy, release builds and the Rust 1.88 all-target/all-feature check passed. The same 230 CPU tests also passed under Rust 1.88; this is an overlapping compatibility run, not 230 additional unique cases.

Hardware validation used an NVIDIA RTX 5070 (12,227 MiB reported VRAM), driver
620.70.06 and CUDA 13.4.3 Compute Sanitizer. Six CUDA integration tests completed
with `ERROR SUMMARY: 0 errors`. The real-model HTTP smoke passed all 19 checks.
The SmolLM2-135M-Instruct F32 GGUF was 541,758,112 bytes, SHA-256
`359c79b411b06f2a9f5b392273ab70bf766cecca44480867be80454e973041f6`.
CPU/CUDA greedy token sequences and the prefill top-ten token ranking matched;
maximum absolute prefill logit difference was 0.000019073486328125. This is a
specific numerical comparison, not a model-quality evaluation.

Build prerequisites are listed in the [README](../README.md). Rust 1.88 is the minimum supported toolchain.

```sh
cargo fmt --all -- --check
cargo test --workspace --locked
cargo check --workspace --no-default-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
NEXUS_REQUIRE_CUDA=1 cargo test --workspace --all-features --locked
cargo +1.88.0 check --workspace --all-targets --all-features --locked
```

CUDA tests may skip device work without a compatible GPU unless `NEXUS_REQUIRE_CUDA=1` is set. A skipped test is not hardware validation. The external tokenizer tests are ignored by default because they need separately obtained assets:

```sh
NEXUS_LLAMA_VOCABS=/path/to/llama.cpp/models \
  cargo test --locked --test external_tokenizer_parity -- --ignored
```

See [reference instructions](../tests/reference/README.txt) for pinned tokenizer revisions and independently generated goldens. Tokenizer compatibility with Qwen/Llama-3 does not add inference architecture support.

For an actual converted model and an NVIDIA GPU:

```sh
cargo build --release --workspace --features cuda --locked
cargo run --release --features cuda --locked --example compare_cuda -- model-f32.gguf
python3 tests/reference/cuda_server_smoke.py model-f32.gguf
NEXUS_REQUIRE_CUDA=1 cargo test --locked --features cuda --test cuda_execution --no-run
# Run the emitted test executable:
NEXUS_REQUIRE_CUDA=1 compute-sanitizer --tool memcheck --error-exitcode 99 \
  /path/to/cuda_execution-test --test-threads=1
```

The CUDA regression covers ragged shapes, F64 reference values, full-engine logits/tokens, growing and shrinking workspaces, budget rejection, concurrent callers, weight replacement and terminal-error handling. Synthetic error tests do not inject an actual hardware fault.

## Parser mutation corpus

```sh
cargo build --release --locked --example parser_corpus
./target/release/examples/parser_corpus --cases 20000 --seed 42 \
  --crash-dir /tmp/nexus-parser-cases
```

The recorded corpus used three seeds (5640011205539672630, 20261115 and 42), 20,000 mutations each, and eleven positive controls per seed. All 60,000 mutations and 33 positive controls completed without a panic or checked invariant failure. Processes had a 512 MiB address-space limit, 90 CPU-second limit and 120-second wall timeout. The corpus mutates small GGUF and safetensors files, checking stream/mmap agreement for accepted GGUFs and supported tensor loading. It is deterministic mutation testing, not coverage-guided fuzzing or a proof that arbitrary hostile files are safe.

## Benchmarks and service soak

```sh
./target/release/nexus bench --model model-f32.gguf --backend cpu-simd \
  --kv-quant none --prompt-lens 32,128 --max-tokens 16 \
  --warmup 3 --iterations 10 --threads 1 --json
./target/release/nexus bench --model model-f32.gguf --backend cuda \
  --kv-quant none --prompt-lens 32,128 --max-tokens 16 \
  --warmup 3 --iterations 10 --json
python3 tests/reference/server_soak.py --binary target/release/nexus-server \
  --model model-f32.gguf --output /tmp/nexus-soak-new \
  --duration-seconds 1800 --backend cuda --concurrency 1,2,4,8
```

Benchmark JSON includes the model hash, effective backend/kernel/thread limit, real generated token count, prefill/decode/first-token timing and nearest-rank percentiles. Loading is excluded from generation latency. Synthetic token IDs measure execution, not language quality. Always compare matching model hash, KV precision, output length and hardware. A small sample's p95 is noisy; ten samples' nearest-rank p95 is the maximum.

The soak starts and stops only its own localhost process. It exercises JSON/SSE, cancellation after response headers, slow readers, concurrency, rejected requests and repeated load/reload/unload. It records per-process RSS/high-water RSS, CPU time, GPU memory and runtime/KV/queue counters. Slow reading alone does not prove TCP/app-buffer saturation. Empty final queues and stable sampled memory are useful observations, not a general no-leak guarantee or a throughput SLA.

## Recorded before/after performance

The same model and settings above were measured on an Intel Core Ultra 5 245KF
and RTX 5070. The baseline was the protected local correctness-package snapshot
before this optimization pass, **not the original GitHub main revision**. Each
cell has ten measured samples in two batches of five, three warmups per batch,
and AB then BA ordering. Prompts contain 32 or 128 synthetic token IDs; every
sample generated exactly 16 tokens with F32 KV and context 4096.

| Backend | Prompt tokens | Before p50 | After p50 | Before p95 | After p95 | p50 ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cpu-simd | 32 | 3.1499 s | 1.1622 s | 3.4082 s | 1.2381 s | 2.71x |
| cpu-simd | 128 | 10.1138 s | 3.6640 s | 11.5421 s | 3.8990 s | 2.76x |
| cuda | 32 | 0.2553 s | 0.2011 s | 0.3034 s | 0.2138 s | 1.27x |
| cuda | 128 | 0.8912 s | 0.7239 s | 0.9717 s | 0.8145 s | 1.23x |

The CPU process maximum RSS was 598,720–600,252 KiB before and
599,420–599,708 KiB after. An earlier candidate eagerly initialized unused KV
pages and increased RSS to about 944,000 KiB; that regression was caught and
fixed before these final measurements. Typed fallible zeroed allocations now
preserve allocator/OS demand-zero behavior. The CPU memory result is unchanged
within this small sample, not a general RAM reduction claim.

Extra CPU thread-limit runs (32 input + 16 output tokens, two warmups and five
measurements each) had p50 totals of 1.0695 s with two threads and 1.0519 s with
four. Those smaller later batches are not mixed into the before/after table.
GPU process RSS varied between batches; no generalized GPU memory-reduction
claim is made.

Model loading is outside these generation times. Child-process maximum RSS
includes load/setup and comes from Linux wait4. OS load and CPU/GPU frequency
were not isolated; other development builds and inference were paused during
the comparison. Results describe this model/hardware/settings combination,
not a universal speedup or service-capacity guarantee. Binary/model hashes,
all timing samples and per-child RSS are in the [machine-readable record](validation/2026-11-15-benchmarks.json).

After these timing runs, a service-only lifecycle change moved model loading
from expiring Tokio blocking threads to a persistent bounded worker. The
CPU/KV/CUDA computation paths were unchanged. The timing record preserves the
measured binary hash and separately records the final release hashes. A later
dependency advisory refresh and CLI download-identifier correction also
required a rebuild; those earlier timing samples are not attributed to the new
binary. The inference algorithms were unchanged by these later edits.

## Model-loader memory investigation

Repeated load/reload/unload with idle gaps exposed host RSS retention. An
isolated diagnostic showed that live glibc arena allocations (`uordblks`) stayed nearly flat
between reloads (477.78 to 477.84 MB), while free arena memory grew by about
455.5 MB; mapped malloc bytes (`hblkhd`) stayed at 149.64 MB. After unload, live arena allocations were about 23.43 MB while the allocator
still retained about 1,364.79 MB of free memory. Thus this observation was free
arena retention, not evidence that old model-weight references remained live.
Immediate repeated loads were stable; waiting beyond the blocking thread's idle
expiry reproduced the increase, also without inference.

The server now constructs loaded and empty engines on one long-lived worker per
server state, with a one-item queue, async replies, per-job panic handling and
cancelled queued-work skipping. The existing lifecycle lock and atomic model
installation remain. Dropping the last sender closes the worker without joining
it on the async reactor. No process-global malloc tuning or forced malloc trim
is used. Regression tests cover idle reuse, serialization, queue capacity,
cancellation, panic recovery and nonblocking shutdown.

## Recorded 30-minute service acceptance

The final server binary (`b7fbfa8c8fea016d23181c550a55adb6aff8d25130b1de44cfddd41b562449f5`)
completed 1,805.22 seconds of mixed CUDA workload with the model above, F32 KV,
context 256 and concurrency 1/2/4/8. The [sanitized run record](validation/2026-11-15-soak.json)
contains the settings, executed harness hash, counts and memory series.

| Observation | Result |
| --- | ---: |
| HTTP calls, including expected rejection/lifecycle cases | 3,584 |
| Nonempty completed JSON/SSE/slow-reader generations | 2,432 |
| Server-confirmed cancellations | 512 |
| Successful load / reload / unload operations | 64 each |
| Unexpected request or sampling failures | 0 |
| Final queued/running requests, active sessions and session KV bytes/pages | 0 |

The worker/backend remained healthy and the owned test server exited after the
final drain. Expected 400/401/503 responses were negative tests. This is a
protocol/lifecycle acceptance run; development builds and package checks
sometimes overlapped, so its timings are not an isolated capacity benchmark.

Linux high-water RSS was 1,982,400 KiB; sampled RSS peaked at 1,862,544 KiB.
At final drain, RSS was 1,261,644 KiB. Supplemental swap observations started
at 15:40 UTC, about fourteen minutes into the workload: the last observation
contained 461,448 KiB of swap, making RSS + swap 1,723,092 KiB. The decrease in
RSS alone is therefore not a memory-saving result, and there is no complete
first-half swap series. Sampled GPU memory peaked at 1,218 MiB during replacement
and ended at 706 MiB. Protocol success, empty final queues and these bounded
observations do not prove absence of all leaks or guarantee other model sizes.

## Dependency advisory check

The locked dependency graph was checked with cargo-audit 0.23.0 against the
RustSec database revision recorded in the [dependency audit](validation/2026-11-15-dependencies.json).
The initial scan reported five vulnerability advisories and four unsoundness
warnings. Patch updates addressed crossbeam-epoch, OpenSSL, anyhow, memmap2,
rustix and Tokio. Moving reqwest from 0.11 to 0.12 removed its old h2 branch;
the locked h2 is now 0.4.20. The resulting scan passed `cargo-audit audit --deny
unsound` with zero vulnerability matches and zero unsoundness warnings. No
advisory was ignored.

One maintenance warning remains: [RUSTSEC-2024-0436](https://rustsec.org/advisories/RUSTSEC-2024-0436.html)
for `paste 1.0.15`, used transitively by tokenizers. This is tracked as dependency
maintenance, not counted as a fixed vulnerability or concealed by an exception.
CI runs the advisory check on pushes and pull requests. Its scanner archive is
pinned by version and SHA-256. An advisory match does not establish that Nexus
calls the affected function; a clean lockfile scan does not audit system TLS,
CUDA, Python/Qt or prove that dependencies have no undiscovered flaws.

## Compatibility changes

- Regenerate Q2_K/Q4_K files written by older Nexus versions from the original model. Their bit layout was incompatible; no heuristic migration is performed.
- Runtime weight storage is F32 even when the GGUF file is quantized. File size is not the required RAM/VRAM. Non-F32 runtime storage options return an error.
- `QuantEngine::quantize_weights`/`dequantize` return `io::Result`; backend `execute` returns owned `KernelOutput` or an error.
- Construct `KvCache` through `new`/`try_new`. Its physical backing storage wraps; use `get_all` for logical oldest-to-newest order. `try_new` exposes invalid-size/allocation errors.
- HTTP request model names must match the serving model. Servers bind to loopback by default; network exposure requires an explicit host setting.
- Desktop and distribution package requirements are in [Linux desktop](linux-desktop.md).

The [roadmap](roadmap.md) separates completed work from the remaining performance, reliability and platform goals.
