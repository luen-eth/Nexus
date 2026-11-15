# Engineering plan

The current release establishes a checked LLaMA inference path, CPU/CUDA numerical regressions, strict format handling, bounded HTTP scheduling and a Linux desktop controller. This plan describes remaining work; it does not imply that the features already exist.

| Priority | Work | Prerequisite | Acceptance evidence |
| --- | --- | --- | --- |
| P1 | Expand quality and model coverage | Current tokenizer/converter/logit references | Pinned real prompt corpora, multiple supported LLaMA sizes, explicit precision tolerances and quality comparison against independent inference |
| P1 | Packed-weight CPU matvec | Correct GGML decoders and SIMD baseline | Avoid full F32 expansion for supported blocks; independent numerical oracle, RSS reduction and matched-model latency measurements |
| P1 | GPU attention and resident activation/KV | Current CUDA workspace and parity suite | Device-side kernels with CPU reference, mask/context/GQA coverage, Compute Sanitizer and reduced measured transfer bytes |
| P1 | Wider parser and I/O fault testing | Deterministic mutation corpus | Coverage-guided GGUF/safetensors fuzzing with resource limits; reproducible minimized failures; disk-full/interrupted-write/restart tests |
| P1 | Long-running service resource tests | Existing mixed-load soak harness | Multiple model sizes/KV formats, longer repeated runs, drain-state RSS/VRAM trend analysis and bounded cancellation/admission latency |
| P2 | Actual batched inference | Efficient matrix/matrix backend and GPU KV | Several requests share matrix execution; isolated RNG/KV/stop behavior, fairness under mixed request sizes and throughput/p95 tradeoffs |
| P2 | Incremental tokenizer decoding | Full-prefix correctness references | Preserve Unicode, stop strings and decoder revision behavior; measure the reduction in long-output detokenization cost |
| P2 | Remove unmaintained transitive crates | RustSec check and current tokenizer references | Upstream tokenizers migration away from paste, unchanged tokenizer goldens and no advisory suppression |
| P2 | Broader desktop compatibility | Arch and Debian-family package paths | Wayland/X11 interaction, HiDPI, accessibility, clean install/upgrade/uninstall, package signatures and distro CI |
| P2 | ARM and other GPU backends | Portable CPU fallback and backend contract | Hardware-backed tests and real-model numerical parity before advertising Metal/Vulkan/WebGPU inference |
| P3 | Speculative decoding and additional architectures | Quality/performance reference corpus | Correct target-distribution behavior and measured net benefit; architecture-specific load/attention/RoPE tests |

For each implementation, preserve the existing regression gates, capture an unchanged baseline binary and source revision, and record the model hash and execution settings. Report end-to-end improvements separately from microbenchmarks. Changes that increase precision loss or memory demand need an explicit compatibility note. Production operation, remote deployment and model distribution are separate from this source and desktop package release.
