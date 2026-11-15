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


## Suggested implementation order

1. **Freeze the quality baseline.** Extend the existing pinned reference assets and
   real-model checks before changing weight precision. Add ordinary text/chat
   prompts, longer outputs, GQA/context boundaries and multiple supported LLaMA
   sizes. Keep model/tokenizer revisions and precision explicit. A tokenizer
   fixture for another family does not qualify its inference architecture.
2. **Develop packed CPU weights and GPU residency in separate branches.** The CPU
   work should introduce an explicit weight-storage representation before adding
   block matvec kernels; the current backend contract accepts F32 vectors. The
   GPU work should first measure transfer bytes, then move activations/KV and
   attention to the device. Preserve atomic replacement and bounded workspaces
   in both paths. Merge each only after its independent numerical reference and
   matched-model memory/latency comparison pass.
3. **Validate service behavior with the resulting memory model.** Re-run mixed
   short/long requests, cancellation, slow consumers and repeated replacement.
   Record RSS, swap, allocator observations and per-process VRAM separately.
   Bound admission/cancellation latency and verify empty final queues and KV
   ownership. Compare throughput only under an isolated, documented workload.
4. **Add actual batching after backend support exists.** Scheduler interleaving
   alone is not batching. Matrix/matrix execution must preserve each session's
   RNG, KV, stop handling and model identity; publish both throughput and p95
   latency changes for mixed request sizes.

Parser fault testing and desktop distribution can proceed alongside the compute
work. The parser track owns minimized malformed inputs and interrupted/disk-full
I/O recovery. The desktop track owns Wayland/X11/HiDPI/accessibility checks,
clean install/upgrade/uninstall and signing. Dependency maintenance owns the
remaining transitive `paste` warning and verifies tokenizer goldens after its
upstream replacement. None of these tracks should weaken a correctness gate to
obtain a performance result.

A reviewable change should include the baseline revision/binary hash, the final
source revision, exact reproduction command, model hash, numerical tolerance,
measured result and compatibility impact. Publish a supported feature only when
its acceptance evidence exists; otherwise retain the explicit unsupported path.
