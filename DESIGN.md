# mxlib — a matrix library for Apple M5 Max

## 1. The three engines (verify each on your machine)

| Engine | What it is | How you reach it | Best at |
|---|---|---|---|
| **CPU SME/SME2** | Arm Scalable Matrix Extension unit shared per CPU cluster (ZA tile array, streaming SVL = 512 bit here). Documented ISA. | Clang intrinsics (`arm_sme.h`, `__arm_streaming`, `__arm_new("za")`), or Accelerate/BLAS which uses it internally | FP32/FP64 and low-latency small/medium GEMM; no launch overhead; shares unified memory |
| **GPU matrix hardware** | `simdgroup_matrix` in all Apple GPUs; on M5 the GPU cores also have per-core neural accelerators exposed via Metal 4 tensor ops (`MetalPerformancePrimitives`) | Metal / MSL kernels, `MTLTensor`, MPSGraph | Large GEMM, FP16/BF16/INT8; highest raw throughput |
| **ANE (Neural Engine)** | Fixed-function inference accelerator | **Only via Core ML** (no public ISA). You compile a tiny MLProgram (a `matmul`/`linear`) and run it | FP16/INT8 matmul at best perf/W; poor for dynamic shapes, high dispatch latency |

### 1a. Verified against this machine's SDK (macOS SDK 27.2, Command Line Tools, clang 21)

| Engine | Confirmed | Evidence |
|---|---|---|
| CPU SME | Yes | `sysctl hw.optional.arm.FEAT_SME/FEAT_SME2 = 1`, `sme_max_svl_b = 64` (512-bit); `arm_sme.h` ships with clang 21 (includes `svmopa`) |
| GPU tensor ops | Yes (API); hardware mapping not | `MetalPerformancePrimitives.framework` has `MPPTensorOpsMatMul2d.h`: `matmul2d_descriptor(m, n, k, ..., relaxed_precision)`, `matmul2d<desc, execution_simdgroups<N>>::run(A, B, C)`, cooperative tensors, scopes `execution_thread/simdgroup/simdgroups<N>`. Host side: `MTLTensor.h`, `-[MTLDevice newTensorWithDescriptor:]` (macOS 26.0+). Availability macros gate on deployment target 26.2 / 27.0. The headers do **not** say these ops run on M5 "neural accelerators"; that mapping is still unconfirmed and must be checked by benchmarking vs `simdgroup_matrix`. |
| ANE | Yes (only route) | `CoreML.framework`: `MLComputeUnitsCPUAndNeuralEngine`. No lower-level API in the SDK. Placement is a request, not a guarantee. |
| Accelerate | Not documented | No SME/streaming mention in Accelerate headers; treat its SME use as an implementation detail and use it only as a baseline. |

Also found: Metal 4 has `MTL4MachineLearningPipeline.h` (ML pipelines encoded on the GPU timeline). Not investigated; it is a Metal/GPU feature, not an ANE path as far as the header shows.

**Toolchain:** Xcode.app is installed but `xcode-select` still points at the Command Line Tools, and the *Metal Toolchain* component is not installed (`xcodebuild -downloadComponent MetalToolchain`). We therefore compile MSL **at runtime** (`newLibraryWithSource:`, language 4.0). This works, including `MetalPerformancePrimitives` and `matmul2d`. Offline `.metallib` compilation would only save the ~tens of ms first-use compile.

Caveats worth knowing before committing to the design:
- ANE cannot be programmed directly; treat it as an opaque "compiled-graph" backend. Shapes/dtypes are baked at compile time, so it needs a plan cache.
- "3 engines" is my reading (SME, GPU, ANE). If you meant AMX + SME + GPU, the CPU side is the same design: AMX is undocumented and only reachable through Accelerate.
- Heterogeneous runs only pay off when the per-engine slice is big enough to beat dispatch overhead. Expect crossover tuning to be most of the work.

## 2. Language decision

**Core in C++20, with a C ABI on top; Metal shaders in MSL; thin Objective-C++ (.mm) shims for Metal and Core ML.** Swift binding is optional later.

- **C++**: Clang's SME intrinsics (`arm_sme.h`, streaming-mode attributes) are first class; Rust has no stable SME intrinsics (inline asm only). Metal-cpp and Obj-C++ mix freely with C++ in one translation unit family.
- **Rust**: great safety story, but SME needs `asm!` with hand-managed streaming mode, and Metal/Core ML go through `objc2` bindings. Worth it only if you value Rust more than velocity here.
- **Swift**: excellent for the Metal/Core ML side and for a friendly API, but cannot express SME kernels; would still need a C/C++ target. Good as a *client* binding.
- **Objective-C**: only as the shim layer (use `.mm`), not as the core.

## 3. Architecture

```
 public C++ / C API (include/mx/mx.h)
        │   mx::gemm(C, A, B, alpha, beta, opts)
        ▼
 ┌───────────────┐   ┌────────────────────┐
 │ Planner       │──▶│ Cost model + tuned │  per (M,N,K,dtype) → split plan
 │ (scheduler)   │   │ crossover table    │  persisted to ~/.cache/mxlib
 └──────┬────────┘   └────────────────────┘
        │ partitions C by rows/cols into tiles, one task per engine
        ▼
 ┌──────────┬──────────────┬──────────────┐
 │ SME      │ GPU (Metal)  │ ANE (CoreML) │   all implement `Backend`
 └──────────┴──────────────┴──────────────┘
        ▲ unified memory: one page-aligned allocation, zero-copy
          (MTLBuffer bytesNoCopy / MLMultiArray over the same pages)
```

Key abstractions (see `src/backend.h`):
- `Backend`: `supports(GemmDesc)`, `gemm(...)` async returning a `Future`, `estimate_ns(GemmDesc)`.
- `Buffer`: page-aligned unified-memory allocation, importable by all three engines without copying.
- `Planner`: static split (initial) → measured throughput feedback (later). Splits C into row-blocks sized proportionally to each engine's measured GFLOP/s.
- Reference backend is always present; it is the correctness oracle for tests.

### Cache/coherence rule
Engines write disjoint tiles of C, so no cross-engine synchronization is needed beyond a join. A and B are read-only during a call.

## 3a. As built (M2/M3)

**GPU (`src/backend_gpu.mm`)**: `matmul2d` kernels (FP32 and FP16), 64x64 tile, 4 simdgroups (tuned: 64x32 was 30-40% slower at 2k+). Tensors are built *inside* the kernel as strided `tensor_inline` views over plain buffers (the legacy compute encoder has no `setTensor`), so padded leading dimensions work. Zero-copy for `mx::alloc_unified` memory (registry in `mx.cpp`, `bytesNoCopy` buffers cached per allocation); other memory is staged with memcpy. alpha/beta via a temporary + epilogue kernel unless alpha=1,beta=0. Synchronous.

**ANE (`src/backend_ane.mm`)**: FP16 only. A Core ML NeuralNetwork model (one batched-matmul layer, no weights) is hand-encoded as protobuf, compiled and loaded with `CPUAndNeuralEngine`, cached per (M,K,N) (32 models). The layer's field number (1045) was found by brute force, not from spec text, so `init()` self-tests numerics **and** uses `MLComputePlan` to confirm the Neural Engine is the preferred device; if either fails the backend reports unavailable. Caveat: placement is verified for the 64^3 self-test model only; other shapes are *requested* on the ANE, and Core ML may place them elsewhere.

## 3b. Measured results (M5 Max, GFLOP/s, square, median of 7)

| n | Accelerate F32 | SME F32 | GPU F32 | GPU F16 | ANE F16 |
|---|---|---|---|---|---|
| 124 | 1368 | 303 | 24 | 27 | 28 |
| 420 | 1616 | 1083 | 933 | 763 | 398 |
| 630 | 1872 | 1514 | 2689 | 2118 | 695 |
| 1420 | 2892 | 2164 | 8990 | 21652 | 1654 |
| 3196 | 2615 | 2712 | 14332 | 40820 | 3161 |

Takeaways: CPU (Accelerate/SME) wins below ~500; the GPU wins above ~600 and its FP16 path is ~3x its FP32 path (consistent with, but not proof of, use of the M5 GPU matrix accelerators). The ANE is the slowest engine at every size in this setup (peaks ~3-4.6 TFLOP/s) and has ~0.25 ms fixed cost; its value is power efficiency and running concurrently with the CPU/GPU, which the planner does not exploit yet. GPU numbers use zero-copy memory; with plain heap memory (`MX_BENCH_HEAP=1`) the GPU pays for staging copies.

## 3c. Memory bandwidth (`make bandwidth`, `bench/bench_bandwidth.mm`)

All engines share one DRAM pool but get very different shares of it (1 GB working set, GB/s; copy counts read+write):

| Engine | Read | Write | Copy |
|---|---|---|---|
| CPU, 1 thread | 70 | 121 | 120 |
| CPU, 6 Super cores | 208 | 282 | 272 |
| CPU, all 18 threads | 266 | 278 | 354 |
| GPU (zero-copy buffers) | 473 | 466 | 437 |
| ANE via Core ML (end to end, effective) | 3-6 | 6 | n/a |

Cache-sized sets (1-16 MB) show cache bandwidth on the CPU (up to ~1.8 TB/s read, all threads), not DRAM. The ANE cannot access memory directly, so its figure is end to end through Core ML (includes Core ML's copies/conversion and some compute, and placement for those shapes was not verified): a lower bound, but 10-100x below the other engines, which is a large part of why the ANE loses on matmul here. The chip's rated peak bandwidth was not checked.

## 3d. ANE matmul speed with the matrix already loaded (`make ane-resident`, `bench/bench_ane_resident.mm`)

One Core ML model chains L matmuls (X1 = A.A, X_L = X_{L-1}.A), so one prediction loads A once. Fitting t(L) = t0 + L*t1 separates per-multiply time (t1) from fixed per-call cost (t0). FP16, all layers confirmed on the Neural Engine via MLComputePlan, chain correctness checked with a permutation matrix.

| n | t1 per matmul (ms) | t0 fixed per call (ms) | TFLOP/s |
|---|---|---|---|
| 256 | 0.003 | 0.20 | 11.0 |
| 512 | 0.015 | 0.37 | 17.7 |
| 1024 | 0.139 | 1.0 | 15.5 |
| 2048 | 1.26 | 4.0 | 13.6 |
| 4096 | 13.5 | 15.2 | 10.2 |

So the ANE's compute rate (10-18 TFLOP/s) is 3-5x the end-to-end rate in 3b; at n=4096 about half of a single call is load/copy/Core ML overhead. The GPU FP16 path (~41 TFLOP/s) is still faster, but that number was not measured with a chained/resident test, so the comparison is not apples to apples. We cannot observe whether the ANE re-reads A from DRAM between layers; the drop in TFLOP/s at large n suggests memory limits it there.

## 3e. GPU matmul speed with the matrix already resident (`make gpu-resident`, `bench/bench_gpu_resident.mm`)

Same experiment as 3d on the GPU: one command buffer of L dependent `matmul2d` dispatches (X1 = A.A, X_L = X_{L-1}.A, two ping-pong buffers, A in one buffer), GPU-timestamp timing, permutation-matrix correctness check. The `@L=10` / `@L=100` columns (time / L) give burst vs sustained rate; the fitted t0 goes negative at large n because per-matmul time grows with L.

**Run-to-run variance is large** (up to ~1.5-2x) on this machine, which had load average 4-7 from other processes and was running repeated benchmarks; no thermal warning was recorded. Ranges over four runs, TFLOP/s per matmul, n = 1024..4096:

| Engine | Range | Best run |
|---|---|---|
| GPU FP16 | ~24-59 | ~59 (n=2048) |
| GPU FP32 | ~7-11 | ~13 |
| ANE FP16 (3d, single run) | 10-18 | 17.7 |

The GPU FP16 path is ~2-4x faster than the ANE even in the noisy runs; GPU FP32 is about level with ANE FP16. Re-measure on an idle machine after a cooldown before quoting precise figures; the ANE numbers were single runs and may carry the same noise.

## 3f. Scaling with hardware used (`make scaling`, `bench/bench_scaling.mm`)

You cannot pin work to specific cores on any engine. CPU: `GemmOptions::threads` (SME backend, 0 = auto) sets the worker-thread count; the OS places threads. GPU: each threadgroup runs on one core, so the number of launched threadgroups T bounds the cores used. Measured on a busy machine (load average 5-7), so rough.

**CPU, SME FP32, n=4096:** 1 thread 1.35 TFLOP/s; 2: 1.86; 3: 2.32; 4: 2.40; 6: 2.36; 18: 2.45. Throughput saturates at ~2.4 TFLOP/s (1.8x) by 3-5 threads; the 12 Performance cores (threads beyond 6) add nothing. Cause not determined (shared SME unit vs memory vs OS placement). The automatic default of 6 threads is more than needed; 3-4 gives the same result.

**GPU, one 64x64 tile per threadgroup, K=8192, M5 Max (40 cores):**
- Shared cache-resident inputs (compute-bound), FP16: 0.72 TFLOP/s for one threadgroup, linear to 16 (16x), peak ~63 TFLOP/s at T=160 (87x). FP32: 0.30 for one, linear to 16, peak ~14 at T=160. The GPU needs several threadgroups per core (~100-160 total) to fill; launching only 40 does not. Odd non-monotonic dips (FP32 at T=20 and 40) are unexplained; other GPU users (display, other apps) are a guess.
- Unique B per threadgroup (streams DRAM): FP16 plateaus at ~33 TFLOP/s (~500 GB/s of B traffic, the measured bandwidth ceiling), FP32 at ~13.5. Low-reuse shapes are memory-bound, not compute-bound.

## 4. Roadmap

1. **M0 (this skeleton)**: API, reference backend, stubs, Makefile, test, bench.
2. **M1 SME FP32 GEMM (done; ~95-105% of Accelerate at n>=2000, 47% at n=280)**: streaming-mode kernel using `svmopa` into ZA tiles (4 x 16x16 FP32 tiles at SVL=512), packing, cache blocking. Benchmark against Accelerate `cblas_sgemm`.
3. **M2 GPU GEMM (done: tensor-op kernels FP32/FP16)**: Metal 4 tensor-ops (`matmul2d`) kernels, FP32 and FP16. Not done: `simdgroup_matrix` fallback, BF16, async.
4. **M3 ANE (done: Core ML batched matmul, FP16; self-tested)**: hand-encoded NeuralNetwork spec (no coremltools needed) + per-shape model cache; FP16 only. Not done: INT8, MLProgram format, verifying placement for every shape.
5. **M4 Planner**: microbenchmarks at install time, split heuristic, co-execution of 2–3 engines.
6. **M5**: batching, transposes, more dtypes, Swift/Python bindings.

## 5. Testing & benchmarking
- Every backend checked against `ref` with dtype-appropriate tolerance (relative Frobenius error).
- Bench reports GFLOP/s per backend and per shape; compare to Accelerate as the baseline to beat.
- Thermals matter on a laptop: bench with warmup and report sustained, not peak.
