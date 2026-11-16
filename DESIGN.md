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
