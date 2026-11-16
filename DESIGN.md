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

**Toolchain gap:** `xcrun metal` is not found: only Command Line Tools are installed, not full Xcode. Offline `.metal` -> `.metallib` compilation therefore won't work yet. Either install Xcode (plus the Metal Toolchain component) or compile MSL at runtime with `-newLibraryWithSource:` (fine for development; the tensor-ops header must then be importable by the runtime compiler, to be tested in M2).

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

## 4. Roadmap

1. **M0 (this skeleton)**: API, reference backend, stubs, Makefile, test, bench.
2. **M1 SME FP32 GEMM**: streaming-mode kernel using `svmopa` into ZA tiles (4 x 16x16 FP32 tiles at SVL=512), packing, cache blocking. Benchmark against Accelerate `cblas_sgemm`.
3. **M2 GPU GEMM**: MSL kernel with `simdgroup_matrix`, then Metal 4 tensor-ops variant; FP16/BF16.
4. **M3 ANE**: Core ML MLProgram matmul builder + plan cache; FP16 only.
5. **M4 Planner**: microbenchmarks at install time, split heuristic, co-execution of 2–3 engines.
6. **M5**: batching, transposes, more dtypes, Swift/Python bindings.

## 5. Testing & benchmarking
- Every backend checked against `ref` with dtype-appropriate tolerance (relative Frobenius error).
- Bench reports GFLOP/s per backend and per shape; compare to Accelerate as the baseline to beat.
- Thermals matter on a laptop: bench with warmup and report sustained, not peak.
