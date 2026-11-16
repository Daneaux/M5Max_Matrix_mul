// How throughput scales with the amount of hardware used.
//
//   ./build/bench_scaling            (MX_SCALE_N=4096 sets the CPU matrix size)
//
// CPU (SME GEMM, F32): sweep the worker-thread count (GemmOptions::threads) at a fixed size. Threads cannot be
//   pinned: the OS picks cores. 6 threads is one cluster of Super cores; beyond 6 the extra threads land on the
//   Performance cores (separate clusters, each with its own SME unit). Speedup is relative to 1 thread.
//
// GPU: you cannot choose GPU cores, but each threadgroup runs on one core, so launching T threadgroups uses at most
//   T cores. We run C[64 x 64T] = A[64 x K] . B[K x 64T] (one 64x64 tile per threadgroup, K=8192), so every
//   threadgroup does identical work, and sweep T. TFLOP/s should grow ~linearly until the GPU is full, then flatten
//   (a core can hold several threadgroups, so saturation may come at more than the core count).
//   Two modes: 'shared' = every threadgroup reads the same A and B tiles (stay in cache; isolates compute scaling),
//   'unique' = each threadgroup streams its own B columns from DRAM (becomes memory-bound at high T; compare with
//   bench_bandwidth). Dispatches are serial in one command buffer, each followed by a small gap; the gap matters
//   most at T=1.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include "mx/mx.h"

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

// ------------------------------------------------------------------ CPU
static void cpu_scaling(size_t n) {
  float* A = (float*)mx::alloc_unified(n * n * 4);
  float* B = (float*)mx::alloc_unified(n * n * 4);
  float* C = (float*)mx::alloc_unified(n * n * 4);
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> d(-1, 1);
  for (size_t i = 0; i < n * n; ++i) { A[i] = d(rng); B[i] = d(rng); C[i] = 0; }
  mx::MatrixView a{A, n, n, n}, b{B, n, n, n}, c{C, n, n, n};
  printf("CPU: SME FP32 GEMM, n=%zu, vs worker threads (hint: 6 = Super cluster, 12 Performance cores beyond that)\n", n);
  printf("%8s | %10s %10s | %8s %10s\n", "threads", "ms", "TFLOP/s", "speedup", "efficiency");
  double t1 = 0;
  for (int th : {1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 18}) {
    mx::GemmOptions o; o.backend = mx::BackendId::SME; o.threads = th;
    if (mx::gemm(c, a, b, o) != 0) { printf("SME unavailable\n"); break; }
    std::vector<double> t;
    for (int i = 0; i < 7; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      mx::gemm(c, a, b, o);
      t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    const double s = median(t), tf = 2.0 * n * n * n / s / 1e12;
    if (th == 1) t1 = s;
    printf("%8d | %10.2f %10.3f | %7.2fx %9.0f%%\n", th, s * 1e3, tf, t1 / s, 100.0 * t1 / s / th);
  }
  mx::free_unified(A); mx::free_unified(B); mx::free_unified(C);
}

// ------------------------------------------------------------------ GPU
static const char* kSrc = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct Dims { uint M, N, K, share; };
template <typename E>
inline void mm_impl(device E* Ap, device E* Bp, device E* Cp, constant Dims& d, uint2 tg) {
  using T2 = tensor<device E, dextents<int32_t, 2>, tensor_inline>;
  T2 A(Ap, dextents<int32_t, 2>(d.K, d.M), array<int32_t, 2>{1, int32_t(d.K)});
  T2 B(Bp, dextents<int32_t, 2>(d.N, d.K), array<int32_t, 2>{1, int32_t(d.N)});
  T2 C(Cp, dextents<int32_t, 2>(d.N, d.M), array<int32_t, 2>{1, int32_t(d.N)});
  constexpr auto desc = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent));
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = A.slice(0, tg.y * 64);
  auto mB = B.slice((d.share ? 0u : tg.x) * 64, 0);  // share: every tile reads the same B columns
  auto mC = C.slice(tg.x * 64, tg.y * 64);
  op.run(mA, mB, mC);
}
kernel void mm_f32(device float* A [[buffer(0)]], device float* B [[buffer(1)]], device float* C [[buffer(2)]],
                   constant Dims& d [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) { mm_impl<float>(A, B, C, d, tg); }
kernel void mm_f16(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device half* C [[buffer(2)]],
                   constant Dims& d [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) { mm_impl<half>(A, B, C, d, tg); }
)MSL";

template <typename T>
static void gpu_scaling(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLComputePipelineState> ps, const char* name, int cores, bool share) {
  const uint32_t K = 8192;
  const int Ts[] = {1, 2, 4, 8, 16, 20, 32, 40, 48, 64, 80, 96, 128, 160, 192};
  const uint32_t Tmax = 192;
  id<MTLBuffer> A = [dev newBufferWithLength:(size_t)64 * K * sizeof(T) options:MTLResourceStorageModeShared];
  id<MTLBuffer> B = [dev newBufferWithLength:(size_t)K * 64 * Tmax * sizeof(T) options:MTLResourceStorageModeShared];
  id<MTLBuffer> C = [dev newBufferWithLength:(size_t)64 * 64 * Tmax * sizeof(T) options:MTLResourceStorageModeShared];
  std::mt19937 rng(2);
  std::normal_distribution<float> nd(0.f, 1.f / std::sqrt((float)K));
  T* a = (T*)A.contents; T* b = (T*)B.contents;
  for (size_t i = 0; i < (size_t)64 * K; ++i) a[i] = (T)nd(rng);
  for (size_t i = 0; i < (size_t)K * 64 * Tmax; ++i) b[i] = (T)nd(rng);

  printf("\nGPU %s, %s: C[64 x 64T] = A[64 x %u] . B[%u x 64T], one 64x64 tile per threadgroup (GPU has %d cores)\n", name,
         share ? "SHARED inputs (cache-resident, compute-bound)" : "UNIQUE B per threadgroup (streams DRAM)", K, K, cores);
  printf("%8s | %10s %10s | %8s %10s | %s\n", "T (tg)", "us/disp", "TFLOP/s", "speedup", "per-tg", "check");
  double t1 = 0;
  for (int Tn : Ts) {
    const uint32_t N = 64 * Tn;
    struct { uint32_t M, N, K, share; } dims{64, N, K, share ? 1u : 0u};
    const int R = 40;
    auto run = [&]() {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:ps];
      [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:B offset:0 atIndex:1]; [e setBuffer:C offset:0 atIndex:2];
      [e setBytes:&dims length:sizeof dims atIndex:3];
      for (int r = 0; r < R; ++r)
        [e dispatchThreadgroups:MTLSizeMake(Tn, 1, 1) threadsPerThreadgroup:MTLSizeMake(ps.threadExecutionWidth * 4, 1, 1)];
      [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
      return (cb.GPUEndTime - cb.GPUStartTime) / R;
    };
    run();
    std::vector<double> t;
    for (int i = 0; i < 7; ++i) t.push_back(run());
    const double s = median(t), tf = 2.0 * 64 * N * K / s / 1e12;
    if (Tn == 1) t1 = tf;
    // spot-check a few elements of the last tile against a double-precision reference
    const T* c = (const T*)C.contents; double worst = 0;
    for (int s2 = 0; s2 < 6; ++s2) {
      const uint32_t i = (s2 * 11) % 64, j = (N - 1 - (s2 * 37) % 64), jb = share ? j - (N - 64) : j;  // B column read
      double r = 0; for (uint32_t k = 0; k < K; ++k) r += (double)(float)a[(size_t)i * K + k] * (float)b[(size_t)k * N + jb];
      worst = std::max(worst, std::fabs((double)(float)c[(size_t)i * N + j] - r) / (std::fabs(r) + 0.05));
    }
    printf("%8d | %10.1f %10.2f | %7.1fx %10.3f | %s\n", Tn, s * 1e6, tf, tf / t1, tf / Tn, worst < 0.02 ? "ok" : "BAD");
  }
}

int main() {
  size_t n = 4096;
  if (const char* e = getenv("MX_SCALE_N")) n = strtoull(e, nullptr, 10);
  @autoreleasepool {
    cpu_scaling(n);
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev) return 0;
    id<MTLCommandQueue> q = [dev newCommandQueue];
    MTLCompileOptions* o = [MTLCompileOptions new];
    o.languageVersion = MTLLanguageVersion4_0;
    NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:o error:&err];
    if (!lib) { fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
    auto f16 = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm_f16"] error:&err];
    auto f32 = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm_f32"] error:&err];
    int cores = 40;  // M5 Max (queried via ioreg: gpu-core-count); the sweep marks 40
    gpu_scaling<_Float16>(dev, q, f16, "FP16", cores, true);
    gpu_scaling<_Float16>(dev, q, f16, "FP16", cores, false);
    gpu_scaling<float>(dev, q, f32, "FP32", cores, true);
    gpu_scaling<float>(dev, q, f32, "FP32", cores, false);
  }
  return 0;
}
