// GPU matmul throughput with the matrix already resident: chained multiplies in ONE command buffer.
//
//   ./build/bench_gpu_resident            (MX_GPU_MAX=4096 to change the largest n)
//
// Same experiment as bench_ane_resident so the two are comparable:  X1 = A.A, X2 = X1.A, ... X_L = X_{L-1}.A
// with A in one buffer for the whole run (X alternates between two buffers). One command buffer holds L
// dependent dispatches; time(L) = t0 + L*t1 is fitted, t1 = time per matmul, t0 = fixed per-submission cost.
// Times are GPU timestamps (GPUEndTime - GPUStartTime), i.e. execution time on the GPU, excluding CPU-side
// encode/commit latency. The caches (GPU L2 / system cache) may keep A on chip between layers; that is part
// of what "resident" means here, and we cannot separate it out.
//
// Note: long chains run the GPU hot for 100s of ms; per-matmul time can rise with L (throttling), which makes the
// fitted t0 negative. The @L=10 / @L=100 columns (time/L) show burst vs sustained rate directly.
//
// Kernel: MetalPerformancePrimitives matmul2d, 64x64 tile, 4 simdgroups (same as src/backend_gpu.mm).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static const char* kSrc = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct Dims { uint n; };
template <typename E>
inline void mm_impl(device E* Ap, device E* Bp, device E* Cp, constant Dims& d, uint2 tg) {
  using T2 = tensor<device E, dextents<int32_t, 2>, tensor_inline>;
  T2 A(Ap, dextents<int32_t, 2>(d.n, d.n), array<int32_t, 2>{1, int32_t(d.n)});
  T2 B(Bp, dextents<int32_t, 2>(d.n, d.n), array<int32_t, 2>{1, int32_t(d.n)});
  T2 C(Cp, dextents<int32_t, 2>(d.n, d.n), array<int32_t, 2>{1, int32_t(d.n)});
  constexpr auto desc = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent));
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = A.slice(0, tg.y * 64);
  auto mB = B.slice(tg.x * 64, 0);
  auto mC = C.slice(tg.x * 64, tg.y * 64);
  op.run(mA, mB, mC);
}
kernel void mm_f32(device float* A [[buffer(0)]], device float* B [[buffer(1)]], device float* C [[buffer(2)]],
                   constant Dims& d [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) { mm_impl<float>(A, B, C, d, tg); }
kernel void mm_f16(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device half* C [[buffer(2)]],
                   constant Dims& d [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) { mm_impl<half>(A, B, C, d, tg); }
)MSL";

struct Ctx {
  id<MTLDevice> dev; id<MTLCommandQueue> q; id<MTLComputePipelineState> f32, f16;
};

// Encodes L chained matmuls; returns GPU execution time in seconds. `out` receives the buffer holding X_L.
static double run_chain(Ctx& c, id<MTLComputePipelineState> ps, id<MTLBuffer> A, id<MTLBuffer> P, id<MTLBuffer> Q,
                        uint32_t n, int L, id<MTLBuffer>* out) {
  id<MTLCommandBuffer> cb = [c.q commandBuffer];
  id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
  [e setComputePipelineState:ps];
  [e setBytes:&n length:4 atIndex:3];
  id<MTLBuffer> prev = A;
  for (int i = 1; i <= L; ++i) {
    id<MTLBuffer> dst = (i % 2) ? P : Q;
    [e setBuffer:prev offset:0 atIndex:0];
    [e setBuffer:A offset:0 atIndex:1];
    [e setBuffer:dst offset:0 atIndex:2];
    [e dispatchThreadgroups:MTLSizeMake(n / 64, n / 64, 1) threadsPerThreadgroup:MTLSizeMake(ps.threadExecutionWidth * 4, 1, 1)];
    prev = dst;
  }
  [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
  if (cb.error) { fprintf(stderr, "gpu error: %s\n", cb.error.localizedDescription.UTF8String); return -1; }
  *out = prev;
  return cb.GPUEndTime - cb.GPUStartTime;
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

template <typename T>
static void fill_random(id<MTLBuffer> b, uint32_t n) {
  T* p = (T*)b.contents;
  std::mt19937 rng(1);
  std::normal_distribution<float> dist(0.f, 1.f / std::sqrt((float)n));  // keeps |X| ~ stable over L multiplies
  for (size_t i = 0; i < (size_t)n * n; ++i) p[i] = (T)dist(rng);
}

template <typename T>
static bool check_permutation(Ctx& c, id<MTLComputePipelineState> ps, const char* name) {
  const uint32_t n = 256;
  bool ok = true;
  for (int L : {1, 7, 20}) {
    id<MTLBuffer> A = [c.dev newBufferWithLength:n * n * sizeof(T) options:MTLResourceStorageModeShared];
    id<MTLBuffer> P = [c.dev newBufferWithLength:n * n * sizeof(T) options:MTLResourceStorageModeShared];
    id<MTLBuffer> Q = [c.dev newBufferWithLength:n * n * sizeof(T) options:MTLResourceStorageModeShared];
    T* a = (T*)A.contents;
    for (size_t i = 0; i < (size_t)n * n; ++i) a[i] = (T)0;
    for (uint32_t i = 0; i < n; ++i) a[i * n + (i + 1) % n] = (T)1;
    id<MTLBuffer> out = nil;
    run_chain(c, ps, A, P, Q, n, L, &out);
    const T* x = (const T*)out.contents;
    int bad = 0;  // X_L = A^(L+1): ones at (i, (i+L+1)%n)
    for (uint32_t i = 0; i < n; ++i) for (uint32_t j = 0; j < n; ++j) if ((float)x[i * n + j] != (j == (i + L + 1) % n ? 1.f : 0.f)) ++bad;
    printf("check %s n=%u L=%-2d (permutation^(L+1)): %s\n", name, n, L, bad ? "WRONG" : "ok");
    ok = ok && !bad;
  }
  return ok;
}

template <typename T>
static void sweep(Ctx& c, id<MTLComputePipelineState> ps, const char* name, int max_n) {
  printf("\n%s: per-matmul GPU time with the matrix already resident (n x n . n x n)\n", name);
  printf("%6s | %8s %8s %8s %8s %8s | %10s %10s | %10s %10s %10s\n", "n", "L=1", "L=10", "L=25", "L=50", "L=100", "t1/matmul", "t0 fixed", "TFLOP/s", "@L=10", "@L=100");
  printf("%6s | %45s | %10s %10s | %10s %10s %10s\n", "", "median GPU ms per submission", "ms", "ms", "(fit)", "(t10/10)", "(t100/100)");
  const int Ls[] = {1, 10, 25, 50, 100};
  for (uint32_t n = 256; n <= (uint32_t)max_n; n *= 2) {
    id<MTLBuffer> A = [c.dev newBufferWithLength:(size_t)n * n * sizeof(T) options:MTLResourceStorageModeShared];
    id<MTLBuffer> P = [c.dev newBufferWithLength:(size_t)n * n * sizeof(T) options:MTLResourceStorageModeShared];
    id<MTLBuffer> Q = [c.dev newBufferWithLength:(size_t)n * n * sizeof(T) options:MTLResourceStorageModeShared];
    fill_random<T>(A, n);
    double t[5];
    for (int k = 0; k < 5; ++k) {
      id<MTLBuffer> out = nil;
      run_chain(c, ps, A, P, Q, n, Ls[k], &out);  // warmup
      std::vector<double> v;
      for (int i = 0; i < 7; ++i) v.push_back(run_chain(c, ps, A, P, Q, n, Ls[k], &out));
      t[k] = median(v);
    }
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int k = 0; k < 5; ++k) { double x = Ls[k]; sx += x; sy += t[k]; sxx += x * x; sxy += x * t[k]; }
    const int m = 5;
    double t1 = (m * sxy - sx * sy) / (m * sxx - sx * sx), t0 = (sy - t1 * sx) / m;
    const double fl = 2.0 * n * n * n;
    printf("%6u | %8.3f %8.3f %8.3f %8.3f %8.3f | %10.4f %10.3f | %10.2f %10.2f %10.2f%s\n", n, t[0] * 1e3, t[1] * 1e3, t[2] * 1e3, t[3] * 1e3,
           t[4] * 1e3, t1 * 1e3, t0 * 1e3, fl / t1 / 1e12, fl / (t[1] / 10) / 1e12, fl / (t[4] / 100) / 1e12,
           t0 < 0 ? "  (t0<0: per-matmul time grows with L, i.e. clocks/thermals; trust the @L columns)" : "");
  }
}

int main() {
  int max_n = 4096;
  if (const char* e = getenv("MX_GPU_MAX")) max_n = atoi(e);
  @autoreleasepool {
    Ctx c;
    c.dev = MTLCreateSystemDefaultDevice();
    if (!c.dev) { fprintf(stderr, "no Metal device\n"); return 1; }
    c.q = [c.dev newCommandQueue];
    MTLCompileOptions* o = [MTLCompileOptions new];
    o.languageVersion = MTLLanguageVersion4_0;
    NSError* err = nil;
    id<MTLLibrary> lib = [c.dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:o error:&err];
    if (!lib) { fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
    c.f32 = [c.dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm_f32"] error:&err];
    c.f16 = [c.dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm_f16"] error:&err];
    if (!c.f32 || !c.f16) { fprintf(stderr, "pipeline failed\n"); return 1; }
    printf("device: %s\n", c.dev.name.UTF8String);
    if (!check_permutation<_Float16>(c, c.f16, "f16") || !check_permutation<float>(c, c.f32, "f32")) return 1;
    sweep<_Float16>(c, c.f16, "FP16", max_n);
    sweep<float>(c, c.f32, "FP32", max_n);
  }
  return 0;
}
