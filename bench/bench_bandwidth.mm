// Memory bandwidth per engine, over a working-set sweep (cache sizes -> DRAM).
//
//   ./build/bench_bandwidth            (MX_BW_MAX_MB=1024 to change the largest working set)
//
// Conventions: "read" counts bytes read, "write" bytes written, "copy" counts read + write bytes.
//  * CPU : NEON loads/stores, 1 / 6 (Super cores) / all threads. QoS is raised but threads cannot be
//          pinned, so the OS decides which cores run them.
//  * GPU : plain Metal kernels (float4, grid-stride) on zero-copy buffers; time from GPU timestamps.
//  * ANE : there is no way to touch memory from the ANE directly. We report *end-to-end effective*
//          rates through Core ML (includes Core ML's copies and the small amount of matmul compute) for
//          a read-heavy op (matrix x vector) and a write-heavy op (outer product). Treat as a lower
//          bound on what the ANE path can move, not as the ANE's raw bandwidth.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <arm_neon.h>
#include <pthread.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include "mx/mx.h"

using Clock = std::chrono::steady_clock;
static volatile float g_sink;

// ---------------------------------------------------------------- CPU kernels
static float cpu_read(const float* p, size_t n) {
  float32x4_t a[8] = {};
  for (size_t i = 0; i + 32 <= n; i += 32)
    for (int k = 0; k < 8; ++k) a[k] = vaddq_f32(a[k], vld1q_f32(p + i + 4 * k));
  float32x4_t s = a[0];
  for (int k = 1; k < 8; ++k) s = vaddq_f32(s, a[k]);
  return vaddvq_f32(s);
}
static void cpu_write(float* p, size_t n) {
  const float32x4_t v = vdupq_n_f32(1.f);
  for (size_t i = 0; i + 32 <= n; i += 32)
    for (int k = 0; k < 8; ++k) vst1q_f32(p + i + 4 * k, v);
}
static void cpu_copy(float* d, const float* s, size_t n) { std::memcpy(d, s, n * sizeof(float)); }

enum Op { READ, WRITE, COPY };

// Run `passes` passes of `op` over [src,dst) split across `nt` threads; returns seconds.
static double cpu_run(Op op, const float* src, float* dst, size_t n, int nt, int passes) {
  std::atomic<int> ready{0}, go{0};
  std::vector<std::thread> th;
  const size_t chunk = (n / nt) / 32 * 32;
  for (int t = 0; t < nt; ++t)
    th.emplace_back([&, t] {
      pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
      const size_t off = t * chunk, len = (t == nt - 1) ? n - off : chunk;
      ready++;
      while (!go.load(std::memory_order_acquire)) {}
      float acc = 0;
      for (int p = 0; p < passes; ++p) {
        if (op == READ) acc += cpu_read(src + off, len);
        else if (op == WRITE) cpu_write(dst + off, len);
        else cpu_copy(dst + off, src + off, len);
      }
      g_sink = acc;
    });
  while (ready.load() < nt) {}
  auto t0 = Clock::now();
  go.store(1, std::memory_order_release);
  for (auto& t : th) t.join();
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

static int passes_for(size_t bytes, double target_bytes = 4e9) { return (int)std::max(1.0, target_bytes / bytes); }

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

static double cpu_gbps(Op op, const float* src, float* dst, size_t n, int nt) {
  const size_t bytes = n * sizeof(float) * (op == COPY ? 2 : 1);
  const int passes = passes_for(n * sizeof(float));
  cpu_run(op, src, dst, n, nt, 1);  // warmup / page-in
  std::vector<double> t;
  for (int i = 0; i < 5; ++i) t.push_back(cpu_run(op, src, dst, n, nt, passes) / passes);
  return bytes / median(t) / 1e9;
}

// ---------------------------------------------------------------- GPU kernels
static const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void rd(device const float4* s [[buffer(0)]], device float* out [[buffer(1)]], constant uint& n [[buffer(2)]],
               uint g [[thread_position_in_grid]], uint T [[threads_per_grid]]) {
  float4 a = 0; uint i = g;
  for (; i + 3 * T < n; i += 4 * T) a += s[i] + s[i + T] + s[i + 2 * T] + s[i + 3 * T];
  for (; i < n; i += T) a += s[i];
  out[g] = a.x + a.y + a.z + a.w;
}
kernel void wr(device float4* d [[buffer(0)]], constant uint& n [[buffer(2)]],
               uint g [[thread_position_in_grid]], uint T [[threads_per_grid]]) {
  uint i = g;
  for (; i + 3 * T < n; i += 4 * T) { d[i] = 1; d[i + T] = 1; d[i + 2 * T] = 1; d[i + 3 * T] = 1; }
  for (; i < n; i += T) d[i] = 1;
}
kernel void cp(device const float4* s [[buffer(0)]], device float4* d [[buffer(1)]], constant uint& n [[buffer(2)]],
               uint g [[thread_position_in_grid]], uint T [[threads_per_grid]]) {
  uint i = g;
  for (; i + 3 * T < n; i += 4 * T) { float4 a = s[i], b = s[i + T], c = s[i + 2 * T], e = s[i + 3 * T];
                                       d[i] = a; d[i + T] = b; d[i + 2 * T] = c; d[i + 3 * T] = e; }
  for (; i < n; i += T) d[i] = s[i];
}
)MSL";

struct Gpu {
  id<MTLDevice> dev; id<MTLCommandQueue> q; id<MTLComputePipelineState> rd, wr, cp; id<MTLBuffer> out;
  bool ok = false;
};

static Gpu gpu_init() {
  Gpu g;
  g.dev = MTLCreateSystemDefaultDevice();
  if (!g.dev) return g;
  g.q = [g.dev newCommandQueue];
  NSError* err = nil;
  id<MTLLibrary> lib = [g.dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:nil error:&err];
  if (!lib) { fprintf(stderr, "gpu compile: %s\n", err.localizedDescription.UTF8String); return g; }
  auto mk = [&](NSString* n) { return [g.dev newComputePipelineStateWithFunction:[lib newFunctionWithName:n] error:&err]; };
  g.rd = mk(@"rd"); g.wr = mk(@"wr"); g.cp = mk(@"cp");
  g.out = [g.dev newBufferWithLength:(1u << 20) * 4 options:MTLResourceStorageModeShared];
  g.ok = g.rd && g.wr && g.cp;
  return g;
}

// GB/s for one op; src/dst are zero-copy wrappers of page-aligned memory.
static double gpu_gbps(Gpu& g, Op op, id<MTLBuffer> src, id<MTLBuffer> dst, size_t bytes) {
  const uint32_t nvec = (uint32_t)(bytes / 16);
  const uint32_t threads = 1u << 20;
  const int passes = passes_for(bytes, 4e9);
  auto once = [&]() {
    id<MTLCommandBuffer> cb = [g.q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    for (int p = 0; p < passes; ++p) {
      id<MTLComputePipelineState> ps = op == READ ? g.rd : op == WRITE ? g.wr : g.cp;
      [e setComputePipelineState:ps];
      [e setBuffer:(op == WRITE ? dst : src) offset:0 atIndex:0];
      [e setBuffer:(op == COPY ? dst : g.out) offset:0 atIndex:1];
      [e setBytes:&nvec length:4 atIndex:2];
      [e dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    }
    [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    return (cb.GPUEndTime - cb.GPUStartTime) / passes;
  };
  once();
  std::vector<double> t;
  for (int i = 0; i < 5; ++i) t.push_back(once());
  return bytes * (op == COPY ? 2 : 1) / median(t) / 1e9;
}

// ---------------------------------------------------------------- ANE (via Core ML, end to end)
static void ane_probe(const char* label, size_t M, size_t K, size_t N, double moved_bytes) {
  const size_t hs = 2;
  _Float16* A = (_Float16*)mx::alloc_unified(M * K * hs);
  _Float16* B = (_Float16*)mx::alloc_unified(K * N * hs);
  _Float16* C = (_Float16*)mx::alloc_unified(M * N * hs);
  for (size_t i = 0; i < M * K; ++i) A[i] = _Float16(0.001f);
  for (size_t i = 0; i < K * N; ++i) B[i] = _Float16(0.001f);
  mx::MatrixView a{A, M, K, K, mx::DType::F16}, b{B, K, N, N, mx::DType::F16}, c{C, M, N, N, mx::DType::F16};
  for (auto id : {mx::BackendId::ANE, mx::BackendId::GPU}) {
    mx::GemmOptions o; o.backend = id;
    if (mx::gemm(c, a, b, o) != 0) { printf("  %-34s %-4s unsupported\n", label, mx::backend_name(id)); continue; }
    std::vector<double> t;
    for (int i = 0; i < 7; ++i) {
      auto t0 = Clock::now();
      mx::gemm(c, a, b, o);
      t.push_back(std::chrono::duration<double>(Clock::now() - t0).count());
    }
    double s = median(t), gf = 2.0 * M * K * N / s / 1e9;
    printf("  %-34s %-4s %8.2f ms  %7.1f GB/s effective  (%6.0f GFLOP/s)\n", label, mx::backend_name(id), s * 1e3,
           moved_bytes / s / 1e9, gf);
  }
  mx::free_unified(A); mx::free_unified(B); mx::free_unified(C);
}

int main() {
  size_t max_mb = 1024;
  if (const char* e = getenv("MX_BW_MAX_MB")) max_mb = strtoull(e, nullptr, 10);
  int super = 6, total = (int)std::thread::hardware_concurrency();
  { int v = 0; size_t sz = sizeof v; if (sysctlbyname("hw.perflevel0.logicalcpu", &v, &sz, nullptr, 0) == 0 && v > 0) super = v; }

  @autoreleasepool {
    Gpu g = gpu_init();
    printf("Memory bandwidth, GB/s (copy counts read+write). CPU threads: 1 / %d (Super) / %d (all)\n\n", super, total);
    printf("%8s | %-23s | %-23s | %-23s | %s\n", "set(MB)", "READ  cpu1  cpuS  cpuAll", "WRITE cpu1  cpuS  cpuAll",
           "COPY  cpu1  cpuS  cpuAll", "GPU read / write / copy");
    for (size_t mb = 1; mb <= max_mb; mb *= 4) {
      const size_t bytes = mb << 20, n = bytes / sizeof(float);
      float* src = (float*)mx::alloc_unified(bytes);
      float* dst = (float*)mx::alloc_unified(bytes);
      for (size_t i = 0; i < n; ++i) { src[i] = 1.f; dst[i] = 0.f; }
      double r[3], w[3], c[3]; const int nts[3] = {1, super, total};
      for (int k = 0; k < 3; ++k) {
        r[k] = cpu_gbps(READ, src, dst, n, nts[k]);
        w[k] = cpu_gbps(WRITE, src, dst, n, nts[k]);
        c[k] = cpu_gbps(COPY, src, dst, n, nts[k]);
      }
      double gr = 0, gw = 0, gc = 0;
      if (g.ok) {
        id<MTLBuffer> bs = [g.dev newBufferWithBytesNoCopy:src length:bytes options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> bd = [g.dev newBufferWithBytesNoCopy:dst length:bytes options:MTLResourceStorageModeShared deallocator:nil];
        gr = gpu_gbps(g, READ, bs, bd, bytes); gw = gpu_gbps(g, WRITE, bs, bd, bytes); gc = gpu_gbps(g, COPY, bs, bd, bytes);
      }
      printf("%8zu | %12.0f %5.0f %5.0f | %12.0f %5.0f %5.0f | %12.0f %5.0f %5.0f | %6.0f / %5.0f / %5.0f\n", mb, r[0], r[1], r[2],
             w[0], w[1], w[2], c[0], c[1], c[2], gr, gw, gc);
      mx::free_unified(src); mx::free_unified(dst);
    }

    printf("\nANE vs GPU through the library (FP16, end to end incl. Core ML / dispatch):\n");
    printf(" 'effective GB/s' = bytes of A+B+C moved / wall time. Compute is included, so this is a lower bound.\n");
    ane_probe("matrix x vector 8192x8192 . 8192x1", 8192, 8192, 1, 8192.0 * 8192 * 2 + 8192 * 2 + 8192 * 2);
    ane_probe("matrix x matrix 8192x8192 . 8192x32", 8192, 8192, 32, 8192.0 * 8192 * 2 + 8192 * 32 * 2 + 8192 * 32 * 2);
    ane_probe("outer product 8192x1 . 1x8192 (write)", 8192, 1, 8192, 8192.0 * 2 + 8192 * 2 + 8192.0 * 8192 * 2);
  }
  return 0;
}
