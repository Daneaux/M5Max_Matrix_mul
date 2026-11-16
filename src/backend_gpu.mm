// GPU backend: Metal 4 tensor ops (MetalPerformancePrimitives matmul2d), FP32 and FP16.
//
// Design notes
//  * Shaders are compiled at runtime (newLibraryWithSource:, language 4.0), so no offline
//    Metal toolchain is needed. Pipelines are created lazily and cached per dtype.
//  * Operands are bound as plain buffers; the kernel builds strided `tensor_inline` views
//    over them (explicit strides, so padded leading dimensions work). Tensor extents are
//    [inner, outer] i.e. A:[K,M] B:[N,K] C:[N,M] for row-major data.
//  * Memory: allocations made with mx::alloc_unified are registered (see mx.cpp) and wrapped
//    zero-copy (newBufferWithBytesNoCopy, cached per allocation). Anything else is staged
//    through shared MTLBuffers with memcpy in/out; that cost is counted in estimate_ns.
//  * C = alpha*A*B + beta*C: the fast path (alpha=1, beta=0) writes C directly; otherwise the
//    product goes to a temporary and a small epilogue kernel combines it with C.
//  * gemm() is synchronous (commit + waitUntilCompleted). TODO: async/Future for the planner.
//  * Tuning knobs for experiments: MX_GPU_TM, MX_GPU_TN, MX_GPU_SG (tile rows/cols, simdgroups).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include "backend.h"

namespace mx {
namespace {

// Aliases: inside Backend subclasses `id` is the BackendId accessor, which shadows ObjC `id`.
using MBuf = id<MTLBuffer>;
using MDev = id<MTLDevice>;
using MQueue = id<MTLCommandQueue>;
using MPso = id<MTLComputePipelineState>;
using MCmd = id<MTLCommandBuffer>;
using MEnc = id<MTLComputeCommandEncoder>;
using MLib = id<MTLLibrary>;
using MFn = id<MTLFunction>;

const char* kShaderSource = R"MSL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

struct Dims { uint M, N, K, lda, ldb, ldc, offA, offB, offC; };
struct Epi  { uint M, N, ldc, offC; float alpha, beta; };

template <typename E>
inline void mm_impl(device E* Ap, device E* Bp, device E* Cp, constant Dims& d, uint2 tg) {
  using T2 = tensor<device E, dextents<int32_t, 2>, tensor_inline>;
  T2 A(Ap + d.offA, dextents<int32_t, 2>(d.K, d.M), array<int32_t, 2>{1, int32_t(d.lda)});
  T2 B(Bp + d.offB, dextents<int32_t, 2>(d.N, d.K), array<int32_t, 2>{1, int32_t(d.ldb)});
  T2 C(Cp + d.offC, dextents<int32_t, 2>(d.N, d.M), array<int32_t, 2>{1, int32_t(d.ldc)});
  constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent));
  matmul2d<desc, execution_simdgroups<SG>> op;
  auto mA = A.slice(0, tg.y * TM);
  auto mB = B.slice(tg.x * TN, 0);
  auto mC = C.slice(tg.x * TN, tg.y * TM);
  op.run(mA, mB, mC);
}

kernel void mm_f32(device float* A [[buffer(0)]], device float* B [[buffer(1)]], device float* C [[buffer(2)]],
                   constant Dims& d [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) { mm_impl<float>(A, B, C, d, tg); }
kernel void mm_f16(device half* A [[buffer(0)]], device half* B [[buffer(1)]], device half* C [[buffer(2)]],
                   constant Dims& d [[buffer(3)]], uint2 tg [[threadgroup_position_in_grid]]) { mm_impl<half>(A, B, C, d, tg); }

template <typename E>
inline void epi_impl(device E* T, device E* C, constant Epi& e, uint2 gid) {
  if (gid.x >= e.N || gid.y >= e.M) return;
  float r = e.alpha * float(T[gid.y * e.N + gid.x]);
  device E* c = C + e.offC + gid.y * e.ldc + gid.x;
  if (e.beta != 0.0f) r += e.beta * float(*c);
  *c = E(r);
}
kernel void epi_f32(device float* T [[buffer(0)]], device float* C [[buffer(1)]], constant Epi& e [[buffer(2)]],
                    uint2 gid [[thread_position_in_grid]]) { epi_impl<float>(T, C, e, gid); }
kernel void epi_f16(device half* T [[buffer(0)]], device half* C [[buffer(1)]], constant Epi& e [[buffer(2)]],
                    uint2 gid [[thread_position_in_grid]]) { epi_impl<half>(T, C, e, gid); }
)MSL";

struct DimsC { uint32_t M, N, K, lda, ldb, ldc, offA, offB, offC; };
struct EpiC { uint32_t M, N, ldc, offC; float alpha, beta; };

size_t env_size(const char* name, size_t def) {
  const char* e = std::getenv(name);
  return e ? static_cast<size_t>(std::strtoull(e, nullptr, 10)) : def;
}
size_t elem_size(DType t) { return t == DType::F16 ? 2 : 4; }
size_t span_bytes(const MatrixView& v) { return ((v.rows - 1) * v.ld + v.cols) * elem_size(v.dtype); }

class GpuBackend final : public Backend {
 public:
  BackendId id() const override { return BackendId::GPU; }
  bool available() const override { return const_cast<GpuBackend*>(this)->init(); }

  bool supports(const GemmDesc& d) const override {
    const DType t = d.C.dtype;
    return (t == DType::F32 || t == DType::F16) && d.A.dtype == t && d.B.dtype == t && d.m() > 0 &&
           d.n() > 0 && d.k() > 0 && d.A.cols == d.B.rows && d.C.rows == d.A.rows && d.C.cols == d.B.cols &&
           d.A.ld >= d.A.cols && d.B.ld >= d.B.cols && d.C.ld >= d.C.cols && d.m() < (1u << 28) &&
           d.n() < (1u << 28) && d.k() < (1u << 28) && span_bytes(d.A) < (1ull << 33) &&
           span_bytes(d.B) < (1ull << 33) && span_bytes(d.C) < (1ull << 33);
  }

  double estimate_ns(const GemmDesc& d) const override {
    // Sync launch + wait is ~100-200 us; tensor-op throughput is far above SME for large sizes.
    double ns = 150e3 + d.flops() / (d.C.dtype == DType::F16 ? 15000.0 : 8000.0);
    for (const MatrixView* v : {&d.A, &d.B, &d.C}) {
      const void* b; size_t sz;
      if (!unified_find(v->data, span_bytes(*v), &b, &sz)) ns += span_bytes(*v) / 50.0;  // staged copy
    }
    return ns;
  }

  int gemm(const GemmDesc& d) override {
    std::lock_guard<std::mutex> lock(mu_);
    if (!init() || !supports(d)) return 1;
    @autoreleasepool {
      const bool f16 = d.C.dtype == DType::F16;
      const bool fast = d.alpha == 1.f && d.beta == 0.f;
      const size_t es = elem_size(d.C.dtype);

      Bound a = bind(d.A, 0, true), b = bind(d.B, 1, true);
      Bound c = bind(d.C, 3, d.beta != 0.f);
      MBuf tmp = nil;
      if (!fast) tmp = staging(2, d.m() * d.n() * es);

      DimsC dims{(uint32_t)d.m(), (uint32_t)d.n(), (uint32_t)d.k(), (uint32_t)d.A.ld, (uint32_t)d.B.ld,
                 fast ? (uint32_t)d.C.ld : (uint32_t)d.n(), a.off, b.off, fast ? c.off : 0};

      MCmd cb = [queue_ commandBuffer];
      MEnc e = [cb computeCommandEncoder];
      MPso mm = f16 ? mm_f16_ : mm_f32_;
      [e setComputePipelineState:mm];
      [e setBuffer:a.buf offset:0 atIndex:0];
      [e setBuffer:b.buf offset:0 atIndex:1];
      [e setBuffer:(fast ? c.buf : tmp) offset:0 atIndex:2];
      [e setBytes:&dims length:sizeof dims atIndex:3];
      [e dispatchThreadgroups:MTLSizeMake((d.n() + tn_ - 1) / tn_, (d.m() + tm_ - 1) / tm_, 1)
          threadsPerThreadgroup:MTLSizeMake(mm.threadExecutionWidth * sg_, 1, 1)];
      if (!fast) {
        EpiC ep{(uint32_t)d.m(), (uint32_t)d.n(), (uint32_t)d.C.ld, c.off, d.alpha, d.beta};
        MPso epi = f16 ? epi_f16_ : epi_f32_;
        [e setComputePipelineState:epi];
        [e setBuffer:tmp offset:0 atIndex:0];
        [e setBuffer:c.buf offset:0 atIndex:1];
        [e setBytes:&ep length:sizeof ep atIndex:2];
        [e dispatchThreads:MTLSizeMake(d.n(), d.m(), 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
      }
      [e endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.error) return 4;
      copy_back(d.C, c);
    }
    return 0;
  }

  void forget(const void* base) {
    std::lock_guard<std::mutex> lock(mu_);
    wrapped_.erase(base);
  }

 private:
  struct Bound {
    MBuf buf = nil;
    uint32_t off = 0;           // element offset of the matrix start inside buf
    bool staged = false;        // true: buf is a staging copy (C needs copy-back)
  };

  bool init() {
    std::lock_guard<std::recursive_mutex> g(init_mu_);
    if (state_ != 0) return state_ > 0;
    state_ = -1;
    @autoreleasepool {
      dev_ = MTLCreateSystemDefaultDevice();
      if (!dev_) return false;
      queue_ = [dev_ newCommandQueue];
      tm_ = env_size("MX_GPU_TM", 64); tn_ = env_size("MX_GPU_TN", 64); sg_ = env_size("MX_GPU_SG", 4);
      MTLCompileOptions* o = [MTLCompileOptions new];
      o.languageVersion = MTLLanguageVersion4_0;
      o.preprocessorMacros = @{@"TM": @(tm_), @"TN": @(tn_), @"SG": @(sg_)};
      NSError* err = nil;
      MLib lib = [dev_ newLibraryWithSource:[NSString stringWithUTF8String:kShaderSource] options:o error:&err];
      if (!lib) { last_error_ = err.localizedDescription.UTF8String; return false; }
      auto pso = [&](NSString* n) -> MPso {
        MFn f = [lib newFunctionWithName:n];
        return f ? [dev_ newComputePipelineStateWithFunction:f error:&err] : nil;
      };
      mm_f32_ = pso(@"mm_f32"); mm_f16_ = pso(@"mm_f16"); epi_f32_ = pso(@"epi_f32"); epi_f16_ = pso(@"epi_f16");
      if (!mm_f32_ || !mm_f16_ || !epi_f32_ || !epi_f16_) return false;
    }
    state_ = 1;
    return true;
  }

  MBuf staging(int slot, size_t bytes) {
    if (!stage_[slot] || stage_[slot].length < bytes)
      stage_[slot] = [dev_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    return stage_[slot];
  }

  // Zero-copy if the operand lies inside a registered unified allocation, else copy it into
  // staging slot `slot` (A=0, B=1, tmp=2, C=3). C is only copied in when its old value is needed.
  Bound bind(const MatrixView& v, int slot, bool copy_in) {
    Bound r;
    const size_t span = span_bytes(v), es = elem_size(v.dtype);
    const void* base; size_t size;
    if (unified_find(v.data, span, &base, &size)) {
      auto it = wrapped_.find(base);
      if (it == wrapped_.end()) {
        MBuf nb = [dev_ newBufferWithBytesNoCopy:const_cast<void*>(base) length:size
                                         options:MTLResourceStorageModeShared deallocator:nil];
        it = wrapped_.emplace(base, nb).first;
      }
      MBuf buf = it->second;
      const size_t byte_off = static_cast<const char*>(v.data) - static_cast<const char*>(base);
      if (buf && byte_off % es == 0) { r.buf = buf; r.off = (uint32_t)(byte_off / es); return r; }
    }
    r.staged = true;
    r.buf = staging(slot, span);
    if (copy_in) std::memcpy(r.buf.contents, v.data, span);
    return r;
  }

  void copy_back(const MatrixView& C, const Bound& c) {
    if (!c.staged) return;
    const size_t es = elem_size(C.dtype);
    auto* src = static_cast<const char*>(c.buf.contents);
    auto* dst = static_cast<char*>(C.data);
    for (size_t i = 0; i < C.rows; ++i) std::memcpy(dst + i * C.ld * es, src + i * C.ld * es, C.cols * es);
  }

  std::mutex mu_;
  std::recursive_mutex init_mu_;
  int state_ = 0;  // 0 = not initialized, 1 = ok, -1 = failed
  std::string last_error_;
  MDev dev_ = nil;
  MQueue queue_ = nil;
  MPso mm_f32_ = nil, mm_f16_ = nil, epi_f32_ = nil, epi_f16_ = nil;
  MBuf stage_[4] = {nil, nil, nil, nil};
  std::map<const void*, MBuf> wrapped_;
  size_t tm_ = 64, tn_ = 32, sg_ = 4;
};

GpuBackend& gpu() { static GpuBackend b; return b; }
}  // namespace

Backend* make_gpu_backend() { return &gpu(); }
void gpu_forget(const void* base) { gpu().forget(base); }
}  // namespace mx
