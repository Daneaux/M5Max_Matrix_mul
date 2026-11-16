// Profiling target for the GPU threadgroup-scaling experiment (see bench_scaling.mm).
//
//   ./build/prof_threadgroups T [seconds]            run FP16 shared-input matmul at T threadgroups, print TFLOP/s
//   MTL_CAPTURE_ENABLED=1 ./build/prof_threadgroups T 0 out.gputrace
//                                                     write a Metal GPU capture (open in Xcode: Performance /
//                                                     Counters, Occupancy, shader profiler). Uses K=2048 so the
//                                                     capture stays small; 20 dispatches in one command buffer.
// Also usable with: xcrun xctrace record --template 'Metal System Trace' --launch -- ./build/prof_threadgroups 40 3
// (note: xctrace's default Metal GPU Counters selection only records "RT Unit Active", so occupancy needs the
//  Xcode GPU capture route).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
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
int main(int argc, char** argv) {
  const int Tn = argc > 1 ? atoi(argv[1]) : 40;
  const double secs = argc > 2 ? atof(argv[2]) : 3.0;
  const char* capture_path = argc > 3 ? argv[3] : nullptr;
  @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q = [dev newCommandQueue];
    MTLCompileOptions* o = [MTLCompileOptions new]; o.languageVersion = MTLLanguageVersion4_0; NSError* err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:o error:&err];
    if (!lib) { fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"mm_f16"] error:&err];
    const uint32_t K = capture_path ? 2048 : 8192;
    id<MTLBuffer> A = [dev newBufferWithLength:(size_t)64*K*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> B = [dev newBufferWithLength:(size_t)K*64*Tn*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> C = [dev newBufferWithLength:(size_t)64*64*Tn*2 options:MTLResourceStorageModeShared];
    _Float16* a = (_Float16*)A.contents; _Float16* b = (_Float16*)B.contents;
    std::mt19937 rng(2); std::normal_distribution<float> nd(0, 0.01f);
    for (size_t i = 0; i < (size_t)64*K; ++i) a[i] = (_Float16)nd(rng);
    for (size_t i = 0; i < (size_t)K*64*Tn; ++i) b[i] = (_Float16)nd(rng);
    struct { uint32_t M, N, K, share; } dims{64, (uint32_t)(64*Tn), K, 1};
    auto encode = [&](int reps) {
      id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      [e setComputePipelineState:ps]; [e setBuffer:A offset:0 atIndex:0]; [e setBuffer:B offset:0 atIndex:1]; [e setBuffer:C offset:0 atIndex:2];
      [e setBytes:&dims length:sizeof dims atIndex:3];
      for (int r = 0; r < reps; ++r) [e dispatchThreadgroups:MTLSizeMake(Tn,1,1) threadsPerThreadgroup:MTLSizeMake(ps.threadExecutionWidth*4,1,1)];
      [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
    };
    if (capture_path) {
      encode(5);  // warmup outside the capture
      MTLCaptureManager* cm = [MTLCaptureManager sharedCaptureManager];
      MTLCaptureDescriptor* d = [MTLCaptureDescriptor new];
      d.captureObject = q; d.destination = MTLCaptureDestinationGPUTraceDocument;
      d.outputURL = [NSURL fileURLWithPath:[NSString stringWithUTF8String:capture_path]];
      if (![cm startCaptureWithDescriptor:d error:&err]) { fprintf(stderr, "capture failed: %s (set MTL_CAPTURE_ENABLED=1)\n", err.localizedDescription.UTF8String); return 1; }
      encode(20);
      [cm stopCapture];
      printf("T=%d  captured 20 dispatches (K=%u) to %s\n", Tn, K, capture_path);
      return 0;
    }
    auto t0 = std::chrono::steady_clock::now(); long disp = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) { encode(200); disp += 200; }
    double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("T=%d  %.1f TFLOP/s  (%ld dispatches)\n", Tn, 2.0*64*64*Tn*K*disp/el/1e12, disp);
  }
}
