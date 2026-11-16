// GPU backend (M2 milestone). Objective-C++ shim around Metal.
//
// Plan:
//   - MTLDevice + one MTLCommandQueue; wrap unified allocations with
//     newBufferWithBytesNoCopy (page-aligned, from mx::alloc_unified).
//   - MSL kernels in src/shaders/*.metal, precompiled to a .metallib:
//       v1: simdgroup_matrix 8x8 tiles (F32/F16)
//       v2: Metal 4 tensor ops (MetalPerformancePrimitives matmul2d) to use the
//           M5 per-core neural accelerators.
//   - Make gemm() async: encode, commit, return a Future signalled by completion handler.
// Link: -framework Metal -framework Foundation (add when implementing).
#include "backend.h"

namespace mx {
namespace {
class GpuBackend final : public Backend {
 public:
  BackendId id() const override { return BackendId::GPU; }
  bool available() const override { return false; }  // TODO(M2): MTLCreateSystemDefaultDevice() != nil
  bool supports(const GemmDesc&) const override { return false; }
  int gemm(const GemmDesc&) override { return 2; }
  double estimate_ns(const GemmDesc& d) const override { return 200e3 + d.flops() / 10000.0; }  // launch + ~10 TFLOP/s
};
}  // namespace

Backend* make_gpu_backend() { static GpuBackend b; return &b; }
}  // namespace mx
