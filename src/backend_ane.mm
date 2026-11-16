// ANE backend (M3 milestone). Objective-C++ shim around Core ML.
//
// There is no public ANE API; the only route is Core ML with
// MLModelConfiguration.computeUnits = MLComputeUnitsCPUAndNeuralEngine.
// Plan:
//   - Build an MLProgram containing one `matmul` for fixed (M,K,N), FP16, compile it
//     (MLModel compileModelAtURL) and cache the compiled model keyed by shape.
//   - Feed A/B via MLMultiArray wrapping the unified buffers (initWithDataPointer).
//   - Verify placement with the Core ML performance report / Instruments; ops can
//     silently fall back to CPU/GPU, so the backend must check and report this.
//   - Only F16 (and INT8 later); estimate_ns must include high dispatch latency.
// Link: -framework CoreML -framework Foundation (add when implementing).
#include "backend.h"

namespace mx {
namespace {
class AneBackend final : public Backend {
 public:
  BackendId id() const override { return BackendId::ANE; }
  bool available() const override { return false; }  // TODO(M3)
  bool supports(const GemmDesc&) const override { return false; }
  int gemm(const GemmDesc&) override { return 2; }
  double estimate_ns(const GemmDesc& d) const override { return 1e6 + d.flops() / 5000.0; }
};
}  // namespace

Backend* make_ane_backend() { static AneBackend b; return &b; }
}  // namespace mx
