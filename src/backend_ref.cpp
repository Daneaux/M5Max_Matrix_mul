// Reference backend: simple, correct, slow. The oracle for all other backends.
#include "backend.h"

namespace mx {
namespace {
class RefBackend final : public Backend {
 public:
  BackendId id() const override { return BackendId::Ref; }
  bool available() const override { return true; }
  bool supports(const GemmDesc& d) const override {
    return d.A.dtype == DType::F32 && d.B.dtype == DType::F32 && d.C.dtype == DType::F32;
  }
  int gemm(const GemmDesc& d) override {
    if (!supports(d) || d.A.cols != d.B.rows || d.C.rows != d.A.rows || d.C.cols != d.B.cols) return 1;
    auto* C = static_cast<float*>(d.C.data);
    auto* A = static_cast<const float*>(d.A.data);
    auto* B = static_cast<const float*>(d.B.data);
    for (size_t i = 0; i < d.m(); ++i)
      for (size_t j = 0; j < d.n(); ++j) {
        float acc = 0;
        for (size_t p = 0; p < d.k(); ++p) acc += A[i * d.A.ld + p] * B[p * d.B.ld + j];
        float& c = C[i * d.C.ld + j];
        c = d.alpha * acc + (d.beta == 0.f ? 0.f : d.beta * c);
      }
    return 0;
  }
  double estimate_ns(const GemmDesc& d) const override { return d.flops() / 2.0; }  // ~2 GFLOP/s
};
}  // namespace

Backend* make_ref_backend() { static RefBackend b; return &b; }
}  // namespace mx
