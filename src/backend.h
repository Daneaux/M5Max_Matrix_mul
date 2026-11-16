// Internal backend interface. Each compute engine implements this.
#pragma once
#include "mx/mx.h"

namespace mx {

struct GemmDesc {
  MatrixView C, A, B;
  float alpha, beta;
  int threads = 0;  // see GemmOptions::threads
  size_t m() const { return C.rows; }
  size_t n() const { return C.cols; }
  size_t k() const { return A.cols; }
  double flops() const { return 2.0 * m() * n() * k(); }
};

class Backend {
 public:
  virtual ~Backend() = default;
  virtual BackendId id() const = 0;
  virtual bool available() const = 0;
  virtual bool supports(const GemmDesc& d) const = 0;
  // Synchronous for now; make async (Future) once the planner co-executes engines.
  virtual int gemm(const GemmDesc& d) = 0;
  // Planner hint: rough runtime in ns, including dispatch overhead.
  virtual double estimate_ns(const GemmDesc& d) const = 0;
};

// Unified-memory registry (mx.cpp): is [p, p+bytes) inside an mx::alloc_unified allocation?
bool unified_find(const void* p, size_t bytes, const void** base, size_t* size);
void gpu_forget(const void* base);  // backend_gpu.mm: drop cached MTLBuffer for a freed allocation

Backend* get_backend(BackendId id);  // singleton per engine
// Planner: choose one engine (later: a split plan across several).
BackendId plan_single(const GemmDesc& d);

}  // namespace mx
