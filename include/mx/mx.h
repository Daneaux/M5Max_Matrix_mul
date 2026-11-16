// mxlib public API.
#pragma once
#include <cstddef>
#include <cstdint>

namespace mx {

enum class DType { F32, F16 };
enum class BackendId { Auto, Ref, SME, GPU, ANE };

// Row-major dense matrix view. Memory is owned by the caller (ideally a mx::Buffer).
struct MatrixView {
  void*   data = nullptr;
  size_t  rows = 0, cols = 0;
  size_t  ld   = 0;          // leading dimension in elements (>= cols)
  DType   dtype = DType::F32;
};

struct GemmOptions {
  BackendId backend = BackendId::Auto;  // Auto = planner picks / splits
  float alpha = 1.f, beta = 0.f;
};

// Page-aligned unified-memory allocation, shareable zero-copy with Metal/CoreML.
void* alloc_unified(size_t bytes);
void  free_unified(void* p);

// C = alpha * A * B + beta * C.  Returns 0 on success, nonzero mx error code.
int gemm(MatrixView C, MatrixView A, MatrixView B, const GemmOptions& opt = {});

const char* backend_name(BackendId id);
bool        backend_available(BackendId id);

}  // namespace mx
