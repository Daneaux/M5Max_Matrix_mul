// Public API + planner.
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <unistd.h>
#include "backend.h"

namespace mx {
Backend* make_ref_backend();
Backend* make_sme_backend();
Backend* make_gpu_backend();
Backend* make_ane_backend();

Backend* get_backend(BackendId id) {
  switch (id) {
    case BackendId::Ref: return make_ref_backend();
    case BackendId::SME: return make_sme_backend();
    case BackendId::GPU: return make_gpu_backend();
    case BackendId::ANE: return make_ane_backend();
    default: return nullptr;
  }
}

// Initial planner: pick the single supported engine with the lowest estimate.
// TODO(M4): split C into row-blocks across engines proportional to measured throughput.
BackendId plan_single(const GemmDesc& d) {
  BackendId best = BackendId::Ref;
  double best_ns = std::numeric_limits<double>::infinity();
  for (BackendId id : {BackendId::Ref, BackendId::SME, BackendId::GPU, BackendId::ANE}) {
    Backend* b = get_backend(id);
    if (!b->available() || !b->supports(d)) continue;
    double ns = b->estimate_ns(d);
    if (ns < best_ns) { best_ns = ns; best = id; }
  }
  return best;
}

void* alloc_unified(size_t bytes) {
  const size_t page = static_cast<size_t>(getpagesize());
  const size_t rounded = (bytes + page - 1) / page * page;
  void* p = nullptr;
  return posix_memalign(&p, page, rounded) == 0 ? p : nullptr;
}
void free_unified(void* p) { free(p); }

int gemm(MatrixView C, MatrixView A, MatrixView B, const GemmOptions& opt) {
  GemmDesc d{C, A, B, opt.alpha, opt.beta};
  BackendId id = opt.backend == BackendId::Auto ? plan_single(d) : opt.backend;
  Backend* b = get_backend(id);
  if (!b || !b->available() || !b->supports(d)) return 3;  // requested engine can't do this
  return b->gemm(d);
}

const char* backend_name(BackendId id) {
  switch (id) {
    case BackendId::Auto: return "auto"; case BackendId::Ref: return "ref";
    case BackendId::SME: return "sme";   case BackendId::GPU: return "gpu";
    case BackendId::ANE: return "ane";
  }
  return "?";
}
bool backend_available(BackendId id) { Backend* b = get_backend(id); return b && b->available(); }
}  // namespace mx
