// Public API + planner.
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <map>
#include <mutex>
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
    if (!b->supports(d) || !b->available()) continue;  // supports() first: it is cheap, available() may initialize
    double ns = b->estimate_ns(d);
    if (ns < best_ns) { best_ns = ns; best = id; }
  }
  return best;
}

namespace {
std::mutex g_reg_mu;
std::map<uintptr_t, size_t> g_reg;  // base -> rounded size, for allocations from alloc_unified
}  // namespace

bool unified_find(const void* p, size_t bytes, const void** base, size_t* size) {
  std::lock_guard<std::mutex> l(g_reg_mu);
  auto it = g_reg.upper_bound(reinterpret_cast<uintptr_t>(p));
  if (it == g_reg.begin()) return false;
  --it;
  const uintptr_t b = it->first, pu = reinterpret_cast<uintptr_t>(p);
  if (pu < b || pu + bytes > b + it->second) return false;
  *base = reinterpret_cast<const void*>(b);
  *size = it->second;
  return true;
}

void* alloc_unified(size_t bytes) {
  const size_t page = static_cast<size_t>(getpagesize());
  const size_t rounded = (bytes + page - 1) / page * page;
  void* p = nullptr;
  if (posix_memalign(&p, page, rounded) != 0) return nullptr;
  std::lock_guard<std::mutex> l(g_reg_mu);
  g_reg[reinterpret_cast<uintptr_t>(p)] = rounded;
  return p;
}
void free_unified(void* p) {
  if (!p) return;
  {
    std::lock_guard<std::mutex> l(g_reg_mu);
    g_reg.erase(reinterpret_cast<uintptr_t>(p));
  }
  gpu_forget(p);  // drop any MTLBuffer wrapping this memory before it is freed
  free(p);
}

int gemm(MatrixView C, MatrixView A, MatrixView B, const GemmOptions& opt) {
  GemmDesc d{C, A, B, opt.alpha, opt.beta, opt.threads};
  BackendId id = opt.backend == BackendId::Auto ? plan_single(d) : opt.backend;
  Backend* b = get_backend(id);
  if (!b || !b->supports(d) || !b->available()) return 3;  // requested engine can't do this
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
