// CPU SME/SME2 backend: FP32 GEMM, row-major.
//
// Hardware model (M5 Max: SVL = 512 bit): a ZA tile for f32 is 16x16; there are four
// (ZA0..ZA3). `svmopa` does tile += outer_product(a[16], b[16]). We use a 2x2 grid of
// tiles, i.e. a 32x32 block of C accumulated over K:
//     ZA0 = C[ 0:16, 0:16]   ZA1 = C[ 0:16,16:32]
//     ZA2 = C[16:32, 0:16]   ZA3 = C[16:32,16:32]
// Per k: load a[0:16], a[16:32] (column k of A panel) and b[0:16], b[16:32] (row k of B
// panel), then 4 MOPAs.
//
// Packing (done in normal mode, zero-padded so the kernel never needs edge cases):
//   Ap : [k][32]       column k of a 32-row slab of A
//   Bp : [jb][k][32]   row k of each 32-col slab of B
// Edges are handled only when storing ZA back to C (masked rows/cols).
//
// Cache blocking (BLIS-style): K outermost; per K block pack all of A once, then for each
// NC-column block pack B once (KC*NC*4 B = 8 MB; the Super-core L2 is 16 MB) and let every 32-row slab of A
// reuse it. Without this each
// slab re-streams all of B from memory and large sizes stall at ~50 GB/s. C accumulates
// across K blocks via ZA preloaded from C.
// Known limitations (TODO): no transposes, F32 only. Threads: MX_SME_THREADS (default: number of Super cores); the SME unit
// is shared per cluster so more threads is not automatically faster.
#include <arm_sme.h>
#include <dispatch/dispatch.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <vector>
#include "backend.h"

namespace mx {
namespace {

constexpr size_t kBlk = 32;   // C block edge = 2 tiles of 16 (SVL=512, f32)
#ifndef MX_KC
#define MX_KC 2048
#endif
#ifndef MX_NC
#define MX_NC 1024
#endif
constexpr size_t kKC = MX_KC;   // K block depth (A panel = KC*128 B)
constexpr size_t kNC = MX_NC;   // N block width (B block = KC*NC*4 B)

bool cpu_has_sme() {
  int v = 0; size_t n = sizeof v;
  return sysctlbyname("hw.optional.arm.FEAT_SME", &v, &n, nullptr, 0) == 0 && v == 1;
}

// Streaming vector length in 32-bit lanes; the kernel is written for 16.
size_t svl_words() {
  int v = 0; size_t n = sizeof v;
  if (sysctlbyname("hw.optional.arm.sme_max_svl_b", &v, &n, nullptr, 0) != 0) return 0;
  return static_cast<size_t>(v) / 4;
}

// Direct ZA <-> memory transfers for one 16x16 tile (ZA tile T at block offset r0,c0).
// Row r of the tile is horizontal slice r. Predicate masks edge columns (masked lanes read as
// zero and are not written), and rows beyond mrows are skipped. alpha is folded into packed A,
// and beta handled by pre-loading C into ZA, so there is no vector round trip (that path
// cost ~2 us per 32x32 block, more than 1000 MOPAs).
template <int T>
inline void load_tile(const float* C, size_t ldc, size_t r0, size_t c0, size_t mrows, size_t ncols)
    __arm_streaming __arm_inout("za") {
  if (mrows <= r0 || ncols <= c0) return;
  const uint32_t rv = static_cast<uint32_t>(std::min<size_t>(16, mrows - r0));
  const svbool_t pc = svwhilelt_b32(uint32_t(0), static_cast<uint32_t>(std::min<size_t>(16, ncols - c0)));
  for (uint32_t r = 0; r < rv; ++r) svld1_hor_za32(T, r, pc, C + (r0 + r) * ldc + c0);
}

template <int T>
inline void store_tile(float* C, size_t ldc, size_t r0, size_t c0, size_t mrows, size_t ncols)
    __arm_streaming __arm_inout("za") {
  if (mrows <= r0 || ncols <= c0) return;
  const uint32_t rv = static_cast<uint32_t>(std::min<size_t>(16, mrows - r0));
  const svbool_t pc = svwhilelt_b32(uint32_t(0), static_cast<uint32_t>(std::min<size_t>(16, ncols - c0)));
  for (uint32_t r = 0; r < rv; ++r) svst1_hor_za32(T, r, pc, C + (r0 + r) * ldc + c0);
}

// One 32-row slab of C (rows already packed into Ap, depth kc) against nc columns of packed B.
__arm_locally_streaming __arm_new("za")
void sme_slab(float* C, size_t ldc, const float* Ap, const float* Bp, size_t kc, size_t nc,
              size_t mrows, bool load_c) {
  const svbool_t pg = svptrue_b32();
  const svcount_t pc2 = svptrue_c32();
  for (size_t j0 = 0; j0 < nc; j0 += kBlk) {
    const float* b = Bp + (j0 / kBlk) * kc * kBlk;
    const float* a = Ap;
    svzero_za();
    if (load_c) {  // accumulate onto existing C (beta==1 semantics)
      const size_t nv = std::min(kBlk, nc - j0);
      load_tile<0>(C + j0, ldc, 0, 0, mrows, nv);
      load_tile<1>(C + j0, ldc, 0, 16, mrows, nv);
      load_tile<2>(C + j0, ldc, 16, 0, mrows, nv);
      load_tile<3>(C + j0, ldc, 16, 16, mrows, nv);
    }
    // 2-vector (SME2) loads: a[0:32] and b[0:32] are contiguous in the packed panels.
#define MX_STEP(o) { \
      svfloat32x2_t av = svld1_f32_x2(pc2, a + (o) * kBlk); \
      svfloat32x2_t bv = svld1_f32_x2(pc2, b + (o) * kBlk); \
      svfloat32_t a0 = svget2_f32(av, 0), a1 = svget2_f32(av, 1); \
      svfloat32_t b0 = svget2_f32(bv, 0), b1 = svget2_f32(bv, 1); \
      svmopa_za32_f32_m(0, pg, pg, a0, b0); svmopa_za32_f32_m(1, pg, pg, a0, b1); \
      svmopa_za32_f32_m(2, pg, pg, a1, b0); svmopa_za32_f32_m(3, pg, pg, a1, b1); }
    size_t k = 0;
    for (; k + 4 <= kc; k += 4, a += 4 * kBlk, b += 4 * kBlk) { MX_STEP(0) MX_STEP(1) MX_STEP(2) MX_STEP(3) }
    for (; k < kc; ++k, a += kBlk, b += kBlk) { MX_STEP(0) }
#undef MX_STEP
    const size_t ncols = std::min(kBlk, nc - j0);
    float* c = C + j0;
    store_tile<0>(c, ldc, 0, 0, mrows, ncols);
    store_tile<1>(c, ldc, 0, 16, mrows, ncols);
    store_tile<2>(c, ldc, 16, 0, mrows, ncols);
    store_tile<3>(c, ldc, 16, 16, mrows, ncols);
  }
}

size_t env_threads() {
  static const size_t t = [] {
    // Default: one thread per Super core (the cluster Accelerate-class SME throughput comes
    // from); measured best on M5 Max. More threads than that hurts (shared SME units).
    int sup = 0; size_t sz = sizeof sup;
    if (sysctlbyname("hw.perflevel0.logicalcpu", &sup, &sz, nullptr, 0) != 0 || sup < 1) sup = 1;
    const char* e = std::getenv("MX_SME_THREADS");
    long v = e ? std::strtol(e, nullptr, 10) : sup;
    return static_cast<size_t>(v < 1 ? 1 : v > 64 ? 64 : v);
  }();
  return t;
}

struct Job {
  const GemmDesc* d;
  const float *A, *B;
  float* C;
  std::vector<float> Bp;  // packed B block [jb][k][32], shared read-only by workers
  std::vector<float> Apk; // packed A for the current K block: [slab][k][32], packed once per K block
  size_t jc = 0, nc = 0, pc = 0, kc = 0;  // current N block / K block
  bool load_c = false;   // current K block accumulates onto C (else starts from zero)
  size_t nslabs_m = 0, nslabs_n = 0;      // M slabs; N slabs in current block
  std::atomic<size_t> next{0};
};

void pack_b_slab(Job& j, size_t jb) {
  const GemmDesc& d = *j.d;
  const size_t j0 = j.jc + jb * kBlk, w = std::min(kBlk, d.n() - j0);
  float* dst = j.Bp.data() + jb * j.kc * kBlk;
  for (size_t k = 0; k < j.kc; ++k) {
    const float* src = j.B + (j.pc + k) * d.B.ld + j0;
    float* o = dst + k * kBlk;
    size_t c = 0;
    for (; c < w; ++c) o[c] = src[c];
    for (; c < kBlk; ++c) o[c] = 0.f;
  }
}

void worker_pack_b(void* ctx, size_t) {
  auto& j = *static_cast<Job*>(ctx);
  for (size_t jb; (jb = j.next.fetch_add(1)) < j.nslabs_n;) pack_b_slab(j, jb);
}

void worker_pack_a(void* ctx, size_t) {
  auto& j = *static_cast<Job*>(ctx);
  const GemmDesc& d = *j.d;
  const size_t M = d.m();
  for (size_t ib; (ib = j.next.fetch_add(1)) < j.nslabs_m;) {
    const size_t i0 = ib * kBlk, h = std::min(kBlk, M - i0);
    // transpose-pack: Ap[k][r] = A[i0+r][pc+k]; k outer so Ap is written sequentially
    const float* rows[kBlk];
    for (size_t r = 0; r < kBlk; ++r) rows[r] = r < h ? j.A + (i0 + r) * d.A.ld + j.pc : nullptr;
    const float alpha = d.alpha;
    float* o = j.Apk.data() + ib * j.kc * kBlk;
    for (size_t k = 0; k < j.kc; ++k, o += kBlk)
      for (size_t r = 0; r < kBlk; ++r) o[r] = rows[r] ? alpha * rows[r][k] : 0.f;
  }
}

void worker_compute(void* ctx, size_t) {
  auto& j = *static_cast<Job*>(ctx);
  const GemmDesc& d = *j.d;
  const size_t M = d.m();
  for (size_t ib; (ib = j.next.fetch_add(1)) < j.nslabs_m;) {
    const size_t i0 = ib * kBlk, h = std::min(kBlk, M - i0);
    sme_slab(j.C + i0 * d.C.ld + j.jc, d.C.ld, j.Apk.data() + ib * j.kc * kBlk, j.Bp.data(), j.kc, j.nc, h,
             j.load_c);
  }
}

class SmeBackend final : public Backend {
 public:
  BackendId id() const override { return BackendId::SME; }
  bool available() const override {
    static const bool a = cpu_has_sme() && svl_words() == 16;  // kernel assumes SVL=512
    return a;
  }
  bool supports(const GemmDesc& d) const override {
    return d.A.dtype == DType::F32 && d.B.dtype == DType::F32 && d.C.dtype == DType::F32 &&
           d.m() > 0 && d.n() > 0 && d.k() > 0 && d.A.cols == d.B.rows && d.C.rows == d.A.rows &&
           d.C.cols == d.B.cols && d.A.ld >= d.A.cols && d.B.ld >= d.B.cols && d.C.ld >= d.C.cols;
  }
  int gemm(const GemmDesc& d) override {
    if (!available() || !supports(d)) return 1;
    Job j;
    j.d = &d;
    j.A = static_cast<const float*>(d.A.data);
    j.B = static_cast<const float*>(d.B.data);
    j.C = static_cast<float*>(d.C.data);
    j.nslabs_m = (d.m() + kBlk - 1) / kBlk;
    const size_t kc_max = std::min(kKC, d.k()), nc_max = std::min(kNC, d.n());  // size buffers to the problem
    j.Bp.resize((nc_max + kBlk - 1) / kBlk * kc_max * kBlk);

    // ~1 thread per 40 MFLOP: dispatch + sync costs more than tiny jobs are worth.
    const size_t nt = std::min({env_threads(), j.nslabs_m, static_cast<size_t>(std::max(1.0, d.flops() / 4e7))});
    dispatch_queue_t q = dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0);
    auto run = [&](void (*fn)(void*, size_t)) {
      j.next = 0;
      if (nt == 1) fn(&j, 0); else dispatch_apply_f(nt, q, &j, fn);
    };
    if (d.beta != 0.f && d.beta != 1.f) {  // C *= beta once; the kernel then only accumulates
      for (size_t i = 0; i < d.m(); ++i) {
        float* row = j.C + i * d.C.ld;
        for (size_t c = 0; c < d.n(); ++c) row[c] *= d.beta;
      }
    }
    j.Apk.resize(j.nslabs_m * kc_max * kBlk);
    for (j.pc = 0; j.pc < d.k(); j.pc += kKC) {  // K outermost: A is packed once per K block
      j.kc = std::min(kKC, d.k() - j.pc);
      j.load_c = j.pc != 0 || d.beta != 0.f;
      run(worker_pack_a);
      for (j.jc = 0; j.jc < d.n(); j.jc += kNC) {
        j.nc = std::min(kNC, d.n() - j.jc);
        j.nslabs_n = (j.nc + kBlk - 1) / kBlk;
        run(worker_pack_b);
        run(worker_compute);
      }
    }
    return 0;
  }
  double estimate_ns(const GemmDesc& d) const override { return 5e3 + d.flops() / 1500.0; }  // ~1.5 TFLOP/s measured (mid sizes)
};
}  // namespace

Backend* make_sme_backend() { static SmeBackend b; return &b; }
}  // namespace mx
