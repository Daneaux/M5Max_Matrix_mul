// Hand-written NEON SIMD matrix multiply (128-bit vectors, 4 floats per instruction). No SME, no BLAS.
//
//   ./build/bench_neon            (MX_NEON_MAX=2048 sets the largest n)
//
// The middle rung between bench_scalar (1 multiply-add per instruction) and SME (16x16 outer product per instruction).
// F32, C = A.B, square, n a multiple of 16. Variants:
//   neon ikj    ikj loop order, c[j..j+3] += a * b[j..j+3] with vfmaq_f32 (one vector fma per load/store pair)
//   neon 4x16   register-blocked: a 4x16 tile of C lives in 16 vector registers; per k: 4 loads of B, 4 broadcasts of A,
//               16 vfmaq. K is blocked by 256 so the 256x16 panel of B stays in L1.
//   4x16 x18    the 4x16 kernel with the row blocks split over 18 threads
#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static void ref_ikj(const float* A, const float* B, float* C, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) C[i * n + j] = 0;
    for (size_t k = 0; k < n; ++k) {
      const float a = A[i * n + k];
      for (size_t j = 0; j < n; ++j) C[i * n + j] += a * B[k * n + j];
    }
  }
}

__attribute__((noinline)) static void neon_ikj(const float* A, const float* B, float* C, size_t n, size_t r0, size_t r1) {
  for (size_t i = r0; i < r1; ++i) {
    float* c = C + i * n;
    for (size_t j = 0; j < n; ++j) c[j] = 0;
    for (size_t k = 0; k < n; ++k) {
      const float32x4_t a = vdupq_n_f32(A[i * n + k]);
      const float* b = B + k * n;
      for (size_t j = 0; j < n; j += 4) vst1q_f32(c + j, vfmaq_f32(vld1q_f32(c + j), a, vld1q_f32(b + j)));
    }
  }
}

// C[i0..i0+4) x C[j0..j0+16) += A[i0.., k0..k1) . B[k0..k1, j0..j0+16)
static inline void tile4x16(const float* A, const float* B, float* C, size_t n, size_t i0, size_t j0, size_t k0, size_t k1) {
  float32x4_t c[4][4];
  for (int r = 0; r < 4; ++r)
    for (int v = 0; v < 4; ++v) c[r][v] = vld1q_f32(C + (i0 + r) * n + j0 + 4 * v);
  const float* a0 = A + i0 * n;
  for (size_t k = k0; k < k1; ++k) {
    const float* b = B + k * n + j0;
    const float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4), b2 = vld1q_f32(b + 8), b3 = vld1q_f32(b + 12);
    for (int r = 0; r < 4; ++r) {
      const float32x4_t a = vdupq_n_f32(a0[r * n + k]);
      c[r][0] = vfmaq_f32(c[r][0], a, b0);
      c[r][1] = vfmaq_f32(c[r][1], a, b1);
      c[r][2] = vfmaq_f32(c[r][2], a, b2);
      c[r][3] = vfmaq_f32(c[r][3], a, b3);
    }
  }
  for (int r = 0; r < 4; ++r)
    for (int v = 0; v < 4; ++v) vst1q_f32(C + (i0 + r) * n + j0 + 4 * v, c[r][v]);
}
// rows [r0, r1) of C, r0/r1 multiples of 4
__attribute__((noinline)) static void neon_tile(const float* A, const float* B, float* C, size_t n, size_t r0, size_t r1) {
  for (size_t i = r0; i < r1; ++i) std::fill(C + i * n, C + (i + 1) * n, 0.f);
  const size_t KC = 256;
  for (size_t k0 = 0; k0 < n; k0 += KC) {
    const size_t k1 = std::min(n, k0 + KC);
    for (size_t j0 = 0; j0 < n; j0 += 16)
      for (size_t i0 = r0; i0 < r1; i0 += 4) tile4x16(A, B, C, n, i0, j0, k0, k1);
  }
}

struct Job { const float* A; const float* B; float* C; size_t n; int T; };
static void worker(void* p, size_t t) {
  auto* j = (Job*)p;
  const size_t blocks = j->n / 4;
  neon_tile(j->A, j->B, j->C, j->n, 4 * (blocks * t / j->T), 4 * (blocks * (t + 1) / j->T));
}

static double time_it(int reps, auto&& f) {
  f();
  std::vector<double> t;
  for (int i = 0; i < reps; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

int main() {
  size_t max_n = 2048;
  if (const char* e = getenv("MX_NEON_MAX")) max_n = strtoull(e, nullptr, 10);
  printf("NEON F32 matmul (hand-written intrinsics, no SME, no BLAS), GFLOP/s, median\n");
  printf("%6s | %9s %9s %9s | %s\n", "n", "neon ikj", "neon 4x16", "4x16 x18", "check");
  for (size_t n : {128, 256, 512, 1024, 2048, 4096}) {
    if (n > max_n) break;
    std::vector<float> A(n * n), B(n * n), R(n * n), C1(n * n), C2(n * n), C3(n * n);
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> d(-1, 1);
    for (auto& x : A) x = d(rng);
    for (auto& x : B) x = d(rng);
    const double fl = 2.0 * n * n * n;
    const int reps = n >= 2048 ? 3 : 7;
    const double g1 = fl / time_it(reps, [&] { neon_ikj(A.data(), B.data(), C1.data(), n, 0, n); }) / 1e9;
    const double g2 = fl / time_it(reps, [&] { neon_tile(A.data(), B.data(), C2.data(), n, 0, n); }) / 1e9;
    Job job{A.data(), B.data(), C3.data(), n, 18};
    const double g3 = fl / time_it(reps, [&] { dispatch_apply_f(18, DISPATCH_APPLY_AUTO, &job, worker); }) / 1e9;
    bool ok = true;
    if (n <= 1024) {
      ref_ikj(A.data(), B.data(), R.data(), n);
      for (size_t i = 0; i < n * n; ++i)
        for (const float* c : {C1.data(), C2.data(), C3.data()})
          if (std::fabs(c[i] - R[i]) > 1e-3 * (std::fabs(R[i]) + 1)) ok = false;
    } else {
      for (size_t i = 0; i < n * n; ++i)  // large n: the three NEON versions must agree with each other
        if (std::fabs(C1[i] - C2[i]) > 1e-2 || std::fabs(C2[i] - C3[i]) > 1e-4) { ok = false; break; }
    }
    printf("%6zu | %9.2f %9.2f %9.2f | %s\n", n, g1, g2, g3, ok ? "ok" : "BAD");
    fflush(stdout);
  }
  return 0;
}
