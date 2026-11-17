// Plain scalar C++ matrix multiply: no SIMD, no BLAS, no blocking tricks beyond loop order.
//
//   ./build/bench_scalar            (MX_SCALAR_MAX=2048 sets the largest n)
//
// Compiled with -fno-vectorize -fno-slp-vectorize so the compiler emits scalar fmadd/fmul/fadd only (NEON is
// never used for the multiplies; verify with `make scalar-asm`). Variants, all F32, C = A.B, square:
//   ijk      textbook triple loop, dot product per element (B is walked down a column: cache-hostile)
//   ikj      same arithmetic, loop order swapped so B and C are read along rows (cache-friendly, still scalar)
//   ikj x18  ikj with the rows split over 18 threads (scalar code on every core)
// This is the "how fast is a hand-written loop" floor to compare with Accelerate / SME / GPU / ANE.
#include <dispatch/dispatch.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

__attribute__((noinline)) static void mm_ijk(const float* A, const float* B, float* C, size_t n) {
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j) {
      float acc = 0;
      for (size_t k = 0; k < n; ++k) acc += A[i * n + k] * B[k * n + j];
      C[i * n + j] = acc;
    }
}
__attribute__((noinline)) static void mm_ikj_rows(const float* A, const float* B, float* C, size_t n, size_t r0, size_t r1) {
  for (size_t i = r0; i < r1; ++i) {
    float* c = C + i * n;
    for (size_t j = 0; j < n; ++j) c[j] = 0;
    for (size_t k = 0; k < n; ++k) {
      const float a = A[i * n + k];
      const float* b = B + k * n;
      for (size_t j = 0; j < n; ++j) c[j] += a * b[j];
    }
  }
}
struct Job { const float* A; const float* B; float* C; size_t n; int T; };
static void worker(void* p, size_t t) {
  auto* j = (Job*)p;
  const size_t r0 = j->n * t / j->T, r1 = j->n * (t + 1) / j->T;
  mm_ikj_rows(j->A, j->B, j->C, j->n, r0, r1);
}

static double time_it(int reps, auto&& f) {
  f();  // warmup
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
  if (const char* e = getenv("MX_SCALAR_MAX")) max_n = strtoull(e, nullptr, 10);
  printf("scalar F32 matmul (no SIMD, no BLAS), GFLOP/s, median\n");
  printf("%6s | %9s %9s %9s | %s\n", "n", "ijk", "ikj", "ikj x18", "check");
  for (size_t n : {128, 256, 512, 1024, 2048}) {
    if (n > max_n) break;
    std::vector<float> A(n * n), B(n * n), C1(n * n), C2(n * n), C3(n * n);
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> d(-1, 1);
    for (auto& x : A) x = d(rng);
    for (auto& x : B) x = d(rng);
    const double fl = 2.0 * n * n * n;
    const int reps = n >= 1024 ? 3 : 7;
    const bool do_ijk = n <= 1024;  // ijk at 2048 takes minutes
    double g1 = 0;
    if (do_ijk) g1 = fl / time_it(reps, [&] { mm_ijk(A.data(), B.data(), C1.data(), n); }) / 1e9;
    const double g2 = fl / time_it(reps, [&] { mm_ikj_rows(A.data(), B.data(), C2.data(), n, 0, n); }) / 1e9;
    Job job{A.data(), B.data(), C3.data(), n, 18};
    const double g3 = fl / time_it(reps, [&] { dispatch_apply_f(18, DISPATCH_APPLY_AUTO, &job, worker); }) / 1e9;
    // ikj vs ikj x18 must agree exactly (same per-element op order); ijk differs only by float rounding.
    double e23 = 0, e12 = 0;
    for (size_t i = 0; i < n * n; ++i) {
      e23 = std::max(e23, (double)std::fabs(C2[i] - C3[i]));
      if (do_ijk) e12 = std::max(e12, (double)std::fabs(C1[i] - C2[i]));
    }
    const bool ok = e23 == 0 && e12 < 1e-2;
    if (do_ijk) printf("%6zu | %9.2f %9.2f %9.2f | %s\n", n, g1, g2, g3, ok ? "ok" : "BAD");
    else        printf("%6zu | %9s %9.2f %9.2f | %s\n", n, "(skipped)", g2, g3, ok ? "ok" : "BAD");
    fflush(stdout);
  }
  return 0;
}
