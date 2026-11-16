// Size sweep: GFLOP/s per engine vs matrix size. CSV on stdout, progress on stderr.
//   ./build/bench_gemm > results.csv
//   MX_BENCH_MAX=4096 ./build/bench_gemm      (largest square size, default 4096)
// Accelerate (cblas_sgemm) is included as the baseline to beat.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>
#include "mx/mx.h"

using Clock = std::chrono::steady_clock;

struct Shape { size_t m, n, k; };

// Median seconds per call. Untimed warmup first; then repeat batches so each
// sample lasts >= min_sample_s, and take the median of `samples` samples.
static double time_median(const std::function<void()>& fn, double min_sample_s = 0.05, int samples = 7) {
  for (int i = 0; i < 2; ++i) fn();  // warmup (shader/model compile, caches)
  size_t reps = 1;
  for (;;) {  // find reps so one sample is long enough
    auto t0 = Clock::now();
    for (size_t i = 0; i < reps; ++i) fn();
    double s = std::chrono::duration<double>(Clock::now() - t0).count();
    if (s >= min_sample_s || reps >= (1u << 20)) break;
    reps *= 2;
  }
  std::vector<double> t;
  for (int i = 0; i < samples; ++i) {
    auto t0 = Clock::now();
    for (size_t r = 0; r < reps; ++r) fn();
    t.push_back(std::chrono::duration<double>(Clock::now() - t0).count() / reps);
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

static void emit(const char* backend, const Shape& s, double sec) {
  double gf = 2.0 * s.m * s.n * s.k / sec / 1e9;
  std::printf("%s,f32,%zu,%zu,%zu,%.6f,%.2f\n", backend, s.m, s.n, s.k, sec * 1e3, gf);
  std::fflush(stdout);
  std::fprintf(stderr, "%-10s %5zux%-5zux%-5zu %10.4f ms %9.2f GFLOP/s\n", backend, s.m, s.n, s.k, sec * 1e3, gf);
}

int main() {
  size_t max_n = 4096;
  if (const char* e = std::getenv("MX_BENCH_MAX")) max_n = std::strtoull(e, nullptr, 10);

  std::vector<Shape> shapes;
  for (size_t n = 16; n <= max_n; n = n * 3 / 2 + (n * 3 / 2) % 2) shapes.push_back({n, n, n});  // ~1.5x steps
  for (size_t n : {256, 1024}) {                                                                  // non-square probes
    if (n > max_n) continue;
    shapes.push_back({n, n, 32});   // skinny K
    shapes.push_back({32, n, n});   // short M
    shapes.push_back({n, 32, n});   // narrow N
  }

  std::printf("backend,dtype,M,N,K,median_ms,gflops\n");
  const mx::BackendId engines[] = {mx::BackendId::Ref, mx::BackendId::SME, mx::BackendId::GPU, mx::BackendId::ANE};

  for (const Shape& s : shapes) {
    std::vector<float> A(s.m * s.k, 0.5f), B(s.k * s.n, 0.25f), C(s.m * s.n);
    mx::MatrixView a{A.data(), s.m, s.k, s.k}, b{B.data(), s.k, s.n, s.n}, c{C.data(), s.m, s.n, s.n};

    // Baseline: Accelerate.
    emit("accelerate", s, time_median([&] {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)s.m, (int)s.n, (int)s.k,
                  1.f, A.data(), (int)s.k, B.data(), (int)s.n, 0.f, C.data(), (int)s.n);
    }));

    for (mx::BackendId id : engines) {
      if (id == mx::BackendId::Ref && s.m * s.n * s.k > 256ull * 256 * 256) continue;  // too slow to be useful
      mx::GemmOptions o; o.backend = id;
      if (mx::gemm(c, a, b, o) != 0) continue;  // engine unavailable / unsupported for this shape
      emit(mx::backend_name(id), s, time_median([&] { mx::gemm(c, a, b, o); }));
    }
  }
}
