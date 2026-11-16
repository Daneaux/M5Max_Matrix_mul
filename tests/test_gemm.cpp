// Checks every backend that claims support against the reference.
#include <array>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <random>
#include <vector>
#include "mx/mx.h"

static double rel_err(const float* a, const float* b, size_t n) {
  double num = 0, den = 0;
  for (size_t i = 0; i < n; ++i) { double d = a[i] - b[i]; num += d * d; den += double(b[i]) * b[i]; }
  return std::sqrt(num / (den > 0 ? den : 1));
}

int main() {
  int failures = 0;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1, 1);
  const size_t shapes[][3] = {{1, 1, 1}, {17, 33, 9}, {64, 64, 64}, {130, 70, 250}};
  for (auto& s : shapes) {
    size_t M = s[0], N = s[1], K = s[2];
    std::vector<float> A(M * K), B(K * N), Cref(M * N), C(M * N);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);
    mx::MatrixView a{A.data(), M, K, K}, b{B.data(), K, N, N}, cr{Cref.data(), M, N, N}, c{C.data(), M, N, N};
    mx::GemmOptions ro; ro.backend = mx::BackendId::Ref;
    if (mx::gemm(cr, a, b, ro) != 0) { std::printf("ref failed\n"); return 1; }
    for (auto id : {mx::BackendId::Auto, mx::BackendId::SME, mx::BackendId::GPU, mx::BackendId::ANE}) {
      mx::GemmOptions o; o.backend = id;
      int rc = mx::gemm(c, a, b, o);
      if (rc == 3) continue;  // engine unavailable / unsupported: skip
      double e = rc == 0 ? rel_err(C.data(), Cref.data(), M * N) : 1.0;
      bool ok = rc == 0 && e < 1e-4;
      std::printf("%-4s %zux%zux%zu rc=%d err=%.2e %s\n", mx::backend_name(id), M, N, K, rc, e, ok ? "ok" : "FAIL");
      failures += !ok;
    }
  }
  // alpha/beta, strided (ld > cols) and larger shapes, SME vs ref.
  for (auto& s : std::vector<std::array<size_t, 3>>{{257, 255, 513}, {70, 600, 1100}, {33, 31, 65}, {96, 96, 96}}) {
    size_t M = s[0], N = s[1], K = s[2], lda = K + 3, ldb = N + 5, ldc = N + 7;
    std::vector<float> A(M * lda), B(K * ldb), C0(M * ldc), C(M * ldc), Cref;
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);
    for (auto& x : C0) x = dist(rng);
    C = Cref = C0;
    mx::MatrixView a{A.data(), M, K, lda}, b{B.data(), K, N, ldb}, cr{Cref.data(), M, N, ldc}, c{C.data(), M, N, ldc};
    mx::GemmOptions o; o.alpha = 1.5f; o.beta = -0.5f;
    o.backend = mx::BackendId::Ref; mx::gemm(cr, a, b, o);
    o.backend = mx::BackendId::SME;
    int rc = mx::gemm(c, a, b, o);
    if (rc == 3) continue;
    double e = rc == 0 ? rel_err(C.data(), Cref.data(), C.size()) : 1.0;  // incl. padding: must be untouched
    bool ok = rc == 0 && e < 1e-4;
    std::printf("sme  %zux%zux%zu alpha/beta/ld rc=%d err=%.2e %s\n", M, N, K, rc, e, ok ? "ok" : "FAIL");
    failures += !ok;
  }
  std::printf("SME available: %d  GPU: %d  ANE: %d\n", mx::backend_available(mx::BackendId::SME),
              mx::backend_available(mx::BackendId::GPU), mx::backend_available(mx::BackendId::ANE));
  return failures ? 1 : 0;
}
