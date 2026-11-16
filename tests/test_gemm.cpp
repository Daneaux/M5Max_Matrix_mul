// Checks every backend that claims support against the reference.
#include <algorithm>
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
      std::fill(C.begin(), C.end(), std::nanf(""));  // poison: a backend that skips writing C must fail
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
    for (auto id : {mx::BackendId::SME, mx::BackendId::GPU, mx::BackendId::ANE}) {
      o.backend = id;
      C = C0;
      int rc = mx::gemm(c, a, b, o);
      if (rc == 3) continue;
      double e = rc == 0 ? rel_err(C.data(), Cref.data(), C.size()) : 1.0;  // incl. padding: must be untouched
      bool ok = rc == 0 && e < 1e-4;
      std::printf("%-4s %zux%zux%zu alpha/beta/ld rc=%d err=%.2e %s\n", mx::backend_name(id), M, N, K, rc, e, ok ? "ok" : "FAIL");
      failures += !ok;
    }
  }
  // Zero-copy path: operands in alloc_unified memory, sub-matrix at an element offset.
  {
    size_t M = 200, N = 130, K = 90, ld = 160;
    float* base = static_cast<float*>(mx::alloc_unified(3 * 300 * ld * sizeof(float)));
    float *A = base, *B = base + 300 * ld, *C = base + 2 * 300 * ld + 5;  // odd offset on purpose
    std::vector<float> Cref(M * N);
    for (size_t i = 0; i < 3 * 300 * ld; ++i) base[i] = dist(rng);
    mx::MatrixView a{A, M, K, ld}, b{B, K, N, ld}, c{C, M, N, ld}, cr{Cref.data(), M, N, N};
    mx::GemmOptions o; o.backend = mx::BackendId::Ref; mx::gemm(cr, a, b, o);
    for (auto id : {mx::BackendId::SME, mx::BackendId::GPU}) {
      o.backend = id;
      for (size_t i = 0; i < M; ++i) for (size_t j = 0; j < N; ++j) C[i * ld + j] = std::nanf("");
      int rc = mx::gemm(c, a, b, o);
      if (rc == 3) continue;
      std::vector<float> got(M * N);
      for (size_t i = 0; i < M; ++i) for (size_t j = 0; j < N; ++j) got[i * N + j] = C[i * ld + j];
      double e = rc == 0 ? rel_err(got.data(), Cref.data(), M * N) : 1.0;
      bool ok = rc == 0 && e < 1e-4;
      std::printf("%-4s unified-memory offset/ld rc=%d err=%.2e %s\n", mx::backend_name(id), rc, e, ok ? "ok" : "FAIL");
      failures += !ok;
    }
    mx::free_unified(base);
  }
  // FP16: GPU and ANE vs an FP32 reference computed from the same half inputs.
  // Tolerance reflects FP16 output rounding (~5e-4 relative) plus accumulation differences.
  for (auto& s : std::vector<std::array<size_t, 3>>{{64, 64, 64}, {17, 33, 9}, {130, 70, 250}, {512, 512, 512}, {300, 200, 700}}) {
    for (int ab = 0; ab < 2; ++ab) {
      size_t M = s[0], N = s[1], K = s[2], lda = K + 8, ldb = N + 8, ldc = N + 8;
      std::vector<_Float16> A(M * lda), B(K * ldb), C0(M * ldc), C(M * ldc);
      for (auto& x : A) x = _Float16(dist(rng));
      for (auto& x : B) x = _Float16(dist(rng));
      for (auto& x : C0) x = _Float16(dist(rng));
      std::vector<float> Af(M * K), Bf(K * N), Cref(M * N);
      for (size_t i = 0; i < M; ++i) for (size_t k = 0; k < K; ++k) Af[i * K + k] = float(A[i * lda + k]);
      for (size_t k = 0; k < K; ++k) for (size_t j = 0; j < N; ++j) Bf[k * N + j] = float(B[k * ldb + j]);
      for (size_t i = 0; i < M; ++i) for (size_t j = 0; j < N; ++j) Cref[i * N + j] = float(C0[i * ldc + j]);
      mx::GemmOptions o; o.backend = mx::BackendId::Ref;
      if (ab) { o.alpha = 1.25f; o.beta = 0.5f; }
      mx::MatrixView af{Af.data(), M, K, K}, bf{Bf.data(), K, N, N}, cf{Cref.data(), M, N, N};
      if (!ab) std::fill(Cref.begin(), Cref.end(), 0.f);
      mx::gemm(cf, af, bf, o);
      mx::MatrixView a{A.data(), M, K, lda, mx::DType::F16}, b{B.data(), K, N, ldb, mx::DType::F16},
          c{C.data(), M, N, ldc, mx::DType::F16};
      for (auto id : {mx::BackendId::Auto, mx::BackendId::GPU, mx::BackendId::ANE}) {
        o.backend = id;
        C = C0;
        if (!ab) for (size_t i = 0; i < M; ++i) for (size_t j = 0; j < N; ++j) C[i * ldc + j] = _Float16(std::nanf(""));
        int rc = mx::gemm(c, a, b, o);
        if (rc == 3) continue;
        double num = 0, den = 0, pad = 0;
        for (size_t i = 0; i < M; ++i)
          for (size_t j = 0; j < N; ++j) {
            double d = double(float(C[i * ldc + j])) - Cref[i * N + j];
            num += d * d; den += double(Cref[i * N + j]) * Cref[i * N + j];
          }
        for (size_t i = 0; i < M; ++i)  // padding columns must be untouched
          for (size_t j = N; j < ldc; ++j) if (!(float(C[i * ldc + j]) == float(C0[i * ldc + j]))) pad += 1;
        double e = rc == 0 ? std::sqrt(num / (den > 0 ? den : 1)) : 1.0;
        bool ok = rc == 0 && e < 3e-3 && pad == 0;
        std::printf("%-4s f16 %zux%zux%zu %s rc=%d err=%.2e pad_clobbered=%.0f %s\n", mx::backend_name(id), M, N, K,
                    ab ? "alpha/beta" : "plain     ", rc, e, pad, ok ? "ok" : "FAIL");
        failures += !ok;
      }
    }
  }
  std::printf("SME available: %d  GPU: %d  ANE: %d\n", mx::backend_available(mx::BackendId::SME),
              mx::backend_available(mx::BackendId::GPU), mx::backend_available(mx::BackendId::ANE));
  return failures ? 1 : 0;
}
