// ANE matmul throughput with the matrix already loaded: chained multiplies inside ONE model.
//
//   ./build/bench_ane_resident            (MX_ANE_MAX=4096 to change the largest n)
//
// Model with L layers:  X1 = A.A, X2 = X1.A, ... X_L = X_{L-1}.A   (A is a single input, n x n FP16).
// One prediction loads A once and runs L matmuls without returning to the app, so per-call costs
// (Core ML dispatch, copying/converting A in and X_L out) are paid once regardless of L.
// We time several L and fit  t(L) = t0 + L * t1 :  t1 = time per matmul, t0 = fixed per-call cost.
//
// Caveat: this removes the per-call load cost, but we cannot observe whether the ANE itself re-reads
// A from DRAM between layers (matrices bigger than its on-chip memory probably are). Compare sizes:
// if TFLOP/s drops for large n, memory is likely the limit, not the matmul units.
//
// The layer field number (1045) and spec layout are the same as src/backend_ane.mm (see there).
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include <unistd.h>

#pragma clang diagnostic ignored "-Wdeprecated-declarations"

using Bytes = std::string;
static void varint(Bytes& b, uint64_t v) { while (v >= 0x80) { b.push_back((char)(v | 0x80)); v >>= 7; } b.push_back((char)v); }
static void tag(Bytes& b, int f, int wt) { varint(b, ((uint64_t)f << 3) | wt); }
static void fvar(Bytes& b, int f, uint64_t v) { tag(b, f, 0); varint(b, v); }
static void fbytes(Bytes& b, int f, const Bytes& v) { tag(b, f, 2); varint(b, v.size()); b += v; }
static Bytes feature(const char* name, int64_t d0, int64_t d1) {
  Bytes packed; varint(packed, d0); varint(packed, d1);
  Bytes arr; fbytes(arr, 1, packed); fvar(arr, 2, 65552);
  Bytes ft; fbytes(ft, 5, arr);
  Bytes fd; fbytes(fd, 1, name); fbytes(fd, 3, ft);
  return fd;
}
static Bytes build_chain(int n, int layers) {
  Bytes desc; fbytes(desc, 1, feature("A", n, n)); fbytes(desc, 10, feature("C", n, n));
  Bytes nn;
  for (int i = 1; i <= layers; ++i) {
    std::string in0 = i == 1 ? "A" : "X" + std::to_string(i - 1);
    std::string out = i == layers ? "C" : "X" + std::to_string(i);
    Bytes layer; fbytes(layer, 1, "mm" + std::to_string(i)); fbytes(layer, 2, in0); fbytes(layer, 2, "A");
    fbytes(layer, 3, out); fbytes(layer, 1045, Bytes());
    fbytes(nn, 1, layer);
  }
  fvar(nn, 5, 1);
  Bytes model; fvar(model, 1, 7); fbytes(model, 2, desc); fbytes(model, 500, nn);
  return model;
}

struct Chain {
  MLModel* model = nil; int ane_layers = 0, total_layers = 0; double build_s = 0;
};

static Chain make_chain(int n, int layers) {
  Chain c;
  @autoreleasepool {
    auto t0 = std::chrono::steady_clock::now();
    Bytes spec = build_chain(n, layers);
    NSString* path = [NSTemporaryDirectory() stringByAppendingPathComponent:[NSString stringWithFormat:@"mx_chain_%d_%d_%d.mlmodel", (int)getpid(), n, layers]];
    [[NSData dataWithBytes:spec.data() length:spec.size()] writeToFile:path atomically:YES];
    NSError* err = nil;
    NSURL* cu = [MLModel compileModelAtURL:[NSURL fileURLWithPath:path] error:&err];
    if (!cu) { fprintf(stderr, "compile failed: %s\n", err.localizedDescription.UTF8String); return c; }
    MLModelConfiguration* cfg = [MLModelConfiguration new];
    cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    c.model = [MLModel modelWithContentsOfURL:cu configuration:cfg error:&err];
    if (!c.model) { fprintf(stderr, "load failed: %s\n", err.localizedDescription.UTF8String); return c; }
    c.build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    __block int on = 0, total = 0;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [MLComputePlan loadContentsOfURL:cu configuration:cfg completionHandler:^(MLComputePlan* plan, NSError*) {
      if (plan && plan.modelStructure.neuralNetwork)
        for (MLModelStructureNeuralNetworkLayer* l in plan.modelStructure.neuralNetwork.layers) {
          total++;
          MLComputePlanDeviceUsage* u = [plan computeDeviceUsageForNeuralNetworkLayer:l];
          if (u && [u.preferredComputeDevice isKindOfClass:[MLNeuralEngineComputeDevice class]]) on++;
        }
      dispatch_semaphore_signal(sem);
    }];
    dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 60 * NSEC_PER_SEC));
    c.ane_layers = on; c.total_layers = total;
  }
  return c;
}

static MLMultiArray* wrap(_Float16* p, int n) {
  NSError* err = nil;
  return [[MLMultiArray alloc] initWithDataPointer:p shape:@[@(n), @(n)] dataType:MLMultiArrayDataTypeFloat16
                                           strides:@[@(n), @1] deallocator:nil error:&err];
}

// Median wall time of one prediction.
static double time_predict(const Chain& c, _Float16* A, _Float16* Cout, int n, int iters) {
  @autoreleasepool {
    NSError* err = nil;
    MLMultiArray* a = wrap(A, n);
    MLMultiArray* cb = wrap(Cout, n);
    MLDictionaryFeatureProvider* in = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"A": a} error:&err];
    MLPredictionOptions* opts = [MLPredictionOptions new];
    opts.outputBackings = @{@"C": cb};
    if (![c.model predictionFromFeatures:in options:opts error:&err]) { fprintf(stderr, "predict failed: %s\n", err.localizedDescription.UTF8String); return -1; }
    std::vector<double> t;
    for (int i = 0; i < iters; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      [c.model predictionFromFeatures:in options:opts error:&err];
      t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
  }
}

int main() {
  int max_n = 4096;
  if (const char* e = getenv("MX_ANE_MAX")) max_n = atoi(e);

  // ---- correctness: A = cyclic shift permutation, so the result is exactly A^(L+1), a shift by L+1 (checks the chain really runs L matmuls).
  {
    const int n = 256;
    for (int L : {1, 7, 20}) {
      Chain c = make_chain(n, L);
      if (!c.model) return 1;
      std::vector<_Float16> A(n * n, _Float16(0)), C(n * n, _Float16(-1));
      for (int i = 0; i < n; ++i) A[i * n + (i + 1) % n] = _Float16(1);  // row i has 1 at column i+1
      time_predict(c, A.data(), C.data(), n, 1);
      // X_1 = A.A, X_L = X_{L-1}.A  =>  X_L = A^(L+1); (A^(L+1))[i][(i+L+1)%n] = 1
      int bad = 0;
      for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) if (float(C[i * n + j]) != (j == (i + L + 1) % n ? 1.f : 0.f)) ++bad;
      printf("check n=%d L=%-2d (permutation^(L+1)): %s  [%d/%d layers on ANE]\n", n, L, bad ? "WRONG" : "ok", c.ane_layers, c.total_layers);
      if (bad) return 1;
    }
  }

  printf("\nPer-matmul ANE time with the matrix already loaded (FP16, n x n . n x n)\n");
  printf("%6s | %8s %8s %8s %8s %8s | %10s %10s | %10s | %s\n", "n", "L=1", "L=10", "L=25", "L=50", "L=100", "t1/matmul", "t0 fixed", "TFLOP/s", "on ANE");
  printf("%6s | %45s | %10s %10s | %10s |\n", "", "median ms per prediction", "ms", "ms", "(2n^3/t1)");
  const int Ls[] = {1, 10, 25, 50, 100};
  for (int n = 256; n <= max_n; n *= 2) {
    std::vector<_Float16> A(n * n), C(n * n);
    std::mt19937 rng(1);
    std::normal_distribution<float> dist(0.f, 1.f / std::sqrt((float)n));  // keeps |X| ~ stable across L multiplies
    for (auto& x : A) x = _Float16(dist(rng));
    double t[5]; int on_all = 1, tot_all = 0, on_sum = 0;
    for (int k = 0; k < 5; ++k) {
      Chain c = make_chain(n, Ls[k]);
      if (!c.model) { t[k] = -1; continue; }
      on_sum += c.ane_layers; tot_all += c.total_layers; if (c.ane_layers != c.total_layers) on_all = 0;
      t[k] = time_predict(c, A.data(), C.data(), n, n >= 2048 ? 5 : 9);
    }
    // least-squares fit t = t0 + L*t1
    double sx = 0, sy = 0, sxx = 0, sxy = 0; int m = 0;
    for (int k = 0; k < 5; ++k) if (t[k] > 0) { double x = Ls[k]; sx += x; sy += t[k]; sxx += x * x; sxy += x * t[k]; ++m; }
    double t1 = (m * sxy - sx * sy) / (m * sxx - sx * sx), t0 = (sy - t1 * sx) / m;
    printf("%6d | %8.2f %8.2f %8.2f %8.2f %8.2f | %10.4f %10.2f | %10.2f | %d/%d%s\n", n, t[0] * 1e3, t[1] * 1e3, t[2] * 1e3,
           t[3] * 1e3, t[4] * 1e3, t1 * 1e3, t0 * 1e3, 2.0 * n * n * n / t1 / 1e12, on_sum, tot_all, on_all ? "" : "  (some layers NOT on ANE)");
  }
  return 0;
}
