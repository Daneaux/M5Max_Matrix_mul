// ANE backend: FP16 GEMM on the Apple Neural Engine via Core ML.
//
// There is no public ANE API. We build, at runtime, a tiny Core ML NeuralNetwork model with a
// single batched-matmul layer (inputs A[M,K], B[K,N] FP16, output C[M,N] FP16, no weights),
// compile it with +[MLModel compileModelAtURL:], load it with computeUnits =
// CPUAndNeuralEngine, and cache it per (M,K,N). Shapes are static (the ANE needs that).
//
// The model is hand-encoded protobuf (CoreML Model.proto / NeuralNetwork.proto). Field numbers
// used (the layer number 1045 was found empirically and verified numerically; the others are
// from the published spec):
//   Model: specificationVersion=1 (>=7 for FP16 features), description=2, neuralNetwork=500
//   ModelDescription: input=1, output=10;  FeatureDescription: name=1, type=3
//   FeatureType.multiArrayType=5;  ArrayFeatureType: shape=1 (packed), dataType=2 (FLOAT16=65552)
//   NeuralNetwork: layers=1, arrayInputShapeMapping=5 (EXACT=1)
//   NeuralNetworkLayer: name=1, input=2, output=3, batchedMatmul=1045
// Because the layer number is not from the spec text, init() runs a self-test: a 64^3 matmul is
// checked numerically against the CPU, and MLComputePlan must report the Neural Engine as the
// preferred device for the layer. If either fails the backend reports itself unavailable.
//
// alpha/beta: not part of the model. alpha==1,beta==0 writes C directly (via output backing);
// otherwise the product goes to a temporary and a CPU epilogue combines it with C.
// Inputs/outputs are wrapped without copying where Core ML allows; the ANE itself copies.
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include "backend.h"

#pragma clang diagnostic ignored "-Wdeprecated-declarations"  // MLMultiArray.dataPointer

namespace mx {
namespace {

using MModel = MLModel*;  // `id` is shadowed by Backend::id() inside the class; avoid ObjC `id` there.

constexpr int kBatchedMatmulField = 1045;
constexpr int kSpecVersion = 7;

// ---- minimal protobuf writer ------------------------------------------------
using Bytes = std::string;
void varint(Bytes& b, uint64_t v) {
  while (v >= 0x80) { b.push_back(static_cast<char>(v | 0x80)); v >>= 7; }
  b.push_back(static_cast<char>(v));
}
void put_tag(Bytes& b, int field, int wt) { varint(b, (static_cast<uint64_t>(field) << 3) | wt); }
void put_varint(Bytes& b, int field, uint64_t v) { put_tag(b, field, 0); varint(b, v); }
void put_bytes(Bytes& b, int field, const Bytes& v) { put_tag(b, field, 2); varint(b, v.size()); b += v; }

Bytes feature(const char* name, int64_t d0, int64_t d1) {
  Bytes packed; varint(packed, d0); varint(packed, d1);
  Bytes arr; put_bytes(arr, 1, packed); put_varint(arr, 2, 65552 /*FLOAT16*/);
  Bytes ft; put_bytes(ft, 5, arr);
  Bytes fd; put_bytes(fd, 1, name); put_bytes(fd, 3, ft);
  return fd;
}

Bytes build_spec(int64_t M, int64_t K, int64_t N) {
  Bytes desc;
  put_bytes(desc, 1, feature("A", M, K));
  put_bytes(desc, 1, feature("B", K, N));
  put_bytes(desc, 10, feature("C", M, N));
  Bytes layer;
  put_bytes(layer, 1, "mm"); put_bytes(layer, 2, "A"); put_bytes(layer, 2, "B"); put_bytes(layer, 3, "C");
  put_bytes(layer, kBatchedMatmulField, Bytes());  // BatchedMatMulLayerParams: all defaults
  Bytes nn; put_bytes(nn, 1, layer); put_varint(nn, 5, 1 /*EXACT_ARRAY_MAPPING*/);
  Bytes model; put_varint(model, 1, kSpecVersion); put_bytes(model, 2, desc); put_bytes(model, 500, nn);
  return model;
}

struct Entry { MModel model = nil; };

class AneBackend final : public Backend {
 public:
  BackendId id() const override { return BackendId::ANE; }
  bool available() const override { return const_cast<AneBackend*>(this)->init(); }

  bool supports(const GemmDesc& d) const override {
    constexpr size_t kMaxDim = 16384;
    return d.A.dtype == DType::F16 && d.B.dtype == DType::F16 && d.C.dtype == DType::F16 && d.m() > 0 &&
           d.n() > 0 && d.k() > 0 && d.m() <= kMaxDim && d.n() <= kMaxDim && d.k() <= kMaxDim &&
           d.A.cols == d.B.rows && d.C.rows == d.A.rows && d.C.cols == d.B.cols && d.A.ld >= d.A.cols &&
           d.B.ld >= d.B.cols && d.C.ld >= d.C.cols;
  }

  double estimate_ns(const GemmDesc& d) const override {
    // Measured on M5 Max: ~0.25 ms fixed per call, ~4.5 TFLOP/s sustained at large sizes.
    double ns = 250e3 + d.flops() / 4500.0;
    if (!is_cached(d.m(), d.k(), d.n())) ns += 30e6;  // build + compile + load a model for a new shape
    return ns;
  }

  int gemm(const GemmDesc& d) override {
    if (!init() || !supports(d)) return 1;
    std::lock_guard<std::mutex> lock(mu_);
    @autoreleasepool {
      MModel model = get_model(d.m(), d.k(), d.n());
      if (!model) return 5;
      return run(model, d);
    }
  }

  const std::string& last_error() const { return error_; }

 private:
  static uint64_t key(size_t m, size_t k, size_t n) { return (uint64_t(m) << 42) ^ (uint64_t(k) << 21) ^ n; }

  bool is_cached(size_t m, size_t k, size_t n) const {
    std::lock_guard<std::mutex> l(cache_mu_);
    return cache_.count(key(m, k, n)) != 0;
  }

  MModel get_model(size_t m, size_t k, size_t n) {
    const uint64_t kk = key(m, k, n);
    {
      std::lock_guard<std::mutex> l(cache_mu_);
      auto it = cache_.find(kk);
      if (it != cache_.end()) return it->second.model;
    }
    MModel model = build_model(m, k, n);
    if (!model) return nil;
    std::lock_guard<std::mutex> l(cache_mu_);
    if (order_.size() >= 32) {  // small LRU-ish (FIFO) cap: each model holds ANE resources
      cache_.erase(order_.front());
      order_.pop_front();
    }
    cache_[kk].model = model;
    order_.push_back(kk);
    return model;
  }

  MModel build_model(size_t m, size_t k, size_t n) {
    @autoreleasepool {
      Bytes spec = build_spec(m, k, n);
      NSString* dir = [NSTemporaryDirectory() stringByAppendingPathComponent:
                          [NSString stringWithFormat:@"mxlib-%d", (int)getpid()]];
      [[NSFileManager defaultManager] createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
      NSString* path = [dir stringByAppendingPathComponent:
                          [NSString stringWithFormat:@"mm_%zux%zux%zu.mlmodel", m, k, n]];
      NSData* data = [NSData dataWithBytes:spec.data() length:spec.size()];
      if (![data writeToFile:path atomically:YES]) { error_ = "cannot write model file"; return nil; }
      NSError* err = nil;
      NSURL* compiled = [MLModel compileModelAtURL:[NSURL fileURLWithPath:path] error:&err];
      if (!compiled) { error_ = err.localizedDescription.UTF8String; return nil; }
      MLModelConfiguration* cfg = [MLModelConfiguration new];
      cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
      MModel model = [MLModel modelWithContentsOfURL:compiled configuration:cfg error:&err];
      if (!model) { error_ = err.localizedDescription.UTF8String; return nil; }
      if (m == 64 && k == 64 && n == 64) placement_on_ane_ = check_placement(compiled, cfg);
      return model;
    }
  }

  // True iff MLComputePlan says the Neural Engine is the preferred device for the matmul layer.
  static bool check_placement(NSURL* compiled, MLModelConfiguration* cfg) {
    __block bool on_ane = false;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    [MLComputePlan loadContentsOfURL:compiled configuration:cfg
                   completionHandler:^(MLComputePlan* plan, NSError*) {
      if (plan && plan.modelStructure.neuralNetwork) {
        for (MLModelStructureNeuralNetworkLayer* l in plan.modelStructure.neuralNetwork.layers) {
          MLComputePlanDeviceUsage* u = [plan computeDeviceUsageForNeuralNetworkLayer:l];
          if (u && [u.preferredComputeDevice isKindOfClass:[MLNeuralEngineComputeDevice class]]) on_ane = true;
        }
      }
      dispatch_semaphore_signal(sem);
    }];
    dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 20 * NSEC_PER_SEC));
    return on_ane;
  }

  static MLMultiArray* wrap(void* p, size_t rows, size_t cols, size_t ld) {
    NSError* err = nil;
    return [[MLMultiArray alloc] initWithDataPointer:p
                                               shape:@[@(rows), @(cols)]
                                            dataType:MLMultiArrayDataTypeFloat16
                                             strides:@[@(ld), @1]
                                         deallocator:nil
                                               error:&err];
  }

  int run(MModel model, const GemmDesc& d) {
    const bool fast = d.alpha == 1.f && d.beta == 0.f;
    std::vector<_Float16> tmp;
    _Float16* cptr = static_cast<_Float16*>(d.C.data);
    size_t cld = d.C.ld;
    if (!fast) { tmp.resize(d.m() * d.n()); cptr = tmp.data(); cld = d.n(); }

    MLMultiArray* A = wrap(d.A.data, d.A.rows, d.A.cols, d.A.ld);
    MLMultiArray* B = wrap(d.B.data, d.B.rows, d.B.cols, d.B.ld);
    MLMultiArray* Cb = wrap(cptr, d.m(), d.n(), cld);
    if (!A || !B || !Cb) return 6;

    NSError* err = nil;
    MLDictionaryFeatureProvider* in = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"A": A, @"B": B} error:&err];
    if (!in) return 6;
    MLPredictionOptions* opts = [MLPredictionOptions new];
    opts.outputBackings = @{@"C": Cb};
    auto out = [model predictionFromFeatures:in options:opts error:&err];
    if (!out) { error_ = err.localizedDescription.UTF8String; return 7; }

    MLMultiArray* Co = [out featureValueForName:@"C"].multiArrayValue;
    if (!Co) return 8;
    if (Co.dataPointer != static_cast<void*>(cptr)) {  // Core ML did not honor the backing: copy out
      const _Float16* src = static_cast<const _Float16*>(Co.dataPointer);
      const size_t sr = Co.strides[0].unsignedLongValue, sc = Co.strides[1].unsignedLongValue;
      for (size_t i = 0; i < d.m(); ++i)
        for (size_t j = 0; j < d.n(); ++j) cptr[i * cld + j] = src[i * sr + j * sc];
    }
    if (!fast) {
      auto* C = static_cast<_Float16*>(d.C.data);
      for (size_t i = 0; i < d.m(); ++i) {
        const _Float16* t = tmp.data() + i * d.n();
        _Float16* c = C + i * d.C.ld;
        for (size_t j = 0; j < d.n(); ++j) {
          float r = d.alpha * static_cast<float>(t[j]);
          if (d.beta != 0.f) r += d.beta * static_cast<float>(c[j]);
          c[j] = static_cast<_Float16>(r);
        }
      }
    }
    return 0;
  }

  // 64^3 numeric check + placement check; protects against the empirical layer number changing.
  bool selftest() {
    const size_t n = 64;
    std::vector<_Float16> a(n * n), b(n * n), c(n * n, _Float16(0));
    for (size_t i = 0; i < n * n; ++i) { a[i] = _Float16(((i * 7) % 13) * 0.1f); b[i] = _Float16(((i * 5) % 11) * 0.1f); }
    placement_on_ane_ = false;
    GemmDesc d{{c.data(), n, n, n, DType::F16}, {a.data(), n, n, n, DType::F16}, {b.data(), n, n, n, DType::F16}, 1.f, 0.f};
    int rc;
    @autoreleasepool {
      MModel model = get_model(n, n, n);
      if (!model) { if (error_.empty()) error_ = "selftest: model build failed"; return false; }
      rc = run(model, d);
    }
    if (rc != 0) { error_ = "selftest: run failed (rc=" + std::to_string(rc) + ") " + error_; return false; }
    for (size_t i = 0; i < n; i += 5)
      for (size_t j = 0; j < n; j += 5) {
        double r = 0;
        for (size_t k = 0; k < n; ++k) r += double(float(a[i * n + k])) * double(float(b[k * n + j]));
        if (std::fabs(double(float(c[i * n + j])) - r) > 0.01 * (std::fabs(r) + 1)) {
          error_ = "selftest: wrong numeric result (layer field number changed?)";
          return false;
        }
      }
    if (!placement_on_ane_) { error_ = "selftest: layer is not placed on the Neural Engine"; return false; }
    return true;
  }

  bool init() {
    std::lock_guard<std::recursive_mutex> g(init_mu_);
    if (state_ != 0) return state_ > 0;
    state_ = -1;
    std::lock_guard<std::mutex> lock(mu_);
    if (selftest()) state_ = 1;
    return state_ > 0;
  }

  std::mutex mu_;
  std::recursive_mutex init_mu_;
  mutable std::mutex cache_mu_;
  int state_ = 0;
  bool placement_on_ane_ = false;
  std::string error_;
  std::unordered_map<uint64_t, Entry> cache_;
  std::deque<uint64_t> order_;
};
}  // namespace

Backend* make_ane_backend() { static AneBackend b; return &b; }
}  // namespace mx
