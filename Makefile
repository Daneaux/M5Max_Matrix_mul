CXX      ?= clang++
# SME kernels (M1) need +sme2; keep it on the SME translation unit only.
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Iinclude -Isrc
FRAMEWORKS := -framework Metal -framework CoreML -framework Foundation
SRC_CPP  := src/mx.cpp src/backend_ref.cpp src/backend_sme.cpp
SRC_MM   := src/backend_gpu.mm src/backend_ane.mm
OBJ      := $(patsubst src/%.cpp,build/%.o,$(SRC_CPP)) $(patsubst src/%.mm,build/%.o,$(SRC_MM))

.PHONY: all test bench bandwidth ane-resident gpu-resident scaling scalar scalar-asm neon clean
all: build/libmx.a

build/%.o: src/%.cpp src/backend.h include/mx/mx.h
	@mkdir -p build && $(CXX) $(CXXFLAGS) -c $< -o $@
build/backend_sme.o: src/backend_sme.cpp src/backend.h include/mx/mx.h
	@mkdir -p build && $(CXX) $(CXXFLAGS) -mcpu=apple-m4+sme2 -c $< -o $@
build/%.o: src/%.mm src/backend.h include/mx/mx.h
	@mkdir -p build && $(CXX) $(CXXFLAGS) -x objective-c++ -fobjc-arc -c $< -o $@

build/libmx.a: $(OBJ)
	ar rcs $@ $^

build/test_gemm: tests/test_gemm.cpp build/libmx.a
	$(CXX) $(CXXFLAGS) $< build/libmx.a $(FRAMEWORKS) -o $@
build/bench_gemm: bench/bench_gemm.cpp build/libmx.a
	$(CXX) $(CXXFLAGS) $< build/libmx.a -framework Accelerate $(FRAMEWORKS) -o $@

build/bench_bandwidth: bench/bench_bandwidth.mm build/libmx.a
	$(CXX) $(CXXFLAGS) -x objective-c++ -fobjc-arc $< -x none build/libmx.a $(FRAMEWORKS) -o $@

build/bench_ane_resident: bench/bench_ane_resident.mm
	$(CXX) $(CXXFLAGS) -x objective-c++ -fobjc-arc $< $(FRAMEWORKS) -o $@

build/bench_gpu_resident: bench/bench_gpu_resident.mm
	$(CXX) $(CXXFLAGS) -x objective-c++ -fobjc-arc $< $(FRAMEWORKS) -o $@

build/bench_scaling: bench/bench_scaling.mm build/libmx.a
	$(CXX) $(CXXFLAGS) -x objective-c++ -fobjc-arc $< -x none build/libmx.a $(FRAMEWORKS) -o $@

build/bench_scalar: bench/bench_scalar.cpp
	$(CXX) $(CXXFLAGS) -fno-vectorize -fno-slp-vectorize $< -o $@

scalar: build/bench_scalar
	./build/bench_scalar
# Shows the compiled inner loops: expect scalar fmadd/fmul/fadd on s registers and zero .4s/.2d vector ops.
scalar-asm:
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -fno-vectorize -fno-slp-vectorize -S bench/bench_scalar.cpp -o build/bench_scalar.s
	@echo "vector (.4s/.2s/.2d) instructions: $$(grep -cE '\.(4s|2s|2d)' build/bench_scalar.s)"
	@grep -E 'fmadd|fmul|fadd' build/bench_scalar.s | awk '{print $$1}' | sort | uniq -c

build/bench_neon: bench/bench_neon.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

neon: build/bench_neon
	./build/bench_neon

build/prof_threadgroups: bench/prof_threadgroups.mm
	$(CXX) $(CXXFLAGS) -x objective-c++ -fobjc-arc $< $(FRAMEWORKS) -o $@

test: build/test_gemm
	./build/test_gemm
bench: build/bench_gemm
	./build/bench_gemm
bandwidth: build/bench_bandwidth
	./build/bench_bandwidth
ane-resident: build/bench_ane_resident
	./build/bench_ane_resident
gpu-resident: build/bench_gpu_resident
	./build/bench_gpu_resident
scaling: build/bench_scaling
	./build/bench_scaling
clean:
	rm -rf build
