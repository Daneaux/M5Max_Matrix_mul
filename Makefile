CXX      ?= clang++
# SME kernels (M1) need +sme2; keep it on the SME translation unit only.
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Iinclude -Isrc
FRAMEWORKS := -framework Metal -framework CoreML -framework Foundation
SRC_CPP  := src/mx.cpp src/backend_ref.cpp src/backend_sme.cpp
SRC_MM   := src/backend_gpu.mm src/backend_ane.mm
OBJ      := $(patsubst src/%.cpp,build/%.o,$(SRC_CPP)) $(patsubst src/%.mm,build/%.o,$(SRC_MM))

.PHONY: all test bench bandwidth ane-resident gpu-resident scaling clean
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
