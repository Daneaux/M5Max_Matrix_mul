CXX      ?= clang++
# SME kernels (M1) need +sme2; keep it on the SME translation unit only.
CXXFLAGS := -std=c++20 -O2 -Wall -Wextra -Iinclude -Isrc
SRC_CPP  := src/mx.cpp src/backend_ref.cpp src/backend_sme.cpp
SRC_MM   := src/backend_gpu.mm src/backend_ane.mm
OBJ      := $(patsubst src/%.cpp,build/%.o,$(SRC_CPP)) $(patsubst src/%.mm,build/%.o,$(SRC_MM))

.PHONY: all test bench clean
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
	$(CXX) $(CXXFLAGS) $< build/libmx.a -o $@
build/bench_gemm: bench/bench_gemm.cpp build/libmx.a
	$(CXX) $(CXXFLAGS) $< build/libmx.a -framework Accelerate -o $@

test: build/test_gemm
	./build/test_gemm
bench: build/bench_gemm
	./build/bench_gemm
clean:
	rm -rf build
