# File: Makefile
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

CC=clang
CXX=clang++
NVCC=/usr/local/cuda/bin/nvcc

OUT_DIR=build
SUPPRESS_WARNINGS=1222 2527 2529 2651 2653 2668 2669 2670 2671 2735 2737 2739 20012 20011 20014
CUDA_FLAGS=-O3 -MMD --device-c -Xcompiler -fPIC -D__NVCC__ -DEIGEN_WARNINGS_DISABLED --expt-relaxed-constexpr --extended-lambda
CUDA_FLAGS += $(foreach warn,$(SUPPRESS_WARNINGS),--diag-suppress=$(warn))
CPP_FLAGS=-O3 -std=c++1z -Wall -Wno-unused-function -Wno-unused-function -Wno-strict-aliasing -Wno-uninitialized -Wno-deprecated-declarations -Wno-unused-lambda-capture -fPIC -MMD -MP
EIGEN_PATH=/usr/include/eigen3

BASE_DIRS=buffer main utility csrmat contact energy eigenanalysis barrier strainlimiting solver
KERNEL_FILES=kernels/reduce.cu kernels/exclusive_scan.cu kernels/vec_ops.cu
CUDA_FILES=$(foreach dir,$(BASE_DIRS),$(dir)/$(dir).cu) $(KERNEL_FILES)
CPP_FILES=$(foreach dir,$(BASE_DIRS),$(dir)/$(dir).cpp)
SIMPLELOG_FILES=simplelog/SimpleLog.cpp

CUDA_OBJS=$(addprefix $(OUT_DIR)/cuda/,$(CUDA_FILES:%.cu=%.o))
CPP_OBJS=$(addprefix $(OUT_DIR)/cpp/,$(CPP_FILES:%.cpp=%.o))
SIMPLELOG_OBJS=$(addprefix $(OUT_DIR)/cpp/,$(SIMPLELOG_FILES:%.cpp=%.o))
LIBDIR=$(OUT_DIR)/lib
DEPS = $(CUDA_OBJS:.o=.d) $(CPP_OBJS:.o=.d) $(SIMPLELOG_OBJS:.o=.d)

all: $(LIBDIR)/libsimbackend_cuda.so $(LIBDIR)/libsimplelog.so

$(OUT_DIR)/cuda/%.o: %.cu
	mkdir -p $(@D)
	$(NVCC) -c $(CUDA_FLAGS) $< -I$(EIGEN_PATH) -o $@

$(OUT_DIR)/cpp/%.o: %.cpp
	mkdir -p $(@D)
	$(CXX) -c $(CPP_FLAGS) $< -I$(EIGEN_PATH) -o $@

$(LIBDIR)/libsimbackend_cuda.so: $(CUDA_OBJS) $(SIMPLELOG_OBJS)
	mkdir -p $(LIBDIR)
	$(NVCC) -shared -o $@ $^

$(LIBDIR)/libsimbackend_cpp.so: $(CPP_OBJS) $(SIMPLELOG_OBJS)
	mkdir -p $(LIBDIR)
	$(CXX) -shared -o $@ $^

$(LIBDIR)/libsimplelog.so: $(SIMPLELOG_OBJS)
	mkdir -p $(LIBDIR)
	$(CXX) -shared -o $@ $^

-include $(DEPS)

.PHONY: clean
clean:
	rm -rf $(OUT_DIR) $(LIBDIR)
