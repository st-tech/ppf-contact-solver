# File: Makefile
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

NVCC=/usr/local/cuda/bin/nvcc

OUT_DIR=build
LIBDIR=$(OUT_DIR)/lib
EIGEN_PATH=/usr/include/eigen3

# Warning suppressions
SUPPRESS_WARNINGS=1222 2527 2529 2651 2653 2668 2669 2670 2671 2735 2737 2739 20012 20011 20014 177 940 1394

# nvcc flags for CUDA files
NVCC_FLAGS=-std=c++17 -O3 -MMD --device-c -Xcompiler -fPIC -D__NVCC__ -DEIGEN_WARNINGS_DISABLED -DTHRUST_IGNORE_DEPRECATED_CPP_DIALECT -DCUB_IGNORE_DEPRECATED_CPP_DIALECT --expt-relaxed-constexpr --extended-lambda -Wno-deprecated-gpu-targets
NVCC_FLAGS += $(foreach warn,$(SUPPRESS_WARNINGS),--diag-suppress=$(warn))

# nvcc flags for C++ files (compiled as host code)
NVCC_CPP_FLAGS=-std=c++17 -O3 -MMD -Xcompiler -fPIC,-Wall,-Wno-unused-function

# Source files
BASE_DIRS=buffer main utility csrmat contact energy eigenanalysis barrier strainlimiting solver lbvh
KERNEL_FILES=kernels/reduce.cu kernels/exclusive_scan.cu kernels/vec_ops.cu kernels/radix_sort.cu
CUDA_FILES=$(foreach dir,$(BASE_DIRS),$(dir)/$(dir).cu) $(KERNEL_FILES)
SIMPLELOG_FILES=simplelog/SimpleLog.cpp
STUB_FILES=stub.cpp

# Object files
CUDA_OBJS=$(addprefix $(OUT_DIR)/cuda/,$(CUDA_FILES:%.cu=%.o))
SIMPLELOG_OBJS=$(addprefix $(OUT_DIR)/cpp/,$(SIMPLELOG_FILES:%.cpp=%.o))
STUB_OBJS=$(addprefix $(OUT_DIR)/cpp/,$(STUB_FILES:%.cpp=%.o))
DEPS = $(CUDA_OBJS:.o=.d) $(SIMPLELOG_OBJS:.o=.d) $(STUB_OBJS:.o=.d)

all: $(LIBDIR)/libsimbackend_cuda.so $(LIBDIR)/libsimplelog.so

# Compile CUDA files
$(OUT_DIR)/cuda/%.o: %.cu
	mkdir -p $(@D)
	$(NVCC) -c $(NVCC_FLAGS) $< -I$(EIGEN_PATH) -o $@

# Compile C++ files with nvcc (uses host compiler internally)
$(OUT_DIR)/cpp/%.o: %.cpp
	mkdir -p $(@D)
	$(NVCC) -c $(NVCC_CPP_FLAGS) $< -I$(EIGEN_PATH) -o $@

# Link CUDA library
$(LIBDIR)/libsimbackend_cuda.so: $(CUDA_OBJS) $(SIMPLELOG_OBJS) $(STUB_OBJS)
	mkdir -p $(LIBDIR)
	$(NVCC) -shared -o $@ $^

# SimpleLog library (for standalone use)
$(LIBDIR)/libsimplelog.so: $(SIMPLELOG_OBJS)
	mkdir -p $(LIBDIR)
	$(NVCC) -shared -o $@ $^

-include $(DEPS)

.PHONY: clean
clean:
	rm -rf $(OUT_DIR)
