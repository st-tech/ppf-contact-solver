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
NVCC_FLAGS=-std=c++17 -O3 -MMD --device-c -Xcompiler -fPIC -D__NVCC__ -DEIGEN_WARNINGS_DISABLED --expt-relaxed-constexpr --extended-lambda -Wno-deprecated-gpu-targets
NVCC_FLAGS += $(foreach warn,$(SUPPRESS_WARNINGS),--diag-suppress=$(warn))

# Device link-time optimization (LTO). Compile each TU to an LTO intermediate
# (code=lto_86); the device link below (-dlto) runs whole-program optimization,
# inlining the cross-TU barrier::compute_stiffness into the contact-Hessian embed
# kernels (a 255-register ABI call otherwise) and roughly halving contact
# matrix-assembly cost. The .so carries an sm_86 (Ampere / RTX 30-series, the
# CUDA 12.8 floor) SASS anchor plus the LTO-optimized compute_86 PTX; the driver
# JITs and caches that PTX for newer GPUs (sm_89, sm_90, ...), so the binary is
# not frozen to today's architectures. (nvcc rejects -dlto beside -gencode at
# compile, but requires both at the link; a PTX-only LTO link is not possible,
# hence the SASS anchor.)
NVCC_FLAGS += -gencode arch=compute_86,code=lto_86

# nvcc flags for C++ files (compiled as host code)
NVCC_CPP_FLAGS=-std=c++17 -O3 -MMD -Xcompiler -fPIC,-Wall,-Wno-unused-function

# Source files
BASE_DIRS=buffer main utility csrmat contact energy eigenanalysis barrier strainlimiting solver lbvh plasticity schwarz
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

# Objects depend on the Makefile so a flags change forces a recompile (nvcc's
# mtime view only tracks sources; without this a flag-only edit relinks stale .o).
$(OUT_DIR)/cuda/%.o: %.cu Makefile
	mkdir -p $(@D)
	$(NVCC) -c $(NVCC_FLAGS) $< -I$(EIGEN_PATH) -o $@

# Compile C++ files with nvcc (uses host compiler internally)
$(OUT_DIR)/cpp/%.o: %.cpp Makefile
	mkdir -p $(@D)
	$(NVCC) -c $(NVCC_CPP_FLAGS) $< -I$(EIGEN_PATH) -o $@

# Link CUDA library: -dlto runs whole-program optimization on the lto_86 objects
# and emits the sm_86 SASS anchor + compute_86 PTX (both gencodes must sit beside
# -dlto here; nvcc only forbids that combination at compile).
$(LIBDIR)/libsimbackend_cuda.so: $(CUDA_OBJS) $(SIMPLELOG_OBJS) $(STUB_OBJS)
	mkdir -p $(LIBDIR)
	$(NVCC) -shared -dlto -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 -o $@ $^

# SimpleLog library (for standalone use)
$(LIBDIR)/libsimplelog.so: $(SIMPLELOG_OBJS)
	mkdir -p $(LIBDIR)
	$(NVCC) -shared -o $@ $^

-include $(DEPS)

.PHONY: clean
clean:
	rm -rf $(OUT_DIR)
