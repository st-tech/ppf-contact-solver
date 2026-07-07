# File: Makefile
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

NVCC=/usr/local/cuda/bin/nvcc

OUT_DIR=build
LIBDIR=$(OUT_DIR)/lib

# Warning suppressions
SUPPRESS_WARNINGS=1222 2527 2529 2651 2653 2668 2669 2670 2671 2735 2737 2739 20012 20011 20014 177 940 1394

# nvcc flags for CUDA files. The device solver has no external linear-algebra
# dependency: the in-house crates/.../cpp/linalg headers replaced Eigen, so
# there is no -I<eigen> or -DEIGEN_WARNINGS_DISABLED here (Eigen now lives only
# in the macOS-only cpp_emul emulator, which carries its own include).
NVCC_FLAGS=-std=c++17 -O3 -MMD --device-c -Xcompiler -fPIC -D__NVCC__ --expt-relaxed-constexpr --extended-lambda -Wno-deprecated-gpu-targets
NVCC_FLAGS += $(foreach warn,$(SUPPRESS_WARNINGS),--diag-suppress=$(warn))

# Device link-time optimization (LTO). Compile each TU to an LTO intermediate
# (code=lto_86); the device link below (-dlto) runs whole-program optimization,
# inlining the cross-TU barrier::compute_stiffness into the contact-Hessian embed
# kernels (a 255-register ABI call otherwise) and roughly halving contact
# matrix-assembly cost.
#
# The LTO IR is at the compute_86 virtual arch, so sm_86 (Ampere) is the floor
# and every device target in the link below emits native SASS from this IR.
# IMPORTANT: device LTO CANNOT embed a JIT-able PTX. The -dlto link lowers its IR
# straight to SASS; a code=compute_XX request is silently dropped (verified with
# cuobjdump: an LTO link yields cubins only, zero PTX). The .so is therefore
# frozen to the exact SASS images we ship, there is no forward-JIT fallback, so
# we enumerate a cubin per supported arch at the link and add a new entry (then
# rebuild) whenever a new GPU generation ships. nvcc rejects -dlto beside -gencode
# at compile, hence code=lto_86 here and the full -gencode list only at the link.
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
	$(NVCC) -c $(NVCC_FLAGS) $< -o $@

# Compile C++ files with nvcc (uses host compiler internally)
$(OUT_DIR)/cpp/%.o: %.cpp Makefile
	mkdir -p $(@D)
	$(NVCC) -c $(NVCC_CPP_FLAGS) $< -o $@

# Link CUDA library: -dlto runs whole-program optimization on the lto_86 objects
# and emits one native SASS cubin per architecture (no PTX, see the note above).
# SASS is binary-compatible forward within a major version, so the sm_86 cubin
# also runs sm_87 (Jetson Orin) and sm_89 (Ada: RTX 40, L40S); each new MAJOR
# needs its own cubin. A brand-new arch not listed here fails to launch ("named
# symbol not found") until its cubin is added and the lib rebuilt.
#   sm_86  Ampere     RTX 30, A10/A40/A6000  (+ sm_87 Orin, sm_89 Ada via compat)
#   sm_89  Ada        RTX 40, L40S, L4       (native; perf on the dev/CI GPU)
#   sm_90  Hopper     H100, H200, GH200
#   sm_100 Blackwell  B200, GB200            (datacenter)
#   sm_120 Blackwell  RTX 50-series          (consumer)
# Keep this list in sync with SUPPORTED_SM in crates/ppf-cts-core/src/utils.rs,
# which gates solver launch against exactly these architectures.
$(LIBDIR)/libsimbackend_cuda.so: $(CUDA_OBJS) $(SIMPLELOG_OBJS) $(STUB_OBJS)
	mkdir -p $(LIBDIR)
	$(NVCC) -shared -dlto \
		-gencode arch=compute_86,code=sm_86 \
		-gencode arch=compute_86,code=sm_89 \
		-gencode arch=compute_86,code=sm_90 \
		-gencode arch=compute_86,code=sm_100 \
		-gencode arch=compute_86,code=sm_120 \
		-o $@ $^

# SimpleLog library (for standalone use)
$(LIBDIR)/libsimplelog.so: $(SIMPLELOG_OBJS)
	mkdir -p $(LIBDIR)
	$(NVCC) -shared -o $@ $^

-include $(DEPS)

.PHONY: clean
clean:
	rm -rf $(OUT_DIR)
