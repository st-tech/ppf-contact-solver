# File: Makefile
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The ROCm backend library, on the same terms as cuda/Makefile beside it: one
# object per generated entry point, this directory's own mechanism, and the C
# boundary the neutral Rust driver calls. No simulation is compiled here.

HIPCC ?= $(ROCM_PATH)/bin/hipcc
OUT_DIR = build
LIBDIR = $(OUT_DIR)/lib

.DEFAULT_GOAL := all

# WHERE THE NEUTRAL KERNEL TREE IS, SUPPLIED BY THE CALLER, for the reason
# cuda/Makefile states: this recipe belongs to ppf-cts-compute and the sources it
# compiles belong to ppf-cts-solver, so the path between them is an input rather
# than a constant.
KERNEL_ROOT ?= ../../ppf-cts-solver/src/kernels
ifeq ($(wildcard $(KERNEL_ROOT)/data.hpp),)
$(error KERNEL_ROOT=$(KERNEL_ROOT) does not hold data.hpp: pass the neutral kernel tree of ppf-cts-solver)
endif

ENTRY_CONTEXT_HEADERS ?= data.hpp linalg/eigsolve.hpp
ENTRY_CONTEXT_FLAGS = $(foreach header,$(ENTRY_CONTEXT_HEADERS),-include $(header))

# AMD targets, read from the manifest beside this file rather than spelled out
# here. rocm_arch.txt records why it is the single source and which consumers
# derive from it. Parsed with awk off a ';'-commented, space-separated file so a
# .bat can read the same file with cmd.exe's `for /f`.
ROCM_ARCH_FILE := rocm_arch.txt
ROCM_TARGETS := $(strip $(shell awk '$$1=="target"{print $$2}' $(ROCM_ARCH_FILE) 2>/dev/null))
# A missing or unparsable manifest stops the build here. Continuing on an empty
# list would link a library carrying no device code, which surfaces much later as
# every GPU being rejected at run time, and that failure names the device rather
# than the file that actually went missing.
ifeq ($(ROCM_TARGETS),)
$(error no 'target' line in $(CURDIR)/$(ROCM_ARCH_FILE))
endif
OFFLOAD_FLAGS = $(foreach t,$(ROCM_TARGETS),--offload-arch=$(t))

# THE NVIDIA PLATFORM IS THE ONLY WAY THIS BACKEND CAN BE EXECUTED while the
# project has no AMD hardware, so the recipe carries it rather than leaving it to
# a hand-typed command line. `HIP_PLATFORM=nvidia` makes hipcc drive nvcc, and
# three things change with it:
#
#   * the AMD target list means nothing; an NVIDIA architecture is named instead
#     (NVIDIA_ARCH, defaulted to the dev fleet's L40S),
#   * nvcc dispatches on the FILE EXTENSION and does not know `.hip`, so `-x cu`
#     is required or it refuses the file outright,
#   * `-x cu` also suppresses the `-isystem <sdk>/include` hipcc would otherwise
#     add, so the HIP headers have to be named explicitly.
#
# `-fgpu-rdc` is an amdclang flag and has no meaning here: nvcc's equivalent is
# `-rdc=true`.
NVIDIA_ARCH ?= sm_89
ifeq ($(HIP_PLATFORM),nvidia)
#
# AND THE CUDA PROLOGUE DIRECTORY IS REQUIRED HERE, which reads like a mistake
# and is not. `seam/seam.hpp` tests `__CUDACC__` BEFORE `__HIPCC__`, and on this
# platform hipcc drives nvcc, so `__CUDACC__` is defined and the CUDA prologue is
# the correct one: the intrinsics available are CUDA's. `seam_hip.hiph` is
# therefore NOT exercised by this leg, which is the staging leg's blind spot:
# the seam file an AMD platform compiles is the one this leg never reads.
# `-x cu` IS A COMPILE-ONLY FLAG and must not reach the link. It tells the
# driver to treat its inputs as CUDA SOURCE, so on the link line it makes the
# compiler read the `.o` files as text: measured, `error: extended character is
# not valid in an identifier` pointing at `backend.o:2`, which names an object
# nobody wrote. Hence two variables rather than one.
PLATFORM_FLAGS = -x cu -arch=$(NVIDIA_ARCH) -rdc=true -isystem $(ROCM_PATH)/include \
                 -I ../cuda --expt-relaxed-constexpr
PLATFORM_LINK_FLAGS = -arch=$(NVIDIA_ARCH) -rdc=true
# nvcc does NOT forward a bare -fPIC to the HOST compiler, so the objects come
# out position DEPENDENT and the shared link fails with `relocation
# R_X86_64_PC32 against symbol stderr can not be used when making a shared
# object`. cuda/Makefile spells it this way for the same reason.
# THE `=` FORM IS REQUIRED, not a style choice. Measured: with
# `-Xcompiler -fPIC` as two tokens, hipcc driving nvcc emits
# `-Xcompiler -x cu` and `-fPIC` is gone, so the objects come out position
# dependent and the shared link fails much later on a text relocation. A single
# token cannot be split that way.
PIC_FLAG = -Xcompiler=-fPIC
OBJ_SUFFIX = -nvidia
else
# __FILE__ MUST NOT NAME THE MACHINE THAT BUILT THIS, and relative recipe
# paths are not enough on this platform. The solver's build script passes
# KERNEL_ROOT and OUT_DIR relative, which covers every path this recipe spells,
# but a body rendering opens with `#line 1 "<absolute neutral file>"` and
# includes neutral headers by absolute path, both written by kernelgen.py, so a
# device-side __FILE__ in a body still names the build tree. Measured with
# TheRock 10.0.0: eight copies of pdrd_lock_projector.kernel.cpp's absolute
# path, one per device target inside .hip_fatbin, which build-linux-native's
# gate C refuses. amdclang applies the map to a name a `#line` supplied, so the
# two roots are mapped to what `build::host` maps them to, a spelling relative
# to the root, which still locates the file in any checkout. The file id the
# diagnostic channel records is a hash of the same string, and this library
# resolves no id back to a path (be_diag_file_path), so no reader depends on
# the absolute spelling. Both variables expand lazily, so KERNELGEN_DIR being
# defined further down is fine.
FILE_PREFIX_FLAGS = -fmacro-prefix-map=$(abspath $(KERNEL_ROOT))/= \
                    -fmacro-prefix-map=$(abspath $(KERNELGEN_DIR))/=
PLATFORM_FLAGS = $(OFFLOAD_FLAGS) -fgpu-rdc $(FILE_PREFIX_FLAGS)
# THE LIBRARY RECORDS WHERE ITS HIP RUNTIME IS. It needs libamdhip64.so.7 from
# the SDK this recipe compiled against, and nothing else on the machine names
# that directory: an SDK unpacked from AMD's tarball registers no loader path the
# way a system package install does. Measured with TheRock 10.0.0, a solver
# linked without this fails to start with `libamdhip64.so.7: cannot open shared
# object file` unless LD_LIBRARY_PATH names the SDK. A distribution rewrites the
# entry to $ORIGIN (build-linux-native/bundle.sh step 6), so the SDK path never
# ships.
PLATFORM_LINK_FLAGS = $(OFFLOAD_FLAGS) -fgpu-rdc -Wl,-rpath,$(ROCM_PATH)/lib
PIC_FLAG = -fPIC
OBJ_SUFFIX =
endif

# THE TWO PLATFORMS MUST NOT SHARE AN OBJECT DIRECTORY, because switching
# platform changes no SOURCE. The renderings are the same bytes either way, so
# they are shared; the objects are amdgcn code objects on one platform and CUDA
# fatbins on the other, and with one directory make finds them up to date by
# mtime and links whichever platform built last. The library keeps ONE name on
# purpose: which platform a linked library came from is answered by
# `be_backend_name()`, which reports `hip-nvidia` for the staging leg and which
# `ppf-contact-solver --backend` prints, rather than by a filename a reader has
# to notice. That identity is load-bearing precisely because no AMD run will
# ever contradict a misattribution.
#
# AND SPLITTING THE OBJECTS IS NOT ENOUGH ON ITS OWN, which is the half that was
# missed the first time. One library NAME means make compares that one file's
# mtime against the objects, so after a build of the OTHER platform the library
# is newer than this platform's objects and `make abi` answers "Nothing to be
# done" while leaving the other platform's library in place. Measured: an AMD
# build after an NVIDIA one left a `.so` carrying `.nv_fatbin` sections, which a
# reader asking for AMD has no reason to inspect. The stamp below is what make
# has to compare against instead: it is named for the platform, so the one for
# the platform being built is ABSENT after a switch, its rule fires, and the
# library relinks. It also deletes the other platform's stamp, so the two can
# never both look current.

# -fgpu-rdc IS REQUIRED, NOT AN OPTIMIZATION. The arena's device-side globals
# (compute::arena::g_bases and its siblings) are defined in this directory's own
# translation unit and referenced from every generated entry, so without
# relocatable device code each entry fails its own device link on an undefined
# protected symbol. It is the HIP counterpart of the CUDA build's --device-c.
#
# NO FAST MATH, AND THAT IS A CORRECTNESS SETTING. `-ffast-math` selects
# different device-library variants through the oclc_* control globals, one of
# which turns off correctly rounded sqrtf. This tree mandates a correctly
# rounded square root at the ACCD sites, so that variant is a non-penetration
# question rather than a speed one. The same applies to the unsafe-atomics flag.
# check-rocm-fp-flags.py resolves these flags THROUGH this recipe rather than
# grepping for the spelling.
HIPCC_FLAGS = -std=c++17 -O3 -MMD -MP $(PIC_FLAG) $(PLATFORM_FLAGS)

KERNELGEN := ../seam/kernelgen.py
KERNELGEN_DIR := $(OUT_DIR)/kernelgen
KERNEL_SRCS := $(shell cd $(KERNEL_ROOT) && find . -name '*.kernel.cpp' | sed 's|^\./||' | sort)
KERNEL_ENTRY_HIP := $(patsubst %.kernel.cpp,$(KERNELGEN_DIR)/%.entry.hip,$(KERNEL_SRCS))
KERNEL_ARGS_HIPH := $(patsubst %.kernel.cpp,$(KERNELGEN_DIR)/%.args.hiph,$(KERNEL_SRCS))
KERNEL_BODY_HIP  := $(patsubst %.kernel.cpp,$(KERNELGEN_DIR)/%.kernel.hip,$(KERNEL_SRCS))

# `.SECONDARY:` because the args and body renderings are produced only as
# prerequisites of an entry object. Without it make DELETES them as intermediate
# files, so the build passes once and fails on every later one.
.SECONDARY: $(KERNEL_ARGS_HIPH) $(KERNEL_BODY_HIP) $(KERNEL_ENTRY_HIP) $(KERNEL_TABLE_INC)

$(KERNELGEN_DIR)/%.entry.hip: $(KERNEL_ROOT)/%.kernel.cpp $(KERNELGEN)
	@mkdir -p $(dir $@)
	python3 $(KERNELGEN) --target hip --emit entry --out $@ $<

$(KERNELGEN_DIR)/%.args.hiph: $(KERNEL_ROOT)/%.kernel.cpp $(KERNELGEN)
	@mkdir -p $(dir $@)
	python3 $(KERNELGEN) --target hip --emit args --out $@ $<

$(KERNELGEN_DIR)/%.kernel.hip: $(KERNEL_ROOT)/%.kernel.cpp $(KERNELGEN)
	@mkdir -p $(dir $@)
	python3 $(KERNELGEN) --target hip --emit body --out $@ $<

# THE KERNEL TABLE IS ONE FRAGMENT PER NEUTRAL SOURCE, CONCATENATED. A kernel
# id is a dense index over every entry the library carries, so no single neutral
# source can know one; the generator emits this file's rows and the order comes
# from the sorted walk of the kernel tree, which is what every generated table is
# built from.
KERNEL_TABLE_INC := $(patsubst %.kernel.cpp,$(KERNELGEN_DIR)/%.table.inc,$(KERNEL_SRCS))
KERNEL_TABLE := $(KERNELGEN_DIR)/kernel_table.inc
# The diagnostic file table: one row per neutral source, mapping the id a
# `DIAG_ASSERT` records to the path it names. A device record cannot carry a
# string, so the site records `file_id(__FILE__)`; this is what reverses it.
DIAG_FILE_INC := $(patsubst %.kernel.cpp,$(KERNELGEN_DIR)/%.diagfile.inc,$(KERNEL_SRCS))
DIAG_FILES := $(KERNELGEN_DIR)/diag_files.inc

$(KERNELGEN_DIR)/%.table.inc: $(KERNEL_ROOT)/%.kernel.cpp $(KERNELGEN)
	@mkdir -p $(dir $@)
	# --kernel-root because a fragment spells the argument header its rows are
	# written against from the generated root, so the concatenation below can be
	# a plain cat rather than an include list.
	python3 -B $(KERNELGEN) --target hip --emit table --kernel-root $(KERNEL_ROOT) --out $@ $<

$(KERNELGEN_DIR)/%.diagfile.inc: $(KERNEL_ROOT)/%.kernel.cpp $(KERNELGEN)
	@mkdir -p $(dir $@)
	# EVERY source gets a row, including one that declares no entry point: an
	# assert does not need an entry to exist.
	python3 -B $(KERNELGEN) --target hip --emit diagfile --kernel-root $(KERNEL_ROOT) --out $@ $<

$(KERNEL_TABLE): $(KERNEL_TABLE_INC)
	@mkdir -p $(dir $@)
	cat $^ > $@

# Order does not matter here, unlike the kernel table: a row carries its own id
# rather than taking one from its position.
$(DIAG_FILES): $(DIAG_FILE_INC)
	@mkdir -p $(dir $@)
	cat $^ > $@

ENTRY_OBJS := $(patsubst $(KERNELGEN_DIR)/%.entry.hip,$(OUT_DIR)/hip-entry$(OBJ_SUFFIX)/%.o,$(KERNEL_ENTRY_HIP))

$(OUT_DIR)/hip-entry$(OBJ_SUFFIX)/%.o: $(KERNELGEN_DIR)/%.entry.hip $(KERNELGEN_DIR)/%.args.hiph \
                          $(KERNELGEN_DIR)/%.kernel.hip Makefile $(ROCM_ARCH_FILE)
	@mkdir -p $(dir $@)
	$(HIPCC) $(HIPCC_FLAGS) -I $(KERNEL_ROOT) -I . -I $(dir $<) \
	    $(ENTRY_CONTEXT_FLAGS) -c $< -o $@

# EVERY RENDERING HAS TO EXIST BEFORE ANY hipcc RUNS, and the pattern rule above
# says so only for an entry's OWN three renderings. A rendered entry includes
# the rendered BODIES of other kernels (`../primitives/vec_ops.kernel.hip`), and
# on a 32 vCPU builder make reached that compile before the render had run:
# run 35175476327 failed with `fatal error: '../primitives/vec_ops.kernel.hip'
# file not found` inside `make: *** Waiting for unfinished jobs`, a race that
# four cores on the GitHub-hosted runner never lost. The CUDA Makefile carries
# this same order-only edge for this same reason. Order-only (the `|`), so a
# re-render does not rebuild every object; the -MMD dependency files carry the
# real mtime relation after the first build.
$(ENTRY_OBJS): | $(KERNEL_BODY_HIP) $(KERNEL_ENTRY_HIP) $(KERNEL_ARGS_HIPH)

MECHANISM_SRCS = backend/backend.hip arena/arena.hip diagnostics/diagnostics.hip \
                 utility/dispatcher.hip
MECHANISM_OBJS = $(patsubst %.hip,$(OUT_DIR)/mech$(OBJ_SUFFIX)/%.o,$(MECHANISM_SRCS))

# THE CONTEXT HEADERS ARE NEEDED HERE TOO, and for the same reason they are
# needed by an entry object: backend.hip includes the concatenated kernel table,
# whose rows include the argument headers, which include the neutral bodies, and
# a neutral body carries no include of its own. Without this the first body in
# the table fails on its own type names (`unknown type name 'AABB'`) and the
# diagnostic points at a kernel nobody touched.
$(OUT_DIR)/mech$(OBJ_SUFFIX)/%.o: %.hip Makefile $(ROCM_ARCH_FILE) $(KERNEL_TABLE)
	@mkdir -p $(dir $@)
	$(HIPCC) $(HIPCC_FLAGS) -I $(KERNEL_ROOT) -I . -I $(KERNELGEN_DIR) \
	    $(ENTRY_CONTEXT_FLAGS) -c $< -o $@
# The mechanism objects include the table, whose rows include the argument
# headers, which include the bodies: the same edge, for the same race.
$(MECHANISM_OBJS): | $(KERNEL_BODY_HIP) $(KERNEL_ENTRY_HIP) $(KERNEL_ARGS_HIPH) $(DIAG_FILES)

.PHONY: all abi entries clean
all: abi

# Renders and compiles every generated entry without linking the library, which
# is what a host with a ROCm toolchain and no AMD GPU can check.
entries: $(ENTRY_OBJS)
	@echo "compiled $(words $(ENTRY_OBJS)) generated entry objects for: $(ROCM_TARGETS)"

abi: $(LIBDIR)/libppfbe_rocm.so

# Named for the platform, so a switch leaves this file ABSENT and the rule below
# fires. See the note beside OBJ_SUFFIX for what that is protecting against.
PLATFORM_ID = $(if $(filter nvidia,$(HIP_PLATFORM)),nvidia,amd)
PLATFORM_STAMP = $(OUT_DIR)/platform.$(PLATFORM_ID)

$(PLATFORM_STAMP):
	@mkdir -p $(dir $@)
	@rm -f $(OUT_DIR)/platform.*
	@touch $@
	@echo "platform: $(PLATFORM_ID)"

# A LINK FLAG CHANGED IN THIS FILE HAS TO RELINK, and no object changes when only
# a flag does, so the link depends on the recipe itself. Without it an existing
# library keeps the flags it was linked with and the change reaches nothing.
THIS_MAKEFILE := $(lastword $(MAKEFILE_LIST))

$(LIBDIR)/libppfbe_rocm.so: $(MECHANISM_OBJS) $(ENTRY_OBJS) $(ROCM_ARCH_FILE) \
                            $(PLATFORM_STAMP) $(THIS_MAKEFILE)
	@mkdir -p $(LIBDIR)
	$(HIPCC) -shared $(PLATFORM_LINK_FLAGS) -o $@ \
	    $(filter-out $(ROCM_ARCH_FILE) $(PLATFORM_STAMP) $(THIS_MAKEFILE),$^)

clean:
	rm -rf $(OUT_DIR)

-include $(shell find $(OUT_DIR) -name '*.d' 2>/dev/null)
