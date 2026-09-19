# File: Makefile
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

# THE TOOLKIT ROOT, AND THE ONE COMPILER PATH EVERY CONSUMER ASKS FOR.
#
# `PPF_CUDA_ROOT` names the toolkit this recipe compiles with. It defaults to
# `/usr/local/cuda`, the root every provisioned host uses. A toolkit assembled
# somewhere else without root, which `build-linux-native/warmup.sh` does and
# which cannot be put behind that symlink, is selected by exporting it. The name
# carries the project prefix because the environment is shared: `CUDA_HOME` is
# exported by several toolkit installers, and honoring it would change which
# toolkit builds on a host that happens to set it.
#
# `crates/ppf-cts-solver/build.rs` checks the release of THIS path, and asks the
# recipe for it through `print-nvcc` below rather than deriving it a second
# time, so the toolkit that is checked and the toolkit that compiles cannot
# differ.
PPF_CUDA_ROOT ?= /usr/local/cuda
NVCC := $(PPF_CUDA_ROOT)/bin/nvcc

OUT_DIR=build
LIBDIR=$(OUT_DIR)/lib

# THE DEFAULT GOAL IS NAMED, NOT INHERITED FROM RULE ORDER.
#
# Without this line make takes its default goal from the first explicit target
# in the file, which is a position rather than a decision: adding a rule above
# `all` silently changes what a bare `make` builds. That is not hypothetical
# here. `crates/ppf-cts-solver/build.rs` invokes this recipe with NO goal on the
# ordinary CUDA path (it names goals only when the ABI library is also wanted),
# so the day the kernel-table concatenation landed above `all`, a from-scratch
# build made one .inc file, exited 0, and produced no library at all. The
# failure surfaced two steps later as build.rs's own guard reporting that the
# recipe reported success and the image does not exist, and it was invisible to
# every warm-cache build, which is every build that had already made the image
# once.
.DEFAULT_GOAL := all

# The compiler this recipe uses, printed for the solver's build script, which
# checks its release before any compile starts. See `PPF_CUDA_ROOT` above.
.PHONY: print-nvcc
print-nvcc:
	@echo '$(NVCC)'

# WHERE THE NEUTRAL KERNEL TREE IS, SUPPLIED BY THE CALLER.
#
# This recipe belongs to ppf-cts-compute and the sources it compiles belong to
# ppf-cts-solver, so the path between them is an input rather than a constant.
# Hardcoding `../../ppf-cts-solver/src/kernels` would make this crate
# unpublishable on its own, which is the one test it is held to. `build::host`
# already takes its kernel root the same way.
#
# The default names the tree in this workspace so `make` in this directory
# works. The solver's build script passes it, and OUT_DIR, relative to this
# directory, for the reason it records beside those arguments: nvcc writes the
# path it is given into every assert's `__FILE__`, so an absolute one would name
# the build machine inside the library.
KERNEL_ROOT ?= ../../ppf-cts-solver/src/kernels
# Refuse before nvcc runs. Every include below resolves through this path, so a
# wrong one surfaces as dozens of missing headers naming files that exist.
ifeq ($(wildcard $(KERNEL_ROOT)/data.hpp),)
$(error KERNEL_ROOT=$(KERNEL_ROOT) does not hold data.hpp: pass the neutral kernel tree of ppf-cts-solver)
endif

# THERE IS NO BACKEND LOGIC TREE ON THE CUDA SIDE, WHICH IS WHY NO ROOT IS NAMED.
#
# This crate is a general compute API: allocation, free, copy and transfer, and
# kernel launch, plus the platform machinery those five need. A Newton loop, a
# PCG recurrence, a contact narrow phase and a preconditioner hierarchy are none
# of the five. The CUDA orchestrator that held them was this backend's logic
# tree until the neutral Rust driver took over driving it, and it is deleted, so
# what this recipe builds is this directory's own mechanism, the C boundary the
# driver calls, and one object per generated entry point: no simulation at all.
#
# Metal still has an orchestrator and still passes a root, but to its OWN recipe
# (../metal/Makefile), which is where that machinery lives now.

# THE DECLARATION HEADERS A STANDALONE ENTRY TRANSLATION UNIT NEEDS, SUPPLIED BY
# THE CALLER FOR THE SAME REASON THE TWO ROOTS ABOVE ARE.
#
# A neutral kernel body carries NO INCLUDE OF ITS OWN, on purpose and by its own
# statement: the matrix and vector names, and the linear-algebra entry points it
# calls, come from whatever declares them for the compiler that is reading it,
# which on the shader side is a prologue rather than a header. Every existing
# consumer of a body is a translation unit that included those declarations
# first, so the property has never been visible. An ENTRY rendering compiled on
# its own has no such caller, and this is the list that stands in for one.
#
# A missing entry here is LOUD: nvcc names the undeclared symbol and the neutral
# file and line it is used at. That is why a list is acceptable at all, and it
# is named from the caller's tree rather than fixed here so this recipe stays
# publishable on its own.
ENTRY_CONTEXT_HEADERS ?= data.hpp linalg/eigsolve.hpp
ENTRY_CONTEXT_FLAGS = $(foreach header,$(ENTRY_CONTEXT_HEADERS),-include $(header))

# CUDA architectures, read from the manifest beside this file rather than
# spelled out here. cuda_arch.txt records why it is the single source and which
# consumers derive from it. It is parsed with awk off a ';'-commented,
# space-separated file so build.bat can read the same file with cmd.exe's
# `for /f` and the two builds cannot come to ship different architectures.
CUDA_ARCH_FILE := cuda_arch.txt
CUDA_ARCH_FLOOR := $(strip $(shell awk '$$1=="floor"{print $$2; exit}' $(CUDA_ARCH_FILE) 2>/dev/null))
CUDA_ARCH_CUBINS := $(strip $(shell awk '$$1=="cubin"{print $$2}' $(CUDA_ARCH_FILE) 2>/dev/null))
# A missing or unparsable manifest stops the build here. Continuing on an empty
# list would link a .so carrying no device code, which surfaces much later as
# every GPU being rejected at run time, and that failure names the device rather
# than the file that actually went missing.
ifeq ($(CUDA_ARCH_FLOOR),)
$(error no 'floor' line in $(CURDIR)/$(CUDA_ARCH_FILE))
endif
ifeq ($(CUDA_ARCH_CUBINS),)
$(error no 'cubin' lines in $(CURDIR)/$(CUDA_ARCH_FILE))
endif
# One -gencode per cubin, every one lowered from the single floor IR.
CUDA_GENCODE := $(foreach arch,$(CUDA_ARCH_CUBINS),-gencode arch=compute_$(CUDA_ARCH_FLOOR),code=sm_$(arch))
# Rendered "sm_86, sm_89, ..." for the text main.cu prints when it rejects a
# device, so that message cannot name a different set from the one linked.
CUDA_ARCH_LIST_STR := $(shell awk '$$1=="cubin"{printf "%ssm_%s", (n++?", ":""), $$2}' $(CUDA_ARCH_FILE))

# Warning suppressions
SUPPRESS_WARNINGS=1222 2527 2529 2651 2653 2668 2669 2670 2671 2735 2737 2739 20012 20011 20014 177 940 1394

# nvcc flags for CUDA files. The device solver has no external linear-algebra
# dependency: the in-house crates/.../kernels/linalg headers replaced Eigen, so
# there is no -I<eigen> or -DEIGEN_WARNINGS_DISABLED here (Eigen now lives only
# in a host-only compile, which carries its own include).
NVCC_FLAGS=-std=c++17 -O3 -MMD -MP --device-c -Xcompiler -fPIC -D__NVCC__ --expt-relaxed-constexpr --extended-lambda -Wno-deprecated-gpu-targets
# Compiling a NEUTRAL kernel directly is a build error, by design.
#
# A `*.kernel.cpp` carries `[[seam::device_fn]]` and `[[seam::thread]]`, and
# kernelgen.py removes both when it renders. So the attributes reach a compiler
# in exactly one situation: a neutral source was fed to it instead of its
# rendering. nvcc reports that as warning 2803, and a warning is not enough,
# for two reasons that are the same defect at different distances.
#
# The near one: an include path that lists this tree before the generated root
# resolves `#include "solver/spmv.kernel.cu"` to the neutral file, which then
# compiles, because an unrecognized attribute is only ignored.
#
# The far one: nvcc treats `[[seam::device_fn]]` as an ignored attribute and the
# function becomes __host__. The DEFINITION still compiles. Only a device call
# site fails, so a body that nothing calls from device code is silently absent
# from the device image, and the first evidence is a link error or a missing
# kernel much later.
#
# --diag-error promotes 2803 for device compilation, and -Werror=attributes the
# host half, so both spellings of the mistake stop the build where it happens.
NVCC_FLAGS += --diag-error=2803 -Xcompiler -Werror=attributes
NVCC_FLAGS += $(foreach warn,$(SUPPRESS_WARNINGS),--diag-suppress=$(warn))

# Device link-time optimization (LTO). Compile each TU to an LTO intermediate
# (code=lto_<floor>); the device link below (-dlto) runs whole-program optimization,
# which is what lets a device function defined in one translation unit inline
# into a kernel in another. Without it such a call is a 255-register ABI call,
# and the caller measured here roughly halved the cost of its hottest kernel by
# gaining that one inline.
#
# The LTO IR sits at the floor virtual arch (compute_61 as shipped), and every
# device target in the link below emits native SASS from that IR.
# IMPORTANT: device LTO CANNOT embed a JIT-able PTX. The -dlto link lowers its IR
# straight to SASS; a code=compute_XX request is silently dropped (verified with
# cuobjdump: an LTO link yields cubins only, zero PTX). The .so is therefore
# frozen to the exact SASS images we ship, there is no forward-JIT fallback, so
# a cubin per supported arch is enumerated at the link. Adding a GPU generation
# is one line in cuda_arch.txt plus a rebuild. nvcc rejects -dlto beside
# -gencode at compile, hence code=lto_<floor> here and the full -gencode list
# only at the link.
NVCC_FLAGS += -gencode arch=compute_$(CUDA_ARCH_FLOOR),code=lto_$(CUDA_ARCH_FLOOR)
# The architectures this image ships, as text, for the run-time rejection
# message. Single-quoted so the embedded ", " survives the shell and reaches
# nvcc as one argument holding a C string literal.
NVCC_FLAGS += -DSHIPPED_ARCH_STR='"$(CUDA_ARCH_LIST_STR)"'
# nvcc flags for C++ files (compiled as host code)
NVCC_CPP_FLAGS=-std=c++17 -O3 -MMD -MP -Xcompiler -fPIC,-Wall,-Wno-unused-function,-Werror=attributes

# ---------------------------------------------------------------------------
# Neutral kernel sources.
#
# A kernel body lives once, as a `*.kernel.cpp` file under $(KERNEL_ROOT):
# plain C++, no preprocessor conditional, no macro of its own, and no spelling
# that belongs to one backend. ../seam/kernelgen.py renders one into the form
# each compiler reads, and this rule produces the nvcc form. The Metal and host
# forms come from the same script, driven by their own recipes.
#
# The output goes under $(OUT_DIR), which cargo sets to the crate's build
# directory, so nothing is written under the neutral tree. That matters more
# than it looks: two build scripts watch a source tree recursively, and cargo
# reads a directory in rerun-if-changed as "rerun if any descendant changes",
# so an artifact anywhere inside costs a full rebuild of ppf-cts-solver AND
# ppf-cts-server on every subsequent build.
#
# The generated tree MIRRORS the neutral one, so a kernel that includes another
# kernel keeps its relative spelling and it resolves. A translation unit names
# a rendered kernel from the generated root, which -I below puts on the search
# path: contact/contact.cu says #include "contact/grain_pair.kernel.cu".
#
# The find needs no prune: $(OUT_DIR) is the caller's build directory and is
# never inside $(KERNEL_ROOT), so the host rendering of a kernel, itself named
# `*.kernel.cpp`, is never offered back as a source. kernelgen.py refuses one
# anyway, by its banner, and that is the loud half of the same guard.
KERNELGEN := ../seam/kernelgen.py
KERNELGEN_DIR := $(OUT_DIR)/kernelgen
KERNEL_SRCS := $(sort $(shell find $(KERNEL_ROOT) -name '*.kernel.cpp' -print \
                                | sed 's|^$(KERNEL_ROOT)/||'))
KERNEL_CU := $(patsubst %.kernel.cpp,$(KERNELGEN_DIR)/%.kernel.cu,$(KERNEL_SRCS))

# The ENTRY rendering, one file per neutral kernel beside its body rendering.
# An entry point is an argument record plus a thread index plus a call into the
# body, all three mechanical, so it is generated rather than written: a
# hand-written entry point is one more mirror pair per kernel with nothing
# linking its halves. A kernel declaring no [[seam::args]] entry still renders a
# file, which is what lets this be a blanket rule instead of a hand-kept list.
KERNEL_ENTRY_CU := $(patsubst %.kernel.cpp,$(KERNELGEN_DIR)/%.entry.cu,$(KERNEL_SRCS))

# The ARGUMENT-RECORD half of that rendering, split out because a record is a
# type and a caller that fills one needs it, while the `__global__` and the
# launcher beside it are definitions that only one translation unit may hold.
# Every hand-written CUDA launch still in the neutral kernel tree sits in a
# header two or three translation units include, so without this split those
# launches cannot become generated entry points at all. The .entry.cu includes
# its .args.cuh, so the record is defined once whichever file a reader opens.
KERNEL_ARGS_CUH := $(patsubst %.kernel.cpp,$(KERNELGEN_DIR)/%.args.cuh,$(KERNEL_SRCS))

# THE KERNEL TABLE, in two halves rendered from one walk of one sorted list.
#
# A kernel id is a dense index over every entry point the ABI library carries,
# so it is a property of the CONCATENATION and not of any source. Both halves
# are concatenated here, from $(KERNEL_SRCS), which is already sorted: the .inc
# is what nvcc compiles into the library's table and the .rs is what the driver
# builds its `KernelDecl` array from. One list and one order is a stronger
# guarantee than two programs independently sorting the same names, and it is
# free, because this recipe already has to walk the tree.
#
# `be_open` still compares the two id by id, by name and by record size.
# That check is not redundant with the shared order: it catches a DRIVER and a
# LIBRARY built from different trees, which is a different mistake from the two
# halves of one build disagreeing.
KERNEL_TABLE_INC := $(patsubst %.kernel.cpp,$(KERNELGEN_DIR)/%.table.inc,$(KERNEL_SRCS))
KERNEL_TABLE_RS := $(patsubst %.kernel.cpp,$(KERNELGEN_DIR)/%.table.rs,$(KERNEL_SRCS))
KERNEL_TABLE := $(KERNELGEN_DIR)/kernel_table.inc
# The diagnostic file table: one row per neutral source, mapping the id a
# `DIAG_ASSERT` records to the path it names. A device record cannot carry a
# string, so the site records `file_id(__FILE__)`; this is what reverses it.
DIAG_FILE_INC := $(patsubst %.kernel.cpp,$(KERNELGEN_DIR)/%.diagfile.inc,$(KERNEL_SRCS))
DIAG_FILES := $(KERNELGEN_DIR)/diag_files.inc
KERNEL_TABLE_RUST := $(KERNELGEN_DIR)/kernel_table.rs

# Three roots, in this order.
#
#   $(KERNELGEN_DIR)  the rendered kernels, which a translation unit names as
#                     `contact/grain_pair.kernel.cu`. First, so a rendering is
#                     never shadowed.
#   .                 this directory, for the three CUDA-only headers that are
#                     mechanism (cuda_vec.hpp, cuda_utils.hpp, seam_cuda.cuh).
#                     A neutral header names one by BARE FILENAME, so the
#                     neutral tree carries no path into this crate and the
#                     backend build is what supplies the spelling; a logic
#                     translation unit names them the same way, for the same
#                     reason applied to the other crate boundary.
#   $(KERNEL_ROOT)    the neutral tree, which a translation unit names from its
#                     root: `contact/contact.hpp`, `data.hpp`. A `..`-relative
#                     spelling would hardcode the distance between two crates
#                     into every file that crosses it.
NVCC_FLAGS += -I$(KERNELGEN_DIR) -I. -I$(KERNEL_ROOT)

$(KERNELGEN_DIR)/%.kernel.cu: $(KERNEL_ROOT)/%.kernel.cpp $(KERNELGEN)
	mkdir -p $(@D)
	python3 -B $(KERNELGEN) --target cu --out $@ $<

$(KERNELGEN_DIR)/%.entry.cu: $(KERNEL_ROOT)/%.kernel.cpp $(KERNELGEN)
	mkdir -p $(@D)
	# --kernel-root is passed rather than left to the generator's fallback:
	# a generated ENTRY point includes headers from the neutral tree, and the
	# transcompiler lives in this crate, so it cannot deduce where that tree
	# is from its own location.
	python3 -B $(KERNELGEN) --target cu --emit entry --kernel-root $(KERNEL_ROOT) --out $@ $<

$(KERNELGEN_DIR)/%.args.cuh: $(KERNEL_ROOT)/%.kernel.cpp $(KERNELGEN)
	mkdir -p $(@D)
	# --kernel-root for the same reason as the entry rule above: the record
	# names ArenaHandle, whose header is in the neutral tree.
	python3 -B $(KERNELGEN) --target cu --emit args --kernel-root $(KERNEL_ROOT) --out $@ $<

$(KERNELGEN_DIR)/%.table.inc: $(KERNEL_ROOT)/%.kernel.cpp $(KERNELGEN)
	mkdir -p $(@D)
	# --kernel-root because a fragment spells the argument header its rows are
	# written against from the generated root, so that the concatenation below
	# can be a plain cat rather than an include list.
	python3 -B $(KERNELGEN) --target cu --emit table --kernel-root $(KERNEL_ROOT) --out $@ $<

$(KERNELGEN_DIR)/%.diagfile.inc: $(KERNEL_ROOT)/%.kernel.cpp $(KERNELGEN)
	mkdir -p $(@D)
	# EVERY source gets a row, including one that declares no entry point: an
	# assert does not need an entry to exist.
	python3 -B $(KERNELGEN) --target cu --emit diagfile --kernel-root $(KERNEL_ROOT) --out $@ $<

$(KERNELGEN_DIR)/%.table.rs: $(KERNEL_ROOT)/%.kernel.cpp $(KERNELGEN)
	mkdir -p $(@D)
	python3 -B $(KERNELGEN) --target rust --emit table --kernel-root $(KERNEL_ROOT) --out $@ $<

# The concatenations. Order is $(KERNEL_SRCS)'s, which is sorted, and the two
# recipes read the same variable, so the id a row gets is the same number on
# both sides of the boundary by construction.
$(KERNEL_TABLE): $(KERNEL_TABLE_INC)
	mkdir -p $(@D)
	cat $^ > $@

# Order does not matter here, unlike the kernel table: a row carries its own
# id rather than taking one from its position.
$(DIAG_FILES): $(DIAG_FILE_INC)
	mkdir -p $(@D)
	cat $^ > $@

# The Rust half is BRACKETED, and that is a requirement of the language rather
# than a decoration: `include!` in expression position parses its file as ONE
# expression, and a comma-separated list of rows is not one. Wrapping the
# concatenation makes it an array literal, which is, so the driver's side reads
# `&include!(..)` and the rows stay exactly what the generator emitted.
$(KERNEL_TABLE_RUST): $(KERNEL_TABLE_RS)
	mkdir -p $(@D)
	{ echo '['; cat $^; echo ']'; } > $@

# Source files, in two sets that are compiled identically and come from two
# different crates.
#
# MECHANISM_DIRS are this crate's own: the arena that hands out device memory
# and the transport half of the diagnostic channel. utility/dispatcher.cu is
# deliberately absent, and is not an omission: dispatcher.hpp ends with
# `#include "dispatcher.cu"`, so it is an include rather than a translation
# unit, and compiling it standalone only produces a dead object.
#
# LOGIC_DIRS AND LOGIC_KERNEL_FILES ARE EMPTY, AND THAT IS THE POINT. They named
# the CUDA orchestrator's translation units, every one of which carried
# simulation, which is why none of them was ever a MECHANISM_DIR. That tree is
# deleted, so a CUDA build compiles no simulation at all. They stay DECLARED
# because check-shared-wiring.py enumerates the CUDA translation units by
# reading these three variables and refuses to report a pass without them.
MECHANISM_DIRS=arena diagnostics
LOGIC_DIRS=
LOGIC_KERNEL_FILES=
MECHANISM_FILES=$(foreach dir,$(MECHANISM_DIRS),$(dir)/$(dir).cu)
# The ABI translation unit: the C boundary the neutral Rust driver calls. It is
# mechanism by the same test as the arena and the diagnostic channel, and it
# names no kernel: its table is generated and its argument records cross as
# opaque bytes.
ABI_FILES=backend/backend.cu
LOGIC_FILES=$(foreach dir,$(LOGIC_DIRS),$(dir)/$(dir).cu) $(LOGIC_KERNEL_FILES)
# Two host C++ translation units, compiled from the neutral tree: the logging
# bridge every backend links and the print_rust stub. Named relative to
# $(KERNEL_ROOT), which the pattern rule below prepends, so the object paths
# stay inside $(OUT_DIR) whatever the caller's kernel root is.
SIMPLELOG_FILES=simplelog/SimpleLog.cpp
STUB_FILES=stub.cpp

# Object files. The two source sets get two object trees, so a path under
# $(OUT_DIR) says which crate its source came from and the two pattern rules
# below cannot both match one object.
MECHANISM_OBJS=$(addprefix $(OUT_DIR)/cuda/,$(MECHANISM_FILES:%.cu=%.o))
ABI_OBJS=$(addprefix $(OUT_DIR)/cuda/,$(ABI_FILES:%.cu=%.o))
# EVERY GENERATED ENTRY POINT IS A TRANSLATION UNIT OF THE ABI LIBRARY. An
# .entry.cu defines a __global__ and a C-linkage launcher, so exactly one
# translation unit may include it; compiling each as its own object is the
# spelling of that rule which needs no hand-kept list, and it is what puts an
# address in the table for every row.
#
# THIS IS WHY THE ABI LIBRARY WAS SEPARATE from the orchestrator's image while
# both existed. The C++ orchestrator reached one of these entry points by
# INCLUDING it into a logic translation unit, so linking both forms into one
# image would have been a duplicate definition; and that image would have grown
# every entry point nothing in it calls, which moves the SASS
# count a release build is compared against for a reason unrelated to the
# solver.
#
# The list is DERIVED rather than kept, by the same predicate the transcompiler
# and the host recipe both use: a declaration opens a line. A neutral kernel
# with no entry declaration renders a banner and nothing else, so compiling it
# would cost an nvcc invocation per source for an object with no symbols, and
# most of the sources are in that state. How many is not written here: it moves
# with every conversion, and a number in a comment nobody re-measures is read as
# current long after it stops being so. `grep -rl` with the pattern below
# answers it.
# `[[seam::entry]]` MAY NAME THE COUNT AND THE INDEX, the way
# `DISPATCH_START(count)` names the count in the reference, so the argument
# form is part of the spelling this reads. Without it the set is EMPTY and
# nvcc compiles no entry at all, which surfaces as the shared library
# referencing every `<name>_entry_launch` symbol with nothing defining them.
# THE LIST COMES FROM THE RENDERER WHEN IT IS THERE. `build.rs` writes one
# line per neutral source it actually rendered an entry for, which is an
# answer rather than a recognition, so a spelling change cannot leave this
# set SHORT. The grep below is the fallback for a hand-run `make`, and it
# is the reading that has gone blind twice.
ENTRY_SOURCE_LIST := $(OUT_DIR)/entry-sources.txt
ifneq ($(wildcard $(ENTRY_SOURCE_LIST)),)
KERNEL_ENTRY_SRCS := $(sort $(shell cat $(ENTRY_SOURCE_LIST)))
else
KERNEL_ENTRY_SRCS := $(sort $(shell grep -l '^\[\[seam::\(args\|entry\)\((.*)\)\?\]\]' \
                        $(addprefix $(KERNEL_ROOT)/,$(KERNEL_SRCS)) 2>/dev/null \
                      | sed 's|^$(KERNEL_ROOT)/||'))
endif

# AN EMPTY SET IS THE SILENT FAILURE, and this recipe is where it would be
# silent. `ENTRY_OBJS` would be empty, the link carries no `-z defs`, and
# libppfbe_cuda.so would build and link clean with not one generated entry
# point in it; the first evidence would be a missing kernel at run time. The
# Metal recipe guards its own check the same way.
ifeq ($(strip $(KERNEL_ENTRY_SRCS)),)
$(error the discovery above found no neutral kernel source declaring \
[[seam::args]] or [[seam::entry]], among $(words $(KERNEL_SRCS)) sources. \
The set is read off the sources by a grep for the attribute spelling, so a \
spelling change with this recipe not updated empties it, and an empty set \
compiles and links cleanly with no generated entry point in the library)
endif

ENTRY_OBJS=$(patsubst %.kernel.cpp,$(OUT_DIR)/cuda-entry/%.o,$(KERNEL_ENTRY_SRCS))
LOGIC_OBJS=$(addprefix $(OUT_DIR)/cuda-logic/,$(LOGIC_FILES:%.cu=%.o))
CUDA_OBJS=$(MECHANISM_OBJS) $(LOGIC_OBJS)
SIMPLELOG_OBJS=$(addprefix $(OUT_DIR)/cpp/,$(SIMPLELOG_FILES:%.cpp=%.o))
STUB_OBJS=$(addprefix $(OUT_DIR)/cpp/,$(STUB_FILES:%.cpp=%.o))
DEPS = $(CUDA_OBJS:.o=.d) $(SIMPLELOG_OBJS:.o=.d) $(STUB_OBJS:.o=.d) \
       $(ABI_OBJS:.o=.d) $(ENTRY_OBJS:.o=.d)

# `abi` IS THE WHOLE PRODUCT, so `all` is an alias for it. There is no second
# library to build.
all: abi

# Objects depend on the Makefile so a flags change forces a recompile (nvcc's
# mtime view only tracks sources; without this a flag-only edit relinks stale .o).
# They depend on the architecture manifest for the same reason and it is not the
# same file: editing it changes both the LTO floor these are compiled to and the
# SHIPPED_ARCH_STR baked into main.cu, neither of which make can see through
# the flags alone. Without this an arch edit leaves every .o stale, and the
# result is caught only later, by arch_guard, as a build failure telling the
# reader to delete the build directory.
$(OUT_DIR)/cuda/%.o: %.cu Makefile $(CUDA_ARCH_FILE)
	mkdir -p $(@D)
	$(NVCC) -c $(NVCC_FLAGS) $< -o $@

# The rendered kernels have to EXIST before nvcc runs, and after that the
# -MMD dependency files carry the real mtime relation: a rendered kernel a
# translation unit includes appears in its .d, so editing the neutral source
# re-renders it and recompiles exactly the objects that read it. Order-only
# (the `|`) is what keeps a re-render from rebuilding every object regardless.
# kernelgen.py writes only when the text changes, so a no-op re-render does not
# move the mtime either.
$(CUDA_OBJS): | $(KERNEL_CU) $(KERNEL_ENTRY_CU) $(KERNEL_ARGS_CUH) $(DIAG_FILES)

# One object per generated entry rendering. The source is under $(OUT_DIR), not
# under either crate's tree, so this rule cannot be confused with the two above
# and no artifact lands where a build script watches.
#
# TWO FLAGS THIS RULE ADDS, both because an entry rendering is being compiled
# STANDALONE here and was written to be included into a translation unit that
# already has a context.
#
#   -include <each of $(ENTRY_CONTEXT_HEADERS)>
#                       The declarations a kernel body is written against and
#                       does not include: the seam names, the linalg
#                       types (`Vec3f`, `Mat3x3f`, `Mat9x9f`) and the eigen
#                       solvers it calls. Supplied by the recipe rather than
#                       added to the renderings, which is what keeps every
#                       rendering byte-identical to the one the other consumer
#                       reads.
#   -Wno-pragma-once-outside-header
#                       An entry rendering carries `#pragma once` because it may
#                       be included; as a main file that is a warning about a
#                       directive doing exactly what it was put there for.
$(OUT_DIR)/cuda-entry/%.o: $(KERNELGEN_DIR)/%.entry.cu Makefile $(CUDA_ARCH_FILE)
	mkdir -p $(@D)
	$(NVCC) -c $(NVCC_FLAGS) $(ENTRY_CONTEXT_FLAGS) \
	        -Xcompiler -Wno-pragma-once-outside-header $< -o $@

# The ABI translation unit reads the concatenated table, which pulls every
# argument record, which pulls every kernel body, so it needs the same
# declarations a standalone entry translation unit does. A rule of its own
# rather than widening the mechanism rule above: the arena and the diagnostic
# channel include nothing from the caller's tree and must keep not needing to.
$(OUT_DIR)/cuda/backend/%.o: backend/%.cu Makefile $(CUDA_ARCH_FILE)
	mkdir -p $(@D)
	$(NVCC) -c $(NVCC_FLAGS) $(ENTRY_CONTEXT_FLAGS) $< -o $@

# The entry objects need every rendering present before nvcc runs, and the ABI
# translation unit needs the concatenated table as well, since it includes it
# twice.
$(ENTRY_OBJS): | $(KERNEL_CU) $(KERNEL_ENTRY_CU) $(KERNEL_ARGS_CUH)
$(ABI_OBJS): | $(KERNEL_CU) $(KERNEL_ENTRY_CU) $(KERNEL_ARGS_CUH)
$(ABI_OBJS): $(KERNEL_TABLE)

# Compile C++ files with nvcc (uses host compiler internally)
$(OUT_DIR)/cpp/%.o: $(KERNEL_ROOT)/%.cpp Makefile
	mkdir -p $(@D)
	$(NVCC) -c $(NVCC_CPP_FLAGS) $< -o $@

# Link CUDA library: -dlto runs whole-program optimization on the lto_<floor> objects
# and emits one native SASS cubin per architecture (no PTX, see the note above).
# SASS is binary-compatible forward within a major version, so the sm_86 cubin
# also runs sm_87 (Jetson Orin) and sm_89 (Ada: RTX 40, L40S); each new MAJOR
# needs its own cubin. A brand-new arch not listed here fails to launch ("named
# symbol not found") until its cubin is added and the lib rebuilt.
#   sm_61  Pascal     GTX 10, Titan Xp       (oldest shipped; issue #141. Covers
#                                             sm_61 and higher minors only, so
#                                             sm_60 (Tesla P100) is NOT included.
#                                             nvcc 12.8 warns that targets below
#                                             75 are deprecated and will be
#                                             dropped in a future CUDA release,
#                                             which is when this row has to go.)
#   sm_75  Turing     RTX 20, GTX 16, T4     (requested, and the self-hosted CI
#                                             runner's card)
#   sm_86  Ampere     RTX 30, A10/A40/A6000  (+ sm_87 Orin, sm_89 Ada via compat)
#   sm_89  Ada        RTX 40, L40S, L4       (native; perf on the dev/CI GPU)
#   sm_90  Hopper     H100, H200, GH200
#   sm_100 Blackwell  B200, GB200            (datacenter)
#   sm_120 Blackwell  RTX 50-series          (consumer)
# Keep this list in sync with SUPPORTED_SM in crates/ppf-cts-core/src/utils.rs,
# which gates solver launch against exactly these architectures.
#
# This link, not the 17 compiles, is what a rebuild costs. Each -gencode is an
# independent whole-program optimization of the same LTO IR down to that arch's
# SASS, and nvcc runs them one after another unless told otherwise, so the link
# costs the arch count times a single arch. Measured on an L40S (16 cores, nvcc
# 12.8): 178.9 s for the five against 33.6 s for one alone, while all 17 compiles
# together take ~6 s wall at -j16.
#
# -t overlaps those per-arch optimizations (0 = one thread per CPU), taking the
# link to 43.9 s, near the floor set by the slowest single arch. It is a
# scheduling flag only: it selects no optimization level and changes no codegen
# decision, and the .so it produces is BYTE-IDENTICAL to the serial one (md5
# verified for -t 0, -t 5 and -t 16 against an unflagged link).
#
# Do NOT reach for --split-compile here. It parallelizes too, and slightly
# faster (39.9 s), but by partitioning the optimizer's scope, so it is not free:
# it emitted 1,150,536 SASS instructions against 1,132,816 for the serial link,
# 17,720 more (+1.6%) in the solver's hottest device code. Build time is not
# worth paying for at run time. --split-compile-extended is worse still: it
# fails this link outright ("nvlink error: Size doesn't match for '__unnamed_1'
# in 'lto.cubin'").
#
# -t belongs on this link ONLY, never on the compile rule above. That rule
# builds a single arch, where -t has nothing to overlap and measured no
# gain (solver.cu: 5.3 s with -t 16 vs 5.4 s without), and make already runs the
# compiles -j wide, so each of the 17 would stack its own thread pool on top.
# The manifest is a prerequisite because the -gencode list is derived from it:
# an arch added or removed there changes this link and nothing else, so without
# it make would consider an image built for the previous list up to date. $^
# would then also feed the manifest to nvcc as an input file, hence $(filter-out).
# THE ABI LIBRARY: the CUDA target as the C boundary the neutral Rust driver
# calls. Arena, diagnostics and the boundary itself, plus one object per
# generated entry point, and NOT one object of the C++ orchestrator: this image
# carries no Newton loop, no PCG recurrence and no contact narrow phase, which
# is what makes it a compute target rather than a second solver.
#
# SimpleLog and the print_rust stub are linked in rather than reached through
# libsimplelog.so, exactly as the orchestrator's image does it, so the two
# libraries are independent artifacts that can be loaded together or apart.
$(LIBDIR)/libppfbe_cuda.so: $(MECHANISM_OBJS) $(ABI_OBJS) $(ENTRY_OBJS) \
                            $(SIMPLELOG_OBJS) $(STUB_OBJS) $(CUDA_ARCH_FILE)
	mkdir -p $(LIBDIR)
	# -Wno-deprecated-gpu-targets is on the compile flags too. It is needed HERE
	# as well because the -gencode list lands at the device link, and nvcc 12.8
	# warns once per deprecated target: the floor and the sm_61 cubin are both
	# below the 75 it now considers current. The Makefile arch table says when
	# that row has to come out; this only keeps the link log readable until then.
	$(NVCC) -shared -dlto -t 0 -Wno-deprecated-gpu-targets $(CUDA_GENCODE) -o $@ $(filter-out $(CUDA_ARCH_FILE),$^)

# Built on request rather than by `all`, because a build that is not routing
# the driver at CUDA pays a second five-architecture device link for an image
# it will not load. The Rust table is a prerequisite so that asking for the
# library also produces the driver's half of the same table.
.PHONY: abi
abi: $(LIBDIR)/libppfbe_cuda.so $(LIBDIR)/libsimplelog.so $(KERNEL_TABLE_RUST)

# SimpleLog library (for standalone use)
$(LIBDIR)/libsimplelog.so: $(SIMPLELOG_OBJS)
	mkdir -p $(LIBDIR)
	$(NVCC) -shared -o $@ $^

# -MP above makes nvcc emit a phony target for every header it names in the
# dependency file. Without it, a `.d` from an earlier build that names a
# header since RENAMED or DELETED stops the next build with
#   make: *** No rule to make target '<old header>', needed by '<object>'
# which reads as a missing source rather than as a stale artifact. That is
# not hypothetical here: the kernel bodies moved to `*.kernel.cpp` in one
# change, so every build directory predating it meets this once, and -MP
# cannot repair a `.d` already written without it. The one-time remedy is
# to delete the stale dependency files (`find <OUT_DIR> -name '*.d'
# -delete`); a build directory created after this change is immune.
# Verified accepted by nvcc 12.8, which emits the phony targets.
-include $(DEPS)

.PHONY: clean
clean:
	rm -rf $(OUT_DIR)
