# File: Makefile
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Builds libppfbe_metal.dylib, the Metal backend, as a DYNAMIC library.
# Dynamic matches the CUDA backend (../cuda/Makefile builds libsimbackend_cuda.so
# the same way), and here it is also required rather than preferred: the
# backend must resolve print_rust to the Rust host at load time, which only a
# dynamic library can do (see the link rule).
#
# Target is arm64 only. Intel Macs are out of scope for this port.

# The default goal, pinned rather than inherited. make takes the first EXPLICIT
# target in the file, and shader_segments.mk is included above `all:` because it
# needs OUT_DIR and KERNELGEN_DIR settled first. Without this line the first
# rule in that fragment becomes the default, so a bare `make` (which is what
# build.rs runs) would build one embedded `.inc` and report success, leaving the
# dylib from the previous build in place.
.DEFAULT_GOAL := all

OUT_DIR ?= build
LIBDIR  := $(OUT_DIR)/lib

# WHERE THE NEUTRAL KERNEL TREE IS, SUPPLIED BY THE CALLER.
#
# This recipe belongs to ppf-cts-compute and the sources it compiles belong to
# ppf-cts-solver, so the path between them is an input rather than a constant.
# Hardcoding `../../ppf-cts-solver/src/kernels` would make this crate
# unpublishable on its own, which is the one test it is held to: this crate must
# build as a general compute library that names no simulation. The CUDA recipe
# beside this one takes its kernel root the same way.
#
# The default names the tree in this workspace so `make` in this directory
# works, and the solver's build script passes an absolute path.
KERNEL_ROOT ?= ../../ppf-cts-solver/src/kernels
# Refuse before the compilers run. Every include and every embed rule below
# resolves through this path, so a wrong one surfaces as dozens of missing
# headers naming files that exist, or worse as an embed rule with no
# prerequisite, which make reports as a missing target rather than as a
# misconfigured root.
ifeq ($(wildcard $(KERNEL_ROOT)/data.hpp),)
$(error KERNEL_ROOT=$(KERNEL_ROOT) does not hold data.hpp: pass the neutral kernel tree of ppf-cts-solver)
endif

# WHERE THE BACKEND LOGIC TREE IS, SUPPLIED BY THE CALLER ON THE SAME TERMS.
#
# This recipe compiles two kinds of source, and the split is the architecture
# rather than a layout preference. What sits beside this Makefile is MECHANISM:
# the device, the arena, the command queue, the shader assembler, the pipeline
# cache and the diagnostic transport, which is the closed list of things a
# general compute API may hold. What sits under $(BACKEND_LOGIC_ROOT) is
# SIMULATION written against Metal: a Newton loop, the shader assembly, the
# parity self-tests and the hand-written MSL entry points. It cannot be here,
# and it cannot be in ppf-cts-solver either, which carries no backend name at
# all, so it lives in a crate whose name says it exists to be deleted.
#
# The path is an input for the same reason KERNEL_ROOT is: a recipe that
# hardcoded the distance to another crate could not be published on its own.
# EMPTY, because there is no backend logic tree on the Metal side any more.
# The C++ orchestrator that was one is deleted: the neutral Rust driver drives
# this backend, and what this recipe builds is this directory's own mechanism,
# the C boundary the driver calls, and one object per generated entry point.
BACKEND_LOGIC_ROOT ?=
ifneq ($(BACKEND_LOGIC_ROOT),)
ifeq ($(wildcard $(BACKEND_LOGIC_ROOT)/shader_segments.mk),)
$(error BACKEND_LOGIC_ROOT=$(BACKEND_LOGIC_ROOT) does not hold shader_segments.mk: pass the backend logic tree)
endif
endif

CXX ?= clang++

# -fobjc-arc: the backend's .mm files hold Objective-C object references in C++
#   containers and enforce ARC with a #error, so this flag is not optional.
# -fPIC: position-independent code for the dynamic library.
# No -DNDEBUG anywhere, matching the CUDA build: the allocator's alignment and
#   capacity assertions are live in release, which is the point of having them.
CXXFLAGS := -std=c++17 -O2 -fPIC -fobjc-arc -arch arm64 \
            -Wall -Wextra -Wno-unused-function

# The neutral kernel tree, reached by name rather than by a `../` spelling of
# the distance between two crates. Every `#include "data.hpp"` and
# `#include "contact/accd.hpp"` in the .mm and .hpp sources resolves through
# this, which is the same arrangement the CUDA recipe uses (`-I$(KERNEL_ROOT)`).
CXXFLAGS += -I$(KERNEL_ROOT)

# `-I.` so a source under $(BACKEND_LOGIC_ROOT) resolves this crate's headers
# (metal_context.hpp, arena.hpp, diagnostics.hpp, bringup.hpp) by name: a quoted
# include resolves relative to the including FILE first, which for those sources
# is the other tree.
#
# THERE IS DELIBERATELY NO `-I$(BACKEND_LOGIC_ROOT)`. The dependency points ONE
# WAY: the logic tree includes this crate's interfaces, and nothing here
# includes a header from there. Adding it would let a mechanism source reach a
# solver type by name, and the build would not object. Its absence is what makes
# that a compile error instead of a review item.
CXXFLAGS += -I.

# Objective-C++ sources, this crate's own: the device, memory, diagnostic,
# shader-assembly and pipeline-cache modules, plus bringup.mm, the one
# definition of the order those modules are started in. Every one of them is
# one of the five verbs or the platform machinery those five require.
#
# The sources carrying simulation are named by $(LOGIC_MM_SRCS), which
# shader_segments.mk defines under $(BACKEND_LOGIC_ROOT), and are compiled by
# their own rule below into their own object directory.
MM_SRCS := bringup.mm metal_context.mm arena.mm diagnostics.mm \
           shader_compiler.mm pipeline_cache.mm

# ---------------------------------------------------------------------------
# Neutral kernel sources.
#
# A kernel body lives once, as a `*.kernel.cpp` file under $(KERNEL_ROOT):
# plain C++, no preprocessor conditional, no macro of its own, and no spelling
# that belongs to one backend. ../seam/kernelgen.py renders one into the form
# each compiler reads, and this backend needs TWO of the three.
#
#   *.kernel.metal  the MSL form, with the address spaces present as keywords,
#                   embedded as shader text by embed_source.py below.
#   *.kernel.cpp    the host form, with the address spaces removed, included by
#                   a source in the backend logic tree so that a self-test's
#                   oracle is the same body the shader runs.
#
# The MSL form is line-for-line identical in count to the neutral source and
# carries no #line of its own, because shader_compiler.mm injects `#line 1
# "<path>"` as it splices the segment and maps a diagnostic back by counting
# lines from there. The host form carries a two-line preamble ending in a
# `#line` naming the neutral file, so a clang diagnostic points at the source
# rather than at the rendering.
#
# find prunes the generated tree for the same reason the CUDA recipe does: the
# host rendering carries the same `.kernel.cpp` name as its source.
KERNELGEN := ../seam/kernelgen.py
KERNELGEN_DIR := $(OUT_DIR)/kernelgen
KERNEL_SRCS := $(sort $(patsubst $(KERNEL_ROOT)/%,%,\
                 $(shell find $(KERNEL_ROOT) -path '$(KERNEL_ROOT)/build' -prune -o \
                                     -name '*.kernel.cpp' -print)))
KERNEL_METAL := $(patsubst %.kernel.cpp,$(KERNELGEN_DIR)/%.kernel.metal,\
                  $(KERNEL_SRCS))
KERNEL_HOST := $(patsubst %.kernel.cpp,$(KERNELGEN_DIR)/%.kernel.cpp,\
                 $(KERNEL_SRCS))

$(KERNELGEN_DIR)/%.kernel.metal: $(KERNEL_ROOT)/%.kernel.cpp $(KERNELGEN)
	mkdir -p $(@D)
	python3 -B $(KERNELGEN) --target metal --out $@ $<

$(KERNELGEN_DIR)/%.kernel.cpp: $(KERNEL_ROOT)/%.kernel.cpp $(KERNELGEN)
	mkdir -p $(@D)
	python3 -B $(KERNELGEN) --target cpp --out $@ $<

# ---------------------------------------------------------------------------
# THE GENERATED ENTRY POINTS, AND A COMPILER THAT READS THEM.
#
# One neutral declaration renders four ways, and the argument for generating
# them is that four declarations cannot disagree. That holds only where a
# compiler reads each one. nvcc reads `*.entry.cu` into the C ABI library and
# the host compiler reads `*.entry.cpp` into the CPU backend's shim; rustc read
# the `*.entry.rs` the driver DISPATCHES and nothing else, which was 22 of 30,
# and the MSL rendering was read by NOTHING at all. The cost of the last was
# measured rather than argued. The rendering of every
# GATHERED STRUCT ELEMENT did not compile at all, across 71 sites in 26
# declarations: `buffer[index]` is a `device` lvalue on Metal, a body parameter
# written `[[seam::thread]] const T &` renders as `thread const T &`, and MSL
# refuses to bind the one to the other. Nothing said so for as long as no
# shader compiler had read one.
#
# WHAT THIS COMPILES AND WHAT IT DOES NOT. Each entry rendering is compiled ON
# ITS OWN, against the backend's real MSL prologue and the real neutral body, by
# `xcrun metal -c`. It does NOT assemble a shader: the shipped libraries are
# spliced in an order that lives with those sources and is not reproduced here,
# and their `#line` and `DIAG_FILE_ID` directives belong to that assembly. So
# this answers "is the generated entry point valid MSL", and the fixture suite
# answers "does the assembled shader compile and dispatch". Neither substitutes
# for the other.
#
# WHY IT IS PART OF `all`. A check nobody invokes is not a check, and this one
# costs a few seconds beside a build that already renders every one of these
# files. Building the Metal backend is what a person does when they change a
# kernel, so that is where the answer belongs.
ENTRY_CHECK_DIR := $(OUT_DIR)/entry-check
MSL_PROLOGUE := $(ENTRY_CHECK_DIR)/msl_prologue.metal

# THE NEUTRAL HEADERS EVERY ENTRY TRANSLATION UNIT NEEDS, SUPPLIED BY THE
# CALLER.
#
# THIS LIST IS A SET, NOT AN ORDER, AND THAT IS THE PROPERTY TO PRESERVE.
#
# Each entry includes what it uses, unconditionally and with a quoted include,
# so naming one of them pulls in whatever declares the types it names:
# `data_records.hpp` reaches `linalg/type_aliases.hpp`, which reaches
# `linalg/smat.hpp`, which reaches `linalg/la_traits.hpp`.
# Permuting the three lines below therefore cannot change what compiles, and the
# alternative was measured rather than argued: as an order this list was already
# the wrong one, and the two orders that suggest themselves fail with `unknown
# type name '__host__'` and a redefinition of a scalar type, neither of which
# names the list. Both were properties of the headers, and both are fixed there.
#
# A quoted include costs the SHIPPED shader nothing, which is what makes this
# available: the run-time assembler neutralizes every quoted include as it
# splices (check-shared-wiring.py rule 3), so the shader keeps taking its
# declarations from its own segment order and only an offline unit follows the
# lines. An ANGLE include still cannot be made unconditional, and every one in
# these files stays behind `#ifndef SM_MSL_CONCAT`.
#
# WHY THERE ARE THREE ENTRIES RATHER THAN NONE. A neutral kernel body names its
# own dependencies and entry_harness.py walks them, so a body needs nothing from
# here. These three are what is reached WITHOUT being named: the material
# headers call `detsqr` without including it, the two spectral bodies call the
# eigensolvers the same way, and a body or an argument record may name any
# record `data_records.hpp` declares. Giving those call sites their includes
# would empty this list; until then a missing entry is LOUD, the shader compiler
# naming the undeclared symbol and the neutral file and line it is used at.
#
# The CUDA recipe carries an ENTRY_CONTEXT_HEADERS of its own for the same
# reason and reads `data.hpp` directly, which nvcc can compile and the shader
# compiler cannot: that header reaches `vec/vec.hpp`, whose raw `T *` has no
# address space. `data_records.hpp` is the half of it that MSL can read.
ENTRY_CONTEXT_HEADERS ?= data_records.hpp linalg/eigsolve.hpp \
                         energy/model/detsqr.hpp

# WHICH KERNELS DECLARE AN ENTRY, read off the sources rather than listed. A
# declaration opens at column zero with one of the two attributes that mark it,
# so a mention in prose (always inside a `//`) is not one. The CUDA recipe
# discovers the same set the same way.
# `[[seam::entry]]` MAY NAME THE COUNT PARAMETER, the way
# `DISPATCH_START(count)` carries it in the reference, so the argument form
# `[[seam::entry(count)]]` is part of the spelling this reads. The guard
# below is what turns a missed spelling into a stopped build rather than an
# empty set that compiles and links with no entry point in the library.
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
                       $(addprefix $(KERNEL_ROOT)/,$(KERNEL_SRCS)) 2>/dev/null | \
                       sed 's|^$(KERNEL_ROOT)/||'))
endif

# The entry-check recipe already refuses an empty set, and this refuses it at
# parse time as well, so a spelling change with this recipe not updated stops
# the build rather than only the check.
ifeq ($(strip $(KERNEL_ENTRY_SRCS)),)
$(error the discovery above found no neutral kernel source declaring \
[[seam::args]] or [[seam::entry]], among $(words $(KERNEL_SRCS)) sources. \
The set is read off the sources by a grep for the attribute spelling, so a \
spelling change with this recipe not updated empties it, and an empty set \
compiles and links cleanly with no generated entry point in the library)
endif

KERNEL_ENTRY_METAL := $(patsubst %.kernel.cpp,$(KERNELGEN_DIR)/%.entry.metal,\
                        $(KERNEL_ENTRY_SRCS))
ENTRY_HARNESS := $(patsubst %.kernel.cpp,$(ENTRY_CHECK_DIR)/%.harness.metal,\
                   $(KERNEL_ENTRY_SRCS))
ENTRY_AIR := $(patsubst %.kernel.cpp,$(ENTRY_CHECK_DIR)/%.air,\
               $(KERNEL_ENTRY_SRCS))

# The HOST half of an entry's argument record. The shader reads the record and
# the ObjC++ that fills it writes the record, and those are different
# translation units, so the record needs a rendering the host can include. It
# carries no launcher and names no CUDA type, which is why the CUDA `.args.cuh`
# cannot stand in for it.
KERNEL_ARGS_H := $(patsubst %.kernel.cpp,$(KERNELGEN_DIR)/%.args.h,\
                   $(KERNEL_ENTRY_SRCS))

$(KERNELGEN_DIR)/%.args.h: $(KERNEL_ROOT)/%.kernel.cpp $(KERNELGEN)
	mkdir -p $(@D)
	python3 -B $(KERNELGEN) --target metal --emit args \
		--kernel-root $(KERNEL_ROOT) --out $@ $<

# THE GENERATED KERNEL TABLE, one fragment per neutral source concatenated in
# SORTED SOURCE ORDER, because a kernel id is a row's POSITION in the result.
# `$(sort ...)` is what fixes that order, and `KERNEL_ENTRY_SRCS` is already
# sorted for the same reason on the CUDA side; the driver builds its half from
# the `rust` fragments the same way, and `be_open`'s caller compares the two id
# by id.
KERNEL_TABLE_FRAGS := $(patsubst %.kernel.cpp,$(KERNELGEN_DIR)/%.table.h,\
                        $(KERNEL_ENTRY_SRCS))
KERNEL_TABLE_METAL := $(KERNELGEN_DIR)/kernel_table_metal.h

$(KERNELGEN_DIR)/%.table.h: $(KERNEL_ROOT)/%.kernel.cpp $(KERNELGEN)
	mkdir -p $(@D)
	python3 -B $(KERNELGEN) --target metal --emit table \
		--kernel-root $(KERNEL_ROOT) --out $@ $<

# A PLAIN `cat`, which is what the fragment's spelling is arranged to allow: a
# metal row is pure data and includes nothing, so the concatenation needs no
# order beyond the sorted one and no second pass.
# THE DRIVER'S HALF OF THE SAME TABLE, rendered here too. Its content is
# target-independent, so this is not a second statement of anything: what makes
# it the Metal recipe's business is that a Metal build's OUT_DIR is where the
# driver and its tests look for the file, and only the recipe that ran knows
# that directory.
#
# The wrapper turns a comma-separated list of rows into an array literal,
# because `include!` needs one expression and a list of rows is not one. The
# driver reads `&include!(..)` and the rows stay exactly what the generator
# emitted.
KERNEL_TABLE_RS := $(patsubst %.kernel.cpp,$(KERNELGEN_DIR)/%.table.rs,\
                     $(KERNEL_ENTRY_SRCS))
KERNEL_TABLE_RUST := $(KERNELGEN_DIR)/kernel_table.rs

$(KERNELGEN_DIR)/%.table.rs: $(KERNEL_ROOT)/%.kernel.cpp $(KERNELGEN)
	mkdir -p $(@D)
	python3 -B $(KERNELGEN) --target rust --emit table \
		--kernel-root $(KERNEL_ROOT) --out $@ $<

$(KERNEL_TABLE_RUST): $(KERNEL_TABLE_RS)
	mkdir -p $(@D)
	{ echo '['; cat $^; echo ']'; } > $@

$(KERNEL_TABLE_METAL): $(KERNEL_TABLE_FRAGS)
	mkdir -p $(@D)
	cat $(KERNEL_TABLE_FRAGS) > $@

$(KERNELGEN_DIR)/%.entry.metal: $(KERNEL_ROOT)/%.kernel.cpp $(KERNELGEN)
	mkdir -p $(@D)
	# --kernel-root because the generator no longer sits inside the neutral
	# tree and an entry rendering resolves headers against it.
	python3 -B $(KERNELGEN) --target metal --emit entry \
		--kernel-root $(KERNEL_ROOT) --out $@ $<

# The prologue is READ out of the two files that define it, never copied. A
# check that compiled an entry point against its own spelling of ARENA_PTR
# would pass while the real one drifted, which is the failure this whole
# arrangement exists to prevent.
$(MSL_PROLOGUE): msl_prologue.py shader_compiler.mm diagnostics.mm arena.hpp
	mkdir -p $(@D)
	python3 -B msl_prologue.py --shader-compiler shader_compiler.mm \
		--diagnostics diagnostics.mm --arena arena.hpp --out $@

$(ENTRY_CHECK_DIR)/%.harness.metal: $(KERNEL_ROOT)/%.kernel.cpp \
                                    entry_harness.py Makefile
	mkdir -p $(@D)
	python3 -B entry_harness.py --kernel-root $(KERNEL_ROOT) --stem $* \
		--prologue $(MSL_PROLOGUE) --context "$(ENTRY_CONTEXT_HEADERS)" \
		--out $@

# -I in three places and each is one root: the harness's own directory for the
# prologue, the neutral tree for the shared headers, and the generated tree for
# the body and entry renderings. They cannot collide, because a rendering never
# carries a `.hpp` name.
#
# The renderings are ORDER-ONLY prerequisites as a set, and the two this unit
# names are real ones: `make -j` may render any file at any time, and the
# harness names its dependencies by path rather than by mtime.
#
# -MMD -MP FOR THE SAME REASON THE HOST OBJECTS BELOW CARRY THEM, and this is
# the compile that SHIPS (see `entries-lib`). The prerequisites above name only
# this unit's own renderings, while a body reaches its material through shared
# headers (`energy/model/*.hpp`) and other renderings by `#include`. Without
# the dependency file an edit to one of those leaves this `.air` newer than
# everything make can see, so the linked metallib keeps the OLD arithmetic and
# the backend runs it without a word, while the CPU and CUDA backends, whose
# compiles do track headers, run the new one.
$(ENTRY_CHECK_DIR)/%.air: $(ENTRY_CHECK_DIR)/%.harness.metal $(MSL_PROLOGUE) \
                          $(KERNELGEN_DIR)/%.entry.metal \
                          $(KERNELGEN_DIR)/%.kernel.metal | $(KERNEL_METAL)
	mkdir -p $(@D)
	xcrun metal $(METAL_ENTRY_FLAGS) -I$(ENTRY_CHECK_DIR) -I$(KERNEL_ROOT) \
		-I$(KERNELGEN_DIR) -MMD -MP -c $< -o $@

-include $(ENTRY_AIR:.air=.d)

# ONE LIBRARY OUT OF EVERY GENERATED ENTRY POINT, WHICH IS WHAT A DISPATCHING
# BACKEND NEEDS AND WHAT `entry-check` DELIBERATELY DOES NOT PRODUCE.
#
# `entry-check` answers "is each generated entry point valid MSL" by compiling
# each one ALONE. That is a per-file answer, and a backend that dispatches by
# kernel id needs a per-LIBRARY one: every entry present, in one library, with
# no symbol collision between translation units that each pulled in the same
# neutral bodies.
#
# THE OFFLINE PATH SIDESTEPS SEGMENT ORDER ENTIRELY, and that is the point. The
# shipped shader is assembled at RUN time by splicing segments in a hand-kept
# order, because `newLibraryWithSource` is handed one string with no filesystem
# behind it. `xcrun metal` is clang and HAS a filesystem, so each entry compiles
# as its own translation unit with live includes and `xcrun metallib` links the
# AIR objects, exactly as the CUDA recipe links one object per entry into
# libppfbe_cuda.so. No order is reproduced here because none is needed.
#
# WHAT IT DOES NOT ANSWER. It says nothing about the shader the SHIPPED Metal
# backend assembles today, which is a different translation unit built a
# different way, and that one is only ever read by a RUN. This target is the precondition for retiring that
# assembly, not a check on it.
ENTRY_METALLIB := $(OUT_DIR)/lib/ppf_entries.metallib

$(ENTRY_METALLIB): $(ENTRY_AIR)
	mkdir -p $(@D)
	xcrun metallib $^ -o $@

# THE SYMBOL COUNT IS PRINTED AND REQUIRED, because a link that produced a
# library exporting NOTHING would exit 0 and read as a pass. That is the same
# trap the FP64 guard names on the CUDA side: report the count first, and treat
# a zero as the extraction having failed rather than as a clean result. The
# count is the number of `kernel void` functions across every rendering, so it
# is what a dispatching backend can create a pipeline for.
# AND A PROGRAM THAT PROVES THE LIBRARY IS USABLE, not merely well formed.
# Loading it says the container parsed; creating a pipeline for every entry is
# the AIR-to-ISA step, which is the larger of Metal's two startup costs and the
# one that can still reject a function.
ENTRY_PROBE := $(OUT_DIR)/entry-check/entries_probe

$(ENTRY_PROBE): entries_probe.mm
	mkdir -p $(@D)
	$(CXX) -std=c++17 -fobjc-arc -framework Metal -framework Foundation \
		-o $@ $<

.PHONY: entries-lib
# NO HARD PREREQUISITE ON THE AIR OBJECTS, and that is the whole shape of this
# recipe. A Mac can build AND RUN this backend with no Metal shader compiler
# installed, because the framework compiles the shader at run time from source;
# making the toolchain a build requirement would break those machines for a
# check. So the guard is the same one `entry-check` uses, tested by RUNNING the
# compiler rather than by locating it, and the account it leaves is the one
# cargo re-emits: `build.rs` captures make's output and prints it only on
# FAILURE, so a skipped check is otherwise silent on exactly the builds it was
# written for.
entries-lib:
	@set -e; \
	mkdir -p $(OUT_DIR); \
	if ! xcrun metallib --version >/dev/null 2>&1; then \
		{ \
		echo "NOT CHECKED: the generated entry points were not linked into a"; \
		echo "  library, because this machine has no Metal shader compiler."; \
		echo "  Test it by RUNNING it: xcrun metallib --version"; \
		echo "  Install it with: xcodebuild -downloadComponent MetalToolchain"; \
		echo "  Nothing else answers whether a dispatching backend could create"; \
		echo "  a pipeline for every entry out of one library."; \
		} | tee $(BUILD_CHECK_STATUS); \
		exit 0; \
	fi; \
	$(MAKE) $(ENTRY_METALLIB) $(ENTRY_PROBE); \
	if [ ! -s $(ENTRY_METALLIB) ]; then \
		echo "ERROR: $(ENTRY_METALLIB) is empty or missing, so the link"; \
		echo "       reported success over nothing."; \
		exit 1; \
	fi; \
	n=$$(xcrun metal-nm $(ENTRY_METALLIB) 2>/dev/null \
	     | grep -c " T .*_entry$$" || true); \
	if [ "$$n" -eq 0 ]; then \
		echo "ERROR: $(ENTRY_METALLIB) exports no entry point. The link"; \
		echo "       succeeded over nothing, which is the failure this"; \
		echo "       count exists to catch."; \
		exit 1; \
	fi; \
	echo "linked $(words $(ENTRY_AIR)) generated entry translation units into"; \
	echo "  $(ENTRY_METALLIB) ($$(wc -c < $(ENTRY_METALLIB)) bytes,"; \
	echo "  $$n entry points a dispatching backend can bind)"; \
	$(ENTRY_PROBE) $(ENTRY_METALLIB)

# make treats a file produced only by a pattern rule on the way to another as
# INTERMEDIATE and deletes it when the chain finishes, which here would throw
# away every harness and every rendering the next build would otherwise reuse,
# and would delete the very text a failing compile names. .SECONDARY with no
# prerequisites turns that off for every target in this file.
.SECONDARY:

# -fmetal-math-mode=safe IS PASSED, AND IT IS A CORRECTNESS SETTING RATHER THAN
# A FLAG SET TO MATCH. Fast math DELETES Kahan
# compensation outright, zero threads keep it, and it damages the
# eigenvalue-floored inverse on the large majority of threads, the worst
# returning a value nine orders out.
#
# THE COMMENT HERE USED TO SAY THIS COMPILE "NEVER SHIPS", AND THAT WAS WRONG.
# The `.air` this recipe produces is `ENTRY_AIR` (see the definition above),
# `$(ENTRY_METALLIB): $(ENTRY_AIR)` links exactly those objects with `xcrun
# metallib`, and `be_open` loads the result by path. So this IS the shipped
# library's compile, and its arithmetic mode is baked in HERE: the run-time
# `mathMode` Safe that `metal_context.mm` sets applies to a library compiled
# from SOURCE at run time and cannot reach a `.metallib` that was already
# linked.
#
# It also used to say the offline default "is Safe". That may be true of the
# toolchain in front of you and it is not something to depend on: a default is a
# property of the machine, and `build-mac-native/build.sh` refuses outright to
# compile an artifact whose flags do not contain this exact spelling. The two
# shipped paths are held to one standard now.
#
# -ferror-limit=0 REMOVES THE TRUNCATION, and it is the diagnostic half of the
# same argument. clang stops after 20 errors and says "too many errors emitted,
# stopping now", which is a sensible default for a compile whose output someone
# is waiting on and the wrong one for a compile whose ENTIRE product is the
# error list. One broken declaration in a shared header spends that budget on
# the cascade inside the first class that reads it, so the twenty lines printed
# can all belong to one root cause while the units that would name a second one
# are never reached. Nothing is emitted from these translation units, so the
# only cost of an unlimited list is log length, and the only cost of the limit
# is the answer.
METAL_ENTRY_FLAGS ?= -Wall -ferror-limit=0 -fmetal-math-mode=safe

# Set to 1 to make a missing Metal toolchain a FAILURE rather than a loud
# warning. The build leaves it at 0 because a Mac can build and RUN this backend
# with no toolchain at all (the shader is compiled at run time by the Metal
# framework, not by `xcrun metal`), so requiring it would refuse a build that
# works. CI sets it to 1, where a silent skip would be the whole leg going
# green over nothing.
ENTRY_CHECK_REQUIRED ?= 0

# A SKIP HAS TO SURVIVE CARGO, WHICH IS WHY IT IS WRITTEN TO A FILE.
#
# `crates/ppf-cts-solver/build.rs` captures this recipe's output and prints it
# only when make FAILS, so a banner echoed here on a machine with no shader
# compiler would be invisible on exactly the builds it is written for. The
# recipe therefore leaves its account in this file, which that build script
# reads and re-emits as a cargo warning, and DELETES when the check ran. The
# name says what it is rather than which check wrote it: a recipe that could not
# run something the caller should know about leaves it here.
BUILD_CHECK_STATUS := $(OUT_DIR)/build-check-status.txt

# EVERY INPUT IS BUILT BY THIS MAKE, AND ONLY THE COMPILES BY THE SUB-MAKE.
#
# `all` names this target, so under `make -j` it runs beside the dylib rules,
# and a recursive make that also built the renderings could be making the same
# file as its parent at the same moment. Naming them here means the outer make
# has already made them, alone, before the recipe starts; the `$(MAKE)` below
# then has nothing left to make but the `.air` files, which no other rule wants.
#
# -k ON THE SUB-MAKE, BECAUSE THIS CHECK IS READ RATHER THAN CONSUMED.
#
# A make that stops at the first failing `.air` reports one translation unit and
# says nothing about the other renderings, and it does not even report a fixed
# one: build.rs runs this recipe under `-j<threads>`, so which unit loses the
# race decides which errors a reader is shown, and two machines compiling the
# same defect print disjoint lists that read as different bugs. Every `.air` is
# independent of every other, so there is no unit whose failure makes a later
# one meaningless, and -k costs a few seconds of compiling on a build that is
# failing anyway. The exit status is unaffected: -k finishes what it can and
# still exits nonzero, which the `set -e` above turns into a failing recipe.
.PHONY: entry-check
entry-check: $(KERNEL_METAL) $(KERNEL_ENTRY_METAL) $(KERNEL_ARGS_H) $(MSL_PROLOGUE) \
             $(ENTRY_HARNESS)
	@set -e; \
	if [ -z "$(KERNEL_ENTRY_SRCS)" ]; then \
		echo "ERROR: no neutral kernel source declares [[seam::args]] or"; \
		echo "       [[seam::entry]], so this check would compile nothing"; \
		echo "       and report success."; \
		exit 1; \
	fi; \
	mkdir -p $(OUT_DIR); \
	if ! xcrun metal --version >/dev/null 2>&1; then \
		{ \
		echo "NOT CHECKED: $(words $(KERNEL_ENTRY_SRCS)) generated MSL entry points were"; \
		echo "  not compiled, because this machine has no Metal shader compiler."; \
		echo "  'xcrun --find metal' resolves whether or not the component is"; \
		echo "  installed, so test it by RUNNING it: xcrun metal --version"; \
		echo "  Install it with: xcodebuild -downloadComponent MetalToolchain"; \
		echo "  Nothing else compiles these renderings, so a defect in one stays"; \
		echo "  invisible until a shader compiler reads it."; \
		} | tee $(BUILD_CHECK_STATUS); \
		if [ "$(ENTRY_CHECK_REQUIRED)" = "1" ]; then exit 1; fi; \
		exit 0; \
	fi; \
	$(MAKE) -k $(ENTRY_AIR); \
	rm -f $(BUILD_CHECK_STATUS); \
	echo "$(words $(ENTRY_AIR)) of $(words $(ENTRY_AIR)) generated MSL entry renderings compiled"

# ---------------------------------------------------------------------------
# The shader text, one `.inc` per segment, and the sources that splice them.
#
# Each `.inc` holds one C string literal, written by embed_source.py and spliced
# into the shader by a shader_add_segment call. Two kinds of input feed them:
#
#   $(KERNEL_ROOT)/<path>.hpp               an ordinary shared header, embedded
#                                           verbatim.
#   $(KERNELGEN_DIR)/<path>.kernel.metal    the MSL rendering of a neutral
#                                           kernel, never the neutral source:
#                                           the shader compiler needs the
#                                           address spaces spelled as keywords,
#                                           and the neutral source carries them
#                                           as [[seam::...]] attributes instead.
#
# WHICH SEGMENTS THERE ARE IS NOT THIS FILE'S BUSINESS. The list names solver
# headers one by one and pairs each with the C identifier a shader_add_segment
# call in face_math.mm expects, so a reader of it learns what the solver
# computes. It travels with those sources instead, and is included from the
# caller-supplied tree.
#
# It is included HERE rather than at the top because it uses OUT_DIR,
# KERNEL_ROOT and KERNELGEN_DIR, all of which are settled above. Make's working
# directory stays this one, so the `embed_source.py` its rules name is the one
# beside this Makefile.
LOGIC_OBJ_DIR := $(OUT_DIR)/metal-logic
ifneq ($(BACKEND_LOGIC_ROOT),)
include $(BACKEND_LOGIC_ROOT)/shader_segments.mk
ifndef LOGIC_MM_SRCS
$(error $(BACKEND_LOGIC_ROOT)/shader_segments.mk defined no LOGIC_MM_SRCS: the backend logic tree would contribute no source and the dylib would link without a Newton loop)
endif
endif

# SimpleLog is the plain-C++ logger the solver writes its per-step *.out data
# files through, shared verbatim with the CUDA backend. It is compiled INTO this
# dylib rather than linked as a separate library, so the Metal backend is one
# artifact and build.rs has one thing to link.
CPP_SRCS := $(KERNEL_ROOT)/simplelog/SimpleLog.cpp

LOGIC_OBJS := $(addprefix $(LOGIC_OBJ_DIR)/,$(LOGIC_MM_SRCS:.mm=.o))
OBJS := $(addprefix $(OUT_DIR)/,$(MM_SRCS:.mm=.o)) $(LOGIC_OBJS) \
        $(addprefix $(OUT_DIR)/,$(notdir $(CPP_SRCS:.cpp=.o)))
DEPS := $(OBJS:.o=.d)

# -undefined dynamic_lookup leaves print_rust to be resolved against the Rust
# host at load time. This is measured, not preferred: over six link
# configurations the determinant of whether a backend log line reaches the host
# is whether the
# DYLIB DEFINES THE SYMBOL ITSELF. With a stub in the dylib the line is
# swallowed under the default two-level namespace, under -flat_namespace and
# under -export_dynamic -flat_namespace alike; without it the call reaches the
# host under all three, with the host and backend addresses identical. So this
# backend ships no print_rust stub (unlike $(KERNEL_ROOT)/stub.cpp, which the
# CUDA recipe compiles) and
# the link below is checked for exactly that.
#
# The flag is blanket, so any OTHER undefined symbol would also become a
# load-time failure instead of a link-time one. The check after the link is what
# keeps that from going unnoticed for the one symbol it matters for.
#
# The other half of the seam is on the Rust side and is equally required:
# build.rs links the solver binary with -Wl,-exported_symbol,_print_rust,
# because the release profile strips symbols and an unbound dynamic_lookup
# reference resolves to address 0 rather than failing at load.
#
# libc++ is not named here because clang++ is the link driver and supplies it;
# naming it as well makes the linker report a duplicate library. A link driven
# by cc or ld directly would need -lc++.
# -install_name @rpath/... rather than letting the linker stamp whatever path
# -o was given. Without it the dylib records an absolute build-tree path (or a
# RELATIVE one for an in-tree `make`, which only loads when cwd happens to be
# this directory), so the binary is not relocatable and dyld ignores a
# co-located copy. It also makes the -Wl,-rpath build.rs emits actually do
# something: an absolute install name is never resolved through an rpath.

# ---------------------------------------------------------------------------
# The offline shader dump tool.
#
# `ppf-metal-shader-dump` brings the backend's shader libraries up with no
# scene and writes, per library, the assembled translation unit and the exact
# `xcrun metal` arguments this backend's compile options amount to. That is what
# build-mac-native/build.sh compiles into the `.metallib` artifacts a run then
# loads instead of compiling its shader.
#
# It links the DYLIB, not the objects, and both halves of that matter. The text
# it dumps is what the shipped backend assembles rather than what a second build
# of the same sources would; and the pre-compiled lookup resolves its directory
# through dladdr, so the tool looks where libppfbe_metal.dylib is, which is
# where the solver will look and where the build installs.
#
# It lands in $(OUT_DIR), never in the source tree: cargo passes its own
# OUT_DIR under target/, and two build scripts watch this directory
# recursively, so an artifact inside it costs a measured 48 s rebuild of two
# crates on every later build.
DUMP_BIN := $(OUT_DIR)/bin/ppf-metal-shader-dump

# `entries-lib` IS PART OF `all` FOR THE REASON `entry-check` IS: a check nobody
# invokes is not a check, and this one is nearly free here. entry-check has
# already compiled every entry to AIR by the time it runs, so what `all` adds is
# one `metallib` link and one probe run over the result. What it buys is a
# standing answer to the question a dispatching backend asks: can a pipeline be
# created for every generated entry, out of one library, with no run-time source
# assembly. It is answered on every Metal build rather than the day someone
# needs it.
# `abi` IS THE WHOLE PRODUCT, so `all` names it.
all: abi entry-check entries-lib

# @loader_path/../lib rather than an -L path: the tool is relocatable with the
# pair of directories it sits in, and the dylib's install name is @rpath-based
# precisely so a co-located copy is what loads.
DUMP_LDFLAGS := -arch arm64 -Wl,-rpath,@loader_path/../lib \
                -framework Metal -framework Foundation

# NOT part of the default build, and it does not link today. shader_dump.mm calls
# metal_backend::backend_bringup, which lives in bringup.mm and is DELIBERATELY not
# in the shipped libppfbe_metal.dylib (the dylib is the runtime ABI entry points only).
# The old tool linked the old orchestrator dylib that carried bringup; restoring the
# shader pre-compile means linking the bring-up + shader-assembly objects here instead.
# Until then build-mac-native/build.sh skips the pre-compile and the backend compiles
# its shader at run time (about 6.5 s, OS-cached), which costs a slower first start only.
$(DUMP_BIN): shader_dump.mm $(LIBDIR)/libppfbe_metal.dylib Makefile
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -I$(OUT_DIR) shader_dump.mm -o $@ \
		-L$(LIBDIR) -lppfbe_metal $(DUMP_LDFLAGS)
	@# The other end of the link contract checked below on the dylib. The
	@# dylib leaves _print_rust undefined so it binds to whatever loaded it;
	@# an executable that does not DEFINE it resolves that reference to
	@# address 0 and dies at the backend's first log line.
	@if ! nm -gU "$@" | grep -q " _print_rust$$"; then \
		echo "ERROR: $@ does not define _print_rust, so the backend's"; \
		echo "       logging seam would resolve to address 0 at run time."; \
		rm -f "$@"; exit 1; \
	fi

# Objects depend on the Makefile so a flags change forces a recompile; the
# compiler's mtime view only tracks sources.
$(OUT_DIR)/%.o: %.mm Makefile
	mkdir -p $(@D)
	$(CXX) -c $(CXXFLAGS) -I$(OUT_DIR) -MMD -MP $< -o $@

# The backend logic tree's sources, into their own object directory so the two
# rules cannot both match one target. They compile with the same flags: a
# source that is not mechanism is still the same language against the same
# runtime, and nothing about the split is a compilation difference.
$(LOGIC_OBJ_DIR)/%.o: $(BACKEND_LOGIC_ROOT)/%.mm Makefile $(BACKEND_LOGIC_ROOT)/shader_segments.mk
	mkdir -p $(@D)
	$(CXX) -c $(CXXFLAGS) -I$(OUT_DIR) -MMD -MP $< -o $@


# The host renderings have to EXIST before clang runs; after that the -MMD
# dependency file carries the real mtime relation, so editing a neutral kernel
# source re-renders it and recompiles exactly the objects that include it.
# kernelgen.py writes only when the text changes, so a no-op re-render moves no
# mtime.
# The argument records join them for the same reason: the host fills a record,
# so it needs the record's type, and the rendering that carries it is generated.
$(OBJS): | $(KERNEL_HOST) $(KERNEL_ARGS_H)

$(OUT_DIR)/SimpleLog.o: $(KERNEL_ROOT)/simplelog/SimpleLog.cpp Makefile
	mkdir -p $(@D)
	$(CXX) -c $(CXXFLAGS) -MMD -MP $< -o $@

# ===========================================================================
# THE ABI LIBRARY: the Metal target as the C boundary the neutral driver calls.
#
# Arena, context, diagnostics and the boundary itself, and NOT one object of the
# holding pen's orchestrator: this image carries no Newton loop, no PCG
# recurrence and no contact narrow phase, which is what makes it a compute
# target rather than a second solver. It assembles no shader either: the kernels
# reach it as `ppf_entries.metallib`, which `entries-lib` links from the
# generated entry renderings, and `be_open` loads by path.
#
# Built on request rather than by `all`, on the same terms as the CUDA one: a
# build that is not routing the driver at Metal pays neither the link nor the
# table generation for an image it will not load.
ABI_MM_SRCS := backend/backend.mm
ABI_OBJS := $(addprefix $(OUT_DIR)/,$(ABI_MM_SRCS:.mm=.o))
# The mechanism half of MM_SRCS. bringup.mm is deliberately absent: it starts
# whatever modules the LINKED BACKEND lists, which is the orchestrator's
# sequence, and this library has no modules of its own to start.
ABI_MECHANISM_OBJS := $(addprefix $(OUT_DIR)/,\
                        metal_context.o arena.o diagnostics.o \
                        pipeline_cache.o shader_compiler.o)

$(OUT_DIR)/backend/backend.o: backend/backend.mm $(KERNEL_TABLE_METAL) Makefile
	mkdir -p $(@D)
	$(CXX) -c $(CXXFLAGS) -I$(KERNELGEN_DIR) -MMD -MP $< -o $@

$(LIBDIR)/libppfbe_metal.dylib: $(ABI_OBJS) $(ABI_MECHANISM_OBJS) \
                                $(addprefix $(OUT_DIR)/,$(notdir $(CPP_SRCS:.cpp=.o)))
	mkdir -p $(LIBDIR)
	$(CXX) -dynamiclib -arch arm64 -undefined dynamic_lookup \
		-install_name @rpath/libppfbe_metal.dylib \
		-framework Metal -framework Foundation -o $@ $^

.PHONY: abi
abi: $(LIBDIR)/libppfbe_metal.dylib $(ENTRY_METALLIB) $(KERNEL_TABLE_RUST)

	@# The measured link contract, verified rather than assumed: print_rust must
	@# be UNDEFINED in this dylib. A definition here (a stub, or an accidental
	@# one pulled in from a shared header) silently swallows every backend log
	@# line at run time, with nothing to see anywhere.
	@if nm -gU "$(LIBDIR)/libppfbe_metal.dylib" | grep -q " _print_rust$$"; then \
		echo "ERROR: libppfbe_metal.dylib DEFINES _print_rust. The dylib must leave it undefined"; \
		echo "       so it resolves to the Rust host; a local definition"; \
		echo "       swallows every solver log line silently."; \
		rm -f "$(LIBDIR)/libppfbe_metal.dylib"; exit 1; \
	fi
	@# There is deliberately NO "must reference _print_rust" check here, unlike the
	@# CUDA recipe. print_rust is the CUDA backend's log seam; the Metal ABI backend
	@# routes every log line through the g_open_log_context callback the host
	@# installs at be_open (backend/backend.mm), so its dylib legitimately references
	@# no _print_rust and a check that demanded one failed every COLD build while the
	@# backend logged to the host correctly (measured: a fixture prints
	@# check_intersection and the per-step lines with print_rust absent from the
	@# dylib). The one hazard that IS Metal's, a LOCAL definition swallowing the
	@# lines, is what the check above catches.

# -MP makes the compiler emit a phony target for every header it names in the
# dependency file. Without it, a `.d` from an earlier build that names a header
# since RENAMED or DELETED stops the next build with
#   make: *** No rule to make target '<old header>', needed by '<object>'
# which reads as a missing source rather than as a stale artifact. That is not
# hypothetical here: the kernel bodies moved to `*.kernel.cpp` in one change, so
# every build directory predating it meets this once, and -MP cannot repair a
# `.d` already written without it. The one-time remedy is to delete the stale
# dependency files (`find <OUT_DIR> -name '*.d' -delete`); a build directory
# created after this change is immune.
-include $(DEPS)

# ---------------------------------------------------------------------------
# The fixture gates.
#
# `make fixtures` runs every fixture script against the backend the last
# `cargo build --release` produced. Each fixture asserts its own invariants and
# exits non-zero when one fails, so the loop under `set -e` is the whole gate.
# This does not build: cargo drives this Makefile, so a build from inside it
# would re-enter cargo.
#
# `make fixtures-poisoned` runs the same suite with PPF_METAL_POISON_ALLOC=1,
# which fills every fresh arena block with a non-zero pattern (arena.mm). That
# is the acceptance run for uninitialized device memory, and the plain run
# cannot substitute for it. Metal returns the bytes that are there rather than
# faulting, and a fresh MTLBuffer's bytes are zero, so a buffer the backend
# never writes reads as zero and the solver returns a plausible trajectory that
# the plain suite scores as a pass. Under the pattern that same read changes
# the answer and the fixture's assertion fails. Run both: the poisoned run is
# the only one that can see a missing initialization, and the plain run is the
# configuration that ships.
#
# An empty fixture list fails rather than passes. A gate that runs nothing
# reports success, which is the loudest possible reading of the quietest
# possible failure.
REPO_ROOT := ../../..
FIXTURE_PYTHON ?= $(HOME)/.local/share/ppf-cts/venv/bin/python
# A fixture is a Python script under the repository that carries the line
# `# metal-fixture`, so the list is found by what a file declares rather than
# by where it sits. METAL_FIXTURES overrides it with paths relative to the
# repository root, and `make fixtures-list` prints what would run.
METAL_FIXTURES ?= $(shell cd $(REPO_ROOT) && grep -rlx --include='*.py' \
                    --exclude-dir=target --exclude-dir=.git --exclude-dir=.venv \
                    '\# metal-fixture' . 2>/dev/null | sed 's|^\./||' | sort)

# An empty value reads as "not set" in the backend, so the plain target passes
# the variable through with nothing in it rather than needing a second recipe.
fixtures: METAL_POISON :=
fixtures-poisoned: METAL_POISON := 1

.PHONY: fixtures-list
fixtures-list:
	@for fixture in $(METAL_FIXTURES); do echo "$$fixture"; done

.PHONY: fixtures fixtures-poisoned
fixtures fixtures-poisoned:
	@set -e; \
	if [ -z "$(METAL_FIXTURES)" ]; then \
		echo "ERROR: no fixtures: no .py under $(REPO_ROOT) carries the line"; \
		echo "       '# metal-fixture', and METAL_FIXTURES names none."; \
		echo "       A gate that runs nothing would report success."; \
		exit 1; \
	fi; \
	if [ ! -x "$(FIXTURE_PYTHON)" ]; then \
		echo "ERROR: no frontend interpreter at $(FIXTURE_PYTHON)."; \
		echo "       Set FIXTURE_PYTHON to one carrying the frontend deps."; \
		exit 1; \
	fi; \
	cd $(REPO_ROOT); \
	for fixture in $(METAL_FIXTURES); do \
		echo "=== $$fixture (poison '$(METAL_POISON)')"; \
		PPF_ALLOW_METAL_SKELETON=1 PYTHONPATH=. \
			PPF_METAL_POISON_ALLOC=$(METAL_POISON) \
			"$(FIXTURE_PYTHON)" "$$fixture"; \
	done; \
	echo "$(words $(METAL_FIXTURES)) of $(words $(METAL_FIXTURES)) fixtures passed"

.PHONY: clean
clean:
	rm -rf $(OUT_DIR)
