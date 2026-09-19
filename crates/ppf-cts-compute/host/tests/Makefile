# Host (no-CUDA) builds for the model-level regression tests.
# Both compile with a plain C++ compiler (clang++ on macOS, g++ on Linux) and
# need no nvcc: each takes the `cpp` rendering of the kernel it pins down (see
# below). Run: make test
#
# THE RECIPE IS HERE AND THE TEST SOURCES ARE NOT. Rendering a neutral kernel
# for a target and compiling the result is backend-specific work, so it belongs
# to this crate; what the two tests ASSERT is the bending convention and the
# polar decomposition, which is simulation, so those sources stay in
# ppf-cts-solver where the kernels they pin down are. $(TESTS) is the one
# variable that names them.
CXX ?= c++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

# WHERE THE NEUTRAL KERNEL TREE IS, SUPPLIED BY THE CALLER. This recipe belongs
# to ppf-cts-compute and the sources it compiles belong to ppf-cts-solver, so
# the path between them is an input rather than a constant.
KERNEL_ROOT ?= ../../../ppf-cts-solver/src/kernels
ifeq ($(wildcard $(KERNEL_ROOT)/data.hpp),)
$(error KERNEL_ROOT=$(KERNEL_ROOT) does not hold data.hpp: pass the neutral kernel tree of ppf-cts-solver)
endif

# The two test sources, which stayed behind for the reason above.
TESTS = $(KERNEL_ROOT)/energy/model/tests

# Where the binaries land, resolving to crates/ppf-cts-compute/build-tests.
#
# THE DEPTH IS LOAD-BEARING AND FAILS SILENTLY: a wrong one still builds and
# still passes while writing inside a watched source tree, and the only
# evidence is the rebuild cost afterward. Three trees are watched: the solver's
# build script watches crates/ppf-cts-solver/src/kernels and this crate's cuda
# directory, and the server's watches crates/ppf-cts-solver/src. cargo reads a
# directory in rerun-if-changed as "rerun if any descendant changes", so an
# artifact inside one costs a 48 s recompile of ppf-cts-solver and
# ppf-cts-server on the next cargo build. .gitignore covering a path buys
# nothing here, because cargo's mtime walk does not consult it.
OUT ?= ../../build-tests

# ---------------------------------------------------------------------------
# Neutral kernel sources.
#
# The arithmetic these tests pin down lives once, as a `*.kernel.cpp` file
# under $(KERNEL_ROOT): plain C++, no preprocessor conditional, no macro of its
# own, and no spelling that belongs to one backend. ../../seam/kernelgen.py
# renders one into the form each compiler reads, and these tests take the `cpp`
# rendering, which drops the address spaces and the execution space and keeps
# everything else. A test names it from the generated root that -I puts on the
# search path:
#   #include "energy/model/<name>.kernel.cpp"
#
# The rendering lands under $(OUT), beside the binaries and outside every
# watched tree, for the reason above.
KERNELGEN := ../../seam/kernelgen.py
KERNELGEN_DIR := $(OUT)/kernelgen

CXXFLAGS += -I$(KERNELGEN_DIR)
# AND THE KERNEL ROOT, so a rendering can reach the SHARED HEADERS it names.
# A `.kernel.cpp` brings no vocabulary of its own: the spellings and the
# records come from whichever shared header the includer pulled in first, and
# a rendering compiled alone has nothing behind it. Without this the body
# parses with `VertexProp` unknown and fails on a member of "non-class type
# const int", which reads as a defect in the body rather than a missing -I.
CXXFLAGS += -I$(KERNEL_ROOT)

$(KERNELGEN_DIR)/%.kernel.cpp: $(KERNEL_ROOT)/%.kernel.cpp $(KERNELGEN)
	mkdir -p $(@D)
	python3 $(KERNELGEN) --target cpp --out $@ $<

PDRD_RIGID_HOST := $(KERNELGEN_DIR)/energy/model/pdrd_rigid.kernel.cpp
# A RENDERED BODY'S OWN INCLUDES HAVE TO BE RENDERED BESIDE IT. The renderings
# keep their source's relative includes verbatim, so
# `energy/model/pdrd_rigid.kernel.cpp` reaching `../../csrmat/fixed_csr.kernel.cpp`
# resolves inside KERNELGEN_DIR and not back in the solver tree. Rendering only
# the body under test therefore compiles until that body gains a neutral
# dependency, and then fails naming a file the test never mentions.
#
# It is listed rather than derived because the set is two files and a scanner
# for `#include` lines in a kernel body is a second, weaker copy of what the
# transcompiler already knows.
PDRD_RIGID_DEPS_HOST := $(KERNELGEN_DIR)/csrmat/fixed_csr.kernel.cpp
SHELL_BEND_STIFFNESS_HOST := \
	$(KERNELGEN_DIR)/energy/model/shell_bend_stiffness.kernel.cpp

POLAR_EXE = $(OUT)/test_pdrd_polar
ANISO_EXE = $(OUT)/test_bend_aniso

all: $(POLAR_EXE) $(ANISO_EXE)

$(POLAR_EXE): $(TESTS)/test_pdrd_polar.cpp \
              $(KERNEL_ROOT)/energy/model/pdrd_polar.hpp $(PDRD_RIGID_HOST) \
              $(PDRD_RIGID_DEPS_HOST)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ $(TESTS)/test_pdrd_polar.cpp

$(ANISO_EXE): $(TESTS)/test_bend_aniso.cpp $(SHELL_BEND_STIFFNESS_HOST)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ $(TESTS)/test_bend_aniso.cpp

# Build and run the regressions (nonzero exit on any failure).
test: $(POLAR_EXE) $(ANISO_EXE)
	./$(POLAR_EXE)
	./$(ANISO_EXE)

clean:
	rm -f $(POLAR_EXE) $(ANISO_EXE)
	rm -rf $(KERNELGEN_DIR)

.PHONY: all test clean
