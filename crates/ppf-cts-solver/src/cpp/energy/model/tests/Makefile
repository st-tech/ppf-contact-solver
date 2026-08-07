# Host (no-CUDA) builds for the model-level regression tests.
# Both compile with a plain C++ compiler (clang++ on macOS, g++ on Linux) and
# need no nvcc: pdrd_polar.hpp is dependency-free, and data.hpp is the same
# header the CUDA-free emulator already compiles host-side. Run: make test
CXX ?= c++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

# Where the binaries land, resolving to crates/ppf-cts-solver/build-tests.
#
# This MUST stay outside crates/ppf-cts-solver/src. Two build scripts watch
# that tree recursively, and cargo reads a directory in rerun-if-changed as
# "rerun if any descendant changes", so an artifact written under it
# invalidates both and costs a 48 s recompile of ppf-cts-solver and
# ppf-cts-server on the next cargo build. src/cpp/build is not far enough out,
# because cargo's mtime walk does not consult .gitignore.
OUT ?= ../../../../../build-tests

POLAR_EXE = $(OUT)/test_pdrd_polar
ANISO_EXE = $(OUT)/test_bend_aniso

all: $(POLAR_EXE) $(ANISO_EXE)

$(POLAR_EXE): test_pdrd_polar.cpp ../pdrd_polar.hpp
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ test_pdrd_polar.cpp

$(ANISO_EXE): test_bend_aniso.cpp ../../../data.hpp
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ test_bend_aniso.cpp

# Build and run the regressions (nonzero exit on any failure).
test: $(POLAR_EXE) $(ANISO_EXE)
	./$(POLAR_EXE)
	./$(ANISO_EXE)

clean:
	rm -f $(POLAR_EXE) $(ANISO_EXE)

.PHONY: all test clean
