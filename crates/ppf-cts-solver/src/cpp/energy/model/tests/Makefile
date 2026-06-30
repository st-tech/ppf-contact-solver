# Host (no-CUDA) build for the PDRD polar-fit regression test.
# pdrd_polar.hpp is dependency-free, so this compiles with a plain C++ compiler
# (clang++ on macOS, g++ on Linux) and needs no nvcc. Run: make test
CXX ?= c++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

TEST_SRC = test_pdrd_polar.cpp
TEST_EXE = test_pdrd_polar

all: $(TEST_EXE)

$(TEST_EXE): $(TEST_SRC) ../pdrd_polar.hpp
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_SRC)

# Build and run the regression (nonzero exit on any failure).
test: $(TEST_EXE)
	./$(TEST_EXE)

clean:
	rm -f $(TEST_EXE)

.PHONY: all test clean
