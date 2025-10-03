# Makefile for testing reduce.cu
NVCC = nvcc
NVCC_FLAGS = -std=c++17 -O3 -arch=sm_70 -I../.. --extended-lambda
DEBUG_FLAGS = -g -G -DDEBUG

# Source files
REDUCE_SRC = ../reduce.cu
TEST_SRC = test_reduce.cu

# Object files
REDUCE_OBJ = reduce.o
TEST_OBJ = test_reduce.o

# Executable
TEST_EXE = test_reduce

# Default target
all: $(TEST_EXE)

# Build the test executable
$(TEST_EXE): $(REDUCE_OBJ) $(TEST_OBJ)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Build reduce.o
$(REDUCE_OBJ): $(REDUCE_SRC) ../reduce.hpp ../../common.hpp
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $(REDUCE_SRC)

# Build test_reduce.o
$(TEST_OBJ): $(TEST_SRC) ../reduce.hpp
	$(NVCC) $(NVCC_FLAGS) -c -o $@ $(TEST_SRC)

# Debug build
debug: NVCC_FLAGS += $(DEBUG_FLAGS)
debug: clean $(TEST_EXE)

# Run the tests
test: $(TEST_EXE)
	./$(TEST_EXE)

# Run tests in verbose mode
test-verbose: $(TEST_EXE)
	./$(TEST_EXE) -v

# Clean build artifacts
clean:
	rm -f $(REDUCE_OBJ) $(TEST_OBJ) $(TEST_EXE)

.PHONY: all clean test test-verbose debug