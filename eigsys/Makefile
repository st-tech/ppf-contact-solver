# File: Makefile
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

CXX = g++
CXXFLAGS = -O3 -std=c++17

EXEC2 = eigen2
EXEC3 = eigen3
BENCH3 = bench3
SRC2 = eigsys_2.cpp
SRC3 = eigsys_3.cpp
BENCH3_SRC = bench3.cpp
INC2 = eig-hpp/eigsolve2x2.hpp
INC3 = eig-hpp/eigsolve3x3.hpp

EIGEN_DIR = /usr/include/eigen3

all: $(EXEC2) $(EXEC3) $(BENCH3)

$(EXEC2): $(SRC2) $(INC2)
	$(CXX) $(CXXFLAGS) -I$(EIGEN_DIR) $(SRC2) -o $(EXEC2)

$(EXEC3): $(SRC3) $(INC3)
	$(CXX) $(CXXFLAGS) -I$(EIGEN_DIR) $(SRC3) -o $(EXEC3)

$(BENCH3): $(BENCH3_SRC) $(INC3)
	$(CXX) $(CXXFLAGS) -I$(EIGEN_DIR) $(BENCH3_SRC) -o $(BENCH3)

clean:
	rm -f $(EXEC2) $(EXEC3) $(EXEC4) $(BENCH3)

run2: $(EXEC2)
	./$(EXEC2)

run3: $(EXEC3)
	./$(EXEC3)

run_bench3: $(BENCH3)
	./$(BENCH3)
	
.PHONY: clean run2 run3 bench3

