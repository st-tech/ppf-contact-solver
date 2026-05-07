# File: Makefile
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Builds libsimbackend_cpu as a STATIC archive (.a) that gets linked
# directly into the Rust binary at build time. Static linkage avoids
# the DLL search-order surprises we hit on Windows when the rig
# launches the solver via a child process.

OUT_DIR ?= build
LIBDIR  := $(OUT_DIR)/lib

CXX ?= clang++
AR  ?= ar

ifeq ($(shell uname),Darwin)
  EIGEN_PATH ?= $(shell brew --prefix eigen 2>/dev/null)/include/eigen3
else
  EIGEN_PATH ?= /usr/include/eigen3
endif

CXXFLAGS := -std=c++17 -O2 -fPIC -Wall -Wno-unused-function \
            -DEIGEN_WARNINGS_DISABLED \
            -I$(EIGEN_PATH)

EMUL_OBJS := $(OUT_DIR)/main.o
DEPS := $(EMUL_OBJS:.o=.d)

all: $(LIBDIR)/libsimbackend_cpu.a

$(OUT_DIR)/main.o: main.cpp
	mkdir -p $(@D)
	$(CXX) -c $(CXXFLAGS) -MMD $< -o $@

$(LIBDIR)/libsimbackend_cpu.a: $(EMUL_OBJS)
	mkdir -p $(LIBDIR)
	rm -f $@
	$(AR) rcs $@ $^

-include $(DEPS)

.PHONY: clean
clean:
	rm -rf $(OUT_DIR)
