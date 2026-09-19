// File: bitonic.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend:
// no preprocessor conditional, no macro of its own, and no spelling that only
// nvcc or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders
// it into the three forms the three compilers read, and the build hands each
// compiler its own form. The two facts a backend cannot infer are written as
// C++ attributes: `[[seam::device_fn]]` is the execution space, and
// `[[seam::device]]` is the address space of a pointer parameter. MSL requires
// the second on every pointer and reference type; CUDA and the host have one
// address space and are handed the same declarations with it removed.
//
// One step of a bitonic sort over a key array and the permutation that follows
// it. The comparison is on the key first and on the index second, so equal keys
// keep a total order and the sort is deterministic across backends; nothing
// here reaches a special-function unit, a division or a root, so the body is
// bit-exact comparable and its parity gate admits no tolerance.

[[seam::device_fn]] inline void
bitonic_step(unsigned *key, unsigned *index,
                 unsigned count, unsigned thread_index, unsigned subsequence,
                 unsigned stride) {
    const unsigned partner = thread_index ^ stride;
    if (partner <= thread_index || partner >= count) {
        return;
    }
    const bool ascending = (thread_index & subsequence) == 0u;
    const bool swap =
        ascending
            ? (key[thread_index] > key[partner] ||
               (key[thread_index] == key[partner] &&
                index[thread_index] > index[partner]))
            : (key[thread_index] < key[partner] ||
               (key[thread_index] == key[partner] &&
                index[thread_index] < index[partner]));
    if (swap) {
        const unsigned key_value = key[thread_index];
        key[thread_index] = key[partner];
        key[partner] = key_value;
        const unsigned index_value = index[thread_index];
        index[thread_index] = index[partner];
        index[partner] = index_value;
    }
}

// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `bitonic_step_entry` shim a host C++ compiler
// compiles, and the Rust `#[repr(C)]` twin the driver fills.
//
// BOTH ARRAYS ARE BASE POINTERS AND THE THREAD INDEX IS FORWARDED, because a
// comparator reaches two elements, its own and its partner's, and the partner
// is computed inside the body. That is exactly what makes this a forwarded
// index rather than a gather: the addressing is the network's, not the entry
// point's.
//
// TWO COUNTS, AND THEY ARE DIFFERENT NUMBERS. `count` is the sorted array's
// length, which the body compares the partner against so a network wider than
// the data leaves the tail alone. `thread_count` is the guard bound, the number
// of comparators this dispatch covers. Metal never faults on an out-of-bounds
// access, so the in-kernel guard is the only bound there, and a launch rounds
// its grid up to whole threadgroups on every backend.
[[seam::entry(thread_count, thread_index)]] void bitonic_step(
    unsigned *key, unsigned *index,
    unsigned count, unsigned thread_index, unsigned subsequence,
    unsigned stride, unsigned thread_count);
