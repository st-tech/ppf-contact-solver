// File: pair_cache.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE: plain C++ belonging to no backend, rendered into the
// three forms the three compilers read by ppf-cts-compute/seam/kernelgen.py.
// The two facts a compiler cannot infer are C++ attributes,
// `[[seam::device_fn]]` for the execution space and `[[seam::device]]` for the
// address space of a pointer parameter. Everything else is an ordinary name:
// `compute::atomic_uint_t` and the operations on it are defined by whichever
// backend prologue is in scope.
//
// The detect-once contact pair cache's record step. `count` runs past
// `capacity` on purpose: it counts every pair whether or not the pair fit, so
// the requirement is known even on the step that could not meet it, which is
// what lets the cache size itself for the next one. `overflow` is the flag
// stage 1 reads to decide that the replay must be abandoned and the BVH
// re-walked.

[[seam::device_fn]] inline void
pair_cache_record(unsigned *pair_data,
                      compute::atomic_uint_t *count,
                      compute::atomic_uint_t *overflow,
                      unsigned capacity, unsigned a, unsigned b) {
    const unsigned slot = compute::atomic_add(count, 1u);
    if (slot < capacity) {
        pair_data[2 * slot] = a;
        pair_data[2 * slot + 1] = b;
    } else {
        compute::atomic_store(overflow, 1u);
    }
}

// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `pair_cache_record_entry` shim a host C++ compiler
// compiles, and the Rust `#[repr(C)]` twin the driver fills. The two endpoint
// arrays are element gathers; the cache, its counter and its overflow flag are
// BASE pointers, because the slot is claimed inside the body and not chosen by
// the thread index.
//
// TWO COUNTS, AND THEY ARE DIFFERENT NUMBERS. `count` is the claim counter in
// device memory, which the body increments; `pair_count` is the guard bound,
// the number of candidate pairs this dispatch covers. Metal never faults on an
// out-of-bounds access, so the in-kernel guard is the only bound there.
//
// THE CLAIM IS NOT MADE SAFE BY A PARTITION. `compute::atomic_add` is a plain
// read, add and write back on the host seam, and the whole point of the
// counter is that each recorded pair gets a slot no other pair got, which no
// range partition can deliver. A generated entry point covers whatever range
// it is handed and says nothing about how that range may be cut, so the caller
// runs it as one ascending pass.
[[seam::entry(pair_count)]] void pair_cache_record(
    unsigned *pair_data,
    compute::atomic_uint_t *count,
    compute::atomic_uint_t *overflow,
    unsigned capacity, const unsigned *a,
    const unsigned *b,
    unsigned pair_count);

// THE SAME BODY OVER AN INTERLEAVED PAIR LIST, which is a second entry point
// rather than a second body.
//
// Which layout a candidate list arrives in is a property of the TRAVERSAL that
// produced it, not of the recording: this driver's traversal packs its output
// as `(a, b)` adjacent, and de-interleaving it to reach the two-array entry
// above would be a copy per pair to buy nothing. So the pair list arrives as a
// BASE pointer with the thread index forwarded, and the body reaches its own
// two endpoints at `2 * i` and `2 * i + 1`.
//
// It carries its own name because the composition is a new function: the body
// above already has callers under `pair_cache_record`, and C++ would
// resolve an overload on arity for a green build that `check-shared-wiring.py`
// then refuses, its census keying a neutral body on its name.
[[seam::device_fn]] inline void pair_cache_record_interleaved(
    unsigned *pair_data,
    compute::atomic_uint_t *count,
    compute::atomic_uint_t *overflow, unsigned capacity,
    const unsigned *pairs, unsigned i) {
    pair_cache_record(pair_data, count, overflow, capacity, pairs[2 * i],
                          pairs[2 * i + 1]);
}

[[seam::entry(pair_count, i)]] void pair_cache_record_interleaved(
    unsigned *pair_data,
    compute::atomic_uint_t *count,
    compute::atomic_uint_t *overflow,
    unsigned capacity, const unsigned *pairs,
    unsigned i,
    unsigned pair_count);
