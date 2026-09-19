// File: vec.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// The two array VIEWS the whole solver reads through, and nothing that
// allocates one.
//
// A view is a base pointer plus its counts: it says where the elements are and
// how many there are, and it computes no value. Reserving the storage those
// counts describe is a device operation, so it belongs to the backend that
// performs it. THE OWNING FORM IS `ArenaVec<T>` AND `ArenaVecVec<T>`, one
// declaration each backend satisfies for itself: an `ArenaPtr<T>`, which is the
// `ArenaHandle` of `../arena_handle.hpp` given an element type, plus the same
// counts these two carry, converting to a view on the way in. The names say
// what the thing IS rather than which compiler reserves the bytes, so a header
// declaring one is not thereby a header only one target can read.
//
// Keeping view and owner apart is what lets this file be read by every compiler
// in the project without any of them meeting another's allocation vocabulary.

#ifndef VEC_HPP
#define VEC_HPP

// The seam supplies the execution-space annotation and the atomic below. It is
// included here rather than assumed from an including file, because a view is
// reached from translation units that do not go through data.hpp.
#include "../seam/seam.hpp"

#include <cassert>
#include <cstdio>

namespace kernels {
// THE FILL BEHIND clear(), DECLARED HERE AND DEFINED BY THE BACKEND.
//
// A view cannot fill itself. Writing n elements of device memory is a DISPATCH,
// and a dispatch belongs to whichever target is running rather than to a header
// every compiler in this project reads: naming a queue type or a launch here
// would put one backend's vocabulary into all of them, which is exactly what
// splitting this header removed. So the declaration carries the three things a
// fill needs and nothing else, and each backend defines it beside the launcher
// it already has. It is a template, so a translation unit that never fills
// never needs a definition.
template <typename T> void fill_view(T *array, unsigned n, T value);
} // namespace kernels

template <class T> struct VecVec {

    T *data{nullptr};
    unsigned *offset{nullptr};
    unsigned size{0};
    unsigned nnz{0};
    unsigned nnz_allocated{0};
    unsigned offset_allocated{0};

    SM_INLINE_DEVICE_HOST T &operator()(unsigned i, unsigned j) {
        if (i >= size) {
            printf("VecVec: operator() i = %u, size = %u\n", i, size);
            assert(false);
        }
        unsigned k = offset[i] + j;
        if (k >= offset[i + 1]) {
            printf("VecVec: k >= offset[i + 1] failed\n");
            assert(false);
        }
        return data[k];
    }
    SM_INLINE_DEVICE_HOST const T &operator()(unsigned i, unsigned j) const {
        if (i >= size) {
            printf("VecVec: const T &operator() i = %u, size = %u\n", i, size);
            assert(false);
        }
        unsigned k = offset[i] + j;
        if (k >= offset[i + 1]) {
            printf("VecVec: k >= offset[i + 1] failed\n");
            assert(false);
        }
        return data[k];
    }
    SM_INLINE_DEVICE_HOST unsigned count(unsigned i) const {
        if (size == 0) {
            return 0;
        }
        if (i >= size) {
            printf("VecVec: count() i = %u, size = %u\n", i, size);
            assert(false);
        }
        return offset[i + 1] - offset[i];
    }
    SM_INLINE_DEVICE_HOST unsigned count() const {
        if (size == 0) {
            return 0;
        }
        return offset[size];
    }
};

template <class T> struct Vec {

    T *data{nullptr};
    unsigned size{0};
    unsigned allocated{0};

    SM_INLINE_DEVICE_HOST T &operator[](unsigned i) {
        if (i >= size) {
            printf("Vec: operator[] i = %u, size = %u\n", i, size);
            assert(false);
        }
        return data[i];
    }
    SM_INLINE_DEVICE_HOST const T &operator[](unsigned i) const {
        if (i >= size) {
            printf("Vec: const T &operator[] i = %u, size = %u\n", i, size);
            assert(false);
        }
        return data[i];
    }
    template <class A> Vec<A> flatten() {
        Vec<A> result;
        result.data = (A *)data;
        result.size = sizeof(T) / sizeof(A) * size;
        result.allocated = sizeof(T) / sizeof(A) * allocated;
        return result;
    }
    void resize(unsigned size) {
        if (size < this->allocated) {
            this->size = size;
        }
    }
    // Filling is the one operation here that is not a read, and it is
    // deliberately not an allocation either: the storage already exists and
    // this writes over it. The dispatch that does the writing is the backend's,
    // declared above.
    Vec<T> clear(const T val = T()) {
        if (data && size > 0) {
            kernels::fill_view(data, size, val);
        }
        return *this;
    }
    // NO ACCUMULATE HERE, and the absence is the rule rather than a gap. A
    // view holds `T *data`, plain storage, so a member spelled here would
    // atomically accumulate through a pointer that carries no record of being
    // reached by more than one thread. `compute::atomic_float_t` is the type
    // that carries it, and the operations take only that type, so an
    // accumulate is written at the call site as
    // `compute::atomic_add(slot + i, value)` on a pointer an entry point
    // resolved from a field declared as a slot. That is what lets a build read
    // an entry's argument record and say whether the kernel accumulates.
};

#endif
