// File: mem.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// CPU stand-in for cpp/main/mem.hpp. Same template signatures, but
// cudaMalloc / cudaMemcpy / cudaFree become std::malloc / std::memcpy
// / std::free. The "device" mirror is plain CPU memory, so a fetch()
// round-trip is a memcpy.

#ifndef PPF_EMUL_MEM_HPP
#define PPF_EMUL_MEM_HPP

#include "../cpp/vec/vec.hpp"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace mem {

inline std::vector<void *> cuda_malloc_list;

template <typename A> A *malloc_device(const A &host_a) {
    A *dev_a = static_cast<A *>(std::malloc(sizeof(A)));
    std::memcpy(dev_a, &host_a, sizeof(A));
    cuda_malloc_list.push_back(static_cast<void *>(dev_a));
    return dev_a;
}

template <typename A>
A *malloc_device(const A *host_src, unsigned host_count, unsigned alloc_count) {
    A *dev_a = nullptr;
    if (alloc_count) {
        dev_a = static_cast<A *>(std::malloc(alloc_count * sizeof(A)));
        cuda_malloc_list.push_back(static_cast<void *>(dev_a));
        if (host_count) {
            std::memcpy(dev_a, host_src, host_count * sizeof(A));
        }
    }
    return dev_a;
}

template <typename A>
void copy_from_device_to_host(const A *dev_src, A *host_dst,
                              unsigned count = 1) {
    if (count) {
        std::memcpy(host_dst, dev_src, count * sizeof(A));
    }
}

template <typename A>
void copy_from_host_to_device(const A *host_src, A *dev_dst,
                              unsigned count = 1) {
    if (count) {
        std::memcpy(dev_dst, host_src, count * sizeof(A));
    }
}

template <typename T>
VecVec<T> malloc_device(const VecVec<T> &host_src, unsigned alloc_factor = 1) {
    VecVec<T> dev_dst;
    if (host_src.nnz_allocated) {
        dev_dst.size = host_src.size;
        dev_dst.nnz = host_src.nnz;
        dev_dst.nnz_allocated = host_src.nnz_allocated * alloc_factor;
        dev_dst.offset_allocated = (host_src.size + 1) * alloc_factor;
        dev_dst.data = malloc_device<T>(host_src.data, host_src.count(),
                                        dev_dst.nnz_allocated);
        dev_dst.offset = malloc_device<unsigned>(
            host_src.offset, host_src.size + 1, dev_dst.offset_allocated);
    } else {
        dev_dst.size = 0;
        dev_dst.nnz = 0;
        dev_dst.nnz_allocated = 0;
        dev_dst.offset_allocated = 0;
        dev_dst.data = nullptr;
        dev_dst.offset = nullptr;
    }
    return dev_dst;
}

template <typename T>
Vec<T> malloc_device(const Vec<T> &host_src, unsigned alloc_factor = 1) {
    Vec<T> dev_dst;
    dev_dst.size = host_src.size;
    dev_dst.allocated = host_src.size * alloc_factor;
    dev_dst.data =
        malloc_device<T>(host_src.data, host_src.size, dev_dst.allocated);
    return dev_dst;
}

template <typename T>
void copy_to_device(const Vec<T> &host_src, Vec<T> &dev_dst) {
    assert(host_src.size <= dev_dst.allocated);
    dev_dst.size = host_src.size;
    copy_from_host_to_device(host_src.data, dev_dst.data, host_src.size);
}

template <typename T>
void copy_to_device(const VecVec<T> &host_src, VecVec<T> &dev_dst) {
    assert(host_src.nnz <= dev_dst.nnz_allocated);
    assert(host_src.size <= dev_dst.offset_allocated);
    dev_dst.size = host_src.size;
    dev_dst.nnz = host_src.nnz;
    copy_from_host_to_device(host_src.data, dev_dst.data, host_src.nnz);
    copy_from_host_to_device(host_src.offset, dev_dst.offset,
                             host_src.size + 1);
}

inline void device_free() {
    for (auto x : cuda_malloc_list) {
        std::free(x);
    }
    cuda_malloc_list.clear();
}

} // namespace mem

#endif
