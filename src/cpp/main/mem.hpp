// File: mem.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef CUDA_MEM_HPP
#define CUDA_MEM_HPP

#include "../vec/vec.hpp"
#include "cuda_utils.hpp"
#include <cassert>
#include <vector>

namespace mem {

std::vector<void *> cuda_malloc_list;

template <typename A> A *malloc_device(const A &host_a) {
    A *dev_a;
    CUDA_HANDLE_ERROR(cudaMalloc((void **)&dev_a, sizeof(A)));
    CUDA_HANDLE_ERROR(
        cudaMemcpy(dev_a, &host_a, sizeof(A), cudaMemcpyHostToDevice));
    cuda_malloc_list.push_back((void *)dev_a);
    return dev_a;
}

template <typename A> A *malloc_device(unsigned count) {
    A *dev_a = nullptr;
    if (count) {
        CUDA_HANDLE_ERROR(cudaMalloc((void **)&dev_a, count * sizeof(A)));
        CUDA_HANDLE_ERROR(cudaMemset(dev_a, 0, count * sizeof(A)));
        cuda_malloc_list.push_back((void *)dev_a);
    }
    return dev_a;
}

template <typename A>
A *malloc_device(const A *host_src, unsigned host_count, unsigned alloc_count) {
    A *dev_a = nullptr;
    if (alloc_count) {
        CUDA_HANDLE_ERROR(cudaMalloc((void **)&dev_a, alloc_count * sizeof(A)));
        cuda_malloc_list.push_back((void *)dev_a);
        CUDA_HANDLE_ERROR(cudaMemcpy(dev_a, host_src, host_count * sizeof(A),
                                     cudaMemcpyHostToDevice));
    }
    return dev_a;
}

template <typename A>
void copy_from_device_to_host(const A *dev_src, A *host_dst,
                              unsigned count = 1) {
    if (count) {
        CUDA_HANDLE_ERROR(cudaMemcpy(host_dst, dev_src, count * sizeof(A),
                                     cudaMemcpyDeviceToHost));
    }
}

template <typename A>
void copy_from_host_to_device(const A *host_src, A *dev_dst,
                              unsigned count = 1) {
    if (count) {
        CUDA_HANDLE_ERROR(cudaMemcpy(dev_dst, host_src, count * sizeof(A),
                                     cudaMemcpyHostToDevice));
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

void device_free() {
    for (auto x : cuda_malloc_list) {
        CUDA_HANDLE_ERROR(cudaFree(x));
    }
}

} // namespace mem

#endif
