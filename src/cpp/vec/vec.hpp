// File: vec.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef VEC_HPP
#define VEC_HPP

#include "../main/cuda_utils.hpp"
#include "../kernels/vec_ops.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

template <class T> struct VecVec {

    T *data{nullptr};
    unsigned *offset{nullptr};
    unsigned size{0};
    unsigned nnz{0};
    unsigned nnz_allocated{0};
    unsigned offset_allocated{0};

    __host__ static VecVec<T> alloc(unsigned nrow, unsigned max_nnz) {
        VecVec<T> result;
        result.size = nrow;
        result.nnz = 0;
        result.nnz_allocated = max_nnz;
        result.offset_allocated = nrow + 1;
        CUDA_HANDLE_ERROR(
            cudaMalloc(&result.offset, result.offset_allocated * sizeof(T)));
        CUDA_HANDLE_ERROR(
            cudaMalloc(&result.data, result.nnz_allocated * sizeof(T)));
        return result;
    }

    __host__ __device__ T &operator()(unsigned i, unsigned j) {
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
    __host__ __device__ const T &operator()(unsigned i, unsigned j) const {
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
    __host__ __device__ unsigned count(unsigned i) const {
        if (size == 0) {
            return 0;
        }
        if (i >= size) {
            printf("VecVec: count() i = %u, size = %u\n", i, size);
            assert(false);
        }
        return offset[i + 1] - offset[i];
    }
    __host__ __device__ unsigned count() const {
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

    __host__ __device__ T &operator[](unsigned i) {
        if (i >= size) {
            printf("Vec: operator[] i = %u, size = %u\n", i, size);
            assert(false);
        }
        return data[i];
    }
    __host__ __device__ const T &operator[](unsigned i) const {
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
    static Vec<T> alloc(unsigned n, unsigned alloc_factor = 1) {
        Vec<T> result;
        if (n > 0) {
            result.allocated = alloc_factor * n;
            CUDA_HANDLE_ERROR(
                cudaMalloc(&result.data, result.allocated * sizeof(T)));
            result.size = n;
        }
        return result;
    }
    bool free() {
        if (data) {
            CUDA_HANDLE_ERROR(cudaFree((void *)data));
            data = nullptr;
            return true;
        }
        return false;
    }
    Vec<T> clear(const T val = T()) {
        if (data && size > 0) {
            kernels::set(data, size, val);
        }
        return *this;
    }
    __device__ void atomic_add(unsigned i, const T &val) {
        assert(i < size);
        if (val) {
            atomicAdd(&data[i], val);
        }
    }
};

#endif
