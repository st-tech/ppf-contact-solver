// File: vec.hpp
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef VEC_HPP
#define VEC_HPP

#include "../common.hpp"
#include "../main/cuda_utils.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

namespace VecDev {
template <class T> void set(T *array, unsigned size, T init_val);
template <class T> void copy(const T *src, T *dst, unsigned size);
template <class T>
void add_scaled(const T *src, T *dst, T scale, unsigned size);
template <class T>
void combine(const T *src_A, const T *src_B, T *dst, T a, T b, unsigned size);
template <class T> T inner_product(const T *a, const T *b, unsigned size);
} // namespace VecDev

template <class T> struct VecVec {

    T *data{nullptr};
    unsigned *offset{nullptr};
    unsigned size{0};
    unsigned nnz{0};
    unsigned nnz_allocated{0};
    unsigned offset_allocated{0};

    static VecVec<T> alloc(unsigned nrow, unsigned max_nnz) {
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
#if DEBUG_MODE
        if (i >= size) {
            printf("VecVec: operator() i = %u, size = %u\n", i, size);
            assert(false && "T &operator");
        }
#endif
        unsigned k = offset[i] + j;
#if DEBUG_MODE
        if (k >= offset[i + 1]) {
            printf("VecVec: k >= offset[i + 1] failed\n");
            assert(false && "T &operator");
        }
#endif
        return data[k];
    }
    __host__ __device__ const T &operator()(unsigned i, unsigned j) const {
#if DEBUG_MODE
        if (i >= size) {
            printf("VecVec: const T &operator() i = %u, size = %u\n", i, size);
            assert(false);
        }
#endif
        unsigned k = offset[i] + j;
#if DEBUG_MODE
        if (k >= offset[i + 1]) {
            printf("VecVec: k >= offset[i + 1] failed\n");
            assert(false);
            assert(false && "T &operator");
        }
#endif
        return data[k];
    }
    __host__ __device__ unsigned count(unsigned i) const {
        if (size == 0) {
            return 0;
        }
#if DEBUG_MODE
        if (i >= size) {
            printf("VecVec: count() i = %u, size = %u\n", i, size);
            assert(false && "count");
        }
#endif
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

    __device__ T &operator[](unsigned i) {
#if DEBUG_MODE
        if (i >= size) {
            printf("Vec: operator[] i = %u, size = %u\n", i, size);
            assert(false);
            assert(false && "T &operator[]");
        }
#endif
        return data[i];
    }
    __device__ const T &operator[](unsigned i) const {
#if DEBUG_MODE
        if (i >= size) {
            printf("Vec: const T &operator[] i = %u, size = %u\n", i, size);
            assert(false && "const T &operator[]");
        }
#endif
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
            VecDev::set(data, size, val);
        }
        return *this;
    }
    __device__ void atomic_add(unsigned i, const T &val) {
        if (i >= size) {
            printf("Vec: atomic_add i = %u, size = %u, sizeof(T) = %u\n", i,
                   size, sizeof(T));
            assert(false && "atomic_add");
        }
        if (val) {
            atomicAdd(&data[i], val);
        }
    }
    void copy(const Vec<T> &src) const {
        assert(src.size == size);
        VecDev::copy(src.data, this->data, src.size);
    }
    void add_scaled(Vec<T> &b, float c) const {
        assert(b.size == size);
        VecDev::add_scaled(b.data, this->data, c, size);
    }
    void combine(Vec<T> &a, Vec<T> &b, float c, float d) const {
        assert(a.size == size);
        assert(b.size == size);
        VecDev::combine(a.data, b.data, this->data, c, d, size);
    }
    float inner_product(const Vec<float> &b) const {
        assert(b.size == size);
        return VecDev::inner_product(data, b.data, size);
    }
};

#endif
