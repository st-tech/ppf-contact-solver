// File: vec_ops.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef VEC_OPS_HPP
#define VEC_OPS_HPP

namespace kernels {

template <typename T>
void set(T *array, unsigned n, T value);

template <typename T>
void copy(const T *src, T *dst, unsigned n);

template <typename T>
void add_scaled(const T *src, T *dst, T scale, unsigned n);

template <typename T>
void combine(const T *src_A, const T *src_B, T *dst, T a, T b, unsigned n);

} // namespace kernels

#endif // VEC_OPS_HPP