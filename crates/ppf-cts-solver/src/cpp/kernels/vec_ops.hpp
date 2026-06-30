// File: vec_ops.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef VEC_OPS_HPP
#define VEC_OPS_HPP

#include "../main/cuda_utils.hpp" // cudaStream_t

namespace kernels {

// Each launcher takes an optional CUDA stream `queue`. The default (0) is the
// legacy stream, so every existing caller is unchanged; the PCG inner loop
// passes its own stream so the whole iteration queues without a host stall.

template <typename T>
void set(T *array, unsigned n, T value, cudaStream_t queue = 0);

template <typename T>
void copy(const T *src, T *dst, unsigned n, cudaStream_t queue = 0);

template <typename T>
void add_scaled(const T *src, T *dst, T scale, unsigned n,
                cudaStream_t queue = 0);

template <typename T>
void combine(const T *src_A, const T *src_B, T *dst, T a, T b, unsigned n,
             cudaStream_t queue = 0);

// Indirect-coefficient variants for the device-resident PCG loop: the scaling
// factor is read from a device pointer rather than passed as a host argument,
// so an iteration never copies a reduction result back to the host just to
// launch the next scaling kernel. All arithmetic is float32, matching the rest
// of the GPU path.

// dst[i] += sign * (*coeff) * src[i]
void add_scaled_indirect(const float *src, float *dst, const float *coeff,
                         float sign, unsigned n, cudaStream_t queue = 0);

// dst[i] = a * src_A[i] + (*coeff_B) * src_B[i]
void combine_indirect(const float *src_A, const float *src_B, float *dst,
                      float a, const float *coeff_B, unsigned n,
                      cudaStream_t queue = 0);

// *out = (*den > 0) ? (*num / *den) : 0, computed in float32 on the device (the
// host loop it replaces divides in double; the operands are float-precision
// reductions either way, so the quotient differs by <= 1 ULP). On the else
// branch (non-SPD / zero denominator) it writes 0 and sets *breakdown = 1 so the
// batched host convergence test can fail the solve instead of spinning.
// `breakdown` may be null.
void scalar_div(float *out, const float *num, const float *den, int *breakdown,
                cudaStream_t queue = 0);

// *dst = *src for a single device scalar (e.g. carrying rz across iterations).
void scalar_assign(float *dst, const float *src, cudaStream_t queue = 0);

// Sets *flag = 1 when *val is not strictly positive and finite (val <= 0, NaN,
// or +/-Inf). Used to detect a non-SPD preconditioner residual (rz <= 0) on the
// device so the sync-free loop can bail to the block-Jacobi fallback without a
// per-iteration host read. *flag is left untouched otherwise (latching).
void flag_if_nonpositive(const float *val, int *flag, cudaStream_t queue = 0);

} // namespace kernels

#endif // VEC_OPS_HPP
