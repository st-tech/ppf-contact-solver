// File: eigenanalysis.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef EIGANALYSIS_DEF_HPP
#define EIGANALYSIS_DEF_HPP

#include "../data.hpp"

namespace eigenanalysis {

__device__ Mat3x3f expand_U(const Mat3x2f &U);
__device__ Mat3x2f compute_force(const DiffTable2 &table, const Svd3x2 &svd);
__device__ Mat6x6f compute_hessian(const DiffTable2 &table, const Svd3x2 &svd,
                                   float eps);
__device__ Mat3x3f compute_force(const DiffTable3 &table, const Svd3x3 &svd);
__device__ Mat9x9f compute_hessian(const DiffTable3 &table, const Svd3x3 &svd,
                                   float eps);
// Fused tet path: the 12x12 element Hessian built directly from the nine
// eigenmodes (equivalent to convert_hessian(compute_hessian(...)) without the
// 9x9 intermediate); see the definition for the algebra.
__device__ void accumulate_hessian_tet_fused(const DiffTable3 &table,
                                             const Svd3x3 &svd, float eps,
                                             const Mat3x3f &inv_rest3x3,
                                             float mass, Mat12x12f &out);

} // namespace eigenanalysis

#endif
