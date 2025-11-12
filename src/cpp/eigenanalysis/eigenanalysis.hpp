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

} // namespace eigenanalysis

#endif
