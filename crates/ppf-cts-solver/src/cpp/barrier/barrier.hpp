// File: barrier.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef BARRIER_DEF_HPP
#define BARRIER_DEF_HPP

#include "../csrmat/csrmat.hpp"
#include "../data.hpp"

namespace barrier {

__device__ float energy(float g, float ghat, float offset, Barrier barrier);
__device__ float gradient(float g, float ghat, float offset, Barrier barrier);
__device__ float curvature(float g, float ghat, float offset, Barrier barrier);
__device__ Vec3f compute_edge_gradient(const Vec3f &e, float eps, float offset,
                                       Barrier barrier);
template <unsigned N>
__device__ float
compute_stiffness(const Proximity<N> &prox, const SVecf<N> &mass,
                  const FixedCSRMat &hess, const Vec3f &e, float eps,
                  float offset, const ParamSet &param);
__device__ Mat3x3f compute_edge_hessian(const Vec3f &e, float eps, float offset,
                                        Barrier barrier);
__device__ DiffTable2 compute_strainlimiting_diff_table(const Vec2f &a, float eps,
                                                        Barrier barrier);
__device__ float strainlimiting_energy(const Vec2f &a, float eps,
                                       Barrier barrier);

} // namespace barrier

#endif
