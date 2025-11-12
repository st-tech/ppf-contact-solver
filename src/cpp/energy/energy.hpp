// File: energy.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef ENERGY_DEF_HPP
#define ENERGY_DEF_HPP

#include "../csrmat/csrmat.hpp"
#include "../data.hpp"

namespace energy {

void embed_momentum_force_hessian(const DataSet &data,
                                  const Vec<Vec3f> &eval_x,
                                  const Vec<Vec3f> &velocity, float dt,
                                  const Vec<Vec3f> &target, Vec<float> &force,
                                  Vec<Mat3x3f> &fixed_hess,
                                  const ParamSet &param);

void embed_elastic_force_hessian(const DataSet &data, const Vec<Vec3f> &eval_x,
                                 Vec<float> &force, FixedCSRMat &fixed_hess,
                                 float dt, const ParamSet &param);

void embed_stitch_force_hessian(const DataSet &data, const Vec<Vec3f> &eval_x,
                                Vec<float> &force, FixedCSRMat &fixed_hess_out,
                                const ParamSet &param);

} // namespace energy

#endif
