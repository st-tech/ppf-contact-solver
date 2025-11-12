// File: strainlimiting.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef STRAINLIMITING_DEF_HPP
#define STRAINLIMITING_DEF_HPP

#include "../csrmat/csrmat.hpp"
#include "../data.hpp"

namespace strainlimiting {

void embed_strainlimiting_force_hessian(const DataSet &data,
                                        const Vec<Vec3f> &eval_x,
                                        Vec<float> &force,
                                        const FixedCSRMat &fixed_hess_in,
                                        FixedCSRMat &fixed_hess_out,
                                        const ParamSet &param);

float line_search(const DataSet &data, const Vec<Vec3f> &eval_x,
                  const Vec<Vec3f> &prev, Vec<float> &min_gap,
                  const ParamSet &param);

} // namespace strainlimiting

#endif
