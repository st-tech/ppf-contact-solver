// File: solver.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef CG_DEF_HPP
#define CG_DEF_HPP

#include "../csrmat/csrmat.hpp"
#include "../data.hpp"

namespace solver {

bool solve(const DynCSRMat &A, const FixedCSRMat &B, const Vec<Mat3x3f> &C,
           Vec<float> b, float tol, unsigned max_iter, Vec<float> x,
           unsigned &iter, float &resid);

} // namespace solver

#endif
