// File: solver.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef CG_DEF_HPP
#define CG_DEF_HPP

#include "../csrmat/csrmat.hpp"
#include "../data.hpp"

namespace solver {

// `prm.precond` selects the base preconditioner: aggregate-Schwarz or 3x3
// block Jacobi (default). PDRD scenes solve in reduced 6-DOF rigid coordinates and
// return before the general PCG; on scenes with no PDRD bodies this is a plain
// cloth solve. `dt` enters the reduced PDRD per-body Hessian build. `positions`
// doubles as the `eval_x` the PDRD fit reads, and as the (reserved) Schwarz
// aggregation seed. `schwarz_fallback` is set to 1 if the Schwarz base produced
// a non-SPD residual and the solver latched the block-Jacobi fallback for this
// solve, else 0.
bool solve(const DynCSRMat &A, const FixedCSRMat &B, const Vec<Mat3x3f> &C,
           Vec<float> b, float tol, unsigned max_iter, Vec<float> x,
           const Vec<Vec3f> &positions, const ParamSet &prm, unsigned &iter,
           float &resid, unsigned &schwarz_fallback, const DataSet &data,
           float dt, Vec<float> pdrd_dtheta_out);

} // namespace solver

#endif
