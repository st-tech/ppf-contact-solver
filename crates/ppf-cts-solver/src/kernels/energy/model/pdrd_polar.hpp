// File: pdrd_polar.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Pure-float rotation math for the PDRD exact-rigid fit: quaternion <-> matrix
// conversion and the best-fit (polar) rotation of a near-rotation 3x3 matrix.
//
// The arithmetic itself is the neutral kernel pdrd_rigid.kernel.cpp; the three
// names below are wrappers that add none of their own, so every backend runs
// the same expressions in the same order.
//
// THE BODY IS NOT INCLUDED HERE, AND THAT IS FORCED BY THE RENDERING. A neutral
// kernel reaches a compiler as the rendering ppf-cts-compute/seam/kernelgen.py produced for
// it, and the three renderings carry three different names
// (pdrd_rigid.kernel.cu, .kernel.metal, .kernel.cpp). This header is compiled
// by TWO of those compilers, nvcc through pdrd_rigid.hpp and a plain host
// compiler through tests/test_pdrd_polar.cpp, so no single spelling here could
// name the right one for both. The includer supplies the body instead, ahead of
// this header, and it names the rendering its own compiler reads. Omitting it
// fails at the first wrapper below, naming the pdrd_ body that is missing.

#ifndef PDRD_POLAR_HPP
#define PDRD_POLAR_HPP

// Host/device portability: nvcc compiles these for both host and device; a plain
// host compiler (the unit test) sees no annotation.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define PDRD_POLAR_HD __device__ __host__
#else
#define PDRD_POLAR_HD
#endif

namespace PDRD {

// Quaternion (x, y, z, w) to a column-major 3x3 rotation matrix.
PDRD_POLAR_HD inline void rigid_quat_to_mat(const float q[4], float R[9]) {
    pdrd_quat_to_mat(q, R);
}

// Quaternion (x, y, z, w) from a proper column-major rotation matrix R, exact
// at 180 degrees (Shepperd's largest-diagonal branch).
PDRD_POLAR_HD inline void rigid_mat_to_quat(const float R[9], float q[4]) {
    pdrd_mat_to_quat(R, q);
}

// Best-fit rotation (polar factor) of a near-rotation 3x3 M (column-major), via
// the Mueller quaternion fixed-point iteration off a Gram-Schmidt seed. The
// derivation, and why the seed is not the identity, are stated with the body.
PDRD_POLAR_HD inline void rigid_polar_quat(const float M[9], float Rout[9]) {
    pdrd_polar_quat(M, Rout);
}

} // namespace PDRD

#endif // PDRD_POLAR_HPP
