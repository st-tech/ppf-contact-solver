// File: contact_assembly.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE: plain C++ belonging to no backend, rendered into the
// three forms the three compilers read by ppf-cts-compute/seam/kernelgen.py. The two facts
// a compiler cannot infer are C++ attributes, `[[seam::device_fn]]` for the
// execution space and `[[seam::device]]` / `[[seam::thread]]` for the address
// space of a pointer or reference parameter.
//
// THE TWO ADDRESS SPACES HERE ARE NOT INTERCHANGEABLE. The sparsity arrays are
// buffers the whole grid reads, so they are `[[seam::device]]`; the force, the
// Hessian and the extended blocks are one thread's own registers and stack, so
// they are `[[seam::thread]]`. The extended Hessian is `SMatf<3 * N, 3 * N>`,
// 144 floats at the N = 4 the callers instantiate, and it is written through a
// reference rather than returned: a large matrix returned by value is what
// exhausts the small default device stack, so this tree accumulates in place.

#include "../csrmat/fixed_csr.kernel.cpp"

// One contact's force and Hessian spread over the N vertices its barycentric
// weights name. The Hessian block for the pair (i, j) is `w_i w_j H`, so the
// extended matrix is the outer product of the weight vector with itself against
// H, which is PSD whenever H is: the weights enter as a congruence and cannot
// introduce a negative mode.
template <unsigned N>
[[seam::device_fn]] inline void extend_contact_force_hessian(
    const SVecf<N> &weight, const Vec3f &force,
    const Mat3x3f &hessian,
    SMatf<3, N> &extended_force,
    SMatf<3 * N, 3 * N> &extended_hessian) {
    for (unsigned i = 0; i < N; ++i) {
        extended_force.col(i) = weight[i] * force;
    }
    for (unsigned i = 0; i < N; ++i) {
        for (unsigned j = 0; j < N; ++j) {
            extended_hessian.template block<3, 3>(3 * i, 3 * j) =
                weight[i] * weight[j] * hessian;
        }
    }
}

