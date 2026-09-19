// File: contact_barrier.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the three forms the three compilers read. The two facts a backend cannot
// infer are written as C++ attributes: `[[seam::device_fn]]` is the execution
// space, and `[[seam::thread]]` is the address space of a reference parameter,
// which MSL requires on every reference and pointer type.
//
// No include of its own. `Barrier` and the three barrier families `cubic`,
// `quadratic` and `logarithm` arrive from whatever declares them for the
// backend that is compiling, which is barrier.hpp and the three family headers
// under nvcc and on the host, and the segments spliced ahead of this one under
// MSL.
//
// The division goes through `fmath::div` rather than the operator, because MSL
// spells the correctly rounded quotient `precise::divide` and the plain
// operator is a different function there.

[[seam::device_fn]] inline float
barrier_energy(float gap, float ghat, float offset, Barrier kind) {
    if (kind == Barrier::Cubic) {
        return cubic::energy(gap, ghat, offset);
    }
    if (kind == Barrier::Quad) {
        return quadratic::energy(gap, ghat, offset);
    }
    if (kind == Barrier::Log) {
        return logarithm::energy(gap, ghat, offset);
    }
    return 0.0f;
}

[[seam::device_fn]] inline float
barrier_gradient(float gap, float ghat, float offset, Barrier kind) {
    if (kind == Barrier::Cubic) {
        return cubic::gradient(gap, ghat, offset);
    }
    if (kind == Barrier::Quad) {
        return quadratic::gradient(gap, ghat, offset);
    }
    if (kind == Barrier::Log) {
        return logarithm::gradient(gap, ghat, offset);
    }
    return 0.0f;
}

[[seam::device_fn]] inline float
barrier_curvature(float gap, float ghat, float offset, Barrier kind) {
    if (kind == Barrier::Cubic) {
        return cubic::curvature(gap, ghat, offset);
    }
    if (kind == Barrier::Quad) {
        return quadratic::curvature(gap, ghat, offset);
    }
    if (kind == Barrier::Log) {
        return logarithm::curvature(gap, ghat, offset);
    }
    return 0.0f;
}

[[seam::device_fn]] inline Vec3f
barrier_edge_gradient(const Vec3f &edge, float ghat,
                          float offset, Barrier kind) {
    const float norm = edge.norm();
    const Vec3f normal = fmath::div(1.0f, norm) * edge;
    return normal * barrier_gradient(norm, ghat, offset, kind);
}

[[seam::device_fn]] inline Mat3x3f
barrier_edge_hessian(const Vec3f &edge, float ghat,
                         float offset, Barrier kind) {
    const float scale =
        fmath::div(barrier_curvature(edge.norm(), ghat, offset, kind),
                 edge.squaredNorm());
    return scale * edge * edge.transpose();
}
