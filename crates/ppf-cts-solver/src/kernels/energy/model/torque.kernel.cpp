// File: torque.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. Plain C++ belonging to no backend: no preprocessor
// conditional, no macro of its own, and no spelling that only one of the three
// compilers accepts. ppf-cts-compute/seam/kernelgen.py renders it into the .cu, the .metal
// and the .cpp that nvcc, the Metal shader compiler and a host C++ compiler
// read. The two facts a compiler cannot infer are C++ attributes:
// `[[seam::device_fn]]` is the execution space, and `[[seam::thread]]` is the
// address space of a pointer or reference parameter. MSL requires the second on
// every one of them; CUDA and the host have a single address space and are
// handed the same declarations with it removed.
//
// Supplies the `fmath::` arithmetic these bodies name, the single-precision
// bounded cosine among it. The Metal backend resolves this include itself
// (metal/shader_compiler.mm neutralizes a quoted include as it splices the
// segment) and answers with its own prologue, so float_math.hpp, which is what
// the CUDA and host arms of `fmath::cos_bounded` reach, never becomes a segment
// and its CUDA and std spellings stay out of MSL.
#include "../../seam/seam.hpp"

// The commanded-torque constraint, as one body for both backends.
//
// A torque group is a set of vertices given a total torque about an axis the
// solver derives from the group's own shape: the mass-weighted centroid, then
// the principal axes of the member cloud's covariance, then one of those three
// columns chosen by the authored `axis_component` and oriented by a hint
// vertex. Every float operation in that chain lives here; the caller supplies
// only the walk over device memory and the loop control, so a fixture that
// calls these bodies is comparing the kernel's arithmetic rather than a second
// copy of it.
//
// TOLERANCE CLASS. `torque_axis` reaches `fmath::acos` and
// `fmath::cos_bounded` through the Cardano eigenvalue formula, and the two
// backends do not agree to the bit on either: CUDA's bounded cosine is the
// special-function unit and MSL's is the accurate library form. Everything the
// axis feeds is therefore characterized-tolerance, which is the axis itself,
// the perpendicular-radius normalization and the per-vertex force and Hessian.
// The centroid is NOT: it is a multiply-accumulate and one division, with no
// transcendental anywhere in the chain, and the division is spelled
// `fmath::div` so it is correctly rounded on both.

// Accumulates one member into the mass-weighted centroid: the position scaled
// by the member's mass into the running sum, and the mass into the running
// total, so the division in `torque_centroid_finish` recovers the center.
[[seam::device_fn]] inline void
torque_centroid_step(const Vec3f &x, float mass,
                         Vec3f &center,
                         float &total_mass) {
    center += x * float(mass);
    total_mass += mass;
}

// Divides the accumulated centroid by the total mass. A group whose members are
// all massless leaves the centroid at the accumulated value rather than
// dividing by zero, which is the guard CUDA states as `total_mass > 1e-12f`.
[[seam::device_fn]] inline void
torque_centroid_finish(Vec3f &center, float total_mass) {
    if (total_mass > 1e-12f) {
        center *= float(fmath::div(1.0f, total_mass));
    }
}

// Accumulates one member's offset from the centroid into the six independent
// entries of the covariance, in the order {00, 01, 02, 11, 12, 22}.
[[seam::device_fn]] inline void
torque_covariance_step(const Vec3f &d,
                           float *covariance) {
    covariance[0] += d[0] * d[0];
    covariance[1] += d[0] * d[1];
    covariance[2] += d[0] * d[2];
    covariance[3] += d[1] * d[1];
    covariance[4] += d[1] * d[2];
    covariance[5] += d[2] * d[2];
}

// Normalizes the covariance by the member count.
[[seam::device_fn]] inline void
torque_covariance_finish(float *covariance,
                             unsigned count) {
    const float inverse = fmath::div(1.0f, float(count));
    for (unsigned entry = 0; entry < 6; ++entry) {
        covariance[entry] *= inverse;
    }
}

// The symmetric 3x3 eigen-frame, with the eigenvectors ordered by DESCENDING
// eigenvalue. Cardano for the eigenvalues, then a cross product of two rows of
// (A - lambda I) for each eigenvector, then a Gram-Schmidt pass so the frame is
// orthonormal even where two eigenvalues nearly coincide and the two cross
// products nearly align.
//
// The eigenvalues themselves are not returned. They order the columns and
// nothing downstream reads them, and a caller that wanted them would be asking
// a different question than "which way does this cloud point".
[[seam::device_fn]] inline void
torque_sym_eig3x3(float a00, float a01, float a02, float a11, float a12,
                      float a22, Mat3x3f &evecs) {
    const float q = fmath::div(a00 + a11 + a22, 3.0f);
    const float p2 = (a00 - q) * (a00 - q) + (a11 - q) * (a11 - q) +
                     (a22 - q) * (a22 - q) +
                     2.0f * (a01 * a01 + a02 * a02 + a12 * a12);
    const float p = fmath::sqrt(fmath::max(fmath::div(p2, 6.0f), 0.0f));

    float lam0, lam1, lam2;
    if (p < 1e-12f) {
        // A (near) scalar multiple of the identity: the diagonal IS the
        // spectrum and the Cardano branch would divide by p.
        lam0 = a00;
        lam1 = a11;
        lam2 = a22;
    } else {
        const float inv_p = fmath::div(1.0f, p);
        // B = (A - qI) / p, whose determinant is the Cardano discriminant.
        const float b00 = (a00 - q) * inv_p;
        const float b01 = a01 * inv_p;
        const float b02 = a02 * inv_p;
        const float b11 = (a11 - q) * inv_p;
        const float b12 = a12 * inv_p;
        const float b22 = (a22 - q) * inv_p;
        const float det_b = b00 * (b11 * b22 - b12 * b12) -
                            b01 * (b01 * b22 - b12 * b02) +
                            b02 * (b01 * b12 - b11 * b02);
        const float r = fmath::min(1.0f, fmath::max(-1.0f, det_b * 0.5f));
        // acos of a value clamped into [-1, 1] lands in [0, pi], so the two
        // cosine arguments below stay inside [0, pi/3 + 2*pi/3], which is the
        // bound `fmath::cos_bounded` relies on.
        const float phi = fmath::div(fmath::acos(r), 3.0f);
        lam0 = q + 2.0f * p * fmath::cos_bounded(phi);
        lam2 = q + 2.0f * p * fmath::cos_bounded(phi + 2.0943951f);
        lam1 = 3.0f * q - lam0 - lam2;
    }

    if (lam1 > lam0) {
        const float swap = lam0;
        lam0 = lam1;
        lam1 = swap;
    }
    if (lam2 > lam0) {
        const float swap = lam0;
        lam0 = lam2;
        lam2 = swap;
    }
    if (lam2 > lam1) {
        const float swap = lam1;
        lam1 = lam2;
        lam2 = swap;
    }

    const float eigenvalue[3] = {lam0, lam1, lam2};
    for (unsigned column = 0; column < 3; ++column) {
        const float l = eigenvalue[column];
        const Vec3f r0(a00 - l, a01, a02);
        const Vec3f r1(a01, a11 - l, a12);
        const Vec3f r2(a02, a12, a22 - l);
        Vec3f v = r0.cross(r1);
        float vn = v.norm();
        if (vn < 1e-8f) {
            v = r0.cross(r2);
            vn = v.norm();
        }
        if (vn < 1e-8f) {
            v = r1.cross(r2);
            vn = v.norm();
        }
        if (vn > 1e-8f) {
            for (unsigned dimension = 0; dimension < 3; ++dimension) {
                evecs(dimension, column) = fmath::div(v[dimension], vn);
            }
        } else {
            // Every row of (A - lambda I) is parallel to every other, which is
            // what an isotropic block looks like: any direction is an
            // eigenvector, so take the axis one.
            for (unsigned dimension = 0; dimension < 3; ++dimension) {
                evecs(dimension, column) = dimension == column ? 1.0f : 0.0f;
            }
        }
    }

    const Vec3f e0 = evecs.col(0).normalized();
    Vec3f e1 = evecs.col(1) - e0 * e0.dot(evecs.col(1));
    // A REPEATED EIGENVALUE CAN LEAVE THESE TWO COLUMNS PARALLEL, and then
    // `e1` normalizes to zero and `e2 = e0 x e1` goes to zero with it, taking
    // the THIRD column down even though its own eigenvalue is distinct and its
    // eigenvector well determined. A group whose members lie on a circle, which
    // is the most ordinary shape a torque group has, therefore gets a ZERO axis
    // and receives no force at all.
    //
    // THE ABSENT DEGENERATE GUARD IS DELIBERATE, NOT AN OVERSIGHT. Where two
    // eigenvalues coincide, every direction in their plane is an eigenvector,
    // so any replacement picked for `e1` is arbitrary and the commanded torque
    // follows whichever one is picked. That is a decision about what the solver
    // computes, not a local repair.
    // `a_circular_group_reproduces_the_reference_zero_axis` in
    // `src/driver/assemble.rs` pins the current behavior, so a repair turns
    // that test red and has to land deliberately.
    e1.normalize();
    const Vec3f e2 = e0.cross(e1);
    evecs.col(0) = e0;
    evecs.col(1) = e1;
    evecs.col(2) = e2;
}

// The group's rotation axis: the requested principal column, normalized, with
// its sign ambiguity resolved by the hint direction. An axis and its negative
// are the same principal direction, so without the hint the commanded torque
// could come out reversed on a group whose covariance is barely perturbed
// between two frames.
[[seam::device_fn]] inline void
torque_axis(const float *covariance,
                unsigned axis_component,
                const Vec3f &hint_direction,
                Vec3f &axis) {
    Mat3x3f evecs;
    torque_sym_eig3x3(covariance[0], covariance[1], covariance[2],
                          covariance[3], covariance[4], covariance[5], evecs);
    axis = evecs.col(axis_component).normalized();
    if (axis.dot(hint_direction) < 0.0f) {
        axis = -axis;
    }
}

// Accumulates one member's squared perpendicular radius about the axis.
[[seam::device_fn]] inline void
torque_rperp_step(const Vec3f &r,
                      const Vec3f &axis,
                      float &sum) {
    const Vec3f perpendicular = r - axis * axis.dot(r);
    sum += perpendicular.squaredNorm();
}

// The normalization the per-vertex force is scaled by, so the group's TOTAL
// torque is the commanded magnitude. A group with no perpendicular extent at
// all (every member on the axis) gets zero rather than a division by zero, and
// then simply receives no force.
[[seam::device_fn]] inline float torque_rperp_finish(float sum) {
    return sum > 1e-12f ? fmath::div(1.0f, sum) : 0.0f;
}

// The per-vertex torque term.
//
//   force_i = (axis x r_perp_i) * magnitude / sum_j |r_perp_j|^2
//
// which integrates to the commanded torque with zero net linear force. The
// caller SUBTRACTS `force` from the vertex gradient, because the assembly
// accumulates energy gradients and this is the applied force; returning the
// unsigned quantity keeps that sign visible at the call site rather than buried
// here.
//
// THE HESSIAN IS PSD BY CONSTRUCTION, unconditionally and with no flag reaching
// it. The exact derivative of that force has a skew part, which is not usable
// in a symmetric solve, and a symmetric part `scale * (I - axis axis^T)`. The
// projection keeps the symmetric part and clamps its coefficient at zero, which
// matters because `scale` carries the SIGNED magnitude: a reverse torque makes
// `scale * P` negative-semidefinite, and assembling it would put a negative
// eigenvalue into a matrix the PCG guards treat as SPD by construction. The
// force above stays exact, which is the projected-Newton pattern.
[[seam::device_fn]] inline void torque_vertex_force_hessian(
    const Vec3f &y, const Vec3f &center,
    const Vec3f &axis, float magnitude,
    float inv_r_perp_sq_sum, Vec3f &force,
    Mat3x3f &hessian) {
    const Vec3f r = (y - center).cast<float>();
    const Vec3f perpendicular = r - axis * axis.dot(r);
    const float scale = magnitude * inv_r_perp_sq_sum;
    force = axis.cross(perpendicular) * scale;
    const Mat3x3f projection = Mat3x3f::Identity() - axis * axis.transpose();
    hessian = fmath::max(0.0f, scale) * projection;
}

// The per-group pre-pass: one group's centroid, principal axis and radius
// normalization, which every member of that group then scales its own force by.
//
// THREE WALKS OVER THE SAME MEMBERS, in the order `energy.cu` performs them and
// for the reason it cannot fuse them either: the covariance is about the
// centroid, so it cannot start until the first walk finishes, and the
// perpendicular radius is about the axis the covariance produces. Each walk
// accumulates through the bodies above, so the only thing this composition adds
// is which member it reads.
//
// THE MEMBER LIST IS A FLAT BOUND ARRAY WALKED HERE rather than a gather,
// because a group's member count is DATA: `vertex_start` and `vertex_count` are
// read from the group itself, so no `[[seam::indices(N)]]` can name the run and
// the entry hands the whole array in with its length. The two indirections that
// follow, a member's `index` into the positions and into the props, are checked
// against that length and against the vertex count before either is used.
//
// A GROUP WHOSE MEMBERS ARE ALL MASSLESS, or that has no perpendicular extent
// about its axis, is not an error: `torque_centroid_finish` leaves the centroid
// unscaled and `torque_rperp_finish` returns zero, and a zero normalization is
// how such a group receives no force at all.
[[seam::device_fn]] inline void torque_group_frame(
    const TorqueGroup &group,
    const TorqueVertex *member, unsigned member_count,
    const Vec3f *position, unsigned vertex_count,
    const VertexProp *prop,
    TorqueGroupResult *result, DiagHandle diag) {
    const unsigned start = group.vertex_start;
    const unsigned last = start + group.vertex_count;
    // THE RUN AND THE HINT ARE DATA, so the thread guard says nothing about
    // either: a group names its own span of the member array and its own hint
    // vertex, and a scene that names one past the end reads zero on Metal and
    // faults on nothing. Both are checked once here, before any walk, so a
    // malformed group stops the run rather than producing a plausible frame.
    DIAG_ASSERT4(diag, last <= member_count && start <= last,
                static_cast<float>(start), static_cast<float>(last),
                static_cast<float>(member_count), 0.0f);
    DIAG_ASSERT4(diag, group.hint_vertex < vertex_count,
                static_cast<float>(group.hint_vertex),
                static_cast<float>(vertex_count), 0.0f, 0.0f);
    if (last > member_count || start > last ||
        group.hint_vertex >= vertex_count) {
        return;
    }

    Vec3f center = Vec3f::Zero();
    float total_mass = 0.0f;
    for (unsigned k = start; k < last; ++k) {
        const unsigned index = member[k].index;
        // A MEMBER'S OWN INDEX IS DATA TOO, and it reaches two arrays. The
        // check is inside the walk because the run is variable: there is no
        // one place before it where every index this group will read is known.
        DIAG_ASSERT4(diag, index < vertex_count, static_cast<float>(index),
                    static_cast<float>(vertex_count), static_cast<float>(k),
                    0.0f);
        if (index >= vertex_count) {
            return;
        }
        // COPIED INTO A LOCAL FIRST, and this is not a style choice. On Metal
        // `position[index]` is a `const device` lvalue and every body below
        // takes its positions as `[[seam::thread]]` references, so the value
        // must cross address spaces here. A generated entry point does this
        // copy for its `[[seam::gather]]` arguments; a body that walks an array
        // ITSELF, as this one must because the run is data, owes the same copy.
        // CUDA and the host have one address space and compile without it.
        const Vec3f member_position = position[index];
        torque_centroid_step(member_position, prop[index].mass, center,
                                 total_mass);
    }
    torque_centroid_finish(center, total_mass);

    float covariance[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned k = start; k < last; ++k) {
        const Vec3f member_position = position[member[k].index];
        const Vec3f offset = (member_position - center).template cast<float>();
        torque_covariance_step(offset, covariance);
    }
    torque_covariance_finish(covariance, group.vertex_count);

    // The hint vertex resolves the sign ambiguity of a principal axis: an axis
    // and its negative are the same principal direction, so without it the
    // commanded torque could reverse between two frames whose covariance barely
    // differs.
    const Vec3f hint_position = position[group.hint_vertex];
    const Vec3f hint = (hint_position - center).template cast<float>();
    Vec3f axis;
    torque_axis(covariance, group.axis_component, hint, axis);

    float r_perp_sq_sum = 0.0f;
    for (unsigned k = start; k < last; ++k) {
        const Vec3f member_position = position[member[k].index];
        const Vec3f radius = (member_position - center).template cast<float>();
        torque_rperp_step(radius, axis, r_perp_sq_sum);
    }

    result[0].center = center;
    result[0].axis = axis;
    result[0].inv_r_perp_sq_sum = torque_rperp_finish(r_perp_sq_sum);
}

// The entry. One thread per torque group.
//
// THE GROUP IS THE ELEMENT AND THE RESULT IS ITS OWN SLOT, so the group is a
// `[[seam::gather]]` and the result a `[[seam::stride(1)]]`; every other array
// is flat and bound, because what this kernel reads out of them is decided by
// the group's own fields rather than by the thread index.
//
// IT TAKES `[[seam::diag]]` FOR THE THREE INDIRECTIONS THE THREAD GUARD CANNOT
// COVER: the member run, the hint vertex and each member's vertex index are all
// data. On CUDA a bad one trips a live release assert; on Metal it would read
// zero and produce a frame that looks like an answer.
[[seam::entry(count)]] void torque_group_frame(
    const TorqueGroup *group,
    const TorqueVertex *member,
    unsigned member_count,
    const Vec3f *position,
    unsigned vertex_count,
    const VertexProp *prop,
    [[seam::stride(1)]] 
    TorqueGroupResult *result,
    DiagHandle diag,
    unsigned count);
