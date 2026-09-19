// File: rod_strain.kernel.cpp
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
// The seam names this body calls arrive from the includer: seam/seam.hpp under
// nvcc and on the host, which data.hpp includes first, and the prologue in
// ppf-cts-compute/metal/shader_compiler.mm under MSL. The division goes
// through `fmath::div` rather than the operator, because MSL spells the
// correctly rounded quotient `precise::divide` and the plain operator is a
// different function there.
#include "../barrier/cubic.hpp"
#include "../barrier/logarithm.hpp"
#include "../barrier/quadratic.hpp"
#include "../barrier/contact_barrier.kernel.cpp"
#include "../contact/distance.hpp"
#include "../csrmat/fixed_csr.kernel.cpp"

[[seam::device_fn]] inline bool rod_strain_force_hessian(
    const Vec3f &x0, const Vec3f &x1,
    float rest_length, float limit, Barrier barrier,
    Mat3x2f &force, Mat6x6f &hessian,
    float &strain) {
    const Vec3f difference = proximity::difference<float, float>(x1, x0);
    const float length = difference.norm();
    strain = fmath::div(length, rest_length) - 1.0f;
    force = Mat3x2f::Zero();
    hessian = Mat6x6f::Zero();
    if (strain <= 0.0f) {
        return false;
    }
    const float gap = limit - strain;
    const float gradient =
        -barrier_gradient(gap, limit, 0.0f, barrier);
    const float curvature =
        barrier_curvature(gap, limit, 0.0f, barrier);
    // COMPONENT-WISE DIVISION, NOT A PRECOMPUTED RECIPROCAL: `difference /
    // length` rounds once per component, where multiplying by a precomputed
    // `1 / length` would round the reciprocal first and then each product.
    const Vec3f normal = difference / length;
    Vec6f jacobian;
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        jacobian[dimension] = -fmath::div(normal[dimension], rest_length);
        jacobian[3 + dimension] = fmath::div(normal[dimension], rest_length);
    }
    const Mat3x3f projection =
        Mat3x3f::Identity() - normal * normal.transpose();
    const float geometric_scale = fmath::div(1.0f, length * rest_length);
    const float positive_curvature = fmath::max(0.0f, curvature);
    const float positive_gradient = fmath::max(0.0f, gradient);
    hessian =
        positive_curvature * (jacobian * jacobian.transpose());
    for (unsigned a = 0; a < 2; ++a) {
        for (unsigned b = 0; b < 2; ++b) {
            const float sign = a == b ? 1.0f : -1.0f;
            hessian.template block<3, 3>(3 * a, 3 * b) +=
                sign * positive_gradient * geometric_scale * projection;
        }
    }
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        force(dimension, 0) = gradient * jacobian[dimension];
        force(dimension, 1) = gradient * jacobian[3 + dimension];
    }
    return true;
}

// THE TWO GATES BELONG HERE AND NOT IN THE CALLER, and so do the zeroed outputs
// that go with them. An edge whose authored limit is not positive has no
// strain-limit term, and one whose rest length is not positive would be divided
// by it: the body above divides the current length by `rest_length` with no
// test of its own. Both are branches on a physical quantity, so both belong
// here.
//
// A FALSE IS A SKIP, NOT AN ERROR. The caller walks the verdicts and takes only
// the edges that produced a term. All three outputs are zeroed on the refused
// path, so an edge the caller skips cannot carry stale bytes into a scatter,
// which is what the body's own early return already promises for the case it
// decides.
//
// THE COMPARISONS ARE NEGATED ON PURPOSE and must stay that way: `!(x > 0)`
// admits a NaN to the zero branch, where `x <= 0` would send it into the
// barrier and propagate it into the assembled Hessian.
//
// `barrier` ARRIVES AS THE `unsigned` THE RECORD CARRIES, because a record field
// is a scalar the driver fills and `Barrier` is this tree's own enumeration; the
// cast is here so no backend spells it. `strainlimiting/shell_strain.kernel.cpp`
// states the same rule at its own composition.
[[seam::device_fn]] inline bool rod_strain_force_hessian_gated(
    const Vec3f &x0, const Vec3f &x1,
    float rest_length, float limit, unsigned barrier,
    Mat3x2f &force, Mat6x6f &hessian,
    float &strain) {
    if (!(limit > 0.0f) || !(rest_length > 0.0f)) {
        force = Mat3x2f::Zero();
        hessian = Mat6x6f::Zero();
        strain = 0.0f;
        return false;
    }
    return rod_strain_force_hessian(x0, x1, rest_length, limit,
                                        static_cast<Barrier>(barrier), force,
                                        hessian, strain);
}

// TWO POSITIONS THROUGH THE EDGE'S OWN INDEX PAIR, two element gathers read, the
// barrier kind as a scalar every thread shares, three non-const gathers written,
// and the verdict as the scatter that carries the body's return value.
//
// THE VERDICT IS `unsigned`, NOT A BYTE. A scatter takes one of the three scalar
// types a record field may hold, and `shell_bend_force_hessian_verdict`
// spells its own the same way; the driver's buffer is a `Vec<u32>` to match.
[[seam::entry(count)]] void rod_strain_force_hessian_gated(
    [[seam::through]] const Vec3f *x,
    [[seam::indices(2)]] const unsigned *edge,
    [[seam::bound]] unsigned vertex_count,
    const float *rest_length,
    const float *limit,
    unsigned barrier,
    Mat3x2f *force,
    Mat6x6f *hessian,
    float *strain,
    [[seam::scatter]] unsigned *ok,
    unsigned count);

// The elasticity-inclusive dynamic stiffness the rod strain-limit barrier above
// is scaled by, the two-node counterpart of shell_strain_stiffness. Two
// terms: the segment's own assembled Hessian contracted along the centered
// shape of the segment, which carries the surrounding elasticity into the
// barrier's scale, plus an inertia term that diverges as the stretch closes on
// the limit.
//
// VALUES, not containers. `local_hessian` is the 6x6 the caller gathers from
// the fixed CSR at this segment's two vertices, block (ii, jj) holding the
// coupling between local vertex ii and local vertex jj in the segment's own
// ordering. `x0` and `x1` are the positions the ROW OF THE MATRIX belongs to,
// which is the pose the step began from rather than the Newton iterate; the
// caller decides that, and `driver::assemble::rod_strain` passes
// `state.positions`, the committed pose. `strain` is length/rest_length - 1, the
// value
// rod_strain_force_hessian reports, and `limit` is the segment's authored
// strain limit, so `limit - strain` is the stretch it has left.
//
// Unlike the shell, there is one limit and not two: a rod carries no shrink
// factors, so the barrier's ghat and this divisor are the same number.
[[seam::device_fn]] inline float rod_strain_stiffness(
    const Vec3f &x0, const Vec3f &x1,
    const Mat6x6f &local_hessian, float mass, float strain,
    float limit) {
    // The center is formed first and the two offsets are taken as differences
    // from it, so the quantities the Hessian is built from stay at segment
    // scale rather than at the scale of the coordinates themselves.
    // Not named `half`: MSL reserves that as its 16-bit float type, so the
    // declaration would parse as a type name there and the body would stop
    // compiling under the Metal shader compiler.
    const float midpoint_weight(0.5f);
    const Vec3f center = midpoint_weight * x0 + midpoint_weight * x1;
    Mat3x2f centered;
    centered << (x0 - center).cast<float>(), //
        (x1 - center).cast<float>();
    Vec6f shape;
    for (unsigned element = 0; element < 6; ++element) {
        shape[element] = centered.m[element];
    }
    const float gap = limit - strain;
    return shape.dot(local_hessian * shape) + fmath::div(mass, gap * gap);
}

// THE GATHER BELONGS HERE AND NOT IN THE CALLER, plus its gate. The 6x6 above
// is a VALUE the caller assembles from the fixed CSR at this segment's two
// vertices, and assembling it is four `fixed_csr_read` lookups over the
// segment's own two slots, which is arithmetic rather than orchestration and so
// belongs here.
//
// THE EDGE ARRAY IS NAMED TWICE IN THIS ENTRY'S RECORD AND THAT IS DELIBERATE.
// The entry needs it in two roles: as the index list whose two slots are read
// and CHECKED AGAINST THE BOUND, which is what hands this body its two
// positions, and as a base pointer this body walks itself, because the CSR
// lookup takes the slots as ROW AND COLUMN indices and `[[seam::indices]]`
// consumes them. There is no attribute that forwards a checked slot as a value.
// The alternative is to make the positions base pointers too and index them
// here, which is expressible and LOSES THE BOUND, so the redundant field is the
// cheaper of the two. Both fields are filled from one array at the call site.
//
// THE COMPARISON IS NEGATED ON PURPOSE, for the reason the wrappers in
// `strain_toi.kernel.cpp` state: `!(limit > 0)` sends a NaN to the zero branch.
[[seam::device_fn]] inline float rod_strain_stiffness_gated(
    const Vec3f &x0, const Vec3f &x1,
    const unsigned *edge_slots, unsigned element,
    const unsigned *index,
    const unsigned *offset,
    const float *value, unsigned row_count, float mass,
    float strain, float limit) {
    if (!(limit > 0.0f)) {
        return 0.0f;
    }
    Mat6x6f local_hessian = Mat6x6f::Zero();
    for (unsigned ii = 0; ii < 2; ++ii) {
        for (unsigned jj = 0; jj < 2; ++jj) {
            local_hessian.template block<3, 3>(3 * ii, 3 * jj) =
                fixed_csr_read(index, offset, value, row_count,
                                   edge_slots[2 * element + ii],
                                   edge_slots[2 * element + jj]);
        }
    }
    return rod_strain_stiffness(x0, x1, local_hessian, mass, strain, limit);
}

// Two positions through the edge's own index pair with the bound beside them,
// the same array again as a base pointer for the CSR lookup, the pattern and
// the value array as base pointers because the row's own slot range is the
// lookup's business, three element gathers and one element scatter.
[[seam::entry(count, element)]] void rod_strain_stiffness_gated(
    [[seam::through]] const Vec3f *x,
    [[seam::indices(2)]] const unsigned *edge,
    [[seam::bound]] unsigned vertex_count,
    const unsigned *edge_slots,
    unsigned element,
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    unsigned row_count,
    const float *mass,
    const float *strain,
    const float *limit,
    float *stiffness,
    unsigned count);
