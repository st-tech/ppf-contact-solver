// File: shell_strain.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py
// renders it into the three forms the three compilers read, and the build hands
// each compiler its own form. The two facts a backend cannot infer are written
// as C++ attributes: `[[seam::device_fn]]` is the execution space, and
// `[[seam::thread]]` is the address space of a reference parameter. MSL
// requires the second on every reference and pointer type; CUDA and the host
// have one address space and are handed the same declarations with it removed.
//
// The division goes through `fmath::div`, which each backend prologue defines
// with that backend's spelling: a plain quotient under nvcc and on the host,
// `metal::precise::divide` under MSL, where the default quotient is not
// correctly rounded.
#include "../barrier/cubic.hpp"
#include "../barrier/logarithm.hpp"
#include "../barrier/quadratic.hpp"
#include "../barrier/contact_barrier.kernel.cpp"
#include "../csrmat/fixed_csr.kernel.cpp"
#include "strain_toi.kernel.cpp"
// THE THREE `shell_strain_embed` ADDS. The deformation gradient and the SVD
// arrive with `strain_toi`, and the CSR read and push with `fixed_csr`; these
// are the spectral force and Hessian, the two conversions back to position
// space, and the three-vertex force scatter.
#include "../eigenanalysis/face_eigenanalysis.kernel.cpp"
#include "../utility/face_convert.kernel.cpp"
#include "../utility/face_scatter.kernel.cpp"

[[seam::device_fn]] inline DiffTable2
shell_strain_diff_table(const Vec2f &shifted_sigma,
                            float limit, Barrier barrier) {
    DiffTable2 table;
    table.deda = Vec2f::Zero();
    table.d2ed2a = Mat2x2f::Zero();
    for (unsigned i = 0; i < 2; ++i) {
        const float strain = shifted_sigma[i];
        if (strain > 0.0f) {
            const float gap = limit - strain;
            table.deda[i] =
                -barrier_gradient(gap, limit, 0.0f, barrier);
            table.d2ed2a(i, i) =
                barrier_curvature(gap, limit, 0.0f, barrier);
        }
    }
    return table;
}

[[seam::device_fn]] inline float
shell_strain_energy(const Vec2f &shifted_sigma, float limit,
                        Barrier barrier) {
    float result = 0.0f;
    for (unsigned i = 0; i < 2; ++i) {
        const float strain = shifted_sigma[i];
        if (strain > 0.0f) {
            result += barrier_energy(
                limit - strain, limit, 0.0f, barrier);
        }
    }
    return result;
}

// The elasticity-inclusive dynamic stiffness the strain-limit barrier above is
// scaled by. Two terms: the face's own assembled Hessian contracted along the
// centered shape of the face, which carries the surrounding elasticity into the
// barrier's scale, plus an inertia term that diverges as the largest stretch
// closes on the limit.
//
// VALUES, not containers. `local_hessian` is the 9x9 the caller gathers from
// the fixed CSR at this face's three vertices, block (ii, jj) holding the
// coupling between local vertex ii and local vertex jj in the face's own
// ordering. `x0`, `x1` and `x2` are the positions the ROW OF THE MATRIX belongs
// to, which is the pose the step began from rather than the Newton iterate; the
// caller decides that, and `shell_strain_embed` below passes the START-OF-STEP
// positions. `shifted_sigma` is the deformation gradient's singular
// values minus one, and `limit` is the shrink-adjusted strain limit
// (shell_effective_strain_limit), so `limit - max(shifted_sigma)` is the
// stretch the face has left before it reaches the limit.
//
// Note that `limit` is NOT the ghat the barrier itself is evaluated against:
// the table above takes the authored `strainlimit` while this takes the value
// corrected for shrink, and a scene with shrink_x or shrink_y below one has the
// two differing. Passing one where the other belongs is a scale error that no
// test on a shrink-free scene can see.
[[seam::device_fn]] inline float shell_strain_stiffness(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2,
    const Mat9x9f &local_hessian, float mass,
    const Vec2f &shifted_sigma, float limit) {
    // The centroid is formed first and the three offsets are taken as
    // differences from it, so the quantities the Hessian is built from stay at
    // triangle scale rather than at the scale of the coordinates themselves.
    const float third(1.0f / 3.0f);
    const Vec3f center = third * x0 + third * x1 + third * x2;
    Mat3x3f centered;
    centered << (x0 - center).cast<float>(), //
        (x1 - center).cast<float>(),         //
        (x2 - center).cast<float>();
    Vec9f shape;
    for (unsigned element = 0; element < 9; ++element) {
        shape[element] = centered.m[element];
    }
    const float gap = limit - shifted_sigma.maxCoeff();
    return shape.dot(local_hessian * shape) + fmath::div(mass, gap * gap);
}

// The barrier's derivative pair for ONE face, at the AUTHORED limit, with the
// no-limit case decided here rather than by a launcher.
//
// THE GATE IS THE POINT OF THIS WRAPPER. A face whose authored `strainlimit` is
// not positive has no strain limit at all, and its derivative pair is zero; the
// table above must not be evaluated there, because it would form
// `gap = limit - strain` against a non-positive limit and hand the barrier a
// gap on the wrong side of the surface. That decision was spelled in the CPU
// backend's hand-written launcher, which is a branch on a physical quantity
// inside a backend entry point; it is here so the three compilers build it from
// one set of bytes.
//
// THE COMPARISON IS NEGATED ON PURPOSE and must stay that way: `!(limit > 0)`
// admits a NaN limit to the zero branch, where `limit <= 0` would send it to
// the table and propagate the NaN into the assembled Hessian.
//
// TWO OUTPUTS THROUGH REFERENCES, which is the shape `svd3x2`'s entry
// states: a gather hands the body `buffer[index]`, an lvalue, so a non-const
// gather IS the write and the pair needs no scatter. `barrier` arrives as the
// `unsigned` the record carries, because a record field is a scalar the driver
// fills and `Barrier` is this tree's own enumeration; the cast is here so no
// backend spells it.
[[seam::device_fn]] inline void shell_strain_diff_table_gated(
    const Vec2f &shifted_sigma, float authored_limit,
    unsigned barrier, Vec2f &deda,
    Mat2x2f &d2ed2a) {
    if (!(authored_limit > 0.0f)) {
        deda = Vec2f::Zero();
        d2ed2a = Mat2x2f::Zero();
        return;
    }
    const DiffTable2 table = shell_strain_diff_table(
        shifted_sigma, authored_limit, static_cast<Barrier>(barrier));
    deda = table.deda;
    d2ed2a = table.d2ed2a;
}

// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `shell_strain_diff_table_gated_entry` shim a host C++
// compiler compiles, and the Rust `#[repr(C)]` twin `src/driver/kernels.rs`
// fills.
//
// FOUR ELEMENT GATHERS AND NO SCATTER. Two are read and two are written, and
// the two written ones are non-const for exactly that reason. The pod sizes pin
// the element widths in every C++ rendering, so a compiler laying `Vec2f` or
// `Mat2x2f` out differently fails to compile rather than reading the wrong
// bytes.
// The same table with the authored limit read from the face's own records.
//
// THE GATE IS THE LIMITER'S OWN AND IT IS NOT THE MEMBRANE'S. There is NO
// `collider` test here: a spring-held collider's vertices are free, and the
// limit it was authored with is what stops its own mesh from stretching past
// it. The two exclusions that do apply are `fixed`, whose DOF is eliminated,
// and `rest_excluded`.
//
// WHAT IT REPLACES is a host loop that read the scene's material table, wrote
// `material.strainlimit` into a staged array for every face and zeroed the rest,
// and uploaded the array every pass. The limit is a material constant reached
// through the face's own `param_index`, so the device can read it out of two
// records it already holds.
[[seam::device_fn]] inline void shell_strain_diff_table_from_records(
    const Vec2f &shifted_sigma,
    const FaceProp &prop,
    const FaceParam *face_param, unsigned barrier,
    float largest_shifted, Vec2f &deda,
    Mat2x2f &d2ed2a, float &live) {
    float authored_limit = 0.0f;
    if (!prop.fixed && !prop.rest_excluded) {
        authored_limit = face_param[prop.param_index].strainlimit;
    }
    // WHICH FACES THE LATER PASSES ACT ON, written here because this is the
    // first pass that holds BOTH halves of the gate.
    //
    // The two halves are `!prop.fixed && !prop.rest_excluded &&
    // fparam.strainlimit > 0.0f`, and then, once the SVD has run,
    // `largest_shifted > 0.0f`. The first is the face's own records and the
    // second is the SVD's, so no pass before this one can answer it and no pass
    // after it needs to ask twice.
    live = (authored_limit > 0.0f && largest_shifted > 0.0f) ? 1.0f : 0.0f;
    shell_strain_diff_table_gated(shifted_sigma, authored_limit, barrier, deda,
                                  d2ed2a);
}

[[seam::entry(count)]] void shell_strain_diff_table_from_records(
    const Vec2f *shifted_sigma,
    const FaceProp *prop,
    const FaceParam *face_param,
    unsigned barrier,
    const float *largest_shifted,
    Vec2f *deda,
    Mat2x2f *d2ed2a,
    float *live,
    unsigned count);

[[seam::entry(count)]] void shell_strain_diff_table_gated(
    const Vec2f *shifted_sigma,
    const float *authored_limit,
    unsigned barrier,
    Vec2f *deda,
    Mat2x2f *d2ed2a,
    unsigned count);

// THE GATHER HAPPENS HERE RATHER THAN IN THE CALLER, plus its gate: the 3x3
// twin of `strainlimiting/rod_strain.kernel.cpp`'s, at 3x3 blocks over a 9x9
// instead of 2x2 over a 6x6. That file states why the face array is named TWICE
// in this entry's record, and the reason is identical here:
// `[[seam::indices(3)]]` consumes the three slots to hand this body its three
// positions, and the CSR lookup needs the same three as ROW AND COLUMN values.
// Both fields are filled from one array at the call site.
//
// PASSING THE 9x9 IN INSTEAD WOULD COST 81 FLOATS PER FACE of caller-side
// storage to say the same thing, which is why the gather is here rather than in
// a pass of its own.
//
// THE COMPARISON IS NEGATED ON PURPOSE, as everywhere else in this family:
// `!(limit > 0)` sends a NaN to the zero branch.
[[seam::device_fn]] inline float shell_strain_stiffness_gated(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2,
    const unsigned *face_slots, unsigned element,
    const unsigned *index,
    const unsigned *offset,
    const float *value, unsigned row_count, float mass,
    const Vec2f &shifted_sigma, float limit) {
    if (!(limit > 0.0f)) {
        return 0.0f;
    }
    Mat9x9f local_hessian = Mat9x9f::Zero();
    for (unsigned ii = 0; ii < 3; ++ii) {
        for (unsigned jj = 0; jj < 3; ++jj) {
            local_hessian.template block<3, 3>(3 * ii, 3 * jj) =
                fixed_csr_read(index, offset, value, row_count,
                                   face_slots[3 * element + ii],
                                   face_slots[3 * element + jj]);
        }
    }
    return shell_strain_stiffness(x0, x1, x2, local_hessian, mass,
                                      shifted_sigma, limit);
}

// The same stiffness with the MASS and the SHRINK-CORRECTED limit read from the
// face's own records.
//
// TWO STAGED ARRAYS COLLAPSE INTO TWO RECORDS. The mass is a `FaceProp` field
// and the effective limit is `shell_effective_strain_limit` of three
// `FaceParam` fields, so the host loop that wrote both into staged arrays every
// pass was recomputing on the CPU what the device can read and evaluate from
// records it already holds. The gate is the limiter's own, which has NO
// `collider` test.
[[seam::device_fn]] inline float shell_strain_stiffness_from_records(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2,
    const unsigned *face_slots, unsigned element,
    const unsigned *index,
    const unsigned *offset,
    const float *value, unsigned row_count,
    const FaceProp &prop,
    const FaceParam *face_param,
    const Vec2f &shifted_sigma) {
    if (prop.fixed || prop.rest_excluded) {
        return 0.0f;
    }
    const FaceParam param = face_param[prop.param_index];
    if (!(param.strainlimit > 0.0f)) {
        return 0.0f;
    }
    const float limit = shell_effective_strain_limit(
        param.strainlimit, param.shrink_x, param.shrink_y);
    return shell_strain_stiffness_gated(x0, x1, x2, face_slots, element, index,
                                        offset, value, row_count, prop.mass,
                                        shifted_sigma, limit);
}

[[seam::entry(count, element)]] void shell_strain_stiffness_from_records(
    [[seam::through]] const Vec3f *x,
    [[seam::indices(3)]] const unsigned *face,
    [[seam::bound]] unsigned vertex_count,
    const unsigned *face_slots,
    unsigned element,
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    unsigned row_count,
    const FaceProp *prop,
    const FaceParam *face_param,
    const Vec2f *shifted_sigma,
    float *stiffness,
    unsigned count);

// Three positions through the face's own index triple with the bound beside
// them, the same array again as a base pointer for the CSR lookup, the pattern
// and the value array as base pointers because the row's own slot range is the
// lookup's business, three element gathers and one element scatter.
//
// `x` IS THE START-OF-STEP POSE, not the Newton iterate: the row of the matrix
// being contracted belongs to that pose, and the caller decides it.
[[seam::entry(count, element)]] void shell_strain_stiffness_gated(
    [[seam::through]] const Vec3f *x,
    [[seam::indices(3)]] const unsigned *face,
    [[seam::bound]] unsigned vertex_count,
    const unsigned *face_slots,
    unsigned element,
    const unsigned *index,
    const unsigned *offset,
    const float *value,
    unsigned row_count,
    const float *mass,
    const Vec2f *shifted_sigma,
    const float *effective_limit,
    float *stiffness,
    unsigned count);

// THE WHOLE LIMITER IN ONE KERNEL, which is what
// `embed_strainlimiting_force_hessian` is: one `DISPATCH_START(shell_face_count)`
// running the gate, the deformation gradient, the SVD, the diff table, the
// stiffness, the force and Hessian and both embeds. Splitting those steps into
// a dispatch each would cost fifteen of them over sixteen per-face staging
// arrays, materializing the Hessian three times on its way to the CSR, 36 then
// 81 then 81 floats a face; here nothing per-face leaves the thread.
//
// THE STIFFNESS IS TAKEN FROM THE SHIFTED SINGULAR VALUES, BEFORE THE RESTORE,
// AND THAT IS A CORRECTNESS CONSTRAINT RATHER THAN AN ORDERING CONVENIENCE.
// `shell_strain_stiffness_gated` takes `shifted_sigma`, and
// `shell_strain_restore_sigma` runs only after it. Reading the RESTORED sigma
// there moves the stiffness divisor `d` by exactly 1.0 and the term by
// `(limit - s)^2 / (limit - s - 1)^2`, on every scene, shrink-free ones
// included. The two locals below are named apart so a later edit cannot
// silently pass the wrong one.
//
// TWO POSES AND TWO MATRICES. The gradient is taken at the ITERATE and the
// stiffness at the START OF STEP, each read through the face's own slots; and
// the stiffness READS the snapshot passed as `reference_value` while the push
// WRITES the live `fixed_value`.
//
// TWO LIMITS ONE LETTER APART. The barrier's ghat is the AUTHORED
// `strainlimit` and the stiffness divisor is
// `shell_effective_strain_limit`, the authored limit corrected for shrink; they
// are equal exactly when both shrink factors are one. Each composed body reads
// its own, so neither is passed here.
//
// NO DIAGNOSTIC. Nothing on this path asserts: the stiffness returns 0.0 on a
// non-positive limit through a negated comparison, so a NaN takes the zero
// branch, and the entry is `decl_generated` rather than `decl_generated_diag`.
[[seam::device_fn]] inline void shell_strain_embed(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2,
    const Vec3f &current0,
    const Vec3f &current1,
    const Vec3f &current2,
    const unsigned *face_slots,
    const Mat2x2f &inverse_rest,
    const FaceProp &prop,
    const FaceParam *face_param, unsigned barrier,
    float eiganalysis_eps,
    const unsigned *reference_index,
    const unsigned *reference_offset,
    const float *reference_value, unsigned reference_rows,
    compute::atomic_float_t *force,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    compute::atomic_float_t *fixed_value, unsigned row_count,
    compute::atomic_uint_t *refused,
    unsigned *witness, unsigned element) {
    // THE FIRST HALF OF THE GATE, off the face's own records. There is no
    // collider test here and that is not an omission: the limiter's gate is
    // `!fixed && !rest_excluded && strainlimit > 0` and no more, where the
    // membrane's gate also excludes a collider.
    if (prop.fixed || prop.rest_excluded) {
        return;
    }
    const FaceParam param = face_param[prop.param_index];
    if (!(param.strainlimit > 0.0f)) {
        return;
    }
    const Mat3x2f gradient_f =
        face_deformation_gradient(x0, x1, x2, inverse_rest);
    Mat3x2f u;
    Vec2f shifted_sigma;
    Mat2x2f vt;
    float largest_shifted = 0.0f;
    svd3x2_shifted(gradient_f, u, shifted_sigma, vt, largest_shifted);
    // THE SECOND HALF, which no pass before the SVD can answer.
    if (!(largest_shifted > 0.0f)) {
        return;
    }
    Vec2f deda;
    Mat2x2f d2ed2a;
    shell_strain_diff_table_gated(shifted_sigma, param.strainlimit, barrier,
                                  deda, d2ed2a);
    // BEFORE THE RESTORE, on the shifted values, from the START-OF-STEP pose
    // and the SNAPSHOT matrix.
    const float limit = shell_effective_strain_limit(
        param.strainlimit, param.shrink_x, param.shrink_y);
    const float stiffness = shell_strain_stiffness_gated(
        current0, current1, current2, face_slots, element, reference_index,
        reference_offset, reference_value, reference_rows, prop.mass,
        shifted_sigma, limit);
    const Vec2f restored_sigma = shell_strain_restore_sigma(shifted_sigma);
    const Mat3x2f force_f = face_spectral_force(deda, u, vt);
    const Mat6x6f hessian_f =
        face_spectral_hessian(deda, d2ed2a, u, restored_sigma, vt,
                              eiganalysis_eps);
    Mat3x3f gradient_x = face_convert_force(force_f, inverse_rest);
    Mat9x9f hessian_x = face_convert_hessian(hessian_f, inverse_rest);
    gradient_x *= stiffness;
    hessian_x *= stiffness;
    Vec3u triple;
    unsigned slots[3];
    for (unsigned k = 0; k < 3u; ++k) {
        slots[k] = face_slots[3u * element + k];
        triple[k] = slots[k];
    }
    face_atomic_embed_force(triple, gradient_x, force);
    fixed_push_blocks_thread(slots, hessian_x.m, 3u, fixed_index, fixed_offset,
                             fixed_value, row_count, refused, witness);
}

[[seam::entry(count, element)]] void shell_strain_embed(
    [[seam::through]] const Vec3f *x,
    [[seam::through]] const Vec3f *current,
    [[seam::indices(3)]] const unsigned *face,
    [[seam::bound]] unsigned vertex_count,
    const unsigned *face_slots,
    const Mat2x2f *inverse_rest,
    const FaceProp *prop,
    const FaceParam *face_param,
    unsigned barrier,
    float eiganalysis_eps,
    const unsigned *reference_index,
    const unsigned *reference_offset,
    const float *reference_value,
    unsigned reference_rows,
    compute::atomic_float_t *force,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    compute::atomic_float_t *fixed_value,
    unsigned row_count,
    compute::atomic_uint_t *refused,
    unsigned *witness,
    unsigned element,
    unsigned count);

