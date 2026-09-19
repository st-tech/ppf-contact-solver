// File: face_force.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the three forms the three compilers read. The two facts a backend cannot
// infer are written as C++ attributes: `[[seam::device_fn]]` is the execution
// space and `[[seam::thread]]` is the address space MSL requires on every
// reference.
//
// DIAG_ASSERT4 IS NOT A PLATFORM BRANCH AND STAYS A MACRO, as in
// contact/aabb_traversal.kernel.cpp: it records __FILE__ and __LINE__ of the
// failing check, which no function can read for its caller, and each backend
// defines it over its own diagnostic channel.
//
// THE SHELL MEMBRANE AND THE PER-FACE PRESSURE BLOCK, the two terms a triangle
// contributes to the Newton system. What is here is every float operation AND
// the two scatters that place each term: the three-node force embed
// (`utility/face_scatter.kernel.cpp`) and the fixed matrix's atomic block push
// (`csrmat/fixed_csr.kernel.cpp`). What stays in the caller is the walk: which
// faces are in this iteration and what material each carries.

#include "../csrmat/fixed_csr.kernel.cpp"
#include "../utility/face_convert.kernel.cpp"
#include "../utility/face_damping.kernel.cpp"
#include "../utility/face_deformation.kernel.cpp"
#include "../utility/face_scatter.kernel.cpp"
#include "../utility/svd3x2.kernel.cpp"
#include "../utility/svd3x3.kernel.cpp"
#include "../linalg/eigsolve.hpp"
#include "../eigenanalysis/face_eigenanalysis.kernel.cpp"
#include "model/arap.hpp"
#include "model/baraffwitkin.hpp"
#include "model/snhk.hpp"
#include "model/stvk.hpp"
#include "elastic_model.kernel.cpp"
#include "model/face_pressure.kernel.cpp"

// The membrane term: the elastic gradient and its PSD-projected Hessian for one
// triangle, in element (x) space, with the Rayleigh damping block folded in.
//
// RETURNS FALSE, AND ONLY FALSE, FOR A MODEL ID NO BRANCH BELOW HANDLES. The
// caller must then assemble NOTHING for the face, the pressure block included:
// `face_elastic_embed` below returns on that `false` before either embed runs,
// so an unrecognized id produces no force row and no CSR block rather than a
// half-assembled face.
//
// THE MODEL ID IS VALIDATED ONLY WHERE IT IS DISPATCHED, never up front, and
// that is a rule rather than an accident. A PDRD body's faces carry
// `ELASTIC_MODEL_PDRD` with `mu == 0`, because a rigid body's shape is held
// by the reduced rigid solve and not by an elastic energy, so they leave at the
// `mu > 0.0f` gate before any dispatch and must not be refused. Validating up
// front would reject those PDRD faces along with a genuinely unrecognized id;
// validating nowhere would route an unrecognized ELASTIC model to SNHk and
// assemble the wrong material silently.
//
// BARAFFWITKIN NEVER FORMS THE SVD. Keeping it out of the spectral branch is
// deliberate: it is the default cloth path, and the eigensolve's registers are
// not free.
template <typename D>
[[seam::device_fn]] inline bool face_elastic_force_hessian(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2,
    const Vec3f &current0,
    const Vec3f &current1,
    const Vec3f &current2,
    const Mat2x2f &inverse_rest, unsigned model, float mu,
    float lambda, float mass, float deform_damping, float dt,
    float eigenanalysis_eps, unsigned element,
    Mat3x3f &gradient, Mat9x9f &hessian,
    D diag) {
    if (mu > 0.0f) {
        Mat3x2f deformation =
            face_deformation_gradient(x0, x1, x2, inverse_rest);
        if (model == ELASTIC_MODEL_BARAFF_WITKIN) {
            Mat3x2f gradient_f;
            Mat6x6f hessian_f;
            BaraffWitkin::material(deformation, mu, lambda, gradient_f,
                                   hessian_f);
            gradient += mass * face_convert_force(gradient_f, inverse_rest);
            hessian +=
                mass * face_convert_hessian(hessian_f, inverse_rest);
        } else {
            Mat3x2f u;
            Vec2f sigma;
            Mat2x2f vt;
            svd3x2(deformation, u, sigma, vt);
            DiffTable2 table;
            if (model == ELASTIC_MODEL_ARAP) {
                table = ARAP::make_diff_table2(sigma, mu, lambda);
            } else if (model == ELASTIC_MODEL_STVK) {
                table = StVK::make_diff_table2(sigma, mu, lambda);
            } else if (model == ELASTIC_MODEL_SNHK) {
                table = SNHk::make_diff_table2(sigma, mu, lambda);
            } else {
                DIAG_ASSERT4(diag, false, static_cast<float>(model),
                            static_cast<float>(element), mu, lambda);
                return false;
            }
            Mat3x2f gradient_f = face_spectral_force(table.deda, u, vt);
            Mat6x6f hessian_f = face_spectral_hessian(
                table.deda, table.d2ed2a, u, sigma, vt, eigenanalysis_eps);
            gradient += mass * face_convert_force(gradient_f, inverse_rest);
            hessian +=
                mass * face_convert_hessian(hessian_f, inverse_rest);
        }
        face_add_stiffness_damping(x0, x1, x2, current0, current1,
                                       current2, deform_damping, dt, gradient,
                                       hessian);
    }
    return true;
}

// The per-face pressure block, gradient and analytically PSD-projected Hessian.
//
// THE ABSOLUTE POSITION IS SPENT ONCE, HERE, AND THE JUSTIFICATION IS THAT THE
// QUANTITY IS GENUINELY TRANSLATION-VARIANT. A single face's pressure gradient
// depends on where the face sits; only the sum over a closed surface does not.
// So the first vertex enters absolute, and the conditioning that makes that
// safe is in face_pressure_gradient: the two edge vectors are differenced
// first and stay at edge scale, which confines the absolute magnitude to
// well-conditioned small-times-large products and keeps the area vector
// `e1 x e2` a small-times-small one. Crossing two absolute positions instead
// cancels the area vector down to the rounding left over when two large
// coordinates agree in their leading digits, an error that grows with distance
// from the origin, and that is the one re-association this project never makes.
//
// The Hessian still consumes the absolute columns: its SVD is of the absolute
// matrix `[v0|v1|v2]`, so its PSD projection remains a function of where the
// face sits in the world. That is a property of the per-face pressure
// formulation itself and not of this call, and reconditioning it is open work.
[[seam::device_fn]] inline void face_pressure_force_hessian(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2, float pressure,
    Mat3x3f &gradient, Mat9x9f &hessian) {
    Vec3f v0 = x0;
    Vec3f e1 = (x1 - x0).cast<float>();
    Vec3f e2 = (x2 - x0).cast<float>();
    Vec3f v1 = v0 + e1;
    Vec3f v2 = v0 + e2;
    gradient = face_pressure_gradient(pressure, v0, e1, e2);
    hessian = face_pressure_hessian(pressure, v0, v1, v2);
}

// One triangle's nine 3x3 Hessian blocks, pushed into the fixed matrix.
//
// ROW MAJOR OVER THE NINE BLOCKS, which is `atomic_embed_hessian<3>`'s own
// `(ii, jj)` order: the block at `(ii, jj)` is the 3x3 of the 9x9 at rows
// `3 * ii` and columns `3 * jj`, and it lands at
// `(face_vertex[ii], face_vertex[jj])`.
//
// A DROPPED BLOCK IS REPORTED, AND THE TWO REFUSALS MUST NOT BE CONFUSED.
// `fixed_csr_atomic_push` answers false in two cases. A block whose row exceeds
// its column is DECLINED BY DESIGN: the matrix stores only `i <= j` and reaches
// the rest through its transpose index, and the symmetric counterpart of that
// block is pushed by the `(jj, ii)` pass of this same loop. A block with
// `row <= column` and no slot is a LOST COUPLING: the scene's fixed sparsity has
// no place for a pair this element's stencil writes, so the Newton matrix would
// be assembled missing it and would no longer be the SPD-by-assembly matrix the
// PCG guards are entitled to assume. That is the defect that shipped once as the
// rod-bend `(j, k)` stencil bug, masked by damping and a small `dt`, and
// `FixedCSRMat::push`'s CUDA callers do not check it. The payload names the
// block, the element and the slot inside the element's nine, so the failure
// identifies the stencil rather than only the run.
//
// IT TAKES A NAME OF ITS OWN because it is a second body rather than an
// overload: `check-shared-wiring.py`'s census keys a neutral body on its NAME
// and cannot tell two rows apart.
[[seam::device_fn]] inline void face_atomic_push_hessian(
    const Vec3u &face_vertex,
    const Mat9x9f &hessian,
    const unsigned *index,
    const unsigned *offset,
    compute::atomic_float_t *value, unsigned row_count,
    DiagHandle diag, unsigned element) {
    for (unsigned row_vertex = 0; row_vertex < 3; ++row_vertex) {
        for (unsigned column_vertex = 0; column_vertex < 3; ++column_vertex) {
            const unsigned row = face_vertex[row_vertex];
            const unsigned column = face_vertex[column_vertex];
            const Mat3x3f block =
                hessian.block<3, 3>(3 * row_vertex, 3 * column_vertex);
            const bool stored = fixed_csr_atomic_push(index, offset, value,
                                                          row_count, row,
                                                          column, block);
            DIAG_ASSERT4(diag, stored || row > column,
                        static_cast<float>(row), static_cast<float>(column),
                        static_cast<float>(element),
                        static_cast<float>(3 * row_vertex + column_vertex));
        }
    }
}

// THE WHOLE SHELL MEMBRANE AND PRESSURE LAYER OF ONE FACE, EVALUATED AND
// EMBEDDED, with nothing left in a caller but the walk. The deformation
// gradient, the SVD, the material table, the spectral force, the PSD-projected
// 6x6 spectral Hessian, the material-frame conversion and the Rayleigh damping
// block all live in this thread's registers, and what leaves them is the
// scatters below.
//
// TWO TERMS, TWO EMBEDS, AND THE TWO ARE SIBLINGS RATHER THAN NESTED.
// Elasticity runs under `if (mu > 0.0f)` and pressure under
// `if (pressure > 0.0f)`, each forming and embedding its own gradient and
// Hessian, and `face_add_stiffness_damping` is called INSIDE
// `face_elastic_force_hessian` on the elastic accumulators alone. Two
// properties follow structurally, and neither needs a flag: the pressure term
// is never damped, because its block is formed after the elastic embed has
// finished with its own accumulators and never passes through the damping body,
// and a face carrying pressure with no elastic stiffness still assembles its
// pressure.
//
// A FACE THE GATE EXCLUDES CONTRIBUTES NOTHING: not a zero force, not a zero
// block, not a slot lookup. `mu == 0` is how a PDRD face, a zero-stiffness
// material and a face the caller has excluded by seeding its material inert all
// arrive, and `pressure == 0` is the same statement for the second term.
//
// AN UNRECOGNIZED MODEL ID STOPS THE FACE ENTIRELY, the pressure block
// included: `face_elastic_force_hessian` returns `false` having touched only
// this thread's local gradient and Hessian, and this body returns on it before
// reaching either embed, so neither `force` nor `value` is scattered to.
[[seam::device_fn]] inline void face_elastic_embed(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2,
    const Vec3f &current0,
    const Vec3f &current1,
    const Vec3f &current2,
    const Vec3u &face_vertex,
    const Mat2x2f &inverse_rest, unsigned model, float mu,
    float lambda, float mass, float deform_damping, float pressure, float dt,
    float eigenanalysis_eps,
    compute::atomic_float_t *force,
    const unsigned *index,
    const unsigned *offset,
    compute::atomic_float_t *value, unsigned row_count,
    DiagHandle diag, unsigned element) {
    if (mu > 0.0f) {
        Mat3x3f gradient = Mat3x3f::Zero();
        Mat9x9f hessian = Mat9x9f::Zero();
        if (!face_elastic_force_hessian(x0, x1, x2, current0, current1,
                                        current2, inverse_rest, model, mu,
                                        lambda, mass, deform_damping, dt,
                                        eigenanalysis_eps, element, gradient,
                                        hessian, diag)) {
            return;
        }
        // THE FORCE FIRST, THEN THE HESSIAN, which is the order the two embeds
        // run in inside `embed_face_force_hessian`. They are different
        // accumulators, so the order between them changes neither, and it is
        // kept because there is no reason to differ.
        face_atomic_embed_force(face_vertex, gradient, force);
        face_atomic_push_hessian(face_vertex, hessian, index, offset, value,
                                     row_count, diag, element);
    }
    if (pressure > 0.0f) {
        Mat3x3f gradient = Mat3x3f::Zero();
        Mat9x9f hessian = Mat9x9f::Zero();
        face_pressure_force_hessian(x0, x1, x2, pressure, gradient, hessian);
        face_atomic_embed_force(face_vertex, gradient, force);
        face_atomic_push_hessian(face_vertex, hessian, index, offset, value,
                                     row_count, diag, element);
    }
}

// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the range shim a host C++ compiler compiles, and the Rust
// `#[repr(C)]` twin the driver fills.
//
// THE FACE'S THREE SLOTS ARRIVE TWICE, AND THE SECOND ARRIVAL IS NOT REDUNDANT.
// `[[seam::indices(3)]]` is how the entry reads the element's own index list,
// checks each slot against `[[seam::bound]] vertex_count` and gathers the two
// position buffers through it; a slot is DATA rather than the thread index, so
// the count guard says nothing about it and Metal answers an out-of-bounds read
// with 0.0 rather than a fault. That list is deliberately NOT forwarded, and
// this body needs the three numbers themselves: the force rows it writes and
// the CSR blocks it pushes are addressed by the element's own vertices. So the
// same buffer is named a second time as a gathered `Vec3u`, which is the same
// twelve bytes at the same element and reaches the body as a thread-space copy.
// Both fields carry the same handle at every call site.
//
// `force` AND `value` ARE BASE POINTERS AND THE SCATTER IS SERIAL. Two faces
// sharing a vertex have the same force destination and two sharing an edge have
// the same CSR slot, so `compute::atomic_add`, which `seam/seam_host.h` spells
// as a plain read, add and write back on the premise that one thread runs a
// shared body, makes a parallel pass a DATA RACE rather than a different fold
// order. Nothing in this declaration says otherwise: the kernel table's
// `Scatter::Atomic` row is what keeps the range one ascending pass.
//
// `compute::atomic_float_t` is the seam's own name for both destinations'
// element type, `float` under nvcc and on the host and `atomic_float` under
// MSL: a `float *` here would compile on two backends and fail on the third.
// THE SAME EMBED, READING ITS MATERIAL OFF THE DEVICE RECORDS.
//
// The six scalars the body above takes are read here out of `FaceProp` and
// `FaceParam`, which are already resident on the device, so an assembly pass
// uploads no per-element material at all. Flattening them into six host-filled
// arrays instead would upload every face's material on every pass: measured on
// `drape` at 3 frames, that is 72 host-to-device calls carrying 44.4 MB.
//
// THE GATE IS CARRIED BY THE BODY, NOT BY A SEEDED MATERIAL, which is what
// makes reading the records equivalent to flattening them rather than merely
// similar. Writing ZEROS into a flattened array for a face that is inflated but
// not stiff would make the body's `mu > 0` test skip the elastic term; reading
// `param.mu` gives that test the same answer with no seeding at all. `mass` is
// read only inside that test, so taking it from the record unconditionally
// changes nothing either. The faces excluded outright,
// `fixed`, `rest_excluded` and `collider`, never reach a dispatch: the host
// still applies that gate when it builds the active list, which is a
// COMPACTION and not a material.
//
// `face_param` IS NOT GATHERED, because it is indexed per MATERIAL and not per
// face: `param_index` is deduplicated across objects that share a material, so
// the body reaches it through the record's own index and owes the thread-space
// copy an entry would otherwise have made. `shell_stretch_terms` takes the same
// pair the same way.
[[seam::device_fn]] inline void face_elastic_embed_from_records(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2,
    const Vec3f &current0,
    const Vec3f &current1,
    const Vec3f &current2,
    const Vec3u &face_vertex,
    const Mat2x2f &inverse_rest,
    const FaceProp &prop,
    const FaceParam *face_param, float dt,
    float eigenanalysis_eps,
    compute::atomic_float_t *force,
    const unsigned *index,
    const unsigned *offset,
    compute::atomic_float_t *value, unsigned row_count,
    DiagHandle diag, unsigned element) {
    // THE EXCLUSION GATE IS EXPLICIT HERE BECAUSE THE MATERIAL DOES NOT CARRY
    // IT. The dispatch covers EVERY face, not the active list, and the record
    // hands each face its real stiffness, so without this test a pinned face, a
    // collider's face or one whose streamed rest shape is near-singular would
    // assemble an elastic term it must not carry. `shell_stretch_terms` applies
    // the same three in its body for the same reason.
    if (prop.fixed || prop.rest_excluded || prop.collider) {
        return;
    }
    const FaceParam param = face_param[prop.param_index];
    face_elastic_embed(x0, x1, x2, current0, current1, current2, face_vertex,
                       inverse_rest, static_cast<unsigned>(param.model),
                       param.mu, param.lambda, prop.mass, param.deform_damping,
                       param.pressure, dt, eigenanalysis_eps, force, index,
                       offset, value, row_count, diag, element);
}

[[seam::entry(count, element)]] void face_elastic_embed_from_records(
    [[seam::through]] const Vec3f *x,
    [[seam::through]] const Vec3f *current,
    [[seam::indices(3)]] const unsigned *face,
    [[seam::bound]] unsigned vertex_count,
    const Vec3u *face_vertex,
    
    const Mat2x2f *inverse_rest,
    const FaceProp *prop,
    const FaceParam *face_param,
    float dt, float eigenanalysis_eps,
    compute::atomic_float_t *force,
    const unsigned *index,
    const unsigned *offset,
    compute::atomic_float_t *value,
    unsigned row_count,
    DiagHandle diag,
    unsigned element,
    unsigned count);

// THE ELASTIC EMBED HAS ONE ENTRY AND IT IS THE RECORD-READING ONE. The driver
// dispatches only that one, so the six-scalar body it composes needs no entry
// of its own: an entry point reached by nothing is a kernel id, a table row and
// a name for no work. That body still has a caller, which is
// `face_elastic_embed_from_records` just above.
