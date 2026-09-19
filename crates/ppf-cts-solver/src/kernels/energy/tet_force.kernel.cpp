// File: tet_force.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the three forms the three compilers read. `[[seam::device_fn]]` is the
// execution space and `[[seam::thread]]` is the address space MSL requires on
// every reference and pointer.
//
// DIAG_ASSERT4 stays a macro for the reason given in
// contact/aabb_traversal.kernel.cpp: it records __FILE__ and __LINE__ of the
// failing check, which no function can read for its caller.
//
// THE SOLID ELEMENT'S ELASTIC TERM. What is here is every float operation a
// tetrahedron contributes AND the two scatters that place it: the four-node
// force embed (`utility/hinge_scatter.kernel.cpp`, which both four-node
// elements come through) and the fixed matrix's atomic block push
// (`csrmat/fixed_csr.kernel.cpp`). What stays in the caller is the walk: which
// tets are in this iteration and what material each carries.

#include "../csrmat/fixed_csr.kernel.cpp"
#include "../linalg/eigsolve.hpp"
#include "../eigenanalysis/tet_eigenanalysis.kernel.cpp"
#include "../utility/hinge_scatter.kernel.cpp"
#include "../utility/svd3x3.kernel.cpp"
#include "../utility/tet_convert.kernel.cpp"
#include "../utility/tet_damping.kernel.cpp"
#include "elastic_model.kernel.cpp"
#include "model/arap.hpp"
#include "model/snhk.hpp"
#include "model/stvk.hpp"

// ONE DISPATCH FOR THE SPECTRAL HESSIAN AND ITS CONVERSION.
//
// The staged assembly ran `tet_spectral_hessian` and then
// `tet_convert_hessian`, which meant writing the 9x9 to global memory and
// reading all 81 floats of it back to build the 12x12. This composition calls
// THE SAME TWO BODIES in the same order and keeps the 9x9 in the thread's
// registers, so it is the staged pair's arithmetic exactly, by construction
// rather than by assertion: nothing is rewritten, the two calls are nested.
//
// It is deliberately NOT `tet_spectral_hessian_fused`, which computes the same
// matrix by its own expressions. `tests/kernels/tet_hessian_agreement.cpp`
// measures those two as equal, but the fused body is also what `c4bcd850`
// reverted the driver away from, and a composition that cannot differ needs no
// such argument.
[[seam::entry(hessian)]]
[[seam::device_fn]] inline Mat12x12f
tet_spectral_convert_hessian(const Vec3f &gradient_sigma,
                                 const Mat3x3f &hessian_sigma,
                                 const Mat3x3f &u, const Vec3f &sigma,
                                 const Mat3x3f &vt,
                                 const Mat3x3f &inverse_rest,
                                 const float &mass, float eigenanalysis_eps) {
    return tet_convert_hessian(
        tet_spectral_hessian(gradient_sigma, hessian_sigma, u, sigma, vt,
                             eigenanalysis_eps),
        inverse_rest, mass);
}

// The elastic gradient and PSD-projected Hessian of one tetrahedron, in element
// (x) space, with the Rayleigh damping block folded in.
//
// RETURNS FALSE, AND ONLY FALSE, FOR A MODEL ID NO BRANCH BELOW HANDLES. The
// model id is validated where it is dispatched and nowhere else, for the reason
// stated at face_elastic_force_hessian: a PDRD body's elements carry
// ELASTIC_MODEL_PDRD with `mu == 0` and leave at the `mu > 0.0f` gate
// before any dispatch, so an up-front validation would refuse a scene that
// simulates correctly, while no validation at all would route an unrecognized
// ELASTIC model to SNHk and assemble the wrong material silently.
//
// BaraffWitkin is a SHELL model and has no solid form, so this dispatch has
// three arms where the face's has four.
//
// The Hessian goes through the FUSED spectral form, which builds the 12x12
// directly from the nine eigenmodes and never materializes the 9x9 dF-space
// intermediate. It accumulates in place because a 12x12 returned by value blows
// the 1 KB default device stack.
template <typename D>
[[seam::device_fn]] inline bool tet_elastic_force_hessian(
    const Vec3f *x, const Vec3f *current,
    const Mat3x3f &inverse_rest, unsigned model, float mu,
    float lambda, float mass, float deform_damping, float dt,
    float eigenanalysis_eps, unsigned element,
    Mat3x4f &gradient, Mat12x12f &hessian,
    D diag) {
    if (mu > 0.0f) {
        Mat3x3f deformation =
            tet_deformation_gradient(x[0], x[1], x[2], x[3], inverse_rest);
        Mat3x3f u;
        Vec3f sigma;
        Mat3x3f vt;
        svd3x3_rv(deformation, u, sigma, vt);
        DiffTable3 table;
        if (model == ELASTIC_MODEL_ARAP) {
            table = ARAP::make_diff_table3(sigma, mu, lambda);
        } else if (model == ELASTIC_MODEL_STVK) {
            table = StVK::make_diff_table3(sigma, mu, lambda);
        } else if (model == ELASTIC_MODEL_SNHK) {
            table = SNHk::make_diff_table3(sigma, mu, lambda);
        } else {
            DIAG_ASSERT4(diag, false, static_cast<float>(model),
                        static_cast<float>(element), mu, lambda);
            return false;
        }
        Mat3x3f gradient_f = tet_spectral_force(table.deda, u, vt);
        gradient += tet_convert_force(gradient_f, inverse_rest, mass);
        tet_spectral_hessian_fused(table.deda, table.d2ed2a, u, sigma, vt,
                                       eigenanalysis_eps, inverse_rest, mass,
                                       hessian);
        tet_add_stiffness_damping(x, current, deform_damping, dt, gradient,
                                      hessian);
    }
    return true;
}

// THE WHOLE TET ELASTIC LAYER OF ONE ELEMENT, EVALUATED AND EMBEDDED, with
// nothing left in a caller but the walk over the elements. The deformation
// gradient, its factorization, the material table, the spectral force, the
// fused 12x12 spectral Hessian and the
// Rayleigh damping block all live in this thread's registers, and what leaves
// them is the two scatters below: the four vertices' force rows and the sixteen
// 3x3 blocks the element contributes to the fixed matrix.
//
// IT TAKES A NAME OF ITS OWN rather than overloading
// `tet_elastic_force_hessian`. C++ would resolve the two on arity and build
// clean; `check-shared-wiring.py`'s census keys a neutral body on its NAME and
// cannot tell two rows apart, so a composition is a new name by rule.
//
// THE `mu > 0` GATE IS OUTSIDE THE CALL, not a shortcut. This body returns
// before either embed when `mu <= 0.0f`, so an element carrying no shear
// modulus contributes NOTHING: not a zero force, not a zero block, not a slot
// lookup. A PDRD element and a
// zero-stiffness material both arrive that way, and so does an element the
// caller has excluded by seeding its material inert. The gate is repeated
// inside `tet_elastic_force_hessian` because that body is also called with a
// gradient and a Hessian the caller keeps; here it decides whether anything is
// written at all.
//
// A DROPPED HESSIAN BLOCK IS REPORTED, AND IT IS THE ONE VERDICT THIS BODY
// RAISES. `fixed_csr_atomic_push` answers false in two cases that must not be
// confused. A block whose row exceeds its column is DECLINED BY DESIGN: the
// matrix stores only `i <= j` and reaches the rest through its transpose index,
// and the symmetric counterpart of that block is pushed by the `(jj, ii)` pass
// of this same loop. A block with `row <= column` and no slot is a LOST
// COUPLING: the scene's fixed sparsity has no place for a pair this element's
// stencil writes, so the Newton matrix would be assembled missing it and would
// no longer be the SPD-by-assembly matrix the PCG guards are entitled to
// assume. That is the defect that shipped once as the rod-bend `(j, k)`
// stencil bug, masked by damping and a small `dt`, and `FixedCSRMat::push`'s
// CUDA callers do not check it. The payload names the block, the element and
// the slot inside the element's sixteen, so the failure identifies the stencil
// rather than only the run.
[[seam::device_fn]] inline void tet_elastic_embed(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2, const Vec3f &x3,
    const Vec3f &current0,
    const Vec3f &current1,
    const Vec3f &current2,
    const Vec3f &current3,
    const Vec4u &tet_vertex,
    const Mat3x3f &inverse_rest, unsigned model, float mu,
    float lambda, float mass, float deform_damping, float dt,
    float eigenanalysis_eps,
    compute::atomic_float_t *force,
    const unsigned *index,
    const unsigned *offset,
    compute::atomic_float_t *value, unsigned row_count,
    DiagHandle diag, unsigned element) {
    if (mu <= 0.0f) {
        return;
    }
    const Vec3f x[4] = {x0, x1, x2, x3};
    const Vec3f current[4] = {current0, current1, current2, current3};
    Mat3x4f gradient = Mat3x4f::Zero();
    Mat12x12f hessian = Mat12x12f::Zero();
    if (!tet_elastic_force_hessian(x, current, inverse_rest, model, mu, lambda,
                                   mass, deform_damping, dt, eigenanalysis_eps,
                                   element, gradient, hessian, diag)) {
        return;
    }
    // THE FORCE FIRST, THEN THE HESSIAN. They are different accumulators, so
    // the order between them changes neither result.
    hinge_atomic_embed_force(tet_vertex, gradient, force);
    // ROW MAJOR OVER THE SIXTEEN BLOCKS, in `(ii, jj)` order: the block at
    // `(ii, jj)` is the 3x3 of the 12x12 at rows `3 * ii` and columns
    // `3 * jj`, and it lands at `(tet_vertex[ii], tet_vertex[jj])`.
    for (unsigned row_vertex = 0; row_vertex < 4; ++row_vertex) {
        for (unsigned column_vertex = 0; column_vertex < 4; ++column_vertex) {
            const unsigned row = tet_vertex[row_vertex];
            const unsigned column = tet_vertex[column_vertex];
            const Mat3x3f block =
                hessian.block<3, 3>(3 * row_vertex, 3 * column_vertex);
            const bool stored = fixed_csr_atomic_push(index, offset, value,
                                                          row_count, row,
                                                          column, block);
            DIAG_ASSERT4(diag, stored || row > column,
                        static_cast<float>(row), static_cast<float>(column),
                        static_cast<float>(element),
                        static_cast<float>(4 * row_vertex + column_vertex));
        }
    }
}

// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the range shim a host C++ compiler compiles, and the Rust
// `#[repr(C)]` twin the driver fills.
//
// THE TET'S FOUR SLOTS ARRIVE TWICE, AND THE SECOND ARRIVAL IS NOT REDUNDANT.
// `[[seam::indices(4)]]` is how the entry reads the element's own index list,
// checks each slot against `[[seam::bound]] vertex_count` and gathers the two
// position buffers through it; a slot is DATA rather than the thread index, so
// the count guard says nothing about it and Metal answers an out-of-bounds read
// with 0.0 rather than a fault. That list is deliberately NOT forwarded, and
// this body needs the four numbers themselves: the force rows it writes and the
// CSR blocks it pushes are addressed by the element's own vertices. So the same
// buffer is named a second time as a gathered `Vec4u`, which is the same
// sixteen bytes at the same element and reaches the body as a thread-space
// copy. Both fields carry the same handle at every call site.
//
// `force` AND `value` ARE BASE POINTERS AND THE SCATTER IS SERIAL. Two tets
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
[[seam::entry(count, element)]] void tet_elastic_embed(
    [[seam::through]] const Vec3f *x,
    [[seam::through]] const Vec3f *current,
    [[seam::indices(4)]] const unsigned *tet,
    [[seam::bound]] unsigned vertex_count,
    const Vec4u *tet_vertex,
    
    const Mat3x3f *inverse_rest,
    const unsigned *model,
    const float *mu,
    const float *lambda,
    const float *mass,
    const float *deform_damping,
    float dt, float eigenanalysis_eps,
    compute::atomic_float_t *force,
    const unsigned *index,
    const unsigned *offset,
    compute::atomic_float_t *value,
    unsigned row_count,
    DiagHandle diag,
    unsigned element,
    unsigned count);

// THE TET'S MATERIAL, GATHERED FROM ITS OWN RECORDS IN ITS OWN THREAD, and that
// is the point rather than a side effect. Both inputs, the per-tet `TetProp`
// and the `TetParam` it indexes, are device arrays, so no part of a tet's
// material crosses the seam per Newton iteration and the five outputs are
// written where they are read.
//
// THE INERT SEED IS WRITTEN UNCONDITIONALLY AND FIRST. The stages downstream
// run over the WHOLE tet range rather than an active list, so a gated-out tet
// must carry values every one of them is well defined on: `ARAP` with a zero
// `mu` is recognized by the dispatch, so its verdict means what it says, and
// is zero everywhere it is used. A tet left holding the previous iteration's
// material would be a stale read that produces a plausible number rather than
// a crash.
//
// THE GATE IS IN TWO PARTS, in this order: the element's own
// `!fixed && !rest_excluded`, then `mu > 0` from the material record.
[[seam::entry(element)]]
[[seam::device_fn]] inline void tet_material_from_records(
    const TetProp &prop,
    const TetParam *tet_param,
    unsigned *model, float *mu,
    float *lambda, float *mass,
    float *damping, unsigned element) {
    model[element] = ELASTIC_MODEL_ARAP;
    mu[element] = 0.0f;
    lambda[element] = 0.0f;
    mass[element] = 0.0f;
    damping[element] = 0.0f;
    if (prop.fixed || prop.rest_excluded) {
        return;
    }
    const TetParam material = tet_param[prop.param_index];
    if (!(material.mu > 0.0f)) {
        return;
    }
    model[element] = static_cast<unsigned>(material.model);
    mu[element] = material.mu;
    lambda[element] = material.lambda;
    mass[element] = prop.mass;
    damping[element] = material.deform_damping;
}

