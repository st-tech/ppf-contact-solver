// File: material_diff_table.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py
// renders it into the three forms the three compilers read.
// `[[seam::device_fn]]` is the execution space and `[[seam::thread]]` /
// `[[seam::device]]` are the address spaces MSL requires on every reference and
// pointer type.
//
// THE MATERIAL DIFF TABLE AS A STAGED PASS, one body per element arity.
// `energy/face_force.kernel.cpp` and `energy/tet_force.kernel.cpp` carry the
// same dispatch for a caller that holds one element in registers and wants its
// force and Hessian in one call. This file carries it for a caller that stages
// the elastic pass over a whole range of one element kind, keeping the singular
// values and the table in device memory between passes.
//
// NEITHER SPELLS A MATERIAL. Both call `ARAP::make_diff_table*`,
// `StVK::make_diff_table*` and `SNHk::make_diff_table*` (`model/arap.hpp`,
// `model/stvk.hpp`, `model/snhk.hpp`), which is where each material is
// composed, so the two arrangements cannot disagree about what an ARAP element
// is.
//
// THE MODEL ID REACHES THE BODY RATHER THAN THE ENTRY POINT, for the reason
// every dispatch in this tree validates where it dispatches: a declaration
// cannot express a branch, and this branch decides which material the element
// has. Three properties of that branch are load-bearing and none of them is the
// obvious one.
//
// AN UNRECOGNIZED MODEL ID DOES NOT FALL THROUGH TO SNHk. The usual dispatch
// shape (`if ARAP ... else if StVK ... else SNHk`) turns a bad id into a
// silently wrong MATERIAL, which runs to completion and looks like a physics
// disagreement rather than like a defect. The fused bodies already refuse one;
// so do these.
//
// MODEL ID 4 (`ELASTIC_MODEL_PDRD`) IS ALLOWED, NOT REJECTED. A PDRD body's
// faces and tets carry that id with `mu == 0`, because their shape is held by
// the reduced rigid solve rather than by an elastic energy, and `energy.cu`
// never validates the id at all: it guards the whole term on `mu > 0`, so such
// an element leaves before any dispatch. A backend that validates ids therefore
// has to admit a non-elastic model or it refuses a scene CUDA runs. Its table
// is zero, which is the correct energy for an element that has none.
//
// THE VERDICT IS THE RETURN VALUE AND THE CALLER MUST HONOR IT. A table the
// dispatch did not write holds the zero fill, which is NOT a zero
// CONTRIBUTION: the element's material is unknown, so the run must stop rather
// than assemble it. That is why each body returns a code beside its two
// outputs. The two destinations are pointers to this element's own slot rather
// than references, because that is the shape a generated entry point hands a
// body whose output lives in device memory: `[[seam::stride(1)]]` is the buffer
// advanced to the element, and the address space travels with it.

#include "arap.hpp"
#include "snhk.hpp"
#include "stvk.hpp"
#include "../elastic_model.kernel.cpp"

// The TET verdict, two-valued. A tet has no BaraffWitkin form, that being a
// shell model, so the only question a solid's dispatch answers is whether the
// id was one it knows.
enum : unsigned {
    TET_TABLE_UNKNOWN = 0u,
    TET_TABLE_ACCEPTED = 1u
};

// The FACE verdict, FOUR-valued, and the extra codes are the shell model set's
// doing rather than this file's. A plain boolean would have to call a
// BaraffWitkin face "accepted" while writing it a ZERO table, and a caller that
// then skipped the BaraffWitkin pass would assemble a cloth with no membrane at
// all: a full run of plausible frames, computed for a material the scene did
// not ask for. The code says which of the two families owns the face, so that
// mistake is a verdict the caller can check rather than an omission nothing can
// see.
//
// FACE_TABLE_NO_ENERGY is not a synonym for FACE_TABLE_ACCEPTED even
// though both leave a zero table. `energy.cu` never reaches it, since such a
// face leaves at `mu > 0`; a face that arrives here with `mu > 0` is the one
// energy.cu answers with a live device assert, so a caller must treat it on an
// ACTIVE face exactly as it treats an unknown id.
enum : unsigned {
    FACE_TABLE_UNKNOWN = 0u,
    FACE_TABLE_ACCEPTED = 1u,
    FACE_TABLE_BARAFF_WITKIN = 2u,
    FACE_TABLE_NO_ENERGY = 3u
};

// One tet's `(de/da, d2e/da2)` table in the singular-value basis, written where
// the staged pipeline expects it, and the verdict on its model id.
[[seam::device_fn]] inline unsigned tet_material_diff_table(
    unsigned model, const Vec3f &sigma, float mu, float lambda,
    Vec3f *gradient_sigma,
    Mat3x3f *hessian_sigma) {
    DiffTable3 table;
    unsigned verdict = TET_TABLE_ACCEPTED;
    if (model == ELASTIC_MODEL_ARAP) {
        table = ARAP::make_diff_table3(sigma, mu, lambda);
    } else if (model == ELASTIC_MODEL_STVK) {
        table = StVK::make_diff_table3(sigma, mu, lambda);
    } else if (model == ELASTIC_MODEL_SNHK) {
        table = SNHk::make_diff_table3(sigma, mu, lambda);
    } else if (model == ELASTIC_MODEL_PDRD) {
        table.deda = Vec3f::Zero();
        table.d2ed2a = Mat3x3f::Zero();
    } else {
        table.deda = Vec3f::Zero();
        table.d2ed2a = Mat3x3f::Zero();
        verdict = TET_TABLE_UNKNOWN;
    }
    gradient_sigma[0] = table.deda;
    hessian_sigma[0] = table.d2ed2a;
    return verdict;
}

// The 2x2 analogue, and the one difference is the verdict's vocabulary above.
//
// BARAFFWITKIN NEVER FORMS THE SVD, which is why it is a code here rather than
// an arm. It goes from the deformation gradient to the gradient and the Hessian
// directly, and `energy/model/baraffwitkin.kernel.cpp` is the pass that then
// OVERWRITES the zero table this one leaves.
[[seam::device_fn]] inline unsigned face_material_diff_table(
    unsigned model, const Vec2f &sigma, float mu, float lambda,
    Vec2f *gradient_sigma,
    Mat2x2f *hessian_sigma) {
    DiffTable2 table;
    unsigned verdict = FACE_TABLE_ACCEPTED;
    if (model == ELASTIC_MODEL_ARAP) {
        table = ARAP::make_diff_table2(sigma, mu, lambda);
    } else if (model == ELASTIC_MODEL_STVK) {
        table = StVK::make_diff_table2(sigma, mu, lambda);
    } else if (model == ELASTIC_MODEL_SNHK) {
        table = SNHk::make_diff_table2(sigma, mu, lambda);
    } else if (model == ELASTIC_MODEL_BARAFF_WITKIN) {
        table.deda = Vec2f::Zero();
        table.d2ed2a = Mat2x2f::Zero();
        verdict = FACE_TABLE_BARAFF_WITKIN;
    } else if (model == ELASTIC_MODEL_PDRD) {
        table.deda = Vec2f::Zero();
        table.d2ed2a = Mat2x2f::Zero();
        verdict = FACE_TABLE_NO_ENERGY;
    } else {
        table.deda = Vec2f::Zero();
        table.d2ed2a = Mat2x2f::Zero();
        verdict = FACE_TABLE_UNKNOWN;
    }
    gradient_sigma[0] = table.deda;
    hessian_sigma[0] = table.d2ed2a;
    return verdict;
}

// The two entry points, each declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `ppf_<stem>_entry` shim a host C++ compiler compiles, and the
// Rust `#[repr(C)]` twin the driver fills.
//
// FOUR ELEMENT GATHERS, TWO ELEMENT DESTINATIONS AND ONE SCATTER. The table's
// two halves are strides of one because a scatter carries the body's RETURN
// value and one call returns one value; the verdict IS that value, so it is the
// scatter. The verdict buffer is `unsigned` rather than a byte because a
// generated record's buffers address 4-byte pointees, which is what keeps a
// record free of padding a later field could hide in.
[[seam::entry(count)]] void tet_material_diff_table(
    const unsigned *model,
    const Vec3f *sigma,
    const float *mu,
    const float *lambda,
    [[seam::stride(1)]] Vec3f *gradient_sigma,
    [[seam::stride(1)]] Mat3x3f *hessian_sigma,
    [[seam::scatter]] unsigned *accepted,
    unsigned count);

[[seam::entry(count)]] void face_material_diff_table(
    const unsigned *model,
    const Vec2f *sigma,
    const float *mu,
    const float *lambda,
    [[seam::stride(1)]] Vec2f *gradient_sigma,
    [[seam::stride(1)]] Mat2x2f *hessian_sigma,
    [[seam::scatter]] unsigned *dispatch,
    unsigned count);
