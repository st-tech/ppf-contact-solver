// File: detsqr.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef DETSQR_HPP
#define DETSQR_HPP

// `Vec2f`, `Vec3f` and the two `DiffTable` spellings, unconditionally: this
// header names them and the shader compiler cannot be handed `data.hpp`, which
// is where a CUDA or host unit gets them from. The assembler neutralizes the
// quoted line at splice time, so the shipped shader is unchanged.
#include "../../linalg/type_aliases.hpp"

#ifndef SM_MSL_CONCAT
#include "../../common.hpp"
#include "../../data.hpp"
#endif

// SM_INLINE, NOT A BARE `__device__`, AND THAT IS LINKAGE RATHER THAN A HINT.
// Under nvcc's device LTO a non-inline `__device__` function has EXTERNAL
// linkage, so two translation units of one device link that both include this
// header define the same symbol and nvlink refuses the link by mangled name.
// That is not hypothetical here: a generated entry point compiles the body it
// wraps, and the library's kernel table compiles every argument record, which
// pulls each body in a second time. `model/baraffwitkin.hpp` carries the same
// guard for the same reason.
#ifndef SM_INLINE
#define SM_INLINE __device__ inline
#define DETSQR_UNDEF_INLINE
#endif
#ifndef SM_THREAD
#define SM_THREAD
#define DETSQR_UNDEF_THREAD
#endif

namespace detsqr {

SM_INLINE float sqr(float x) { return x * x; }

// Volume term: 0.5 * lambda * (J - 1)^2 with J the Jacobian, i.e. the PRODUCT of
// the principal stretches. The two functions below and the diff tables that
// follow must describe the same energy, so both read J straight off `a`.
//
// These once subtracted 1 from each stretch BEFORE multiplying, i.e. they
// evaluated 0.5*lambda*((a0-1)(a1-1) - 1)^2, which is not the Jacobian and not
// what the diff tables differentiate (those are the exact derivatives of the
// form below: dE/da0 = lambda*a1*(J-1), d2E/da0da1 = lambda*(2*a0*a1 - 1)). The
// solver drives Newton from the tables and never calls these, so nothing was
// mis-simulated, but the mismatch would have silently poisoned the first
// consumer to evaluate the energy itself, such as an energy-based line search.
SM_INLINE float energy(SM_THREAD const Vec2f &a, float lmd) {
    float J = a[0] * a[1];
    return lmd * 0.5f * sqr(J - 1.0f);
}

SM_INLINE float energy(SM_THREAD const Vec3f &a, float lmd) {
    float J = a[0] * a[1] * a[2];
    return lmd * 0.5f * sqr(J - 1.0f);
}

SM_INLINE DiffTable2 make_diff_table2(SM_THREAD const Vec2f &a, float lmd) {
    DiffTable2 table;
    float J = a[0] * a[1];
    table.deda[0] = lmd * a[1] * (J - 1.0f);
    table.deda[1] = lmd * a[0] * (J - 1.0f);
    table.d2ed2a(0, 0) = lmd * sqr(a[1]);
    table.d2ed2a(1, 1) = lmd * sqr(a[0]);
    table.d2ed2a(0, 1) = lmd * (2.0f * a[0] * a[1] - 1.0f);
    table.d2ed2a(1, 0) = table.d2ed2a(0, 1);
    return table;
}

SM_INLINE DiffTable3 make_diff_table3(SM_THREAD const Vec3f &a, float lmd) {
    DiffTable3 table;
    float J = a[0] * a[1] * a[2];
    table.deda[0] = lmd * a[1] * a[2] * (J - 1.0f);
    table.deda[1] = lmd * a[0] * a[2] * (J - 1.0f);
    table.deda[2] = lmd * a[0] * a[1] * (J - 1.0f);
    table.d2ed2a(0, 0) = lmd * sqr(a[1] * a[2]);
    table.d2ed2a(1, 1) = lmd * sqr(a[0] * a[2]);
    table.d2ed2a(2, 2) = lmd * sqr(a[0] * a[1]);
    table.d2ed2a(0, 1) = lmd * a[2] * (2.0f * J - 1.0f);
    table.d2ed2a(0, 2) = lmd * a[1] * (2.0f * J - 1.0f);
    table.d2ed2a(1, 2) = lmd * a[0] * (2.0f * J - 1.0f);
    table.d2ed2a(1, 0) = table.d2ed2a(0, 1);
    table.d2ed2a(2, 0) = table.d2ed2a(0, 2);
    table.d2ed2a(2, 1) = table.d2ed2a(1, 2);
    return table;
}

} // namespace detsqr

#ifdef DETSQR_UNDEF_THREAD
#undef SM_THREAD
#undef DETSQR_UNDEF_THREAD
#endif

#ifdef DETSQR_UNDEF_INLINE
#undef SM_INLINE
#undef DETSQR_UNDEF_INLINE
#endif

#endif