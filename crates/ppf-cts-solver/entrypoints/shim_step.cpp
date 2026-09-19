// File: entrypoints/shim_step.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// The CPU backend's STEP entry points: the ones the Newton driver calls, as
// opposed to the element kernels in kernel_shim.cpp.
//
// A SECOND TRANSLATION UNIT, A SPLIT BY SUBJECT rather
// than a new mechanism. Every rule kernel_shim.cpp states applies here word for
// word: this file is a loop, a gather and a scatter, and every value it
// produces comes from a neutral body under ../src/kernels or from a shared header
// on the `SM_` seam. If a float expression appears below that is not a call
// into one of those, the architecture's one hard constraint has been crossed.
//
// WHY THESE ARE HERE AND NOT THERE. kernel_shim.cpp is the element layer: it
// takes flat arrays of one element kind and writes flat arrays back. The
// entries below take the SCENE: `VertexProp`, `ParamSet`, `FixPair`, the
// vertex-face neighbor table. Those are `repr(C)` records Rust's `data.rs`
// mirrors field for field, so passing them whole is a reinterpretation and not
// a conversion, and it keeps the driver from spelling out sixty scalars per
// call. Keeping the two layers in separate files also keeps a rebuild of one
// off the other.
//
// ONE SAFETY RULE, THE CALLER'S, as in kernel_shim.cpp:
// `dirichlet_lift_entry` reaches `compute::atomic_add`, which the host seam
// spells as a plain read, add and write back. Run it serially, or partition so
// no two ranges reach the same destination row.
//
// WHAT IS DELIBERATELY ABSENT FROM `momentum_embed_entry`, and why the
// absence is safe rather than a silent gap: `embed_vertex_force_hessian`
// (`energy/energy.cu`) also carries a TORQUE-GROUP loop, which needs the
// per-group centroid and PCA axis that `energy::compute_torque_groups`
// produces. That pre-pass is not ported, so torque groups are refused by name
// in `src/driver/refusal.rs` and the driver asserts the count is zero before
// calling this. The term is therefore unreachable rather than skipped. When the
// pre-pass lands, the loop is added HERE and the refusal comes off in the same
// change.

#include "../src/kernels/data.hpp"

// NO SHARED ENERGY HEADER IS REACHED FROM HERE. The momentum layer is this
// file's only subject that would name `momentum.hpp`, `air_damper.hpp` and
// `fix.hpp`, and it is a generated entry point, so those three are included by
// the neutral body in `src/kernels/main/momentum.kernel.cpp` instead. Naming
// them here would not be merely untidy: `air_damper.hpp` declares its functions
// without `inline`, so a second translation unit including it is a duplicate
// symbol at the link.
//
// NO ELASTIC MATERIAL IS REACHED FROM HERE EITHER. Every one of the four this solver carries
// is reached through a GENERATED entry point instead, from a declaration beside
// its neutral body: the three diff-table models through
// `energy/model/material_diff_table.kernel.cpp` and BaraffWitkin, which forms
// no SVD and so has a pass of its own, through
// `energy/model/baraffwitkin.kernel.cpp`.

#include <cstddef>
#include <cstdint>

// The same reinterpretations kernel_shim.cpp licenses, restated here because a
// translation unit cannot borrow another's static helpers and a second spelling
// of a cast is exactly what these assertions exist to prevent.
static_assert(sizeof(float) == sizeof(int32_t),
              "a position component must be one 32-bit word for the int32 "
              "array views below to be reinterpretations, not conversions");
static_assert(sizeof(Vec3f) == 3 * sizeof(int32_t),
              "Vec3f must be three packed 32-bit components");
static_assert(sizeof(Mat3x3f) == 9 * sizeof(float),
              "SMat must be packed column-major storage with no padding");
static_assert(alignof(Mat3x3f) == alignof(float),
              "SMat must carry natural alignment, not a vector alignment");

static inline const Vec3f *step_position(const int32_t *p) {
    return reinterpret_cast<const Vec3f *>(p);
}
static inline Vec3f *step_position_mut(int32_t *p) {
    return reinterpret_cast<Vec3f *>(p);
}
template <class M> static inline M *step_mat_mut(float *p) {
    static_assert(sizeof(M) == sizeof(float) * M::Size,
                  "an SMat must be exactly its elements, with no padding");
    static_assert(alignof(M) == alignof(float),
                  "an SMat must carry natural alignment");
    return reinterpret_cast<M *>(p);
}

// ---------------------------------------------------------------------------
// The step's position arithmetic: the seven bodies promoted out of main.cu.
// ---------------------------------------------------------------------------

#include "main/dirichlet.kernel.cpp"
#include "main/dx_norm.kernel.cpp"
#include "main/dx_seed.kernel.cpp"
#include "main/fix_xz_drag.kernel.cpp"
#include "main/position_accept.kernel.cpp"
#include "main/position_step.kernel.cpp"
#include "main/rewind_fix.kernel.cpp"
#include "main/velocity.kernel.cpp"

extern "C" {

// The largest absolute coordinate the position representation can carry.
//
// A CONSTANT OF THE REPRESENTATION, not a scene parameter, and it crosses the
// boundary rather than being spelled in Rust because a second spelling would be
// a second definition of the domain the step's own overflow check is judged
// against. The magnitude relative to the ORIGIN is the entire measurement, so
// an absolute coordinate is the right quantity here.
//
// Spelled as a literal rather than as `FLT_MAX`, which `common.hpp` redefines
// to a small sentinel for this tree's own use: taking it here would report a
// domain about thirty orders of magnitude too small.
float position_domain_abi() {
    return 3.402823466e+38f;
}

// A3's three per-vertex quantities are NOT here. The velocity itself, its
// squared speed and the vertex's distance from the origin are computed by
// `velocity_terms`, which declares its own entry point beside the body
// (`src/kernels/main/velocity.kernel.cpp`), so the range shim is rendered into
// `entrypoints/entries.cpp`. It remains ONE pass for all three, as main.cu
// does: they read the same two position arrays, and splitting them would read
// those arrays three times.

// B3'S PIN REWIND IS NOT HERE. `rewind_fix` declares its own entry point
// beside its body (`src/kernels/main/rewind_fix.kernel.cpp`), where the gate on
// a pin's `kinematic` flag belongs, and the range shim is rendered into
// `entrypoints/entries.cpp`. It still rewrites the pin array in place, which is
// what main.cu does and is safe for the same reason: the host hands over a
// fresh constraint every step and this runs once per step.

// B4. Seed the search direction on every prescribed row with its exact
// increment, so the initial residual on that row is zero and the eliminated
// column keeps `(A p)` zero on it for every PCG iteration afterwards.
// The pass itself is NOT here. `dx_seed` declares its own entry point
// beside its body (`src/kernels/main/dx_seed.kernel.cpp`), where the gate on
// whether a row was removed belongs, and the range shim is rendered into
// `entrypoints/entries.cpp`.

// B19 IS NOT HERE. The per-vertex magnitude of the search direction is one
// base pointer, one thread index and one returned float, so `dx_magnitude`
// declares its own entry point beside the body
// (`src/kernels/main/dx_norm.kernel.cpp`) and the range shim is rendered into
// `entrypoints/entries.cpp`. The caller max-reduces the result into `max_dx`.

// B20 IS NOT HERE. The rescaled Newton step is one element in and one element
// out, so `position_step` declares its own entry point beside the body
// (`src/kernels/main/position_step.kernel.cpp`) and the range shim is rendered
// into `entrypoints/entries.cpp`. The gather and the scatter name one buffer,
// which is the in-place update a launcher spells as `x[i] = f(x[i], ...)`.

// THE `fix-xz` DRAG'S POSITION HALF IS NOT HERE. It runs after the Newton
// position step and before the line search, where main.cu puts it, but the
// whole per-vertex answer including the DOF-removed case is `fix_xz_drag`,
// which declares its own entry point beside the body
// (`src/kernels/main/fix_xz_drag.kernel.cpp`), so the range shim is rendered
// into `entrypoints/entries.cpp`. The gather and the scatter name one buffer,
// which is the in-place update a launcher spells as `x[i] = f(x[i], ...)`.
//
// A DOF-REMOVED VERTEX GETS NO DRAG, and that is not an optimization. Such a
// vertex is prescribed exactly by its pin, so dragging its x and z would move it
// off the keyframe the Dirichlet row holds it to, and the two would then
// disagree about where it is. The predicate travels with the body rather than
// with the launcher, which is why the range shim carries none.
//
// This half ships with the momentum half in `momentum_embed_entry`,
// never alone: the momentum term is what the linear system is built from and
// this clamp is what the accepted step does, so a backend carrying one would
// solve a system that does not describe the step it then takes.

// B31 IS NOT HERE. Accepting the line search's fraction reads one position
// from each of two buffers and writes one back to the second, so
// `position_accept` declares its own entry point beside the body
// (`src/kernels/main/position_accept.kernel.cpp`) and the range shim is
// rendered into `entrypoints/entries.cpp`. The gather and the scatter name one buffer,
// which is the in-place update a launcher spells as
// `x[i] = f(y[i], x[i], ...)`.

} // extern "C"

// ---------------------------------------------------------------------------
// The momentum layer: inertia, the aerodynamic term, the pull spring, the
// isotropic drag and the `fix-xz` drag, per vertex.
//
// This is `embed_vertex_force_hessian` (`energy/energy.cu:92`) with the torque
// loop absent for the reason stated at the top of this file. Every term is read
// from its shared header in main.cu's order, and the order is load-bearing: `f`
// and `H` are fp32 running sums, so re-ordering the accumulation changes the
// last bits of the assembled system.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The Newton operator, `A + B + C`, and the PCG inner-product terms.
// ---------------------------------------------------------------------------

#include "solver/pcg.kernel.cpp"

extern "C" {

// One row's contribution to `a . b`, and the magnitude of that contribution.
//
// The caller folds both arrays under `src/driver/reduce.rs`'s fixed shape, which
// is where this backend's determinism lives. The two must be folded over the
// same rows in the same shape: the magnitude is the round-off bound the
// curvature verdict is read against, and a bound measured over a different
// decomposition is a number about a different fold.
// THE DOT-PRODUCT TERMS ARE NOT HERE. `pcg_dot_terms` declares its own
// entry point beside its body (`src/kernels/solver/pcg.kernel.cpp`) and the
// range shim is rendered into `entrypoints/entries.cpp`.

} // extern "C"

// ---------------------------------------------------------------------------
// One shared-body composition the driver would otherwise have to spell in
// Rust: the preconditioner's diagonal sum.
// ---------------------------------------------------------------------------

#include "csrmat/fixed_csr.kernel.cpp"
#include "primitives/vec_ops.kernel.cpp"

extern "C" {

// THE PER-ELEMENT MASS SCALE IS NOT HERE. It declares its own entry point
// beside its body (`src/kernels/primitives/vec_ops.kernel.cpp`) and the range shim
// is rendered into `entrypoints/entries.cpp`.

} // extern "C"
