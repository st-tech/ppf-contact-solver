// File: fix_xz_drag.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. Plain C++ belonging to no backend: no preprocessor
// conditional, no macro of its own, and no spelling that only nvcc or only the
// Metal shader compiler accepts. `[[seam::device_fn]]` is the execution space
// and `[[seam::thread]]` is the address space MSL requires on every reference.
//
// THE POSITION-SIDE HALF OF THE `fix-xz` DRAG.
//
// `fix-xz` is TWO halves and they ship together. The momentum side
// (momentum::fix_xz_active, fix_xz_gradient, fix_xz_hessian) is the force and
// Hessian the Newton system carries; this is what the accepted step then does
// to the position. A backend that applies one without the other solves a system
// that does not describe the step it takes.

// Pull a vertex's horizontal position back toward its previous pose, once the
// vertex has risen above `threshold`.
//
// `next` is the position the Newton step proposes and `previous` is the pose
// the drag pulls toward; the vertical coordinate is untouched. Below the
// threshold the term is ABSENT rather than zero, which is why the predicate is
// here and not folded into a weight.
//
// THE RAMP IS THE EXPRESSION THE MOMENTUM SIDE ALSO USES
// (`energy/model/momentum.hpp`, `fix_xz_ramp`): `min(1, y - threshold)`,
// saturating one unit above the threshold. The two must stay the same
// expression, because this body moves the position while that one supplies the
// gradient and Hessian of the same term, so a backend writing its own copy of
// either owes the other.
[[seam::device_fn]] inline Vec3f
fix_xz_drag_position(const Vec3f &next,
                         const Vec3f &previous,
                         float threshold) {
    Vec3f dragged = next;
    if (dragged[1] > float(threshold)) {
        float y = fmath::min(
            1.0f, static_cast<float>(dragged[1] - float(threshold)));
        dragged[0] -= y * (dragged[0] - previous[0]);
        dragged[2] -= y * (dragged[2] - previous[2]);
    }
    return dragged;
}

// The drag as one vertex's whole answer, including the case where it has none.
//
// A COMPOSITION AND NOTHING ELSE. It calls the body above with the arguments
// that body has always been given; what it adds is the DOF-removed predicate,
// which every caller of the body already spelled for itself, so a single
// declaration has something to be generated from and no arithmetic moved.
//
// THE PREDICATE IS PART OF THE OPERATION, not a launcher optimization. A
// DOF-removed vertex is prescribed exactly by its pin, and its Dirichlet row
// holds it on the keyframe; dragging its x and z would move it off, and the row
// and the position would then disagree about where the vertex is.
//
// A SKIPPED VERTEX RETURNS ITS OWN POSITION, which is what makes the predicate
// expressible as a return value. The launchers this replaces skipped the STORE
// as well as the arithmetic, and a generated entry always writes what the body
// returns; the two agree because the value written is the value read, from the
// one thread that owns the element, so the store is the same bits and no other
// thread observes it. What must not change is which vertices the ARITHMETIC
// runs for, and that is the predicate above.
[[seam::device_fn]] inline Vec3f
fix_xz_drag(const Vec3f &next,
                const Vec3f &previous, unsigned dof_removed,
                float threshold) {
    if (dof_removed != 0u) {
        return next;
    }
    return fix_xz_drag_position(next, previous, threshold);
}

// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `fix_xz_drag_entry` shim a host C++ compiler compiles, and
// the Rust `#[repr(C)]` twin the driver fills.
//
// POSITIONS MOVE AS WHOLE TRIPLES. Both position buffers are declared as
// pointers to `Vec3f`, whose 12 bytes the generated `[[seam::pod(12)]]`
// assertion pins in every C++ rendering, so a gather reads one position and the
// scatter writes one back. The entry computes nothing: the ramp and the two
// horizontal corrections are formed inside the body above.
//
// `eval_x` carries [[seam::gather]] and [[seam::scatter]] together, which is
// the in-place element update the range shim spelled `x[i] = f(x[i], ...)`: the
// buffer is resolved once and read inside the call.
[[seam::entry(count)]] void fix_xz_drag(
    [[seam::gather]] [[seam::scatter]]
    Vec3f *eval_x,
    const Vec3f *previous,
    const unsigned *dof_removed, float threshold,
    unsigned count);
