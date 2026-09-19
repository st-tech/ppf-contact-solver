// File: rewind_fix.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. Plain C++ belonging to no backend: no preprocessor
// conditional, no macro of its own, and no spelling that only nvcc or only the
// Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it into the
// three forms the three compilers read. `[[seam::device_fn]]` is the execution
// space and `[[seam::thread]]` the address space of a reference parameter;
// `Vec3f`, `Vec3f` and `float` arrive from whichever backend prologue is in
// scope.

// WALK A KINEMATIC PIN BACK TO THE FRACTION OF ITS STEP THAT WAS ACTUALLY
// TRAVELED.
//
// The host aims a step at some end time and writes each pin's pose there. The
// CCD line search may only get `toi` of the way through that span, and the
// clock is then advanced by the same fraction, so a pin left at the full step's
// pose would sit where the animation puts it at a time the simulation never
// reached. The collider would outrun its own animation for the rest of the step
// and snap back on the next one: a jitter on the collider, and a spurious
// contact impulse handed to whatever it is touching. An exact-Dirichlet fix pin
// makes this sharper rather than softer than a penalty pin would, because its
// degrees of freedom are eliminated and it therefore lands on the full-step
// pose exactly.
//
// A pin's path bends only where its keyframes sit, which is far apart relative
// to one step, so over a step it is straight to within O(dt^2) and the pose at
// fraction `toi` is `position - (1 - toi) * step_delta`. `back` is that
// `1 - toi`. A static pin carries a zero delta and does not move.
//
// THE ARGUMENTS CARRY THE DISTINCTION. `position` is a POSITION and
// `step_delta` is a per-step DISPLACEMENT, held as the difference it already
// is: `back` scales the displacement, and the product is subtracted from the
// position once, so the pin's own coordinate multiplies nothing and the rewind
// resolves the travel rather than the coordinate.
[[seam::device_fn]] inline Vec3f
rewind_fix_position(const Vec3f &position,
                        const Vec3f &step_delta, float back) {
    return position - (back * step_delta).cast<float>();
}

// THE PIN'S OWN RECORD IS WHAT THE STEP UPDATES, so this composition takes the
// `FixPair` rather than its three fields. The gate and the write-back were both
// spelled in the CPU backend's hand-written launcher, and a gate on a pin's
// `kinematic` flag is a branch on what the scene contains, which a backend
// library may not hold: it is here, in the neutral body the three compilers
// share, and the entry declaration below expresses only the addressing.
//
// A STATIC PIN IS SKIPPED RATHER THAN REWOUND BY ZERO. Its `step_delta` is
// zero, so the two agree on the value; they do not agree on what the record
// says, because a skipped element is not written at all and a rewind by zero
// stores the same bits back. Keeping the gate is what lets a reader see that a
// static pin is not part of this pass.
// ONE ELEMENT GATHER AND NO SCATTER: the body returns void and writes through
// the gathered element, which is an lvalue, so the pin's record is read and
// written in place. `back` is one scalar the whole dispatch shares, the
// fraction of the aimed step the line search did not reach.
[[seam::entry]]
[[seam::device_fn]] inline void
rewind_fix(FixPair &fix, float back) {
    if (fix.kinematic) {
        fix.position =
            rewind_fix_position(fix.position, fix.step_delta, back);
    }
}
