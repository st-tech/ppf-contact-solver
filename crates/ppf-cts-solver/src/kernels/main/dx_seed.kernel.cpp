// File: dx_seed.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. Plain C++ belonging to no backend: no preprocessor
// conditional, no macro of its own, and no spelling that only nvcc or only the
// Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it into the
// three forms the three compilers read. `[[seam::device_fn]]` is the execution
// space and `[[seam::thread]]` the address space of a reference parameter;
// `Vec3f` and `Vec3f` arrive from whichever backend prologue is in scope.

// THE PRESCRIBED INCREMENT OF A REMOVED ROW, AND IT HAS THREE CONSUMERS.
//
// A fix pin is an exact Dirichlet boundary condition, so its vertex owns no
// degree of freedom in the Newton system: its row and column are eliminated and
// its correction is known in closed form. That correction is this value, the
// displacement from where the iterate currently sits to where the pin's
// prescribed path says the vertex belongs.
//
// It is written once and read three times in one Newton iteration:
//
//   1. as the SEED, loaded into `dx` for the removed rows before the solve, so
//      the initial residual on those rows is exactly zero and the eliminated
//      column keeps it zero for every iteration afterwards;
//   2. as the LIFTING TERM's right-hand factor, multiplied by each coupling
//      Hessian block and moved to the free row's right-hand side, which is the
//      only signal a free vertex gets that a prescribed one is advancing into
//      it (main/dirichlet.kernel.cpp);
//   3. as the right-hand side of the removed row itself, once that row has been
//      replaced by the identity.
//
// One definition, so the three cannot disagree. They must not: the seed and the
// row's right-hand side agreeing is what makes the residual on a removed row
// identically zero, and the lift is the same quantity seen from the other side
// of the coupling.
//
// THE INCREMENT IS A DIFFERENCE AND NOTHING ELSE, which is the reason this is a
// kernel body rather than driver bookkeeping. Both operands are POSITIONS that
// agree in their leading digits, so subtracting them cancels those digits and
// leaves the increment; anything that scaled or offset a position before the
// subtraction would spend precision on the absolute magnitude and leave the
// increment as the rounding that survived, an error that is zero at the origin
// and grows across the domain.
[[seam::device_fn]] inline Vec3f
prescribed_increment(const Vec3f &current,
                         const Vec3f &target) {
    return (current - target).cast<float>();
}

// THE SEED PASS, which is consumer 1 of the three above.
//
// THE GATE IS THE COMPOSITION'S WHOLE REASON TO EXIST. `fix_index > 0` asks
// whether this vertex's row was removed, which is a branch on what the scene
// contains, and a declaration cannot express one. It sat in the CPU backend's
// hand-written launcher, so a rule about which rows are prescribed lived in a
// backend entry point; here it is in the neutral body all three compilers
// build.
//
// A FREE ROW IS LEFT ALONE RATHER THAN SEEDED WITH ZERO, and the difference is
// not cosmetic: the caller clears `dx` before this pass and the free rows carry
// whatever the Newton driver put there, so writing a zero here would be this
// pass making a claim about rows it does not own. Only a removed row has a
// closed-form correction, and only a removed row is written.
// FOUR ELEMENT GATHERS AND NO SCATTER. The body returns void and writes through
// the last gathered element, which is an lvalue; a scatter would write on every
// thread, which is exactly what the gate above forbids. Both positions are read
// as `Vec3f`, so the entry computes nothing: the one subtraction happens inside
// the body above.
[[seam::entry]]
[[seam::device_fn]] inline void
dx_seed(const Vec3f &eval_x,
            const Vec3f &target,
            const VertexProp &prop,
            Vec3f &dx) {
    if (prop.fix_index > 0) {
        dx = prescribed_increment(eval_x, target);
    }
}
