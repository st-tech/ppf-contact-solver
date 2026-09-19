// File: translation_lock_check.kernel.cpp
// License: Apache v2.0

// THE READ-ONLY HALF OF THE TRANSLATION LOCK. It accumulates the two
// quantities `check_invariant` needs in order to JUDGE the constraint and
// modifies no position. That is the whole design of lock verification: the
// solver never snaps a state, because a snap applied after the CCD line search
// would reintroduce penetration, so every step of the check is a reduction and
// the verdict is reached on the host.

#include "translation_lock_math.hpp"

// One vertex's contribution to its lock group's perpendicular center-of-mass
// drift and to the largest per-axis displacement anywhere in that group.
//
// THE TWO ACCUMULATORS ARE DIFFERENT KINDS, AND THE SECOND ONE IS WHY THE SEAM
// CARRIES `compute::atomic_max` AT ALL. The drift is a SUM, so it is a float
// atomic add over the group's three components. The displacement is a MAXIMUM,
// and there is no float atomic maximum on every backend. The seam's
// `compute::atomic_max` therefore takes UINT-typed storage and a NON-NEGATIVE
// float, the IEEE-754 bit pattern of a non-negative float ordering the same way
// as the float itself. The value below is a maximum of magnitudes and so is
// non-negative by construction, which is the precondition that rule needs; the
// host converts the bits back to a float after the readback, so no neutral body
// spells a bitcast.
// `locks`, `positions`, `initial` and `prop` are BASE POINTERS rather than
// gathers: `locks[lock_index[i]]` is a two-level indirection whose inner index
// is DATA, and a record carries one index list. The bound on each is therefore
// the array's own length, which is what `[[seam::bound]]` would name; the two
// accumulators are indexed by a group id read out of `lock_index`, so their
// widths are the group count rather than the thread count.
[[seam::entry(i)]]
[[seam::device_fn]] inline void translation_lock_drift_row(
    const unsigned *lock_index,
    const TranslationLock *locks,
    const Vec3f *positions,
    const Vec3f *initial,
    const VertexProp *prop,
    compute::atomic_float_t *drift,
    compute::atomic_uint_t *max_displacement, unsigned i) {
    const unsigned li = lock_index[i];
    if (li == translation_lock::UNSET) {
        return;
    }
    const TranslationLock lock = locks[li];
    if (!translation_lock::translation_lock_enabled(lock)) {
        return;
    }
    // A DIFFERENCE of two positions, so the anchor's own magnitude leaves the
    // displacement before the mass scales it. Both operands are copied into
    // thread space
    // first: a `[[seam::device]]` base pointer yields a device-space lvalue,
    // which MSL cannot bind to the thread-space reference the subtraction takes.
    const Vec3f current = positions[i];
    const Vec3f anchor = initial[i];
    const Vec3f delta = (current - anchor).cast<float>();
    // THE SAME HELPER THE DRIFT ACCUMULATION USES, so the end-of-step invariant
    // and the constraint it verifies cannot disagree about which part of the
    // displacement is locked.
    const Vec3f weighted =
        prop[i].mass * translation_lock::constrained_translation(lock, delta);
    compute::atomic_add(drift + 3u * li + 0u, weighted[0]);
    compute::atomic_add(drift + 3u * li + 1u, weighted[1]);
    compute::atomic_add(drift + 3u * li + 2u, weighted[2]);
    compute::atomic_max(max_displacement + li,
                        fmath::max(fmath::abs(delta[0]),
                                   fmath::max(fmath::abs(delta[1]),
                                              fmath::abs(delta[2]))));
}
