// File: stitch.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. Plain C++ belonging to no backend: no preprocessor
// conditional, no macro of its own, and no spelling that only one of the three
// compilers accepts. ppf-cts-compute/seam/kernelgen.py renders it into the .cu, the .metal
// and the .cpp that nvcc, the Metal shader compiler and a host C++ compiler
// read. The two facts a compiler cannot infer are C++ attributes:
// `[[seam::device_fn]]` is the execution space, and `[[seam::thread]]` is the
// address space of a pointer or reference parameter. MSL requires the second on
// every one of them; CUDA and the host have a single address space and are
// handed the same declarations with it removed.
//
// No include of its own. `Vec3f`, `SMatf`, `Mat3x3f`, `float` and the
// `fmath::` arithmetic all arrive from whatever the backend compiles ahead of
// this body, which is data.hpp under nvcc and on the host, and the shader
// prologue plus the earlier segments under MSL.

// The cross-stitch spring, as one body for both backends.
//
// A stitch is a 6-slot barycentric-to-barycentric pair: slots 0..2 name the
// SOURCE triangle and slots 3..5 the TARGET, each with barycentric weights that
// sum to one. A non-SOLID endpoint degenerates to index {s, s, s} with weights
// {1, 0, 0}, which recovers single-vertex behavior through the same arithmetic;
// no branch distinguishes the two, and the degenerate form folds three
// contributions onto one vertex and onto one CSR slot rather than three.
//
// THE HESSIAN IS PSD BY CONSTRUCTION, unconditionally and with no flag reaching
// it. Both terms are positive-semidefinite forms, `g g^T` and `dtdx^T dtdx`,
// and the two coefficients that scale them are clamped at zero by `fmath::max`.
// That clamp IS the projection: past the length cap the raw second derivative
// of the spring energy turns indefinite, and the clamp drops exactly the
// offending mode while the gradient above it stays exact, which is the
// projected-Newton pattern the rest of the assembly uses.
//
// EVERY SCALAR DIVISION IS SPELLED `fmath::div`, which is `/` on CUDA (so the
// CUDA arithmetic is unchanged) and `precise::divide` in MSL. The distinction
// is load-bearing: MSL's plain `/` is not correctly rounded, and the parity
// fixture compares this chain to the bit against a host image of the same body,
// where `/` is. The one division that is NOT spelled here is inside
// `SMat::norm`, which reaches the seam's square root and is correctly rounded
// on both.
//
// The caller passes the six per-vertex `ghat` and `offset` values rather than
// the length cap, so the whole numeric chain from scene data to scattered
// blocks lives in one body and a fixture cannot certify a transliteration of
// half of it.
[[seam::device_fn]] inline void stitch_force_hessian(
    const Vec3f *x, const float *w,
    const float *ghat, const float *offset,
    float stitch_length_factor, float stiffness,
    Mat3x6f &gradient,
    Mat18x18f &hessian) {
    // The rest length is the mean contact gap over the six slots, weighted by
    // the same barycentric weights the spring itself uses, halved because a gap
    // is measured between two surfaces.
    const float l0 =
        fmath::div(w[0] * ghat[0] + w[1] * ghat[1] + w[2] * ghat[2] +
                     w[3] * ghat[3] + w[4] * ghat[4] + w[5] * ghat[5],
                 2.0f);
    // The two endpoints' contact offsets, each the LARGEST over that
    // endpoint's three slots. A max rather than a barycentric blend, because
    // the offset is the shell the barrier keeps clear and a blend would let the
    // cap fall inside the thickest of the three.
    const float source_offset =
        fmath::max(fmath::max(offset[0], offset[1]), offset[2]);
    const float target_offset =
        fmath::max(fmath::max(offset[3], offset[4]), offset[5]);
    const float l_cap =
        stitch_length_factor * l0 + source_offset + target_offset;

    // The centroid of the six slots, formed first so the differences below stay
    // at slot-separation scale. Differencing two large nearby coordinates
    // cancels their leading digits and leaves the round-off as the answer, so
    // the offsets are taken against this centroid rather than against each
    // other.
    const float s(fmath::div(1.0f, 6.0f));
    const Vec3f cog =
        s * x[0] + s * x[1] + s * x[2] + s * x[3] + s * x[4] + s * x[5];
    Vec3f z0 = w[0] * (x[0] - cog).cast<float>() +
               w[1] * (x[1] - cog).cast<float>() +
               w[2] * (x[2] - cog).cast<float>();
    Vec3f z1 = w[3] * (x[3] - cog).cast<float>() +
               w[4] * (x[4] - cog).cast<float>() +
               w[5] * (x[5] - cog).cast<float>();
    const Vec3f t = z0 - z1;
    // The cap enters as a MINIMUM on the measured length, not as a branch: past
    // it the spring stops lengthening and `r` saturates, which is what makes
    // the two coefficients below reach their clamps.
    const float l = fmath::min(l_cap, t.norm());
    Vec3f n;
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        n[dimension] = fmath::div(t[dimension], l);
    }

    SMatf<3, 18> dtdx;
    dtdx << w[0] * Mat3x3f::Identity(), w[1] * Mat3x3f::Identity(),
        w[2] * Mat3x3f::Identity(), -w[3] * Mat3x3f::Identity(),
        -w[4] * Mat3x3f::Identity(), -w[5] * Mat3x3f::Identity();
    const Vec3f dedt = (fmath::div(l, l0) - 1.0f) * n;
    const SVecf<18> g = dtdx.transpose() * n;
    const float r = fmath::div(l - l0, l);
    const float c0 = fmath::div(fmath::max(0.0f, 1.0f - r), l0);
    const float c1 = fmath::max(0.0f, fmath::div(r, l0));

    gradient.col(0) = w[0] * dedt;
    gradient.col(1) = w[1] * dedt;
    gradient.col(2) = w[2] * dedt;
    gradient.col(3) = -w[3] * dedt;
    gradient.col(4) = -w[4] * dedt;
    gradient.col(5) = -w[5] * dedt;
    hessian = c0 * g * g.transpose() + c1 * dtdx.transpose() * dtdx;

    // The per-object stiffness is a RAW force factor: no mass and no time-scale
    // normalization, so the authored number is what the spring pulls with. It
    // is applied in place rather than as `stiffness * gradient` at the scatter
    // so the 18x18 temporary that spelling would materialize never exists;
    // IEEE multiplication is commutative to the bit, so the two are the same
    // float.
    for (unsigned element = 0; element < 18; ++element) {
        gradient.m[element] = stiffness * gradient.m[element];
    }
    for (unsigned element = 0; element < 324; ++element) {
        hessian.m[element] = stiffness * hessian.m[element];
    }
}

// THE THREE THREAD ARRAYS THE CALLER USED TO REBUILD ABOVE THE CALL. The body
// above takes its six slots as arrays, and a generated entry passes a gathered
// buffer's elements as SEPARATE arguments, one per slot, so the rebuild has to
// happen somewhere; here is the only place it can be written once for three
// backends.
//
// ONE INDEX LIST SERVES ALL THREE GATHERS, because the position, the contact
// gap and the offset are read at the SAME six vertex slots. That is what makes
// eighteen arguments one declaration rather than three.
//
// THE 18x18 IS A BASE POINTER AND THE 3x6 IS NOT, AND THAT ASYMMETRY IS THE
// WHOLE REASON THIS CONVERTS. A generated entry COPIES a gathered element into
// thread space on MSL: at 1296 bytes the Hessian would not fit a device stack
// measured in kilobytes, while the 72-byte gradient is unremarkable. So the
// Hessian arrives as the array plus this element's index and the body writes
// through device memory, which is what the hand-written launcher already did.
//
// THE WEIGHTS ARE COPIED TOO, and for an address space rather than a size:
// `[[seam::stride(6)]]` hands over the buffer's base advanced by `6 * element`,
// which is a DEVICE pointer, and the body's parameter is a thread one.
[[seam::device_fn]] inline void stitch_force_hessian_gathered(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2, const Vec3f &x3,
    const Vec3f &x4, const Vec3f &x5,
    float ghat0, float ghat1, float ghat2, float ghat3, float ghat4,
    float ghat5, float offset0, float offset1, float offset2, float offset3,
    float offset4, float offset5, const float *weight,
    float stitch_length_factor, float stiffness,
    Mat3x6f &gradient, Mat18x18f *hessian,
    unsigned element) {
    const Vec3f node[6] = {x0, x1, x2, x3, x4, x5};
    const float gap[6] = {ghat0, ghat1, ghat2, ghat3, ghat4, ghat5};
    const float shell[6] = {offset0, offset1, offset2, offset3, offset4, offset5};
    float share[6];
    for (unsigned k = 0; k < 6; ++k) {
        share[k] = weight[k];
    }
    // The body computes into a thread-space block; placing the result is
    // the entry's business, which is the only side that knows the address
    // space its destination lives in.
    Mat18x18f block;
    stitch_force_hessian(node, share, gap, shell, stitch_length_factor,
                         stiffness, gradient, block);
    hessian[element] = block;
}

// Six positions, six contact gaps and six offsets read THROUGH the stitch's own
// six-slot index list, the six barycentric weights as this element's own run of
// the weight array, two per-element scalars, the gradient as a non-const gather
// and the Hessian as a base pointer with the thread index forwarded beside it.
[[seam::entry(count, element)]] void stitch_force_hessian_gathered(
    [[seam::through]] const Vec3f *x,
    [[seam::indices(6)]] const unsigned *stitch_index,
    [[seam::bound]] unsigned vertex_count,
    [[seam::through]] const float *vertex_ghat,
    [[seam::through]] const float *vertex_offset,
    [[seam::stride(6)]] const float *stitch_weight,
    const float *length_factor,
    const float *stiffness,
    Mat3x6f *gradient,
    Mat18x18f *hessian,
    unsigned element,
    unsigned count);
