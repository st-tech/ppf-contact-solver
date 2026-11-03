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
// EVERY SUM OF PRODUCTS IS SPELLED AS AN `fmath::fma` NEST. A contracting
// compiler (nvcc --fmad=true, Metal, the host at -ffp-contract=fast) may fuse
// `a * b + c * d` either way or not at all, and each choice rounds
// differently. CUDA 13.2 fused the centroid and the weighted sums the other
// way from CUDA 12 and the host compilers, and the stitch golden moved by a few
// ulp. With every product either alone or inside an explicit fma, no compiler
// has a choice left, so all backends give the same bits.
//
// EVERY SCALAR DIVISION IS SPELLED `fmath::div`, which is `/` on CUDA (so the
// CUDA arithmetic is unchanged) and `precise::divide` in MSL. The distinction
// is load-bearing: MSL's plain `/` is not correctly rounded, and a backend whose
// division is not would give this chain a different answer from the host image
// of the same body, where `/` is. The one division that is NOT spelled here is
// inside `SMat::norm`, which reaches the seam's square root and is correctly
// rounded on both.
//
// THE REST LENGTH IS EITHER AUTHORED OR DERIVED, AND A FLAG SAYS WHICH. A stitch
// between two bodies that rest apart (a couching tack from a floss strand to
// the ground yarn under it, whose center lines sit about two yarn radii apart)
// carries its own rest length, so it holds that distance with no preload. A
// stitch without one takes half the barycentric mean contact gap, which closes
// a seam to contact distance. The choice is a separate `has_rest_length` flag
// rather than a reserved value of `rest_length`, because every float is a
// length someone could mean, and the derived length is computed either way so
// the unflagged path is the same arithmetic it has always been. The energy is
// in STRAIN units, `l / l0 - 1`, so a longer rest length makes a given
// stiffness a softer spring in absolute terms: the force at an extension `d`
// is `stiffness * d / l0`.
//
// The caller passes the six per-vertex `ghat` and `offset` values rather than
// the length cap, so the whole numeric chain from scene data to scattered
// blocks lives in one body and a fixture cannot certify a transliteration of
// half of it.
[[seam::device_fn]] inline void stitch_force_hessian(
    const Vec3f *x, const float *w,
    const float *ghat, const float *offset,
    float stitch_length_factor, float stiffness,
    float rest_length, unsigned has_rest_length,
    Mat3x6f &gradient,
    Mat18x18f &hessian) {
    // The derived rest length is the mean contact gap over the six slots,
    // weighted by the same barycentric weights the spring itself uses, halved
    // because a gap is measured between two surfaces. An authored one replaces
    // it; the host has already refused one that is not finite and positive.
    float gap_sum = w[0] * ghat[0];
    for (unsigned k = 1; k < 6; ++k) {
        gap_sum = fmath::fma(w[k], ghat[k], gap_sum);
    }
    const float gap_rest_length = fmath::div(gap_sum, 2.0f);
    const float l0 = has_rest_length != 0u ? rest_length : gap_rest_length;
    // The two endpoints' contact offsets, each the LARGEST over that
    // endpoint's three slots. A max rather than a barycentric blend, because
    // the offset is the shell the barrier keeps clear and a blend would let the
    // cap fall inside the thickest of the three.
    const float source_offset =
        fmath::max(fmath::max(offset[0], offset[1]), offset[2]);
    const float target_offset =
        fmath::max(fmath::max(offset[3], offset[4]), offset[5]);
    const float l_cap =
        fmath::fma(stitch_length_factor, l0, source_offset) + target_offset;

    // The centroid of the six slots, formed first so the differences below stay
    // at slot-separation scale. Differencing two large nearby coordinates
    // cancels their leading digits and leaves the round-off as the answer, so
    // the offsets are taken against this centroid rather than against each
    // other. The centroid is a plain sum over six, with no product to fuse.
    Vec3f cog;
    Vec3f t;
    for (unsigned d = 0; d < 3; ++d) {
        float sum = x[0][d];
        for (unsigned k = 1; k < 6; ++k) {
            sum = sum + x[k][d];
        }
        cog[d] = fmath::div(sum, 6.0f);
        float z0 = w[0] * (x[0][d] - cog[d]);
        float z1 = w[3] * (x[3][d] - cog[d]);
        for (unsigned k = 1; k < 3; ++k) {
            z0 = fmath::fma(w[k], x[k][d] - cog[d], z0);
            z1 = fmath::fma(w[k + 3], x[k + 3][d] - cog[d], z1);
        }
        t[d] = z0 - z1;
    }
    // The length as an fma nest too; `SMat::norm` sums plain products.
    const float t_length = fmath::sqrt(
        fmath::fma(t[2], t[2], fmath::fma(t[1], t[1], t[0] * t[0])));
    // The cap enters as a MINIMUM on the measured length, not as a branch: past
    // it the spring stops lengthening and `r` saturates, which is what makes
    // the two coefficients below reach their clamps.
    const float l = fmath::min(l_cap, t_length);
    Vec3f n;
    for (unsigned dimension = 0; dimension < 3; ++dimension) {
        n[dimension] = fmath::div(t[dimension], l);
    }

    // dt/dx is w_k I for the source slots and -w_k I for the target ones, so
    // g = dtdx^T n and dtdx^T dtdx are written entry by entry from the signed
    // weights rather than as 18-wide matrix products.
    float sw[6];
    for (unsigned k = 0; k < 6; ++k) {
        sw[k] = k < 3 ? w[k] : -w[k];
    }
    const Vec3f dedt = (fmath::div(l, l0) - 1.0f) * n;
    SVecf<18> g;
    for (unsigned k = 0; k < 6; ++k) {
        for (unsigned d = 0; d < 3; ++d) {
            g[3 * k + d] = sw[k] * n[d];
        }
    }
    const float r = fmath::div(l - l0, l);
    const float c0 = fmath::div(fmath::max(0.0f, 1.0f - r), l0);
    const float c1 = fmath::max(0.0f, fmath::div(r, l0));

    gradient.col(0) = w[0] * dedt;
    gradient.col(1) = w[1] * dedt;
    gradient.col(2) = w[2] * dedt;
    gradient.col(3) = -w[3] * dedt;
    gradient.col(4) = -w[4] * dedt;
    gradient.col(5) = -w[5] * dedt;
    // H = c0 g g^T + c1 dtdx^T dtdx. The second term is nonzero only where the
    // two axes match, and there it is one fma with the first.
    for (unsigned j = 0; j < 18; ++j) {
        for (unsigned i = 0; i < 18; ++i) {
            const float outer = c0 * g[i];
            hessian(i, j) = i % 3 == j % 3
                                ? fmath::fma(outer, g[j], (c1 * sw[i / 3]) * sw[j / 3])
                                : outer * g[j];
        }
    }

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
    float rest_length, unsigned has_rest_length,
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
                         stiffness, rest_length, has_rest_length, gradient,
                         block);
    hessian[element] = block;
}

// Six positions, six contact gaps and six offsets read THROUGH the stitch's own
// six-slot index list, the six barycentric weights as this element's own run of
// the weight array, four per-element scalars (the length factor, the stiffness,
// the authored rest length and the flag saying whether there is one), the
// gradient as a non-const gather and the Hessian as a base pointer with the
// thread index forwarded beside it.
[[seam::entry(count, element)]] void stitch_force_hessian_gathered(
    [[seam::through]] const Vec3f *x,
    [[seam::indices(6)]] const unsigned *stitch_index,
    [[seam::bound]] unsigned vertex_count,
    [[seam::through]] const float *vertex_ghat,
    [[seam::through]] const float *vertex_offset,
    [[seam::stride(6)]] const float *stitch_weight,
    const float *length_factor,
    const float *stiffness,
    const float *rest_length,
    const unsigned *has_rest_length,
    Mat3x6f *gradient,
    Mat18x18f *hessian,
    unsigned element,
    unsigned count);
