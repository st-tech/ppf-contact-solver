// File: svd3x2.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE: plain C++ that belongs to no backend, with no
// preprocessor conditional, no macro of its own, and no spelling that only one
// of the three compilers accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the form each compiler reads. `[[seam::device_fn]]` is the execution
// space and `[[seam::thread]]` is the address space MSL requires on every
// reference type. The `fmath::` functions come from the backend prologue:
// cpp/seam under nvcc and on the host, and `kMslMacroSeam` in
// metal/shader_compiler.mm under MSL.

// The entry point, declared once and rendered for four targets: the
// `__global__` nvcc compiles, the `kernel void` the Metal shader compiler
// compiles, the `svd3x2_entry` shim a host C++ compiler compiles, and
// the Rust `#[repr(C)]` twin the driver fills.
//
// FOUR ELEMENT GATHERS AND NO SCATTER, because the body returns void and
// writes its three factors through references. A gather hands the body
// `buffer[index]`, which is an lvalue, so a non-const gather IS the write: the
// generated statement is the launcher's
// `svd3x2(in[i], u_out[i], sigma_out[i], vt_out[i])` character for
// character. The pod sizes pin the element widths in every C++ rendering, so a
// compiler laying one of these matrices out differently fails to compile
// rather than reading the wrong bytes.
[[seam::entry]]
[[seam::device_fn]] inline void
svd3x2(const Mat3x2f &input, Mat3x2f &u,
           Vec2f &sigma, Mat2x2f &vt) {
    float a[3][2];
    for (unsigned row = 0; row < 3; ++row) {
        a[row][0] = input(row, 0);
        a[row][1] = input(row, 1);
    }
    float v[2][2] = {{1.0f, 0.0f}, {0.0f, 1.0f}};

    float aa = 0.0f;
    float bb = 0.0f;
    float gamma = 0.0f;
    for (unsigned row = 0; row < 3; ++row) {
        aa += a[row][0] * a[row][0];
        bb += a[row][1] * a[row][1];
        gamma += a[row][0] * a[row][1];
    }
    const float zeta = fmath::div(bb - aa, 2.0f * gamma);
    const float t =
        gamma == 0.0f
            ? 0.0f
            : fmath::div(zeta >= 0.0f ? 1.0f : -1.0f,
                       fmath::abs(zeta) + fmath::sqrt(1.0f + zeta * zeta));
    const float cosine = fmath::div(1.0f, fmath::sqrt(1.0f + t * t));
    const float sine = cosine * t;
    for (unsigned row = 0; row < 3; ++row) {
        const float p = a[row][0];
        const float q = a[row][1];
        a[row][0] = cosine * p - sine * q;
        a[row][1] = sine * p + cosine * q;
    }
    for (unsigned row = 0; row < 2; ++row) {
        const float p = v[row][0];
        const float q = v[row][1];
        v[row][0] = cosine * p - sine * q;
        v[row][1] = sine * p + cosine * q;
    }

    float singular_value[2];
    float normalized[3][2];
    for (unsigned column = 0; column < 2; ++column) {
        float norm_squared = 0.0f;
        for (unsigned row = 0; row < 3; ++row) {
            norm_squared += a[row][column] * a[row][column];
        }
        const float norm = fmath::sqrt(norm_squared);
        singular_value[column] = norm;
        const float inverse = norm > 0.0f ? fmath::div(1.0f, norm) : 0.0f;
        for (unsigned row = 0; row < 3; ++row) {
            normalized[row][column] = a[row][column] * inverse;
        }
    }
    const unsigned first =
        singular_value[1] < singular_value[0] ? 1u : 0u;
    const unsigned second = 1u - first;
    sigma[0] = singular_value[first];
    sigma[1] = singular_value[second];
    for (unsigned row = 0; row < 3; ++row) {
        u(row, 0) = normalized[row][first];
        u(row, 1) = normalized[row][second];
    }
    for (unsigned row = 0; row < 2; ++row) {
        vt(0, row) = v[row][first];
        vt(1, row) = v[row][second];
    }
}

// The same factorization with its singular values SHIFTED by one, plus the
// larger of the two shifted values.
//
// THE SHIFT IS THE STRAIN LIMITER'S OWN COORDINATE and not a rescaling: a
// singular value of one is an unstretched direction, so `sigma - 1` is the
// strain the barrier is a function of and its larger entry is the gate the
// limiter admits a face on. Both were spelled in the CPU backend's hand-written
// launcher, which is host logic in a backend entry point; they are here so the
// three compilers build them from one set of bytes.
//
// THE ROUND TRIP IS NOT THE IDENTITY IN fp32, which is why the restore beside
// it is a separate pass rather than a discarded intermediate: `sigma - 1` loses
// the low bits of `sigma` once `|sigma - 1| > 1`, and adding one back does not
// return them, so the spectral force and Hessian are evaluated at the restored
// value and not at the value this factorization produced.
// FIVE ELEMENT GATHERS AND NO SCATTER, for the reason `svd3x2`'s entry
// above states: the body returns void and writes its four outputs through
// references, and a gather hands the body `buffer[index]`, which is an lvalue.
// The last output is one float per element rather than a matrix, so it carries
// no `[[seam::pod]]`: there is no layout for a compiler to disagree about.
[[seam::entry]]
[[seam::device_fn]] inline void svd3x2_shifted(
    const Mat3x2f &input, Mat3x2f &u,
    Vec2f &shifted_sigma, Mat2x2f &vt,
    float &largest_shifted) {
    Vec2f sigma;
    svd3x2(input, u, sigma, vt);
    shifted_sigma = sigma - Vec2f::Ones();
    largest_shifted = shifted_sigma.maxCoeff();
}

// THE RESTORE, `sigma = shifted_sigma + 1`, which puts the singular values back
// in the coordinate the two spectral stages read them in.
//
// IT IS A PASS AND NOT A DISCARDED INTERMEDIATE, for the reason the shift above
// states: `sigma - 1` drops the low bits of `sigma` once the face is stretched
// past two, and adding one back does not return them, so the spectral force and
// Hessian are evaluated at THIS value rather than at the one the factorization
// produced. Reconstructing it from the shift instead of storing it would put
// the same rounding back.
//
// The arithmetic was spelled in the CPU backend's hand-written launcher, which
// is a value the three compilers each have to agree on written in a place only
// one of them reads. Here it is one body and one declaration.
[[seam::device_fn]] inline Vec2f shell_strain_restore_sigma(
    const Vec2f &shifted_sigma) {
    return shifted_sigma + Vec2f::Ones();
}

// ONE ELEMENT GATHER AND ONE ELEMENT SCATTER, the smallest shape an entry
// declaration takes: the body is handed the element and its return value is the
// destination.
[[seam::entry(count)]] void shell_strain_restore_sigma(
    const Vec2f *shifted,
    Vec2f *restored,
    unsigned count);
