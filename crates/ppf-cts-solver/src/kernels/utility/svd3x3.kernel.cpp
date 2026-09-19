// File: svd3x3.kernel.cpp
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

[[seam::device_fn]] inline void
svd3x3(const Mat3x3f &input, Mat3x3f &u,
           Vec3f &sigma, Mat3x3f &vt) {
    float a[3][3];
    float v[3][3];
    for (unsigned row = 0; row < 3; ++row) {
        for (unsigned column = 0; column < 3; ++column) {
            a[row][column] = input(row, column);
            v[row][column] = row == column ? 1.0f : 0.0f;
        }
    }
    for (unsigned sweep = 0; sweep < 5; ++sweep) {
        for (unsigned p = 0; p < 3; ++p) {
            for (unsigned q = p + 1; q < 3; ++q) {
                float aa = 0.0f;
                float bb = 0.0f;
                float gamma = 0.0f;
                for (unsigned row = 0; row < 3; ++row) {
                    aa += a[row][p] * a[row][p];
                    bb += a[row][q] * a[row][q];
                    gamma += a[row][p] * a[row][q];
                }
                const float zeta = fmath::div(bb - aa, 2.0f * gamma);
                const float t =
                    gamma == 0.0f
                        ? 0.0f
                        : fmath::div(zeta >= 0.0f ? 1.0f : -1.0f,
                                   fmath::abs(zeta) +
                                       fmath::sqrt(1.0f + zeta * zeta));
                const float cosine =
                    fmath::div(1.0f, fmath::sqrt(1.0f + t * t));
                const float sine = cosine * t;
                for (unsigned row = 0; row < 3; ++row) {
                    const float ap = a[row][p];
                    const float aq = a[row][q];
                    a[row][p] = cosine * ap - sine * aq;
                    a[row][q] = sine * ap + cosine * aq;
                }
                for (unsigned row = 0; row < 3; ++row) {
                    const float vp = v[row][p];
                    const float vq = v[row][q];
                    v[row][p] = cosine * vp - sine * vq;
                    v[row][q] = sine * vp + cosine * vq;
                }
            }
        }
    }

    float singular_value[3];
    float normalized[3][3];
    for (unsigned column = 0; column < 3; ++column) {
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
    unsigned index[3] = {0u, 1u, 2u};
    for (unsigned i = 0; i < 3; ++i) {
        for (unsigned j = i + 1; j < 3; ++j) {
            if (singular_value[index[j]] < singular_value[index[i]]) {
                const unsigned temporary = index[i];
                index[i] = index[j];
                index[j] = temporary;
            }
        }
    }
    for (unsigned column = 0; column < 3; ++column) {
        sigma[column] = singular_value[index[column]];
        for (unsigned row = 0; row < 3; ++row) {
            u(row, column) = normalized[row][index[column]];
            vt(column, row) = v[row][index[column]];
        }
    }
}

// The two entry points, declared once each and rendered for four targets. Both
// are four element gathers and no scatter: each body returns void and writes
// its three factors through references, so a non-const gather IS the write and
// each generated statement is its launcher's
// `f(in[i], u_out[i], sigma_out[i], vt_out[i])` character for character.
//
// WHAT THE DRIVER DISPATCHES IS THE REFLECTION-CORRECTED FORM. The plain
// factorization below it carries an entry all the same, because the alternative
// is a hand-written launcher for it, which is a mirror pair with nothing linking
// its halves. Its id is declared past `id::COUNT` in `src/driver/kernels.rs` with no
// row in the dispatch table, the arrangement `vec_fill` and
// `vec_combine_indirect` already use: `decl_of` then answers
// `Fault::MissingKernel` by name if anything ever dispatches it, rather than
// reading past the table.
[[seam::entry]]
[[seam::device_fn]] inline void
svd3x3_rv(const Mat3x3f &input, Mat3x3f &u,
              Vec3f &sigma, Mat3x3f &vt) {
    svd3x3(input, u, sigma, vt);
    const float determinant_u = u.determinant();
    const float determinant_vt = vt.determinant();
    unsigned minimum = 0;
    if (sigma[1] < sigma[minimum]) {
        minimum = 1;
    }
    if (sigma[2] < sigma[minimum]) {
        minimum = 2;
    }
    if (determinant_u < 0.0f && determinant_vt > 0.0f) {
        for (unsigned row = 0; row < 3; ++row) {
            u(row, minimum) = -u(row, minimum);
        }
        sigma[minimum] = -sigma[minimum];
    } else if (determinant_u > 0.0f && determinant_vt < 0.0f) {
        for (unsigned column = 0; column < 3; ++column) {
            vt(minimum, column) = -vt(minimum, column);
        }
        sigma[minimum] = -sigma[minimum];
    }
}

[[seam::entry(count)]] void svd3x3(
    const Mat3x3f *input,
    Mat3x3f *u,
    Vec3f *sigma,
    Mat3x3f *vt,
    unsigned count);
