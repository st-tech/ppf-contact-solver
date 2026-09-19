// File: face_pressure.kernel.cpp
// Code: GitHub Copilot
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. This file is plain C++ and belongs to no backend: no
// preprocessor conditional, no macro of its own, and no spelling that only nvcc
// or only the Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py
// renders it into the three forms the three compilers read, and the build hands
// each compiler its own form. The two facts a backend cannot infer are written
// as C++ attributes: `[[seam::device_fn]]` is the execution space, and
// `[[seam::thread]]` is the address space of a reference parameter. MSL
// requires an address space on every reference; CUDA and the host have one
// address space and are handed the same declarations with it removed.
//
// The matrix and vector names and linalg::eig::symm3x3 come from whatever
// declares them for the backend that is compiling, which is data.hpp under nvcc
// and on the host and the shader prologue's aliases under MSL. `fmath::abs`
// comes from that backend's prologue in the same way: cpp/seam under nvcc and on
// the host, the prologue in metal/shader_compiler.mm under MSL. `svd3x3_rv` is
// the one name this file includes for itself, below, because it is the one that
// no prologue declares.

// THE HESSIAN BELOW CALLS `svd3x3_rv`, so this file names it and must include
// it. Until this file carried an entry of its own it was only ever compiled
// inside `energy/face_force.kernel.cpp`, which includes `svd3x3.kernel.cpp`
// nine lines above including this one, and inside the host translation unit,
// which reaches it the same way; both supplied the name by accident of order.
// A generated entry rendering is compiled ALONE, where nothing does.
#include "../../utility/svd3x3.kernel.cpp"

[[seam::device_fn]] inline Mat3x3f
face_pressure_gradient(float pressure, const Vec3f &v0,
                           const Vec3f &e1,
                           const Vec3f &e2) {
    const float coefficient = -pressure / 6.0f;
    Mat3x3f gradient;
    gradient.col(0) =
        coefficient * (e1.cross(e2) + (e1 - e2).cross(v0));
    gradient.col(1) = coefficient * e2.cross(v0);
    gradient.col(2) = coefficient * v0.cross(e1);
    return gradient;
}

[[seam::device_fn]] inline Mat9x9f
face_pressure_hessian(float pressure, const Vec3f &v0,
                          const Vec3f &v1,
                          const Vec3f &v2) {
    Mat3x3f positions;
    positions.col(0) = v0;
    positions.col(1) = v1;
    positions.col(2) = v2;
    Mat3x3f u;
    Vec3f sigma;
    Mat3x3f vt;
    svd3x3_rv(positions, u, sigma, vt);

    const Mat3x3f v = vt.transpose();
    Mat3x3f diagonal_subspace;
    diagonal_subspace << 0.0f, sigma[2], sigma[1], sigma[2], 0.0f, sigma[0],
        sigma[1], sigma[0], 0.0f;
    Vec3f eigenvalues;
    Mat3x3f eigenvectors;
    linalg::eig::symm3x3(diagonal_subspace, eigenvalues, eigenvectors);

    constexpr float inverse_sqrt_two = 0.7071067811865475f;
    const float coefficient = pressure / 6.0f;
    float weights[5];
    Vec3f columns[5][3];
    unsigned kept = 0;
    const unsigned swap_i[3] = {1u, 2u, 2u};
    const unsigned swap_j[3] = {0u, 0u, 1u};
    const float swap_sigma[3] = {sigma[2], sigma[1], sigma[0]};
    for (unsigned pair = 0; pair < 3; ++pair) {
        const float weight = coefficient * swap_sigma[pair];
        if (weight > 0.0f) {
            const unsigned i = swap_i[pair];
            const unsigned j = swap_j[pair];
            const Vec3f ui = u.col(i);
            const Vec3f uj = u.col(j);
            weights[kept] = weight;
            for (unsigned vertex_index = 0; vertex_index < 3;
                 ++vertex_index) {
                columns[kept][vertex_index] =
                    inverse_sqrt_two *
                    (v(vertex_index, j) * ui + v(vertex_index, i) * uj);
            }
            ++kept;
        }
    }
    for (unsigned mode = 0; mode < 3; ++mode) {
        if (eigenvalues[mode] < -1.0e-12f) {
            const float weight = coefficient * fmath::abs(eigenvalues[mode]);
            if (weight > 0.0f) {
                const Vec3f abc = eigenvectors.col(mode);
                weights[kept] = weight;
                for (unsigned vertex_index = 0; vertex_index < 3;
                     ++vertex_index) {
                    columns[kept][vertex_index] =
                        abc[0] * v(vertex_index, 0) * u.col(0) +
                        abc[1] * v(vertex_index, 1) * u.col(1) +
                        abc[2] * v(vertex_index, 2) * u.col(2);
                }
                ++kept;
            }
        }
    }

    Mat9x9f hessian = Mat9x9f::Zero();
    for (unsigned mode = 0; mode < kept; ++mode) {
        for (unsigned a = 0; a < 3; ++a) {
            for (unsigned b = 0; b < 3; ++b) {
                hessian.block<3, 3>(3 * a, 3 * b) +=
                    weights[mode] * columns[mode][a] *
                    columns[mode][b].transpose();
            }
        }
    }
    return hessian;
}

// ---------------------------------------------------------------------------
// The pass that puts the two above into the face accumulators.
// ---------------------------------------------------------------------------

// ONE FACE'S PRESSURE TERM, ADDED to the elastic gradient and Hessian this face
// already carries.
//
// `v0` IS ABSOLUTE ON PURPOSE. The per-face pressure gradient is genuinely
// translation-VARIANT, only the sum over a closed surface being invariant, so
// there is no origin to difference against. What `face_pressure_gradient` does
// instead is re-associate, rewriting `v1 x v2` as `e1 x e2 + (e1 - e2) x v0`
// over the edge vectors `e1 = x1 - x0` and `e2 = x2 - x0`. Those are small and
// well determined, so the absolute magnitude enters only products whose other
// factor is an edge, never a cross product of two nearly parallel large
// vectors, whose leading digits would cancel. The Hessian still SVDs the
// absolute position matrix, which is a property of the per-face formulation
// rather than of this call.
//
// A FACE WITH NO PRESSURE ADDS NOTHING, tested here rather than by the caller,
// because the caller dispatches over every shell face and the parameter is
// per-face.
//
// THE TWO DESTINATIONS ARE `[[seam::device]]`, matching `face_baraffwitkin`
// below the same reasoning: a `[[seam::stride(N)]]` argument is the buffer
// advanced to this element's own slot, so it stays in device memory and the
// address space travels with the pointer. Only the three positions and the
// pressure are `[[seam::thread]]`, those being the gathered values the MSL
// entry copies into thread space before the call. CUDA and the host have one
// address space and compile either spelling, so this is a Metal-only
// distinction that no other backend's build can check.
[[seam::device_fn]] inline void face_pressure_embed(
    const Vec3f &x0, const Vec3f &x1,
    const Vec3f &x2, float pressure,
    float *gradient, float *hessian) {
    if (!(pressure > 0.0f)) {
        return;
    }
    const Vec3f v0 = x0;
    const Vec3f e1 = (x1 - x0).template cast<float>();
    const Vec3f e2 = (x2 - x0).template cast<float>();
    const Vec3f v1 = v0 + e1;
    const Vec3f v2 = v0 + e2;
    const Mat3x3f g = face_pressure_gradient(pressure, v0, e1, e2);
    const Mat9x9f h = face_pressure_hessian(pressure, v0, v1, v2);
    // COLUMN-MAJOR, as every other face accumulator in this tree: column `k` of
    // the gradient is vertex `k`'s three components.
    for (unsigned c = 0; c < 3u; ++c) {
        for (unsigned r = 0; r < 3u; ++r) {
            gradient[3u * c + r] += g(r, c);
        }
    }
    for (unsigned c = 0; c < 9u; ++c) {
        for (unsigned r = 0; r < 9u; ++r) {
            hessian[9u * c + r] += h(r, c);
        }
    }
}

// The entry. One thread per shell face.
//
// TWO ACCUMULATORS AND NO SCATTER, which is the seam's rule read rather than
// worked around: a thread ADDS into a fixed run of each, nine floats of the
// gradient and eighty-one of the Hessian, and that is the `[[seam::stride(N)]]`
// shape. There is no return value, so there is nothing for a
// `[[seam::scatter]]` to carry and the entry declares none.
//
// THE PRESSURE IS ONE FLOAT PER FACE, not a `FaceParam` gather, because
// `FaceParam` is DEDUPLICATED across faces with identical materials and the
// index that resolves it is the driver's. Every other per-face material this
// pass needs already travels that way, `mass`, `lambda` and `damping` each
// being a float array the host fills through `params.get(prop.param_index)`,
// and taking a different route for this one would be a second answer to a
// question the file already answers.
[[seam::entry(count)]] void face_pressure_embed(
    [[seam::through]] const Vec3f *vert,
    [[seam::indices(3)]] const unsigned *face,
    [[seam::bound]] unsigned vertex_count,
    const float *pressure,
    [[seam::stride(9)]] float *gradient,
    [[seam::stride(81)]] float *hessian,
    unsigned count);
