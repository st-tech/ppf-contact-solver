// File: friction_frame.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE: the friction's coupling across Cosserat frames in
// the augmented solve's operator, which only the augmented solve dispatches.
//
// `distance.hpp` FIRST, for the type vocabulary a kernel body does not include
// itself (`contact_narrow.kernel.cpp` says why the order is load-bearing), then
// the friction memory, whose record layout and slip Jacobian this applies.
#include "distance.hpp"
#include "friction_memory.kernel.cpp"

// THE FRICTION'S FRAME COUPLING, applied matrix free in the augmented solve's
// operator (`pcg::solve_cosserat`), one thread per frozen record entry. A
// framed pair's frozen friction is `J^T K J` over its vertices and its sides'
// frames, `K = stiffness P` the tangential stiffness the frozen pass kept
// (`frozen_stiffness`, its multiplicity in it) and `J = [J_x | G_a | G_b]` the
// slip's Jacobian (`friction_framed_jacobian`). The vertex block `J_x^T K J_x`
// is in the assembled matrix and each frame's own `G_s^T K G_s` is in its
// diagonal block, so this adds only the rest: on a vertex row
// `J_k^T K (sum_s G_s u_s)`, and on a frame row `G_s^T K` of the slip of
// everything but that frame, `J_x u_x + G_t u_t`. The augmented operator then
// applies the pair's friction exactly, so its Newton direction is the true one.
// The couplings must use the blocks' own Hessian: one stiffness throughout is
// PSD, a coupling stiffer than its blocks is indefinite. `input` and `output`
// are augmented vectors, three vertex rows a vertex then three frame rows a
// slot from `theta_base`.
//
// ITS TERMS ENTER THE CURVATURE GUARD'S ROUND-OFF SCALE (`absolute`, the vertex
// operator's per-row `sum|contribution|`, which `solve_cosserat` folds). The
// coupling is formed by cancellation of stiff stuck-friction terms, so the
// computed `p^T A p` carries their round-off; left out of the scale, a sliding
// yarn's curvature of order 1e-5 of that scale was read as an indefinite operator
// and stopped the run. Each term's magnitude is taken before any cancellation:
// `|J|` and `|G|` elementwise, the projection bounded by twice the identity.
// The guard's verdict is sound only when every rounded term is in its scale.
//
// A VERTEX THE DIRICHLET PASS REMOVED IS NOT A ROW OF THIS SYSTEM. That pass
// zeroes a removed vertex's blocks and gives it the identity, so its step is
// its prescribed value and nothing couples to it; `dof_mask` (one where the
// vertex is removed, `vertex_dof_removal_mask`) keeps the coupling off its row
// and its column alike, or the operator would stop being the reduced matrix.
// Elementwise absolute values, for the round-off scale's term magnitudes.
[[seam::device_fn]] inline Mat3x3f friction_frame_abs(const Mat3x3f &m) {
    Mat3x3f out;
    for (unsigned e = 0; e < 9u; ++e) {
        out.m[e] = fmath::abs(m.m[e]);
    }
    return out;
}

[[seam::device_fn]] inline Vec3f friction_frame_abs(const Vec3f &v) {
    return Vec3f(fmath::abs(v[0]), fmath::abs(v[1]), fmath::abs(v[2]));
}

[[seam::entry(entry)]]
[[seam::device_fn]] inline void friction_frame_apply(
    const Vec3f *x0, const unsigned *record_vertex, const float *record_float,
    const unsigned *record_count, unsigned record_capacity,
    const float *frozen_stiffness, const unsigned *dof_mask, const float *input,
    compute::atomic_float_t *output, compute::atomic_float_t *absolute,
    unsigned theta_base, unsigned entry) {
    if (entry >= record_count[0] || entry >= record_capacity) {
        return;
    }
    const float stiffness = frozen_stiffness[entry];
    if (!(stiffness > 0.0f)) {
        return;
    }
    const float *in = record_float + FRICTION_MEMORY_FLOATS * entry;
    unsigned v[FRICTION_MEMORY_VERTICES];
    float w[FRICTION_MEMORY_VERTICES];
    for (unsigned k = 0; k < FRICTION_MEMORY_VERTICES; ++k) {
        v[k] = record_vertex[FRICTION_RECORD_WORDS * entry + k];
        w[k] = in[6u + k];
    }
    const unsigned slot[2] = {record_vertex[FRICTION_RECORD_WORDS * entry + 4u],
                              record_vertex[FRICTION_RECORD_WORDS * entry + 5u]};
    const float radius[2] = {in[14], in[15]};
    const Vec3f normal(in[3], in[4], in[5]);
    Mat3x3f jacobian[FRICTION_MEMORY_VERTICES];
    Mat3x3f frame_jacobian[2];
    friction_framed_jacobian(x0, v, w, slot, radius, normal, jacobian,
                             frame_jacobian);
    // K u = stiffness (u - n (n . u)).
    Vec3f vertex_slip = Vec3f::Zero();
    for (unsigned k = 0; k < FRICTION_MEMORY_VERTICES; ++k) {
        if (v[k] != FRICTION_MEMORY_UNUSED && dof_mask[v[k]] == 0u) {
            const Vec3f u(input[3u * v[k]], input[3u * v[k] + 1u],
                          input[3u * v[k] + 2u]);
            vertex_slip += jacobian[k] * u;
        }
    }
    Vec3f frame_slip[2] = {Vec3f::Zero(), Vec3f::Zero()};
    for (unsigned s = 0; s < 2u; ++s) {
        if (slot[s] != FRICTION_MEMORY_UNUSED) {
            const unsigned t = theta_base + 3u * slot[s];
            const Vec3f u(input[t], input[t + 1u], input[t + 2u]);
            frame_slip[s] = frame_jacobian[s] * u;
        }
    }
    const Vec3f frames = frame_slip[0] + frame_slip[1];
    // The magnitudes of the slips' terms, before they cancel.
    Vec3f vertex_size = Vec3f::Zero();
    for (unsigned k = 0; k < FRICTION_MEMORY_VERTICES; ++k) {
        if (v[k] != FRICTION_MEMORY_UNUSED && dof_mask[v[k]] == 0u) {
            vertex_size += friction_frame_abs(jacobian[k]) *
                           friction_frame_abs(Vec3f(input[3u * v[k]], input[3u * v[k] + 1u],
                                                    input[3u * v[k] + 2u]));
        }
    }
    Vec3f frame_size[2] = {Vec3f::Zero(), Vec3f::Zero()};
    for (unsigned s = 0; s < 2u; ++s) {
        if (slot[s] != FRICTION_MEMORY_UNUSED) {
            const unsigned t = theta_base + 3u * slot[s];
            frame_size[s] = friction_frame_abs(frame_jacobian[s]) *
                            friction_frame_abs(Vec3f(input[t], input[t + 1u], input[t + 2u]));
        }
    }
    float magnitude = 0.0f;
    const Vec3f k_frames = stiffness * (frames - normal * normal.dot(frames));
    for (unsigned k = 0; k < FRICTION_MEMORY_VERTICES; ++k) {
        if (v[k] == FRICTION_MEMORY_UNUSED || dof_mask[v[k]] != 0u) {
            continue;
        }
        const Vec3f y = jacobian[k].transpose() * k_frames;
        for (unsigned d = 0; d < 3u; ++d) {
            if (y[d] != 0.0f) {
                compute::atomic_add(output + 3u * v[k] + d, y[d]);
            }
        }
        const Vec3f size = friction_frame_abs(jacobian[k]).transpose() *
                           (2.0f * stiffness * (frame_size[0] + frame_size[1]));
        for (unsigned d = 0; d < 3u; ++d) {
            magnitude += fmath::abs(input[3u * v[k] + d]) * size[d];
        }
    }
    for (unsigned s = 0; s < 2u; ++s) {
        if (slot[s] == FRICTION_MEMORY_UNUSED) {
            continue;
        }
        const Vec3f rest = vertex_slip + frames - frame_slip[s];
        const Vec3f y = frame_jacobian[s].transpose() *
                        (stiffness * (rest - normal * normal.dot(rest)));
        const unsigned t = theta_base + 3u * slot[s];
        for (unsigned d = 0; d < 3u; ++d) {
            compute::atomic_add(output + t + d, y[d]);
        }
        const Vec3f size = friction_frame_abs(frame_jacobian[s]).transpose() *
                           (2.0f * stiffness * (vertex_size + frame_size[1u - s]));
        for (unsigned d = 0; d < 3u; ++d) {
            magnitude += fmath::abs(input[t + d]) * size[d];
        }
    }
    // Only the scale's total is read, so the entry's sum lands on one row.
    unsigned row = FRICTION_MEMORY_UNUSED;
    for (unsigned k = 0; k < FRICTION_MEMORY_VERTICES; ++k) {
        if (row == FRICTION_MEMORY_UNUSED && v[k] != FRICTION_MEMORY_UNUSED) {
            row = v[k];
        }
    }
    if (row != FRICTION_MEMORY_UNUSED && magnitude > 0.0f) {
        compute::atomic_add(absolute + row, magnitude);
    }
}
