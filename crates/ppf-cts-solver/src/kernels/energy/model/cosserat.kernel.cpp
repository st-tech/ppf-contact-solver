// File: cosserat.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. Plain C++ belonging to no backend: no preprocessor
// conditional, no macro of its own, and no spelling that only nvcc or only the
// Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the three forms the three compilers read.
//
// COSSERAT RODS. A Cosserat rod's edge carries a material frame R in SO(3)
// beside its two node positions ("Painless Differentiable Rotation Dynamics",
// Romanya-Serrasolsas, Casafranca and Otaduy, ACM TOG 44(4) 2025, section 6.2,
// on the CoRdE discretization of Spillmann and Teschner). The frames are built
// on the host (`cosserat.rs`); this file holds what the Newton step does with
// them: three energies (shear, bend and twist, rotational inertia), their
// Dirichlet reduction, the augmented operator and its preconditioner the reduced
// solve applies (`driver/pcg.rs`, `solve_cosserat`), and the updates of the
// frame state. The rod's stretch is the plain rod stretch (`rod_force.kernel.cpp`).
//
// A frame is perturbed on the LEFT, R -> exp(skew(theta)) R, and every gradient
// and Hessian is with respect to that tangent vector at theta = 0. A slot is one
// Cosserat edge in the frame builder's order, its three theta entries following
// the 3 * vertex_count position entries of the reduced vector (`theta_base`).
// A frame is a unit quaternion (x, y, z, w), four floats per slot.
//
// THE PRODUCTS ARE WRITTEN AS SCALAR LOOPS. A device matrix product or outer
// product through the matrix type's operators is silently wrong on CUDA
// (`solver/block_jacobi.kernel.cpp`), so every product below spells its sum.
//
// EVERY HESSIAN BLOCK IS PSD-PROJECTED before it lands anywhere, unconditionally
// (SPD-by-assembly): the shear, bend and twist, and inertia Hessians are all
// indefinite in general. The gradients are exact.

#include "../../csrmat/fixed_csr.kernel.cpp"
#include "../../linalg/eigsolve.hpp"
#include "../../main/dx_seed.kernel.cpp"
#include "../../utility/rod_scatter.kernel.cpp"
#include "pdrd_rigid.kernel.cpp"
#include "rod_bend_stiffness.kernel.cpp"

// ------------------------------------------------------------- 3x3 arithmetic

// a * b.
[[seam::host_device_fn]] inline Mat3x3f cosserat_mul(const Mat3x3f &a,
                                                     const Mat3x3f &b) {
    Mat3x3f c;
    for (unsigned i = 0; i < 3u; ++i) {
        for (unsigned j = 0; j < 3u; ++j) {
            float sum = 0.0f;
            for (unsigned k = 0; k < 3u; ++k) {
                sum += a(i, k) * b(k, j);
            }
            c(i, j) = sum;
        }
    }
    return c;
}

// a^T * b.
[[seam::host_device_fn]] inline Mat3x3f cosserat_mul_tn(const Mat3x3f &a,
                                                        const Mat3x3f &b) {
    Mat3x3f c;
    for (unsigned i = 0; i < 3u; ++i) {
        for (unsigned j = 0; j < 3u; ++j) {
            float sum = 0.0f;
            for (unsigned k = 0; k < 3u; ++k) {
                sum += a(k, i) * b(k, j);
            }
            c(i, j) = sum;
        }
    }
    return c;
}

// a * b^T.
[[seam::host_device_fn]] inline Mat3x3f cosserat_mul_nt(const Mat3x3f &a,
                                                        const Mat3x3f &b) {
    Mat3x3f c;
    for (unsigned i = 0; i < 3u; ++i) {
        for (unsigned j = 0; j < 3u; ++j) {
            float sum = 0.0f;
            for (unsigned k = 0; k < 3u; ++k) {
                sum += a(i, k) * b(j, k);
            }
            c(i, j) = sum;
        }
    }
    return c;
}

// m * v.
[[seam::host_device_fn]] inline Vec3f cosserat_apply(const Mat3x3f &m,
                                                     const Vec3f &v) {
    Vec3f out;
    for (unsigned i = 0; i < 3u; ++i) {
        out[i] = m(i, 0) * v[0] + m(i, 1) * v[1] + m(i, 2) * v[2];
    }
    return out;
}

// m^T * v.
[[seam::host_device_fn]] inline Vec3f cosserat_apply_t(const Mat3x3f &m,
                                                       const Vec3f &v) {
    Vec3f out;
    for (unsigned i = 0; i < 3u; ++i) {
        out[i] = m(0, i) * v[0] + m(1, i) * v[1] + m(2, i) * v[2];
    }
    return out;
}

// The axial vector of m - m^T: (m21 - m12, m02 - m20, m10 - m01).
[[seam::host_device_fn]] inline Vec3f cosserat_axial_diff(const Mat3x3f &m) {
    return Vec3f(m(2, 1) - m(1, 2), m(0, 2) - m(2, 0), m(1, 0) - m(0, 1));
}

// The frame of a stored quaternion, column major as Mat3x3f stores it.
[[seam::host_device_fn]] inline Mat3x3f
cosserat_frame(const float *quat, unsigned slot) {
    float q[4];
    for (unsigned k = 0; k < 4u; ++k) {
        q[k] = quat[4u * slot + k];
    }
    float r[9];
    pdrd_quat_to_mat(q, r);
    Mat3x3f frame;
    for (unsigned e = 0; e < 9u; ++e) {
        frame.data()[e] = r[e];
    }
    return frame;
}

// Store `frame` at `slot` as a unit quaternion.
[[seam::host_device_fn]] inline void cosserat_store_frame(float *quat,
                                                          unsigned slot,
                                                          const Mat3x3f &frame) {
    float r[9];
    for (unsigned e = 0; e < 9u; ++e) {
        r[e] = frame.data()[e];
    }
    float q[4];
    pdrd_mat_to_quat(r, q);
    const float norm =
        fmath::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    for (unsigned k = 0; k < 4u; ++k) {
        quat[4u * slot + k] = norm > 0.0f ? fmath::div(q[k], norm) : q[k];
    }
}

// The rotation carrying the unit direction `from` onto the unit direction `to`
// with no turn about either, I + [v]x + [v]x^2 / (1 + c), v = from x to,
// c = from . to: exact, with no trigonometry. `1 + c` is the caller's to check,
// since the two directions opposing each other leave the turn's axis undefined.
[[seam::host_device_fn]] inline Mat3x3f cosserat_transport(const Vec3f &from,
                                                           const Vec3f &to,
                                                           float one_plus_c) {
    const Mat3x3f k = pdrd_skew(from.cross(to));
    const Mat3x3f k2 = cosserat_mul(k, k);
    const float inverse = fmath::div(1.0f, one_plus_c);
    Mat3x3f r;
    for (unsigned i = 0; i < 3u; ++i) {
        for (unsigned j = 0; j < 3u; ++j) {
            r(i, j) = (i == j ? 1.0f : 0.0f) + k(i, j) + inverse * k2(i, j);
        }
    }
    return r;
}

// THE SMALLEST 1 + cos(turn) A PRESCRIBED EDGE MAY HAVE OVER ONE STEP: a turn
// within about 179 degrees. An edge whose two vertices are both prescribed and
// that reverses its direction in one step has no defined turn axis, and the
// frame it carries cannot follow it; that is a scene whose pins swap ends
// within a substep, and it is refused through the diagnostic lane rather than
// given an arbitrary frame.
[[seam::host_device_fn]] inline float cosserat_reversal_floor() {
    return 1.0e-4f;
}

// --------------------------------------------------------------- the energies

// SHEAR: E = c (1 - d . p), c = 1/2 k_X l, d the edge's unit director, p = R e_z
// its frame's third column. DOF order [x0, x1, theta], L the current length.
// Writes the exact gradient and Hessian.
[[seam::host_device_fn]] inline void
cosserat_shear_grad_hess(const Vec3f &d, const Vec3f &p, float c, float length,
                         Vec3f &grad_x0, Vec3f &grad_x1, Vec3f &grad_theta,
                         Mat9x9f &hessian) {
    const float s = d.dot(p);
    const float cl = fmath::div(c, length);
    const float cl2 = fmath::div(c, length * length);
    for (unsigned i = 0; i < 3u; ++i) {
        const float pp = p[i] - s * d[i];
        grad_x0[i] = cl * pp;
        grad_x1[i] = -cl * pp;
    }
    grad_theta = c * d.cross(p);
    for (unsigned i = 0; i < 9u; ++i) {
        for (unsigned j = 0; j < 9u; ++j) {
            hessian(i, j) = 0.0f;
        }
    }
    // The position block cl2 (p d^T + d p^T + s I - 3 s d d^T), with the signs
    // of x1 - x0.
    for (unsigned i = 0; i < 3u; ++i) {
        for (unsigned j = 0; j < 3u; ++j) {
            float v = p[i] * d[j] + d[i] * p[j] - 3.0f * s * d[i] * d[j];
            if (i == j) {
                v += s;
            }
            v *= cl2;
            hessian(i, j) = v;
            hessian(3u + i, 3u + j) = v;
            hessian(i, 3u + j) = -v;
            hessian(3u + i, j) = -v;
        }
    }
    // The frame's own block c s I - c/2 (d p^T + p d^T).
    for (unsigned i = 0; i < 3u; ++i) {
        for (unsigned j = 0; j < 3u; ++j) {
            float v = -0.5f * c * (d[i] * p[j] + p[i] * d[j]);
            if (i == j) {
                v += c * s;
            }
            hessian(6u + i, 6u + j) = v;
        }
    }
    // The coupling: theta against x0 is cl (skew(p) - (p x d) d^T), against x1
    // its negative, and the position rows carry the transposes.
    const Vec3f pxd = p.cross(d);
    const Mat3x3f sp = pdrd_skew(p);
    for (unsigned i = 0; i < 3u; ++i) {
        for (unsigned j = 0; j < 3u; ++j) {
            const float a = cl * (sp(i, j) - pxd[i] * d[j]);
            hessian(6u + i, j) = a;
            hessian(6u + i, 3u + j) = -a;
            hessian(j, 6u + i) = a;
            hessian(3u + j, 6u + i) = -a;
        }
    }
}

// THE SMALLEST 1 + tr(G0) A BEND SITE MAY HAVE: a relative turn within about
// 179 degrees between two consecutive frames. Past it the Darboux vector below
// has no defined direction; a site there is refused through the diagnostic lane.
[[seam::host_device_fn]] inline float cosserat_half_turn_floor() {
    return 1.0e-4f;
}

// BEND AND TWIST: E = 1/2 l (w - w_rest)^T M K M (w - w_rest), K = diag(kb, kb, kt)
// and M the site's REST METRIC (`along` on the rest Darboux vector's axis,
// `across` off it; `crate::cosserat::rest_metric` says why it makes the second
// variation at rest exact), at the node between two consecutive frames, with w
// the DISCRETE DARBOUX VECTOR
// of their relative rotation G0 = R_prev^T R_next:
//
//   w = (2 / l) vec(q(G0)) = u / (l s),  u = axial_diff(G0),  s = sqrt(1 + tr G0),
//
// q(G0) the relative quaternion on the hemisphere of positive scalar part. A
// relative turn by phi about the unit axis a gives w = (2 / l) sin(phi / 2) a,
// which is ONE-TO-ONE for every turn below a half turn. That is the property a
// signed rest curvature needs: the axial vector of G0's antisymmetric part alone,
// sin(phi) a / l, peaks at a quarter turn, so a site resting at phi0 has a second
// state of zero energy at pi - phi0, which a crimp of a radian reaches under load.
//
// DOF order [theta_next, theta_prev]. Writes the exact gradient of the next
// frame (the previous one's is its negative) and the exact Hessian, and returns
// false when the relative turn is too close to a half turn to define w.
//
// THE DERIVATIVES. In v = R_prev^T theta_next, G = exp(skew(v)) G0, and to
// second order
//   u(v) = u0 + J v + 1/2 [ (G0^T v) x v - |v|^2 u0 ],  J = tr(G0) I - G0,
//   t(v) = t0 - u0 . v + 1/2 v^T (sym(G0) - t0 I) v,     s = sqrt(1 + t),
// so w's Jacobian and Hessians follow from the quotient rule; the curvature part
// sums them against y = K (w - w_rest) in closed form, sum_i y_i d2u_i =
// sym(-G0 skew(y)) - (u0 . y) I. The previous frame enters through
// exp(-theta_prev) exp(theta_next), whose log is
// theta_next - theta_prev - 1/2 theta_prev x theta_next, so both diagonal blocks
// are the same H and the coupling is -H + 1/2 skew(g) with the previous frame's
// rows, g the next frame's gradient.
//
// THE CURVATURE IS FORMED FROM THE RELATIVE ROTATION G0, never as a difference
// of the two absolute frames, which would cancel in fp32.
[[seam::host_device_fn]] inline bool
cosserat_bend_grad_hess(const Mat3x3f &frame_prev, const Mat3x3f &frame_next,
                        const Vec3f &rest_darboux, float kb, float kt,
                        float inv_len, float along, float across,
                        Vec3f &grad_next, Mat6x6f &hessian) {
    const Mat3x3f g0 = cosserat_mul_tn(frame_prev, frame_next);
    const Vec3f u0 = cosserat_axial_diff(g0);
    const float t0 = g0(0, 0) + g0(1, 1) + g0(2, 2);
    const float one_plus = 1.0f + t0;
    if (!(one_plus > cosserat_half_turn_floor())) {
        return false;
    }
    const float s0 = fmath::sqrt(one_plus);
    const float inv_s = fmath::div(1.0f, s0);
    const float length = fmath::div(1.0f, inv_len);
    // The stiffness M K M, M = across I + (along - across) a a^T, a the rest
    // Darboux vector's unit axis (M = I at a straight rest, where both are 1).
    const float rest_norm = rest_darboux.norm();
    const Vec3f axis = rest_norm > 0.0f ? rest_darboux / rest_norm
                                        : Vec3f(0.0f, 0.0f, 0.0f);
    Mat3x3f metric;
    for (unsigned i = 0; i < 3u; ++i) {
        for (unsigned j = 0; j < 3u; ++j) {
            metric(i, j) = (i == j ? across : 0.0f) + (along - across) * axis[i] * axis[j];
        }
    }
    const float diagonal[3] = {kb, kb, kt};
    Mat3x3f stiffness;
    for (unsigned i = 0; i < 3u; ++i) {
        for (unsigned j = 0; j < 3u; ++j) {
            float sum = 0.0f;
            for (unsigned k = 0; k < 3u; ++k) {
                sum += metric(i, k) * diagonal[k] * metric(k, j);
            }
            stiffness(i, j) = sum;
        }
    }
    // w and the stiffness-weighted misfit y = M K M (w - w_rest).
    Vec3f misfit;
    for (unsigned i = 0; i < 3u; ++i) {
        misfit[i] = inv_len * inv_s * u0[i] - rest_darboux[i];
    }
    const Vec3f y = cosserat_apply(stiffness, misfit);
    // s's derivatives in v: D_a = -u0_a / (2 s), D_ab = t_ab / (2 s) - u0_a u0_b / (4 s^3).
    Vec3f d1;
    for (unsigned a = 0; a < 3u; ++a) {
        d1[a] = -0.5f * inv_s * u0[a];
    }
    Mat3x3f d2;
    for (unsigned a = 0; a < 3u; ++a) {
        for (unsigned b = 0; b < 3u; ++b) {
            const float t_ab = 0.5f * (g0(a, b) + g0(b, a)) - (a == b ? t0 : 0.0f);
            d2(a, b) = 0.5f * inv_s * t_ab -
                       0.25f * inv_s * inv_s * inv_s * u0[a] * u0[b];
        }
    }
    // dw_i / dv_a = inv_len (J_ia / s - u0_i D_a / s^2).
    Mat3x3f dw;
    for (unsigned i = 0; i < 3u; ++i) {
        for (unsigned a = 0; a < 3u; ++a) {
            const float j_ia = (i == a ? t0 : 0.0f) - g0(i, a);
            dw(i, a) = inv_len * (j_ia * inv_s - u0[i] * d1[a] * inv_s * inv_s);
        }
    }
    // The gradient in v, then in theta_next = R_prev v.
    Vec3f grad_v;
    for (unsigned a = 0; a < 3u; ++a) {
        float sum = 0.0f;
        for (unsigned i = 0; i < 3u; ++i) {
            sum += dw(i, a) * y[i];
        }
        grad_v[a] = length * sum;
    }
    grad_next = cosserat_apply(frame_prev, grad_v);
    // The Hessian in v: l [ dw^T K dw + sum_i y_i d2w_i ], with
    // sum_i y_i d2w_i = inv_len [ (1/s) sum_i y_i d2u_i
    //   - (1/s^2) ((J^T y)_a D_b + (J^T y)_b D_a) - (u0 . y / s^2) D_ab
    //   + 2 (u0 . y) D_a D_b / s^3 ].
    float u_dot_y = 0.0f;
    for (unsigned i = 0; i < 3u; ++i) {
        u_dot_y += u0[i] * y[i];
    }
    Vec3f jt_y;
    for (unsigned a = 0; a < 3u; ++a) {
        float sum = 0.0f;
        for (unsigned i = 0; i < 3u; ++i) {
            sum += ((i == a ? t0 : 0.0f) - g0(i, a)) * y[i];
        }
        jt_y[a] = sum;
    }
    const Mat3x3f g0_skew_y = cosserat_mul(g0, pdrd_skew(y));
    Mat3x3f h_v;
    for (unsigned a = 0; a < 3u; ++a) {
        for (unsigned b = 0; b < 3u; ++b) {
            float gauss_newton = 0.0f;
            for (unsigned i = 0; i < 3u; ++i) {
                for (unsigned j = 0; j < 3u; ++j) {
                    gauss_newton += dw(i, a) * stiffness(i, j) * dw(j, b);
                }
            }
            const float y_d2u = -0.5f * (g0_skew_y(a, b) + g0_skew_y(b, a)) -
                                (a == b ? u_dot_y : 0.0f);
            const float curvature =
                inv_len * (inv_s * y_d2u -
                           inv_s * inv_s * (jt_y[a] * d1[b] + jt_y[b] * d1[a]) -
                           inv_s * inv_s * u_dot_y * d2(a, b) +
                           2.0f * inv_s * inv_s * inv_s * u_dot_y * d1[a] * d1[b]);
            h_v(a, b) = length * (gauss_newton + curvature);
        }
    }
    // H in theta: R_prev H_v R_prev^T.
    const Mat3x3f h = cosserat_mul_nt(cosserat_mul(frame_prev, h_v), frame_prev);
    const Mat3x3f skew_g = pdrd_skew(grad_next);
    for (unsigned i = 0; i < 3u; ++i) {
        for (unsigned j = 0; j < 3u; ++j) {
            const float sym = 0.5f * (h(i, j) + h(j, i));
            hessian(i, j) = sym;
            hessian(3u + i, 3u + j) = sym;
            // (next, prev) = -H - 1/2 skew(g); (prev, next) is its transpose.
            hessian(i, 3u + j) = -0.5f * (h(i, j) + h(j, i)) - 0.5f * skew_g(i, j);
        }
    }
    for (unsigned i = 0; i < 3u; ++i) {
        for (unsigned j = 0; j < 3u; ++j) {
            hessian(3u + i, j) = hessian(j, 3u + i);
        }
    }
    return true;
}

// ROTATIONAL INERTIA: E = -tr(M0), M0 = (j / dt^2) R R_pred^T, the frame against
// its inertial prediction. Writes the exact gradient axial(M0 - M0^T) and Hessian
// tr(M0) I - sym(M0).
[[seam::host_device_fn]] inline void
cosserat_inertia_grad_hess(const Mat3x3f &m0, Vec3f &gradient, Mat3x3f &hessian) {
    gradient = cosserat_axial_diff(m0);
    const float trace = m0(0, 0) + m0(1, 1) + m0(2, 2);
    for (unsigned i = 0; i < 3u; ++i) {
        for (unsigned j = 0; j < 3u; ++j) {
            hessian(i, j) = (i == j ? trace : 0.0f) - 0.5f * (m0(i, j) + m0(j, i));
        }
    }
}

// The nearest PSD matrix to the symmetric part of `m`, in place.
template <int N>
[[seam::host_device_fn]] inline void
cosserat_project(linalg::SMat<float, N, N> &m) {
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            const float s = 0.5f * (m(i, j) + m(j, i));
            m(i, j) = s;
            m(j, i) = s;
        }
    }
    linalg::psd_project_symmetric<N>(m, 0.0f);
}

// ----------------------------------------------------------------- assembly

// The orientation buffers a Newton iteration assembles, per slot and per site,
// as flat floats: `gradient` 3 per slot, `self_block` 9 per slot (the frame's
// own 3x3), `coupling_from` and `coupling_to` 9 per slot (position rows against
// the frame's columns, for the slot's two endpoints), and `site_block` 9 per
// site (the next frame's rows against the previous frame's columns). Every 3x3
// is column major, as Mat3x3f stores it. The driver clears them before the
// shear pass.

// Add a 3x3 into a slot's block with atomic adds.
[[seam::device_fn]] inline void cosserat_atomic_add_block(
    compute::atomic_float_t *blocks, unsigned slot, const Mat3x3f &m) {
    for (unsigned e = 0; e < 9u; ++e) {
        compute::atomic_add(blocks + 9u * slot + e, m.data()[e]);
    }
}

// THE SHEAR PASS, one thread per slot. The position half goes where the rod
// stretch's goes: the force into `force` and the 6x6 into the fixed matrix
// through the rod edge's own slots, in the MESH order of the edge's two
// vertices, which is the order those slots were registered in. The frame half
// goes into the orientation buffers.
[[seam::entry(slot)]]
[[seam::device_fn]] inline void cosserat_shear_row(
    const Vec3f *x,
    const Vec2u *edge_nodes,
    const unsigned *slot_edge,
    const Vec2u *rod_edge,
    const EdgeProp *edge_prop,
    const EdgeParam *edge_param,
    const float *quat,
    const unsigned *hess_slots,
    unsigned has_hess_slots,
    compute::atomic_float_t *force,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    compute::atomic_float_t *fixed_value,
    unsigned row_count,
    compute::atomic_uint_t *refused,
    unsigned *witness,
    compute::atomic_float_t *gradient,
    compute::atomic_float_t *self_block,
    compute::atomic_float_t *coupling_from,
    compute::atomic_float_t *coupling_to,
    unsigned slot) {
    const unsigned edge = slot_edge[slot];
    const EdgeProp prop = edge_prop[edge];
    const EdgeParam material = edge_param[prop.param_index];
    if (!(material.cosserat_shear > 0.0f)) {
        return;
    }
    const Vec2u nodes = edge_nodes[slot];
    const Vec3f x_from = x[nodes[0]];
    const Vec3f x_to = x[nodes[1]];
    const Vec3f e = (x_to - x_from).cast<float>();
    const float length = e.norm();
    if (!(length > 0.0f)) {
        return;
    }
    const Vec3f d = e / length;
    const Mat3x3f frame = cosserat_frame(quat, slot);
    const Vec3f p(frame(0, 2), frame(1, 2), frame(2, 2));
    // Density-normalized, as the rod stretch's stiffness times mass: with mass
    // = density * l this is the paper's 1/2 k_X l, and the balance of the
    // elastic terms against inertia does not depend on the density.
    const float c = 0.5f * material.cosserat_shear * prop.mass;
    Vec3f grad_from, grad_to, grad_theta;
    Mat9x9f hessian;
    cosserat_shear_grad_hess(d, p, c, length, grad_from, grad_to, grad_theta,
                             hessian);
    cosserat_project<9>(hessian);

    // The position force and 6x6, in the mesh edge's vertex order.
    const Vec2u mesh = rod_edge[edge];
    const bool reversed = mesh[0] != nodes[0];
    Mat3x2f position_gradient;
    Mat6x6f position_hessian;
    for (unsigned i = 0; i < 3u; ++i) {
        position_gradient(i, 0) = reversed ? grad_to[i] : grad_from[i];
        position_gradient(i, 1) = reversed ? grad_from[i] : grad_to[i];
    }
    for (unsigned a = 0; a < 2u; ++a) {
        const unsigned source_a = reversed ? 1u - a : a;
        for (unsigned b = 0; b < 2u; ++b) {
            const unsigned source_b = reversed ? 1u - b : b;
            for (unsigned i = 0; i < 3u; ++i) {
                for (unsigned j = 0; j < 3u; ++j) {
                    position_hessian(3u * a + i, 3u * b + j) =
                        hessian(3u * source_a + i, 3u * source_b + j);
                }
            }
        }
    }
    rod_atomic_embed_force(mesh, position_gradient, force);
    unsigned vertex_slots[2];
    vertex_slots[0] = mesh[0];
    vertex_slots[1] = mesh[1];
    if (has_hess_slots != 0u) {
        fixed_push_blocks_thread_at(hess_slots, position_hessian.m, 2u,
                                    fixed_value, edge);
    } else {
        fixed_push_blocks_thread(vertex_slots, position_hessian.m, 2u,
                                 fixed_index, fixed_offset, fixed_value,
                                 row_count, refused, witness);
    }

    // The frame half: its gradient, its own block, and the coupling of each
    // endpoint's position rows to its columns.
    Mat3x3f own, from, to;
    for (unsigned i = 0; i < 3u; ++i) {
        for (unsigned j = 0; j < 3u; ++j) {
            own(i, j) = hessian(6u + i, 6u + j);
            from(i, j) = hessian(i, 6u + j);
            to(i, j) = hessian(3u + i, 6u + j);
        }
    }
    for (unsigned i = 0; i < 3u; ++i) {
        compute::atomic_add(gradient + 3u * slot + i, grad_theta[i]);
    }
    cosserat_atomic_add_block(self_block, slot, own);
    cosserat_atomic_add_block(coupling_from, slot, from);
    cosserat_atomic_add_block(coupling_to, slot, to);
}

// THE ROTATIONAL INERTIA PASS, one thread per slot, after the shear pass and
// before the bend pass.
//
// THE PREDICTION CARRIES ANGULAR MOMENTUM. With dR = R_prev R_prev2^T the last
// step's incremental rotation, the frame is predicted at
// (I + tr skew(w)) R_prev, w = axial(dR - dR^T) / 2, tr = dt / prev_dt: the
// rotational twin of the position predictor x + (x - x_prev) dt / prev_dt, and
// to first order the paper's (2 I - dR^T) R_prev at tr = 1. dR is a RELATIVE
// rotation, never a difference of two absolute frames. At rest dR = I and the
// prediction is R_prev.
//
// ISOTROPIC INERTIA j = m l^2 / 12 per edge. It is what keeps the twist about
// the tangent, an exact null mode of shear plus bend and twist, from leaving the
// system singular.
//
// Each slot is written by this thread alone, so the buffers take plain adds.
[[seam::entry(slot)]]
[[seam::device_fn]] inline void cosserat_inertia_row(
    const unsigned *slot_edge,
    const EdgeProp *edge_prop,
    const float *quat,
    const float *quat_prev,
    const float *quat_prev2,
    float dt,
    float tr,
    float *gradient,
    float *self_block,
    unsigned slot) {
    const EdgeProp prop = edge_prop[slot_edge[slot]];
    const float inertia = prop.mass * prop.length * prop.length / 12.0f;
    if (!(inertia > 0.0f) || !(dt > 0.0f)) {
        return;
    }
    const Mat3x3f current = cosserat_frame(quat, slot);
    const Mat3x3f previous = cosserat_frame(quat_prev, slot);
    const Mat3x3f before = cosserat_frame(quat_prev2, slot);
    const Vec3f axial = cosserat_axial_diff(cosserat_mul_nt(previous, before));
    const Vec3f w = (0.5f * tr) * axial;
    Mat3x3f step = pdrd_skew(w);
    for (unsigned i = 0; i < 3u; ++i) {
        step(i, i) += 1.0f;
    }
    const Mat3x3f predicted = cosserat_mul(step, previous);
    Mat3x3f m0 = cosserat_mul_nt(current, predicted);
    const float scale = fmath::div(inertia, dt * dt);
    for (unsigned e = 0; e < 9u; ++e) {
        m0.data()[e] *= scale;
    }
    Vec3f g;
    Mat3x3f h;
    cosserat_inertia_grad_hess(m0, g, h);
    // Projected like every other term: the exact inertia Hessian is PSD only
    // while the step's rotation stays below about pi / 2.
    cosserat_project<3>(h);
    for (unsigned i = 0; i < 3u; ++i) {
        gradient[3u * slot + i] += g[i];
    }
    for (unsigned e = 0; e < 9u; ++e) {
        self_block[9u * slot + e] += h.data()[e];
    }
}

// THE BEND AND TWIST PASS, one thread per site. A site owns its `site_block`
// entry, so that one is stored.
//
// THE RIGIDITIES FOLLOW THE PLAIN ROD'S CONVENTION (`rod_bend_stiffness.kernel.cpp`):
// B = bend * linear density * REF_LENGTH^2, the two edges' mean `bend`, so one
// `cosserat-bend` value describes one rod at any segment length. That helper
// returns B / l, the coefficient of a squared turning angle; this energy
// integrates 1/2 B |w - w_rest|^2 over the site's own length, so its coefficient
// is that stiffness times l. Scaling B by the node's mass alone would make it
// grow with l, so halving every segment would halve the rod's rigidity.
[[seam::entry(site)]]
[[seam::device_fn]] inline void cosserat_bend_row(
    const Vec2u *bend_slots,
    const unsigned *slot_edge,
    const EdgeProp *edge_prop,
    const EdgeParam *edge_param,
    const float *inv_len,
    const float *rest_metric,
    const Vec3f *rest_darboux,
    const float *quat,
    compute::atomic_float_t *gradient,
    compute::atomic_float_t *self_block,
    float *site_block,
    DiagHandle diag,
    unsigned site) {
    const Vec2u slots = bend_slots[site];
    const unsigned slot_prev = slots[0];
    const unsigned slot_next = slots[1];
    const EdgeProp prop_prev = edge_prop[slot_edge[slot_prev]];
    const EdgeProp prop_next = edge_prop[slot_edge[slot_next]];
    const EdgeParam material_prev = edge_param[prop_prev.param_index];
    const EdgeParam material_next = edge_param[prop_next.param_index];
    const float node_mass = 0.5f * (prop_prev.mass + prop_next.mass);
    const float voronoi = 0.5f * (prop_prev.length + prop_next.length);
    const float kb =
        rod_bend_stiffness(rod_bend_segment_average(material_prev.cosserat_bend,
                                                    material_next.cosserat_bend),
                           node_mass, prop_prev.length, prop_next.length) *
        voronoi;
    const float kt =
        rod_bend_stiffness(rod_bend_segment_average(material_prev.cosserat_twist,
                                                    material_next.cosserat_twist),
                           node_mass, prop_prev.length, prop_next.length) *
        voronoi;
    if (!(kb > 0.0f) && !(kt > 0.0f)) {
        return;
    }
    const Vec3f rest = rest_darboux[site];
    Vec3f grad_next;
    Mat6x6f hessian;
    const bool defined = cosserat_bend_grad_hess(
        cosserat_frame(quat, slot_prev), cosserat_frame(quat, slot_next), rest, kb,
        kt, inv_len[site], rest_metric[2u * site], rest_metric[2u * site + 1u],
        grad_next, hessian);
    DIAG_ASSERT4(diag, defined, static_cast<float>(site),
                 static_cast<float>(slot_prev), static_cast<float>(slot_next), kb);
    if (!defined) {
        return;
    }
    cosserat_project<6>(hessian);
    Mat3x3f next, prev, coupling;
    for (unsigned i = 0; i < 3u; ++i) {
        for (unsigned j = 0; j < 3u; ++j) {
            next(i, j) = hessian(i, j);
            prev(i, j) = hessian(3u + i, 3u + j);
            coupling(i, j) = hessian(i, 3u + j);
        }
    }
    for (unsigned i = 0; i < 3u; ++i) {
        compute::atomic_add(gradient + 3u * slot_next + i, grad_next[i]);
        compute::atomic_add(gradient + 3u * slot_prev + i, -grad_next[i]);
    }
    cosserat_atomic_add_block(self_block, slot_next, next);
    cosserat_atomic_add_block(self_block, slot_prev, prev);
    for (unsigned e = 0; e < 9u; ++e) {
        site_block[9u * site + e] = coupling.data()[e];
    }
}

// THE DIRICHLET REDUCTION OF THE COUPLING, one thread per slot, run whenever any
// vertex's degree of freedom is removed. The fixed matrix's own pass
// (`main/dirichlet.kernel.cpp`) cannot see the coupling blocks, which live
// here, so this mirrors it: for each endpoint whose row is removed, LIFT the
// known increment p onto the frame's right-hand side, g_theta -= H^T p, then
// ZERO the block. Without it a pinned vertex beside a free frame leaves a live
// coupling against an identity row, and the augmented matrix is indefinite as
// soon as the coupling exceeds the frame's own block, which a stiff shear always
// does.
[[seam::entry(slot)]]
[[seam::device_fn]] inline void cosserat_dirichlet_row(
    const Vec2u *edge_nodes,
    const unsigned *dof_mask,
    const Vec3f *eval_x,
    const Vec3f *target,
    float *gradient,
    float *coupling_from,
    float *coupling_to,
    unsigned slot) {
    const Vec2u nodes = edge_nodes[slot];
    for (unsigned end = 0; end < 2u; ++end) {
        const unsigned vert = nodes[end];
        if (dof_mask[vert] == 0u) {
            continue;
        }
        // The two arrays are indexed where they are read and written, and not
        // through a local pointer to the one this end uses: a pointer into a
        // device array has no address space of its own in a shader.
        Mat3x3f coupling;
        for (unsigned e = 0; e < 9u; ++e) {
            coupling.data()[e] = end == 0u ? coupling_from[9u * slot + e]
                                           : coupling_to[9u * slot + e];
        }
        const Vec3f current = eval_x[vert];
        const Vec3f goal = target[vert];
        const Vec3f lift =
            cosserat_apply_t(coupling, prescribed_increment(current, goal));
        for (unsigned i = 0; i < 3u; ++i) {
            gradient[3u * slot + i] -= lift[i];
        }
        for (unsigned e = 0; e < 9u; ++e) {
            if (end == 0u) {
                coupling_from[9u * slot + e] = 0.0f;
            } else {
                coupling_to[9u * slot + e] = 0.0f;
            }
        }
    }
}

// THE FRAMES A STEP PRESCRIBES, one thread per slot, before the augmented
// solve. A frame is HELD when `cosserat-fix-ends` holds it, or when both of its
// edge's vertices are prescribed this step: an edge fully held by fixed pins is
// a clamp, and a clamp holds the cross-section too, so the frame moves with its
// edge. A single held vertex is a hinge and leaves the frame free.
//
// `held` is the per-step mask the solve projects with, and `prescribed` the
// frame rows of its seed: zero for a frame `cosserat-fix-ends` holds, and for a
// clamped edge the turn carrying its direction now onto its prescribed one, in
// the step's sign convention (`R <- exp(-dtheta) R`, as `x <- x - dx`). The
// solve uses it only to lift the clamp's motion into the free rows it couples
// to, so its first-order form, 2 (a x b) / (1 + a . b), exact to the cube of
// the turn, is enough; the frame itself is moved by the exact transport in
// `cosserat_step_row`.
[[seam::entry(slot)]]
[[seam::device_fn]] inline void cosserat_prescribe_row(
    const Vec2u *edge_nodes,
    const unsigned *fixed,
    const unsigned *dof_mask,
    const Vec3f *eval_x,
    const Vec3f *target,
    unsigned *held,
    float *prescribed,
    DiagHandle diag,
    unsigned slot) {
    const Vec2u nodes = edge_nodes[slot];
    Vec3f turn(0.0f, 0.0f, 0.0f);
    unsigned hold = fixed[slot] != 0u ? 1u : 0u;
    if (dof_mask[nodes[0]] != 0u && dof_mask[nodes[1]] != 0u) {
        const Vec3f from0 = eval_x[nodes[0]];
        const Vec3f from1 = eval_x[nodes[1]];
        const Vec3f to0 = target[nodes[0]];
        const Vec3f to1 = target[nodes[1]];
        const Vec3f now = (from1 - from0).cast<float>();
        const Vec3f goal = (to1 - to0).cast<float>();
        const Vec3f a = now / now.norm();
        const Vec3f b = goal / goal.norm();
        const float one_plus_c = 1.0f + a.dot(b);
        DIAG_ASSERT4(diag, one_plus_c > cosserat_reversal_floor(), one_plus_c,
                     static_cast<float>(slot), static_cast<float>(nodes[0]),
                     static_cast<float>(nodes[1]));
        if (one_plus_c > cosserat_reversal_floor()) {
            turn = fmath::div(-2.0f, one_plus_c) * a.cross(b);
        }
        hold = 1u;
    }
    held[slot] = hold;
    for (unsigned i = 0; i < 3u; ++i) {
        prescribed[3u * slot + i] = turn[i];
    }
}

// ------------------------------------------------------ the reduced operator

// The frame rows of y = A u and the frames' share of its position rows, one
// thread per slot, after the per-vertex operator has written y's position rows:
// y_theta = H_self u_theta + H_from^T u_from + H_to^T u_to, stored, and
// y_from += H_from u_theta, y_to += H_to u_theta, added. `output` and
// `output_atomic` are the same allocation.
[[seam::entry(slot)]]
[[seam::device_fn]] inline void cosserat_apply_slot_row(
    const Vec2u *edge_nodes,
    const float *self_block,
    const float *coupling_from,
    const float *coupling_to,
    const float *input,
    float *output,
    compute::atomic_float_t *output_atomic,
    unsigned theta_base,
    unsigned slot) {
    const Vec2u nodes = edge_nodes[slot];
    Mat3x3f own, from, to;
    for (unsigned e = 0; e < 9u; ++e) {
        own.data()[e] = self_block[9u * slot + e];
        from.data()[e] = coupling_from[9u * slot + e];
        to.data()[e] = coupling_to[9u * slot + e];
    }
    const unsigned t = theta_base + 3u * slot;
    const Vec3f u_theta(input[t], input[t + 1u], input[t + 2u]);
    const Vec3f u_from(input[3u * nodes[0]], input[3u * nodes[0] + 1u],
                       input[3u * nodes[0] + 2u]);
    const Vec3f u_to(input[3u * nodes[1]], input[3u * nodes[1] + 1u],
                     input[3u * nodes[1] + 2u]);
    const Vec3f y_theta = cosserat_apply(own, u_theta) +
                          cosserat_apply_t(from, u_from) +
                          cosserat_apply_t(to, u_to);
    for (unsigned i = 0; i < 3u; ++i) {
        output[t + i] = y_theta[i];
    }
    const Vec3f y_from = cosserat_apply(from, u_theta);
    const Vec3f y_to = cosserat_apply(to, u_theta);
    for (unsigned i = 0; i < 3u; ++i) {
        compute::atomic_add(output_atomic + 3u * nodes[0] + i, y_from[i]);
        compute::atomic_add(output_atomic + 3u * nodes[1] + i, y_to[i]);
    }
}

// The bend coupling's share of y's frame rows, one thread per site, after the
// slot pass: y_next += H_site u_prev and y_prev += H_site^T u_next.
[[seam::entry(site)]]
[[seam::device_fn]] inline void cosserat_apply_site_row(
    const Vec2u *bend_slots,
    const float *site_block,
    const float *input,
    compute::atomic_float_t *output,
    unsigned theta_base,
    unsigned site) {
    const Vec2u slots = bend_slots[site];
    Mat3x3f coupling;
    for (unsigned e = 0; e < 9u; ++e) {
        coupling.data()[e] = site_block[9u * site + e];
    }
    const unsigned prev = theta_base + 3u * slots[0];
    const unsigned next = theta_base + 3u * slots[1];
    const Vec3f u_prev(input[prev], input[prev + 1u], input[prev + 2u]);
    const Vec3f u_next(input[next], input[next + 1u], input[next + 2u]);
    const Vec3f y_next = cosserat_apply(coupling, u_prev);
    const Vec3f y_prev = cosserat_apply_t(coupling, u_next);
    for (unsigned i = 0; i < 3u; ++i) {
        compute::atomic_add(output + next + i, y_next[i]);
        compute::atomic_add(output + prev + i, y_prev[i]);
    }
}

// THE PRECONDITIONER: OVERLAPPING BLOCKS, ONE PER COSSERAT EDGE.
//
// The shear ties each frame to its edge's direction, and its position half
// puts a stiffness of order GA / l on the diagonal of both vertices. A bending
// mode whose frames follow their edges costs only the bend energy, so a
// preconditioner that sees vertices and frames separately takes it for a stiff
// mode, and its iterations grow with GA over EI: measured on a dense woven strip,
// block-Jacobi with a per-frame Schur complement met PCG's 10,000-iteration cap
// at the first step, and a tenth of the shear did not. Solving each edge's two
// vertices and its frame TOGETHER sees that mode for what it is. The block is a
// principal submatrix of the augmented matrix (each vertex's full diagonal
// block, the fixed matrix's block between the two vertices, the couplings and
// the frame's own block), so it is SPD wherever the system is, and the sum over
// the overlapping blocks is an additive Schwarz preconditioner, SPD as well. A
// vertex on no Cosserat edge keeps its block-Jacobi inverse.

// The Cholesky factor of one edge's 9x9 block, lower triangle packed by rows,
// 45 floats per slot. A held frame's rows are decoupled and set to the
// identity: the solve keeps them at their seed.
//
// EVERY DIAGONAL ENTRY MUST BE POSITIVE AND FINITE, and one that is not is an
// assembly defect, reported through the diagnostic lane. A PIVOT IS FLOORED AT
// 1e-6 OF ITS OWN DIAGONAL ENTRY: a block holds a contact-stiffened vertex beside
// a frame whose bending pivot is four or five orders smaller, and the shear's
// coupling cancels most of that frame's diagonal on the way down, so in single
// precision a pivot of an SPD block can round to zero or below. The floor is
// the cancellation's own scale, raises an eigenvalue only, and keeps the factor
// SPD; it changes how fast PCG converges, never what it converges to.
[[seam::entry(slot)]]
[[seam::device_fn]] inline void cosserat_schwarz_factor_row(
    const Vec2u *edge_nodes,
    const unsigned *held,
    const float *vertex_diagonal,
    const unsigned *fixed_index,
    const unsigned *fixed_offset,
    const float *fixed_value,
    unsigned row_count,
    const float *self_block,
    const float *coupling_from,
    const float *coupling_to,
    float *factor,
    DiagHandle diag,
    unsigned slot) {
    const Vec2u nodes = edge_nodes[slot];
    float k[81];
    for (unsigned e = 0; e < 81u; ++e) {
        k[e] = 0.0f;
    }
    // Column-major 3x3 blocks into the row-major 9x9 `k[9 * row + column]`.
    for (unsigned i = 0; i < 3u; ++i) {
        for (unsigned j = 0; j < 3u; ++j) {
            k[9u * i + j] = vertex_diagonal[9u * nodes[0] + i + 3u * j];
            k[9u * (3u + i) + 3u + j] = vertex_diagonal[9u * nodes[1] + i + 3u * j];
        }
    }
    const Mat3x3f between = fixed_csr_read(fixed_index, fixed_offset, fixed_value,
                                           row_count, nodes[0], nodes[1]);
    for (unsigned i = 0; i < 3u; ++i) {
        for (unsigned j = 0; j < 3u; ++j) {
            k[9u * i + 3u + j] = between(i, j);
            k[9u * (3u + j) + i] = between(i, j);
        }
    }
    if (held[slot] != 0u) {
        for (unsigned i = 0; i < 3u; ++i) {
            k[9u * (6u + i) + 6u + i] = 1.0f;
        }
    } else {
        for (unsigned i = 0; i < 3u; ++i) {
            for (unsigned j = 0; j < 3u; ++j) {
                const float own = self_block[9u * slot + i + 3u * j];
                const float from = coupling_from[9u * slot + i + 3u * j];
                const float to = coupling_to[9u * slot + i + 3u * j];
                k[9u * (6u + i) + 6u + j] = own;
                // `from` and `to` are position rows against frame columns.
                k[9u * i + 6u + j] = from;
                k[9u * (6u + j) + i] = from;
                k[9u * (3u + i) + 6u + j] = to;
                k[9u * (6u + j) + 3u + i] = to;
            }
        }
    }
    unsigned bad = 9u;
    for (unsigned i = 0; i < 9u; ++i) {
        const float d = k[10u * i];
        if (bad == 9u && !(d > 0.0f && d < 3.0e38f)) {
            bad = i;
        }
    }
    DIAG_ASSERT4(diag, bad == 9u, static_cast<float>(bad),
                 static_cast<float>(slot), static_cast<float>(nodes[0]),
                 static_cast<float>(nodes[1]));
    float l[81];
    for (unsigned e = 0; e < 81u; ++e) {
        l[e] = 0.0f;
    }
    for (unsigned j = 0; j < 9u; ++j) {
        float pivot = k[10u * j];
        for (unsigned m = 0; m < j; ++m) {
            pivot -= l[9u * j + m] * l[9u * j + m];
        }
        const float floor = 1.0e-6f * k[10u * j];
        if (!(pivot > floor)) {
            pivot = floor;
        }
        const float root = fmath::sqrt(pivot);
        l[10u * j] = root;
        for (unsigned i = j + 1u; i < 9u; ++i) {
            float sum = k[9u * i + j];
            for (unsigned m = 0; m < j; ++m) {
                sum -= l[9u * i + m] * l[9u * j + m];
            }
            l[9u * i + j] = fmath::div(sum, root);
        }
    }
    unsigned at = 45u * slot;
    for (unsigned i = 0; i < 9u; ++i) {
        for (unsigned j = 0; j <= i; ++j) {
            factor[at] = l[9u * i + j];
            at += 1u;
        }
    }
}

// The vertices on no Cosserat edge, one thread per vertex: their block-Jacobi
// inverse, stored. A vertex an edge covers is left at zero here, for the edge
// blocks below to add into.
[[seam::entry(vertex_id)]]
[[seam::device_fn]] inline void cosserat_vertex_precond_row(
    const unsigned *covered,
    const float *inverse_diagonal,
    const float *residual,
    float *output,
    unsigned vertex_id) {
    for (unsigned i = 0; i < 3u; ++i) {
        float value = 0.0f;
        if (covered[vertex_id] == 0u) {
            for (unsigned j = 0; j < 3u; ++j) {
                value += inverse_diagonal[9u * vertex_id + i + 3u * j] *
                         residual[3u * vertex_id + j];
            }
        }
        output[3u * vertex_id + i] = value;
    }
}

// One edge block's solve, one thread per slot: its nine residual rows through
// the factor, the vertex rows added (two edges share an interior vertex) and
// the frame rows stored (a frame is one edge's own).
[[seam::entry(slot)]]
[[seam::device_fn]] inline void cosserat_schwarz_apply_row(
    const Vec2u *edge_nodes,
    const float *factor,
    const float *residual,
    float *output,
    compute::atomic_float_t *output_atomic,
    unsigned theta_base,
    unsigned slot) {
    const Vec2u nodes = edge_nodes[slot];
    const unsigned t = theta_base + 3u * slot;
    float y[9];
    for (unsigned i = 0; i < 3u; ++i) {
        y[i] = residual[3u * nodes[0] + i];
        y[3u + i] = residual[3u * nodes[1] + i];
        y[6u + i] = residual[t + i];
    }
    const unsigned base = 45u * slot;
    // Forward: L y' = y, row i of L starting at i (i + 1) / 2.
    for (unsigned i = 0; i < 9u; ++i) {
        const unsigned row = base + i * (i + 1u) / 2u;
        float sum = y[i];
        for (unsigned j = 0; j < i; ++j) {
            sum -= factor[row + j] * y[j];
        }
        y[i] = fmath::div(sum, factor[row + i]);
    }
    // Backward: L^T z = y'.
    for (unsigned ii = 0; ii < 9u; ++ii) {
        const unsigned i = 8u - ii;
        float sum = y[i];
        for (unsigned j = i + 1u; j < 9u; ++j) {
            sum -= factor[base + j * (j + 1u) / 2u + i] * y[j];
        }
        y[i] = fmath::div(sum, factor[base + i * (i + 1u) / 2u + i]);
    }
    for (unsigned i = 0; i < 3u; ++i) {
        compute::atomic_add(output_atomic + 3u * nodes[0] + i, y[i]);
        compute::atomic_add(output_atomic + 3u * nodes[1] + i, y[3u + i]);
        output[t + i] = y[6u + i];
    }
}

// THE FRAME ROWS OF A RESIDUAL IN FORCE UNITS, one thread per slot:
// `|r_theta|_1 / l`, l the slot's edge rest length. A frame row is a torque, and
// the torque a frame carries is balanced by forces of that size over its edge's
// length at the edge's vertices, so this puts the two blocks of the augmented
// residual in one unit and the solve measures one norm over both. Measured per
// block instead, the frames' seeded residual is round-off at a step whose frames
// start at rest, and a tolerance relative to it cannot be met.
[[seam::entry(slot)]]
[[seam::device_fn]] inline void cosserat_torque_scale_row(
    const unsigned *slot_edge,
    const EdgeProp *edge_prop,
    const float *residual,
    float *scaled,
    unsigned theta_base,
    unsigned slot) {
    const unsigned t = theta_base + 3u * slot;
    const float sum = fmath::abs(residual[t]) + fmath::abs(residual[t + 1u]) +
                      fmath::abs(residual[t + 2u]);
    scaled[slot] = fmath::div(sum, edge_prop[slot_edge[slot]].length);
}

// THE COARSE SPACE OF THE FRAME PRECONDITIONER: each chain's rigid roll.
//
// A chain no clamp and no `cosserat-fix-ends` holds can turn every frame about
// its own axis by one angle, which changes no shear, no bend and no twist, so
// only the rotational inertia resists it. The per-frame preconditioner above
// sees each frame's twist coupling to its neighbors instead, so this one mode
// sits about j / (dt^2 GJ / l) below the rest of the preconditioned spectrum,
// smaller as the segments shorten, and PCG spends its iterations on it. The
// correction adds it back exactly, one scalar per chain:
// z += e_c (e_c . r) / (e_c . K e_c), with e_c the chain's roll vector, which is
// SPD added to an SPD preconditioner. A held frame's roll is zero in e, so a
// chain a clamp holds carries no near-null roll and its correction is small.

// The roll vector, one thread per slot: the frame's own third axis, or zero for
// a frame this step holds.
[[seam::entry(slot)]]
[[seam::device_fn]] inline void cosserat_roll_row(
    const unsigned *held,
    const float *quat,
    float *roll,
    unsigned slot) {
    const Mat3x3f frame = cosserat_frame(quat, slot);
    const bool free_frame = held[slot] == 0u;
    for (unsigned i = 0; i < 3u; ++i) {
        roll[3u * slot + i] = free_frame ? frame(i, 2) : 0.0f;
    }
}

// One thread per chain: `dot[chain]` is the sum over the chain's slots of
// `roll_s . x_s`, `x` being an augmented vector whose frame rows start at
// `theta_base`. Serial within the chain, so the sum's order is fixed.
[[seam::entry(chain)]]
[[seam::device_fn]] inline void cosserat_chain_dot_row(
    const unsigned *chain_offset,
    const unsigned *chain_slot,
    const float *roll,
    const float *x,
    float *dot,
    unsigned theta_base,
    unsigned chain) {
    float sum = 0.0f;
    for (unsigned k = chain_offset[chain]; k < chain_offset[chain + 1u]; ++k) {
        const unsigned slot = chain_slot[k];
        const unsigned t = theta_base + 3u * slot;
        for (unsigned i = 0; i < 3u; ++i) {
            sum += roll[3u * slot + i] * x[t + i];
        }
    }
    dot[chain] = sum;
}

// One thread per chain: `output_s += roll_s (numerator / denominator)` over the
// chain's slots. A chain whose every frame is held has a zero roll vector and a
// zero denominator, and takes no correction.
[[seam::entry(chain)]]
[[seam::device_fn]] inline void cosserat_chain_add_row(
    const unsigned *chain_offset,
    const unsigned *chain_slot,
    const float *roll,
    const float *numerator,
    const float *denominator,
    float *output,
    unsigned theta_base,
    unsigned chain) {
    const float d = denominator[chain];
    if (!(d > 0.0f)) {
        return;
    }
    const float weight = fmath::div(numerator[chain], d);
    for (unsigned k = chain_offset[chain]; k < chain_offset[chain + 1u]; ++k) {
        const unsigned slot = chain_slot[k];
        const unsigned t = theta_base + 3u * slot;
        for (unsigned i = 0; i < 3u; ++i) {
            output[t + i] += weight * roll[3u * slot + i];
        }
    }
}

// The held frames' rows of a reduced vector set to zero, one thread per slot:
// applied to every residual, operator output and preconditioned residual, so the
// solve stays in the free subspace and a held frame's step stays at its seed,
// which `cosserat_prescribe_row` wrote. `held` is that row's per-step mask.
[[seam::entry(slot)]]
[[seam::device_fn]] inline void cosserat_project_fixed_row(
    const unsigned *held,
    float *vector,
    unsigned theta_base,
    unsigned slot) {
    if (held[slot] != 0u) {
        const unsigned t = theta_base + 3u * slot;
        vector[t] = 0.0f;
        vector[t + 1u] = 0.0f;
        vector[t + 2u] = 0.0f;
    }
}

// ----------------------------------------------------- the frame state updates

// The Newton step on a frame, one thread per slot: R <- exp(-scale dtheta) R,
// the rotational twin of eval_x -= scale dx, `scale` the fraction of the step the
// line search accepted. `step` holds the slots' three entries each, from 0.
//
// A FRAME ON A CLAMPED EDGE, both vertices prescribed, is moved instead by the
// exact transport of its edge from `before` to `after`, the iterate's positions
// on either side of the accepted step, so it follows its edge to round-off
// whatever fraction was accepted.
[[seam::entry(slot)]]
[[seam::device_fn]] inline void cosserat_step_row(
    const float *step,
    float scale,
    const Vec2u *edge_nodes,
    const unsigned *dof_mask,
    const Vec3f *before,
    const Vec3f *after,
    float *quat,
    DiagHandle diag,
    unsigned slot) {
    const Vec2u nodes = edge_nodes[slot];
    Mat3x3f rotation;
    if (dof_mask[nodes[0]] != 0u && dof_mask[nodes[1]] != 0u) {
        const Vec3f from0 = before[nodes[0]];
        const Vec3f from1 = before[nodes[1]];
        const Vec3f to0 = after[nodes[0]];
        const Vec3f to1 = after[nodes[1]];
        const Vec3f now = (from1 - from0).cast<float>();
        const Vec3f next = (to1 - to0).cast<float>();
        const Vec3f a = now / now.norm();
        const Vec3f b = next / next.norm();
        const float one_plus_c = 1.0f + a.dot(b);
        DIAG_ASSERT4(diag, one_plus_c > cosserat_reversal_floor(), one_plus_c,
                     static_cast<float>(slot), static_cast<float>(nodes[0]),
                     static_cast<float>(nodes[1]));
        if (!(one_plus_c > cosserat_reversal_floor())) {
            return;
        }
        rotation = cosserat_transport(a, b, one_plus_c);
    } else {
        const Vec3f increment(-scale * step[3u * slot],
                              -scale * step[3u * slot + 1u],
                              -scale * step[3u * slot + 2u]);
        rotation = pdrd_exp_so3(increment);
    }
    cosserat_store_frame(quat, slot,
                         cosserat_mul(rotation, cosserat_frame(quat, slot)));
}

// THE END DRIVE, one thread per slot, once per committed step: a held end whose
// rate is not zero turns about its own tangent by rate * dt, so the total turn
// is the rate times the elapsed time whatever the step size. The two ends of a
// chain carry opposite rates. A turn about the tangent leaves the director, so
// only the twist at the end's bend site feels it.
[[seam::entry(slot)]]
[[seam::device_fn]] inline void cosserat_drive_row(
    const float *end_omega,
    float dt,
    float *quat,
    unsigned slot) {
    const float rate = end_omega[slot];
    if (rate == 0.0f) {
        return;
    }
    const Mat3x3f frame = cosserat_frame(quat, slot);
    const float angle = rate * dt;
    const Vec3f turn(angle * frame(0, 2), angle * frame(1, 2), angle * frame(2, 2));
    cosserat_store_frame(quat, slot, cosserat_mul(pdrd_exp_so3(turn), frame));
}
