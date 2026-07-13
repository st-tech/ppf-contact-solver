// File: eigsolve.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// In-house closed-form symmetric eigensolvers (2x2, 3x3) in float32, replacing
// Eigen::SelfAdjointEigenSolver<...>::computeDirect on the device. Ported from
// the FD-validated eigsys/ reference (eig-hpp/eigsolve{2x2,3x3}.hpp), with:
//   * every literal float-ified (no double promotion on device),
//   * the Cardano acos argument clamped to [-1,1] (float32 hardening: near a
//     repeated/degenerate eigenvalue r -> +/-1 and acos'(r) -> inf),
//   * output eigenvalues sorted ASCENDING with eigenvectors permuted to match,
//     reproducing Eigen's convention that eigenanalysis.cu relies on.

#ifndef PPF_LINALG_EIGSOLVE_HPP
#define PPF_LINALG_EIGSOLVE_HPP

#include "smat.hpp"

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#define LA_HD __host__ __device__

namespace linalg {
namespace eig {

using M2 = SMat<float, 2, 2>;
using V2 = SMat<float, 2, 1>;
using M3 = SMat<float, 3, 3>;
using V3 = SMat<float, 3, 1>;

static constexpr float kPi = 3.14159265358979323846f;

// ============================== 2x2 =========================================
static LA_HD V2 eigvalues2(const M2 &A) {
    float a00 = A(0, 0), a01 = A(0, 1), a11 = A(1, 1);
    float tmp = a00 - a11;
    float D = 0.5f * sqrtf(tmp * tmp + 4.0f * a01 * a01);
    float mid = 0.5f * (a00 + a11);
    return V2(mid - D, mid + D); // ascending
}
static LA_HD V2 rot90(const V2 &x) { return V2(x[1], -x[0]); }
static LA_HD V2 find_ortho2(const M2 &A, const V2 &x, float sqr_eps) {
    V2 u = rot90(A.col(0));
    V2 v = rot90(A.col(1));
    if (u.squaredNorm() > sqr_eps)
        return u.normalized();
    else if (v.squaredNorm() > sqr_eps)
        return v.normalized();
    else
        return rot90(x);
}
static LA_HD M2 eigvectors2(const M2 &A, const V2 &lmd) {
    float eps = 1e-7f;
    float sqr_eps = eps * eps;
    V2 u = find_ortho2(A - lmd[0] * M2::Identity(), V2(0.0f, 1.0f), sqr_eps);
    // A symmetric 2x2 has EXACTLY orthogonal eigenvectors: the 2nd is the exact
    // orthogonal complement rot90(u). Solving (A - lmd[1] I) independently makes
    // u,v collapse toward parallel for clustered lmd -> ortho up to 0.29. rot90 is
    // a pure swap+negate (no division, |rot90(u)|=|u|), so it never NaNs.
    V2 v = rot90(u);
    M2 result;
    result << u, v;
    return result;
}

// Symmetric 2x2 eigensolve; eigenvalues ASCENDING, eigenvectors as columns.
static LA_HD void symm2x2(const M2 &A, V2 &val, M2 &vec) {
    // Infinity-norm (max-abs) scale, NOT Frobenius A.norm(): squaring an entry
    // ~1e20 gives ~1e40 > FLT_MAX (3.4e38) -> scale=inf -> B=0 -> val=inf*0=NaN.
    // fabsf/fmaxf never square, so no overflow on huge inputs (and no subnormal
    // underflow on tiny ones).
    float scale = 0.0f;
    scale = fmaxf(scale, fabsf(A.m[0]));
    scale = fmaxf(scale, fabsf(A.m[1]));
    scale = fmaxf(scale, fabsf(A.m[2]));
    scale = fmaxf(scale, fabsf(A.m[3]));
    if (scale <= 0.0f) {
        val = V2(0.0f, 0.0f);
        vec = M2::Identity();
        return;
    }
    M2 B = A / scale;
    V2 lmd = eigvalues2(B);       // ascending
    M2 evec = eigvectors2(B, lmd);
    val = scale * lmd;
    vec = evec;
}

// ============================== 3x3 =========================================
static LA_HD V3 eigvalues3(const M3 &A) {
    float p1 = A(0, 1) * A(0, 1) + A(0, 2) * A(0, 2) + A(1, 2) * A(1, 2);
    float q = A.trace() / 3.0f;
    float p2 = (A(0, 0) - q) * (A(0, 0) - q) + (A(1, 1) - q) * (A(1, 1) - q) +
               (A(2, 2) - q) * (A(2, 2) - q) + 2.0f * p1;
    float p = sqrtf(p2 / 6.0f);
    if (fabsf(p) < 1e-8f) {
        return V3(0.0f, 0.0f, 0.0f);
    }
    M3 B = (1.0f / p) * (A - q * M3::Identity());
    float r = B.determinant() / 2.0f;
    // float32 hardening: clamp before acos (near-degenerate blows up otherwise).
    if (r < -1.0f)
        r = -1.0f;
    else if (r > 1.0f)
        r = 1.0f;
    float phi = acosf(r) / 3.0f;
    float eig1 = q + 2.0f * p * cosf(phi);                    // largest
    float eig3 = q + 2.0f * p * cosf(phi + 2.0f * kPi / 3.0f); // smallest
    float eig2 = 3.0f * q - eig1 - eig3;
    return V3(eig1, eig2, eig3); // descending: eig1 >= eig2 >= eig3
}

static LA_HD V3 pick_largest(const V3 &a, const V3 &b, const V3 &c) {
    float an = a.squaredNorm(), bn = b.squaredNorm(), cn = c.squaredNorm();
    if (an > bn) {
        return (an > cn) ? a : c;
    } else {
        return (bn > cn) ? b : c;
    }
}

struct ortho3 {
    V3 v1, v2;
};
static LA_HD ortho3 find_ortho3x3(const M3 &A) {
    float eps = 1e-7f;
    V3 u = A.col(0), v = A.col(1), w = A.col(2);
    V3 uv = u.cross(v), vw = v.cross(w), wu = w.cross(u);
    V3 q = pick_largest(uv, vw, wu);
    if (q.squaredNorm() < eps) {
        V3 pp = pick_largest(u, v, w);
        V3 x = pp.cross(V3(1.0f, 0.0f, 0.0f));
        if (x.squaredNorm() < eps) {
            x = pp.cross(V3(0.0f, 1.0f, 0.0f));
        }
        V3 y = pp.cross(x);
        ortho3 o;
        o.v1 = x.normalized();
        o.v2 = y.normalized();
        return o;
    } else {
        ortho3 o;
        o.v1 = q.normalized();
        o.v2 = V3(0.0f, 0.0f, 0.0f);
        return o;
    }
}

static LA_HD M3 eigvectors3x3(const M3 &A, const V3 &lmd) {
    ortho3 uv = find_ortho3x3(A - lmd[0] * M3::Identity());
    if (uv.v2.squaredNorm() == 0.0f) {
        ortho3 tmp = find_ortho3x3(A - lmd[1] * M3::Identity());
        // Clustered lmd[0]~lmd[1]: the cross-product null extraction for lmd[1]
        // loses ~1e-4 (cofactor ~ product of eigen-gaps), so tmp.v1 carries a
        // spurious v1-component. Gram-Schmidt against the accurate anchor v1
        // removes it (subtraction is division-free).
        V3 g = tmp.v1 - uv.v1.dot(tmp.v1) * uv.v1;
        uv.v2 = g.normalized();
    }
    // Right-handed orthonormal completion. normalized() guards a (near-)parallel
    // v1,v2 (zero cross) -> zero, never 0/0. Recomputing v2 = w x v1 makes all
    // three columns mutually orthonormal to float precision.
    V3 w = uv.v1.cross(uv.v2).normalized();
    uv.v2 = w.cross(uv.v1);
    M3 result;
    result << uv.v1, uv.v2, w;
    return result;
}

// Symmetric 3x3 eigensolve; eigenvalues ASCENDING, eigenvectors as columns.
static LA_HD void symm3x3(const M3 &A, V3 &val, M3 &vec) {
    // Infinity-norm (max-abs) scale (see symm2x2): Frobenius A.norm() squares
    // each entry and overflows float32 for large-magnitude matrices -> NaN.
    float scale = 0.0f;
#pragma unroll
    for (int i = 0; i < 9; ++i)
        scale = fmaxf(scale, fabsf(A.m[i]));
    if (scale <= 0.0f) {
        val = V3(0.0f, 0.0f, 0.0f);
        vec = M3::Identity();
        return;
    }
    M3 B = (1.0f / scale) * A;
    V3 lmd = eigvalues3(B); // descending
    M3 evec;
    if (lmd.squaredNorm() > 0.0f) {
        evec = eigvectors3x3(B, lmd);
    } else {
        // Isotropic B: eigvalues3 hit its p~0 return, so the triple eigenvalue is
        // q = trace/3 exactly, WITH sign. The old column-norm average is always
        // non-negative and flipped the sign for negative-definite A (A=-5*I -> +5),
        // which then survived the downstream SPD clamp as spurious +stiffness. B is
        // bounded (entries in [-1,1] after the max-abs scale; A=0 is caught by the
        // scale<=0 guard above), so trace/3 never divides by anything data-dependent.
        evec = M3::Identity();
        float q = B.trace() / 3.0f;
        lmd = V3(q, q, q);
    }
    // eigsys emits descending (lmd[0] largest, column 0 its vector). Reverse to
    // ascending to match Eigen's SelfAdjointEigenSolver ordering.
    V3 asc = V3(scale * lmd[2], scale * lmd[1], scale * lmd[0]);
    M3 vasc;
    vasc.col(0) = evec.col(2);
    vasc.col(1) = evec.col(1);
    vasc.col(2) = evec.col(0);
    val = asc;
    vec = vasc;
}

} // namespace eig

// Project a symmetric NxN matrix (in place) onto its nearest positive-
// semidefinite matrix in the Frobenius sense: eigen-decompose and clamp every
// eigenvalue up to `floor` (pass 0 for the nearest PSD). Cyclic Jacobi in
// float32; meant for the small dense bending Hessians (rod bend N=9, shell bend
// N=12). The caller must pass a numerically symmetric matrix. This is the
// projected-Newton step: it keeps the exact Hessian's positive-curvature
// content and only removes the indefinite directions, so the assembled Newton
// system stays SPD (the true bending Hessian, g g^T plus (theta-theta0) d2theta,
// is otherwise indefinite and breaks the SPD PCG solve).
template <int N> static LA_HD void psd_project_symmetric(SMat<float, N, N> &A,
                                                         float floor) {
    SMat<float, N, N> V = SMat<float, N, N>::Identity();
    float scale = 0.0f;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            scale = fmaxf(scale, fabsf(A(i, j)));
    if (scale == 0.0f) {
        if (floor > 0.0f)
            for (int i = 0; i < N; ++i)
                A(i, i) = floor;
        return;
    }
    const float tiny = 1e-7f * scale;
    for (int sweep = 0; sweep < 24; ++sweep) {
        float off = 0.0f;
        for (int p = 0; p < N; ++p)
            for (int q = p + 1; q < N; ++q)
                off = fmaxf(off, fabsf(A(p, q)));
        if (off <= tiny)
            break;
        for (int p = 0; p < N; ++p) {
            for (int q = p + 1; q < N; ++q) {
                const float apq = A(p, q);
                if (fabsf(apq) <= tiny)
                    continue;
                const float app = A(p, p);
                const float aqq = A(q, q);
                // Jacobi rotation angle that zeroes A(p, q).
                const float tau = (aqq - app) / (2.0f * apq);
                const float t = (tau >= 0.0f ? 1.0f : -1.0f) /
                                (fabsf(tau) + sqrtf(1.0f + tau * tau));
                const float c = 1.0f / sqrtf(1.0f + t * t);
                const float s = t * c;
                A(p, p) = c * c * app - 2.0f * s * c * apq + s * s * aqq;
                A(q, q) = s * s * app + 2.0f * s * c * apq + c * c * aqq;
                A(p, q) = 0.0f;
                A(q, p) = 0.0f;
                for (int k = 0; k < N; ++k) {
                    if (k == p || k == q)
                        continue;
                    const float akp = A(k, p);
                    const float akq = A(k, q);
                    const float np = c * akp - s * akq;
                    const float nq = s * akp + c * akq;
                    A(k, p) = np;
                    A(p, k) = np;
                    A(k, q) = nq;
                    A(q, k) = nq;
                }
                for (int k = 0; k < N; ++k) {
                    const float vkp = V(k, p);
                    const float vkq = V(k, q);
                    V(k, p) = c * vkp - s * vkq;
                    V(k, q) = s * vkp + c * vkq;
                }
            }
        }
    }
    // Eigenvalues are the converged diagonal; clamp and reconstruct
    // A = V diag(max(lambda, floor)) V^T.
    float lam[N];
    for (int i = 0; i < N; ++i)
        lam[i] = fmaxf(A(i, i), floor);
    for (int i = 0; i < N; ++i) {
        for (int j = i; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += V(i, k) * lam[k] * V(j, k);
            A(i, j) = sum;
            A(j, i) = sum;
        }
    }
}

} // namespace linalg

#undef LA_HD
#endif
