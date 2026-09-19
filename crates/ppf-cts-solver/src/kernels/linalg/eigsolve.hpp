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

#ifndef LINALG_EIGSOLVE_HPP
#define LINALG_EIGSOLVE_HPP

// Unconditional, so an offline unit that names this header gets `linalg::SMat`
// without depending on having been handed it first; the run-time assembler
// neutralizes the line as it splices. See the same split in smat.hpp.
#include "smat.hpp"

#ifndef SM_MSL_CONCAT
// Supplies the single-precision transcendentals used by the CUDA and host
// fallbacks below. The Metal path never reaches this include (the shader
// prologue defines SM_MSL_CONCAT and supplies its own SM_COS / SM_ACOS), which
// is what keeps float_math.hpp's CUDA and std spellings out of MSL.
#include "../float_math.hpp"
#endif

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#define LA_HD __host__ __device__

#ifndef SM_THREAD
#define SM_THREAD
#define EIG_UNDEF_SM_THREAD
#endif
#ifndef SM_SQRT
#define SM_SQRT sqrtf
#define EIG_UNDEF_SM_SQRT
#endif
#ifndef SM_ABS
#define SM_ABS fabsf
#define EIG_UNDEF_SM_ABS
#endif
#ifndef SM_MAX
#define SM_MAX fmaxf
#define EIG_UNDEF_SM_MAX
#endif
// acosf carries no argument reduction (its argument is inherently bounded to
// [-1, 1]), so it emits no FP64 and needs no replacement.
#ifndef SM_ACOS
#define SM_ACOS acosf
#define EIG_UNDEF_SM_ACOS
#endif
// NOT cosf: the library cosf reduces its argument with 64-bit integer and
// double arithmetic on its accurate path, so a kernel that merely calls it
// emits FP64, which the solver's single-precision-on-GPU rule forbids.
// fmath::cos_bounded is the hardware special-function unit, float throughout,
// and asserts the bound it relies on. The precondition holds here by
// construction: the Cardano angle phi lies in [0, pi/3], so both arguments
// below stay inside one turn. The Metal backend defines SM_COS itself, where
// the shader compiler's cos is already single precision.
#ifndef SM_COS
#define SM_COS fmath::cos_bounded
#define EIG_UNDEF_SM_COS
#endif

namespace linalg {
namespace eig {

using M2 = SMat<float, 2, 2>;
using V2 = SMat<float, 2, 1>;
using M3 = SMat<float, 3, 3>;
using V3 = SMat<float, 3, 1>;

// A constexpr FUNCTION, not a constexpr variable: this header compiles as both
// CUDA and MSL, MSL rejects a program-scope constexpr variable, and a float
// cannot be an enumerator. LA_HD is empty under MSL, so this reads there as the
// plain `constexpr float kPi() { ... }` form.
LA_HD constexpr float kPi() { return 3.14159265358979323846f; }

// ============================== 2x2 =========================================
static LA_HD V2 eigvalues2(SM_THREAD const M2 &A) {
    float a00 = A(0, 0), a01 = A(0, 1), a11 = A(1, 1);
    float tmp = a00 - a11;
    float D = 0.5f * SM_SQRT(tmp * tmp + 4.0f * a01 * a01);
    float mid = 0.5f * (a00 + a11);
    return V2(mid - D, mid + D); // ascending
}
static LA_HD V2 rot90(SM_THREAD const V2 &x) { return V2(x[1], -x[0]); }
static LA_HD V2 find_ortho2(SM_THREAD const M2 &A,
                            SM_THREAD const V2 &x, float sqr_eps) {
    V2 u = rot90(A.col(0));
    V2 v = rot90(A.col(1));
    if (u.squaredNorm() > sqr_eps)
        return u.normalized();
    else if (v.squaredNorm() > sqr_eps)
        return v.normalized();
    else
        return rot90(x);
}
static LA_HD M2 eigvectors2(SM_THREAD const M2 &A,
                            SM_THREAD const V2 &lmd) {
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
static LA_HD void symm2x2(SM_THREAD const M2 &A, SM_THREAD V2 &val,
                          SM_THREAD M2 &vec) {
    // Infinity-norm (max-abs) scale, NOT Frobenius A.norm(): squaring an entry
    // ~1e20 gives ~1e40 > FLT_MAX (3.4e38) -> scale=inf -> B=0 -> val=inf*0=NaN.
    // fabsf/fmaxf never square, so no overflow on huge inputs (and no subnormal
    // underflow on tiny ones).
    float scale = 0.0f;
    scale = SM_MAX(scale, SM_ABS(A.m[0]));
    scale = SM_MAX(scale, SM_ABS(A.m[1]));
    scale = SM_MAX(scale, SM_ABS(A.m[2]));
    scale = SM_MAX(scale, SM_ABS(A.m[3]));
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
static LA_HD V3 eigvalues3(SM_THREAD const M3 &A) {
    float p1 = A(0, 1) * A(0, 1) + A(0, 2) * A(0, 2) + A(1, 2) * A(1, 2);
    float q = A.trace() / 3.0f;
    float p2 = (A(0, 0) - q) * (A(0, 0) - q) + (A(1, 1) - q) * (A(1, 1) - q) +
               (A(2, 2) - q) * (A(2, 2) - q) + 2.0f * p1;
    float p = SM_SQRT(p2 / 6.0f);
    if (SM_ABS(p) < 1e-8f) {
        return V3(0.0f, 0.0f, 0.0f);
    }
    M3 B = (1.0f / p) * (A - q * M3::Identity());
    float r = B.determinant() / 2.0f;
    // float32 hardening: clamp before acos (near-degenerate blows up otherwise).
    if (r < -1.0f)
        r = -1.0f;
    else if (r > 1.0f)
        r = 1.0f;
    // phi lands in [0, pi/3], so both cosine arguments are inside one turn,
    // which is the bound SM_COS relies on (see the seam block above).
    float phi = SM_ACOS(r) / 3.0f;
    float eig1 = q + 2.0f * p * SM_COS(phi); // largest
    float eig3 =
        q + 2.0f * p * SM_COS(phi + 2.0f * kPi() / 3.0f); // smallest
    float eig2 = 3.0f * q - eig1 - eig3;
    return V3(eig1, eig2, eig3); // descending: eig1 >= eig2 >= eig3
}

static LA_HD V3 pick_largest(SM_THREAD const V3 &a,
                             SM_THREAD const V3 &b,
                             SM_THREAD const V3 &c) {
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
static LA_HD ortho3 find_ortho3x3(SM_THREAD const M3 &A) {
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

static LA_HD M3 eigvectors3x3(SM_THREAD const M3 &A,
                              SM_THREAD const V3 &lmd) {
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
    // Degenerate-triad fallback: for a NEAR-isotropic A whose eigenvalue split
    // sits between eigvalues3's exact-isotropic cutoff (p < 1e-8) and fp32
    // eigenvector resolution, every quantity find_ortho3x3 tests is of order
    // (split)^2 and falls below its absolute eps, both axis probes included,
    // and normalized() of the resulting exact zeros yields an all-zero triad.
    // Downstream that zero basis reconstructs a ZERO matrix function (found on
    // the dev-diffsim fork as an all-zero block-Jacobi preconditioner block on
    // a free sand grain's mass/dt^2-dominated diagonal block, fed through the
    // eigendecomposition invert(): the vertex's DOF silently froze, since an
    // exactly-zero block contributes nothing to r.z/pAp and no SPD guard
    // fires, and PCG stalled). A zero triad can only arise when A is isotropic
    // to within fp32 eigenvector resolution, where ANY orthonormal basis
    // diagonalizes A to working precision: return Identity. Healthy paths
    // always produce unit columns, so the 0.5 threshold cannot misfire.
    if (uv.v1.squaredNorm() < 0.5f || w.squaredNorm() < 0.5f) {
        return M3::Identity();
    }
    M3 result;
    result << uv.v1, uv.v2, w;
    return result;
}

// Symmetric 3x3 eigensolve; eigenvalues ASCENDING, eigenvectors as columns.
static LA_HD void symm3x3(SM_THREAD const M3 &A, SM_THREAD V3 &val,
                          SM_THREAD M3 &vec) {
    // Infinity-norm (max-abs) scale (see symm2x2): Frobenius A.norm() squares
    // each entry and overflows float32 for large-magnitude matrices -> NaN.
    float scale = 0.0f;
#pragma unroll
    for (int i = 0; i < 9; ++i)
        scale = SM_MAX(scale, SM_ABS(A.m[i]));
    if (scale <= 0.0f) {
        val = V3(0.0f, 0.0f, 0.0f);
        vec = M3::Identity();
        return;
    }
    M3 D = (1.0f / scale) * A;
    // Assembly promises symmetry, but averaging the mirrored entries makes the
    // promise structural at the eigensolver boundary and matches the proven
    // shared_math implementation.
    for (int p = 0; p < 3; ++p) {
        for (int q = p + 1; q < 3; ++q) {
            float s = 0.5f * (D(p, q) + D(q, p));
            D(p, q) = s;
            D(q, p) = s;
        }
    }
    M3 V = M3::Identity();
    for (int sweep = 0; sweep < 8; ++sweep) {
        for (int p = 0; p < 2; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                float apq = D(p, q);
                if (SM_ABS(apq) > 1.0e-20f) {
                    float app = D(p, p);
                    float aqq = D(q, q);
                    float theta = 0.5f * (aqq - app) / apq;
                    float sign = theta >= 0.0f ? 1.0f : -1.0f;
                    float t =
                        sign / (SM_ABS(theta) +
                                SM_SQRT(theta * theta + 1.0f));
                    float c = 1.0f / SM_SQRT(t * t + 1.0f);
                    float s = t * c;
                    for (int k = 0; k < 3; ++k) {
                        float dkp = D(k, p), dkq = D(k, q);
                        D(k, p) = c * dkp - s * dkq;
                        D(k, q) = s * dkp + c * dkq;
                    }
                    for (int k = 0; k < 3; ++k) {
                        float dpk = D(p, k), dqk = D(q, k);
                        D(p, k) = c * dpk - s * dqk;
                        D(q, k) = s * dpk + c * dqk;
                    }
                    for (int k = 0; k < 3; ++k) {
                        float vkp = V(k, p), vkq = V(k, q);
                        V(k, p) = c * vkp - s * vkq;
                        V(k, q) = s * vkp + c * vkq;
                    }
                }
            }
        }
    }
    V3 eigenvalues(scale * D(0, 0), scale * D(1, 1), scale * D(2, 2));
    // Sort ascending, carrying columns of V with their eigenvalues.
    for (int i = 0; i < 2; ++i) {
        int smallest = i;
        for (int j = i + 1; j < 3; ++j) {
            if (eigenvalues[j] < eigenvalues[smallest]) {
                smallest = j;
            }
        }
        if (smallest != i) {
            float l = eigenvalues[i];
            eigenvalues[i] = eigenvalues[smallest];
            eigenvalues[smallest] = l;
            V3 column = V.col(i);
            V.col(i) = V.col(smallest);
            V.col(smallest) = column;
        }
    }
    val = eigenvalues;
    vec = V;
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
template <int N>
static LA_HD void
psd_project_symmetric(SM_THREAD SMat<float, N, N> &A, float floor) {
    SMat<float, N, N> V = SMat<float, N, N>::Identity();
    float scale = 0.0f;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            scale = SM_MAX(scale, SM_ABS(A(i, j)));
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
                off = SM_MAX(off, SM_ABS(A(p, q)));
        if (off <= tiny)
            break;
        for (int p = 0; p < N; ++p) {
            for (int q = p + 1; q < N; ++q) {
                const float apq = A(p, q);
                if (SM_ABS(apq) <= tiny)
                    continue;
                const float app = A(p, p);
                const float aqq = A(q, q);
                // Jacobi rotation angle that zeroes A(p, q).
                const float tau = (aqq - app) / (2.0f * apq);
                const float t = (tau >= 0.0f ? 1.0f : -1.0f) /
                                (SM_ABS(tau) + SM_SQRT(1.0f + tau * tau));
                const float c = 1.0f / SM_SQRT(1.0f + t * t);
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
        lam[i] = SM_MAX(A(i, i), floor);
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

#ifdef EIG_UNDEF_SM_COS
#undef SM_COS
#undef EIG_UNDEF_SM_COS
#endif
#ifdef EIG_UNDEF_SM_ACOS
#undef SM_ACOS
#undef EIG_UNDEF_SM_ACOS
#endif
#ifdef EIG_UNDEF_SM_MAX
#undef SM_MAX
#undef EIG_UNDEF_SM_MAX
#endif
#ifdef EIG_UNDEF_SM_ABS
#undef SM_ABS
#undef EIG_UNDEF_SM_ABS
#endif
#ifdef EIG_UNDEF_SM_SQRT
#undef SM_SQRT
#undef EIG_UNDEF_SM_SQRT
#endif
#ifdef EIG_UNDEF_SM_THREAD
#undef SM_THREAD
#undef EIG_UNDEF_SM_THREAD
#endif
#undef LA_HD
#endif
