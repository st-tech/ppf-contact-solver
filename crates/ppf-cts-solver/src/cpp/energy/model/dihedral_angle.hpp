// File: dihedral_angle.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef DIHEDRAL_ANGLE_HPP
#define DIHEDRAL_ANGLE_HPP

#include "../../common.hpp"
#include "../../data.hpp"
#include "../../linalg/eigsolve.hpp"

namespace dihedral_angle {

__device__ static Vec4u remap(Vec4u hinge) {
    return Vec4u(hinge[2], hinge[1], hinge[0], hinge[3]);
}

__device__ static Mat3x4f face_dihedral_angle_grad(const Vec3f &v2,
                                                   const Vec3f &v0,
                                                   const Vec3f &v1,
                                                   const Vec3f &v3) {
    Mat3x4f result;
    const Vec3f e0 = (v1 - v0);
    const Vec3f e1 = (v2 - v0);
    const Vec3f e2 = (v3 - v0);
    const Vec3f e3 = (v2 - v1);
    const Vec3f e4 = (v3 - v1);
    const Vec3f n1 = e0.cross(e1);
    const Vec3f n2 = e2.cross(e0);
    const float n1_sqnm = n1.squaredNorm();
    const float n2_sqnm = n2.squaredNorm();
    const float e0_norm = e0.norm();
    assert(n1_sqnm > 0.0f);
    assert(n2_sqnm > 0.0f);
    assert(e0_norm > 0.0f);
    result << -e0_norm / n1_sqnm * n1,
        -e0.dot(e3) / (e0_norm * n1_sqnm) * n1 -
            e0.dot(e4) / (e0_norm * n2_sqnm) * n2,
        e0.dot(e1) / (e0_norm * n1_sqnm) * n1 +
            e0.dot(e2) / (e0_norm * n2_sqnm) * n2,
        -e0_norm / n2_sqnm * n2;
    return result;
}

// Float-position variant of face_dihedral_angle_grad (identical math on plain
// float positions). Used by the element-local finite difference of the angle
// gradient that builds the exact bending Hessian term (theta - theta0)
// d2theta/dx2 in face_compute_force_hessian. Positions are local floats (exact
// differences of the fixed-point positions), so the FD step is independent of
// the global position lattice. Degenerate configurations return zero instead
// of asserting, since the FD probes can brush against them.
__device__ static Mat3x4f face_dihedral_angle_grad_f(const Vec3f &v2,
                                                     const Vec3f &v0,
                                                     const Vec3f &v1,
                                                     const Vec3f &v3) {
    Mat3x4f result;
    const Vec3f e0 = v1 - v0;
    const Vec3f e1 = v2 - v0;
    const Vec3f e2 = v3 - v0;
    const Vec3f e3 = v2 - v1;
    const Vec3f e4 = v3 - v1;
    const Vec3f n1 = e0.cross(e1);
    const Vec3f n2 = e2.cross(e0);
    const float n1_sqnm = n1.squaredNorm();
    const float n2_sqnm = n2.squaredNorm();
    const float e0_norm = e0.norm();
    if (n1_sqnm <= 0.0f || n2_sqnm <= 0.0f || e0_norm <= 0.0f) {
        return Mat3x4f::Zero();
    }
    result << -e0_norm / n1_sqnm * n1,
        -e0.dot(e3) / (e0_norm * n1_sqnm) * n1 -
            e0.dot(e4) / (e0_norm * n2_sqnm) * n2,
        e0.dot(e1) / (e0_norm * n1_sqnm) * n1 +
            e0.dot(e2) / (e0_norm * n2_sqnm) * n2,
        -e0_norm / n2_sqnm * n2;
    return result;
}

__device__ static float face_dihedral_angle(const Vec3f &v0, const Vec3f &v1,
                                            const Vec3f &v2,
                                            const Vec3f &v3) {
    const Vec3f n1 = (v1 - v0).cross((v2 - v0));
    const Vec3f n2 = (v2 - v3).cross((v1 - v3));
    // atan2 form of the signed dihedral angle. The acosf(clamped dot) form is
    // ill-conditioned near flat hinges (d acos / d dot -> inf as dot -> 1),
    // amplifying one fp32 ulp of the normalized dot into ~ulp/theta of angle
    // error; for near-flat cloth that deterministic error reaches
    // ~1e-5..1e-4 rad and, multiplied by the bending stiffness, dominates the
    // force-evaluation noise floor. atan2(sin, cos) has O(1) conditioning at
    // every angle and preserves the old sign convention (angle > 0 iff
    // (n2 x n1) . (v1 - v2) > 0): sin and cos share the common positive
    // factor |n1||n2|, which cancels in atan2.
    const Vec3f e = (v1 - v2);
    const float e_norm = e.norm();
    if (e_norm <= 0.0f) {
        return 0.0f;
    }
    const float sin_t = n2.cross(n1).dot(e) / e_norm;
    const float cos_t = n1.dot(n2);
    return atan2f(sin_t, cos_t);
}

__device__ static float hinge_compute_energy(const Vec<Vec3f> &vertex,
                                             Vec4u hinge, float rest_angle) {
    hinge = remap(hinge);
    const Vec3f x0 = vertex[hinge[0]];
    const Vec3f x1 = vertex[hinge[1]];
    const Vec3f x2 = vertex[hinge[2]];
    const Vec3f x3 = vertex[hinge[3]];
    const float angle = face_dihedral_angle(x0, x1, x2, x3);
    const float diff = angle - rest_angle;
    return 0.5f * diff * diff;
}

// Bending force and analytically PSD-projected Hessian for the shell hinge
// energy E = 1/2 (theta - theta0)^2 (per unit stiffness). force is the exact
// gradient (unchanged). The exact Hessian is in general INDEFINITE and breaks
// the SPD PCG solve, so we assemble the projected-Newton Hessian via the
// closed-form angle-energy eigensystem of Wu & Kim 2023 ("An Eigenanalysis of
// Angle-Based Deformation Energies", PACMCGIT 6(3)). Unlike the rod, a shell has
// d2F/dx2 != 0, so the Hessian is J^T d2Psi/dF2 J + (dPsi/dF)^T d2F/dx2: the
// first term is the section-3 6-eigenpair F-Hessian (mapped by the projected-
// edge Jacobian, Eqns 43-44) and the second is a rank-4 term W (Eqns 45-57). We
// PSD-project each part and add them. Verified against finite differences over
// thousands of random configs (which also exposed four typos in the paper:
// the rod u-vector b-sign; the shell section-3 sign S3=-sign(tb.t1); Eqn 57's
// p3 x0-block c-signs; and a missing kappa1 factor in Eqn 49's d-coefficients).
__device__ static void face_compute_force_hessian(const Vec<Vec3f> &vertex,
                                                  Vec4u &hinge,
                                                  float rest_angle,
                                                  Mat3x4f &force,
                                                  Mat12x12f &hess) {
    hinge = remap(hinge);
    const Vec3f X0 = vertex[hinge[0]];
    const Vec3f X1 = vertex[hinge[1]];
    const Vec3f X2 = vertex[hinge[2]];
    const Vec3f X3 = vertex[hinge[3]];
    const float angle = face_dihedral_angle(X0, X1, X2, X3);
    const Mat3x4f angle_grad = face_dihedral_angle_grad(X0, X1, X2, X3);
    force = (angle - rest_angle) * angle_grad;
    hess = Mat12x12f::Zero();

    // Wu & Kim hinge stencil in the codebase's remapped vertex order: the shared
    // edge is (X2, X1) and the flaps are X0, X3, so the paper stencil
    // (P0,P1=edge, P2,P3=flaps) maps to (P0,P1,P2,P3) = (X2, X1, X0, X3). All
    // geometry below is in the paper P-order; the final block-permutation
    // (swap blocks 0<->2) rewrites the 12x12 into the codebase (X0,X1,X2,X3)
    // order for embedding. theta = face_dihedral_angle == the paper's angle.
    const Vec3f y1 = (X1 - X2); // P1 - P0
    const Vec3f y2 = (X0 - X2); // P2 - P0
    const Vec3f y3 = (X3 - X2); // P3 - P0
    const float ly1 = y1.norm();
    if (ly1 <= 1e-9f) {
        return;
    }
    const Vec3f t1 = y1 / ly1;
    const Mat3x3f sigma = Mat3x3f::Identity() - t1 * t1.transpose();
    const Vec3f z0 = y2 - t1 * t1.dot(y2);
    const Vec3f z1 = y3 - t1 * t1.dot(y3);
    const float l0 = z0.norm();
    const float l1 = z1.norm();
    Vec3f bcr = z0.cross(z1);
    const float lb = bcr.norm();
    if (l0 <= 1e-9f || l1 <= 1e-9f || lb <= 1e-9f * l0 * l1) {
        return;
    }
    const Vec3f tb = bcr / lb;
    const float Sgeom = (tb.dot(t1) >= 0.0f) ? 1.0f : -1.0f;
    const float g = angle - rest_angle; // = theta - theta0 (stiffness mu = 1)

    // Local 12x12 in the paper P-order; permuted into hess at the end.
    Mat12x12f Hp = Mat12x12f::Zero();

    // ---- F-part: section-3 6-eigenpair Hessian (S3 = -Sgeom), mapped to the 12
    // DOFs by J = dF/dx (Eqns 43-44). J^T [qa; qb] has blocks (P0,P1,P2,P3):
    //   P0 = (-eta2-sigma)^T qa + (-eta3-sigma)^T qb,  P1 = eta2^T qa + eta3^T qb,
    //   P2 = sigma qa,  P3 = sigma qb   (sigma is symmetric).
    const Vec3f e0p = z0.cross(tb);
    const Vec3f e1p = z1.cross(tb);
    const float S3 = -Sgeom;
    const float w = S3 * g; // h = 1
    const float gamma = (l1 * l1) / (l0 * l0);
    const float gm1 = gamma - 1.0f;
    const float gp1 = gamma + 1.0f;
    const float r = sqrtf(4.0f * w * w * (gm1 / gp1) * (gm1 / gp1) + 1.0f);
    const float Rm = sqrtf(fmaxf(0.0f, 2.0f * (2.0f * w * w + 1.0f - r)));
    const float Rp = sqrtf(fmaxf(0.0f, 2.0f * (2.0f * w * w + 1.0f + r)));
    const float Lh = 1.0f / (l0 * l0) + 1.0f / (l1 * l1);
    const float cc = 4.0f * w / gp1;
    const Mat3x3f eta2 =
        (-1.0f / ly1) * (t1.dot(y2) * sigma + t1 * z0.transpose());
    const Mat3x3f eta3 =
        (-1.0f / ly1) * (t1.dot(y3) * sigma + t1 * z1.transpose());
    const Mat3x3f e2T = eta2.transpose();
    const Mat3x3f e3T = eta3.transpose();
    const Mat3x3f m0 = -e2T - sigma; // (-eta2-sigma)^T
    const Mat3x3f m1 = -e3T - sigma; // (-eta3-sigma)^T
    auto accF = [&](float lam, const Vec3f &qa, const Vec3f &qb) {
        if (lam <= 0.0f) {
            return;
        }
        const float n2 = qa.dot(qa) + qb.dot(qb);
        if (n2 <= 1e-20f) {
            return;
        }
        const Vec3f b0 = m0 * qa + m1 * qb;
        const Vec3f b1 = e2T * qa + e3T * qb;
        const Vec3f b2 = sigma * qa;
        const Vec3f b3 = sigma * qb;
        float p12[12];
        for (int c = 0; c < 3; ++c) {
            p12[c] = b0[c];
            p12[3 + c] = b1[c];
            p12[6 + c] = b2[c];
            p12[9 + c] = b3[c];
        }
        const float s = lam / n2;
        for (int i = 0; i < 12; ++i) {
            for (int j = 0; j < 12; ++j) {
                Hp(i, j) += s * p12[i] * p12[j];
            }
        }
    };
    const int sr[4] = {-1, -1, 1, 1};
    const float RR[4] = {Rm, Rm, Rp, Rp};
    const int sR[4] = {-1, 1, -1, 1};
    for (int mm = 0; mm < 4; ++mm) {
        const float R = RR[mm];
        const float a = w * (gm1 * gm1 / gp1 + sr[mm] * gp1 * r + sR[mm] * gm1 * R);
        const float b = -0.5f * gamma * R * R + 2.0f * w * w -
                        sR[mm] * 0.5f * (gm1 + sr[mm] * gp1 * r) * R;
        const float d = 1.0f + sr[mm] * r + sR[mm] * R;
        accF(0.25f * Lh * d, a * z0 + b * e0p, cc * z1 + d * e1p);
    }
    const float cos_t = z0.dot(z1) / (l0 * l1);
    const float sin_t = lb / (l0 * l1);
    const float beta = (l1 / l0 - l0 / l1) * cos_t;
    const float alpha0 = 0.5f * (-beta + sqrtf(beta * beta + 4.0f));
    const float ginv = w / sin_t;
    accF(ginv * (cos_t / (l1 * l1) - alpha0 / (l0 * l1)), alpha0 * tb, tb);
    accF(ginv * (cos_t / (l1 * l1) + 1.0f / (alpha0 * l0 * l1)), tb,
         (-alpha0) * tb);

    // ---- W-part (rank-4 first term, Eqns 45-57): p-basis + symmetric
    // generalized 4x4 eig, PSD-projected. Gram is block-diagonal because
    // p0 _|_ p1 and {p0,p1} _|_ {p2,p3}.
    const Vec3f tau0 = z0 / l0;
    const Vec3f tau1 = z1 / l1;
    const Vec3f tau1p = tau1.cross(tb);
    const float a0 = t1.dot(y2) / (ly1 * l0);
    const float a1 = t1.dot(y3) / (ly1 * l1);
    const float c0 = 1.0f / l0;
    const float c1 = 1.0f / l1;
    const float k0 = -4.0f * tau0.dot(tau1p) / (ly1 * ly1);
    const float k1 = 2.0f * tau0.dot(tau1p) / ly1;
    const Vec3f s01 = tau0 + tau1;
    const Vec3f d01v = tau0 - tau1;
    float pc[4][12];
    for (int c = 0; c < 3; ++c) {
        pc[0][c] = s01[c];  pc[0][3 + c] = -s01[c];  pc[0][6 + c] = 0.0f;             pc[0][9 + c] = 0.0f;
        pc[1][c] = d01v[c]; pc[1][3 + c] = -d01v[c]; pc[1][6 + c] = 0.0f;             pc[1][9 + c] = 0.0f;
        pc[2][c] = (-a1 - a0 + c1 + c0) * t1[c]; pc[2][3 + c] = (a1 + a0) * t1[c]; pc[2][6 + c] = -c0 * t1[c]; pc[2][9 + c] = -c1 * t1[c];
        pc[3][c] = (-a1 + a0 + c1 - c0) * t1[c]; pc[3][3 + c] = (a1 - a0) * t1[c]; pc[3][6 + c] =  c0 * t1[c]; pc[3][9 + c] = -c1 * t1[c];
    }
    // Gram diagonal / 2x2 block and the W p-basis coefficients (Eqn 49 with the
    // corrected kappa1 factor).
    float gg[4], g23 = 0.0f;
    for (int i = 0; i < 4; ++i) {
        float s = 0.0f;
        for (int m = 0; m < 12; ++m) {
            s += pc[i][m] * pc[i][m];
        }
        gg[i] = s;
    }
    for (int m = 0; m < 12; ++m) {
        g23 += pc[2][m] * pc[3][m];
    }
    // Skip the whole W-part when the p-basis is fp32-degenerate: near-flat /
    // near-fold hinges make gg[0] or gg[1] = 4(1 +/- cos phi) round to 0 (so the
    // 1/sqrt(gg) Cholesky is inf/NaN), and collinear p2,p3 make the 2x2 block
    // singular. The F-part above is already PSD on its own, and the omitted
    // W-term is negligible there (near-flat => theta ~ 0 => g ~ 0 => -gS W ~ 0).
    const float det23 = gg[2] * gg[3] - g23 * g23;
    if (gg[0] > 1e-5f && gg[1] > 1e-5f && det23 > 1e-6f * gg[2] * gg[3]) {
    // E = -g*Sgeom * P^T W P. The explicit W-p3-basis d-coefficients (Eqn 49)
    // blow up ~1/|p0|^2 at near-flat hinges, but P^T W P itself stays finite;
    // using its closed form directly avoids the 0*inf. Symmetric by construction.
    const float we = -g * Sgeom;
    float E[4][4] = {{0.0f}};
    E[0][0] = we * (-k0 * gg[0]);
    E[1][1] = we * (k0 * gg[1]);
    E[0][2] = we * (k1 * gg[2]);
    E[2][0] = E[0][2];
    E[0][3] = we * (k1 * g23);
    E[3][0] = E[0][3];
    E[1][2] = we * (k1 * g23);
    E[2][1] = E[1][2];
    E[1][3] = we * (k1 * gg[3]);
    E[3][1] = E[1][3];
    // Block Cholesky of the block-diagonal Gram -> Li (lower inverse).
    float Li[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            Li[i][j] = 0.0f;
    Li[0][0] = 1.0f / sqrtf(gg[0]);
    Li[1][1] = 1.0f / sqrtf(gg[1]);
    const float L22 = sqrtf(gg[2]);
    const float L32 = g23 / L22;
    const float L33 = sqrtf(fmaxf(1e-30f, gg[3] - L32 * L32));
    Li[2][2] = 1.0f / L22;
    Li[3][3] = 1.0f / L33;
    Li[3][2] = -L32 / (L22 * L33);
    // Ssym = Li E Li^T (4x4), PSD-project, then Core = Li^T Ssym Li.
    float LiE[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k)
                s += Li[i][k] * E[k][j];
            LiE[i][j] = s;
        }
    SMat<float, 4, 4> Ss;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k)
                s += LiE[i][k] * Li[j][k];
            Ss(i, j) = s;
        }
    linalg::psd_project_symmetric<4>(Ss, 0.0f);
    float LtS[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k)
                s += Li[k][i] * Ss(k, j);
            LtS[i][j] = s;
        }
    float Core[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) {
            float s = 0.0f;
            for (int k = 0; k < 4; ++k)
                s += LtS[i][k] * Li[k][j];
            Core[i][j] = s;
        }
    // T_spd = P Core P^T; accumulate into Hp. Precompute PC = P Core (12x4).
    float PC[12][4];
    for (int m = 0; m < 12; ++m)
        for (int j = 0; j < 4; ++j) {
            float s = 0.0f;
            for (int i = 0; i < 4; ++i)
                s += pc[i][m] * Core[i][j];
            PC[m][j] = s;
        }
    for (int m = 0; m < 12; ++m)
        for (int n = 0; n < 12; ++n) {
            float s = 0.0f;
            for (int j = 0; j < 4; ++j)
                s += PC[m][j] * pc[j][n];
            Hp(m, n) += s;
        }
    } // end W-part (fp32-degeneracy guard)

    // ---- Permute P-order -> codebase (X0,X1,X2,X3) order: swap blocks 0<->2.
    const int perm[4] = {2, 1, 0, 3};
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    hess(3 * perm[a] + i, 3 * perm[b] + j) = Hp(3 * a + i, 3 * b + j);
}

__device__ static float face_energy(const Vec3f &v0, const Vec3f &v1,
                                    const Vec3f &v2, const Vec3f &v3,
                                    float rest_angle) {
    float angle = face_dihedral_angle(v0, v1, v2, v3);
    float diff = angle - rest_angle;
    return 0.5f * diff * diff;
}

__device__ static float strand_energy(const Vec3f &x0, const Vec3f &x1,
                                      const Vec3f &x2, float rest_angle) {
    Vec3f e0 = (x0 - x1);
    Vec3f e1 = (x2 - x1);
    // atan2 form of the bending angle between the two rod segments (same
    // rationale as face_dihedral_angle): acosf(dot) is ill-conditioned near
    // straight segments (dot -> +-1), whereas atan2(|e0 x e1|, e0 . e1) is
    // O(1)-conditioned and gives the identical value on [0, pi].
    float theta = atan2f(e0.cross(e1).norm(), e0.dot(e1));
    float diff = theta - rest_angle;
    return 0.5f * diff * diff;
}

__device__ static Mat3x2f gradient_theta(const Vec3f &e0, const Vec3f &e1) {
    // Guard the degenerate / near-degenerate case e0 parallel or
    // antiparallel to e1 (a perfectly straight or back-folded rod
    // segment). |e0 x e1|^2 = |e0|^2 |e1|^2 sin^2(theta), so we reject
    // when sin^2(theta) is below a scale-invariant threshold. Without
    // this, exact collinearity produces a zero-vector cross whose
    // ``.normalized()`` returns NaN, which propagates through both the
    // bending force and the g*g^T Hessian and breaks PCG monotonicity
    // (alpha flips sign and the residual diverges). The bending force
    // is (theta - theta_0) * grad(theta) which vanishes at the
    // singularity anyway, so contributing a zero gradient (and thus a
    // zero Hessian block) is also the right physics for that vertex.
    Vec3f cross = e0.cross(e1);
    float cross_sqnm = cross.squaredNorm();
    float e0_sqnm = e0.dot(e0);
    float e1_sqnm = e1.dot(e1);
    constexpr float kSinSqEps = 1e-12f;
    if (cross_sqnm <= kSinSqEps * e0_sqnm * e1_sqnm) {
        return Mat3x2f::Zero();
    }
    Vec3f n = cross / sqrtf(cross_sqnm);
    Vec3f e0perp = e0.cross(n);
    Vec3f e1perp = e1.cross(n);
    Vec3f g0 = e0perp / e0_sqnm;
    Vec3f g1 = -e1perp / e1_sqnm;
    Mat3x2f G;
    G << g0, g1;
    return G;
}

__device__ static Mat3x3f strand_gradient(const Vec3f &x0, const Vec3f &x1,
                                          const Vec3f &x2, float rest_angle) {
    Vec3f e0 = (x0 - x1);
    Vec3f e1 = (x2 - x1);
    float theta = atan2f(e0.cross(e1).norm(), e0.dot(e1));
    Mat3x2f Pfinal = (theta - rest_angle) * gradient_theta(e0, e1);
    Mat3x3f PK1;
    PK1 << Pfinal.col(0), -Pfinal.col(0) - Pfinal.col(1), Pfinal.col(1);
    return PK1;
}

// Force and analytically PSD-projected Hessian for the rod-bend energy
// E = 1/2 (theta - theta0)^2.  force = (theta - theta0) * grad(theta) (the exact
// gradient, unchanged). The exact Hessian d2E/dx2 = grad(theta)(x)grad(theta) +
// (theta-theta0) d2theta/dx2 is in general INDEFINITE, which breaks the SPD PCG
// solve when the inertia term is too weak to mask it. We instead assemble the
// projected-Newton Hessian: the nearest PSD matrix that keeps all positive-
// curvature eigenmodes and drops the indefinite ones, via the closed-form
// angle-energy eigensystem of Wu & Kim 2023 ("An Eigenanalysis of Angle-Based
// Deformation Energies", PACMCGIT 6(3)). For a rod, d2F/dx2 = 0 (F = [e0 e1]),
// so projecting the 6-D F-space Hessian is sufficient; the 6 eigenpairs are
// mapped to the 9 DOFs by the constant Jacobian J = d[e0;e1]/d[x0;x1;x2] =
// [[I,-I,0],[0,-I,I]]. Verified against finite differences over 3000 random
// configs (the paper's u-vector b-component R-sign is a typo, corrected below).
__device__ static void strand_compute_force_hessian(
    const Vec3f &x0, const Vec3f &x1, const Vec3f &x2, float rest_angle,
    Mat3x3f &force, Mat9x9f &hess) {
    Vec3f e0 = (x0 - x1);
    Vec3f e1 = (x2 - x1);
    float theta = atan2f(e0.cross(e1).norm(), e0.dot(e1));
    Mat3x2f gtheta = gradient_theta(e0, e1);
    Mat3x3f angle_grad;
    angle_grad << gtheta.col(0), -gtheta.col(0) - gtheta.col(1), gtheta.col(1);
    force = (theta - rest_angle) * angle_grad;
    hess = Mat9x9f::Zero();

    const float l0 = e0.norm();
    const float l1 = e1.norm();
    Vec3f bcr = e0.cross(e1);
    const float lb = bcr.norm();
    // Straight / near-collinear rod: grad(theta) already returned zero above, so
    // the force vanishes and the bending Hessian is zero. Guard the divisions.
    if (lb <= 1e-9f * l0 * l1 || l0 <= 1e-9f || l1 <= 1e-9f) {
        return;
    }
    const Vec3f tb = bcr / lb;
    const Vec3f e0p = e0.cross(tb); // in-plane orthogonal edge, |e0p| = l0
    const Vec3f e1p = e1.cross(tb);
    const float cos_t = e0.dot(e1) / (l0 * l1);
    const float sin_t = lb / (l0 * l1);
    // Energy 1/2 (theta-theta0)^2 with stiffness mu = 1 (embed_rod_bend scales
    // the result by stiff_k = bend*mass afterwards): g = theta-theta0, h = 1,
    // so w = S g / h = theta - theta0 with the sign convention S = 1.
    const float w = theta - rest_angle;
    const float gamma = (l1 * l1) / (l0 * l0);
    const float gm1 = gamma - 1.0f;
    const float gp1 = gamma + 1.0f;
    const float r = sqrtf(4.0f * w * w * (gm1 / gp1) * (gm1 / gp1) + 1.0f);
    const float Rm = sqrtf(fmaxf(0.0f, 2.0f * (2.0f * w * w + 1.0f - r)));
    const float Rp = sqrtf(fmaxf(0.0f, 2.0f * (2.0f * w * w + 1.0f + r)));
    const float Lh = 1.0f / (l0 * l0) + 1.0f / (l1 * l1); // h = 1
    const float cc = 4.0f * w / gp1;

    // Accumulate lam * (J^T qhat)(J^T qhat)^T for a kept (lam > 0) eigenpair,
    // where the 6-D unit eigenvector is [qa; qb] and J^T maps it to the 9 DOFs
    // as [qa; -qa-qb; qb]. Normalization (/n2) folds |[qa;qb]|^2 in.
    auto accum = [&](float lam, const Vec3f &qa, const Vec3f &qb) {
        if (lam <= 0.0f) {
            return;
        }
        const float n2 = qa.dot(qa) + qb.dot(qb);
        if (n2 <= 1e-20f) {
            return;
        }
        float p9[9];
        for (int c = 0; c < 3; ++c) {
            p9[c] = qa[c];
            p9[3 + c] = -qa[c] - qb[c];
            p9[6 + c] = qb[c];
        }
        const float s = lam / n2;
        for (int i = 0; i < 9; ++i) {
            for (int jj = 0; jj < 9; ++jj) {
                hess(i, jj) += s * p9[i] * p9[jj];
            }
        }
    };

    // Four in-plane eigenpairs (Wu & Kim Eqns 16-24). The b (2nd) component's
    // R-term sign is FLIPPED relative to the paper (Eqns 16-19 typo; the printed
    // +sign_R must be -sign_R), verified numerically.
    const int sr[4] = {-1, -1, 1, 1};
    const float RR[4] = {Rm, Rm, Rp, Rp};
    const int sR[4] = {-1, 1, -1, 1};
    for (int m = 0; m < 4; ++m) {
        const float R = RR[m];
        const float a = w * (gm1 * gm1 / gp1 + sr[m] * gp1 * r + sR[m] * gm1 * R);
        const float b = -0.5f * gamma * R * R + 2.0f * w * w -
                        sR[m] * 0.5f * (gm1 + sr[m] * gp1 * r) * R;
        const float d = 1.0f + sr[m] * r + sR[m] * R;
        const float lam = 0.25f * Lh * d; // Lh (1 + sr r + sR R) / 4
        Vec3f qa = a * e0 + b * e0p;
        Vec3f qb = cc * e1 + d * e1p;
        accum(lam, qa, qb);
    }

    // Two out-of-plane eigenpairs (Wu & Kim Eqns 30-32).
    const float beta = (l1 / l0 - l0 / l1) * cos_t;
    const float alpha0 = 0.5f * (-beta + sqrtf(beta * beta + 4.0f));
    const float ginv = w / sin_t; // g / sin(theta), g = w
    const float lam4 = ginv * (cos_t / (l1 * l1) - alpha0 / (l0 * l1));
    const float lam5 = ginv * (cos_t / (l1 * l1) + 1.0f / (alpha0 * l0 * l1));
    accum(lam4, alpha0 * tb, tb);
    accum(lam5, tb, (-alpha0) * tb);
}

} // namespace dihedral_angle

#endif
