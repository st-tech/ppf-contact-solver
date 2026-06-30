// File: energy.cu
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#include "../eigenanalysis/eigenanalysis.hpp"
#include "../utility/dispatcher.hpp"
#include "../utility/utility.hpp"
#include "model/pdrd_rigid.hpp"
#include "model/air_damper.hpp"
#include "model/arap.hpp"
#include "model/inflate.hpp"
#include "model/baraffwitkin.hpp"
#include "model/dihedral_angle.hpp"
#include "model/hook.hpp"
#include "model/momentum.hpp"
#include "model/snhk.hpp"
#include "model/stvk.hpp"

// 3x3 symmetric eigendecomposition for PCA on device.
// Returns eigenvectors sorted by descending eigenvalue.
__device__ void eigen_sym3x3(
    float a00, float a01, float a02,
    float a11, float a12, float a22,
    Vec3f evals, Mat3x3f &evecs) {

    // Eigenvalues via Cardano's formula for symmetric 3x3
    float q = (a00 + a11 + a22) / 3.0f;
    float p2 = (a00 - q) * (a00 - q) + (a11 - q) * (a11 - q) +
               (a22 - q) * (a22 - q) + 2.0f * (a01 * a01 + a02 * a02 + a12 * a12);
    float p = sqrtf(fmaxf(p2 / 6.0f, 0.0f));

    float lam0, lam1, lam2;
    if (p < 1e-12f) {
        // Matrix is (near) scalar multiple of identity
        lam0 = a00; lam1 = a11; lam2 = a22;
    } else {
        float inv_p = 1.0f / p;
        // B = (A - qI) / p
        float b00 = (a00 - q) * inv_p, b01 = a01 * inv_p, b02 = a02 * inv_p;
        float b11 = (a11 - q) * inv_p, b12 = a12 * inv_p;
        float b22 = (a22 - q) * inv_p;
        float det_b = b00 * (b11 * b22 - b12 * b12)
                    - b01 * (b01 * b22 - b12 * b02)
                    + b02 * (b01 * b12 - b11 * b02);
        float r = fminf(1.0f, fmaxf(-1.0f, det_b * 0.5f));
        float phi = acosf(r) / 3.0f;
        lam0 = q + 2.0f * p * cosf(phi);
        lam2 = q + 2.0f * p * cosf(phi + 2.0943951f); // 2π/3
        lam1 = 3.0f * q - lam0 - lam2;
    }

    // Sort descending
    if (lam1 > lam0) { float t = lam0; lam0 = lam1; lam1 = t; }
    if (lam2 > lam0) { float t = lam0; lam0 = lam2; lam2 = t; }
    if (lam2 > lam1) { float t = lam1; lam1 = lam2; lam2 = t; }
    evals = Vec3f(lam0, lam1, lam2);

    // Eigenvectors via cross product of rows of (A - λI)
    float vals[3] = {lam0, lam1, lam2};
    for (int c = 0; c < 3; ++c) {
        float l = vals[c];
        Vec3f r0(a00 - l, a01, a02);
        Vec3f r1(a01, a11 - l, a12);
        Vec3f r2(a02, a12, a22 - l);
        Vec3f v = r0.cross(r1);
        float vn = v.norm();
        if (vn < 1e-8f) { v = r0.cross(r2); vn = v.norm(); }
        if (vn < 1e-8f) { v = r1.cross(r2); vn = v.norm(); }
        if (vn > 1e-8f) {
            evecs.col(c) = v / vn;
        } else {
            evecs.col(c) = Vec3f::Unit(c);
        }
    }

    // Gram-Schmidt to ensure orthogonality
    Vec3f e0 = evecs.col(0).normalized();
    Vec3f e1 = evecs.col(1) - e0 * e0.dot(evecs.col(1));
    e1.normalize();
    Vec3f e2 = e0.cross(e1);
    evecs.col(0) = e0;
    evecs.col(1) = e1;
    evecs.col(2) = e2;
}

namespace energy {

// Stiffness-proportional Rayleigh damping. Reuses the already-assembled,
// SPD-projected per-element elastic Hessian K (= d2edx2) as the damping
// operator: it adds the lagged damping gradient (beta/dt) * K * (x - x^n) to
// the element gradient and the constant SPD block (beta/dt) * K to the element
// Hessian, with x^n = data.vertex.curr (start of step). The d(K v)/dx term is
// dropped (semi-implicit), so the contribution stays symmetric and PSD and the
// Newton/PCG solve is unaffected. Call AFTER the elastic dedx/d2edx2 are fully
// built and BEFORE the atomic scatter. No-op when beta <= 0.
template <unsigned N, class IdxVec, class DedxMat, class HessMat>
__device__ void
add_stiffness_damping(const DataSet &data, const Vec<Vec3f> &eval_x,
                      const IdxVec &elem, float beta, float dt, DedxMat &dedx,
                      HessMat &d2edx2) {
    if (beta <= 0.0f || dt <= 0.0f) {
        return;
    }
    const float s = beta / dt;
    // Per-node displacement during the step, u_k = x_k - x_k^n, with
    // x^n = data.vertex.curr (start of step). Flattened to 3N.
    SVecf<3 * N> u;
    for (unsigned k = 0; k < N; ++k) {
        unsigned idx = elem[k];
        Vec3f d = (eval_x[idx] - data.vertex.curr[idx]);
        u[3 * k + 0] = d[0];
        u[3 * k + 1] = d[1];
        u[3 * k + 2] = d[2];
    }
    // Damping gradient (beta/dt) * K * u added to the element gradient, with an
    // explicit matvec over the still-elastic K (raw element access avoids the
    // device Eigen matvec pitfalls; K is reused as the SPD damping operator).
    for (unsigned a = 0; a < N; ++a) {
        float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f;
        for (unsigned b = 0; b < 3 * N; ++b) {
            float ub = u[b];
            f0 += d2edx2(3 * a + 0, b) * ub;
            f1 += d2edx2(3 * a + 1, b) * ub;
            f2 += d2edx2(3 * a + 2, b) * ub;
        }
        dedx(0, a) += s * f0;
        dedx(1, a) += s * f1;
        dedx(2, a) += s * f2;
    }
    // Hessian += (beta/dt) * K, i.e. elastic K plus the damping block.
    d2edx2 *= (1.0f + s);
}

// Lagged variant for rapidly-varying (bending) Hessians. The damping operator
// K_lag is evaluated at the START-OF-STEP positions (data.vertex.curr), so the
// damping force (beta/dt) K_lag (x - x^n) is exactly the gradient of the convex
// potential (beta/2dt)(x - x^n)^T K_lag (x - x^n) and is therefore guaranteed
// dissipative. The current-iterate form above drops the d(K v)/dx term, which
// is negligible for the smoothly-varying membrane/solid Hessian but large for
// the dihedral bending Hessian (g g^T with a fast-changing angle gradient g),
// where it can otherwise inject energy. Adds s*K_lag to both the element
// gradient (contracted with x - x^n) and the element Hessian.
template <unsigned N, class IdxVec, class DedxMat, class HessMat>
__device__ void
add_stiffness_damping_lagged(const DataSet &data, const Vec<Vec3f> &eval_x,
                             const IdxVec &elem, float beta, float dt,
                             DedxMat &dedx, HessMat &d2edx2,
                             const HessMat &K_lag) {
    if (beta <= 0.0f || dt <= 0.0f) {
        return;
    }
    const float s = beta / dt;
    SVecf<3 * N> u;
    for (unsigned k = 0; k < N; ++k) {
        unsigned idx = elem[k];
        Vec3f d = (eval_x[idx] - data.vertex.curr[idx]);
        u[3 * k + 0] = d[0];
        u[3 * k + 1] = d[1];
        u[3 * k + 2] = d[2];
    }
    for (unsigned a = 0; a < N; ++a) {
        float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f;
        for (unsigned b = 0; b < 3 * N; ++b) {
            float ub = u[b];
            f0 += K_lag(3 * a + 0, b) * ub;
            f1 += K_lag(3 * a + 1, b) * ub;
            f2 += K_lag(3 * a + 2, b) * ub;
        }
        dedx(0, a) += s * f0;
        dedx(1, a) += s * f1;
        dedx(2, a) += s * f2;
    }
    for (unsigned r = 0; r < 3 * N; ++r) {
        for (unsigned c = 0; c < 3 * N; ++c) {
            d2edx2(r, c) += s * K_lag(r, c);
        }
    }
}

__device__ void embed_vertex_force_hessian(
    const DataSet &data, const Vec<Vec3f> &eval_x, const Vec<Vec3f> &velocity,
    const Vec<Vec3f> &target, Vec<float> &force, Vec<Mat3x3f> &diag_hess,
    float dt, const ParamSet &param,
    const Vec<TorqueGroupResult> &torque_result, unsigned i) {

    float mass = data.prop.vertex[i].mass;
    float area = data.prop.vertex[i].area;

    const Vec3f &x = data.vertex.curr[i];
    const Vec3f &y = eval_x[i];
    const Vec3f normal = utility::compute_vertex_normal(data, eval_x, i);

    Vec3f wind = Vec3f::Zero();
    if (!param.inactive_momentum) {
        wind = utility::get_wind_weight(param.time) * param.wind;
    }

    Vec3f f = Vec3f::Zero();
    Mat3x3f H = Mat3x3f::Zero();
    if (normal.isZero() == false && param.air_density) {
        f += area * param.air_density *
             air_damper::face_gradient(dt, y, x, normal, wind, param);
        H += area * param.air_density *
             air_damper::face_hessian(dt, normal, param);
    }

    bool pulled(false);
    for (unsigned j = 0; j < data.constraint.pull.size; ++j) {
        if (i == data.constraint.pull[j].index) {
            Vec3f position = data.constraint.pull[j].position;
            float weight = data.constraint.pull[j].weight;
            f += weight * (y - position);
            H += weight * Mat3x3f::Identity();
            pulled = true;
            break;
        }
    }

    for (unsigned j = 0; j < data.constraint.torque_vertices.size; ++j) {
        if (i == data.constraint.torque_vertices[j].index) {
            float mag = data.constraint.torque_vertices[j].magnitude;
            unsigned gid = data.constraint.torque_vertices[j].group_id;

            // Read pre-computed centroid and PCA axis from pre-pass
            Vec3f center = torque_result[gid].center;
            Vec3f axis = torque_result[gid].axis;

            Vec3f r = (y - center);
            Vec3f r_perp = r - axis * axis.dot(r);
            // F_i = axis × r_perp_i * magnitude / Σ|r_perp_j|²
            // ensures total torque = magnitude and zero net linear force
            float scale = mag * torque_result[gid].inv_r_perp_sq_sum;
            Vec3f torque_force = axis.cross(r_perp) * scale;
            f -= torque_force;
            // Symmetric Hessian approximation so Newton converges for
            // the torque term regardless of iteration count.
            // ∂(axis×r_perp)/∂y has a skew part (unusable) and a
            // symmetric magnitude part scale*(I - axis⊗axis).
            Mat3x3f P = Mat3x3f::Identity() - axis * axis.transpose();
            H += scale * P;
        }
    }

    if (!pulled) {
        f += mass * momentum::gradient(dt, y, target[i]);
        H += mass * momentum::hessian(dt);
    }

    if (param.isotropic_air_friction) {
        f += param.isotropic_air_friction * (y - x) / (dt * dt);
        H += (param.isotropic_air_friction / (dt * dt)) * Mat3x3f::Identity();
    }

    if (param.fix_xz && y[1] > float(param.fix_xz)) {
        float t = fmin(1.0f, y[1] - float(param.fix_xz));
        Vec3f n(0.0f, 1.0f, 0.0f);
        Mat3x3f P = Mat3x3f::Identity() - n * n.transpose();
        f += P * t * mass * (y - x) / (dt * dt);
        H += P * t * mass * (dt * dt);
    }

    Map<Vec3f>(force.data + 3 * i) += f;
    diag_hess[i] += H;
}

__device__ void embed_rod_force_hessian(const DataSet &data,
                                        const Vec<Vec3f> &eval_x,
                                        Vec<float> &force, FixedCSRMat &hess,
                                        float dt, const ParamSet &param,
                                        unsigned i) {
    const Vec2u &edge = data.mesh.mesh.edge[i];
    const Vec3f &x0 = eval_x[edge[0]];
    const Vec3f &x1 = eval_x[edge[1]];

    const EdgeProp &prop = data.prop.edge[i];
    const EdgeParam &edge_param = data.param_arrays.edge[prop.param_index];
    float l0 = prop.length;
    Vec3f t = x1 - x0;
    float l = t.norm();
    float mass = prop.mass;
    float stiffness = edge_param.stiffness;
    if (stiffness > 0.0f) {
        Mat3x2f dedx;
        Mat6x6f d2edx2;
        hook::make_diff_table(x0, x1, l0, stiffness * mass, dedx, d2edx2);
        add_stiffness_damping<2>(data, eval_x, edge, edge_param.deform_damping,
                                 dt, dedx, d2edx2);
        utility::atomic_embed_force<2>(edge, dedx, force);
        utility::atomic_embed_hessian<2>(edge, d2edx2, hess);
    }
}

__device__ void embed_face_force_hessian(const DataSet &data,
                                         const Vec<Vec3f> &eval_x,
                                         Vec<float> &force, FixedCSRMat &hess,
                                         float dt, const ParamSet &param,
                                         unsigned i) {
    const Vec3u &face = data.mesh.mesh.face[i];
    const FaceProp &prop = data.prop.face[i];
    const FaceParam &face_param = data.param_arrays.face[prop.param_index];
    const Vec3f &x0 = eval_x[face[0]];
    const Vec3f &x1 = eval_x[face[1]];
    const Vec3f &x2 = eval_x[face[2]];
    Mat3x3f dedx = Mat3x3f::Zero();
    Mat9x9f d2edx2 = Mat9x9f::Zero();
    float mass = prop.mass;
    float mu = face_param.mu;
    if (mu > 0.0f) {
        Mat3x2f F;
        Mat3x3f X;
        X << x0, x1, x2;
        F = utility::compute_deformation_grad(X, data.inv_rest2x2[i]);
        const Svd3x2 svd = utility::svd3x2(F);
        if (face_param.model == Model::BaraffWitkin) {
            Mat3x2f de0dF = BaraffWitkin::stretch_gradient(F, mu);
            Mat3x2f de1dF = BaraffWitkin::shear_gradient(F, face_param.lambda);
            Mat6x6f d2e0dF2 = BaraffWitkin::stretch_hessian(F, mu);
            Mat6x6f d2e1dF2 = BaraffWitkin::shear_hessian(F, face_param.lambda);
            Mat3x2f dedF = de0dF + de1dF;
            Mat6x6f d2edF2 = d2e0dF2 + d2e1dF2;
            dedx += mass * utility::convert_force(dedF, data.inv_rest2x2[i]);
            d2edx2 +=
                mass * utility::convert_hessian(d2edF2, data.inv_rest2x2[i]);
        } else {
            DiffTable2 table;
            Mat3x2f dedF;
            Mat6x6f d2edF2;
            if (face_param.model == Model::ARAP) {
                table = ARAP::make_diff_table2(svd.S, mu, face_param.lambda);
            } else if (face_param.model == Model::StVK) {
                table = StVK::make_diff_table2(svd.S, mu, face_param.lambda);
            } else if (face_param.model == Model::SNHk) {
                table = SNHk::make_diff_table2(svd.S, mu, face_param.lambda);
            } else {
                assert(false);
            }
            dedF = eigenanalysis::compute_force(table, svd);
            d2edF2 = eigenanalysis::compute_hessian(table, svd,
                                                    param.eiganalysis_eps);
            dedx += mass * utility::convert_force(dedF, data.inv_rest2x2[i]);
            d2edx2 +=
                mass * utility::convert_hessian(d2edF2, data.inv_rest2x2[i]);
        }
        add_stiffness_damping<3>(data, eval_x, face, face_param.deform_damping,
                                 dt, dedx, d2edx2);
        utility::atomic_embed_force<3>(face, dedx, force);
        utility::atomic_embed_hessian<3>(face, d2edx2, hess);
    }

    if (face_param.pressure > 0.0f) {
        Vec3f v0 = x0;
        Vec3f v1 = x1;
        Vec3f v2 = x2;
        utility::atomic_embed_force<3>(
            face, inflate::face_gradient(face_param.pressure, v0, v1, v2), force);
        utility::atomic_embed_hessian<3>(
            face, inflate::face_hessian(face_param.pressure, v0, v1, v2), hess);
    }
}

__device__ void embed_tet_force_hessian(const DataSet &data,
                                        const Vec<Vec3f> &eval_x,
                                        Vec<float> &force, FixedCSRMat &hess,
                                        float dt, const ParamSet &param,
                                        unsigned i) {
    const Vec4u &tet = data.mesh.mesh.tet[i];
    const TetProp &prop = data.prop.tet[i];
    const TetParam &tet_param = data.param_arrays.tet[prop.param_index];
    const float mass = prop.mass;
    const float mu = tet_param.mu;
    if (mu > 0.0f) {
        const float lambda = tet_param.lambda;
        const Vec3f &x0 = eval_x[tet[0]];
        const Vec3f &x1 = eval_x[tet[1]];
        const Vec3f &x2 = eval_x[tet[2]];
        const Vec3f &x3 = eval_x[tet[3]];
        Mat3x4f X;
        X << x0, x1, x2, x3;
        Mat3x3f F = utility::compute_deformation_grad(X, data.inv_rest3x3[i]);
        Svd3x3 svd = utility::svd3x3_rv(F);
        DiffTable3 table;
        Mat3x3f dedF;
        Mat9x9f d2edF2;
        Mat3x4f dedx = Mat3x4f::Zero();
        Mat12x12f d2edx2 = Mat12x12f::Zero();
        if (tet_param.model == Model::ARAP) {
            table = ARAP::make_diff_table3(svd.S, mu, lambda);
        } else if (tet_param.model == Model::StVK) {
            table = StVK::make_diff_table3(svd.S, mu, lambda);
        } else if (tet_param.model == Model::SNHk) {
            table = SNHk::make_diff_table3(svd.S, mu, lambda);
        } else {
            assert(false);
        }
        dedF = eigenanalysis::compute_force(table, svd);
        d2edF2 =
            eigenanalysis::compute_hessian(table, svd, param.eiganalysis_eps);
        dedx += mass * utility::convert_force(dedF, data.inv_rest3x3[i]);
        d2edx2 += mass * utility::convert_hessian(d2edF2, data.inv_rest3x3[i]);
        add_stiffness_damping<4>(data, eval_x, tet, tet_param.deform_damping, dt,
                                 dedx, d2edx2);
        utility::atomic_embed_force<4>(tet, dedx, force);
        utility::atomic_embed_hessian<4>(tet, d2edx2, hess);
    }
}

__device__ void embed_hinge_force_hessian(const DataSet &data,
                                          const Vec<Vec3f> &eval_x,
                                          Vec<float> &force, FixedCSRMat &hess,
                                          float dt, const ParamSet &param,
                                          unsigned i) {
    const HingeProp &prop = data.prop.hinge[i];
    const HingeParam &hinge_param = data.param_arrays.hinge[prop.param_index];
    float length = prop.length;
    float area = prop.area;  // combined rest area of the two incident triangles
    float bend = hinge_param.bend;
    Vec4u hinge = data.mesh.mesh.hinge[i];
    // Density-normalize the shell bending stiffness by the local areal density
    // (kg/m^2), so the bent shape is invariant to density. This matches the rest
    // of the solver: the membrane and the rod bend already scale by mass, so
    // shell bending was the lone elastic term whose shape depended on density
    // (drape scaled with bend/density). With this factor `bend` alone sets the
    // bent shape and density becomes a free knob -- which lets a very light
    // fabric (e.g. silk) be mixed with dense bodies without conditioning
    // trouble. Areal density (mass/area), not raw vertex mass, keeps the bend
    // mesh-independent.
    float areal_density = 0.0f;
    {
        int cnt = 0;
        for (int k = 0; k < 4; ++k) {
            float a = data.prop.vertex[hinge[k]].area;
            if (a > 0.0f) {
                areal_density += data.prop.vertex[hinge[k]].mass / a;
                ++cnt;
            }
        }
        if (cnt > 0) {
            areal_density /= cnt;
        }
    }
    // Resolution-independent Discrete Shells bending coefficient. The convergent
    // per-hinge stiffness is k = B * |e|/h_e proportional to |e|^2 / (A1 + A2),
    // which is scale-invariant under mesh refinement (|e|^2 / area stays O(1)), so
    // the bent shape no longer depends on resolution. The old form used a bare |e|
    // factor, which shrank as |e| -> |e|/s under refinement, so finer cloth drooped
    // more. (Convergence: the integrated mean curvature on an edge is |e|*theta and
    // the edge dual area is (A1+A2)/3, so the sum converges to int B*kappa^2 dA;
    // Grinspun et al. 2003, Tamstorf-Grinspun 2013, Wang 2023.) B = bend*areal_density
    // is the density-normalized flexural rigidity (density-invariant shape).
    //
    // BEND_SCALE only sets the numeric range of the user `bend` parameter; it does
    // NOT affect resolution-independence (that is the |e|^2/area factor). It is
    // calibrated (calibration/cusick_drape) so the established fabric bend values,
    // and existing scenes, keep their look at the usual mesh density after the
    // switch from the old resolution-dependent |e|*ghat factor (which it replaces:
    // |e|*ghat was ~3e5x weaker than |e|^2/area at that mesh). Guard near-degenerate
    // triangles where area -> 0 would blow the stiffness up.
    const float BEND_SCALE = 1.28e-5f;
    float stiff_k = (area > 1e-12f)
                        ? BEND_SCALE * bend * (length * length / area) * areal_density
                        : 0.0f;
    if (stiff_k > 0.0f) {
        Mat3x4f dedx;
        Mat12x12f d2edx2;
        dihedral_angle::face_compute_force_hessian(eval_x, hinge,
                                                   prop.rest_angle, dedx,
                                                   d2edx2);
        // Scale to the true bending stiffness first, then damp with that K.
        dedx *= stiff_k;
        d2edx2 *= stiff_k;
        // Lagged bending damping: build the damping Hessian at the start-of-step
        // positions (guaranteed dissipative). `hinge` is remapped in place by
        // face_compute_force_hessian, so pass a fresh copy for the lagged eval;
        // both remap identically, so the node order matches.
        if (hinge_param.bend_damping > 0.0f) {
            Vec4u hinge_c = data.mesh.mesh.hinge[i];
            Mat3x4f f_lag;
            Mat12x12f K_lag;
            dihedral_angle::face_compute_force_hessian(
                data.vertex.curr, hinge_c, prop.rest_angle, f_lag, K_lag);
            K_lag *= stiff_k;
            add_stiffness_damping_lagged<4>(data, eval_x, hinge,
                                            hinge_param.bend_damping, dt, dedx,
                                            d2edx2, K_lag);
        }
        utility::atomic_embed_force<4>(hinge, dedx, force);
        utility::atomic_embed_hessian<4>(hinge, d2edx2, hess);
    }
}

__device__ void
embed_rod_bend_force_hessian(const DataSet &data, const Vec<Vec3f> &eval_x,
                             Vec<float> &force, FixedCSRMat &hess, float dt,
                             const ParamSet &param, unsigned i) {
    if (data.mesh.neighbor.vertex.edge.count(i) == 2 &&
        data.mesh.neighbor.vertex.face.count(i) == 0) {
        unsigned edge_idx_0 = data.mesh.neighbor.vertex.edge(i, 0);
        unsigned edge_idx_1 = data.mesh.neighbor.vertex.edge(i, 1);
        const EdgeParam &edge_param_0 = data.param_arrays.edge[data.prop.edge[edge_idx_0].param_index];
        const EdgeParam &edge_param_1 = data.param_arrays.edge[data.prop.edge[edge_idx_1].param_index];
        float bend_0 = edge_param_0.bend;
        float bend_1 = edge_param_1.bend;
        float bend = 0.5f * (bend_0 + bend_1);
        float bend_damping =
            0.5f * (edge_param_0.bend_damping + edge_param_1.bend_damping);
        float mass = data.prop.vertex[i].mass;
        float stiff_k = bend * mass;
        if (mass > 0.0f && stiff_k > 0.0f) {
            Vec2u edge_0 = data.mesh.mesh.edge[edge_idx_0];
            Vec2u edge_1 = data.mesh.mesh.edge[edge_idx_1];
            unsigned j = edge_0[0] == i ? edge_0[1] : edge_0[0];
            unsigned k = edge_1[0] == i ? edge_1[1] : edge_1[0];
            Vec3u element = Vec3u(j, i, k);
            Vec3f x0 = eval_x[j];
            Vec3f x1 = eval_x[i];
            Vec3f x2 = eval_x[k];
            float rest_angle = data.prop.vertex[i].rest_bend_angle;
            Mat3x3f dedx;
            Mat9x9f d2edx2;
            dihedral_angle::strand_compute_force_hessian(
                x0, x1, x2, rest_angle, dedx, d2edx2);
            // Scale to the true bending stiffness first, then damp with that K.
            dedx *= stiff_k;
            d2edx2 *= stiff_k;
            // Lagged bending damping: damping Hessian at start-of-step positions
            // (guaranteed dissipative; the current-iterate form injects energy
            // for the fast-varying dihedral Hessian).
            if (bend_damping > 0.0f) {
                Vec3f c0 = data.vertex.curr[j];
                Vec3f c1 = data.vertex.curr[i];
                Vec3f c2 = data.vertex.curr[k];
                Mat3x3f f_lag;
                Mat9x9f K_lag;
                dihedral_angle::strand_compute_force_hessian(c0, c1, c2,
                                                             rest_angle, f_lag,
                                                             K_lag);
                K_lag *= stiff_k;
                add_stiffness_damping_lagged<3>(data, eval_x, element,
                                                bend_damping, dt, dedx, d2edx2,
                                                K_lag);
            }
            utility::atomic_embed_force<3>(element, dedx, force);
            utility::atomic_embed_hessian<3>(element, d2edx2, hess);
        }
    }
}

void compute_torque_groups(const DataSet &data,
                           const Vec<Vec3f> &eval_x,
                           Vec<TorqueGroupResult> &result) {
    DISPATCH_START(data.constraint.torque_groups.size)
    [data, eval_x, result] __device__(unsigned g) mutable {
        const auto &grp = data.constraint.torque_groups[g];
        unsigned vs = grp.vertex_start;
        unsigned vn = grp.vertex_count;

        // Mass-weighted centroid (center of gravity)
        Vec3f center = Vec3f::Zero();
        float total_mass = 0.0f;
        for (unsigned k = vs; k < vs + vn; ++k) {
            unsigned idx = data.constraint.torque_vertices[k].index;
            float m = data.prop.vertex[idx].mass;
            center += eval_x[idx] * float(m);
            total_mass += m;
        }
        if (total_mass > 1e-12f) {
            center *= float(1.0f / total_mass);
        }

        // 3x3 covariance matrix
        float c00 = 0, c01 = 0, c02 = 0, c11 = 0, c12 = 0, c22 = 0;
        for (unsigned k = vs; k < vs + vn; ++k) {
            Vec3f d = (eval_x[data.constraint.torque_vertices[k].index] - center);
            c00 += d[0] * d[0]; c01 += d[0] * d[1]; c02 += d[0] * d[2];
            c11 += d[1] * d[1]; c12 += d[1] * d[2]; c22 += d[2] * d[2];
        }
        float inv_n = 1.0f / (float)vn;
        c00 *= inv_n; c01 *= inv_n; c02 *= inv_n;
        c11 *= inv_n; c12 *= inv_n; c22 *= inv_n;

        // Eigendecompose
        Vec3f evals;
        Mat3x3f evecs;
        eigen_sym3x3(c00, c01, c02, c11, c12, c22, evals, evecs);

        result[g].center = center;
        Vec3f ax = evecs.col(grp.axis_component).normalized();
        // Resolve sign ambiguity using hint vertex: orient axis so
        // dot(axis, hint_vertex - centroid) > 0
        Vec3f hint_dir = (eval_x[grp.hint_vertex] - center);
        if (ax.dot(hint_dir) < 0.0f) ax = -ax;
        result[g].axis = ax;

        // Compute Σ|r_perp_i|² for torque normalization so total torque = magnitude
        float r_perp_sq_sum = 0.0f;
        for (unsigned k = vs; k < vs + vn; ++k) {
            Vec3f r = (eval_x[data.constraint.torque_vertices[k].index] - center);
            Vec3f r_perp = r - ax * ax.dot(r);
            r_perp_sq_sum += r_perp.squaredNorm();
        }
        result[g].inv_r_perp_sq_sum = (r_perp_sq_sum > 1e-12f) ? 1.0f / r_perp_sq_sum : 0.0f;
    } DISPATCH_END;
}

void embed_momentum_force_hessian(const DataSet &data,
                                  const Vec<Vec3f> &eval_x,
                                  const Vec<Vec3f> &velocity, float dt,
                                  const Vec<Vec3f> &target, Vec<float> &force,
                                  Vec<Mat3x3f> &diag_hess,
                                  const ParamSet &param,
                                  const Vec<TorqueGroupResult> &torque_result) {
    DISPATCH_START(data.vertex.curr.size)
    [data, eval_x, velocity, dt, target, force, diag_hess,
     param, torque_result] __device__(unsigned i) mutable {
        if (data.prop.vertex[i].fix_index == 0) {
            energy::embed_vertex_force_hessian(data, eval_x, velocity, target,
                                               force, diag_hess, dt, param,
                                               torque_result, i);
        }
    } DISPATCH_END;
}

void embed_elastic_force_hessian(const DataSet &data, const Vec<Vec3f> &eval_x,
                                 Vec<float> &force, FixedCSRMat &fixed_hess,
                                 float dt, const ParamSet &param) {
    unsigned surface_vert_count = data.surface_vert_count;
    unsigned hinge_count = data.mesh.mesh.hinge.size;
    unsigned shell_face_count = data.shell_face_count;
    unsigned rod_count = data.rod_count;
    unsigned tet_count = data.mesh.mesh.tet.size;

    DISPATCH_START(surface_vert_count)
    [data, eval_x, force, fixed_hess, dt,
     param] __device__(unsigned i) mutable {
        if (data.prop.vertex[i].fix_index == 0) {
            energy::embed_rod_bend_force_hessian(data, eval_x, force,
                                                 fixed_hess, dt, param, i);
        }
    } DISPATCH_END;

    if (rod_count > 0) {
        DISPATCH_START(rod_count)
        [data, eval_x, force, fixed_hess, dt,
         param] __device__(unsigned i) mutable {
            if (!data.prop.edge[i].fixed) {
                energy::embed_rod_force_hessian(data, eval_x, force, fixed_hess,
                                                dt, param, i);
            }
        } DISPATCH_END;
    }

    if (shell_face_count > 0) {
        DISPATCH_START(shell_face_count)
        [data, eval_x, force, fixed_hess, dt,
         param] __device__(unsigned i) mutable {
            if (!data.prop.face[i].fixed && !data.prop.face[i].rest_excluded) {
                energy::embed_face_force_hessian(data, eval_x, force,
                                                 fixed_hess, dt, param, i);
            }
        } DISPATCH_END;
    }

    if (tet_count > 0) {
        DISPATCH_START(tet_count)
        [data, eval_x, force, fixed_hess, dt,
         param] __device__(unsigned i) mutable {
            if (!data.prop.tet[i].fixed && !data.prop.tet[i].rest_excluded) {
                energy::embed_tet_force_hessian(data, eval_x, force, fixed_hess,
                                                dt, param, i);
            }
        } DISPATCH_END;
    }

    if (hinge_count > 0) {
        DISPATCH_START(hinge_count)
        [data, eval_x, force, fixed_hess, dt,
         param] __device__(unsigned i) mutable {
            if (data.prop.hinge[i].fixed == false &&
                (data.mesh.type.hinge[i] & 1) == 0) {
                energy::embed_hinge_force_hessian(data, eval_x, force,
                                                  fixed_hess, dt, param, i);
            }
        } DISPATCH_END;
    }

    // Painless Differentiable Rotation Dynamics bodies are EXACTLY rigid: no penalty energy is
    // assembled here. Rigidity is enforced by the reduced 6-DOF rigid solve
    // (solver.cu) plus the per-iteration rigid reconstruct in main.cu. The
    // assembled matrix carries only per-vertex inertia + contact on PDRD
    // vertices, which the reduced operator R = P^T M P projects exactly.
}

void embed_stitch_force_hessian(const DataSet &data, const Vec<Vec3f> &eval_x,
                                Vec<float> &force, FixedCSRMat &fixed_out,
                                const ParamSet &param) {
    unsigned seam_count = data.constraint.stitch.size;
    if (seam_count) {
        DISPATCH_START(seam_count)
        [data, eval_x, force, fixed_out, param] __device__(unsigned i) mutable {
            const Stitch &stitch = data.constraint.stitch[i];
            // 6-slot barycentric-barycentric stitch: index[0..2] /
            // weight[0..2] is the source barycentric (weights sum to 1),
            // index[3..5] / weight[3..5] is the target barycentric. A
            // non-SOLID source degenerates to index[0..2] = {s, s, s},
            // weight[0..2] = {1, 0, 0}, recovering single-vertex behavior.
            Vec6u index = stitch.index;
            const Vec3f &x0 = eval_x[index[0]];
            const Vec3f &x1 = eval_x[index[1]];
            const Vec3f &x2 = eval_x[index[2]];
            const Vec3f &x3 = eval_x[index[3]];
            const Vec3f &x4 = eval_x[index[4]];
            const Vec3f &x5 = eval_x[index[5]];
            const VertexParam &vp0 = data.param_arrays.vertex[data.prop.vertex[index[0]].param_index];
            const VertexParam &vp1 = data.param_arrays.vertex[data.prop.vertex[index[1]].param_index];
            const VertexParam &vp2 = data.param_arrays.vertex[data.prop.vertex[index[2]].param_index];
            const VertexParam &vp3 = data.param_arrays.vertex[data.prop.vertex[index[3]].param_index];
            const VertexParam &vp4 = data.param_arrays.vertex[data.prop.vertex[index[4]].param_index];
            const VertexParam &vp5 = data.param_arrays.vertex[data.prop.vertex[index[5]].param_index];
            float w[] = {stitch.weight[0], stitch.weight[1], stitch.weight[2],
                         stitch.weight[3], stitch.weight[4], stitch.weight[5]};
            float l0 = (w[0] * vp0.ghat + w[1] * vp1.ghat + w[2] * vp2.ghat +
                        w[3] * vp3.ghat + w[4] * vp4.ghat + w[5] * vp5.ghat) /
                       2.0f;
            float source_offset = fmaxf(fmaxf(vp0.offset, vp1.offset), vp2.offset);
            float target_offset = fmaxf(fmaxf(vp3.offset, vp4.offset), vp5.offset);
            float l_cap = param.stitch_length_factor * l0 + source_offset + target_offset;
            float s(1.0f / 6.0f);
            const Vec3f cog = s * x0 + s * x1 + s * x2 + s * x3 + s * x4 + s * x5;
            Vec3f z0 = w[0] * (x0 - cog) +
                       w[1] * (x1 - cog) +
                       w[2] * (x2 - cog);
            Vec3f z1 = w[3] * (x3 - cog) +
                       w[4] * (x4 - cog) +
                       w[5] * (x5 - cog);
            Vec3f t = z0 - z1;
            float l = fmin(l_cap, t.norm());
            Vec3f n = t / l;
            using Mat3x18f = Eigen::Matrix<float, 3, 18>;
            using Vec18f = Eigen::Vector<float, 18>;
            using Mat18x18f = Eigen::Matrix<float, 18, 18>;
            Mat3x18f dtdx;
            dtdx << w[0] * Mat3x3f::Identity(),
                    w[1] * Mat3x3f::Identity(),
                    w[2] * Mat3x3f::Identity(),
                    -w[3] * Mat3x3f::Identity(),
                    -w[4] * Mat3x3f::Identity(),
                    -w[5] * Mat3x3f::Identity();
            Vec3f dedt = (l / l0 - 1.0f) * n;
            Vec18f g = dtdx.transpose() * n;
            float r = (l - l0) / l;
            float c0 = fmaxf(0.0f, 1.0f - r) / l0;
            float c1 = fmaxf(0.0f, r / l0);
            Eigen::Matrix<float, 3, 6> gradient;
            gradient.col(0) = w[0] * dedt;
            gradient.col(1) = w[1] * dedt;
            gradient.col(2) = w[2] * dedt;
            gradient.col(3) = -w[3] * dedt;
            gradient.col(4) = -w[4] * dedt;
            gradient.col(5) = -w[5] * dedt;
            Mat18x18f hessian =
                c0 * g * g.transpose() + c1 * dtdx.transpose() * dtdx;
            // Raw stitch stiffness: scale the gradient and Hessian directly by
            // this stitch's per-object stiffness (no mass / time-scale
            // normalization), so the value is a direct force factor and stays
            // simple to control. Resolved per object at scene-build time.
            utility::atomic_embed_force<6>(
                index, stitch.stiffness * gradient, force);
            utility::atomic_embed_hessian<6>(
                index, stitch.stiffness * hessian, fixed_out);
        } DISPATCH_END;
    }
}

} // namespace energy
