// File: dihedral_angle.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef DIHEDRAL_ANGLE_HPP
#define DIHEDRAL_ANGLE_HPP

#include "../../common.hpp"
#include "../../data.hpp"

namespace dihedral_angle {

__device__ static Vec4u remap(Vec4u hinge) {
    return Vec4u(hinge[2], hinge[1], hinge[0], hinge[3]);
}

__device__ static Mat3x4f face_dihedral_angle_grad(const Vec3f &v2,
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

__device__ static float face_dihedral_angle(const Vec3f &v0, const Vec3f &v1,
                                            const Vec3f &v2,
                                            const Vec3f &v3) {
    const Vec3f n1 = (v1 - v0).cross(v2 - v0);
    const Vec3f n2 = (v2 - v3).cross(v1 - v3);
    float dot = n1.dot(n2) / sqrt(n1.squaredNorm() * n2.squaredNorm());
    float angle = acosf(fmaxf(-1.0f, fminf(1.0f, dot)));
    if (n2.cross(n1).dot(v1 - v2) < 0.0f) {
        angle = -angle;
    }
    return angle;
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

__device__ static void face_compute_force_hessian(const Vec<Vec3f> &vertex,
                                                  Vec4u &hinge,
                                                  float rest_angle,
                                                  Mat3x4f &force,
                                                  Mat12x12f &hess) {
    hinge = remap(hinge);
    const Vec3f x0 = vertex[hinge[0]];
    const Vec3f x1 = vertex[hinge[1]];
    const Vec3f x2 = vertex[hinge[2]];
    const Vec3f x3 = vertex[hinge[3]];
    const float angle = face_dihedral_angle(x0, x1, x2, x3);
    const Mat3x4f angle_grad = face_dihedral_angle_grad(x0, x1, x2, x3);
    const Vec12f g = Map<const Vec12f>(angle_grad.data());
    force = (angle - rest_angle) * angle_grad;
    hess = g * g.transpose();
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
    Vec3f e0 = x0 - x1;
    Vec3f e1 = x2 - x1;
    float theta = acosf(e0.dot(e1) / (e0.norm() * e1.norm()));
    float diff = theta - rest_angle;
    return 0.5f * diff * diff;
}

__device__ static Mat3x2f gradient_theta(const Vec3f &e0, const Vec3f &e1) {
    Vec3f n = e0.cross(e1).normalized();
    Vec3f e0perp = e0.cross(n);
    Vec3f e1perp = e1.cross(n);
    Vec3f g0 = e0perp / e0.dot(e0);
    Vec3f g1 = -e1perp / e1.dot(e1);
    Mat3x2f G;
    G << g0, g1;
    return G;
}

__device__ static Mat3x3f strand_gradient(const Vec3f &x0, const Vec3f &x1,
                                          const Vec3f &x2, float rest_angle) {
    Vec3f e0 = x0 - x1;
    Vec3f e1 = x2 - x1;
    float cosTheta =
        fmaxf(-1.0f, fminf(1.0f, e0.dot(e1) / (e0.norm() * e1.norm())));
    float theta = acosf(cosTheta);
    Mat3x2f Pfinal = (theta - rest_angle) * gradient_theta(e0, e1);
    Mat3x3f PK1;
    PK1 << Pfinal.col(0), -Pfinal.col(0) - Pfinal.col(1), Pfinal.col(1);
    return PK1;
}

// Force + Gauss-Newton Hessian for the rod-bend energy E = ½(θ-θ₀)²,
// matching the face_compute_force_hessian pattern. The Hessian uses the
// raw gradient of θ outer-producted with itself; using strand_gradient's
// output (which already carries the (θ-θ₀) factor) for the outer product
// gives k(θ-θ₀)²(∇θ⊗∇θ), which collapses to zero at the rest pose and
// leaves the rod with no curvature in the Newton solve, so any FP drift
// grows unbounded.
__device__ static void strand_compute_force_hessian(
    const Vec3f &x0, const Vec3f &x1, const Vec3f &x2, float rest_angle,
    Mat3x3f &force, Mat9x9f &hess) {
    Vec3f e0 = x0 - x1;
    Vec3f e1 = x2 - x1;
    float cosTheta =
        fmaxf(-1.0f, fminf(1.0f, e0.dot(e1) / (e0.norm() * e1.norm())));
    float theta = acosf(cosTheta);
    Mat3x2f gtheta = gradient_theta(e0, e1);
    Mat3x3f angle_grad;
    angle_grad << gtheta.col(0), -gtheta.col(0) - gtheta.col(1), gtheta.col(1);
    Vec9f g = Map<Vec9f>(angle_grad.data());
    force = (theta - rest_angle) * angle_grad;
    hess = g * g.transpose();
}

} // namespace dihedral_angle

#endif
