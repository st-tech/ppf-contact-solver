// File: friction.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef FRICTION_HPP
#define FRICTION_HPP

#include "../../common.hpp"
#include "../../data.hpp"

struct Friction {
    const Vec3f &dx;
    Mat3x3f P;
    float lambda;
    float mu;
    float contact;
    __device__ Friction(const Vec3f &force_contact, const Vec3f &dx,
                        const Vec3f &normal, float mu, float min_dx)
        : dx(dx), mu(mu) {
        contact = -normal.dot(force_contact);
        P = get_projection(normal);

        if (mu > 0.0f) {
            float denom = (P * dx).squaredNorm();
            if (denom > 0.0f) {
                lambda = mu * contact / fmax(min_dx, sqrt(denom));
            } else {
                lambda = mu * contact / min_dx;
            }
        } else {
            lambda = 0.0f;
        }
    }
    __device__ Vec3f gradient() const { return lambda * (P * dx); }
    __device__ Mat3x3f hessian() const { return lambda * P; }
    __device__ Mat3x3f get_projection(const Vec3f &normal) {
        return Mat3x3f::Identity() - normal * normal.transpose();
    }
};

#endif
