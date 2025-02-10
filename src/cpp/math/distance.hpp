// File: distance.hpp
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef DISTANCE_HPP
#define DISTANCE_HPP

#include "../common.hpp"
#include "../data.hpp"
#include <Eigen/Cholesky>

namespace distance {

__device__ bool solve(const Mat2x2f &a, const Vec2f &b, Vec2f &x) {
    float det = a(0, 0) * a(1, 1) - a(1, 0) * a(0, 1);
    if (det) {
        Mat2x2f a_inv;
        a_inv << a(1, 1) / det, -a(0, 1) / det, -a(1, 0) / det, a(0, 0) / det;
        x = a_inv * b;
        return true;
    }
    return false;
}

template <class T>
__device__ Vec2f point_edge_distance_coeff(const SVec<T, 3> &p,
                                           const SVec<T, 3> &e0,
                                           const SVec<T, 3> &e1) {
    Vec3f r = (e1 - e0).template cast<float>();
    float d = r.squaredNorm();
    if (d > EPSILON) {
        float x = r.dot((p - e0).template cast<float>()) / d;
        return Vec2f(1.0f - x, x);
    } else {
        return Vec2f(0.5f, 0.5f);
    }
}

template <class T>
__device__ Vec3f point_triangle_distance_coeff(const SVec<T, 3> &p,
                                               const SVec<T, 3> &t0,
                                               const SVec<T, 3> &t1,
                                               const SVec<T, 3> &t2) {
    Vec3f r0 = (t1 - t0).template cast<float>();
    Vec3f r1 = (t2 - t0).template cast<float>();
    Mat3x2f a;
    a << r0, r1;
    Eigen::Transpose<Mat3x2f> a_t = a.transpose();
    Vec2f c;
    if (!solve(a_t * a, a_t * (p - t0).template cast<float>(), c)) {
        c = Vec2f(1.0f / 3.0f, 1.0f / 3.0f);
    }
    return Vec3f(1.0f - c[0] - c[1], c[0], c[1]);
}

template <class T>
__device__ Vec4f edge_edge_distance_coeff(const SVec<T, 3> &ea0,
                                          const SVec<T, 3> &ea1,
                                          const SVec<T, 3> &eb0,
                                          const SVec<T, 3> &eb1) {
    Vec3f r0 = (ea1 - ea0).template cast<float>();
    Vec3f r1 = (eb1 - eb0).template cast<float>();
    Mat3x2f a;
    a << r0, -r1;
    Eigen::Transpose<Mat3x2f> a_t = a.transpose();
    Vec2f x;
    if (solve(a.transpose() * a,
              a.transpose() * (eb0 - ea0).template cast<float>(), x)) {
        return Vec4f(1.0f - x[0], x[0], 1.0f - x[1], x[1]);
    } else {
        return Vec4f(0.5f, 0.5f, 0.5f, 0.5f);
    }
}

template <class T>
__device__ Vec3f point_triangle_distance_coeff_unclassified(
    const SVec<T, 3> &p, const SVec<T, 3> &t0, const SVec<T, 3> &t1,
    const SVec<T, 3> &t2) {

    Vec3f c = point_triangle_distance_coeff(p, t0, t1, t2);
    if (c[0] >= 0.0f && c[0] <= 1.0f && c[1] >= 0.0f && c[1] <= 1.0f &&
        c[2] >= 0.0f && c[2] <= 1.0f) {
        return c;
    } else if (c[0] < 0.0f) {
        Vec2f c = point_edge_distance_coeff(p, t1, t2);
        if (c(0) >= 0.0f && c(0) <= 1.0f) {
            return Vec3f(0.0f, c(0), c(1));
        } else {
            if (c(0) > 1.0f) {
                return Vec3f(0.0f, 1.0f, 0.0f);
            } else {
                return Vec3f(0.0f, 0.0f, 1.0f);
            }
        }
    } else if (c[1] < 0.0f) {
        Vec2f c = point_edge_distance_coeff(p, t0, t2);
        if (c(0) >= 0.0f && c(0) <= 1.0f) {
            return Vec3f(c(0), 0.0f, c(1));
        } else {
            if (c(0) > 1.0f) {
                return Vec3f(1.0f, 0.0f, 0.0f);
            } else {
                return Vec3f(0.0f, 0.0f, 1.0f);
            }
        }
    } else {
        Vec2f c = point_edge_distance_coeff(p, t0, t1);
        if (c(0) >= 0.0f && c(0) <= 1.0f) {
            return Vec3f(c(0), c(1), 0.0f);
        } else {
            if (c(0) > 1.0f) {
                return Vec3f(1.0f, 0.0f, 0.0f);
            } else {
                return Vec3f(0.0f, 1.0f, 0.0f);
            }
        }
    }
}

template <class T>
__device__ float point_triangle_distance_squared_unclassified(
    const SVec<T, 3> &p, const SVec<T, 3> &t0, const SVec<T, 3> &t1,
    const SVec<T, 3> &t2) {
    Vec3f c = point_triangle_distance_coeff_unclassified(p, t0, t1, t2);
    Vec3f x = c(0) * (t0 - p).template cast<float>() +
              c(1) * (t1 - p).template cast<float>() +
              c(2) * (t2 - p).template cast<float>();
    return x.squaredNorm();
}

template <class T>
__device__ Vec4f edge_edge_distance_coeff_unclassified(const SVec<T, 3> &ea0,
                                                       const SVec<T, 3> &ea1,
                                                       const SVec<T, 3> &eb0,
                                                       const SVec<T, 3> &eb1) {

    Vec4f c = edge_edge_distance_coeff(ea0, ea1, eb0, eb1);
    if (c[0] >= 0.0f && c[0] <= 1.0f && c[1] >= 0.0f && c[1] <= 1.0f &&
        c[2] >= 0.0f && c[2] <= 1.0f && c[3] >= 0.0f && c[3] <= 1.0f) {
        return c;
    } else {
        Vec2f c1 = point_edge_distance_coeff(ea0, eb0, eb1);
        Vec2f c2 = point_edge_distance_coeff(ea1, eb0, eb1);
        Vec2f c3 = point_edge_distance_coeff(eb0, ea0, ea1);
        Vec2f c4 = point_edge_distance_coeff(eb1, ea0, ea1);
        if (c1(0) < 0.0f) {
            c1 = Vec2f(0.0f, 1.0f);
        } else if (c1(0) > 1.0f) {
            c1 = Vec2f(1.0f, 0.0f);
        }
        if (c2(0) < 0.0f) {
            c2 = Vec2f(0.0f, 1.0f);
        } else if (c2(0) > 1.0f) {
            c2 = Vec2f(1.0f, 0.0f);
        }
        if (c3(0) < 0.0f) {
            c3 = Vec2f(0.0f, 1.0f);
        } else if (c3(0) > 1.0f) {
            c3 = Vec2f(1.0f, 0.0f);
        }
        if (c4(0) < 0.0f) {
            c4 = Vec2f(0.0f, 1.0f);
        } else if (c4(0) > 1.0f) {
            c4 = Vec2f(1.0f, 0.0f);
        }
        Vec4f types[] = {
            Vec4f(1.0f, 0.0f, c1(0), c1(1)), Vec4f(0.0f, 1.0f, c2(0), c2(1)),
            Vec4f(c3(0), c3(1), 1.0f, 0.0f), Vec4f(c4(0), c4(1), 0.0f, 1.0f)};
        unsigned index = 0;
        float di = FLT_MAX;
        for (unsigned i = 0; i < 4; ++i) {
            const auto &c = types[i];
            Vec3f x0 = c(0) * ea0.template cast<float>() +
                       c(1) * ea1.template cast<float>();
            Vec3f x1 = c(2) * eb0.template cast<float>() +
                       c(3) * eb1.template cast<float>();
            float d = (x0 - x1).squaredNorm();
            if (d < di) {
                index = i;
                di = d;
            }
        }
        return types[index];
    }
}

template <class T>
__device__ float edge_edge_distance_squared_unclassified(
    const SVec<T, 3> &ea0, const SVec<T, 3> &ea1, const SVec<T, 3> &eb0,
    const SVec<T, 3> &eb1) {
    Vec4f c = edge_edge_distance_coeff_unclassified(ea0, ea1, eb0, eb1);
    Vec3f x0 = c[0] * ea0.template cast<float>() +
               c[1] * ea1.template cast<float>();
    Vec3f x1 = c[2] * eb0.template cast<float>() +
               c[3] * eb1.template cast<float>();
    return (x1 - x0).squaredNorm();
}

} // namespace distance

#endif
