// File: accd.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef ACCD_HPP
#define ACCD_HPP

#include "distance.hpp"
#include <cassert>

namespace accd {

template <class T, unsigned R, unsigned C>
__device__ void centerize(SMat<T, R, C> &x) {
    SVec<T, R> mov = SVec<T, R>::Zero();
    T scale(1.0 / C);
    for (int k = 0; k < C; k++) {
        mov += scale * x.col(k);
    }
    for (int k = 0; k < C; k++) {
        x.col(k) -= mov;
    }
}

template <class T, unsigned R, unsigned C>
__device__ float max_relative_u(const SMat<T, R, C> &u) {
    float max_u = 0.0f;
    for (int i = 0; i < C; i++) {
        for (int j = i + 1; j < C; j++) {
            SVec<float, R> du = (u.col(i) - u.col(j)).template cast<float>();
            max_u = std::max<float>(max_u, du.squaredNorm());
        }
    }
    return sqrt(max_u);
}

template <typename F, typename T, unsigned R, unsigned C>
__device__ float ccd_helper(const SMat<T, R, C> &x0, const SMat<T, R, C> &dx,
                            float u_max, F square_dist_func, float offset,
                            const ParamSet &param) {
    float toi = 0.0f;
    float max_t = param.line_search_max_t;
    float eps = param.ccd_reduction * (sqrtf(square_dist_func(x0)) - offset);
    float target = eps + offset;
    float eps_sqr = eps * eps;
    float inv_u_max = 1.0f / u_max;
    for (unsigned k = 0; k < param.ccd_max_iter; k++) {
        float d2 = square_dist_func(x0 + toi * dx);
        float d_minus_target = (d2 - target * target) / (sqrtf(d2) + target);
        if ((max_t - toi) * u_max < d_minus_target - eps) {
            toi = max_t;
            break;
        } else if (toi > 0.0f && d_minus_target * d_minus_target < eps_sqr) {
            break;
        }
        float toi_next = toi + d_minus_target * inv_u_max;
        if (toi_next != toi) {
            toi = toi_next;
        } else {
            break;
        }
        if (toi > max_t) {
            toi = max_t;
            break;
        }
    }
    assert(toi > 0.0f);
    return toi;
}

template <typename T, typename Y> struct EdgeEdgeSquaredDist {
    __device__ float operator()(const Mat3x4<T> &x) {
        const Vec3<T> &p0 = x.col(0);
        const Vec3<T> &p1 = x.col(1);
        const Vec3<T> &q0 = x.col(2);
        const Vec3<T> &q1 = x.col(3);
        Vec4<T> c = distance::edge_edge_distance_coeff_unclassified<T, Y>(
                        p0, p1, q0, q1)
                        .template cast<T>();
        Vec3<T> x0 = c[0] * p0 + c[1] * p1;
        Vec3<T> x1 = c[2] * q0 + c[3] * q1;
        return Y((x1 - x0).dot(x1 - x0));
    }
};

template <typename T, typename Y> struct PointEdgeSquaredDist {
    __device__ float operator()(const Mat3x3<T> &x) {
        const Vec3<T> &p = x.col(0);
        const Vec3<T> &q0 = x.col(1);
        const Vec3<T> &q1 = x.col(2);
        Vec2<T> c =
            distance::point_edge_distance_coeff_unclassified<T, Y>(p, q0, q1)
                .template cast<T>();
        Vec3<T> q = c[0] * q0 + c[1] * q1;
        return Y((p - q).dot(p - q));
    }
};

template <typename T, typename Y> struct PointTriangleSquaredDist {
    __device__ float operator()(const Mat3x4<T> &x) {
        const Vec3<T> &p = x.col(0);
        const Vec3<T> &t0 = x.col(1);
        const Vec3<T> &t1 = x.col(2);
        const Vec3<T> &t2 = x.col(3);
        Vec3<T> c = distance::point_triangle_distance_coeff_unclassified<T, Y>(
                        p, t0, t1, t2)
                        .template cast<T>();
        Vec3<T> y = c(0) * (t0 - p) + c(1) * (t1 - p) + c(2) * (t2 - p);
        return Y(y.dot(y));
    }
};

__device__ float point_triangle_ccd(const Vec3f &p0, const Vec3f &p1,
                                    const Vec3f &t00, const Vec3f &t01,
                                    const Vec3f &t02, const Vec3f &t10,
                                    const Vec3f &t11, const Vec3f &t12,
                                    float offset, const ParamSet &param) {
    Vec3f dp = p1 - p0;
    Vec3f dt0 = t10 - t00;
    Vec3f dt1 = t11 - t01;
    Vec3f dt2 = t12 - t02;
    Mat3x4f x0;
    Mat3x4f dx;
    x0 << p0, t00, t01, t02;
    dx << dp, dt0, dt1, dt2;
    centerize<float, 3, 4>(x0);
    centerize<float, 3, 4>(dx);
    float u_max = max_relative_u<float, 3, 4>(dx);
    if (u_max) {
        PointTriangleSquaredDist<float, float> dist_func;
        return ccd_helper<PointTriangleSquaredDist<float, float>, float, 3, 4>(
            x0, dx, u_max, dist_func, offset, param);
    } else {
        return param.line_search_max_t;
    }
}

__device__ float point_edge_ccd(const Vec3f &p0, const Vec3f &p1,
                                const Vec3f &e00, const Vec3f &e01,
                                const Vec3f &e10, const Vec3f &e11,
                                float offset, const ParamSet &param) {
    Vec3f dp = p1 - p0;
    Vec3f dt0 = e10 - e00;
    Vec3f dt1 = e11 - e01;
    Mat3x3f x0;
    Mat3x3f dx;
    x0 << p0, e00, e01;
    dx << dp, dt0, dt1;
    centerize<float, 3, 3>(x0);
    centerize<float, 3, 3>(dx);
    float u_max = max_relative_u<float, 3, 3>(dx);
    if (u_max) {
        PointEdgeSquaredDist<float, float> dist_func;
        return ccd_helper<PointEdgeSquaredDist<float, float>, float, 3, 3>(
            x0, dx, u_max, dist_func, offset, param);
    } else {
        return param.line_search_max_t;
    }
}

__device__ float edge_edge_ccd(const Vec3f &ea00, const Vec3f &ea01,
                               const Vec3f &eb00, const Vec3f &eb01,
                               const Vec3f &ea10, const Vec3f &ea11,
                               const Vec3f &eb10, const Vec3f &eb11,
                               float offset, const ParamSet &param) {
    Vec3f dea0 = ea10 - ea00;
    Vec3f dea1 = ea11 - ea01;
    Vec3f deb0 = eb10 - eb00;
    Vec3f deb1 = eb11 - eb01;
    Mat3x4f x0;
    Mat3x4f dx;
    x0 << ea00, ea01, eb00, eb01;
    dx << dea0, dea1, deb0, deb1;
    centerize<float, 3, 4>(x0);
    centerize<float, 3, 4>(dx);
    float u_max = max_relative_u<float, 3, 4>(dx);
    if (u_max) {
        EdgeEdgeSquaredDist<float, float> dist_func;
        return ccd_helper<EdgeEdgeSquaredDist<float, float>, float, 3, 4>(
            x0, dx, u_max, dist_func, offset, param);
    } else {
        return param.line_search_max_t;
    }
}

} // namespace accd

#endif
