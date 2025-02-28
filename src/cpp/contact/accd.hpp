// File: accd.hpp
// Author: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef ACCD_HPP
#define ACCD_HPP

#include "distance.hpp"
#include <cassert>

namespace accd {

template <class T> __device__ void centerize(Mat3x4<T> &x) {
    Vec3f mov = Vec3f::Zero();
    T scale(0.25f);
    for (int k = 0; k < 4; k++) {
        mov += scale * x.col(k);
    }
    for (int k = 0; k < 4; k++) {
        x.col(k) -= mov;
    }
}

template <typename F>
__device__ float ccd_helper(const Mat3x4f &x0, const Mat3x4f &dx, float u_max,
                            F square_dist_func, float offset,
                            const ParamSet &param) {
    float toi = 0.0f;
    float max_t = param.line_search_max_t;
    float eps = param.ccd_reduction * (sqrtf(square_dist_func(x0)) - offset);
    float target = eps + offset;
    float eps_sqr = eps * eps;
    float inv_u_max = 1.0f / u_max;
    while (true) {
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

struct EdgeEdgeSquaredDist {
    __device__ float operator()(const Mat3x4f &x) {
        const Vec3f &p0 = x.col(0);
        const Vec3f &p1 = x.col(1);
        const Vec3f &q0 = x.col(2);
        const Vec3f &q1 = x.col(3);
        return distance::edge_edge_distance_squared_unclassified(p0, p1, q0,
                                                                 q1);
    }
};

struct PointTriangleSquaredDist {
    __device__ float operator()(const Mat3x4f &x) {
        const Vec3f &p = x.col(0);
        const Vec3f &t0 = x.col(1);
        const Vec3f &t1 = x.col(2);
        const Vec3f &t2 = x.col(3);
        return distance::point_triangle_distance_squared_unclassified(p, t0, t1,
                                                                      t2);
    }
};

__device__ float max_relative_u(const Mat3x4f &u) {
    float max_u = 0.0f;
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            Vec3f du = u.col(i) - u.col(j);
            max_u = std::max(max_u, du.squaredNorm());
        }
    }
    return sqrt(max_u);
}

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
    centerize(x0);
    dx << dp, dt0, dt1, dt2;
    centerize(dx);
    float u_max = max_relative_u(dx);
    if (u_max) {
        PointTriangleSquaredDist dist_func;
        return ccd_helper(x0, dx, u_max, dist_func, offset, param);
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
    centerize(x0);
    dx << dea0, dea1, deb0, deb1;
    centerize(dx);
    float u_max = max_relative_u(dx);
    if (u_max) {
        EdgeEdgeSquaredDist dist_func;
        return ccd_helper(x0, dx, u_max, dist_func, offset, param);
    } else {
        return param.line_search_max_t;
    }
}

} // namespace accd

#endif
