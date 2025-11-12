// File: distance.hpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#ifndef DISTANCE_HPP
#define DISTANCE_HPP

#include "../common.hpp"
#include "../data.hpp"
#include <Eigen/Cholesky>

namespace distance {

template <class Y>
__device__ void solve(const Mat2x2<Y> &a, const Vec2<Y> &b, Vec2<Y> &x,
                      Y &det) {
    det = a(0, 0) * a(1, 1) - a(1, 0) * a(0, 1);
    Mat2x2<Y> adj;
    adj << a(1, 1), -a(0, 1), -a(1, 0), a(0, 0);
    x = adj * b;
}

template <class T, class Y>
__device__ Vec2<Y> point_edge_distance_coeff(const Vec3<T> &p,
                                             const Vec3<T> &e0,
                                             const Vec3<T> &e1) {
    Vec3<Y> r = (e1 - e0).template cast<Y>();
    Y d = r.dot(r);
    if (d > Y(0.0)) {
        Y x = r.dot((p - e0).template cast<Y>());
        Vec2<Y> w(d - x, x);
        return w / d;
    } else {
        return Vec2<Y>::Ones() / Y(2.0);
    }
}

template <class T, class Y>
__device__ Vec3<Y>
point_triangle_distance_coeff(const Vec3<T> &p, const Vec3<T> &t0,
                              const Vec3<T> &t1, const Vec3<T> &t2) {
    Vec3<Y> r0 = (t1 - t0).template cast<Y>();
    Vec3<Y> r1 = (t2 - t0).template cast<Y>();
    Mat3x2<Y> a;
    a << r0, r1;
    Eigen::Transpose<Mat3x2<Y>> a_t = a.transpose();
    Y det;
    Vec2<Y> c;
    solve<Y>(a_t * a, a_t * (p - t0).template cast<Y>(), c, det);
    if (det) {
        Vec3<Y> w(det - c[0] - c[1], c[0], c[1]);
        return w / det;
    } else {
        assert(false && "This should be handled with care");
    }
}

template <class T, class Y>
__device__ Vec4<Y>
edge_edge_distance_coeff(const Vec3<T> &ea0, const Vec3<T> &ea1,
                         const Vec3<T> &eb0, const Vec3<T> &eb1) {
    Vec3<Y> r0 = (ea1 - ea0).template cast<Y>();
    Vec3<Y> r1 = (eb1 - eb0).template cast<Y>();
    Mat3x2<Y> a;
    a << r0, -r1;
    Eigen::Transpose<Mat3x2<Y>> a_t = a.transpose();
    Vec2<Y> x;
    Y det;
    solve<Y>(a.transpose() * a, a.transpose() * (eb0 - ea0).template cast<Y>(),
             x, det);
    if (det) {
        x = x / det;
        Vec2<Y> w0, w1;
        Vec3<Y> q0 = (eb0 - ea0).template cast<Y>();
        Vec3<Y> q1 = (eb1 - ea0).template cast<Y>();
        Vec3<Y> p0 = (ea0 - eb0).template cast<Y>();
        Vec3<Y> p1 = (ea1 - eb0).template cast<Y>();
        for (int k = 0; k < 4; k++) {
            w0 = point_edge_distance_coeff<Y, Y>(x(0) * r0, q0, q1);
            w1 = point_edge_distance_coeff<Y, Y>(x(1) * r1, p0, p1);
            x(0) = Y(0.5) * (w1(1) + x(0));
            x(1) = Y(0.5) * (w0(1) + x(1));
        }
        return Vec4<Y>(1.0 - x(0), x(0), 1.0 - x(1), x(1));
    } else {
        return Vec4<Y>(1.0, 0.0, 1.0, 0.0);
    }
}

template <class T, class Y>
__device__ Vec3<Y> point_triangle_distance_coeff_unclassified(
    const Vec3<T> &p, const Vec3<T> &t0, const Vec3<T> &t1, const Vec3<T> &t2) {

    Vec3<Y> c = point_triangle_distance_coeff<T, Y>(p, t0, t1, t2);
    if (c.minCoeff() >= Y(0.0) && c.maxCoeff() <= Y(1.0)) {
        return c;
    } else if (c[0] < Y(0.0)) {
        Vec2<Y> c = point_edge_distance_coeff<T, Y>(p, t1, t2);
        if (c(0) >= Y(0.0) && c(0) <= Y(1.0)) {
            return Vec3<Y>(Y(0.0), c(0), c(1));
        } else {
            if (c(0) > Y(1.0)) {
                return Vec3<Y>(Y(0.0), Y(1.0), Y(0.0));
            } else {
                return Vec3<Y>(Y(0.0), Y(0.0), Y(1.0));
            }
        }
    } else if (c[1] < Y(0.0)) {
        Vec2<Y> c = point_edge_distance_coeff<T, Y>(p, t0, t2);
        if (c(0) >= Y(0.0) && c(0) <= Y(1.0)) {
            return Vec3<Y>(c(0), Y(0.0), c(1));
        } else {
            if (c(0) > Y(1.0)) {
                return Vec3<Y>(Y(1.0), Y(0.0), Y(0.0));
            } else {
                return Vec3<Y>(Y(0.0), Y(0.0), Y(1.0));
            }
        }
    } else {
        Vec2<Y> c = point_edge_distance_coeff<T, Y>(p, t0, t1);
        if (c(0) >= Y(0.0) && c(0) <= Y(1.0)) {
            return Vec3<Y>(c(0), c(1), Y(0.0));
        } else {
            if (c(0) > Y(1.0)) {
                return Vec3<Y>(Y(1.0), Y(0.0), Y(0.0));
            } else {
                return Vec3<Y>(Y(0.0), Y(1.0), Y(0.0));
            }
        }
    }
}

template <class T, class Y>
__device__ Vec2<Y> point_edge_distance_coeff_unclassified(const Vec3<T> &p,
                                                          const Vec3<T> &e0,
                                                          const Vec3<T> &e1) {
    Vec2<Y> c = point_edge_distance_coeff<T, Y>(p, e0, e1);
    if (c(0) >= Y(0.0) && c(0) <= Y(1.0)) {
        return c;
    } else {
        if (c(0) > Y(1.0)) {
            return Vec2<Y>(Y(1.0), Y(0.0));
        } else {
            return Vec2<Y>(Y(0.0), Y(1.0));
        }
    }
}

template <class T, class Y>
__device__ Vec4<Y>
edge_edge_distance_coeff_unclassified(const Vec3<T> &ea0, const Vec3<T> &ea1,
                                      const Vec3<T> &eb0, const Vec3<T> &eb1) {

    Vec4<Y> c = edge_edge_distance_coeff<T, Y>(ea0, ea1, eb0, eb1);
    if (c.minCoeff() >= Y(0.0) && c.maxCoeff() <= Y(1.0)) {
        return c;
    } else {
        Vec2<Y> c1 = point_edge_distance_coeff<T, Y>(ea0, eb0, eb1);
        Vec2<Y> c2 = point_edge_distance_coeff<T, Y>(ea1, eb0, eb1);
        Vec2<Y> c3 = point_edge_distance_coeff<T, Y>(eb0, ea0, ea1);
        Vec2<Y> c4 = point_edge_distance_coeff<T, Y>(eb1, ea0, ea1);
        if (c1(0) < Y(0.0)) {
            c1 = Vec2<Y>(Y(0.0), Y(1.0));
        } else if (c1(0) > Y(1.0)) {
            c1 = Vec2<Y>(Y(1.0), Y(0.0));
        }
        if (c2(0) < Y(0.0)) {
            c2 = Vec2<Y>(Y(0.0), Y(1.0));
        } else if (c2(0) > Y(1.0)) {
            c2 = Vec2<Y>(Y(1.0), Y(0.0));
        }
        if (c3(0) < Y(0.0)) {
            c3 = Vec2<Y>(Y(0.0), Y(1.0));
        } else if (c3(0) > Y(1.0)) {
            c3 = Vec2<Y>(Y(1.0), Y(0.0));
        }
        if (c4(0) < Y(0.0)) {
            c4 = Vec2<Y>(Y(0.0), Y(1.0));
        } else if (c4(0) > Y(1.0)) {
            c4 = Vec2<Y>(Y(1.0), Y(0.0));
        }
        Vec4<Y> types[] = {Vec4<Y>(Y(1.0), Y(0.0), c1(0), c1(1)),
                           Vec4<Y>(Y(0.0), Y(1.0), c2(0), c2(1)),
                           Vec4<Y>(c3(0), c3(1), Y(1.0), Y(0.0)),
                           Vec4<Y>(c4(0), c4(1), Y(0.0), Y(1.0))};
        unsigned index = 0;
        Y di = std::numeric_limits<Y>::max();
        T s(0.25);
        Vec3<T> cog = s * ea0 + s * ea1 + s * eb0 + s * eb1;
        Vec3<T> _ea0 = ea0 - cog;
        Vec3<T> _ea1 = ea1 - cog;
        Vec3<T> _eb0 = eb0 - cog;
        Vec3<T> _eb1 = eb1 - cog;
        for (unsigned i = 0; i < sizeof(types) / sizeof(types[0]); ++i) {
            Vec4<T> c = types[i].template cast<T>();
            Vec3<T> x0 = c(0) * _ea0 + c(1) * _ea1;
            Vec3<T> x1 = c(2) * _eb0 + c(3) * _eb1;
            Vec3<Y> r = (x1 - x0).template cast<Y>();
            Y d = r.dot(r);
            if (d < di) {
                index = i;
                di = d;
            }
        }
        return types[index];
    }
}

} // namespace distance

#endif
