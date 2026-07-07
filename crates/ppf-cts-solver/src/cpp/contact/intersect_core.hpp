// File: intersect_core.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Single source of truth for the edge-triangle self-intersection
// predicate. Shared by the device contact kernels (contact.cu, compiled
// by nvcc) and the host build-time self-intersection check (ppf-cts-core,
// via the extern "C" shim in ppf-cts-core/cpp/intersect_ffi.cpp).
//
// Dependency-free (no STL, no CUDA runtime, no Eigen) and templated on the
// scalar type so each caller keeps its own precision: the device
// instantiates with float (the solver's fp32 math), the host build-check
// with double (its f64 math). The routine is translation-invariant (it
// only ever uses coordinate differences), so the device wrapper may hand
// it vectors already relative to a shared origin while the host passes
// absolute coordinates; both are exact.

#ifndef PPF_CTS_INTERSECT_CORE_HPP
#define PPF_CTS_INTERSECT_CORE_HPP

#if defined(__CUDACC__)
#define PPF_ISECT_HD __host__ __device__
#else
#define PPF_ISECT_HD
#endif

namespace ppf_isect {

template <class T> PPF_ISECT_HD inline T dot3(const T *a, const T *b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// True if `p` lies inside the triangle `(0, d1, d2)`. 2x2 Gram (Cramer)
// solve; degenerate (zero-area) triangles return false.
template <class T>
PPF_ISECT_HD inline bool point_triangle_inside(const T *p, const T *d1,
                                               const T *d2) {
    T a00 = dot3(d1, d1);
    T a01 = dot3(d1, d2);
    T a11 = dot3(d2, d2);
    T b0 = dot3(d1, p);
    T b1 = dot3(d2, p);
    T det = a00 * a11 - a01 * a01;
    if (det == T(0)) {
        return false;
    }
    T w0 = (a11 * b0 - a01 * b1) / det;
    T w1 = (a00 * b1 - a01 * b0) / det;
    T w2 = T(1) - w0 - w1;
    T wmin = w0 < w1 ? w0 : w1;
    wmin = wmin < w2 ? wmin : w2;
    T wmax = w0 > w1 ? w0 : w1;
    wmax = wmax > w2 ? wmax : w2;
    return wmin >= T(0) && wmax <= T(1);
}

// True iff segment `(e0, e1)` strictly crosses triangle `(v0, v1, v2)`.
// Coplanar / touching cases return false; those are handled by the
// coplanar-overlap fallback in the host build-check.
template <class T>
PPF_ISECT_HD inline bool edge_triangle_intersect(const T *e0, const T *e1,
                                                 const T *v0, const T *v1,
                                                 const T *v2) {
    T d1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    T d2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
    T a0[3] = {e0[0] - v0[0], e0[1] - v0[1], e0[2] - v0[2]};
    T a1[3] = {e1[0] - v0[0], e1[1] - v0[1], e1[2] - v0[2]};
    T n[3] = {d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2],
              d1[0] * d2[1] - d1[1] * d2[0]};
    T s1 = dot3(a0, n);
    T s2 = dot3(a1, n);
    if (s1 * s2 < T(0)) {
        T t = s1 / (s1 - s2);
        T r[3] = {(T(1) - t) * a0[0] + t * a1[0],
                  (T(1) - t) * a0[1] + t * a1[1],
                  (T(1) - t) * a0[2] + t * a1[2]};
        return point_triangle_inside(r, d1, d2);
    }
    return false;
}

} // namespace ppf_isect

#endif // PPF_CTS_INTERSECT_CORE_HPP
