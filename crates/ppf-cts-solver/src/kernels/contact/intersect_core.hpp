// File: intersect_core.hpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Single source of truth for the edge-triangle self-intersection
// predicate. Shared by the device intersection kernels
// (contact/intersect_geometry.kernel.cpp) and the host build-time
// self-intersection check (ppf-cts-core, via the extern "C" shim in
// ppf-cts-core/cpp/intersect_ffi.cpp).
//
// Dependency-free (no STL, no CUDA runtime, no Eigen) and templated on the
// scalar type so each caller keeps its own precision: the device
// instantiates with float (matching the solver's fp32 math), the host
// build-check with double. The
// routine is translation-invariant (it only ever uses coordinate
// differences), so the device wrapper may pass coordinates pre-translated
// relative to a shared origin, preserving float precision near large
// magnitudes, while the host passes absolute coordinates; both are exact.

#ifndef CTS_INTERSECT_CORE_HPP
#define CTS_INTERSECT_CORE_HPP

#if defined(__CUDACC__) || defined(__HIPCC__)
#define ISECT_HD __host__ __device__
#else
#define ISECT_HD
#endif

#ifndef SM_THREAD
#define SM_THREAD
#define ISECT_UNDEF_THREAD
#endif
#ifndef SM_DIV
#define SM_DIV(a, b) ((a) / (b))
#define ISECT_UNDEF_DIV
#endif

namespace isect {

template <class T>
ISECT_HD inline T dot3(SM_THREAD const T *a, SM_THREAD const T *b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// True if `p` lies inside the triangle `(0, d1, d2)`. 2x2 Gram (Cramer)
// solve; degenerate (zero-area) triangles return false.
template <class T>
ISECT_HD inline bool
point_triangle_inside(SM_THREAD const T *p, SM_THREAD const T *d1,
                      SM_THREAD const T *d2) {
    T a00 = dot3(d1, d1);
    T a01 = dot3(d1, d2);
    T a11 = dot3(d2, d2);
    T b0 = dot3(d1, p);
    T b1 = dot3(d2, p);
    T det = a00 * a11 - a01 * a01;
    if (det == T(0)) {
        return false;
    }
    T w0 = SM_DIV(a11 * b0 - a01 * b1, det);
    T w1 = SM_DIV(a00 * b1 - a01 * b0, det);
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
ISECT_HD inline bool edge_triangle_intersect(
    SM_THREAD const T *e0, SM_THREAD const T *e1, SM_THREAD const T *v0,
    SM_THREAD const T *v1, SM_THREAD const T *v2) {
    T d1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    T d2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
    T a0[3] = {e0[0] - v0[0], e0[1] - v0[1], e0[2] - v0[2]};
    T a1[3] = {e1[0] - v0[0], e1[1] - v0[1], e1[2] - v0[2]};
    T n[3] = {d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2],
              d1[0] * d2[1] - d1[1] * d2[0]};
    T s1 = dot3(a0, n);
    T s2 = dot3(a1, n);
    // Compare the two SIGNS instead of the sign of their product. The two are
    // equivalent in exact arithmetic, and this form agrees with `s1 * s2 < 0`
    // on every input WHOSE PRODUCT DOES NOT UNDERFLOW, including zero, signed
    // zero, infinity and NaN. On the inputs whose product does underflow the
    // two deliberately disagree, and that disagreement is the whole point: see
    // below. The sign form never MATERIALIZES the product, which the product
    // form must.
    //
    // That matters because s1 and s2 are signed volumes: each is the triangle's
    // area vector dotted with an endpoint offset, so for an edge lying nearly in
    // the triangle's plane both are tiny while still being ordinary normal
    // floats, and their product underflows. Measured on the Apple GPU, which
    // flushes subnormals to zero in every math mode, the product form reported
    // `-0` and therefore NO CROSSING for 5 of 8 genuine crossings, the first at
    // s1 = 1e-19, s2 = -1e-19. This routine is the pierce predicate behind
    // contact::check_intersection, the final penetration gate, so a missed
    // crossing is a missed interpenetration.
    //
    // It is not only a Metal concern, though the two backends fail at different
    // magnitudes and the distinction matters. A backend with gradual underflow
    // (CUDA, nvcc's default -ftz=false) keeps the product as a subnormal and
    // stays correct down to the smallest subnormal, ~1.4e-45; below that it
    // underflows to -0 and misses the crossing too. In the same measurement the
    // product form missed 1e-25 * -1e-25 and q^3 * -q^3 on the HOST reference
    // as well (Apple clang, arm64), for q = 7.5e-9, so a scalar triple product
    // of edge vectors whose components reach that magnitude underflows on any
    // backend. The sign form has no threshold at all: it caught 8 of 8.
    const bool s1_neg = s1 < T(0), s1_pos = s1 > T(0);
    const bool s2_neg = s2 < T(0), s2_pos = s2 > T(0);
    if ((s1_neg && s2_pos) || (s1_pos && s2_neg)) {
        T t = SM_DIV(s1, s1 - s2);
        T r[3] = {(T(1) - t) * a0[0] + t * a1[0],
                  (T(1) - t) * a0[1] + t * a1[1],
                  (T(1) - t) * a0[2] + t * a1[2]};
        return point_triangle_inside(r, d1, d2);
    }
    return false;
}

} // namespace isect

#ifdef ISECT_UNDEF_DIV
#undef SM_DIV
#undef ISECT_UNDEF_DIV
#endif
#ifdef ISECT_UNDEF_THREAD
#undef SM_THREAD
#undef ISECT_UNDEF_THREAD
#endif

#endif // CTS_INTERSECT_CORE_HPP
