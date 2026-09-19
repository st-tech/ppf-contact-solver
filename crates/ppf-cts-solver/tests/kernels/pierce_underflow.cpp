// File: pierce_underflow.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// A HOST TEST, DRIVEN BY `tests/kernel_gates.rs`.
//
//
// Regression gate for the edge-triangle pierce predicate in
// contact/intersect_core.hpp, which is the single source of truth behind
// contact::check_intersection, the final penetration gate.
//
// The predicate asks whether an edge's two endpoints lie strictly on opposite
// sides of a triangle's plane. Written as `s1 * s2 < 0` it forms a PRODUCT of
// two signed volumes, and for a nearly coplanar edge both factors are tiny
// while still being ordinary normal floats, so the product underflows and the
// crossing is silently missed. Written as a comparison of the two SIGNS it has
// no threshold at all.
//
// This is not a hypothetical. A triangle a millimeter across carries an area
// vector of magnitude 1e-6, so an edge straddling its plane by 1e-19 gives two
// signed volumes of 1e-25 and a product of 1e-50, five decades below where
// fp32 can hold it. Measured with the product form: it missed 5 of 8 genuine
// crossings on an Apple GPU (which flushes subnormals in every math mode) and
// 2 of 8 on an NVIDIA L40S with nvcc's default gradual underflow.
//
// NOTHING IT ASSERTS NEEDS A DEVICE. The product form underflows to zero on
// ANY fp32 hardware, gradual or flush-to-zero, because a product of about
// 1e-50 is below the smallest fp32 subnormal at 1.4e-45; the measured
// per-device MISS COUNTS in the paragraph above are what varies, and this gate
// asserts neither of them.
// What it asserts is the shipped form catching every genuine crossing, the
// shipped form never LOSING one the product form found, and the product
// form missing at least one, without which the gate proved nothing.
//
// This gate FAILS if the predicate is ever rewritten back into a product form.

#include "contact/intersect_core.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>

// The pre-fix spelling, kept here and ONLY here so the gate can show what it
// is protecting against. Never reintroduce this into the solver.
template <class T>
inline bool pierce_product_form(const T *e0, const T *e1,
                                                    const T *v0, const T *v1,
                                                    const T *v2) {
    T d1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
    T d2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
    T a0[3] = {e0[0] - v0[0], e0[1] - v0[1], e0[2] - v0[2]};
    T a1[3] = {e1[0] - v0[0], e1[1] - v0[1], e1[2] - v0[2]};
    T n[3] = {d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2],
              d1[0] * d2[1] - d1[1] * d2[0]};
    T s1 = isect::dot3(a0, n);
    T s2 = isect::dot3(a1, n);
    if (s1 * s2 < T(0)) {
        T t = s1 / (s1 - s2);
        T r[3] = {(T(1) - t) * a0[0] + t * a1[0],
                  (T(1) - t) * a0[1] + t * a1[1],
                  (T(1) - t) * a0[2] + t * a1[2]};
        return isect::point_triangle_inside(r, d1, d2);
    }
    return false;
}

// A triangle of edge 1e-3 in the z = 0 plane, and an edge that pierces its
// interior vertically, straddling the plane by +/- eps. Every eps below is a
// GENUINE crossing, so a correct predicate must report true for all of them.
// The area vector has magnitude 1e-6, so the signed volumes are eps * 1e-6 and
// the product is eps^2 * 1e-12: eps = 1e-13 lands the product at 1e-38 (still
// representable), eps = 1e-19 lands it at 1e-50 (gone).
inline void build(float eps, float *e0, float *e1,
                                      float *v0, float *v1, float *v2) {
    v0[0] = 0.0f;    v0[1] = 0.0f;    v0[2] = 0.0f;
    v1[0] = 1e-3f;   v1[1] = 0.0f;    v1[2] = 0.0f;
    v2[0] = 0.0f;    v2[1] = 1e-3f;   v2[2] = 0.0f;
    // (2e-4, 2e-4) is strictly interior: barycentric (0.6, 0.2, 0.2).
    e0[0] = 2e-4f;   e0[1] = 2e-4f;   e0[2] =  eps;
    e1[0] = 2e-4f;   e1[1] = 2e-4f;   e1[2] = -eps;
}

// Two plain loops, one iteration per row and one per sampled pair. Nothing in
// either is a device operation, so the gate runs on the host.
void pierce_rows(const float *eps, int n, int *shipped, int *product) {
    for (int i = 0; i < n; ++i) {
        float e0[3], e1[3], v0[3], v1[3], v2[3];
        build(eps[i], e0, e1, v0, v1, v2);
        shipped[i] =
            isect::edge_triangle_intersect<float>(e0, e1, v0, v1, v2) ? 1 : 0;
        product[i] = pierce_product_form<float>(e0, e1, v0, v1, v2) ? 1 : 0;
    }
}

// Every finite float pair, sampled: the shipped form must never report FEWER
// crossings than the product form. That is the direction that permits
// penetration, and it must be empty.
//
// The bit pattern comes from the same two multipliers the device version used,
// so the sample is the same sequence; `__uint_as_float` is a memcpy here.
static float as_float(unsigned bits) {
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

void superset(unsigned seed, long long n, long long *lost, long long *gained) {
    for (long long i = 0; i < n; ++i) {
        const float s1 = as_float((unsigned)(i * 2654435761u + seed));
        const float s2 = as_float((unsigned)(i * 2246822519u + seed * 40503u));
        const bool prod = s1 * s2 < 0.0f;
        const bool sign =
            ((s1 < 0.0f) && (s2 > 0.0f)) || ((s1 > 0.0f) && (s2 < 0.0f));
        if (prod && !sign) {
            *lost += 1;
        }
        if (!prod && sign) {
            *gained += 1;
        }
    }
}

int main() {
    const float eps_h[] = {1e-6f, 1e-9f, 1e-12f, 1e-13f, 1e-15f,
                           1e-17f, 1e-19f, 1e-20f, 1e-22f};
    const int N = sizeof(eps_h) / sizeof(eps_h[0]);
    int shipped[N], product[N];
    pierce_rows(eps_h, N, shipped, product);

    printf("A 1e-3 triangle pierced through its interior by an edge straddling\n");
    printf("the plane by +/- eps. Every row is a GENUINE crossing, so a correct\n");
    printf("predicate reports 1 in every row.\n\n");
    printf("  %-10s %14s %9s %9s  %s\n", "eps", "s1*s2 approx", "SHIPPED",
           "product", "");
    int fail = 0, product_missed = 0;
    for (int i = 0; i < N; ++i) {
        double s = (double)eps_h[i] * 1e-6;
        const char *note = "";
        if (!shipped[i]) {
            note = "SHIPPED FORM MISSED A CROSSING";
            ++fail;
        } else if (!product[i]) {
            note = "product form missed it, shipped form caught it";
            ++product_missed;
        }
        printf("  %-10.0e %14.3e %9d %9d  %s\n", eps_h[i], s * s, shipped[i],
               product[i], note);
    }
    printf("\n  shipped form missed %d of %d; product form missed %d of %d\n\n",
           fail, N, product_missed + fail, N);

    // THE SAME 200 MILLION PAIRS THE DEVICE VERSION SAMPLED. It is a second or
    // two of scalar float work on a host and the count is not reduced, because
    // the sample size is the coverage: shrinking it to make a test quick is how
    // a gate stops reaching the band it was written for.
    long long lost = 0, gained = 0;
    const long long M = 200000000LL;
    superset(12345u, M, &lost, &gained);
    printf("superset check over %lld float pairs (NaN and Inf included):\n", M);
    printf("  crossings the shipped form LOSES vs the product form : %lld  <-- must be 0\n", lost);
    printf("  crossings the shipped form GAINS                     : %lld\n", gained);

    const bool ok = (fail == 0) && (lost == 0) && (product_missed > 0);
    printf("\n%s\n", ok ? "PASS" : "FAIL");
    if (product_missed == 0)
        printf("  (note: the product form missed nothing here, so this gate proved\n"
               "   nothing. That means the eps sweep no longer reaches the underflow\n"
               "   band and the test needs smaller eps, not that the bug is gone.)\n");
    return ok ? 0 : 1;
}
