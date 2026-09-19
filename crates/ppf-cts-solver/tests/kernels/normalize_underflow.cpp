// File: normalize_underflow.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// A HOST TEST, DRIVEN BY `tests/kernel_gates.rs`.
//
// Regression gate for SMat::normalize() in linalg/smat.hpp.
//
// normalize() guards on the norm being positive so a zero-length vector stays
// zero instead of becoming 0/0. The subtlety is that norm() is
// la_sqrt(squaredNorm()), and SQUARING HALVES THE USABLE EXPONENT RANGE: a
// vector whose largest component is below sqrt(FLT_MIN) = 1.0842e-19 has a
// squared norm at or below the subnormal floor even though the vector itself
// is twenty decades clear of it. When that squared norm reaches zero the guard
// declines to act and normalize() silently returns the original NON-UNIT
// vector, which is not what the guard's comment promises and not what any
// caller expects.
//
// It bites on both backends, at different magnitudes. On an Apple GPU, which
// flushes subnormals in every math mode, it starts at |v| ~ 1e-19. On CUDA,
// with nvcc's default gradual underflow, it starts around |v| ~ 1e-23. Either
// way the failure is silent: no NaN, no assert, just a vector that is not a
// unit vector being used as one. There are 39 normalize call sites, including
// three collision-mesh contact normals in contact.cu.
//
// Usage (from ppf-cts-compute/cuda/tests, which is where this Makefile runs;
// its sibling rules name ../kernels/reduce.cu, so the directory matters):
//   make test_normalize_underflow
//   ./test_normalize_underflow
// or both underflow gates at once:  make test-underflow
//
// The gate asserts three things:
//   1. the fix works: a small non-zero vector comes back unit;
//   2. it costs nothing: the fast path is BIT-IDENTICAL to the pre-fix code,
//      so the solver's normal-range arithmetic is untouched;
//   3. it never regresses: no input that used to normalize correctly stops.

#include "linalg/smat.hpp"

// Vec3f lives in data.hpp, which drags in the whole scene ABI. SVec<float,3>
// is exactly what it aliases, so using it directly keeps this gate dependent
// only on the header under test. Both it and la_sqrt sit in namespace linalg;
// the rest of the tree sees them unqualified because data.hpp opens that
// namespace, and this file deliberately does not.
using V3 = linalg::SVec<float, 3>;
using linalg::la_sqrt;

#include <cmath>
#include <cstring>

// The two CUDA bit-cast intrinsics this file used, spelled for a host: a memcpy
// is what both compile to and is the one form that is not undefined behaviour.
static inline float as_float(unsigned bits) {
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}
static inline unsigned as_uint(float value) {
    unsigned out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}
#include <cstdio>

// The pre-fix spelling, kept here and ONLY here so the gate can compare
// against it. Never reintroduce this into the solver.
inline void normalize_pre_fix(float *v, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i)
        acc = acc + v[i] * v[i];
    float len = la_sqrt(acc);
    if (len > 0.0f)
        for (int i = 0; i < n; ++i)
            v[i] = v[i] / len;
}

inline float len3(const float *v) {
    // Computed through a rescale so the measurement itself cannot underflow.
    float mx = 0.0f;
    for (int i = 0; i < 3; ++i) {
        float a = v[i] < 0.0f ? -v[i] : v[i];
        if (a > mx)
            mx = a;
    }
    if (!(mx > 0.0f))
        return 0.0f;
    float acc = 0.0f;
    for (int i = 0; i < 3; ++i) {
        float q = v[i] / mx;
        acc = acc + q * q;
    }
    return mx * la_sqrt(acc);
}

// The loop this replaces was a `__global__` kernel, one thread per row; a lane
// index became a loop counter and nothing else changed.
void sweep_rows(const float *mag, int n, float *shipped_len,
                float *prefix_len) {
    for (int i = 0; i < n; ++i) {
    V3 a;
    a[0] = mag[i];
    a[1] = mag[i];
    a[2] = mag[i];
    float b[3] = {mag[i], mag[i], mag[i]};
    a.normalize();
    normalize_pre_fix(b, 3);
    float av[3] = {a[0], a[1], a[2]};
        shipped_len[i] = len3(av);
        prefix_len[i] = len3(b);
    }
}

void equivalence(unsigned seed, long long n, long long *fastpath_diff,
                 long long *regressed, long long *fixed) {
    for (long long i = 0; i < n; ++i) {
    float v[3] = {as_float((unsigned)(i * 2654435761u + seed)),
                  as_float((unsigned)(i * 2246822519u + seed * 40503u)),
                  as_float((unsigned)(i * 3266489917u + seed * 2654u))};
    // `continue`, NOT `return`. In the kernel this was a thread declining to
    // do its own row; in a loop a `return` would abandon every remaining pair
    // and the sample would silently be as long as its first non-finite entry.
    if (!std::isfinite(v[0]) || !std::isfinite(v[1]) ||
        !std::isfinite(v[2])) {
        continue;
    }

    float sq = 0.0f;
    for (int j = 0; j < 3; ++j)
        sq = sq + v[j] * v[j];

    V3 a;
    a[0] = v[0]; a[1] = v[1]; a[2] = v[2];
    float b[3] = {v[0], v[1], v[2]};
    a.normalize();
    normalize_pre_fix(b, 3);

    const bool same = as_uint(a[0]) == as_uint(b[0]) &&
                      as_uint(a[1]) == as_uint(b[1]) &&
                      as_uint(a[2]) == as_uint(b[2]);
    // On the fast path the two MUST agree bit for bit.
    if (!same && sq >= 1.17549435e-38f)  // the shipped fast-path condition
        *fastpath_diff += 1;

    const bool nonzero = v[0] != 0.0f || v[1] != 0.0f || v[2] != 0.0f;
    if (nonzero) {
        float av[3] = {a[0], a[1], a[2]};
        const float ls = len3(av), lp = len3(b);
        const bool shipped_ok = fabsf(ls - 1.0f) < 1e-3f;
        const bool prefix_ok = fabsf(lp - 1.0f) < 1e-3f;
        if (prefix_ok && !shipped_ok)
            *regressed += 1;
        if (!prefix_ok && shipped_ok)
            *fixed += 1;
        }
    }
}

int main() {
    const float mag_h[] = {1e-3f, 1e-10f, 1e-19f, 1e-20f, 1e-22f,
                           1e-23f, 1e-25f, 1e-30f, 0.0f};
    const int N = sizeof(mag_h) / sizeof(mag_h[0]);
    float shipped[N], prefix[N];
    sweep_rows(mag_h, N, shipped, prefix);

    printf("Isotropic vectors (m, m, m). A non-zero input must come back with\n");
    printf("length 1; the all-zero input must stay 0.\n\n");
    printf("  %-10s %12s %14s %14s  %s\n", "component", "squaredNorm",
           "SHIPPED |v|", "pre-fix |v|", "");
    int fail = 0, prefix_failed = 0;
    for (int i = 0; i < N; ++i) {
        const double sq = 3.0 * (double)mag_h[i] * (double)mag_h[i];
        const bool want_unit = mag_h[i] != 0.0f;
        const bool s_ok = want_unit ? fabsf(shipped[i] - 1.0f) < 1e-3f
                                    : shipped[i] == 0.0f;
        const bool p_ok = want_unit ? fabsf(prefix[i] - 1.0f) < 1e-3f
                                    : prefix[i] == 0.0f;
        const char *note = "";
        if (!s_ok) { note = "SHIPPED FORM FAILED"; ++fail; }
        else if (!p_ok) { note = "pre-fix failed here, shipped form is correct"; ++prefix_failed; }
        printf("  %-10.0e %12.4g %14.6f %14.6f  %s\n", mag_h[i], sq,
               shipped[i], prefix[i], note);
    }
    printf("\n  shipped form failed %d of %d; pre-fix form failed %d of %d\n\n",
           fail, N, prefix_failed + fail, N);

    // THE SAME 200 MILLION VECTORS THE DEVICE VERSION SAMPLED. The count is
    // not reduced: the sample size is the coverage, and shrinking it to make a
    // test quick is how a gate stops reaching the band it was written for.
    long long fp = 0, rg = 0, fx = 0;
    const long long M = 200000000LL;
    equivalence(999u, M, &fp, &rg, &fx);
    printf("equivalence over %lld random finite 3-vectors:\n", M);
    printf("  bit differences on the FAST PATH (squaredNorm > 0) : %lld  <-- must be 0\n", fp);
    printf("  inputs the pre-fix form normalized and this one does not : %lld  <-- must be 0\n", rg);
    printf("  inputs this form normalizes and the pre-fix form did not : %lld\n", fx);

    const bool ok = (fail == 0) && (fp == 0) && (rg == 0) && (prefix_failed > 0);
    printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
