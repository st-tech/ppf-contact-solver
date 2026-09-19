// File: crates/ppf-cts-solver/tests/kernels/distance_coeff.cpp
// Code: Claude Code and Codex
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Gate for the closest-point routines in contact/distance.hpp and for the CCD
// that sizes its conservative advance from the distances they report.
//
// The distance between two primitives is what ACCD divides by to bound how far
// the sweep may advance before they can touch. Reporting it too LARGE is the
// unsafe direction: the advance is then sized for a separation the pair does
// not have, so the sweep can step across the true time of impact and certify a
// trajectory that passes through. Reporting it too small only costs iterations.
// The coefficients name points ON the primitives, so under-reporting is
// structurally impossible and the property required is exactness, not a
// one-sided bound.
//
// Two regimes matter, one per primitive pair. Two edges nearly parallel: a
// stationary point recovered from the Gram expression a e - b^2 subtracts two
// products agreeing to within the SQUARED sine of the angle between them, so in
// single precision it keeps no digits below roughly 3e-4 rad. A point against a
// triangle with one wide interior angle: classifying by the sign of the
// barycentric coordinates splits the plane by the triangle's edge lines, which
// separates the three edge regions only when the triangle is acute.
//
// THE REFERENCE SHARES NOTHING WITH THE ROUTINE IT CHECKS. Every reference here
// is computed in double, and the routines under test are float throughout, so
// the two cannot share a blind spot. Each case generator also writes out the
// geometry it built, so the comparison reads the exact numbers that were
// evaluated instead of reconstructing them.
//
// The references share no implementation with the routines they check. g(s),
// the squared distance from A(s) to the SEGMENT B, is the squared distance from
// a point to a convex set composed with an affine map, hence convex on [0, 1];
// its derivative is 2 (A(s) - C(s)) . r0, with C(s) held fixed by the envelope
// theorem. Bisecting that derivative converges to the true minimizer from a
// different formulation. The triangle reference is exhaustive instead: the
// perpendicular projection when it lands inside, otherwise the smallest of the
// three clamped edge projections, which enumerates every feature.
//
// WHY THIS IS A HOST PROGRAM. It was a `.cu` that launched six kernels, and
// every one of them ran the routine under test at one case per thread with no
// cooperation of any kind: no shared memory, no barrier, no warp intrinsic, no
// atomic. The grid was a loop bound and nothing else. What the kernels computed
// is float arithmetic out of headers a host compiler reads the same way, and
// the entire reference half already ran on the host in double, so it is what
// dominated the run either way.
//
// THE THREAD DECOMPOSITION IS REPRODUCED RATHER THAN FLATTENED, and only for
// the two gates that need it: `run_sound` and `run_sound_tangential` seed a
// per-thread RNG from the thread index and then walk a grid-stride loop, so the
// case a given draw belongs to is a function of the launch geometry. Reproducing
// `<<<32, 256>>>` exactly is what keeps this test looking at the same twenty
// thousand sweeps the CUDA one did. The other four generators are pure functions
// of the case index, so a flat loop over it is identical by construction.

// `data.hpp` FIRST. A shared header takes its type vocabulary from whichever
// header the includer pulled in ahead of it, and neither of the two below
// brings its own.
#include "data.hpp"

#include "contact/accd.hpp"
#include "contact/distance.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

// ---------------------------------------------------------------------------
// Case generators. Each writes out the geometry it built alongside the
// routine's answer, so the comparison reads the exact numbers that were
// evaluated rather than reconstructing them. That is also why `sinf` and `cosf`
// are spelled plainly here where the CUDA original spelled `__sinf` and
// `__cosf`: that spelling existed to keep the double-precision argument
// reduction on the library slow path out of the DEVICE binary, which this
// program does not have, and accuracy is irrelevant either way because whatever
// geometry comes out is what the reference is computed against.
// ---------------------------------------------------------------------------

enum : int {
    N_ANG = 60,
    N_OFF = 7,
    N_LEN = 5,
    N_SHIFT = 5,
    N_EE = N_ANG * N_OFF * N_LEN * N_SHIFT,
    N_APEX = 12,
    N_GRID = 13,
    N_POFF = 6,
    N_PT = N_APEX * N_GRID * N_GRID * N_POFF,
    N_APEX5 = 14,
    N_GRID5 = 21,
    N_BARY = N_APEX5 * N_GRID5 * N_GRID5,
    N_SOUND = 20000,
    N_BINS = 10,
    N_SAMPLES = 400,
    N_BLOCKS = 32,
    N_THREADS = 256,
    // The launch geometry the two RNG gates were run under, reproduced so each
    // draw lands in the case it landed in there.
    N_LANES = N_BLOCKS * N_THREADS
};

// Two segments: A fixed on the x axis, B rotated by `ang`, scaled by `len`,
// shifted along x and lifted out of plane by `off`.
static inline void gen_ee(int c, float *a0, float *a1, float *b0, float *b1) {
    int si = c % N_SHIFT;
    int li = (c / N_SHIFT) % N_LEN;
    int oi = (c / (N_SHIFT * N_LEN)) % N_OFF;
    int ai = c / (N_SHIFT * N_LEN * N_OFF);
    float ang = 1.0e-6f * powf(10.0f, 6.2f * float(ai) / float(N_ANG - 1));
    float off = 1.0e-6f * powf(10.0f, 5.0f * float(oi) / float(N_OFF - 1));
    float len = powf(10.0f, 2.0f * float(li) / float(N_LEN - 1) - 1.0f);
    float shift = -0.8f + 0.4f * float(si);
    float ca = cosf(ang), sa = sinf(ang);
    a0[0] = -0.5f; a0[1] = 0.0f; a0[2] = 0.0f;
    a1[0] = 0.5f;  a1[1] = 0.0f; a1[2] = 0.0f;
    b0[0] = shift - 0.5f * len * ca; b0[1] = -0.5f * len * sa; b0[2] = off;
    b1[0] = shift + 0.5f * len * ca; b1[1] = 0.5f * len * sa;  b1[2] = off;
}

// A point against a triangle whose apex height shrinks geometrically, taking
// the widest interior angle from about 90 degrees to within a rounding of 180.
static inline void gen_pt(int c, float *p, float *t0, float *t1, float *t2) {
    int oi = c % N_POFF;
    int gy = (c / N_POFF) % N_GRID;
    int gx = (c / (N_POFF * N_GRID)) % N_GRID;
    int ai = c / (N_POFF * N_GRID * N_GRID);
    float h = powf(10.0f, -0.5f * float(ai));
    t0[0] = -0.5f; t0[1] = 0.0f; t0[2] = 0.0f;
    t1[0] = 0.5f;  t1[1] = 0.0f; t1[2] = 0.0f;
    t2[0] = 0.12f; t2[1] = h;    t2[2] = 0.0f;
    p[0] = -1.4f + 2.8f * float(gx) / float(N_GRID - 1);
    p[1] = -0.9f + 2.2f * float(gy) / float(N_GRID - 1);
    p[2] = 1.0e-6f * powf(10.0f, 5.0f * float(oi) / float(N_POFF - 1));
}

// The same family, on a grid hugging the triangle, because a classification is
// only interesting near the boundary it is deciding.
static inline void gen_bary(int c, float *p, float *t0, float *t1, float *t2) {
    int gy = c % N_GRID5;
    int gx = (c / N_GRID5) % N_GRID5;
    int ai = c / (N_GRID5 * N_GRID5);
    float h = powf(10.0f, -0.5f * float(ai));
    t0[0] = -0.5f; t0[1] = 0.0f; t0[2] = 0.0f;
    t1[0] = 0.5f;  t1[1] = 0.0f; t1[2] = 0.0f;
    t2[0] = 0.12f; t2[1] = h;    t2[2] = 0.0f;
    p[0] = -0.8f + 1.6f * float(gx) / float(N_GRID5 - 1);
    p[1] = -0.4f * h + 1.8f * h * float(gy) / float(N_GRID5 - 1);
    p[2] = 1.0e-5f;
}

// The configuration that produced a penetration in examples/large-animals: a
// vertex outside a sliver face whose widest interior angle is 153.56 degrees.
static inline void gen_regression(float *p, float *t0, float *t1, float *t2) {
    p[0] = -1.493152976e-01f; p[1] = -1.083186531e+01f; p[2] = -6.033576727e-01f;
    t0[0] = -1.840408295e-01f; t0[1] = -1.082791328e+01f; t0[2] = -5.791632533e-01f;
    t1[0] = -1.123973131e-01f; t1[1] = -1.081860638e+01f; t1[2] = -6.160872579e-01f;
    t2[0] = -1.463371068e-01f; t2[1] = -1.083091545e+01f; t2[2] = -6.040185690e-01f;
}

// ---------------------------------------------------------------------------
// References, in double. They see none of the code under test.
// ---------------------------------------------------------------------------

struct D3 {
    double x, y, z;
};

static inline D3 d3(const float *v) {
    return {double(v[0]), double(v[1]), double(v[2])};
}
static inline D3 sub(const D3 &a, const D3 &b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
static inline D3 mad(const D3 &a, double s, const D3 &d) {
    return {a.x + s * d.x, a.y + s * d.y, a.z + s * d.z};
}
static inline double dot(const D3 &a, const D3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
static inline D3 cross(const D3 &a, const D3 &b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x};
}
static inline double len(const D3 &a) { return sqrt(dot(a, a)); }

// Closest point of segment [b0, b0 + r1] to p, by exact clamped projection.
// A point-to-SEGMENT projection carries no near-parallel cancellation.
static D3 closest_on_seg(const D3 &p, const D3 &b0, const D3 &r1, double e) {
    if (!(e > 0.0)) {
        return b0;
    }
    double t = dot(sub(p, b0), r1) / e;
    t = t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
    return mad(b0, t, r1);
}

// True minimum distance between segments [a0, a0 + r0] and [b0, b0 + r1], by
// bisecting the convex derivative.
static double ref_edge_edge(const D3 &a0, const D3 &r0, const D3 &b0,
                            const D3 &r1) {
    double e = dot(r1, r1);
    auto deriv = [&](double s) {
        D3 p = mad(a0, s, r0);
        return dot(sub(p, closest_on_seg(p, b0, r1, e)), r0);
    };
    double s;
    if (deriv(0.0) >= 0.0) {
        s = 0.0;
    } else if (deriv(1.0) <= 0.0) {
        s = 1.0;
    } else {
        double lo = 0.0, hi = 1.0;
        for (int i = 0; i < 100; ++i) {
            double mid = 0.5 * (lo + hi);
            if (deriv(mid) < 0.0) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        s = 0.5 * (lo + hi);
    }
    D3 p = mad(a0, s, r0);
    return len(sub(p, closest_on_seg(p, b0, r1, e)));
}

// True minimum distance from a point to a triangle, by exhaustive feature test.
static double ref_point_triangle(const D3 &p, const D3 &t0, const D3 &t1,
                                 const D3 &t2) {
    D3 r0 = sub(t1, t0), r1 = sub(t2, t0), d = sub(p, t0);
    D3 n = cross(r0, r1);
    double nn = dot(n, n);
    if (nn > 0.0) {
        double u = dot(cross(d, r1), n) / nn;
        double v = dot(cross(r0, d), n) / nn;
        if (u >= 0.0 && v >= 0.0 && u + v <= 1.0) {
            return fabs(dot(d, n)) / sqrt(nn);
        }
    }
    double best = 1.0e300;
    const D3 *v0[3] = {&t0, &t1, &t2};
    const D3 *v1[3] = {&t1, &t2, &t0};
    for (int k = 0; k < 3; ++k) {
        D3 r = sub(*v1[k], *v0[k]);
        double e = dot(r, r);
        double s = e > 0.0 ? dot(sub(p, *v0[k]), r) / e : 0.0;
        s = s < 0.0 ? 0.0 : (s > 1.0 ? 1.0 : s);
        double dd = len(sub(p, mad(*v0[k], s, r)));
        if (dd < best) {
            best = dd;
        }
    }
    return best;
}

// Barycentric coordinates of p projected onto the triangle's plane.
static void ref_bary(const D3 &p, const D3 &t0, const D3 &t1, const D3 &t2,
                     double *out) {
    D3 r0 = sub(t1, t0), r1 = sub(t2, t0), d = sub(p, t0);
    D3 n = cross(r0, r1);
    double nn = dot(n, n);
    double u = dot(cross(d, r1), n) / nn;
    double v = dot(cross(r0, d), n) / nn;
    out[0] = 1.0 - u - v;
    out[1] = u;
    out[2] = v;
}

// The distance a set of barycentric weights actually realizes, evaluated in
// double from the float weights the routine produced. This measures what the
// routine's ANSWER is worth, independently of the arithmetic that formed it.
static double realized_ee(const float *a0, const float *a1, const float *b0,
                          const float *b1, const float *w) {
    D3 A0 = d3(a0), A1 = d3(a1), B0 = d3(b0), B1 = d3(b1);
    D3 p = mad(mad({0, 0, 0}, double(w[0]), A0), double(w[1]), A1);
    D3 q = mad(mad({0, 0, 0}, double(w[2]), B0), double(w[3]), B1);
    return len(sub(p, q));
}

static double realized_pt(const float *t0, const float *t1, const float *t2,
                          const float *p, const float *w) {
    D3 T0 = d3(t0), T1 = d3(t1), T2 = d3(t2);
    D3 q = mad(mad(mad({0, 0, 0}, double(w[0]), T0), double(w[1]), T1),
               double(w[2]), T2);
    return len(sub(d3(p), q));
}

// The pair spans about one unit, so single precision resolves a difference
// vector to about that times the unit round-off; an excess at that level is the
// arithmetic's own resolution rather than a property of the algorithm. The
// relative allowance covers the opposite end, where the true separation is
// itself near that resolution and a small absolute excess is a large ratio. A
// case counts against a gate only when it clears BOTH.
static inline double abs_allow() { return 16.0 * 1.19209290e-7; }
static inline double ratio_allow() { return 1.01; }

// ---------------------------------------------------------------------------
// The passes under test. Each runs the routine and writes its raw output.
// ---------------------------------------------------------------------------

static void run_ee(float *geom, float *out) {
    for (int c = 0; c < N_EE; ++c) {
        float a0[3], a1[3], b0[3], b1[3];
        gen_ee(c, a0, a1, b0, b1);
        for (int k = 0; k < 3; ++k) {
            geom[12 * c + k] = a0[k];
            geom[12 * c + 3 + k] = a1[k];
            geom[12 * c + 6 + k] = b0[k];
            geom[12 * c + 9 + k] = b1[k];
        }
        Vec4f w = proximity::edge_edge_distance_coeff<float, float>(
            Vec3f(a0[0], a0[1], a0[2]), Vec3f(a1[0], a1[1], a1[2]),
            Vec3f(b0[0], b0[1], b0[2]), Vec3f(b1[0], b1[1], b1[2]));
        for (int k = 0; k < 4; ++k) {
            out[4 * c + k] = w[k];
        }
    }
}

static void run_pt(float *geom, float *out) {
    for (int c = 0; c < N_PT; ++c) {
        float p[3], t0[3], t1[3], t2[3];
        gen_pt(c, p, t0, t1, t2);
        for (int k = 0; k < 3; ++k) {
            geom[12 * c + k] = p[k];
            geom[12 * c + 3 + k] = t0[k];
            geom[12 * c + 6 + k] = t1[k];
            geom[12 * c + 9 + k] = t2[k];
        }
        Vec3f w =
            proximity::point_triangle_distance_coeff_unclassified<float, float>(
                Vec3f(p[0], p[1], p[2]), Vec3f(t0[0], t0[1], t0[2]),
                Vec3f(t1[0], t1[1], t1[2]), Vec3f(t2[0], t2[1], t2[2]));
        for (int k = 0; k < 3; ++k) {
            out[3 * c + k] = w[k];
        }
    }
}

static void run_bary(float *geom, float *out) {
    for (int c = 0; c < N_BARY; ++c) {
        float p[3], t0[3], t1[3], t2[3];
        gen_bary(c, p, t0, t1, t2);
        for (int k = 0; k < 3; ++k) {
            geom[12 * c + k] = p[k];
            geom[12 * c + 3 + k] = t0[k];
            geom[12 * c + 6 + k] = t1[k];
            geom[12 * c + 9 + k] = t2[k];
        }
        Vec3f w = proximity::point_triangle_distance_coeff<float, float>(
            Vec3f(p[0], p[1], p[2]), Vec3f(t0[0], t0[1], t0[2]),
            Vec3f(t1[0], t1[1], t1[2]), Vec3f(t2[0], t2[1], t2[2]));
        for (int k = 0; k < 3; ++k) {
            out[3 * c + k] = w[k];
        }
    }
}

static void run_regression(float *geom, float *out) {
    float p[3], t0[3], t1[3], t2[3];
    gen_regression(p, t0, t1, t2);
    for (int k = 0; k < 3; ++k) {
        geom[k] = p[k];
        geom[3 + k] = t0[k];
        geom[6 + k] = t1[k];
        geom[9 + k] = t2[k];
    }
    Vec3f w =
        proximity::point_triangle_distance_coeff_unclassified<float, float>(
            Vec3f(p[0], p[1], p[2]), Vec3f(t0[0], t0[1], t0[2]),
            Vec3f(t1[0], t1[1], t1[2]), Vec3f(t2[0], t2[1], t2[2]));
    for (int k = 0; k < 3; ++k) {
        out[k] = w[k];
    }
}

static inline unsigned rng(unsigned &s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}
static inline float uf(unsigned &s) {
    return float(rng(s) & 0xFFFFFF) / float(0xFFFFFF);
}
static inline float sf(unsigned &s) { return 2.0f * uf(s) - 1.0f; }

// Constructed crossings: edge B lies along the x axis and edge A translates
// from +h to -h straight through B's plane, so a configuration of zero
// separation exists somewhere in the sweep by construction. Rejection sampling
// was tried and produced four usable cases in twenty thousand draws.
//
// The geometry is written out as the coordinate differences the sweep actually
// presents, formed by subtraction so an absolute coordinate never becomes a
// float. The comparison then sees exactly the rounded geometry that was
// evaluated, and quantization cannot masquerade as a missed crossing.
static void run_sound(float *geom, float *toi_out, float *ang_out) {
    for (int tid = 0; tid < N_LANES; ++tid) {
        unsigned seed = 12345u + 7919u * unsigned(tid);
        ParamSet param;
        param.line_search_max_t = 1.25f;
        param.ccd_eps = 1e-7f;

        for (int c = tid; c < N_SOUND; c += N_LANES) {
            float ang = 1.0e-4f * powf(10.0f, 4.0f * uf(seed));
            float h = 1.0e-4f * powf(10.0f, 3.0f * uf(seed));
            float length = powf(10.0f, 2.0f * sf(seed));
            float shift = 0.8f * sf(seed);
            ang_out[c] = -1.0f;

            // The coordinate type has a bounded domain, so a case reaching
            // outside it is not one the solver can ever be handed. Enforced
            // against the parameters rather than by choosing constants that
            // happen to fit.
            const float kDomainLimit = 8.0f;
            if (fabsf(shift) + 0.5f * length > kDomainLimit) {
                continue;
            }
            float ca = cosf(ang), sa = sinf(ang);
            Vec3f b0(-1.0f, 0.0f, 0.0f), b1(1.0f, 0.0f, 0.0f);
            Vec3f a0(shift - 0.5f * length * ca, -0.5f * length * sa, h);
            Vec3f a1(shift + 0.5f * length * ca, 0.5f * length * sa, h);
            Vec3f a0e(shift - 0.5f * length * ca, -0.5f * length * sa, -h);
            Vec3f a1e(shift + 0.5f * length * ca, 0.5f * length * sa, -h);

            Vec3f g[5] = {(a0 - b0).template cast<float>(),
                          (a1 - b0).template cast<float>(),
                          (a0e - b0).template cast<float>(),
                          (a1e - b0).template cast<float>(),
                          (b1 - b0).template cast<float>()};
            for (int i = 0; i < 5; ++i) {
                for (int k = 0; k < 3; ++k) {
                    geom[15 * c + 3 * i + k] = g[i][k];
                }
            }
            // The overlap record is an out-parameter rather than the device
            // global it used to be, because a mutable device global has no
            // Metal equivalent. This test only reads the returned toi, so the
            // record is written and discarded; it still has to be supplied,
            // since ccd_helper dereferences it unconditionally.
            accd::OverlapInfo overlap{-1.0f, -1.0f};
            toi_out[c] = accd::edge_edge_ccd(
                a0, a1, b0, b1, a0e, a1e, b0, b1, 0.0f, 1e-3f,
                param.line_search_max_t, param.ccd_eps, &overlap);
            ang_out[c] = ang;
        }
    }
}

// Gate 6's sweeps. Same crossing geometry as run_sound, plus a large TANGENTIAL
// glide laid on top of the head-on descent, which is the regime the directional
// advance certificate in accd.hpp exists for: the direction-agnostic Lipschitz
// bound sees the whole glide speed while the surfaces approach only at the
// descent rate, so the ratio between the two is `glide / (2 h)` and reaches
// three orders of magnitude here. A certificate that read the glide as approach
// would merely be slow; one that read the approach as glide would certify a
// trajectory straight through the contact, which is what this gate rules out.
static void run_sound_tangential(float *geom, float *toi_out, float *ang_out) {
    for (int tid = 0; tid < N_LANES; ++tid) {
        unsigned seed = 60077u + 7919u * unsigned(tid);
        ParamSet param;
        param.line_search_max_t = 1.25f;
        param.ccd_eps = 1e-7f;

        for (int c = tid; c < N_SOUND; c += N_LANES) {
            float ang = 1.0e-4f * powf(10.0f, 4.0f * uf(seed));
            float h = 1.0e-4f * powf(10.0f, 3.0f * uf(seed));
            float length = powf(10.0f, 2.0f * sf(seed));
            float shift = 0.8f * sf(seed);
            // Glide spans "comparable to the descent" through "a thousand times
            // it", so the gate covers the whole range over which the two bounds
            // diverge.
            float glide = h * powf(10.0f, 3.0f * uf(seed));
            float gdir = 6.2831853f * uf(seed);
            float gx = glide * cosf(gdir), gy = glide * sinf(gdir);
            ang_out[c] = -1.0f;

            const float kDomainLimit = 8.0f;
            if (fabsf(shift) + 0.5f * length + glide > kDomainLimit) {
                continue;
            }
            float ca = cosf(ang), sa = sinf(ang);
            Vec3f b0(-1.0f, 0.0f, 0.0f), b1(1.0f, 0.0f, 0.0f);
            Vec3f a0(shift - 0.5f * length * ca, -0.5f * length * sa, h);
            Vec3f a1(shift + 0.5f * length * ca, 0.5f * length * sa, h);
            Vec3f a0e(shift - 0.5f * length * ca + gx,
                       -0.5f * length * sa + gy, -h);
            Vec3f a1e(shift + 0.5f * length * ca + gx,
                       0.5f * length * sa + gy, -h);

            Vec3f g[5] = {(a0 - b0).template cast<float>(),
                          (a1 - b0).template cast<float>(),
                          (a0e - b0).template cast<float>(),
                          (a1e - b0).template cast<float>(),
                          (b1 - b0).template cast<float>()};
            for (int i = 0; i < 5; ++i) {
                for (int k = 0; k < 3; ++k) {
                    geom[15 * c + 3 * i + k] = g[i][k];
                }
            }
            accd::OverlapInfo overlap{-1.0f, -1.0f};
            toi_out[c] = accd::edge_edge_ccd(
                a0, a1, b0, b1, a0e, a1e, b0, b1, 0.0f, 1e-3f,
                param.line_search_max_t, param.ccd_eps, &overlap);
            ang_out[c] = ang;
        }
    }
}

int main() {
    int failures = 0;
    const float kMaxT = 1.25f;

    // ----------------------------------------------------------- gate 1
    {
        std::vector<float> ee(4 * N_EE, 0.0f), ee_geom(12 * N_EE, 0.0f);
        run_ee(ee_geom.data(), ee.data());

        double worst_ratio = 0.0, worst_excess = 0.0;
        unsigned n_bad = 0u;
        for (int c = 0; c < N_EE; ++c) {
            const float *a0 = &ee_geom[12 * c], *a1 = &ee_geom[12 * c + 3];
            const float *b0 = &ee_geom[12 * c + 6], *b1 = &ee_geom[12 * c + 9];
            double got = realized_ee(a0, a1, b0, b1, &ee[4 * c]);
            D3 A0 = d3(a0), B0 = d3(b0);
            double ref = ref_edge_edge(A0, sub(d3(a1), A0), B0, sub(d3(b1), B0));
            double excess = got - ref;
            double ratio = ref > 0.0 ? got / ref : 1.0;
            if (excess > worst_excess) {
                worst_excess = excess;
            }
            if (ratio > worst_ratio) {
                worst_ratio = ratio;
            }
            // Judged per case: the worst excess and the worst ratio are
            // generally attained by different pairs, so comparing the two
            // maxima against the two allowances would clear a pair violating
            // neither alone while violating both together.
            if (excess > abs_allow() && ratio > ratio_allow()) {
                ++n_bad;
            }
        }
        printf("=== gate 1: edge-edge reported vs truth (%d pairs) ===\n", N_EE);
        printf("  worst ratio    reported/true = %.6f\n", worst_ratio);
        printf("  worst absolute excess        = %.3e\n", worst_excess);
        printf("  pairs clearing both          = %u\n", n_bad);
        printf("  %s\n\n", n_bad == 0u ? "PASS"
                                       : "FAIL: the advance can be sized for a "
                                         "separation the pair does not have");
        if (n_bad) {
            ++failures;
        }
    }

    // ----------------------------------------------------------- gate 2
    {
        std::vector<float> geom(15 * N_SOUND, 0.0f), toi(N_SOUND, 0.0f),
            angs(N_SOUND, -1.0f);
        run_sound(geom.data(), toi.data(), angs.data());

        unsigned bin_tot[N_BINS] = {0}, bin_miss[N_BINS] = {0};
        for (int c = 0; c < N_SOUND; ++c) {
            if (angs[c] < 0.0f) {
                continue;
            }
            const float *g = &geom[15 * c];
            D3 A0 = d3(g), A1 = d3(g + 3), A0E = d3(g + 6), A1E = d3(g + 9),
               R1 = d3(g + 12);
            D3 zero{0.0, 0.0, 0.0};
            auto sep = [&](double u) {
                D3 s0 = mad(A0, u, sub(A0E, A0));
                D3 s1 = mad(A1, u, sub(A1E, A1));
                return ref_edge_edge(s0, sub(s1, s0), zero, R1);
            };
            double d_start = sep(0.0), dmin = d_start;
            for (int i = 1; i <= N_SAMPLES; ++i) {
                double d = sep(double(i) / double(N_SAMPLES));
                if (d < dmin) {
                    dmin = d;
                }
            }
            // Admit only cases that start apart and genuinely reach contact.
            if (!(d_start > 1e-6) || dmin > 1e-7) {
                continue;
            }
            int bin = int(float(N_BINS) * angs[c]);
            bin = bin < 0 ? 0 : (bin >= N_BINS ? N_BINS - 1 : bin);
            ++bin_tot[bin];
            if (toi[c] >= kMaxT) {
                ++bin_miss[bin];
            }
        }
        unsigned tot = 0u, miss = 0u;
        for (int i = 0; i < N_BINS; ++i) {
            tot += bin_tot[i];
            miss += bin_miss[i];
        }
        printf("=== gate 2: sweeps that genuinely collide (%u constructed) ===\n",
               tot);
        printf("  certified as collision-free : %u\n", miss);
        for (int i = 0; i < N_BINS; ++i) {
            printf("    [%.1f, %.1f) rad: %6u crossings, %5u certified  %6.2f%%\n",
                   0.1 * i, 0.1 * (i + 1), bin_tot[i], bin_miss[i],
                   bin_tot[i] ? 100.0 * double(bin_miss[i]) / double(bin_tot[i])
                              : 0.0);
        }
        // No allowance at all: one certified crossing is one trajectory the
        // line search would have accepted through a collision.
        bool g2 = (tot > 0u) && (miss == 0u);
        printf("  %s\n", tot == 0u ? "FAIL: no crossing was constructed"
                                   : (miss == 0u ? "PASS"
                                                 : "FAIL: a colliding trajectory "
                                                   "was certified"));
        if (!g2) {
            ++failures;
        }
    }

    // ----------------------------------------------------------- gate 3
    {
        std::vector<float> pt(3 * N_PT, 0.0f), pt_geom(12 * N_PT, 0.0f);
        run_pt(pt_geom.data(), pt.data());

        double r3 = 0.0, e3 = 0.0;
        unsigned bad3 = 0u;
        for (int c = 0; c < N_PT; ++c) {
            const float *p = &pt_geom[12 * c], *t0 = &pt_geom[12 * c + 3];
            const float *t1 = &pt_geom[12 * c + 6], *t2 = &pt_geom[12 * c + 9];
            double got = realized_pt(t0, t1, t2, p, &pt[3 * c]);
            double ref = ref_point_triangle(d3(p), d3(t0), d3(t1), d3(t2));
            double excess = got - ref, ratio = ref > 0.0 ? got / ref : 1.0;
            if (excess > e3) {
                e3 = excess;
            }
            if (ratio > r3) {
                r3 = ratio;
            }
            if (excess > abs_allow() && ratio > ratio_allow()) {
                ++bad3;
            }
        }
        printf("\n=== gate 3: point vs triangle, reported vs truth (%d cases) ===\n",
               N_PT);
        printf("  worst ratio    reported/true = %.6f\n", r3);
        printf("  worst absolute excess        = %.3e\n", e3);
        printf("  cases clearing both          = %u\n", bad3);
        printf("  %s\n\n", bad3 == 0u ? "PASS"
                                      : "FAIL: the advance can be sized for a "
                                        "separation the point does not have");
        if (bad3) {
            ++failures;
        }
    }

    // ----------------------------------------------------------- gate 4
    {
        std::vector<float> rg(3, 0.0f), rg_geom(12, 0.0f);
        run_regression(rg_geom.data(), rg.data());
        const float *p = &rg_geom[0], *t0 = &rg_geom[3], *t1 = &rg_geom[6],
                    *t2 = &rg_geom[9];
        double got = realized_pt(t0, t1, t2, p, rg.data());
        double ref = ref_point_triangle(d3(p), d3(t0), d3(t1), d3(t2));
        double ratio = ref > 0.0 ? got / ref : 1.0;
        printf("=== gate 4: the large-animals sliver (153.56 deg) ===\n");
        printf("  reported %.6e   true %.6e   ratio %.4f\n", got, ref, ratio);
        bool g4 = ratio <= ratio_allow();
        printf("  %s\n\n", g4 ? "PASS"
                              : "FAIL: over-reports on the geometry that "
                                "penetrated");
        if (!g4) {
            ++failures;
        }
    }

    // ----------------------------------------------------------- gate 5
    //
    // point_triangle_distance_coeff returns the unclamped projection, and its
    // callers consume only the verdict it implies: they proceed on
    // `c.minCoeff() > 0`. So what matters at those call sites is whether that
    // comparison lands on the correct side. A disagreement counts only where
    // the reference is UNAMBIGUOUS, since a projection genuinely on the
    // boundary is entitled to be called either way.
    {
        std::vector<float> bary(3 * N_BARY, 0.0f), bary_geom(12 * N_BARY, 0.0f);
        run_bary(bary_geom.data(), bary.data());

        double err5 = 0.0;
        unsigned dis5 = 0u;
        float bad_apex = 0.0f, bad_px = 0.0f, bad_py = 0.0f;
        const double kTol = 1.0e-4;
        for (int c = 0; c < N_BARY; ++c) {
            const float *p = &bary_geom[12 * c], *t0 = &bary_geom[12 * c + 3];
            const float *t1 = &bary_geom[12 * c + 6],
                        *t2 = &bary_geom[12 * c + 9];
            double ref[3];
            ref_bary(d3(p), d3(t0), d3(t1), d3(t2), ref);
            const float *w = &bary[3 * c];
            for (int k = 0; k < 3; ++k) {
                double e = fabs(double(w[k]) - ref[k]);
                if (e > err5) {
                    err5 = e;
                }
            }
            double ref_min = ref[0] < ref[1]
                                 ? (ref[0] < ref[2] ? ref[0] : ref[2])
                                 : (ref[1] < ref[2] ? ref[1] : ref[2]);
            if (fabs(ref_min) > kTol) {
                float got_min = w[0] < w[1] ? (w[0] < w[2] ? w[0] : w[2])
                                            : (w[1] < w[2] ? w[1] : w[2]);
                if ((got_min > 0.0f) != (ref_min > 0.0)) {
                    ++dis5;
                    bad_apex = t2[1];
                    bad_px = p[0];
                    bad_py = p[1];
                }
            }
        }
        printf("=== gate 5: barycentric verdict on slivers (%d cases) ===\n",
               N_BARY);
        printf("  worst coordinate error         = %.3e\n", err5);
        printf("  unambiguous misclassifications = %u\n", dis5);
        if (dis5) {
            printf("    e.g. apex height %.3e at point (%.4f, %.4f)\n",
                   double(bad_apex), double(bad_px), double(bad_py));
        }
        printf("  %s\n\n", dis5 == 0u ? "PASS"
                                      : "FAIL: the inside test lands on the "
                                        "wrong side");
        if (dis5) {
            ++failures;
        }
    }

    // ----------------------------------------------------------- gate 6
    {
        std::vector<float> tg(15 * N_SOUND, 0.0f), tt(N_SOUND, 0.0f),
            ta(N_SOUND, -1.0f);
        run_sound_tangential(tg.data(), tt.data(), ta.data());

        unsigned tot = 0u, miss = 0u;
        for (int c = 0; c < N_SOUND; ++c) {
            if (ta[c] < 0.0f) {
                continue;
            }
            const float *g = &tg[15 * c];
            D3 A0 = d3(g), A1 = d3(g + 3), A0E = d3(g + 6), A1E = d3(g + 9),
               R1 = d3(g + 12);
            D3 zero{0.0, 0.0, 0.0};
            auto sep = [&](double u) {
                D3 s0 = mad(A0, u, sub(A0E, A0));
                D3 s1 = mad(A1, u, sub(A1E, A1));
                return ref_edge_edge(s0, sub(s1, s0), zero, R1);
            };
            double d_start = sep(0.0), dmin = d_start;
            for (int i = 1; i <= N_SAMPLES; ++i) {
                double d = sep(double(i) / double(N_SAMPLES));
                if (d < dmin) {
                    dmin = d;
                }
            }
            if (!(d_start > 1e-6) || dmin > 1e-7) {
                continue;
            }
            ++tot;
            if (tt[c] >= kMaxT) {
                ++miss;
            }
        }
        printf("=== gate 6: gliding sweeps that genuinely collide (%u "
               "constructed) ===\n", tot);
        printf("  certified as collision-free : %u\n", miss);
        bool g6 = (tot > 0u) && (miss == 0u);
        printf("  %s\n", tot == 0u ? "FAIL: no crossing was constructed"
                                   : (miss == 0u ? "PASS"
                                                 : "FAIL: a colliding gliding "
                                                   "trajectory was certified"));
        if (!g6) {
            ++failures;
        }
    }

    printf("%s\n", failures == 0 ? "all contact-distance gates passed"
                                 : "contact-distance gates FAILED");
    return failures == 0 ? 0 : 1;
}
