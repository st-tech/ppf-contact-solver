// File: accd_degenerate.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// A HOST TEST, DRIVEN BY `tests/kernel_gates.rs`. Both stages run on one
// thread and do no device work.
//
// Gate for the degenerate sweep frame in accd::rescale (contact/accd.hpp).
//
// rescale normalizes a CCD sweep by that sweep's own largest coordinate. The
// scaling loop below the division exits only on `scale <= s`, so it terminates
// only for a positive finite quotient: an infinite one satisfies `scale > s`
// on every iteration while `scale /= s` leaves it unchanged. It is a device
// loop entered from all four *_ccd entry points, for every contact pair of
// every line search, so the failure presents as a GPU hang with no assert and
// no output. This pins the guard that reports such a pair as coincident.
//
// Two frames produce that infinity. A frame with no extent (every column on
// the centroid at both ends, so the two primitives are coincident) divides by
// zero. A frame whose extent is small enough that 0.99f divided by it leaves
// the float range overflows to the same value, which takes an extent of about
// 2.9e-39: a denormal no scene authors, but one a float coordinate can hold,
// so the cases below construct it directly and cover that second frame from
// here.
//
// AGAINST UNGUARDED CODE STAGE 3 NEVER RETURNS, BY CONSTRUCTION, so this file
// supplies its own deadline: the `std::async` wait below gives stage 3 30 s
// and, on reaching it, prints a located failure and exits.
// The alternative, guarding each call behind the same precondition the fix
// adds, was rejected: rescale is called from inside the entry points, so no
// caller-side guard reaches it, and a test that only exercised a guard it
// installed itself would pass against the unguarded solver and gate nothing.
// The cases are instead ordered so everything that can terminate runs and
// reports first. Stage 1 (controls and sanity) and stage 2 (the collapse
// witnesses) complete either way, and only stage 3 hangs, immediately after a
// host line naming it, so a timeout kill reads as a located failure rather
// than as a mute stall.
//
// Stage 2 exists because a coincident configuration does not by itself present
// a zero extent to rescale. centerize subtracts a centroid weighted by 1/C,
// which is a power of two for C = 2 and C = 4 and is not representable for
// C = 3. A two or four column frame therefore collapses to exactly zero for
// any coincident position, while a three column frame collapses only where its
// columns already sit at the origin: elsewhere the centroid rounding leaves a
// residue of an ulp or two, which rescale amplifies instead of dividing by
// zero. That is why the point-edge configuration is the one placed at the
// origin. The witnesses measure the extent each stage 3 input actually
// presents to rescale and require it to be zero, so a change to the centroid
// rounding fails here with a clear message instead of quietly routing stage 3
// past the branch it covers.
//
// Stage 1 carries the control that gives the test its value: two SEPARATED
// primitives translating together also have zero relative motion, so u_max is
// zero for them exactly as it is for a collapsed frame. They must still be
// granted a full step and must not be reported. A guard keyed on the relative
// motion rather than on the frame extent passes every collapsed case here and
// silently breaks those.
//
// Usage (from `crates/ppf-cts-solver`):
//   cargo test --test kernel_gates accd_returns_on_a_degenerate_frame

// `data.hpp` FIRST, which is what `entrypoints/shim_contact.cpp` does and for
// the same reason: a shared header takes its vocabulary from whichever header
// the includer pulled in first, and `accd.hpp` names `max`, `SM_THREAD` and
// the shared vector types without including anything itself.
#include "data.hpp"

// `accd::rescale` spells `max` unqualified. Production instantiates it with
// `float`, and on the shipped call sites nvcc is what supplies the name: it
// force includes a global `max(float, float)` ahead of every user header, and
// a host C++ compiler does not. Declaring it here, BEFORE the header, puts it
// in scope for unqualified lookup at the template's definition point; ADL
// cannot reach it, `float` having no associated namespace. It stays in this
// translation unit rather than in the seam, which owes the host only what a
// neutral body may name.
inline float max(float a, float b) {
    return ::fmaxf(a, b);
}

#include "contact/accd.hpp"
#include <cmath>
#include <cstdio>
#include <ctime>
#include <chrono>
#include <future>

enum Case {
    // Stage 1: bounded. These terminate whether or not rescale is guarded.
    NORMAL_FACTOR = 0,
    NORMAL_EXTENT,
    FLOAT_NORMAL_FACTOR,
    FLOAT_NORMAL_EXTENT,
    CTRL_COMOVING_EXTENT,
    CTRL_COMOVING,
    SANITY_APPROACH,
    // Stage 2: bounded. Each measures the extent one stage 3 input hands to
    // rescale, without calling rescale.
    WITNESS_PP,
    WITNESS_PT,
    WITNESS_PE,
    WITNESS_EE,
    // Stage 3: unbounded against an unguarded rescale.
    RESCALE_ZERO_2,
    RESCALE_ZERO_3,
    RESCALE_ZERO_4,
    FLOAT_ZERO,
    FLOAT_UNDERFLOW,
    FLOAT_NAN,
    PP_COINCIDENT,
    PT_COLLAPSED,
    PE_COLLAPSED,
    EE_COLLAPSED,
    PP_REPORT_D2,
    PP_REPORT_OFFSET,
    N_CASES
};

static const char *CASE_NAME[N_CASES] = {
    "rescale: an ordinary frame returns a positive factor",
    "rescale: an ordinary frame normalizes the extent to just under one",
    "rescale<float>: an ordinary frame returns a positive factor",
    "rescale<float>: an ordinary frame normalizes the extent",
    "witness: the co-moving control presents a nonzero extent",
    "control: a separated pair moving together still gets a full step",
    "sanity: an ordinary approach stops at the parked gap",
    "witness: the point-point input collapses to a zero extent",
    "witness: the point-triangle input collapses to a zero extent",
    "witness: the point-edge input collapses to a zero extent",
    "witness: the edge-edge input collapses to a zero extent",
    "rescale: a zero two-column frame returns zero",
    "rescale: a zero three-column frame returns zero",
    "rescale: a zero four-column frame returns zero",
    "rescale<float>: a zero frame returns zero",
    "rescale<float>: an extent that overflows the quotient returns zero",
    "rescale<float>: a NaN extent returns zero",
    "point_point_ccd: a coincident pair reports and grants no advance",
    "point_triangle_ccd: a collapsed triangle reports",
    "point_edge_ccd: a collapsed edge reports",
    "edge_edge_ccd: collapsed edges report",
    "point_point_ccd: a collapsed frame records a zero start distance",
    "point_point_ccd: a collapsed frame leaves the offset unset",
};

#define MAX_T 1.25f
#define OFFSET 1e-3f
#define GHAT 1e-3f

// Small enough that 0.99f divided by it leaves the float range: a denormal no
// scene authors, but one a float coordinate can hold.
#define TINY_EXTENT 1e-40f

static ParamSet make_param() {
    ParamSet param = {};
    param.line_search_max_t = MAX_T;
    param.ccd_eps = 1e-7f;
    return param;
}

// A CCD reports an overlapping start through an accd::OverlapInfo the CALLER
// owns; there is no device global to read. accd.hpp states the contract, and
// contact.cu is written against it: a report grants no advance, so a returned
// toi of exactly zero is the signal that the fields were written, and the
// solver records the pair on exactly that test. This test reads the result the
// same way, and separately checks whether the fields really were written, so
// the two halves of a report are distinguishable: a zero returned without a
// record, or a record written on a step that was granted, both fail rather
// than passing on the strength of the other half.
#define OVERLAP_POISON (-7.0f)

enum ReportBit {
    REPORT_TOI_ZERO = 1,
    REPORT_WRITTEN = 2,
    REPORTED = REPORT_TOI_ZERO | REPORT_WRITTEN,
};

// The destination handed to every call below. Both fields start at a value
// neither answer can be confused with. It cannot be -1.0f: that is the sentinel
// the collapsed path leaves in the offset, so poisoning with it would make the
// unset-offset case vacuous. A written d2 is a squared distance and a written
// offset is a contact offset, so neither is ever negative either.
static accd::OverlapInfo poisoned_overlap() {
    accd::OverlapInfo overlap;
    overlap.d2 = OVERLAP_POISON;
    overlap.offset = OVERLAP_POISON;
    return overlap;
}

static unsigned report_bits(float toi,
                                       const accd::OverlapInfo &overlap) {
    unsigned bits = 0u;
    if (toi == 0.0f) {
        bits |= (unsigned)REPORT_TOI_ZERO;
    }
    if (overlap.d2 != OVERLAP_POISON || overlap.offset != OVERLAP_POISON) {
        bits |= (unsigned)REPORT_WRITTEN;
    }
    return bits;
}

// A sweep whose columns share one position at the start and one at the end.
struct Sweep {
    Vec3f a;
    Vec3f b;
};

// The four collapsed configurations, each named once so the witness stage and
// the entry-point stage speak about exactly the same geometry. Every column of
// the pair sits on `a` at the start of the sweep and on `b` at the end, so the
// frame is collapsed at BOTH ends and what collapses is the frame rather than
// the motion.
//
// The coordinates are exact binary fractions, which is what makes the centroid
// subtraction exact for the two and four column frames. The point-edge pair is
// the exception described in the file header: its centroid weight is 1/3,
// which is not representable, so it sits at the origin, where the subtraction
// is exact for any weight, and its two ends coincide.
static Sweep pp_sweep() {
    return {Vec3f(1.0f, 2.0f, 3.0f), Vec3f(1.5f, 2.0f, 3.0f)};
}
static Sweep pt_sweep() {
    return {Vec3f(-2.0f, 0.75f, 4.0f), Vec3f(-2.0f, 0.75f, 4.5f)};
}
static Sweep pe_sweep() {
    return {Vec3f(0.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, 0.0f)};
}
static Sweep ee_sweep() {
    return {Vec3f(3.0f, -1.0f, 0.5f), Vec3f(3.0f, -1.0f, 0.25f)};
}

// The magnitude rescale divides by, measured the way rescale measures it. The
// frames are taken by value so a witness cannot disturb its caller.
//
// The magnitude read here is ABSOLUTE for the reason it is absolute inside
// rescale: the quantity wanted IS the size of the centered sweep frame, and
// reading it is the whole purpose of the witness.
template <int C>
static float frame_extent(SMat<float, 3, C> x0,
                                     SMat<float, 3, C> dx, float max_t) {
    accd::centerize<float, 3, C>(x0);
    accd::centerize<float, 3, C>(dx);
    float max_entry =
        max(x0.cwiseAbs().maxCoeff(),
            (x0 + float(max_t) * dx).cwiseAbs().maxCoeff());
    return max_entry;
}

// Stage 1 and stage 2. Every case here terminates against an unguarded
// rescale, so its verdict is available even when stage 3 has to be killed.
void run_bounded(float *value, unsigned *flagged) {
    ParamSet param = make_param();

    // An ordinary frame. The factor must be positive, and applying it must
    // leave the largest coordinate just under one, which is the property every
    // caller relies on when it divides its own lengths by the same factor.
    {
        Mat3x2f x0, dx;
        x0 << Vec3f(0.1f, 0.0f, 0.0f), Vec3f(-0.1f, 0.0f, 0.0f);
        dx << Vec3f(0.0f, 0.01f, 0.0f), Vec3f(0.0f, -0.01f, 0.0f);
        accd::centerize<float, 3, 2>(x0);
        accd::centerize<float, 3, 2>(dx);
        accd::OverlapInfo overlap = poisoned_overlap();
        value[NORMAL_FACTOR] =
            accd::rescale<float, 3, 2>(x0, dx, MAX_T, &overlap);
        float max_entry =
            max(x0.cwiseAbs().maxCoeff(),
                (x0 + float(MAX_T) * dx).cwiseAbs().maxCoeff());
        value[NORMAL_EXTENT] = max_entry;
    }

    // The same, through the float instantiation, so the ordinary path of a
    // float-coordinate build is covered alongside its degenerate one.
    {
        Mat3x2f x0, dx;
        x0 << Vec3f(0.1f, 0.0f, 0.0f), Vec3f(-0.1f, 0.0f, 0.0f);
        dx << Vec3f(0.0f, 0.01f, 0.0f), Vec3f(0.0f, -0.01f, 0.0f);
        accd::OverlapInfo overlap = poisoned_overlap();
        value[FLOAT_NORMAL_FACTOR] =
            accd::rescale<float, 3, 2>(x0, dx, MAX_T, &overlap);
        value[FLOAT_NORMAL_EXTENT] =
            max(x0.cwiseAbs().maxCoeff(),
                (x0 + MAX_T * dx).cwiseAbs().maxCoeff());
    }

    // The control. Two points a real distance apart, translating together.
    // Their separation never changes, so the CCD must not restrict the step
    // and must not report an overlap. Its extent is recorded alongside to make
    // the discrimination explicit: u_max is zero here exactly as it is for the
    // collapsed cases, and the extent is the only thing that tells them apart.
    {
        Vec3f a0(0.0f, 0.0f, 0.0f), b0(0.5f, 0.0f, 0.0f);
        Vec3f a1(0.1f, 0.0f, 0.0f), b1(0.6f, 0.0f, 0.0f);
        Mat3x2f x0, dx;
        x0 << a0, b0;
        dx << a1 - a0, b1 - b0;
        value[CTRL_COMOVING_EXTENT] = frame_extent<2>(x0, dx, MAX_T);

        accd::OverlapInfo overlap = poisoned_overlap();
        value[CTRL_COMOVING] =
            accd::point_point_ccd(a0, a1, b0, b1, OFFSET, GHAT,
                                  param.line_search_max_t, param.ccd_eps,
                                  &overlap);
        flagged[CTRL_COMOVING] = report_bits(value[CTRL_COMOVING], overlap);
    }

    // A head-on approach, so a rescale broken outright cannot pass by
    // reporting everything as coincident. The pair starts 0.6 apart and closes
    // at 1.0 per unit of t, and the CCD parks it at the contact offset plus
    // park_floor(ghat), so the answer is (0.6 - 1e-3 - 1e-5) / 1.0 = 0.59899.
    // The toi is invariant under rescale, which scales x, dx and every length
    // it is compared against by one common factor, so this value also checks
    // that the normalization round-trips.
    {
        Vec3f a0(0.0f, 0.0f, 0.0f), b0(0.6f, 0.0f, 0.0f);
        Vec3f a1(0.5f, 0.0f, 0.0f), b1(0.1f, 0.0f, 0.0f);
        accd::OverlapInfo overlap = poisoned_overlap();
        value[SANITY_APPROACH] =
            accd::point_point_ccd(a0, a1, b0, b1, OFFSET, GHAT,
                                  param.line_search_max_t, param.ccd_eps,
                                  &overlap);
        flagged[SANITY_APPROACH] = report_bits(value[SANITY_APPROACH], overlap);
    }

    // The witnesses. Each rebuilds the frame its entry point builds, in the
    // same column order, and measures what rescale is about to divide by.
    {
        Sweep s = pp_sweep();
        Vec3f d = s.b - s.a;
        Mat3x2f x0, dx;
        x0 << s.a, s.a;
        dx << d, d;
        value[WITNESS_PP] = frame_extent<2>(x0, dx, MAX_T);
    }
    {
        Sweep s = pt_sweep();
        Vec3f d = s.b - s.a;
        Mat3x4f x0, dx;
        x0 << s.a, s.a, s.a, s.a;
        dx << d, d, d, d;
        value[WITNESS_PT] = frame_extent<4>(x0, dx, MAX_T);
    }
    {
        Sweep s = pe_sweep();
        Vec3f d = s.b - s.a;
        Mat3x3f x0, dx;
        x0 << s.a, s.a, s.a;
        dx << d, d, d;
        value[WITNESS_PE] = frame_extent<3>(x0, dx, MAX_T);
    }
    {
        Sweep s = ee_sweep();
        Vec3f d = s.b - s.a;
        Mat3x4f x0, dx;
        x0 << s.a, s.a, s.a, s.a;
        dx << d, d, d, d;
        value[WITNESS_EE] = frame_extent<4>(x0, dx, MAX_T);
    }
}

// Stage 3. Every case here hands rescale a frame the witnesses just measured
// as degenerate, so an unguarded rescale never returns from the first of them.
void run_collapsed(float *value, unsigned *flagged) {
    ParamSet param = make_param();

    // rescale on its own, at each of the three frame widths the entry points
    // use. A zeroed frame is the defect in its plainest form and needs no
    // argument about representability to construct.
    {
        Mat3x2f x = Mat3x2f::Zero(), dx = Mat3x2f::Zero();
        accd::OverlapInfo overlap = poisoned_overlap();
        value[RESCALE_ZERO_2] =
            accd::rescale<float, 3, 2>(x, dx, MAX_T, &overlap);
    }
    {
        Mat3x3f x = Mat3x3f::Zero(), dx = Mat3x3f::Zero();
        accd::OverlapInfo overlap = poisoned_overlap();
        value[RESCALE_ZERO_3] =
            accd::rescale<float, 3, 3>(x, dx, MAX_T, &overlap);
    }
    {
        Mat3x4f x = Mat3x4f::Zero(), dx = Mat3x4f::Zero();
        accd::OverlapInfo overlap = poisoned_overlap();
        value[RESCALE_ZERO_4] =
            accd::rescale<float, 3, 4>(x, dx, MAX_T, &overlap);
    }

    // The same instantiation again, with the two frames a float coordinate can
    // present: the zero frame is the same divide by zero, and the tiny frame
    // is the quotient overflow.
    {
        Mat3x4f x = Mat3x4f::Zero(), dx = Mat3x4f::Zero();
        accd::OverlapInfo overlap = poisoned_overlap();
        value[FLOAT_ZERO] = accd::rescale<float, 3, 4>(x, dx, MAX_T, &overlap);
    }
    {
        Mat3x4f x = Mat3x4f::Zero(), dx = Mat3x4f::Zero();
        x(0, 0) = TINY_EXTENT;
        accd::OverlapInfo overlap = poisoned_overlap();
        value[FLOAT_UNDERFLOW] =
            accd::rescale<float, 3, 4>(x, dx, MAX_T, &overlap);
    }

    // A NaN coordinate. maxCoeff seeds best = m[0] and replaces on
    // `best < m[i]`, so a NaN in slot 0 is never replaced and the extent comes
    // back NaN, which is the branch the guard's comment claims to cover.
    {
        Mat3x4f x = Mat3x4f::Zero();
        Mat3x4f dx = Mat3x4f::Zero();
        x(0, 0) = nanf("");
        accd::OverlapInfo overlap = poisoned_overlap();
        value[FLOAT_NAN] = accd::rescale<float, 3, 4>(x, dx, MAX_T, &overlap);
    }

    // Two grains at the same position, moving together. Two distinct grains
    // closer together than one ulp of their coordinates share a position
    // exactly, so a scene reaches this configuration.
    {
        accd::OverlapInfo overlap = poisoned_overlap();
        Sweep s = pp_sweep();
        value[PP_COINCIDENT] = accd::point_point_ccd(
            s.a, s.b, s.a, s.b, OFFSET, GHAT, param.line_search_max_t,
            param.ccd_eps, &overlap);
        flagged[PP_COINCIDENT] = report_bits(value[PP_COINCIDENT], overlap);
        // The values the host reads to build its OverlappingStart crash. The
        // distance is exactly zero, which needs no units. The offset is left
        // at its unset sentinel, because ccd_helper's flag sites record it in
        // the frame rescale gave them and a collapsed frame has no such scale.
        value[PP_REPORT_D2] = overlap.d2;
        value[PP_REPORT_OFFSET] = overlap.offset;
    }

    // A triangle collapsed onto the query point at both ends.
    {
        accd::OverlapInfo overlap = poisoned_overlap();
        Sweep s = pt_sweep();
        value[PT_COLLAPSED] =
            accd::point_triangle_ccd(s.a, s.b, s.a, s.a, s.a, s.b, s.b, s.b,
                                     OFFSET, GHAT, param.line_search_max_t,
                                     param.ccd_eps, &overlap);
        flagged[PT_COLLAPSED] = report_bits(value[PT_COLLAPSED], overlap);
    }

    // A zero-length edge coincident with the query point.
    {
        accd::OverlapInfo overlap = poisoned_overlap();
        Sweep s = pe_sweep();
        value[PE_COLLAPSED] =
            accd::point_edge_ccd(s.a, s.b, s.a, s.a, s.b, s.b, OFFSET, GHAT,
                                 param.line_search_max_t, param.ccd_eps,
                                 &overlap);
        flagged[PE_COLLAPSED] = report_bits(value[PE_COLLAPSED], overlap);
    }

    // Two zero-length edges at the same point.
    {
        accd::OverlapInfo overlap = poisoned_overlap();
        Sweep s = ee_sweep();
        value[EE_COLLAPSED] =
            accd::edge_edge_ccd(s.a, s.a, s.a, s.a, s.b, s.b, s.b, s.b, OFFSET,
                                GHAT, param.line_search_max_t, param.ccd_eps,
                                &overlap);
        flagged[EE_COLLAPSED] = report_bits(value[EE_COLLAPSED], overlap);
    }
}

static bool g_ok = true;

static void check(bool cond, int c, float measured) {
    printf("  %-4s %-66s %.9g\n", cond ? "ok" : "FAIL", CASE_NAME[c], measured);
    g_ok = g_ok && cond;
}

int main() {
    float value[N_CASES] = {};
    unsigned flagged[N_CASES] = {};

    run_bounded(value, flagged);

    check(value[NORMAL_FACTOR] > 0.0f, NORMAL_FACTOR, value[NORMAL_FACTOR]);
    check(value[NORMAL_EXTENT] > 0.9f && value[NORMAL_EXTENT] < 1.0f,
          NORMAL_EXTENT, value[NORMAL_EXTENT]);
    check(value[FLOAT_NORMAL_FACTOR] > 0.0f, FLOAT_NORMAL_FACTOR,
          value[FLOAT_NORMAL_FACTOR]);
    check(value[FLOAT_NORMAL_EXTENT] > 0.9f && value[FLOAT_NORMAL_EXTENT] < 1.0f,
          FLOAT_NORMAL_EXTENT, value[FLOAT_NORMAL_EXTENT]);
    check(value[CTRL_COMOVING_EXTENT] > 0.0f, CTRL_COMOVING_EXTENT,
          value[CTRL_COMOVING_EXTENT]);
    check(value[CTRL_COMOVING] == MAX_T && flagged[CTRL_COMOVING] == 0u,
          CTRL_COMOVING, value[CTRL_COMOVING]);
    check(value[SANITY_APPROACH] > 0.59f && value[SANITY_APPROACH] < 0.61f &&
              flagged[SANITY_APPROACH] == 0u,
          SANITY_APPROACH, value[SANITY_APPROACH]);

    const int witnesses[] = {WITNESS_PP, WITNESS_PT, WITNESS_PE, WITNESS_EE};
    bool collapsed = true;
    for (int c : witnesses) {
        bool zero = value[c] == 0.0f;
        check(zero, c, value[c]);
        collapsed = collapsed && zero;
    }

    if (!collapsed) {
        // Running stage 3 now would prove nothing: an input that no longer
        // presents a degenerate frame exercises the ordinary path and reports
        // a pass that means nothing. Stop, and say which precondition broke.
        printf("FAIL: a stage 3 input no longer reaches the branch under test "
               "(see the nonzero extent above); stage 3 not run\n");
        return 1;
    }

    printf("  -- stage 3: an unguarded rescale does not return from these. A "
           "hang here IS the failure.\n");
    fflush(stdout);

    // STAGE 3 GETS A DEADLINE, NOT A BLOCKING CALL, and that is the whole
    // reason it is written this way. An unguarded rescale never returns on a
    // collapsed frame, so running it inline would hang this binary and the
    // failure would present as a test that never finishes rather than as a
    // located one. The device version polled a non-blocking stream against the
    // same 30 s deadline; `std::async` is the host's spelling of it.
    //
    // The thread is DETACHED rather than joined on the timeout path, because a
    // future that never becomes ready cannot be joined: the process reports the
    // located failure and exits, which is the outcome that matters.
    auto stage3 = std::async(std::launch::async, [&]() {
        run_collapsed(value, flagged);
    });
    const auto deadline = std::chrono::seconds(30);
    if (stage3.wait_for(deadline) != std::future_status::ready) {
        printf("  FAIL stage 3 did not retire within %lld s: rescale did not "
               "return on a collapsed frame\n", (long long)deadline.count());
        printf("FAIL: %d cases\n", (int)N_CASES);
        // `_Exit` rather than `return`: a detached task still holding `value`
        // makes an ordinary unwind a race, and the verdict is already printed.
        std::fflush(stdout);
        std::_Exit(1);
    }

    const int zeros[] = {RESCALE_ZERO_2, RESCALE_ZERO_3, RESCALE_ZERO_4,
                         FLOAT_ZERO, FLOAT_UNDERFLOW, FLOAT_NAN};
    for (int c : zeros) {
        check(value[c] == 0.0f, c, value[c]);
    }

    const int reporting[] = {PP_COINCIDENT, PT_COLLAPSED, PE_COLLAPSED,
                             EE_COLLAPSED};
    for (int c : reporting) {
        // Both halves of the report are required: no advance granted, and the
        // caller's OverlapInfo actually written.
        check(value[c] == 0.0f && flagged[c] == (unsigned)REPORTED, c,
              value[c]);
    }
    check(value[PP_REPORT_D2] == 0.0f, PP_REPORT_D2, value[PP_REPORT_D2]);
    check(value[PP_REPORT_OFFSET] == -1.0f, PP_REPORT_OFFSET,
          value[PP_REPORT_OFFSET]);

    printf("%s: %d cases\n", g_ok ? "PASS" : "FAIL", (int)N_CASES);
    return g_ok ? 0 : 1;
}
