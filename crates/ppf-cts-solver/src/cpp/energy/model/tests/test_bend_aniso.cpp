// File: tests/test_bend_aniso.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// Host regression for hinge_bend_directional (cpp/data.hpp), which combines a
// shell hinge's isotropic bending stiffness with the two directional ones
// according to where its shared edge sits in the UV material frame:
//
//   k(psi) = bend + warp * sin^2(psi) + weft * cos^2(psi)
//
// This is the ONLY place the sin/cos pairing is pinned down. A build with warp
// and weft exchanged is still directional, still non-negative, still SPD, and
// simply drapes wrong by 90 degrees; nothing downstream notices, so the anchors
// in `anchors_pin_the_sign_convention` below are load-bearing.

#include "../../../data.hpp"

#include <cmath>
#include <cstdio>

namespace {

int g_failures = 0;

void expect_near(const char *what, float got, float want, float tol) {
    if (!(std::fabs(got - want) <= tol)) {
        std::printf("  FAIL %-52s got %.9g want %.9g\n", what, got, want);
        ++g_failures;
    }
}

void expect_true(const char *what, bool cond) {
    if (!cond) {
        std::printf("  FAIL %s\n", what);
        ++g_failures;
    }
}

// A hinge folds ABOUT its shared edge, so the surface curves ACROSS it: an
// edge lying along warp bends the sheet in the weft sense and picks up `weft`,
// while an edge along weft bends the warp fibers and picks up `warp`. That
// 90-degree swap is why sin^2 pairs with warp.
void anchors_pin_the_sign_convention() {
    const float bend = 3.0f, warp = 11.0f, weft = 5.0f;
    expect_near("edge along warp (sin2=0) -> bend + weft",
                hinge_bend_directional(bend, warp, weft, 0.0f), bend + weft,
                1e-6f);
    expect_near("edge along weft (sin2=1) -> bend + warp",
                hinge_bend_directional(bend, warp, weft, 1.0f), bend + warp,
                1e-6f);
    // The 45 degree value is not independently settable under this form; it is
    // the midpoint. Asserted so a future change to a bias-carrying form has to
    // come here and say so.
    expect_near("edge at 45 degrees -> bend + (warp + weft)/2",
                hinge_bend_directional(bend, warp, weft, 0.5f),
                bend + 0.5f * (warp + weft), 1e-6f);
}

// The isotropic default must return `bend` EXACTLY at every orientation, not
// merely close: the kernel scales the hinge block by this, so any drift
// perturbs every pre-existing scene.
void isotropic_defaults_are_bit_exact() {
    int off = 0;
    for (int i = 0; i <= 100000; ++i) {
        const float s = static_cast<float>(i) / 100000.0f;
        if (hinge_bend_directional(7.5f, 0.0f, 0.0f, s) != 7.5f) {
            ++off;
        }
    }
    expect_true("zero warp and weft return bend exactly at 100001 orientations",
                off == 0);
    if (off) {
        std::printf("       %d orientations deviated\n", off);
    }
}

// A mesh with no UV carries the sentinel and must fall back to the isotropic
// stiffness rather than to whatever the expression makes of a negative sin^2.
void no_uv_sentinel_is_isotropic() {
    expect_true("sentinel -> bend exactly",
                hinge_bend_directional(2.25f, 11.0f, 5.0f, -1.0f) == 2.25f);
}

// SPD-by-assembly: the kernel scales a PSD hinge block by this, so a negative
// value would hand the solver an indefinite matrix. Every term is
// non-negative, and so are sin^2 and cos^2, which is the whole argument; there
// is no stability condition beyond scene.rs's non-negativity asserts.
void stiffness_is_never_negative() {
    int neg = 0;
    for (int bi = 0; bi <= 20; ++bi) {
        const float bend = 0.5f * static_cast<float>(bi);
        for (int wi = 0; wi <= 40; ++wi) {
            const float warp = 2.5f * static_cast<float>(wi);
            for (int fi = 0; fi <= 40; ++fi) {
                const float weft = 2.5f * static_cast<float>(fi);
                for (int i = 0; i <= 100; ++i) {
                    const float s = static_cast<float>(i) / 100.0f;
                    if (hinge_bend_directional(bend, warp, weft, s) < 0.0f) {
                        ++neg;
                    }
                }
            }
        }
    }
    expect_true("no non-negative input yields a negative stiffness", neg == 0);
    if (neg) {
        std::printf("       %d negative samples\n", neg);
    }
}

// The stiffness never leaves the interval the two directional values bracket,
// so a hinge can be no softer than the softest axis and no stiffer than the
// stiffest. That is what makes the two knobs readable as "stiffness in this
// direction" rather than as coefficients of some interpolation.
void stiffness_stays_between_the_axes() {
    const float bend = 1.5f, warp = 9.0f, weft = 2.0f;
    const float lo = bend + (warp < weft ? warp : weft);
    const float hi = bend + (warp > weft ? warp : weft);
    int out = 0;
    for (int i = 0; i <= 10000; ++i) {
        const float s = static_cast<float>(i) / 10000.0f;
        const float k = hinge_bend_directional(bend, warp, weft, s);
        if (k < lo - 1e-5f || k > hi + 1e-5f) {
            ++out;
        }
    }
    expect_true("stiffness stays within [bend+min, bend+max]", out == 0);
}

// The directional part reads a DIRECTION, so scaling the UV layout cannot
// change it. Verified at the sin^2 level, which is what builder.rs computes.
void direction_only_so_scale_free() {
    const float bend = 2.0f, warp = 6.0f, weft = 1.0f;
    for (int i = 1; i < 200; ++i) {
        const float du = 0.01f * static_cast<float>(i) - 1.0f;
        const float dv = 1.0f - 0.007f * static_cast<float>(i);
        const float len2 = du * du + dv * dv;
        if (!(len2 > 0.0f)) {
            continue;
        }
        const float s = dv * dv / len2;
        const float base = hinge_bend_directional(bend, warp, weft, s);
        for (float k : {0.001f, 0.5f, 3.0f, 1000.0f, -1.0f}) {
            const float du2 = k * du, dv2 = k * dv;
            const float s2 = dv2 * dv2 / (du2 * du2 + dv2 * dv2);
            expect_near("UV scale / edge reversal leaves the stiffness alone",
                        hinge_bend_directional(bend, warp, weft, s2), base,
                        1e-4f);
        }
    }
}

} // namespace

int main() {
    std::printf("test_bend_aniso\n");
    anchors_pin_the_sign_convention();
    isotropic_defaults_are_bit_exact();
    no_uv_sentinel_is_isotropic();
    stiffness_is_never_negative();
    stiffness_stays_between_the_axes();
    direction_only_so_scale_free();
    if (g_failures == 0) {
        std::printf("  all checks passed\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", g_failures);
    return 1;
}
