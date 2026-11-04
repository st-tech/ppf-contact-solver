// File: lane4_block.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0
//
// A HOST TEST, DRIVEN BY `tests/kernel_gates.rs`.
//
// Regression gate for the four-lane block helpers in seam/seam_host.h
// (`compute::lane4_block_columns`, `..._transposed`, `lane4_block_product`,
// `lane4_block_magnitude`), which the host twin of the folded dynamic apply
// uses in place of the scalar `mat3_mul`, `mat3_transpose_mul` and the two
// entry-magnitude helpers of solver/spmv.kernel.cpp.
//
// THE CLAIM IS BIT IDENTITY, lane `a` against scalar component `a`, so this
// compares bit patterns, not values within a tolerance. It is compiled with
// contraction off (`kernel_gates.rs` passes `-ffp-contract=off`): with it on,
// the compiler may fuse a different product in each form, which is a choice
// both forms already have and not a property of the lane layout.
//
// The scalar forms are restated here from solver/spmv.kernel.cpp, word for
// word, because that file defines the cooperative body and its serial twin
// under one name and cannot be included by a single host translation unit.
//
// Inputs cover ordinary values, magnitudes spread over 2^-40 .. 2^40, signed
// zeros, infinities and NaNs. Each block sits at the END of an exact-size
// array, so a load reaching past the nine floats would read outside it (and
// fault under a sanitizer build).

#include "data.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

Vec3f mat3_mul(const float *matrix, const float *vector) {
    Vec3f result;
    result[0] =
        matrix[0] * vector[0] + matrix[3] * vector[1] + matrix[6] * vector[2];
    result[1] =
        matrix[1] * vector[0] + matrix[4] * vector[1] + matrix[7] * vector[2];
    result[2] =
        matrix[2] * vector[0] + matrix[5] * vector[1] + matrix[8] * vector[2];
    return result;
}

Vec3f mat3_transpose_mul(const float *matrix, const float *vector) {
    Vec3f result;
    result[0] =
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2];
    result[1] =
        matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2];
    result[2] =
        matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2];
    return result;
}

void magnitude(Vec3f &abssum, const float *matrix, const float *vector) {
    for (unsigned a = 0; a < 3; ++a) {
        abssum[a] += fmath::abs(matrix[a]) * fmath::abs(vector[0]) +
                     fmath::abs(matrix[3 + a]) * fmath::abs(vector[1]) +
                     fmath::abs(matrix[6 + a]) * fmath::abs(vector[2]);
    }
}

void magnitude_transpose(Vec3f &abssum, const float *matrix,
                         const float *vector) {
    for (unsigned a = 0; a < 3; ++a) {
        abssum[a] += fmath::abs(matrix[3 * a]) * fmath::abs(vector[0]) +
                     fmath::abs(matrix[3 * a + 1]) * fmath::abs(vector[1]) +
                     fmath::abs(matrix[3 * a + 2]) * fmath::abs(vector[2]);
    }
}

float sample(std::mt19937 &g, int mode) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    float v = u(g);
    if (mode >= 1) v = std::ldexp(v, int(g() % 81) - 40);
    if (mode == 2) {
        switch (g() % 16) {
        case 0: v = 0.0f; break;
        case 1: v = -0.0f; break;
        case 2: v = INFINITY; break;
        case 3: v = -INFINITY; break;
        case 4: v = NAN; break;
        default: break;
        }
    }
    return v;
}

unsigned bit_pattern(float f) {
    unsigned b;
    std::memcpy(&b, &f, 4);
    return b;
}

}  // namespace

int main() {
    const unsigned cases = 200000;
    unsigned differ[4] = {0, 0, 0, 0};
    // A NaN against a NaN of another payload is counted apart: which operand's
    // payload an operation propagates is the compiler's choice wherever it
    // commutes an operand pair, in the scalar form as much as in this one, so
    // it is not a property of the lane layout. Every other bit must agree.
    unsigned payload[4] = {0, 0, 0, 0};
    auto tally = [](float got, float want, unsigned &bad, unsigned &nan_payload) {
        if (bit_pattern(got) == bit_pattern(want)) return;
        if (got != got && want != want) {
            ++nan_payload;
        } else {
            ++bad;
        }
    };
    for (unsigned c = 0; c < cases; ++c) {
        std::mt19937 g(c);
        const int mode = int(c % 3);
        // Exact-size arrays, the block last, so a read past it leaves them.
        std::vector<float> block(9), right(3), start(3);
        for (auto &v : block) v = sample(g, mode);
        for (auto &v : right) v = sample(g, mode);
        for (auto &v : start) v = std::fabs(sample(g, mode));
        const float *m = block.data();
        const float *x = right.data();
        for (int transposed = 0; transposed < 2; ++transposed) {
            compute::lane4 c0, c1, c2;
            if (transposed) {
                compute::lane4_block_columns_transposed(m, c0, c1, c2);
            } else {
                compute::lane4_block_columns(m, c0, c1, c2);
            }
            const compute::lane4 product =
                compute::lane4_block_product(c0, c1, c2, x);
            compute::lane4 acc = compute::lane4_splat(0.0f);
            for (unsigned a = 0; a < 3; ++a) acc[a] = start[a];
            acc = acc + compute::lane4_block_magnitude(c0, c1, c2, x);
            const Vec3f want = transposed ? mat3_transpose_mul(m, x) : mat3_mul(m, x);
            Vec3f want_abs;
            for (unsigned a = 0; a < 3; ++a) want_abs[a] = start[a];
            if (transposed) {
                magnitude_transpose(want_abs, m, x);
            } else {
                magnitude(want_abs, m, x);
            }
            for (unsigned a = 0; a < 3; ++a) {
                tally(product[a], want[a], differ[2 * transposed], payload[2 * transposed]);
                tally(acc[a], want_abs[a], differ[2 * transposed + 1], payload[2 * transposed + 1]);
            }
        }
    }
    const char *names[4] = {"block product", "block magnitude", "transposed product", "transposed magnitude"};
    for (unsigned k = 0; k < 4; ++k) {
        printf("lane4 %-21s differs in %u of %u components (NaN payload only: %u)\n", names[k], differ[k], 3 * cases, payload[k]);
    }
    const bool pass = !differ[0] && !differ[1] && !differ[2] && !differ[3];
    printf("%s: the lane form is the scalar form bit for bit on %u blocks\n", pass ? "PASS" : "FAIL", cases);
    return pass ? 0 : 1;
}
