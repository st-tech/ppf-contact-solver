// File: external_field.kernel.cpp
// Code: Claude Code
// Review: Ryoichi Ando (ryoichi.ando@zozo.com)
// License: Apache v2.0

#pragma once

// A NEUTRAL KERNEL SOURCE. Plain C++ belonging to no backend: no preprocessor
// conditional, no macro of its own, and no spelling that only nvcc or only the
// Metal shader compiler accepts. ppf-cts-compute/seam/kernelgen.py renders it
// into the forms the compilers read.
//
// THE EXTERNAL FORCE FIELD: a per-vertex acceleration `a(x, t)` in m/s^2, and
// a per-vertex air velocity for the aerodynamic drag, evaluated ONCE PER STEP
// at the step's starting position and time.
//
// Two sources are summed, and a scene may carry either, both or neither:
//
//   * SAMPLED GRIDS. Each grid is a world-space box sampled at W x H x D cell
//     corners and at T instants, trilinear in space and linear in time. A grid
//     is ZERO outside its box; time outside its samples clamps to the end
//     sample. A grid's KIND says which output it feeds.
//   * ONE EXACT SCRIPT, a bytecode program the frontend compiled from a Python
//     function `eval(x, y, z, t)`. It is evaluated at the vertex's own
//     position, so it has no resolution and no domain.
//
// WHY THE STEP'S STARTING POSITION AND NOT THE NEWTON ITERATE. The result is
// added to the inertial target beside gravity, so the Newton system gains no
// Hessian term: nothing asymmetric (the Jacobian of a curl field) has to be
// projected, and SPD-by-assembly is untouched. Steps are TOI-limited
// substeps, so an object moving through the field still meets it at the
// substep rate.
//
// POSITIONS ARE READ AS ABSOLUTE WORLD COORDINATES. Both sources are functions
// of WHERE a vertex is: a grid resolves a position to a cell, and a script is a
// formula the user wrote in world coordinates. Neither differences two
// positions.

#include "../contact/distance.hpp"

// The bytecode. The frontend compiler (`frontend/_force_field_.py`) and the
// solver's loader (`src/force_field.rs`) spell the same numbers; the loader
// refuses a program naming anything else, so this table is closed.
//
// A word is `opcode | operand << 8`. Every jump is FORWARD, which the loader
// verifies, so a program runs at most its own length of instructions and
// always terminates. The maximum stack depth is proven by the loader too, so
// the bounds checks below are guarantee-class asserts rather than the control
// flow.
enum FieldOp : unsigned {
    FIELD_OP_CONST = 1u,
    FIELD_OP_LOAD = 2u,
    FIELD_OP_STORE = 3u,
    FIELD_OP_ADD = 4u,
    FIELD_OP_SUB = 5u,
    FIELD_OP_MUL = 6u,
    FIELD_OP_DIV = 7u,
    FIELD_OP_NEG = 8u,
    FIELD_OP_MOD = 9u,
    FIELD_OP_POW = 10u,
    FIELD_OP_LT = 11u,
    FIELD_OP_LE = 12u,
    FIELD_OP_GT = 13u,
    FIELD_OP_GE = 14u,
    FIELD_OP_EQ = 15u,
    FIELD_OP_NE = 16u,
    FIELD_OP_NOT = 17u,
    FIELD_OP_AND = 18u,
    FIELD_OP_OR = 19u,
    FIELD_OP_JUMP = 20u,
    FIELD_OP_JUMP_IF_FALSE = 21u,
    FIELD_OP_RETURN = 22u,
    FIELD_OP_SQRT = 32u,
    FIELD_OP_SIN = 33u,
    FIELD_OP_COS = 34u,
    FIELD_OP_TAN = 35u,
    FIELD_OP_EXP = 36u,
    FIELD_OP_LOG = 37u,
    FIELD_OP_ABS = 38u,
    FIELD_OP_FLOOR = 39u,
    FIELD_OP_CEIL = 40u,
    FIELD_OP_TANH = 41u,
    FIELD_OP_ASIN = 42u,
    FIELD_OP_ACOS = 43u,
    FIELD_OP_ATAN = 44u,
    FIELD_OP_SINH = 45u,
    FIELD_OP_COSH = 46u,
    FIELD_OP_ATAN2 = 48u,
    FIELD_OP_MIN = 49u,
    FIELD_OP_MAX = 50u,
    FIELD_OP_HYPOT = 51u,
    // Pop x, y, z, w, seed; the operand is the octave count. NOISE pushes
    // one value, CURL pushes three. w is how far the pattern has evolved.
    FIELD_OP_NOISE = 52u,
    FIELD_OP_CURL = 53u,
};

// Capacities of the interpreter's thread-local storage. The loader refuses a
// program that needs more, so these are limits of the format, stated once in
// `src/force_field.rs` as well.
enum FieldLimit : unsigned {
    FIELD_STACK = 32u,
    FIELD_VARS = 64u,
    FIELD_MAX_OCTAVES = 8u,
};

// Words per grid in the header table, and floats per grid in the box table.
enum FieldGridLayout : unsigned {
    FIELD_GRID_HEADER = 8u,
    FIELD_GRID_BOX = 6u,
    FIELD_KIND_ACCELERATION = 0u,
    FIELD_KIND_AIR_VELOCITY = 1u,
    // Words per script in the script table: code offset, code length,
    // constant offset, constant count, target bit.
    FIELD_SCRIPT_HEADER = 5u,
    // A target bit of this value reaches every vertex; below it, the bit a
    // vertex's target mask must carry.
    FIELD_ALL_TARGETS = 32u,
};

// Floor through truncation, which is what the special-function-free float
// arithmetic of every backend agrees on. Exact for |x| < 2^31; the scripts
// this serves compute cell or period indices, far inside that.
[[seam::device_fn]] inline float field_floor(float x) {
    const float truncated = static_cast<float>(static_cast<int>(x));
    return truncated > x ? truncated - 1.0f : truncated;
}

// cos through the periodic sine, the one unbounded-argument transcendental the
// seam carries on every backend.
[[seam::device_fn]] inline float field_cos(float x) {
    return fmath::sin_periodic(x + 1.57079632679f);
}

// Python's `**`. A positive base goes through exp and log; zero and negative
// bases follow the real-valued cases Python defines, and a negative base with
// a non-integer exponent is NaN, which the caller's finiteness assert reports.
[[seam::device_fn]] inline float field_pow(float base, float exponent) {
    if (base > 0.0f) {
        return fmath::exp(exponent * fmath::log(base));
    }
    if (base == 0.0f) {
        return exponent > 0.0f ? 0.0f : (exponent == 0.0f ? 1.0f : fmath::infinity());
    }
    if (field_floor(exponent) != exponent) {
        return fmath::infinity() - fmath::infinity();
    }
    const float magnitude = fmath::exp(exponent * fmath::log(-base));
    const float halved = exponent * 0.5f;
    return field_floor(halved) == halved ? magnitude : -magnitude;
}


// THE NOISE, one algorithm in three places: here in single precision, and in
// `frontend/_noise_.py` and `blender_addon/core/noise.py` (byte-identical) in
// double. Every constant is part of the definition. A four-dimensional
// grid of hashed gradients from the thirty-two hypercube edge midpoints,
// blended with the quintic fade; the first three coordinates are space and
// the fourth, w, is how far the pattern has evolved. The analytic gradient in
// SPACE comes with it, which is what the curl reads.

// A 32-bit hash of a grid point and a seed. Unsigned multiplication wraps
// on every backend, which is the arithmetic the Python copies mask to.
[[seam::device_fn]] inline unsigned field_hash(unsigned ix, unsigned iy,
                                               unsigned iz, unsigned iw,
                                               unsigned seed) {
    unsigned h = seed * 0x9E3779B9u;
    h ^= ix * 0x85EBCA6Bu;
    h ^= iy * 0xC2B2AE35u;
    h ^= iz * 0x27D4EB2Fu;
    h ^= iw * 0x165667B1u;
    h ^= h >> 15u;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12u;
    h *= 0x297A2D39u;
    h ^= h >> 15u;
    return h;
}

// Component `axis` of gradient `index` of the thirty-two: index 8 * a + s has
// a zero on axis a and, on the other three axes in order, -1 where bit 0, 1,
// 2 of s is set, else +1. The same table the Python copies spell out.
[[seam::device_fn]] inline float field_gradient(unsigned index, unsigned axis) {
    const unsigned zero = index >> 3u;
    if (axis == zero) {
        return 0.0f;
    }
    const unsigned bit = axis < zero ? axis : axis - 1u;
    return ((index >> bit) & 1u) != 0u ? -1.0f : 1.0f;
}

[[seam::device_fn]] inline float field_fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

[[seam::device_fn]] inline float field_dfade(float t) {
    return 30.0f * t * t * (t - 1.0f) * (t - 1.0f);
}

// One octave of gradient noise at (p, w): the value, and its gradient in
// space in `grad`.
[[seam::device_fn]] inline float field_gradient_noise(float px, float py,
                                                      float pz, float pw,
                                                      unsigned seed,
                                                      Vec3f &grad) {
    const float p[4] = {px, py, pz, pw};
    unsigned cell[4];
    float f[4];
    float u[4];
    float du[4];
    for (unsigned d = 0u; d < 4u; ++d) {
        const float c = field_floor(p[d]);
        cell[d] = static_cast<unsigned>(static_cast<int>(c));
        f[d] = p[d] - c;
        u[d] = field_fade(f[d]);
        du[d] = field_dfade(f[d]);
    }
    float value = 0.0f;
    grad = Vec3f::Zero();
    for (unsigned corner = 0u; corner < 16u; ++corner) {
        unsigned o[4];
        for (unsigned d = 0u; d < 4u; ++d) {
            o[d] = (corner >> d) & 1u;
        }
        const unsigned index =
            field_hash(cell[0] + o[0], cell[1] + o[1], cell[2] + o[2],
                       cell[3] + o[3], seed) %
            32u;
        float g[4];
        float w[4];
        float dw[4];
        float dot = 0.0f;
        for (unsigned d = 0u; d < 4u; ++d) {
            g[d] = field_gradient(index, d);
            dot += g[d] * (f[d] - static_cast<float>(o[d]));
            w[d] = o[d] != 0u ? u[d] : 1.0f - u[d];
            dw[d] = o[d] != 0u ? du[d] : -du[d];
        }
        const float weight = w[0] * w[1] * w[2] * w[3];
        value += weight * dot;
        grad[0] += dw[0] * w[1] * w[2] * w[3] * dot + weight * g[0];
        grad[1] += w[0] * dw[1] * w[2] * w[3] * dot + weight * g[1];
        grad[2] += w[0] * w[1] * dw[2] * w[3] * dot + weight * g[2];
    }
    return value;
}

// `octaves` octaves at halving amplitude and doubling frequency (in space and
// in w alike), normalized by the amplitudes' sum; the gradient likewise.
[[seam::device_fn]] inline float field_fractal_noise(float px, float py,
                                                     float pz, float pw,
                                                     unsigned octaves,
                                                     unsigned seed,
                                                     Vec3f &grad) {
    float value = 0.0f;
    grad = Vec3f::Zero();
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float total = 0.0f;
    for (unsigned k = 0u; k < octaves; ++k) {
        Vec3f g;
        const float v = field_gradient_noise(px * frequency, py * frequency,
                                             pz * frequency, pw * frequency,
                                             seed + k * 1013u, g);
        value += amplitude * v;
        grad += (amplitude * frequency) * g;
        total += amplitude;
        amplitude *= 0.5f;
        frequency *= 2.0f;
    }
    grad *= fmath::div(1.0f, total);
    return fmath::div(value, total);
}

// The curl in space of three decorrelated noise potentials at one w:
// divergence-free at every instant.
[[seam::device_fn]] inline Vec3f field_curl_noise(float px, float py, float pz,
                                                  float pw, unsigned octaves,
                                                  unsigned seed) {
    Vec3f g0;
    Vec3f g1;
    Vec3f g2;
    field_fractal_noise(px, py, pz, pw, octaves, seed, g0);
    field_fractal_noise(px + 31.416f, py + 47.853f, pz + 12.679f, pw, octaves,
                        seed + 7919u, g1);
    field_fractal_noise(px + 71.231f, py + 5.117f, pz + 93.402f, pw, octaves,
                        seed + 2u * 7919u, g2);
    Vec3f c;
    c[0] = g2[1] - g1[2];
    c[1] = g0[2] - g2[0];
    c[2] = g1[0] - g0[1];
    return c;
}

// One run of the script. Returns false when the program ran past its end
// without a RETURN, which the loader's proof makes unreachable; the caller
// asserts on it.
//
// The program is `length` words at `code_offset` in the shared code table, and
// its constants start at `constant_offset` in the shared constant table; the
// loader guarantees both lie inside their tables.
[[seam::device_fn]] inline bool field_script_eval(
    const unsigned *code, unsigned code_offset, unsigned length,
    const float *constants, unsigned constant_offset, unsigned constant_count,
    float x, float y, float z, float t,
    DiagHandle diag, Vec3f &out) {
    float stack[FIELD_STACK];
    float vars[FIELD_VARS];
    vars[0] = x;
    vars[1] = y;
    vars[2] = z;
    vars[3] = t;
    for (unsigned k = 4u; k < FIELD_VARS; ++k) {
        vars[k] = 0.0f;
    }
    unsigned sp = 0u;
    unsigned pc = 0u;
    while (pc < length) {
        const unsigned word = code[code_offset + pc];
        const unsigned op = word & 0xffu;
        const unsigned arg = word >> 8u;
        pc += 1u;
        if (op == FIELD_OP_CONST) {
            DIAG_ASSERT4(diag, arg < constant_count && sp < FIELD_STACK,
                         static_cast<float>(arg), static_cast<float>(sp), 0.0f,
                         0.0f);
            stack[sp] = constants[constant_offset + arg];
            sp += 1u;
        } else if (op == FIELD_OP_LOAD) {
            DIAG_ASSERT4(diag, arg < FIELD_VARS && sp < FIELD_STACK,
                         static_cast<float>(arg), static_cast<float>(sp), 1.0f,
                         0.0f);
            stack[sp] = vars[arg];
            sp += 1u;
        } else if (op == FIELD_OP_STORE) {
            DIAG_ASSERT4(diag, arg < FIELD_VARS && sp >= 1u,
                         static_cast<float>(arg), static_cast<float>(sp), 2.0f,
                         0.0f);
            sp -= 1u;
            vars[arg] = stack[sp];
        } else if (op == FIELD_OP_JUMP) {
            pc = arg;
        } else if (op == FIELD_OP_JUMP_IF_FALSE) {
            DIAG_ASSERT4(diag, sp >= 1u, static_cast<float>(sp), 3.0f, 0.0f,
                         0.0f);
            sp -= 1u;
            if (stack[sp] == 0.0f) {
                pc = arg;
            }
        } else if (op == FIELD_OP_RETURN) {
            DIAG_ASSERT4(diag, sp >= 3u, static_cast<float>(sp), 4.0f, 0.0f,
                         0.0f);
            out[0] = stack[sp - 3u];
            out[1] = stack[sp - 2u];
            out[2] = stack[sp - 1u];
            return true;
        } else if (op == FIELD_OP_NOISE || op == FIELD_OP_CURL) {
            DIAG_ASSERT4(diag,
                         sp >= 5u && arg >= 1u && arg <= FIELD_MAX_OCTAVES &&
                             (op == FIELD_OP_NOISE || sp + 3u - 5u <= FIELD_STACK),
                         static_cast<float>(sp), static_cast<float>(op),
                         static_cast<float>(arg), 9.0f);
            // The seed truncates toward zero, as Python's int() does.
            const unsigned seed =
                static_cast<unsigned>(static_cast<int>(stack[sp - 1u]));
            const float nx = stack[sp - 5u];
            const float ny = stack[sp - 4u];
            const float nz = stack[sp - 3u];
            const float nw = stack[sp - 2u];
            sp -= 5u;
            if (op == FIELD_OP_NOISE) {
                Vec3f unused;
                stack[sp] =
                    field_fractal_noise(nx, ny, nz, nw, arg, seed, unused);
                sp += 1u;
            } else {
                const Vec3f c = field_curl_noise(nx, ny, nz, nw, arg, seed);
                stack[sp] = c[0];
                stack[sp + 1u] = c[1];
                stack[sp + 2u] = c[2];
                sp += 3u;
            }
        } else if (op == FIELD_OP_NEG || op == FIELD_OP_NOT ||
                   (op >= FIELD_OP_SQRT && op <= FIELD_OP_COSH)) {
            DIAG_ASSERT4(diag, sp >= 1u, static_cast<float>(sp),
                         static_cast<float>(op), 5.0f, 0.0f);
            const float a = stack[sp - 1u];
            float r = 0.0f;
            if (op == FIELD_OP_NEG) {
                r = -a;
            } else if (op == FIELD_OP_NOT) {
                r = a == 0.0f ? 1.0f : 0.0f;
            } else if (op == FIELD_OP_SQRT) {
                r = fmath::sqrt(a);
            } else if (op == FIELD_OP_SIN) {
                r = fmath::sin_periodic(a);
            } else if (op == FIELD_OP_COS) {
                r = field_cos(a);
            } else if (op == FIELD_OP_TAN) {
                r = fmath::div(fmath::sin_periodic(a), field_cos(a));
            } else if (op == FIELD_OP_EXP) {
                r = fmath::exp(a);
            } else if (op == FIELD_OP_LOG) {
                r = fmath::log(a);
            } else if (op == FIELD_OP_ABS) {
                r = fmath::abs(a);
            } else if (op == FIELD_OP_FLOOR) {
                r = field_floor(a);
            } else if (op == FIELD_OP_CEIL) {
                r = -field_floor(-a);
            } else if (op == FIELD_OP_TANH) {
                // Saturated before the exponential can overflow: tanh is
                // within one ulp of +-1 well before |a| reaches 10.
                const float c = fmath::min(fmath::max(a, -10.0f), 10.0f);
                const float e = fmath::exp(2.0f * c);
                r = fmath::div(e - 1.0f, e + 1.0f);
            } else if (op == FIELD_OP_ASIN) {
                r = fmath::atan2(a, fmath::sqrt(1.0f - a * a));
            } else if (op == FIELD_OP_ACOS) {
                r = fmath::acos(a);
            } else if (op == FIELD_OP_ATAN) {
                r = fmath::atan2(a, 1.0f);
            } else if (op == FIELD_OP_SINH) {
                r = 0.5f * (fmath::exp(a) - fmath::exp(-a));
            } else if (op == FIELD_OP_COSH) {
                r = 0.5f * (fmath::exp(a) + fmath::exp(-a));
            }
            stack[sp - 1u] = r;
        } else {
            DIAG_ASSERT4(diag, sp >= 2u, static_cast<float>(sp),
                         static_cast<float>(op), 6.0f, 0.0f);
            const float b = stack[sp - 1u];
            const float a = stack[sp - 2u];
            float r = 0.0f;
            if (op == FIELD_OP_ADD) {
                r = a + b;
            } else if (op == FIELD_OP_SUB) {
                r = a - b;
            } else if (op == FIELD_OP_MUL) {
                r = a * b;
            } else if (op == FIELD_OP_DIV) {
                r = fmath::div(a, b);
            } else if (op == FIELD_OP_MOD) {
                // Python's floored modulo: the result takes the sign of b.
                r = a - b * field_floor(fmath::div(a, b));
            } else if (op == FIELD_OP_POW) {
                r = field_pow(a, b);
            } else if (op == FIELD_OP_LT) {
                r = a < b ? 1.0f : 0.0f;
            } else if (op == FIELD_OP_LE) {
                r = a <= b ? 1.0f : 0.0f;
            } else if (op == FIELD_OP_GT) {
                r = a > b ? 1.0f : 0.0f;
            } else if (op == FIELD_OP_GE) {
                r = a >= b ? 1.0f : 0.0f;
            } else if (op == FIELD_OP_EQ) {
                r = a == b ? 1.0f : 0.0f;
            } else if (op == FIELD_OP_NE) {
                r = a != b ? 1.0f : 0.0f;
            } else if (op == FIELD_OP_AND) {
                r = (a != 0.0f && b != 0.0f) ? 1.0f : 0.0f;
            } else if (op == FIELD_OP_OR) {
                r = (a != 0.0f || b != 0.0f) ? 1.0f : 0.0f;
            } else if (op == FIELD_OP_ATAN2) {
                r = fmath::atan2(a, b);
            } else if (op == FIELD_OP_MIN) {
                r = fmath::min(a, b);
            } else if (op == FIELD_OP_MAX) {
                r = fmath::max(a, b);
            } else if (op == FIELD_OP_HYPOT) {
                r = fmath::sqrt(a * a + b * b);
            } else {
                DIAG_ASSERT4(diag, false, static_cast<float>(op), 7.0f, 0.0f,
                             0.0f);
            }
            sp -= 1u;
            stack[sp - 1u] = r;
        }
    }
    return false;
}

// One grid at one point and one time. Returns false, and leaves `out`
// untouched, when the point is outside the grid's box.
//
// Layout: `data[((k * D + iz) * H + iy) * W + ix) * 3 + c]`, cell corners at
// `min + (max - min) * i / (N - 1)`. The loader guarantees every extent is at
// least 2 and every box non-degenerate, so no division here can be by zero.
[[seam::device_fn]] inline bool field_grid_sample(
    const unsigned *header, const float *box, const float *times,
    const float *data, float px, float py, float pz, float t,
    Vec3f &out) {
    const unsigned extent[3] = {header[0], header[1], header[2]};
    const unsigned samples = header[3];
    const unsigned data_offset = header[5];
    const unsigned time_offset = header[6];
    const float p[3] = {px, py, pz};
    unsigned cell[3];
    float frac[3];
    for (unsigned d = 0u; d < 3u; ++d) {
        const float lo = box[d];
        const float hi = box[3u + d];
        if (!(p[d] >= lo && p[d] <= hi)) {
            return false;
        }
        const float scaled = fmath::div(p[d] - lo, hi - lo) *
                             static_cast<float>(extent[d] - 1u);
        unsigned index = static_cast<unsigned>(scaled);
        if (index > extent[d] - 2u) {
            index = extent[d] - 2u;
        }
        cell[d] = index;
        frac[d] = fmath::min(fmath::max(scaled - static_cast<float>(index), 0.0f), 1.0f);
    }

    // The two time samples bracketing `t`, clamped at both ends.
    unsigned k0 = 0u;
    unsigned k1 = 0u;
    float blend = 0.0f;
    if (samples > 1u) {
        const float first = times[time_offset];
        const float last = times[time_offset + samples - 1u];
        if (t >= last) {
            k0 = samples - 1u;
            k1 = samples - 1u;
        } else if (t > first) {
            unsigned k = 0u;
            while (k + 2u < samples && times[time_offset + k + 1u] <= t) {
                k += 1u;
            }
            k0 = k;
            k1 = k + 1u;
            const float t0 = times[time_offset + k0];
            const float t1 = times[time_offset + k1];
            blend = fmath::min(fmath::max(fmath::div(t - t0, t1 - t0), 0.0f), 1.0f);
        }
    }

    const unsigned W = extent[0];
    const unsigned H = extent[1];
    const unsigned D = extent[2];
    Vec3f result = Vec3f::Zero();
    for (unsigned s = 0u; s < 2u; ++s) {
        const unsigned k = s == 0u ? k0 : k1;
        const float ws = s == 0u ? 1.0f - blend : blend;
        if (ws == 0.0f) {
            continue;
        }
        for (unsigned corner = 0u; corner < 8u; ++corner) {
            const unsigned ox = corner & 1u;
            const unsigned oy = (corner >> 1u) & 1u;
            const unsigned oz = (corner >> 2u) & 1u;
            const float wx = ox != 0u ? frac[0] : 1.0f - frac[0];
            const float wy = oy != 0u ? frac[1] : 1.0f - frac[1];
            const float wz = oz != 0u ? frac[2] : 1.0f - frac[2];
            const float w = ws * wx * wy * wz;
            const unsigned node =
                ((k * D + cell[2] + oz) * H + cell[1] + oy) * W + cell[0] + ox;
            const unsigned base = data_offset + 3u * node;
            result[0] += w * data[base + 0u];
            result[1] += w * data[base + 1u];
            result[2] += w * data[base + 2u];
        }
    }
    out = result;
    return true;
}

// Whether a source whose target bit is `bit` reaches a vertex whose target
// mask is `mask`. FIELD_ALL_TARGETS reaches every vertex.
[[seam::device_fn]] inline bool field_applies(unsigned bit, unsigned mask) {
    return bit >= FIELD_ALL_TARGETS || ((mask >> bit) & 1u) != 0u;
}

// The whole field for one vertex.
//
// `acceleration` and `air_velocity` are WRITTEN, never accumulated: this is
// the vertex's whole contribution for the step. `outside` is 1 for a free
// vertex that some grid targets but that lies outside every grid targeting
// it, which is what the per-frame log line counts; a vertex no grid can reach
// gets exactly the force the scripts (or nothing) give it, and that is said
// out loud rather than guessed at.
//
// EACH SOURCE CARRIES A TARGET BIT, and a vertex's `target_mask` says which
// bits reach it: a grid or script meant for some object groups only is
// skipped everywhere else. FIELD_ALL_TARGETS is the source that reaches every
// vertex, and a scene whose sources all do carries no mask at all.
//
// A FIX-PINNED VERTEX GETS ZERO, because its target is its pin and nothing
// here could move it; a weight of zero skips the evaluation for the same
// reason, which is also what makes a group's opt-out free.
[[seam::entry(i)]]
[[seam::device_fn]] inline void external_field(
    const Vec3f *current, const VertexProp *prop,
    const float *weight, unsigned has_weight,
    const unsigned *target_mask, unsigned has_mask,
    const unsigned *grid_header, const float *grid_box,
    const float *grid_times, const float *grid_data, unsigned grid_count,
    const unsigned *script_header, unsigned script_count,
    const unsigned *script_code, const float *script_constants,
    float time, float inv_world_scaling, float *acceleration, float *air_velocity,
    unsigned has_air, unsigned *outside, DiagHandle diag, unsigned i) {
    Vec3f accel = Vec3f::Zero();
    Vec3f air = Vec3f::Zero();
    unsigned out_of_domain = 0u;
    const float w = has_weight != 0u ? weight[i] : 1.0f;
    const unsigned mask = has_mask != 0u ? target_mask[i] : 0u;
    if (prop[i].fix_index == 0u && w != 0.0f) {
        const Vec3f position = current[i];
        // BACK TO SCENE UNITS. The solver simulates the scene multiplied by
        // World Scaling, and a grid's box and a script's formula are written
        // in the units the scene was authored in, so the position is divided
        // back here once for both. The result is an acceleration or an air
        // velocity, physical quantities World Scaling never touches.
        const float px = position[0] * inv_world_scaling;
        const float py = position[1] * inv_world_scaling;
        const float pz = position[2] * inv_world_scaling;
        bool targeted = false;
        bool inside_any = false;
        for (unsigned g = 0u; g < grid_count; ++g) {
            const unsigned *header = grid_header + FIELD_GRID_HEADER * g;
            if (!field_applies(header[7], mask)) {
                continue;
            }
            targeted = true;
            Vec3f sample = Vec3f::Zero();
            if (field_grid_sample(header, grid_box + FIELD_GRID_BOX * g,
                                  grid_times, grid_data, px, py, pz, time,
                                  sample)) {
                inside_any = true;
                if (header[4] == FIELD_KIND_AIR_VELOCITY) {
                    air += sample;
                } else {
                    accel += sample;
                }
            }
        }
        if (targeted && !inside_any) {
            out_of_domain = 1u;
        }
        for (unsigned k = 0u; k < script_count; ++k) {
            const unsigned *header = script_header + FIELD_SCRIPT_HEADER * k;
            if (!field_applies(header[4], mask)) {
                continue;
            }
            Vec3f value = Vec3f::Zero();
            const bool returned = field_script_eval(
                script_code, header[0], header[1], script_constants, header[2],
                header[3], px, py, pz, time, diag, value);
            DIAG_ASSERT4(diag, returned, static_cast<float>(i),
                         static_cast<float>(k), 8.0f, 0.0f);
            if (returned) {
                accel += value;
            }
        }
        accel *= w;
        air *= w;
        // A NON-FINITE FIELD IS A FAILED RUN, NOT A ZERO FORCE. A script that
        // divides by zero or takes the root of a negative produces NaN, and a
        // NaN in the target would surface frames later as an unrelated line
        // search failure; this names the vertex and the value instead.
        const bool finite = !fmath::isnan(accel[0]) && !fmath::isinf(accel[0]) &&
                            !fmath::isnan(accel[1]) && !fmath::isinf(accel[1]) &&
                            !fmath::isnan(accel[2]) && !fmath::isinf(accel[2]);
        DIAG_ASSERT4(diag, finite, static_cast<float>(i), px, py, pz);
        if (!finite) {
            accel = Vec3f::Zero();
        }
    }
    acceleration[3u * i + 0u] = accel[0];
    acceleration[3u * i + 1u] = accel[1];
    acceleration[3u * i + 2u] = accel[2];
    if (has_air != 0u) {
        air_velocity[3u * i + 0u] = air[0];
        air_velocity[3u * i + 1u] = air[1];
        air_velocity[3u * i + 2u] = air[2];
    }
    outside[i] = out_of_domain;
}
