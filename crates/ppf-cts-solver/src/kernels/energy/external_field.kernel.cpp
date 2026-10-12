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
// at the step's starting position, velocity and time.
//
// Two sources are summed, and a scene may carry either, both or neither:
//
//   * FIELD OBJECTS. Each is a Blender-style field (Force, Wind, Vortex,
//     Turbulence, Magnetic) evaluated EXACTLY at the vertex from its settings
//     and a track of per-frame samples (origin, axes, animatable values),
//     linear between samples and holding the end sample outside them. What
//     each type computes was measured on Blender as a black box. A Wind
//     feeds the air velocity; the rest accelerate.
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
// A FIELD OBJECT READS A VERTEX RELATIVE TO ITS ORIGIN. Its laws are functions
// of that difference, so the host uploads the origin of every track sample in
// solver units and the kernel subtracts it from the vertex before anything
// else is computed.
//
// A SCRIPT, AND A TURBULENCE IN GLOBAL COORDINATES, READ THE ABSOLUTE
// POSITION: a script is a formula the user wrote in world coordinates, and a
// global turbulence pins its pattern to the world.

#include "../contact/distance.hpp"
#include "../utility/vertex_normal.kernel.cpp"

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

// The field-object tables, spelled once more in `src/force_field.rs`.
enum FieldLayout : unsigned {
    // Words per field in the header table: type, shape, falloff, Z direction,
    // flags, seed, target bit, first track sample, track sample count, then
    // a Texture's image offset (in floats), width, height, mode, extension
    // and texture flags, or a Fluid's index and density flag (zero for every
    // other type).
    FIELD_HEADER = 15u,
    // Floats per field in the timing table: first sample's instant, spacing.
    FIELD_TIMING = 2u,
    // Floats per track sample in the origin table: the origin, in solver
    // units.
    FIELD_TRACK_ORIGIN = 3u,
    // Floats per track sample: axes X, Y, Z (columns, world), strength,
    // falloff power, min, max, radial power, radial min, radial max, size,
    // noise amount, noise scale X Y Z, wind factor, linear drag, quadratic
    // drag, harmonic damping, rest length, inflow, a Texture's nabla, the
    // origin's motion to the next sample. The noise scale is the field
    // object's scale along its local X, Y and Z: it sizes Noise Amount's
    // gusts and scales an Object Coordinates texture's lookup.
    FIELD_TRACK = 31u,
    FIELD_TRACK_STRENGTH = 9u,
    FIELD_TRACK_POWER = 10u,
    FIELD_TRACK_MIN = 11u,
    FIELD_TRACK_MAX = 12u,
    FIELD_TRACK_RADIAL_POWER = 13u,
    FIELD_TRACK_RADIAL_MIN = 14u,
    FIELD_TRACK_RADIAL_MAX = 15u,
    FIELD_TRACK_SIZE = 16u,
    FIELD_TRACK_NOISE = 17u,
    FIELD_TRACK_NOISE_SCALE = 18u,
    FIELD_TRACK_WIND_FACTOR = 21u,
    FIELD_TRACK_LINEAR_DRAG = 22u,
    FIELD_TRACK_QUADRATIC_DRAG = 23u,
    FIELD_TRACK_HARMONIC_DAMPING = 24u,
    FIELD_TRACK_REST_LENGTH = 25u,
    FIELD_TRACK_INFLOW = 26u,
    FIELD_TRACK_NABLA = 27u,
    FIELD_TRACK_DELTA = 28u,
    // Words per script in the script table: code offset, code length,
    // constant offset, constant count, target bit.
    FIELD_SCRIPT_HEADER = 5u,
    // A target bit of this value reaches every vertex; below it, the bit a
    // vertex's target mask must carry.
    FIELD_ALL_TARGETS = 32u,
};

enum FieldType : unsigned {
    FIELD_FORCE = 0u,
    FIELD_WIND = 1u,
    FIELD_VORTEX = 2u,
    FIELD_TURBULENCE = 3u,
    FIELD_MAGNETIC = 4u,
    // Evaluated at the Newton iterate by `external_field_implicit`, whose
    // force and Hessian join the momentum row.
    FIELD_DRAG = 5u,
    FIELD_HARMONIC = 6u,
    // An image read at the point.
    FIELD_TEXTURE = 7u,
    // A fluid's velocity volume, wholly implicit.
    FIELD_FLUID = 8u,
};

// A Fluid field's volume tables, spelled once more in `src/force_field.rs`
// and `src/driver/force_field.rs`. The volume is cut into BRICKS of
// FLUID_BRICK^3 cells; a frame is a table over the bricks, x fastest, naming
// each brick's slot or FLUID_EMPTY, and the slots' cells, x fastest inside a
// brick, four floats each (velocity in the domain's local axes, in solver
// units per second, then density). Two frames of each field are resident,
// the two bracketing the instant the step evaluates. A frame of zero cells
// is EMPTY: the fluid is absent there.
enum FieldFluid : unsigned {
    FLUID_BRICK = 8u,
    FLUID_BRICK_CELLS = 512u,
    FLUID_CHANNELS = 4u,
    FLUID_EMPTY = 0xffffffffu,
    // Words per resident frame: table offset, data offset, resolution x, y
    // and z.
    FLUID_FRAME_WORDS = 5u,
    // Words per field: two resident frames.
    FLUID_WORDS = 10u,
    // Floats per resident frame: the domain's axes (columns, world), the
    // cell along local x, y and z in scene units, then the domain corner in
    // solver units.
    FLUID_FRAME_FLOATS = 15u,
    // Where a resident frame's domain corner starts in its floats.
    FLUID_FRAME_CORNER = 12u,
    // Floats per field: the blend toward the second frame, then two frames.
    FLUID_FLOATS = 31u,
    // The header words a Fluid field uses in place of a Texture's: its index
    // into the fluid tables, and whether the density weighs the push.
    FIELD_HEADER_FLUID_INDEX = 9u,
    FIELD_HEADER_FLUID_DENSITY = 10u,
};

enum FieldTexture : unsigned {
    FIELD_TEXTURE_RGB = 0u,
    FIELD_TEXTURE_GRADIENT = 1u,
    FIELD_TEXTURE_CURL = 2u,
    FIELD_EXTEND = 0u,
    FIELD_REPEAT = 1u,
    FIELD_CLIP = 2u,
    FIELD_CLIP_CUBE = 3u,
    FIELD_TEXTURE_INTERPOLATE = 1u,
    FIELD_TEXTURE_OBJECT_COORDINATES = 2u,
    // Blender's axes (Z up) for the lookup in world coordinates and for the
    // vector the image gives; solver axes (Y up) without it.
    FIELD_TEXTURE_Z_UP = 4u,
};

enum FieldShape : unsigned {
    FIELD_POINT = 0u,
    FIELD_PLANE = 1u,
};

enum FieldFalloff : unsigned {
    FIELD_SPHERE = 0u,
    FIELD_TUBE = 1u,
    FIELD_CONE = 2u,
};

enum FieldZDirection : unsigned {
    FIELD_Z_BOTH = 0u,
    FIELD_Z_POSITIVE = 1u,
    FIELD_Z_NEGATIVE = 2u,
};

enum FieldFlag : unsigned {
    FIELD_USE_MIN = 1u,
    FIELD_USE_MAX = 2u,
    FIELD_USE_RADIAL_MIN = 4u,
    FIELD_USE_RADIAL_MAX = 8u,
    FIELD_GRAVITATION = 16u,
    FIELD_GLOBAL_COORDINATES = 32u,
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

// A field's falloff factor at distance `d`: 1 inside the
// minimum, 0 past the maximum, else `(1 + max(d - m, 0))^-power` with `m` the
// minimum when it is used and 0 otherwise. The base is at least 1, so the
// power goes through exp and log with no domain question.
[[seam::device_fn]] inline float field_falloff(float d, bool use_min, float dmin,
                                               bool use_max, float dmax,
                                               float power) {
    if (use_max && d > dmax) {
        return 0.0f;
    }
    if (use_min && d < dmin) {
        return 1.0f;
    }
    if (power == 0.0f) {
        return 1.0f;
    }
    const float base = 1.0f + fmath::max(d - (use_min ? dmin : 0.0f), 0.0f);
    return fmath::exp(-power * fmath::log(base));
}

// Three decorrelated noise channels at one point and w = 0: the frontend's
// `noise_vector(x, y, z, 1, seed)`, which a Turbulence field is made of.
[[seam::device_fn]] inline Vec3f field_noise_vector(float px, float py,
                                                    float pz, unsigned seed) {
    Vec3f unused;
    Vec3f out;
    out[0] = field_fractal_noise(px, py, pz, 0.0f, 1u, seed, unused);
    out[1] = field_fractal_noise(px + 31.416f, py + 47.853f, pz + 12.679f, 0.0f,
                                 1u, seed + 7919u, unused);
    out[2] = field_fractal_noise(px + 71.231f, py + 5.117f, pz + 93.402f, 0.0f,
                                 1u, seed + 2u * 7919u, unused);
    return out;
}

// One field at one point: its frame and settings blended at the instant, the
// point in its local frame, and its falloff weight. Both entries
// read a field through this one function, so the explicit evaluation at the
// step's start and the implicit one at the Newton iterate cannot disagree
// about the track, the frame, the origin difference or the falloff.
struct FieldPoint {
    unsigned type;
    unsigned shape;
    unsigned flags;
    unsigned seed;
    Vec3f ax;
    Vec3f ay;
    Vec3f az;
    float px;
    float py;
    float pz;
    float r;
    float distance;
    float axial;
    float W;
    // 0 where a Gravitation field meets the point exactly at its origin,
    // whose weight is infinite there; the caller fails the run.
    unsigned weight_defined;
    float strength;
    float size;
    float noise_amount;
    float wind_factor;
    float linear_drag;
    float quadratic_drag;
    float harmonic_damping;
    float rest_length;
    float inflow;
    float nabla;
    Vec3f noise_scale;
};

[[seam::device_fn]] inline float field_blend(const float *field_track,
                                             unsigned t0, unsigned t1, float b,
                                             unsigned at) {
    return (1.0f - b) * field_track[t0 + at] + b * field_track[t1 + at];
}

[[seam::device_fn]] inline void field_point(
    const unsigned *field_header, const float *field_timing,
    const float *field_origin, const float *field_track, unsigned f,
    const Vec3f &position, float time, float inv_world_scaling,
    FieldPoint &out) {
    const unsigned h = FIELD_HEADER * f;
    out.type = field_header[h + 0u];
    out.shape = field_header[h + 1u];
    const unsigned falloff = field_header[h + 2u];
    const unsigned z_direction = field_header[h + 3u];
    out.flags = field_header[h + 4u];
    out.seed = field_header[h + 5u];
    const unsigned first = field_header[h + 7u];
    const unsigned samples = field_header[h + 8u];

    // The two track samples bracketing `time`, and the blend between them;
    // outside the track the end sample holds.
    unsigned k = 0u;
    float b = 0.0f;
    if (samples > 1u) {
        const float u = fmath::div(time - field_timing[FIELD_TIMING * f],
                                   field_timing[FIELD_TIMING * f + 1u]);
        if (u >= static_cast<float>(samples - 1u)) {
            k = samples - 1u;
        } else if (u > 0.0f) {
            k = static_cast<unsigned>(u);
            b = fmath::min(fmath::max(u - static_cast<float>(k), 0.0f), 1.0f);
        }
    }
    const unsigned s0 = first + k;
    const unsigned s1 = k + 1u < samples ? s0 + 1u : s0;
    const unsigned t0 = FIELD_TRACK * s0;
    const unsigned t1 = FIELD_TRACK * s1;

    // The frame: the two samples' axes blended, then made orthonormal again
    // so a rotation between samples is never a shear.
    Vec3f ax;
    Vec3f az;
    for (unsigned c = 0u; c < 3u; ++c) {
        ax[c] = field_blend(field_track, t0, t1, b, c);
        az[c] = field_blend(field_track, t0, t1, b, 6u + c);
    }
    az *= fmath::div(1.0f, az.norm());
    ax -= ax.dot(az) * az;
    ax *= fmath::div(1.0f, ax.norm());
    out.ax = ax;
    out.az = az;
    out.ay = az.cross(ax);

    out.strength = field_blend(field_track, t0, t1, b, FIELD_TRACK_STRENGTH);
    const float power = field_blend(field_track, t0, t1, b, FIELD_TRACK_POWER);
    const float dmin = field_blend(field_track, t0, t1, b, FIELD_TRACK_MIN);
    const float dmax = field_blend(field_track, t0, t1, b, FIELD_TRACK_MAX);
    const float radial_power =
        field_blend(field_track, t0, t1, b, FIELD_TRACK_RADIAL_POWER);
    const float radial_min = field_blend(field_track, t0, t1, b, FIELD_TRACK_RADIAL_MIN);
    const float radial_max = field_blend(field_track, t0, t1, b, FIELD_TRACK_RADIAL_MAX);
    out.size = field_blend(field_track, t0, t1, b, FIELD_TRACK_SIZE);
    out.noise_amount = field_blend(field_track, t0, t1, b, FIELD_TRACK_NOISE);
    out.wind_factor = field_blend(field_track, t0, t1, b, FIELD_TRACK_WIND_FACTOR);
    out.linear_drag = field_blend(field_track, t0, t1, b, FIELD_TRACK_LINEAR_DRAG);
    out.quadratic_drag =
        field_blend(field_track, t0, t1, b, FIELD_TRACK_QUADRATIC_DRAG);
    out.harmonic_damping =
        field_blend(field_track, t0, t1, b, FIELD_TRACK_HARMONIC_DAMPING);
    out.rest_length = field_blend(field_track, t0, t1, b, FIELD_TRACK_REST_LENGTH);
    out.inflow = field_blend(field_track, t0, t1, b, FIELD_TRACK_INFLOW);
    out.nabla = field_blend(field_track, t0, t1, b, FIELD_TRACK_NABLA);
    for (unsigned c = 0u; c < 3u; ++c) {
        out.noise_scale[c] =
            field_blend(field_track, t0, t1, b, FIELD_TRACK_NOISE_SCALE + c);
    }

    // The point relative to the field's origin, in scene units and world
    // axes: the difference at sample s0, less the part of the origin's
    // motion to s1 that the blend has covered.
    Vec3f d;
    for (unsigned c = 0u; c < 3u; ++c) {
        d[c] = (position[c] - field_origin[FIELD_TRACK_ORIGIN * s0 + c]) *
                   inv_world_scaling -
               b * field_track[t0 + FIELD_TRACK_DELTA + c];
    }
    out.px = d.dot(ax);
    out.py = d.dot(out.ay);
    out.pz = d.dot(az);
    out.r = fmath::sqrt(out.px * out.px + out.py * out.py);
    out.distance = fmath::sqrt(out.px * out.px + out.py * out.py + out.pz * out.pz);

    // The falloff weight. A Plane Vortex falls off with the
    // distance from its axis, measured.
    float axial = fmath::abs(out.pz);
    if (out.type == FIELD_VORTEX && out.shape == FIELD_PLANE) {
        axial = out.r;
    } else if (falloff == FIELD_SPHERE && out.shape == FIELD_POINT) {
        axial = out.distance;
    }
    float W = field_falloff(axial, (out.flags & FIELD_USE_MIN) != 0u, dmin,
                            (out.flags & FIELD_USE_MAX) != 0u, dmax, power);
    if (falloff == FIELD_TUBE || falloff == FIELD_CONE) {
        // A Cone measures its radial term as the angle from +Z IN DEGREES, its
        // radial limits too (measured).
        const float radial = falloff == FIELD_TUBE
                                 ? out.r
                                 : fmath::atan2(out.r, out.pz) * 57.2957795f;
        W *= field_falloff(radial, (out.flags & FIELD_USE_RADIAL_MIN) != 0u,
                           radial_min, (out.flags & FIELD_USE_RADIAL_MAX) != 0u,
                           radial_max, radial_power);
    }
    if ((z_direction == FIELD_Z_POSITIVE && out.pz < 0.0f) ||
        (z_direction == FIELD_Z_NEGATIVE && out.pz > 0.0f)) {
        W = 0.0f;
    }
    out.weight_defined = 1u;
    if ((out.flags & FIELD_GRAVITATION) != 0u && W != 0.0f) {
        // Exact and unbounded, as measured. At d = 0 the weight is infinite,
        // and the caller fails the run rather than let a
        // direction guard turn it into a quiet zero.
        const float gd = out.shape == FIELD_PLANE ? axial : out.distance;
        if (gd > 0.0f) {
            W = fmath::div(W, gd * gd);
        } else {
            out.weight_defined = 0u;
        }
    }
    out.axial = axial;
    out.W = W;
}

// One cell of a resident fluid frame, its index clamped into the domain:
// zero velocity and density where its brick is empty.
[[seam::device_fn]] inline void field_fluid_cell(
    const unsigned *fluid_table, const float *fluid_data, unsigned table_offset,
    unsigned data_offset, int nx, int ny, int nz, int i, int j, int k,
    Vec3f &velocity, float &density) {
    i = i < 0 ? 0 : (i >= nx ? nx - 1 : i);
    j = j < 0 ? 0 : (j >= ny ? ny - 1 : j);
    k = k < 0 ? 0 : (k >= nz ? nz - 1 : k);
    const unsigned b = FLUID_BRICK;
    const unsigned bx = (static_cast<unsigned>(nx) + b - 1u) / b;
    const unsigned by = (static_cast<unsigned>(ny) + b - 1u) / b;
    const unsigned ui = static_cast<unsigned>(i);
    const unsigned uj = static_cast<unsigned>(j);
    const unsigned uk = static_cast<unsigned>(k);
    const unsigned slot =
        fluid_table[table_offset + ui / b + bx * (uj / b + by * (uk / b))];
    if (slot == FLUID_EMPTY) {
        velocity = Vec3f::Zero();
        density = 0.0f;
        return;
    }
    const unsigned at =
        data_offset +
        FLUID_CHANNELS * (slot * FLUID_BRICK_CELLS + ui % b + b * (uj % b + b * (uk % b)));
    velocity = Vec3f(fluid_data[at], fluid_data[at + 1u], fluid_data[at + 2u]);
    density = fluid_data[at + 3u];
}

// One resident fluid frame at a point: the velocity in WORLD axes, the
// density, trilinear between cell centers and clamped at the outer cells,
// and whether the point is inside the domain at all; outside, all three are
// zero. The point's offset from the domain
// corner is taken as its offset from a field's origin is, and each
// local axis has its own cell, so a non-cubic domain's cells are boxes.
//
// AN EMPTY FRAME (a zero resolution) IS OUTSIDE EVERYWHERE, decided before
// anything else of it is read: it has no table and no bricks.
[[seam::device_fn]] inline void field_fluid_frame(
    const unsigned *fluid_words, const float *fluid_floats,
    const unsigned *fluid_table, const float *fluid_data, unsigned wb, unsigned fb,
    const Vec3f &position, float inv_world_scaling, Vec3f &velocity,
    float &density, float &presence) {
    velocity = Vec3f::Zero();
    density = 0.0f;
    presence = 0.0f;
    const int n[3] = {static_cast<int>(fluid_words[wb + 2u]),
                      static_cast<int>(fluid_words[wb + 3u]),
                      static_cast<int>(fluid_words[wb + 4u])};
    if (n[0] <= 0 || n[1] <= 0 || n[2] <= 0) {
        return;
    }
    Vec3f d;
    for (unsigned c = 0u; c < 3u; ++c) {
        d[c] = (position[c] - fluid_floats[fb + FLUID_FRAME_CORNER + c]) *
               inv_world_scaling;
    }
    float g[3];
    for (unsigned c = 0u; c < 3u; ++c) {
        const Vec3f axis(fluid_floats[fb + 3u * c], fluid_floats[fb + 3u * c + 1u],
                         fluid_floats[fb + 3u * c + 2u]);
        g[c] = d.dot(axis) * fmath::div(1.0f, fluid_floats[fb + 9u + c]);
        if (!(g[c] >= 0.0f && g[c] <= static_cast<float>(n[c]))) {
            return;
        }
    }
    presence = 1.0f;
    int i0[3];
    float t[3];
    for (unsigned c = 0u; c < 3u; ++c) {
        const float q = g[c] - 0.5f;
        const float fl = field_floor(q);
        i0[c] = static_cast<int>(fl);
        t[c] = q - fl;
    }
    Vec3f local = Vec3f::Zero();
    for (unsigned corner = 0u; corner < 8u; ++corner) {
        const unsigned ox = corner & 1u;
        const unsigned oy = (corner >> 1u) & 1u;
        const unsigned oz = (corner >> 2u) & 1u;
        const float w = (ox != 0u ? t[0] : 1.0f - t[0]) *
                        (oy != 0u ? t[1] : 1.0f - t[1]) *
                        (oz != 0u ? t[2] : 1.0f - t[2]);
        Vec3f u;
        float r;
        field_fluid_cell(fluid_table, fluid_data, fluid_words[wb], fluid_words[wb + 1u],
                         n[0], n[1], n[2], i0[0] + static_cast<int>(ox),
                         i0[1] + static_cast<int>(oy), i0[2] + static_cast<int>(oz), u,
                         r);
        local += w * u;
        density += w * r;
    }
    // The grid holds the domain's LOCAL axes; turn it to the world's.
    for (unsigned c = 0u; c < 3u; ++c) {
        velocity[0] += local[c] * fluid_floats[fb + 3u * c];
        velocity[1] += local[c] * fluid_floats[fb + 3u * c + 1u];
        velocity[2] += local[c] * fluid_floats[fb + 3u * c + 2u];
    }
}

// A Fluid field's volume at a point, blended between its two resident
// frames by the host's blend for the instant the step evaluates: the
// velocity, the density and the presence (1 inside the domain, 0 outside).
[[seam::device_fn]] inline void field_fluid(
    const unsigned *fluid_words, const float *fluid_floats,
    const unsigned *fluid_table, const float *fluid_data, unsigned index,
    const Vec3f &position, float inv_world_scaling, Vec3f &velocity,
    float &density, float &presence) {
    const unsigned wb = FLUID_WORDS * index;
    const unsigned fb = FLUID_FLOATS * index;
    const float b = fluid_floats[fb];
    Vec3f u0;
    Vec3f u1;
    float r0;
    float r1;
    float p0;
    float p1;
    field_fluid_frame(fluid_words, fluid_floats, fluid_table, fluid_data, wb, fb + 1u,
                      position, inv_world_scaling, u0, r0, p0);
    field_fluid_frame(fluid_words, fluid_floats, fluid_table, fluid_data,
                      wb + FLUID_FRAME_WORDS, fb + 1u + FLUID_FRAME_FLOATS, position,
                      inv_world_scaling, u1, r1, p1);
    velocity = (1.0f - b) * u0 + b * u1;
    density = (1.0f - b) * r0 + b * r1;
    presence = (1.0f - b) * p0 + b * p1;
}

// One pixel of an RGB image, its index brought inside by the extension; a
// CLIP extension reads color 0 for a pixel outside.
[[seam::device_fn]] inline Vec3f field_texel(const float *field_image, unsigned offset,
                                             int width, int height, unsigned extension,
                                             int ix, int iy) {
    if (extension == FIELD_REPEAT) {
        ix = ((ix % width) + width) % width;
        iy = ((iy % height) + height) % height;
    } else if (extension == FIELD_EXTEND) {
        ix = ix < 0 ? 0 : (ix >= width ? width - 1 : ix);
        iy = iy < 0 ? 0 : (iy >= height ? height - 1 : iy);
    } else if (ix < 0 || iy < 0 || ix >= width || iy >= height) {
        return Vec3f::Zero();
    }
    const unsigned at = offset + 3u * static_cast<unsigned>(iy * width + ix);
    return Vec3f(field_image[at], field_image[at + 1u], field_image[at + 2u]);
}

// The image's color at texture coordinates (x, y, z), the image spanning x
// and y in [-1, 1]. The nearest pixel without
// interpolation, matching Blender; bilinear between pixel centers with it,
// which is this project's own sampling (Blender filters by its filter size).
[[seam::device_fn]] inline Vec3f field_texture_color(
    const float *field_image, unsigned offset, unsigned width, unsigned height,
    unsigned extension, bool interpolate, float x, float y, float z) {
    if ((extension == FIELD_CLIP || extension == FIELD_CLIP_CUBE) &&
        (x < -1.0f || x > 1.0f || y < -1.0f || y > 1.0f)) {
        return Vec3f::Zero();
    }
    if (extension == FIELD_CLIP_CUBE && (z < -1.0f || z > 1.0f)) {
        return Vec3f::Zero();
    }
    const int w = static_cast<int>(width);
    const int h = static_cast<int>(height);
    const float u = 0.5f * (x + 1.0f) * static_cast<float>(w);
    const float v = 0.5f * (y + 1.0f) * static_cast<float>(h);
    if (!interpolate) {
        return field_texel(field_image, offset, w, h, extension,
                           static_cast<int>(field_floor(u)),
                           static_cast<int>(field_floor(v)));
    }
    const float fu = u - 0.5f;
    const float fv = v - 0.5f;
    const float bu = field_floor(fu);
    const float bv = field_floor(fv);
    const float tu = fu - bu;
    const float tv = fv - bv;
    const int i0 = static_cast<int>(bu);
    const int j0 = static_cast<int>(bv);
    return (1.0f - tu) * (1.0f - tv) *
               field_texel(field_image, offset, w, h, extension, i0, j0) +
           tu * (1.0f - tv) *
               field_texel(field_image, offset, w, h, extension, i0 + 1, j0) +
           (1.0f - tu) * tv *
               field_texel(field_image, offset, w, h, extension, i0, j0 + 1) +
           tu * tv * field_texel(field_image, offset, w, h, extension, i0 + 1, j0 + 1);
}

[[seam::device_fn]] inline bool field_finite(const Vec3f &v) {
    return !fmath::isnan(v[0]) && !fmath::isinf(v[0]) && !fmath::isnan(v[1]) &&
           !fmath::isinf(v[1]) && !fmath::isnan(v[2]) && !fmath::isinf(v[2]);
}

// Whether a source whose target bit is `bit` reaches a vertex whose target
// mask is `mask`. FIELD_ALL_TARGETS reaches every vertex.
[[seam::device_fn]] inline bool field_applies(unsigned bit, unsigned mask) {
    return bit >= FIELD_ALL_TARGETS || ((mask >> bit) & 1u) != 0u;
}

// The whole field for one vertex.
//
// `acceleration` and `air_velocity` are WRITTEN, never accumulated: this is
// the vertex's whole contribution for the step.
//
// EACH SOURCE CARRIES A TARGET BIT, and a vertex's `target_mask` says which
// bits reach it: a field object or script meant for some object groups only
// is skipped everywhere else. FIELD_ALL_TARGETS is the source that reaches
// every vertex, and a scene whose sources all do carries no mask at all.
//
// A FIX-PINNED VERTEX GETS ZERO, because its target is its pin and nothing
// here could move it; a weight of zero skips the evaluation for the same
// reason, which is also what makes a group's opt-out free.
//
// WIND FACTOR WEIGHS A PUSH BY HOW THE SURFACE FACES IT (measured): with `n`
// the vertex's area-weighted normal at the step's starting pose, folded from
// the same vertex-face table the drag reads, a field's acceleration becomes
// `(1 - wf) a + wf (a . n) n`. A vertex that presents no surface (on a rod,
// or inside a solid) has a zero normal, so there is nothing to weigh and its
// push stays whole. A Wind does not read its own Wind Factor: it acts through
// the drag, which already weighs a surface by how it faces the flow.
//
// A NON-FINITE FIELD IS A FAILED RUN, NOT A ZERO FORCE. A script that divides
// by zero, or a Gravitation field meeting a vertex exactly at its origin
// (Blender's `1 / d^2` is unbounded, and this solver evaluates
// it exactly), produces a NaN or an infinity, which in the target would
// surface frames later as an unrelated line search failure; this names the
// vertex and the source instead.
[[seam::entry(i)]]
[[seam::device_fn]] inline void external_field(
    const Vec3f *current, const Vec3f *previous, float previous_dt,
    const VertexProp *prop, const unsigned *neighbor_index,
    const unsigned *neighbor_offset, unsigned has_neighbor, const unsigned *face,
    const float *weight, unsigned has_weight,
    const unsigned *target_mask, unsigned has_mask,
    const unsigned *field_header, const float *field_timing,
    const float *field_origin, const float *field_track, unsigned field_count,
    const float *field_image,
    const unsigned *script_header, unsigned script_count,
    const unsigned *script_code, const float *script_constants,
    float time, float inv_world_scaling, float *acceleration, float *air_velocity,
    unsigned has_air, DiagHandle diag, unsigned i) {
    Vec3f accel = Vec3f::Zero();
    Vec3f air = Vec3f::Zero();
    const float w = has_weight != 0u ? weight[i] : 1.0f;
    const unsigned mask = has_mask != 0u ? target_mask[i] : 0u;
    if (prop[i].fix_index == 0u && w != 0.0f) {
        const Vec3f position = current[i];
        // The surface normal Wind Factor weighs by, folded only when a field
        // asks for it below.
        bool normal_ready = false;
        Vec3f surface_normal = Vec3f::Zero();
        for (unsigned f = 0u; f < field_count; ++f) {
            const unsigned h = FIELD_HEADER * f;
            // A Fluid field is wholly implicit (external_field_implicit).
            if (field_header[h] == FIELD_FLUID ||
                !field_applies(field_header[h + 6u], mask)) {
                continue;
            }
            FieldPoint fp;
            field_point(field_header, field_timing, field_origin, field_track, f,
                        position, time, inv_world_scaling, fp);
            DIAG_ASSERT4(diag, fp.weight_defined != 0u, static_cast<float>(i),
                         static_cast<float>(f), fp.axial, 11.0f);
            const unsigned type = fp.type;
            const unsigned shape = fp.shape;
            const unsigned flags = fp.flags;
            const unsigned seed = fp.seed;
            const Vec3f ax = fp.ax;
            const Vec3f ay = fp.ay;
            const Vec3f az = fp.az;
            const float px = fp.px;
            const float py = fp.py;
            const float pz = fp.pz;
            const float r = fp.r;
            const float distance = fp.distance;
            const float axial = fp.axial;
            const float W = fp.W;
            const float strength = fp.strength;
            const float size = fp.size;
            const float noise_amount = fp.noise_amount;
            const Vec3f noise_scale = fp.noise_scale;
            const bool plane_vortex = type == FIELD_VORTEX && shape == FIELD_PLANE;

            // The field type's law, in the field's local frame.
            Vec3f local = Vec3f::Zero();
            Vec3f local_air = Vec3f::Zero();
            Vec3f tangent = Vec3f::Zero();
            if (r > 0.0f) {
                tangent[0] = fmath::div(-py, r);
                tangent[1] = fmath::div(px, r);
            }
            const float sw = strength * W;
            if (type == FIELD_FORCE) {
                if (shape == FIELD_PLANE) {
                    local[2] = pz >= 0.0f ? sw : -sw;
                } else if (distance > 0.0f) {
                    local[0] = sw * fmath::div(px, distance);
                    local[1] = sw * fmath::div(py, distance);
                    local[2] = sw * fmath::div(pz, distance);
                }
            } else if (type == FIELD_WIND) {
                local_air[2] = sw;
            } else if (type == FIELD_VORTEX) {
                if (plane_vortex) {
                    // A rotation with its centripetal term: tangent u and an
                    // inward |u|^2 / r (measured).
                    const float speed = sw * r;
                    local = speed * tangent;
                    if (r > 0.0f) {
                        const float inward = fmath::div(speed * speed, r);
                        local[0] -= inward * fmath::div(px, r);
                        local[1] -= inward * fmath::div(py, r);
                    }
                } else {
                    local = (sw * distance) * tangent;
                }
            } else if (type == FIELD_TURBULENCE) {
                Vec3f q;
                if ((flags & FIELD_GLOBAL_COORDINATES) != 0u) {
                    // The pattern is pinned to the world by definition here,
                    // so the absolute position is the argument.
                    for (unsigned c = 0u; c < 3u; ++c) {
                        q[c] = fmath::div(position[c] *
                                              inv_world_scaling,
                                          size);
                    }
                    const Vec3f n = field_noise_vector(q[0], q[1], q[2], seed);
                    // A world-axis pattern, turned into the local frame so
                    // the common rotation back below leaves it as it is.
                    local[0] = sw * n.dot(ax);
                    local[1] = sw * n.dot(ay);
                    local[2] = sw * n.dot(az);
                } else {
                    local = sw * field_noise_vector(fmath::div(px, size),
                                                    fmath::div(py, size),
                                                    fmath::div(pz, size), seed);
                }
            } else if (type == FIELD_MAGNETIC) {
                DIAG_ASSERT4(diag, previous_dt > 0.0f, static_cast<float>(i),
                             static_cast<float>(f), previous_dt, 10.0f);
                // Thread-space copies: MSL will not bind a device-space
                // element to the reference the difference takes.
                // The velocity is PHYSICAL (solver units per second), as the
                // drag compares the wind against: World Scaling scales a
                // field's lengths, never a speed.
                const Vec3f before = previous[i];
                const Vec3f v = fmath::div(1.0f, previous_dt) *
                                (position - before).cast<float>();
                const Vec3f vl(v.dot(ax), v.dot(ay), v.dot(az));
                const Vec3f axis = shape == FIELD_PLANE ? Vec3f(0.0f, 0.0f, 1.0f)
                                                        : tangent;
                local = sw * vl.cross(axis);
            } else if (type == FIELD_TEXTURE) {
                const unsigned hh = FIELD_HEADER * f;
                const unsigned image_offset = field_header[hh + 9u];
                const unsigned width = field_header[hh + 10u];
                const unsigned height = field_header[hh + 11u];
                const unsigned mode = field_header[hh + 12u];
                const unsigned extension = field_header[hh + 13u];
                const unsigned texture_flags = field_header[hh + 14u];
                const bool interpolate = (texture_flags & FIELD_TEXTURE_INTERPOLATE) != 0u;
                const bool z_up = (texture_flags & FIELD_TEXTURE_Z_UP) != 0u;
                // WORLD coordinates by default, which pins the image to the
                // world by definition, so the absolute position is the
                // argument. With Z up the world is Blender's: the solver
                // point (x, y, z) is Blender's (x, -z, y), so the image spans
                // Blender's horizontal x and y.
                //
                // OBJECT COORDINATES read the field object's own space, its
                // SCALE INCLUDED: the point in the field's local frame divided
                // axis by axis by the object's scale along that axis (the
                // noise scale), which together apply the object's full
                // inverse matrix (measured: Blender looks the image up there,
                // so a scaled object stretches the pattern). Nabla steps in
                // that scaled space too. The field's own frame is the same
                // frame in either axis convention.
                Vec3f q;
                if ((texture_flags & FIELD_TEXTURE_OBJECT_COORDINATES) != 0u) {
                    q = Vec3f(fmath::div(px, noise_scale[0]),
                              fmath::div(py, noise_scale[1]),
                              fmath::div(pz, noise_scale[2]));
                } else {
                    Vec3f world;
                    for (unsigned c = 0u; c < 3u; ++c) {
                        world[c] = position[c] * inv_world_scaling;
                    }
                    q = z_up ? Vec3f(world[0], -world[2], world[1]) : world;
                }
                const Vec3f c0 = field_texture_color(field_image, image_offset, width,
                                                     height, extension, interpolate,
                                                     q[0], q[1], q[2]);
                // THE VECTOR IS IN WORLD AXES (measured), so it is turned into
                // the local frame here for the common rotation back below.
                Vec3f tex = Vec3f::Zero();
                if (mode == FIELD_TEXTURE_RGB) {
                    tex = sw * (Vec3f(0.5f, 0.5f, 0.5f) - c0);
                } else {
                    const float n = fp.nabla;
                    // Forward differences along the coordinate axes:
                    // d[k][ch] is channel ch's derivative along axis k.
                    float d[3][3];
                    for (unsigned k = 0u; k < 3u; ++k) {
                        Vec3f qk = q;
                        qk[k] += n;
                        const Vec3f ck = field_texture_color(
                            field_image, image_offset, width, height, extension,
                            interpolate, qk[0], qk[1], qk[2]);
                        for (unsigned ch = 0u; ch < 3u; ++ch) {
                            d[k][ch] = fmath::div(ck[ch] - c0[ch], n);
                        }
                    }
                    if (mode == FIELD_TEXTURE_GRADIENT) {
                        for (unsigned k = 0u; k < 3u; ++k) {
                            tex[k] = -sw * (d[k][0] + d[k][1] + d[k][2]) * (1.0f / 3.0f);
                        }
                    } else {
                        tex[0] = sw * (d[1][2] - d[2][1]);
                        tex[1] = sw * (d[2][0] - d[0][2]);
                        tex[2] = sw * (d[0][1] - d[1][0]);
                    }
                }
                // With Z up the vector is a Blender-world one, Blender's
                // (x, y, z) being the solver's (x, z, -y): a rotation, so a
                // gradient and a curl turn with it alike.
                if (z_up) {
                    tex = Vec3f(tex[0], tex[2], -tex[1]);
                }
                local[0] = tex.dot(ax);
                local[1] = tex.dot(ay);
                local[2] = tex.dot(az);
            }

            // Noise Amount: a curl-noise vector, a ratio of the
            // field's own strength, on the field's output. Its length is the
            // turbulence's size on a Turbulence and the object's scale
            // otherwise; on a Wind the pattern is carried by the flow.
            if (noise_amount != 0.0f && W != 0.0f) {
                Vec3f q;
                if (type == FIELD_TURBULENCE) {
                    q = Vec3f(fmath::div(px, size), fmath::div(py, size),
                              fmath::div(pz, size));
                } else {
                    const float carried = type == FIELD_WIND ? strength * time : 0.0f;
                    q = Vec3f(fmath::div(px, noise_scale[0]),
                              fmath::div(py, noise_scale[1]),
                              fmath::div(pz - carried, noise_scale[2]));
                }
                const Vec3f gust =
                    (noise_amount * fmath::abs(strength) * W) *
                    field_curl_noise(q[0], q[1], q[2], 0.0f, 1u, seed);
                if (type == FIELD_WIND) {
                    local_air += gust;
                } else {
                    local += gust;
                }
            }

            Vec3f world = local[0] * ax + local[1] * ay + local[2] * az;
            const float wind_factor = fp.wind_factor;
            if (wind_factor != 0.0f && type != FIELD_WIND) {
                if (!normal_ready && has_neighbor != 0u) {
                    Vec3f sum = Vec3f::Zero();
                    for (unsigned j = neighbor_offset[i]; j < neighbor_offset[i + 1u];
                         ++j) {
                        const unsigned fc = neighbor_index[j];
                        // Thread-space copies, as the drag's fold takes them:
                        // MSL will not bind a device-space element to the
                        // helper's reference parameters.
                        const Vec3f z0 = current[face[3u * fc + 0u]];
                        const Vec3f z1 = current[face[3u * fc + 1u]];
                        const Vec3f z2 = current[face[3u * fc + 2u]];
                        sum += vertex_normal_face_term(z0, z1, z2);
                    }
                    surface_normal = vertex_normal_finalize(sum);
                }
                normal_ready = true;
                if (!surface_normal.isZero()) {
                    world = (1.0f - wind_factor) * world +
                            (wind_factor * world.dot(surface_normal)) * surface_normal;
                }
            }
            const Vec3f world_air =
                local_air[0] * ax + local_air[1] * ay + local_air[2] * az;
            const bool finite = field_finite(world) && field_finite(world_air);
            DIAG_ASSERT4(diag, finite, static_cast<float>(i), static_cast<float>(f),
                         W, axial);
            if (finite) {
                accel += world;
                air += world_air;
            }
        }
        for (unsigned k = 0u; k < script_count; ++k) {
            const unsigned *header = script_header + FIELD_SCRIPT_HEADER * k;
            if (!field_applies(header[4], mask)) {
                continue;
            }
            const float px = position[0] * inv_world_scaling;
            const float py = position[1] * inv_world_scaling;
            const float pz = position[2] * inv_world_scaling;
            Vec3f value = Vec3f::Zero();
            const bool returned = field_script_eval(
                script_code, header[0], header[1], script_constants, header[2],
                header[3], px, py, pz, time, diag, value);
            DIAG_ASSERT4(diag, returned, static_cast<float>(i),
                         static_cast<float>(k), 8.0f, 0.0f);
            const bool finite = field_finite(value);
            DIAG_ASSERT4(diag, finite, static_cast<float>(i), px, py, pz);
            if (returned && finite) {
                accel += value;
            }
        }
        accel *= w;
        air *= w;
    }
    acceleration[3u * i + 0u] = accel[0];
    acceleration[3u * i + 1u] = accel[1];
    acceleration[3u * i + 2u] = accel[2];
    if (has_air != 0u) {
        air_velocity[3u * i + 0u] = air[0];
        air_velocity[3u * i + 1u] = air[1];
        air_velocity[3u * i + 2u] = air[2];
    }
}
