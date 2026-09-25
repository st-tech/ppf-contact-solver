# File: _noise_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Gradient noise, fractal noise and curl noise for force fields.

THIS FILE EXISTS TWICE, BYTE FOR BYTE: as ``frontend/_noise_.py`` and as
``blender_addon/core/noise.py``, because the add-on ships without the
frontend and both need it (the frontend's compiler cross-checks a script's
bytecode against the Python function, and the add-on draws the script and
samples its Turbulence field). ``addon_host_tests/_force_field_noise_copies_.py``
fails when the two differ. The solver's kernel
(``crates/ppf-cts-solver/src/kernels/energy/external_field.kernel.cpp``)
implements the SAME algorithm in single precision, so a script's noise on the
solver matches this one to float32 rounding.

This is this project's own noise, written from the textbook construction and
sharing no code with any other implementation: a four-dimensional grid of
pseudo-random gradients, blended with the quintic fade ``6t^5 - 15t^4 +
10t^3``. The first three coordinates are space; the fourth, ``w``, is how far
the pattern has EVOLVED, so moving along it changes the pattern in place
rather than sliding it. Every constant below is part of the definition;
changing one changes every field that uses it, on all three copies at once.

* ``noise(x, y, z, octaves=1, seed=0, time=0.0, frequency=1.0, decay=0.0)``:
  a smooth scalar in about [-1, 1]. With more octaves it sums finer copies at
  half the amplitude each (fractal noise), normalized back to the same range.
  The pattern evolves at ``w = time * frequency`` (``frequency`` in Hz when
  ``time`` is in seconds: about one complete change per ``1 / frequency``
  seconds), and its amplitude fades as ``exp(-decay * time)``.
* ``curl_noise(x, y, z, octaves=1, seed=0, time=0.0, frequency=1.0,
  decay=0.0)``: the curl (in space) of three independent noise potentials, a
  swirling vector field with no sources or sinks, so it stirs without
  bunching things up. Returns ``(cx, cy, cz)``. ``time``, ``frequency`` and
  ``decay`` mean what they mean for ``noise``.

The solver's bytecode carries only ``w``: the compiler turns ``time *
frequency`` into it and applies the decay with ordinary arithmetic, so the
kernel's noise has one input more than space and nothing else.

Inputs may be Python floats or numpy arrays of one shape.
"""

import numpy as np

MASK = 0xFFFFFFFF
MAX_OCTAVES = 8

# The thirty-two edge midpoints of a 4D hypercube, the gradient set a grid
# point draws from: index ``8 * a + s`` has a zero on axis ``a`` and, on the
# other three axes in order, -1 where bit 0, 1, 2 of ``s`` is set, else +1.
def _gradients_4d():
    out = []
    for zero in range(4):
        for signs in range(8):
            g, bit = [], 0
            for axis in range(4):
                if axis == zero:
                    g.append(0.0)
                else:
                    g.append(-1.0 if (signs >> bit) & 1 else 1.0)
                    bit += 1
            out.append(g)
    return np.array(out, dtype=np.float64)


GRADIENTS = _gradients_4d()

# Per-octave and per-potential seed strides, and the offsets that decorrelate
# the three curl potentials.
OCTAVE_SEED_STRIDE = 1013
POTENTIAL_SEED_STRIDE = 7919
POTENTIAL_OFFSETS = ((0.0, 0.0, 0.0), (31.416, 47.853, 12.679), (71.231, 5.117, 93.402))


def _hash(ix, iy, iz, iw, seed):
    """A 32-bit integer hash of a grid point and a seed (uint64 arrays)."""
    h = (seed * 0x9E3779B9) & MASK
    h ^= (ix * 0x85EBCA6B) & MASK
    h ^= (iy * 0xC2B2AE35) & MASK
    h ^= (iz * 0x27D4EB2F) & MASK
    h ^= (iw * 0x165667B1) & MASK
    h ^= h >> 15
    h = (h * 0x2C1B3C6D) & MASK
    h ^= h >> 12
    h = (h * 0x297A2D39) & MASK
    h ^= h >> 15
    return h


def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _dfade(t):
    return 30.0 * t * t * (t - 1.0) * (t - 1.0)


def _gradient_noise(x, y, z, w, seed):
    """One octave: the value and its gradient in SPACE, float64 arrays."""
    p = (x, y, z, w)
    cell = [np.floor(c) for c in p]
    f = [c - k for c, k in zip(p, cell)]
    i = [k.astype(np.int64).astype(np.uint64) & MASK for k in cell]
    u = [_fade(t) for t in f]
    du = [_dfade(t) for t in f]
    seed_u = np.uint64(int(seed) & MASK)
    value = np.zeros_like(x)
    grad = [np.zeros_like(x) for _ in range(3)]
    for corner in range(16):
        o = [(corner >> k) & 1 for k in range(4)]
        h = _hash(*[(i[k] + np.uint64(o[k])) & MASK for k in range(4)], seed_u)
        g = GRADIENTS[(h % np.uint64(32)).astype(np.int64)]
        d = [f[k] - o[k] for k in range(4)]
        dot = g[..., 0] * d[0] + g[..., 1] * d[1] + g[..., 2] * d[2] + g[..., 3] * d[3]
        wt = [u[k] if o[k] else 1.0 - u[k] for k in range(4)]
        dw = [du[k] if o[k] else -du[k] for k in range(3)]
        weight = wt[0] * wt[1] * wt[2] * wt[3]
        value = value + weight * dot
        grad[0] = grad[0] + dw[0] * wt[1] * wt[2] * wt[3] * dot + weight * g[..., 0]
        grad[1] = grad[1] + wt[0] * dw[1] * wt[2] * wt[3] * dot + weight * g[..., 1]
        grad[2] = grad[2] + wt[0] * wt[1] * dw[2] * wt[3] * dot + weight * g[..., 2]
    return value, grad


def _fractal(x, y, z, w, octaves, seed):
    """Octaves summed at halving amplitude, normalized: value and gradient."""
    octaves = int(octaves)
    if not 1 <= octaves <= MAX_OCTAVES:
        raise ValueError(f"octaves must be 1 to {MAX_OCTAVES}, got {octaves}")
    value = np.zeros_like(x)
    grad = [np.zeros_like(x) for _ in range(3)]
    amplitude, frequency, total = 1.0, 1.0, 0.0
    for k in range(octaves):
        v, g = _gradient_noise(x * frequency, y * frequency, z * frequency,
                               w * frequency, int(seed) + k * OCTAVE_SEED_STRIDE)
        value = value + amplitude * v
        for a in range(3):
            grad[a] = grad[a] + amplitude * frequency * g[a]
        total += amplitude
        amplitude *= 0.5
        frequency *= 2.0
    return value / total, [g / total for g in grad]


def _arrays(x, y, z, time):
    arrays = np.broadcast_arrays(np.asarray(x, dtype=np.float64),
                                 np.asarray(y, dtype=np.float64),
                                 np.asarray(z, dtype=np.float64),
                                 np.asarray(time, dtype=np.float64))
    scalar = arrays[0].ndim == 0
    return [np.atleast_1d(a).astype(np.float64) for a in arrays], scalar


def noise(x, y, z, octaves=1, seed=0, time=0.0, frequency=1.0, decay=0.0):
    """Smooth scalar noise in about [-1, 1] at ``(x, y, z)``, evolved to
    ``time * frequency`` and faded by ``exp(-decay * time)``."""
    (x, y, z, time), scalar = _arrays(x, y, z, time)
    value, _ = _fractal(x, y, z, time * frequency, octaves, seed)
    value = value * np.exp(-decay * time)
    return float(value[0]) if scalar else value


def curl_noise(x, y, z, octaves=1, seed=0, time=0.0, frequency=1.0, decay=0.0):
    """The curl of three noise potentials at ``(x, y, z)``: ``(cx, cy, cz)``."""
    (x, y, z, time), scalar = _arrays(x, y, z, time)
    w = time * frequency
    grads = []
    for a, off in enumerate(POTENTIAL_OFFSETS):
        _, g = _fractal(x + off[0], y + off[1], z + off[2], w, octaves,
                        int(seed) + a * POTENTIAL_SEED_STRIDE)
        grads.append(g)
    fade = np.exp(-decay * time)
    cx = (grads[2][1] - grads[1][2]) * fade
    cy = (grads[0][2] - grads[2][0]) * fade
    cz = (grads[1][0] - grads[0][1]) * fade
    if scalar:
        return float(cx[0]), float(cy[0]), float(cz[0])
    return cx, cy, cz


def noise_vector(x, y, z, octaves=1, seed=0, time=0.0, frequency=1.0, decay=0.0):
    """Three independent noise channels at ``(x, y, z)``: ``(nx, ny, nz)``."""
    out = []
    for a, off in enumerate(POTENTIAL_OFFSETS):
        out.append(noise(np.asarray(x) + off[0], np.asarray(y) + off[1],
                         np.asarray(z) + off[2], octaves,
                         int(seed) + a * POTENTIAL_SEED_STRIDE,
                         time, frequency, decay))
    return tuple(out)
