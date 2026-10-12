# File: _field_eval_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""The reference evaluator of a field object, in double precision.

The solver evaluates a field object exactly at each vertex on the device
(``kernels/energy/external_field.kernel.cpp``); this module computes the same
laws with numpy, so a test can hold the kernel to them and a viewer can draw
what the solver will apply. Each law is Blender's, measured on Blender as a
black box.

A field is the record :meth:`frontend.ForceField.add` builds
(``frontend._force_field_.make_field``): its discrete settings, and its
continuous ones as per-sample arrays (``position`` T x 3, ``axes`` T x 3 x 3
with the local axes as columns, ``noise_scale`` T x 3, the field object's
scale along those axes, the rest T long), spaced ``step`` seconds from
``start``. Every length is in scene units and solver world axes.

THIS MODULE IMPORTS NOTHING FROM THE SOLVER, and nothing from the rest of the
frontend but the noise, so the add-on can carry a byte-identical copy
(``blender_addon/core/field_eval.py``), as it does the noise.
"""

from __future__ import annotations

import math

import numpy as np

try:  # the frontend names its noise `_noise_`, the add-on's copy `noise`
    from . import _noise_ as _noise
except ImportError:  # pragma: no cover
    from . import noise as _noise  # type: ignore

DEGREES_PER_RADIAN = 57.2957795

# A fluid volume (a Fluid Flow field's) is cut into bricks of FLUID_BRICK^3
# cells, x fastest, as the solver holds it.
FLUID_BRICK = 8
FLUID_EMPTY = 0xFFFFFFFF


def _real_array(value, name: str):
    """``value`` as an array of real numbers in its own dtype, or refused."""
    arr = np.asarray(value)
    if arr.dtype.kind not in "biuf":
        raise TypeError(f"a fluid {name} frame holds {arr.dtype} values; give real numbers")
    return arr


def fluid_bricks(velocity, density=None):
    """Cut one frame into the solver's bricks.

    ``velocity`` is ``(nx, ny, nz, 3)`` in the domain's local axes and
    ``density`` ``(nx, ny, nz)``, or None for a zero density channel; either
    may be any real dtype. The frame is read in its own dtype, ``FLUID_BRICK``
    z-planes at a time, and stored as the float32 the solver reads, so the cut
    holds the input, one slab and the kept bricks (twice those while the last
    are joined), however little of the box is occupied. A frame with a zero
    extent is EMPTY: no table entries and no bricks.

    Returns ``(table, bricks)``: a uint32 table over the bricks, x fastest,
    naming each kept brick's slot or ``FLUID_EMPTY``, and the kept bricks as
    float32 ``(count, 512, 4)``, cells x fastest inside a brick, channels
    velocity then density. A brick is kept when any of its float32 values is
    not zero, so an empty brick and a zero one read alike.

    Raises:
        TypeError: A frame holds values that are not real numbers.
        ValueError: A shape is wrong, or a value is not finite as the float32
            the solver reads (a NaN, an infinity, or a magnitude past
            float32's 3.4e38), naming the cell ``(i, j, k)``.
    """
    u = _real_array(velocity, "velocity")
    if u.ndim != 4 or u.shape[3] != 3:
        raise ValueError(f"a fluid velocity frame has shape {u.shape}; give (nx, ny, nz, 3)")
    n = tuple(int(v) for v in u.shape[:3])
    rho = None
    if density is not None:
        rho = _real_array(density, "density")
        if rho.shape != n:
            raise ValueError(f"a fluid density frame has shape {rho.shape}; give {n}")
    b = FLUID_BRICK
    if min(n) == 0:
        return np.zeros(0, dtype=np.uint32), np.zeros((0, b ** 3, 4), dtype=np.float32)
    nb = [(k + b - 1) // b for k in n]
    table = np.full(nb[0] * nb[1] * nb[2], FLUID_EMPTY, dtype=np.uint32)
    kept = []
    count = 0
    # One slab of bricks, FLUID_BRICK planes thick, padded to whole bricks.
    slab = np.empty((nb[0] * b, nb[1] * b, b, 4), dtype=np.float32)
    for bz in range(nb[2]):
        z0 = bz * b
        depth = min(b, n[2] - z0)
        slab.fill(0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            slab[:n[0], :n[1], :depth, :3] = u[:, :, z0:z0 + depth]
            if rho is not None:
                slab[:n[0], :n[1], :depth, 3] = rho[:, :, z0:z0 + depth]
        finite = np.isfinite(slab)
        if not finite.all():
            i, j, k, c = (int(v) for v in np.argwhere(~finite)[0])
            name = "velocity" if c < 3 else "density"
            value = float(u[i, j, z0 + k, c] if c < 3 else rho[i, j, z0 + k])
            beyond = "" if not math.isfinite(value) else (
                ", past the float32 range the solver reads")
            raise ValueError(f"the fluid {name} is not finite at cell ({i}, {j}, {z0 + k}): "
                             f"{value!r}{beyond}")
        del finite
        # (bx, 8, by, 8, 8z * 4) -> occupied (bx, by); taken by-major, bx
        # fastest, which is the table's order within the slab.
        occupied = (slab != 0.0).reshape(nb[0], b, nb[1], b, b * 4).any(axis=(1, 3, 4))
        by, bx = np.nonzero(occupied.T)
        if len(bx):
            table[bx + nb[0] * (by + nb[1] * bz)] = np.arange(
                count, count + len(bx), dtype=np.uint32)
            count += len(bx)
            # (bx, 8x, by, 8y, 8z, 4) -> (by, bx, 8z, 8y, 8x, 4): the cells of
            # a brick x fastest; only the kept bricks are copied.
            cells = slab.reshape(nb[0], b, nb[1], b, b, 4).transpose(2, 0, 4, 3, 1, 5)
            kept.append(cells[by, bx].reshape(len(bx), b ** 3, 4))
    if not kept:
        return table, np.zeros((0, b ** 3, 4), dtype=np.float32)
    return table, (kept[0] if len(kept) == 1 else np.concatenate(kept))


def fluid_bracket(volume: dict, time: float):
    """The two frames bracketing ``time`` and the blend toward the second;
    the end frame holds outside the sequence."""
    n = len(volume["tables"])
    if n <= 1:
        return 0, 0, 0.0
    u = (time - volume["start"]) / volume["step"]
    if u >= n - 1:
        return n - 1, n - 1, 0.0
    if u > 0.0:
        k = int(u)
        return k, k + 1, min(max(u - k, 0.0), 1.0)
    return 0, 0, 0.0


def fluid_frame(volume: dict, k: int, points):
    """Frame ``k`` at world ``points`` (N x 3): the velocity in world axes,
    the density, trilinear between cell centers and clamped at the outer
    cells, and the presence, 1 inside the domain; outside, all three are zero
    (measured on Blender's Fluid Flow). The frame's ``cell`` is its size
    along each of the domain's axes (one number for all three also reads).
    An EMPTY frame, a zero ``resolution``, holds no domain: it is outside
    everywhere."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = np.asarray(volume["resolution"][k], dtype=np.int64)
    if np.any(n == 0):
        return np.zeros((len(p), 3)), np.zeros(len(p)), np.zeros(len(p))
    axes = np.asarray(volume["axes"][k], dtype=np.float64)
    corner = np.asarray(volume["corner"][k], dtype=np.float64)
    cell = np.broadcast_to(np.asarray(volume["cell"][k], dtype=np.float64), (3,))
    table = np.asarray(volume["tables"][k])
    bricks = np.asarray(volume["bricks"][k], dtype=np.float64).reshape(-1, FLUID_BRICK ** 3, 4)
    g = ((p - corner) @ axes) / cell
    inside = np.all((g >= 0.0) & (g <= n), axis=1)
    q = g - 0.5
    i0 = np.floor(q).astype(np.int64)
    t = q - i0
    b = FLUID_BRICK
    nb = (n + b - 1) // b
    out = np.zeros((len(p), 4))
    for corner_bits in range(8):
        o = np.array([(corner_bits >> c) & 1 for c in range(3)])
        idx = np.clip(i0 + o, 0, n - 1)
        w = np.prod(np.where(o == 1, t, 1.0 - t), axis=1)
        brick = idx // b
        slot = table[brick[:, 0] + nb[0] * (brick[:, 1] + nb[1] * brick[:, 2])]
        full = slot != FLUID_EMPTY
        local = idx % b
        within = local[:, 0] + b * (local[:, 1] + b * local[:, 2])
        value = np.zeros((len(p), 4))
        if np.any(full):
            value[full] = bricks[slot[full].astype(np.int64), within[full]]
        out += w[:, None] * value
    out[~inside] = 0.0
    return out[:, :3] @ axes.T, out[:, 3], inside.astype(np.float64)


def fluid_at(volume: dict, points, time: float):
    """The volume at ``time``, its two bracketing frames blended: velocity,
    density and presence."""
    k0, k1, b = fluid_bracket(volume, time)
    u0, r0, p0 = fluid_frame(volume, k0, points)
    if b == 0.0:
        return u0, r0, p0
    u1, r1, p1 = fluid_frame(volume, k1, points)
    return (1.0 - b) * u0 + b * u1, (1.0 - b) * r0 + b * r1, (1.0 - b) * p0 + b * p1


def _falloff(d, use_min, dmin, use_max, dmax, power):
    """Blender's falloff: 1 inside the minimum, 0 past the maximum, else
    ``(1 + max(d - m, 0))^-power``."""
    d = np.asarray(d, dtype=np.float64)
    m = dmin if use_min else 0.0
    if power == 0.0:
        f = np.ones_like(d)
    else:
        f = np.power(1.0 + np.maximum(d - m, 0.0), -power)
    if use_min:
        f = np.where(d < dmin, 1.0, f)
    if use_max:
        f = np.where(d > dmax, 0.0, f)
    return f


def _texel(image, extension, ix, iy):
    """Pixels of an (H, W, 3) image at integer indices, brought inside by
    the extension; CLIP reads color 0 outside."""
    h, w = image.shape[:2]
    ix = np.asarray(ix)
    iy = np.asarray(iy)
    if extension == "repeat":
        ix, iy = np.mod(ix, w), np.mod(iy, h)
    elif extension == "extend":
        ix, iy = np.clip(ix, 0, w - 1), np.clip(iy, 0, h - 1)
    outside = (ix < 0) | (iy < 0) | (ix >= w) | (iy >= h)
    out = image[np.clip(iy, 0, h - 1), np.clip(ix, 0, w - 1)]
    return np.where(outside[:, None], 0.0, out)


def texture_color(field: dict, q):
    """The texture's color at texture coordinates q (N x 3), the image
    spanning x and y in [-1, 1]: the nearest pixel, or bilinear between pixel
    centers with interpolation (measured on Blender's Texture field)."""
    image = np.asarray(field["image"], dtype=np.float64)
    h, w = image.shape[:2]
    ext = field["extension"]
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    u = 0.5 * (x + 1.0) * w
    v = 0.5 * (y + 1.0) * h
    if not field["interpolation"]:
        color = _texel(image, ext, np.floor(u).astype(np.int64), np.floor(v).astype(np.int64))
    else:
        fu, fv = u - 0.5, v - 0.5
        bu, bv = np.floor(fu), np.floor(fv)
        tu, tv = (fu - bu)[:, None], (fv - bv)[:, None]
        i0, j0 = bu.astype(np.int64), bv.astype(np.int64)
        color = ((1 - tu) * (1 - tv) * _texel(image, ext, i0, j0)
                 + tu * (1 - tv) * _texel(image, ext, i0 + 1, j0)
                 + (1 - tu) * tv * _texel(image, ext, i0, j0 + 1)
                 + tu * tv * _texel(image, ext, i0 + 1, j0 + 1))
    clip = np.zeros(len(q), dtype=bool)
    if ext in ("clip", "clip_cube"):
        clip |= (x < -1.0) | (x > 1.0) | (y < -1.0) | (y > 1.0)
    if ext == "clip_cube":
        clip |= (z < -1.0) | (z > 1.0)
    return np.where(clip[:, None], 0.0, color)


def _sample(field: dict, time: float):
    """The track sample index, the blend toward the next, and the blended
    values at ``time``; the end sample holds outside the track."""
    n = len(field["position"])
    k, b = 0, 0.0
    if n > 1:
        u = (time - field["start"]) / field["step"]
        if u >= n - 1:
            k = n - 1
        elif u > 0.0:
            k = int(u)
            b = min(max(u - k, 0.0), 1.0)
    k1 = k + 1 if k + 1 < n else k

    def lerp(key):
        a = np.asarray(field[key][k], dtype=np.float64)
        c = np.asarray(field[key][k1], dtype=np.float64)
        return (1.0 - b) * a + b * c

    return k, b, lerp


def evaluate(field: dict, points, velocities=None, time: float = 0.0, normals=None,
             part: str = "all", fluid_time=None, playback: float = 1.0):
    """One field at ``points`` (N x 3), with the vertices' ``velocities``
    (N x 3, needed by Magnetic, Drag, Harmonic and Vortex) and surface
    ``normals`` (N x 3 unit vectors, or zero where a vertex presents no
    surface; needed by a Wind Factor) at ``time``.

    ``part`` picks what the solver evaluates where: ``"explicit"`` is what it
    evaluates once at the step's start, ``"implicit"`` what it evaluates at
    the Newton iterate (Harmonic's spring, and the damping of Drag, Harmonic
    and Vortex), and ``"all"`` their sum. Wind Factor weighs the explicit
    part only, as the solver does. A Fluid field reads its volume at
    ``fluid_time`` (default ``time``): the solver reads it at the step's END,
    blending the two frames that bracket it. Blender instead holds frame
    F + 1's grid for every substep from frame F to F + 1; the two agree at
    every step that ends on a frame.

    ``playback`` is the step's Playback Speed. A step advances the vertices
    by ``dt * playback`` seconds of motion and the scene clock by ``dt``,
    and a Fluid's velocity ``U`` is per second of the scene clock its frames
    are placed on, so the solver divides ``U`` by ``playback`` to compare it
    with the vertex's velocity; the Flow damping is not divided. Nothing
    else depends on it.

    A Texture record's ``z_up`` says its image lies in Blender's axes (Z up):
    in world coordinates the image is read at the point's Blender coordinates,
    ``(x, -z, y)`` of the solver point, and in either coordinate mode the
    texture's vector, a Blender-world vector, is turned into solver axes.
    With ``object_coordinates`` the image is read in the field object's own
    frame, scale included: at the local point divided by ``noise_scale``
    along each axis, as Blender reads it under the object's whole inverse
    world matrix (measured), and the differences ``nabla`` apart are taken
    there.

    Returns ``(acceleration, air_velocity)``, each N x 3 in solver axes.
    A Fluid field's Flow is its record's ``inflow`` (Blender stores the two
    as one value).
    """
    if part not in ("all", "explicit", "implicit"):
        raise ValueError(f"part {part!r} is not all, explicit or implicit")
    playback = float(playback)
    if not (math.isfinite(playback) and playback > 0.0):
        raise ValueError(f"playback {playback!r} is not a positive speed")
    p_world = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(p_world)
    k, b, lerp = _sample(field, time)

    axes = lerp("axes")
    az = axes[:, 2] / np.linalg.norm(axes[:, 2])
    ax = axes[:, 0] - np.dot(axes[:, 0], az) * az
    ax = ax / np.linalg.norm(ax)
    ay = np.cross(az, ax)
    frame = np.stack([ax, ay, az], axis=1)  # columns: local X, Y, Z

    origin = lerp("position")
    d = p_world - origin
    local_p = d @ frame
    px, py, pz = local_p[:, 0], local_p[:, 1], local_p[:, 2]
    r = np.hypot(px, py)
    distance = np.linalg.norm(local_p, axis=1)

    kind = field["type"]
    shape = field["shape"]
    falloff = field["falloff"]
    strength = float(lerp("strength"))

    plane_vortex = kind == "vortex" and shape == "plane"
    if plane_vortex:
        axial = r
    elif falloff == "sphere" and shape == "point":
        axial = distance
    else:
        axial = np.abs(pz)
    W = _falloff(axial, field["use_min"], float(lerp("min_distance")),
                 field["use_max"], float(lerp("max_distance")), float(lerp("power")))
    if falloff in ("tube", "cone"):
        radial = r if falloff == "tube" else np.arctan2(r, pz) * DEGREES_PER_RADIAN
        W = W * _falloff(radial, field["use_radial_min"], float(lerp("radial_min")),
                         field["use_radial_max"], float(lerp("radial_max")),
                         float(lerp("radial_power")))
    if field["z_direction"] == "positive":
        W = np.where(pz < 0.0, 0.0, W)
    elif field["z_direction"] == "negative":
        W = np.where(pz > 0.0, 0.0, W)
    if field["gravitation"]:
        gd = axial if shape == "plane" else distance
        if np.any((gd == 0.0) & (W != 0.0)):
            # The solver fails the run here too: the weight is infinite.
            raise ValueError("a gravitation field meets a point exactly at its origin")
        with np.errstate(divide="ignore", invalid="ignore"):
            W = np.where(W != 0.0, W / (gd * gd), 0.0)

    sw = strength * W
    local = np.zeros((n, 3))
    local_air = np.zeros((n, 3))
    safe_r = np.where(r > 0.0, r, 1.0)
    tangent = np.stack([-py / safe_r, px / safe_r, np.zeros(n)], axis=1)
    tangent[r <= 0.0] = 0.0

    if kind == "force":
        if shape == "plane":
            local[:, 2] = np.where(pz >= 0.0, sw, -sw)
        else:
            safe_d = np.where(distance > 0.0, distance, 1.0)
            local = (sw / safe_d * (distance > 0.0))[:, None] * local_p
    elif kind == "wind":
        local_air[:, 2] = sw
    elif kind == "vortex":
        if plane_vortex:
            speed = sw * r
            local = speed[:, None] * tangent
            inward = np.where(r > 0.0, speed * speed / safe_r, 0.0)
            local[:, 0] -= inward * px / safe_r
            local[:, 1] -= inward * py / safe_r
        else:
            local = (sw * distance)[:, None] * tangent
    elif kind == "turbulence":
        size = float(lerp("size"))
        if field["global_coordinates"]:
            q = p_world / size
            nv = np.stack(_noise.noise_vector(q[:, 0], q[:, 1], q[:, 2], 1, field["seed"]),
                          axis=1)
            local = sw[:, None] * (nv @ frame)
        else:
            q = local_p / size
            local = sw[:, None] * np.stack(
                _noise.noise_vector(q[:, 0], q[:, 1], q[:, 2], 1, field["seed"]), axis=1)
    elif kind == "magnetic":
        if velocities is None:
            raise ValueError("a magnetic field needs the vertices' velocities")
        v = np.asarray(velocities, dtype=np.float64).reshape(-1, 3) @ frame
        axis = np.array([0.0, 0.0, 1.0]) if shape == "plane" else tangent
        local = sw[:, None] * np.cross(v, axis)
    elif kind in ("harmonic", "drag", "fluid"):
        pass  # wholly implicit, below
    elif kind == "texture":
        z_up = bool(field["z_up"])
        if field["object_coordinates"]:
            # Under the field object's whole inverse world matrix, so its
            # scale stretches the image as Blender's does.
            q = local_p / lerp("noise_scale")
        elif z_up:
            q = np.stack([p_world[:, 0], -p_world[:, 2], p_world[:, 1]], axis=1)
        else:
            q = p_world
        c0 = texture_color(field, q)
        mode = field["texture_mode"]
        if mode == "rgb":
            tex = sw[:, None] * (0.5 - c0)
        else:
            nabla = float(lerp("nabla"))
            d = np.stack([(texture_color(field, q + nabla * e) - c0) / nabla
                          for e in np.eye(3)], axis=1)  # d[:, k, ch]
            if mode == "gradient":
                tex = -sw[:, None] * d.mean(axis=2)
            else:
                tex = sw[:, None] * np.stack([d[:, 1, 2] - d[:, 2, 1],
                                               d[:, 2, 0] - d[:, 0, 2],
                                               d[:, 0, 1] - d[:, 1, 0]], axis=1)
        if z_up:
            # Blender's world axes to the solver's.
            tex = np.stack([tex[:, 0], tex[:, 2], -tex[:, 1]], axis=1)
        # A world-axes vector (measured), turned into the local frame for the
        # common rotation back.
        local = tex @ frame
    else:
        raise ValueError(f"field type {kind!r} has no reference law here")

    # THE IMPLICIT TERMS, which the solver evaluates at the
    # Newton iterate: Harmonic's spring, and the damping of Drag, Harmonic and
    # Vortex.
    implicit = np.zeros((n, 3))
    if kind == "harmonic":
        if shape == "plane":
            implicit[:, 2] = -sw * pz
        else:
            rest = float(lerp("rest_length"))
            safe_d = np.where(distance > 0.0, distance, 1.0)
            implicit = (-sw * (distance - rest) / safe_d * (distance > 0.0))[:, None] * local_p
    damping = np.zeros(n)
    if kind == "fluid":
        U, rho, presence = fluid_at(field["fluid"], p_world,
                                    time if fluid_time is None else fluid_time)
        if not field["use_density"]:
            # Outside the domain nothing acts, the damping included (measured).
            rho = presence
        swr = strength * W * rho
        implicit = (swr[:, None] * (U / playback)) @ frame
    if kind in ("drag", "harmonic", "vortex", "fluid"):
        if velocities is None:
            raise ValueError(f"a {kind} field needs the vertices' velocities")
        v_world = np.asarray(velocities, dtype=np.float64).reshape(-1, 3)
        if kind == "drag":
            speed = np.linalg.norm(v_world, axis=1)
            damping = W * (float(lerp("linear_drag")) * speed + float(lerp("quadratic_drag")))
        elif kind == "fluid":
            damping = swr * float(lerp("inflow"))
        elif kind == "harmonic":
            damping = np.where(W > 0.0, 2.0 * float(lerp("harmonic_damping"))
                               * np.sqrt(abs(strength)), 0.0)
        elif shape == "plane":
            damping = np.where(W > 0.0, 1.0 + float(lerp("inflow")) * W, 0.0)
        else:
            damping = float(lerp("inflow")) * W
        implicit = implicit - damping[:, None] * (v_world @ frame)

    noise_amount = float(lerp("noise"))
    if noise_amount != 0.0:
        if kind == "turbulence":
            q = local_p / float(lerp("size"))
        else:
            scale = lerp("noise_scale")
            carried = strength * time if kind == "wind" else 0.0
            q = np.stack([px / scale[0], py / scale[1], (pz - carried) / scale[2]], axis=1)
        curl = np.stack(_noise.curl_noise(q[:, 0], q[:, 1], q[:, 2], 1, field["seed"]),
                        axis=1)
        gust = (noise_amount * abs(strength) * W)[:, None] * curl
        if kind == "wind":
            local_air = local_air + gust
        else:
            local = local + gust

    accel = local @ frame.T
    wind_factor = float(lerp("wind_factor"))
    if wind_factor != 0.0 and kind != "wind":
        if normals is None:
            raise ValueError("a field with a wind factor needs the vertices' normals")
        nrm = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        along = np.sum(accel * nrm, axis=1)
        accel = (1.0 - wind_factor) * accel + wind_factor * along[:, None] * nrm
        # A vertex with no surface keeps its push whole.
        bare = ~np.any(nrm != 0.0, axis=1)
        accel[bare] = (local @ frame.T)[bare]
    if part == "explicit":
        return accel, local_air @ frame.T
    if part == "implicit":
        return implicit @ frame.T, np.zeros_like(accel)
    return accel + implicit @ frame.T, local_air @ frame.T
