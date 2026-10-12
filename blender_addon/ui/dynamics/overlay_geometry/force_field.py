# File: force_field.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The force field's Visualize overlay: arrows at a sparse PREVIEW grid around
# the simulated objects, for the field at the current frame.
#
# DRAWN FROM THE SOLVER'S OWN LAWS (`core.force_field.evaluate`, through
# `core/field_eval.py`, the frontend's reference evaluator byte for byte), in
# the record Transfer sends, so the arrows are what the solver computes at
# those points, not a second reading of the settings. The preview resolution
# is for drawing only: the solver evaluates a field at every vertex exactly.
#
# Arrow length is the local magnitude over the largest one drawn, times one
# preview cell, so the field's SHAPE reads at any strength; the panel does not
# claim a scale for them.
#
# The drawn box covers every simulated object at the DRAWN frame, grown by a
# tenth of its largest extent (`core.force_field.objects_box`).
#
# A FIELD WHOSE SOURCE CANNOT BE READ is left out of the drawing and named in
# the panel (`preview_errors`), and the other fields and the script still
# draw (`core.force_field.evaluate`'s ``errors``). The core keeps each field's
# record under its `field_key`, so a zoom, which changes only the arrow
# thickness, does not read a fluid or an image again; the script's arrows are
# kept here the same way.

import hashlib
import math

import numpy as np

import gpu  # pyright: ignore
from gpu_extras.batch import batch_for_shader  # pyright: ignore
from mathutils import Vector  # pyright: ignore

from .primitives import _line_to_tris, _orthonormal_basis

# A preview never draws more arrows than this, whatever the resolution says.
_MAX_ARROWS = 48 * 48 * 48


# Why each field left out of the last drawing was left out, each message
# naming its field.
_preview_errors: list = []
# The script's arrows under the key they were evaluated for: (key, vectors).
_script_memo: list = [None, None]


def preview_errors() -> list:
    """The message of each field the last drawing left out because its
    source could not be read, each naming the field; empty while Visualize
    is off."""
    return list(_preview_errors)


def force_field_key(scene, state):
    """What the drawn arrows depend on, so the overlay rebuilds exactly when
    one of them changes: the settings, every field object's field
    (`core.force_field.field_key`: its transform, settings, texture and
    fluid source) and whether it is refused, the simulated objects, the
    script text, and the frame."""
    from ....core import force_field as ff
    from .. import force_field_status as status

    if not state.force_field_visualize:
        return None
    parts = [
        tuple(state.force_field_preview_resolution),
        scene.frame_current,
    ]
    for obj in ff.field_objects(scene, state):
        # The refusal decides whether a field is drawn at all, and can turn
        # on the scene (a Fluid Flow's baked range against the solve's).
        parts.append((ff.field_key(obj), status.static_refusal(obj)))
    for obj in ff.dynamic_objects(scene):
        parts.append((obj.name, tuple(tuple(r) for r in obj.matrix_world)))
    text = state.force_field_script
    parts.append(hash(text.as_string()) if text is not None else 0)
    return tuple(parts)


def _drawn_time(scene, state) -> float:
    """Seconds into the simulation at the drawn frame, the solver's clock."""
    from ....core.encoder import frame_to_time, resolve_solver_fps, resolve_start_frame

    return frame_to_time(scene.frame_current, resolve_solver_fps(state),
                         resolve_start_frame(state))


def _script_vectors(state, points, t, digest):
    """The exact script at ``points`` at time ``t``, or ``None`` without one
    or on error. ``digest`` names the points, for the memo."""
    from ....core import force_field as ff

    text = state.force_field_script
    if text is None:
        return None
    source = text.as_string()
    key = (source, digest, t)
    if _script_memo[0] == key:
        return _script_memo[1]
    try:
        fn = ff.local_script(source)
    except Exception as e:  # the script's own top level failed
        print(f"[ppf] force field script cannot be drawn: {e}")
        return None
    out = np.zeros_like(points)
    for i, p in enumerate(points):
        try:
            v = fn(float(p[0]), float(p[1]), float(p[2]), t)
            out[i] = (float(v[0]), float(v[1]), float(v[2]))
        except Exception:
            # A point the formula has no value at (a division by zero, a
            # domain error) is left undrawn; the solver would stop the run
            # there, which Compile and Check and the run itself report.
            out[i] = np.nan
    _script_memo[:] = [key, out]
    return out


def preview_fields(scene, state):
    """``(points, cell, [(vectors, color), ...])`` the overlay draws, or
    ``None`` when Visualize is off or there is no box to draw in.

    Separate from the batching so a scenario can check WHAT is drawn without
    a GPU context.
    """
    from ....core import force_field as ff
    from .. import force_field_status as status

    if not state.force_field_visualize:
        _preview_errors.clear()
        return None
    objs = [o for o in ff.field_objects(scene, state) if status.static_refusal(o) is None]
    pushed = ff.dynamic_objects(scene)
    if not pushed:
        _preview_errors.clear()
        return None
    # One box over every simulated object, which holds each field's own box.
    lo, hi = ff.objects_box(pushed)
    res = [max(2, int(v)) for v in state.force_field_preview_resolution]
    while res[0] * res[1] * res[2] > _MAX_ARROWS:
        res = [max(2, v // 2) for v in res]
    pts = ff.grid_points(lo, hi, res).reshape(-1, 3)
    cell = float(np.min((hi - lo) / (np.array(res) - 1)))
    fields = []
    time = _drawn_time(scene, state)
    errors: list = []
    if objs:
        acc, air = ff.evaluate(objs, pts, time, errors=errors)
        if len(errors) < len(objs):
            fields.append((acc, ff.COLOR_ACCELERATION))
            fields.append((air, ff.COLOR_AIR))
    _preview_errors[:] = errors
    digest = hashlib.sha1(np.ascontiguousarray(pts, dtype=np.float64).tobytes()).hexdigest()
    script = _script_vectors(state, pts, time, digest)
    if script is not None:
        fields.append((script, ff.COLOR_SCRIPT))
    return pts, cell, fields


def _field_arrow(start, end, thickness):
    """A thin arrow whose head is sized by ITS OWN length: a quarter of it,
    and no wider than a third of that, so a weak spot draws a small arrow
    rather than a full-size head on a stub."""
    diff = end - start
    length = diff.length
    if length < 1e-9:
        return []
    direction = diff / length
    head = min(0.25 * length, 12.0 * thickness)
    radius = max(head / 3.0, thickness)
    base = end - direction * head
    tris = _line_to_tris(start, base, thickness)
    u, v = _orthonormal_basis(direction)
    ring = [base + (u * math.cos(a) + v * math.sin(a)) * radius
            for a in (2.0 * math.pi * i / 6 for i in range(6))]
    for i in range(6):
        tris.extend((end, ring[i], ring[(i + 1) % 6]))
    return tris


def build_force_field_batches(scene, state, view_distance):
    """``[(batch, color), ...]`` for the Visualize overlay, or ``[]``."""
    preview = preview_fields(scene, state)
    if preview is None:
        return []
    pts, cell, fields = preview
    # Thin enough that neighboring arrows never merge: a hundredth of a cell,
    # but not below what stays visible at this zoom.
    thickness = max(cell * 0.01, view_distance * 0.0006)
    shader = gpu.shader.from_builtin("UNIFORM_COLOR")
    batches = []
    for vec, color in fields:
        mag = np.linalg.norm(vec, axis=1)
        finite = np.isfinite(mag)
        if not finite.any():
            continue
        top = float(mag[finite].max())
        if top <= 1e-12:
            continue
        tris = []
        for p, v, m in zip(pts[finite], vec[finite], mag[finite]):
            if m <= top * 1e-3:
                continue
            start = Vector(p)
            end = start + Vector(v) * (0.9 * cell / top)
            tris.extend(_field_arrow(start, end, thickness))
        if tris:
            batches.append((batch_for_shader(shader, "TRIS", {"pos": tris}), color))
    return batches
