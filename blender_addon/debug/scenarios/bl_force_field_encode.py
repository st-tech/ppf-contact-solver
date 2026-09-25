# File: scenarios/bl_force_field_encode.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The add-on's force-field ENCODE (issue #114): Blender field objects sampled
# into the W x H x D x T grids the solver reads, and the exact script's text.
# A grid covers the objects its fields push, grown by Padding, with points at
# most Spacing apart; nothing else says where or how finely.
#
# What makes the encode worth a scenario is the AXIS BOOKKEEPING, which fails
# silently: the solver's grid walks its own axes (x, z, -y of Blender's), so the
# sampled array has to be transposed and one axis reversed on the way out. A
# mistake there ships a plausible field rotated a quarter turn. Subtest D
# therefore re-derives samples from the decoded payload index by index and
# compares them against the evaluator at the matching Blender point.
#
# Subtests:
#   A. unsupported_type_refused: a Magnet field fails the encode, naming it.
#   B. wind_needs_air_density: a Wind field in a scene with Air Density 0 fails
#      the encode, naming Air Density.
#   C. payload_shape: Turbulence and Wind give one acceleration and one
#      air-velocity grid of shape [T, D, H, W, 3] in solver order, over the
#      sheet's bounds grown by Padding in solver axes, with the counts Spacing
#      gives, at T instants spanning the solve.
#   D. samples_match_the_evaluator_in_solver_axes: see above.
#   E. size_limit_refuses: a Spacing whose grids pass the Size Limit fails the
#      encode, naming the limit, and the estimate line is the one the panel
#      shows.
#   F. animated_field_moves_between_time_samples: a Force field keyframed from
#      x = -1 to x = +1 peaks near -1 in the first sample and near +1 in the
#      last one.
#   G. script_text_travels: the chosen Text's source is in the payload.
#   H. each_grid_covers_the_objects_it_pushes: a field narrowed to a far group
#      gets a box around that group's object only, while a field reaching
#      every group gets one around both, and their points are Spacing apart
#      at most.
#   I. a_box_thinner_than_spacing_is_refused: with Padding 0 the flat sheets
#      give a box 1 mm deep along Z, which they would leave as soon as they
#      move; the encode fails naming the axis and Padding, and so does the
#      plan the panel draws its estimate from.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True
BACKENDS = ("real",)

_DRIVER_BODY = r"""
import math
import traceback
import zlib

import numpy as np
from mathutils import Vector

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

try:
    dh = DriverHelpers(pkg, result)
    ff = __import__(pkg + ".core.force_field", fromlist=["encode"])
    params_mod = __import__(pkg + ".core.encoder.params", fromlist=["_build_param_dict"])
    enc = __import__(pkg + ".core.encoder", fromlist=["resolve_solver_fps"])

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=1.0,
                                    location=(0, 0, 0))
    sheet = bpy.context.object
    sheet.name = "FFSheet"
    dh.save_blend(PROBE_DIR, "force_field_encode.blend")
    root = dh.configure_state(project_name="force_field_encode", frame_count=11)
    state = root.state
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)

    state.force_field_padding = 0.2
    state.force_field_spacing = 0.25

    def field(kind, name, loc=(0.0, 0.0, 0.0)):
        bpy.ops.object.effector_add(type=kind, location=loc)
        o = bpy.context.object
        o.name = name
        return o

    def build():
        return params_mod._build_param_dict(bpy.context)

    def refused(fragment):
        try:
            build()
            return False, "no error"
        except ValueError as e:
            return fragment in str(e), str(e)

    # ----- A --------------------------------------------------------------
    magnet = field("MAGNET", "FFMagnet")
    ok, msg = refused("FFMagnet")
    dh.record("A_unsupported_type_refused", ok, {"error": msg})
    bpy.data.objects.remove(magnet)

    # ----- B --------------------------------------------------------------
    turb = field("TURBULENCE", "FFTurb", (0.1, 0.2, 0.0))
    turb.field.strength = 4.0
    turb.field.size = 0.7
    wind = field("WIND", "FFWind", (0.0, 0.0, -1.0))
    wind.rotation_euler = (0.4, 0.0, 0.3)
    wind.field.strength = 3.0
    ok, msg = refused("Air Density")
    dh.record("B_wind_needs_air_density", ok, {"error": msg})
    state.air_density = 1.0

    # ----- C --------------------------------------------------------------
    state.force_field_time_samples = 3
    built = build()
    ffp = built.get("force_field") or {}
    grids = {g["kind"]: g for g in ffp.get("grids", [])}
    fps = float(enc.resolve_solver_fps(state))
    bpy.context.view_layer.update()
    corners = np.array([sheet.matrix_world @ Vector(c) for c in sheet.bound_box])
    lo, hi = corners.min(axis=0) - 0.2, corners.max(axis=0) + 0.2
    # Blender (W, H, D) along (x, y, z); the solver's grid is (x, z, -y).
    w, h, d = (max(2, int(math.ceil(e / 0.25 - 1e-9)) + 1) for e in hi - lo)
    want_min = [lo[0], lo[2], -hi[1]]
    want_max = [hi[0], hi[2], -lo[1]]
    shape_ok = (
        set(grids) == {"acceleration", "air-velocity"}
        and all(list(g["shape"]) == [3, h, d, w, 3] for g in grids.values())
        and all(np.allclose(g["min"], want_min, atol=1e-5)
                and np.allclose(g["max"], want_max, atol=1e-5) for g in grids.values())
        and all(np.allclose(g["times"], [0.0, 5.0 / fps, 10.0 / fps]) for g in grids.values())
    )
    dh.record("C_payload_shape", shape_ok, {
        "kinds": sorted(grids),
        "shapes": {k: list(g["shape"]) for k, g in grids.items()},
        "want_shape": [3, h, d, w, 3],
        "min": {k: list(g["min"]) for k, g in grids.items()},
        "want_min": [float(v) for v in want_min],
        "times": {k: list(g["times"]) for k, g in grids.items()},
        "fps": fps})

    # ----- D --------------------------------------------------------------
    worst = 0.0
    if shape_ok:
        scene = bpy.context.scene
        start = int(enc.resolve_start_frame(state))
        scene.frame_set(start)
        samples = ff.snapshot(ff.field_objects(scene, state))
        rng = np.random.default_rng(7)
        for kind, g in grids.items():
            T, D, H, W, _ = g["shape"]
            arr = np.frombuffer(zlib.decompress(g["data"]), dtype=np.float32).reshape(g["shape"])
            smin = np.array(g["min"]); smax = np.array(g["max"])
            for _ in range(40):
                iz, iy, ix = rng.integers(D), rng.integers(H), rng.integers(W)
                xs = smin + (smax - smin) * np.array([ix / (W - 1), iy / (H - 1), iz / (D - 1)])
                pb = np.array([xs[0], -xs[2], xs[1]])
                acc, air = ff.evaluate(samples, pb[None, :])
                vb = (acc if kind == "acceleration" else air)[0]
                vs = np.array([vb[0], vb[2], -vb[1]])
                got = arr[0, iz, iy, ix]
                worst = max(worst, float(np.abs(got - vs).max() / (1e-6 + np.abs(vs).max())))
    dh.record("D_samples_match_the_evaluator_in_solver_axes",
              shape_ok and worst < 1e-4, {"worst_relative_error": worst})

    # ----- E --------------------------------------------------------------
    line = ff.estimate_line([(w, h, d), (w, h, d)], 3)
    # A millimeter spacing over the 1.4 m box is about 1400^2 x 400 points,
    # tens of GB at 3 instants, past the default 2000 MB: refused before a
    # single sample is taken.
    state.force_field_spacing = 0.001
    ok, msg = refused("limit")
    state.force_field_spacing = 0.25
    dh.record("E_size_limit_refuses", ok and "MB estimated" in line
              and line.startswith(f"[Info] Force field {w}x{h}x{d}x3, {w}x{h}x{d}x3:"),
              {"error": msg, "line": line})

    # ----- F --------------------------------------------------------------
    bpy.data.objects.remove(turb)
    bpy.data.objects.remove(wind)
    state.air_density = 0.0
    push = field("FORCE", "FFPush", (-1.0, 0.0, 0.3))
    push.field.strength = 5.0
    push.field.use_max_distance = True
    push.field.distance_max = 0.6
    push.keyframe_insert("location", frame=1)
    push.location = (1.0, 0.0, 0.3)
    push.keyframe_insert("location", frame=11)
    # A padding past the field's path, so the box holds it at both ends.
    state.force_field_padding = 1.4
    state.force_field_spacing = 0.1
    bpy.context.view_layer.update()
    state.force_field_time_samples = 2
    ffp = build().get("force_field") or {}
    g = (ffp.get("grids") or [None])[0]
    peaks = []
    if g is not None:
        arr = np.frombuffer(zlib.decompress(g["data"]), dtype=np.float32).reshape(g["shape"])
        xs = np.linspace(g["min"][0], g["max"][0], g["shape"][3])
        for k in range(2):
            mag = np.linalg.norm(arr[k], axis=-1).max(axis=(0, 1))
            # The field is zero outside its max distance, so the occupied
            # stretch of x is where the moving field sat at that sample.
            occupied = xs[mag > 1e-6]
            peaks.append(float(occupied.mean()) if len(occupied) else None)
    dh.record("F_animated_field_moves_between_time_samples",
              len(peaks) == 2 and None not in peaks
              and peaks[0] < -0.6 and peaks[1] > 0.6,
              {"occupied_x_center": peaks})

    # ----- G --------------------------------------------------------------
    text = bpy.data.texts.new("ff_script.py")
    text.from_string("def eval(x, y, z, t):\n    return (0.0, 0.0, 1.0)\n")
    state.force_field_script = text
    ffp = build().get("force_field") or {}
    (sc,) = ffp.get("scripts") or [{}]
    dh.record("G_script_text_travels",
              sc.get("source") == text.as_string() and sc.get("name") == "ff_script.py"
              and sc.get("groups") is None,
              {"script": sc})

    # ----- H --------------------------------------------------------------
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=0.5,
                                    location=(3.0, 0.0, 0.0))
    far = bpy.context.object
    far.name = "FFFar"
    far_group = dh.api.solver.create_group("Far", "SHELL")
    far_group.add(far.name)
    near = field("FORCE", "FFNear", (3.0, 0.0, 0.3))
    near.field.strength = 2.0
    dh.api.solver.set_force_field_targets("FFNear", [far_group])
    state.force_field_padding = 0.1
    state.force_field_spacing = 0.2
    bpy.context.scene.frame_set(1)
    plan = {e["uuids"]: e for e in ff.grid_plan(bpy.context.scene, state)}

    def bounds(objs):
        pts = np.array([o.matrix_world @ Vector(c) for o in objs for c in o.bound_box])
        return pts.min(axis=0) - 0.1, pts.max(axis=0) + 0.1

    ok_boxes = set(plan) == {None, (far_group.uuid,)}
    spacings = []
    if ok_boxes:
        every, narrowed = plan[None], plan[(far_group.uuid,)]
        for entry, objs in ((every, [sheet, far]), (narrowed, [far])):
            lo_w, hi_w = bounds(objs)
            ok_boxes = ok_boxes and np.allclose(entry["lo"], lo_w, atol=1e-5) \
                and np.allclose(entry["hi"], hi_w, atol=1e-5)
            gap = (entry["hi"] - entry["lo"]) / (np.array(entry["shape"]) - 1)
            spacings.append([float(v) for v in gap])
        # 1e-6: the Spacing property holds 0.2 in single precision.
        ok_boxes = ok_boxes and all(v <= 0.2 + 1e-6 for g in spacings for v in g)
    dh.record("H_each_grid_covers_the_objects_it_pushes", ok_boxes,
              {"plan": {str(k): {"lo": [float(v) for v in e["lo"]],
                                 "hi": [float(v) for v in e["hi"]],
                                 "shape": list(e["shape"]),
                                 "fields": [o.name for o in e["fields"]]}
                        for k, e in plan.items()},
               "spacings": spacings})

    # ----- I --------------------------------------------------------------
    state.force_field_padding = 0.0
    ok, msg = refused("Raise Padding")
    try:
        ff.grid_plan(bpy.context.scene, state)
        plan_msg = "no error"
    except ValueError as e:
        plan_msg = str(e)
    state.force_field_padding = 0.1
    dh.record("I_a_box_thinner_than_spacing_is_refused",
              ok and "along Z" in msg and "Raise Padding" in plan_msg,
              {"encode": msg, "plan": plan_msg})

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
