# File: scenarios/bl_time_scale_encoding.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Time Scale encoding contract. state.time_scale re-interprets how much
# simulated time one Blender frame covers, implemented entirely encoder-side
# via resolve_solver_fps = resolve_fps * time_scale. Every quantity the
# solver consumes as SECONDS must scale together, or one animation channel
# desyncs from the rest of the scene when time_scale != 1.
#
# One scene carries every time-carrying channel, encoded twice
# (time_scale 1.0 then 0.5). The PARAM payload carries the whole time base:
# its times double, its fps halves, and it ships the time_scale factor. The
# DATA payload is timing-free (frame offsets / raw animation rates), so it
# must be BYTE-IDENTICAL across the flip: that is what makes a Time Scale
# edit an Update-Params-only change, never a geometry re-transfer.
#
# Channels:
#   * pin move_by + spin operation windows (param side, seconds)
#   * captured pin deformation track times (param side, seconds)
#   * STATIC rigid motion via object fcurves (data side, frame_offset)
#   * STATIC ops incl. SPIN (data side, frame offsets + raw rate)
#   * STATIC captured deformation (data side, rows ARE frame offsets)
#
# Subtests:
#   A. fps_param_halves
#   B. pin_op_times_double
#   C. pin_capture_times_double
#   D. data_payload_invariant (byte-identical Scene CBOR across the flip)
#   E. data_channels_frame_indexed (frame_offset present; no seconds keys)
#   F. structure_and_counts_unchanged
#   G. spin_rate_scales_down (pin-side rate multiplies by time_scale)

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

FRAME_COUNT = 6
FPS = 100

# Time-like leaf keys the walker collects. "time" lists are matched with
# their surrounding path to attribute them to a channel.
_TIME_KEYS = ("t_start", "t_end", "unpin_time")


def _walk(node, path, out):
    # Collect (path, key, value) for every time-like leaf. Values are
    # either a float scalar (t_start/t_end/fps) or a list of floats
    # ("time" arrays). Non-numeric leaves are ignored.
    if isinstance(node, dict):
        for k, v in node.items():
            p = path + "/" + str(k)
            if k in _TIME_KEYS and isinstance(v, (int, float)):
                out.append((p, k, float(v)))
            elif k == "fps" and isinstance(v, (int, float)):
                out.append((p, k, float(v)))
            elif k == "angular_velocity" and isinstance(v, (int, float)):
                out.append((p, k, float(v)))
            elif k == "time" and isinstance(v, (list, tuple)) and v and all(
                isinstance(x, (int, float)) for x in v
            ):
                out.append((p, k, [float(x) for x in v]))
            else:
                _walk(v, p, out)
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            _walk(v, path + "[%d]" % i, out)


def _collect(data_tree, param_tree):
    out = []
    _walk(data_tree, "data", out)
    _walk(param_tree, "param", out)
    return out


def _pairs(leaves1, leaves2):
    # Zip by path; encoding is deterministic so paths must match 1:1.
    d1 = {p: (k, v) for p, k, v in leaves1}
    d2 = {p: (k, v) for p, k, v in leaves2}
    return d1, d2


def g_ops_assigned_iter(dh, group_name):
    root = dh.groups.get_addon_data(bpy.context.scene)
    for i in range(32):
        g = getattr(root, "object_group_%d" % i, None)
        if g is not None and g.active and g.name == group_name:
            for a in g.assigned_objects:
                yield a


def _doubled(a, b, rel=1e-6):
    # b must equal 2*a (zeros stay zero).
    if isinstance(a, list):
        return len(a) == len(b) and all(_doubled(x, y, rel) for x, y in zip(a, b))
    return abs(b - 2.0 * a) <= rel * max(1.0, abs(a))


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    root = dh.configure_state(
        project_name="time_scale_encoding",
        frame_count=FRAME_COUNT,
        frame_rate=FPS,
    )
    state = root.state

    # --- channel 1: SHELL with a move_by pin op --------------------------
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    plane_op = bpy.context.active_object
    plane_op.name = "PlaneOp"
    vg = plane_op.vertex_groups.new(name="OpPin")
    vg.add(list(range(len(plane_op.data.vertices))), 1.0, "REPLACE")
    g_op = dh.api.solver.create_group("ClothOp", "SHELL")
    g_op.add(plane_op.name)
    pin_op = g_op.create_pin(plane_op.name, "OpPin")
    pin_op.move_by(delta=(0.1, 0.0, 0.1), frame_start=1, frame_end=4)
    # A rate-parameterized op alongside: SPIN is authored in ANIMATION
    # degrees/second, so its emitted rate must MULTIPLY by time_scale
    # (halve at 0.5) to keep rotation-per-frame invariant, the opposite
    # direction of the time leaves.
    pin_op.spin(axis=(0.0, 0.0, 1.0), angular_velocity=90.0,
                frame_start=1, frame_end=4)

    # --- channel 2: SHELL with a captured pin deformation cache ----------
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(3.0, 0.0, 0.0))
    plane_cap = bpy.context.active_object
    plane_cap.name = "PlaneCap"
    vg2 = plane_cap.vertex_groups.new(name="CapPin")
    pin_indices = list(range(len(plane_cap.data.vertices)))
    vg2.add(pin_indices, 1.0, "REPLACE")
    g_cap = dh.api.solver.create_group("ClothCap", "SHELL")
    g_cap.add(plane_cap.name)
    g_cap.create_pin(plane_cap.name, "CapPin")

    pc2 = __import__(pkg + ".core.pc2", fromlist=[
        "write_pin_anim_pc2", "write_static_deform_pc2",
    ])
    pin_ops_mod = __import__(pkg + ".ui.dynamics.pin_ops",
                             fromlist=["_ensure_embedded_move_op"])
    # Synthetic dense capture: row k = row 0 shifted linearly. Values are
    # irrelevant to this scenario; only the emitted time list matters.
    cap = np.zeros((5, len(pin_indices), 3), dtype=np.float32)
    for k in range(cap.shape[0]):
        cap[k, :, 1] = 0.05 * k
    pc2.write_pin_anim_pc2(plane_cap, "CapPin", cap)
    addon_root = dh.groups.get_addon_data(bpy.context.scene)
    # Mirror the modal capture finalize (bl_pin_capture_deformation).
    cap_group = None
    for i in range(32):
        g = getattr(addon_root, "object_group_%d" % i, None)
        if g is not None and g.active and g.name == "ClothCap":
            cap_group = g
            break
    pin_item = cap_group.pin_vertex_groups[0]
    pin_item.has_captured_anim = True
    pin_ops_mod._ensure_embedded_move_op(pin_item)

    # --- channel 3: STATIC rigid motion via location fcurves -------------
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(6.0, 0.0, 2.0))
    cube_rigid = bpy.context.active_object
    cube_rigid.name = "CubeRigid"
    cube_rigid.keyframe_insert(data_path="location", frame=1)
    cube_rigid.location = (6.0, 0.0, 2.5)
    cube_rigid.keyframe_insert(data_path="location", frame=5)
    g_rigid = dh.api.solver.create_group("StatRigid", "STATIC")
    g_rigid.add(cube_rigid.name)

    # --- channel 3b: STATIC with UI-assigned ops incl. SPIN --------------
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(12.0, 0.0, 0.0))
    cube_ops = bpy.context.active_object
    cube_ops.name = "CubeOps"
    g_ops = dh.api.solver.create_group("StatOps", "STATIC")
    g_ops.add(cube_ops.name)
    for _a in g_ops_assigned_iter(dh, "StatOps"):
        op = _a.static_ops.add()
        op.op_type = "SPIN"
        op.frame_start = 1
        op.frame_end = 5
        op.spin_axis = (0.0, 0.0, 1.0)
        op.spin_angular_velocity = 90.0

    # --- channel 4: STATIC captured deformation cache --------------------
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(9.0, 0.0, 0.0))
    cube_def = bpy.context.active_object
    cube_def.name = "CubeDeform"
    n_cv = len(cube_def.data.vertices)
    rows = np.zeros((FRAME_COUNT, n_cv, 3), dtype=np.float32)
    for k in range(FRAME_COUNT):
        rows[k, :, 1] = 0.02 * k
    pc2.write_static_deform_pc2(cube_def, rows)
    g_def = dh.api.solver.create_group("StatDeform", "STATIC")
    g_def.add(cube_def.name)

    bpy.context.view_layer.update()

    # --- encode at time_scale 1.0 and 0.5 --------------------------------
    state.time_scale = 1.0
    data1_b, param1_b = dh.encode_payload()
    state.time_scale = 0.5
    data2_b, param2_b = dh.encode_payload()
    state.time_scale = 1.0

    leaves1 = _collect(dh.decode_addon_blob(data1_b), dh.decode_addon_blob(param1_b))
    leaves2 = _collect(dh.decode_addon_blob(data2_b), dh.decode_addon_blob(param2_b))
    d1, d2 = _pairs(leaves1, leaves2)
    same_paths = set(d1.keys()) == set(d2.keys())

    def _channel(pred):
        # (n_leaves, all_doubled, sample_paths)
        paths = [p for p in d1 if pred(p, d1[p][0])]
        oks = [
            p in d2 and _doubled(d1[p][1], d2[p][1])
            for p in paths
        ]
        return len(paths), all(oks) if oks else False, paths[:4]

    # A: every fps leaf exactly halves.
    fps_paths = [p for p in d1 if d1[p][0] == "fps"]
    fps_ok = bool(fps_paths) and all(
        p in d2 and abs(d2[p][1] - 0.5 * d1[p][1]) <= 1e-9 * max(1.0, d1[p][1])
        for p in fps_paths
    )
    dh.record("A_fps_param_halves", fps_ok and same_paths, {
        "fps_1": [d1[p][1] for p in fps_paths],
        "fps_05": [d2[p][1] for p in fps_paths if p in d2],
        "same_paths": same_paths,
    })

    # B: pin op windows (t_start / t_end under operations).
    n_b, ok_b, sample_b = _channel(
        lambda p, k: k in ("t_start", "t_end") and "operation" in p
    )
    dh.record("B_pin_op_times_double", n_b >= 2 and ok_b,
              {"n_leaves": n_b, "sample": sample_b})

    # C: captured pin track times ("time" lists in the param payload,
    # excluding the data-side transform / static-deform channels).
    n_c, ok_c, sample_c = _channel(
        lambda p, k: k == "time" and p.startswith("param")
        and "transform_animation" not in p
        and "static_deform_animation" not in p
    )
    dh.record("C_pin_capture_times_double", n_c >= 1 and ok_c,
              {"n_leaves": n_c, "sample": sample_c})

    # D: the DATA payload is timing-free, so it must be BYTE-IDENTICAL
    # across the time_scale flip (hence hash-equal: a Time Scale edit is an
    # Update-Params-only change). Strictly stronger than any per-leaf check
    # and also catches a static-op rate leaking Time Scale into the data.
    dh.record("D_data_payload_invariant", data1_b == data2_b,
              {"len_1": len(data1_b), "len_05": len(data2_b),
               "identical": data1_b == data2_b})

    # E: the data-side channels ship frame-indexed timing: frame_offset
    # keys present, and no seconds keys ("time" lists / t_start) anywhere
    # in the data payload.
    def _keys(node, out):
        if isinstance(node, dict):
            for k, v in node.items():
                out.add(k)
                _keys(v, out)
        elif isinstance(node, (list, tuple)):
            for v in node:
                _keys(v, out)
    data_keys = set()
    _keys(dh.decode_addon_blob(data1_b), data_keys)
    has_frame_idx = ("frame_offset" in data_keys
                     and "frame_offset_start" in data_keys
                     and "angular_velocity_anim" in data_keys
                     and "vert_frames" in data_keys)
    no_seconds = ("time" not in data_keys and "t_start" not in data_keys
                  and "angular_velocity" not in data_keys)
    dh.record("E_data_channels_frame_indexed", has_frame_idx and no_seconds,
              {"has_frame_idx": has_frame_idx, "no_seconds": no_seconds,
               "keys_sample": sorted(k for k in data_keys if "frame" in k
                                     or k in ("time", "t_start"))})

    # F: structure unchanged — identical leaf paths, identical leaf count,
    # frames param untouched by the time re-interpretation.
    p1 = dh.decode_addon_blob(param1_b)
    p2 = dh.decode_addon_blob(param2_b)

    def _find_frames(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "frames" and isinstance(v, (int, float)):
                    return v
                got = _find_frames(v)
                if got is not None:
                    return got
        elif isinstance(node, (list, tuple)):
            for v in node:
                got = _find_frames(v)
                if got is not None:
                    return got
        return None

    frames1, frames2 = _find_frames(p1), _find_frames(p2)
    dh.record("F_structure_and_counts_unchanged",
              same_paths and len(leaves1) == len(leaves2)
              and frames1 is not None and frames1 == frames2,
              {"n_leaves": (len(leaves1), len(leaves2)),
               "frames": (frames1, frames2), "same_paths": same_paths})

    # G: rate-parameterized op values HALVE (rate * time_scale), keeping
    # rotation-per-frame invariant; they must never double like time leaves.
    rate_paths = [p for p in d1 if d1[p][0] == "angular_velocity"]
    rate_ok = bool(rate_paths) and all(
        p in d2 and abs(d2[p][1] - 0.5 * d1[p][1]) <= 1e-9 * max(1.0, abs(d1[p][1]))
        for p in rate_paths
    )
    dh.record("G_spin_rate_scales_down", rate_ok, {
        "n_rate_leaves": len(rate_paths),
        "rates_1": [d1[p][1] for p in rate_paths],
        "rates_05": [d2[p][1] for p in rate_paths if p in d2],
    })

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 180.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
