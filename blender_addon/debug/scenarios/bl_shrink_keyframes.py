# File: scenarios/bl_shrink_keyframes.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# A keyframed shrink, from the Blender slider to the solver's rest shape.
#
# THE SCENE is three free bodies side by side with gravity off, far enough
# apart that nothing touches: "Sheet", a subdivided plane whose Shell group
# keys Shrink X from 1.0 at frame 1 to SHEET_END at frame KEY_END; "Steady", the
# same plane in a Shell group with no keys, the control; and "Ball", an
# icosphere whose Solid group keys Shrink from 1.0 to BALL_END over the same
# frames. With no gravity and no contact, the keyed rest shape alone decides
# each body's size, so the measured extents are a direct read of the schedule.
# One build and one run serve every check.
#
# Subtests:
#   A_props_offer_keyframes       shrink_x, shrink_y and shrink are animatable
#                                 RNA, and a Rod's Shrink (length_factor) is not.
#   B_payload_carries_schedule    the encoded groups carry `shrink-x` (Sheet) and
#                                 `shrink` (Ball) series from 1.0 to the keyed end.
#   C_session_carries_schedule    the session has param_anim/tri-shrink-x.bin
#                                 (keyframes x triangles) and tet-shrink.bin
#                                 (keyframes x tetrahedra), the layouts the
#                                 solver reads.
#   D_sheet_shrinks_gradually     Sheet's UV-X extent is part-way down at frame
#                                 MID and ends near SHEET_END of its start, its
#                                 UV-Y extent unchanged.
#   E_ball_shrinks_gradually      Ball's extent is part-way down at MID and ends
#                                 near BALL_END of its start.
#   F_control_holds_its_size      Steady keeps its size.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
BACKENDS = ("real",)
NOT_PARALLELIZABLE = True


_DRIVER_BODY = r"""
import glob
import os
import time
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
START = time.time()
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT_NAME = "<<PROJECT_NAME>>"
PROJECT_ROOT = "<<PROJECT_ROOT>>"
FRAMES = 40
KEY_END = 15
MID = 7
SHEET_END = 0.6
BALL_END = 0.7


def phase(name):
    result["phases"].append([name, round(time.time() - START, 2)])


def session_dir():
    hits = [os.path.dirname(p) for p in glob.glob(
        os.path.join(PROJECT_ROOT, "**", "session", "info.toml"),
        recursive=True)]
    if not hits:
        raise RuntimeError("no session/info.toml under %s" % PROJECT_ROOT)
    hits.sort(key=os.path.getmtime, reverse=True)
    return hits[0]


def group_params(param_bytes, object_name):
    for params, objects, _uuids in dh.decode_addon_blob(param_bytes)["group"]:
        if object_name in objects:
            return params
    return None


def pc2_samples(obj):
    path = dh.find_pc2_for(obj)
    if not path or not os.path.isfile(path):
        return 0
    try:
        return len(dh.read_pc2(path))
    except (OSError, ValueError):
        return 0


def fetch_frames(expected, timeout=60.0):
    # Fetch the frames the live fetch has not delivered yet and wait until
    # every body's cache holds them all; those caches are what the checks read.
    dh.com.fetch()
    deadline = time.time() + timeout
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        dh.client.apply_animation()
        state = dh.facade.engine.state
        samples = min(pc2_samples(o) for o in bodies.values())
        if samples >= expected and state.activity.name == "IDLE":
            return samples
        time.sleep(0.1)
    return min(pc2_samples(o) for o in bodies.values())


def extents(obj):
    # Per frame, the body's extent along world X, Y and Z.
    pc2 = np.asarray(dh.read_pc2(dh.find_pc2_for(obj)))
    return pc2.max(axis=1) - pc2.min(axis=1)


def file_floats(path, dtype=np.float32):
    return os.path.getsize(path) // np.dtype(dtype).itemsize if os.path.isfile(path) else 0


try:
    dh = DriverHelpers(pkg, result)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = FRAMES

    bodies = {}
    for name, x in (("Sheet", -0.7), ("Steady", 0.0)):
        # The primitive plane's UV maps U to world X, so Shrink X acts on X.
        bpy.ops.mesh.primitive_plane_add(size=0.4, location=(x, 0.0, 0.5))
        obj = bpy.context.object
        obj.name = name
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.subdivide(number_cuts=8)
        bpy.ops.object.mode_set(mode="OBJECT")
        bodies[name] = obj
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=0.12,
                                          location=(0.7, 0.0, 0.5))
    bodies["Ball"] = bpy.context.object
    bodies["Ball"].name = "Ball"
    for obj in bodies.values():
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    dh.save_blend(PROBE_DIR, "shrink_keyframes.blend")
    root = dh.configure_state(project_name=PROJECT_NAME, frame_count=FRAMES,
                              gravity=(0.0, 0.0, 0.0))
    groups_mod = __import__(pkg + ".models.groups",
                            fromlist=["get_group_slot_index"])
    addon_root = dh.groups.get_addon_data(scene)
    rnas = {}
    for label, name, kind in (("Shrinking", "Sheet", "SHELL"),
                              ("Control", "Steady", "SHELL"),
                              ("Solid", "Ball", "SOLID")):
        group = dh.api.solver.create_group(label, kind)
        group.add(name)
        rnas[name] = getattr(addon_root, "object_group_%d"
                             % groups_mod.get_group_slot_index(scene, group.uuid))
    phase("scene")

    # A: the keyframe offer.
    props = rnas["Sheet"].bl_rna.properties
    offered = {p: props[p].is_animatable for p in ("shrink_x", "shrink_y", "shrink")}
    rod_offered = props["length_factor"].is_animatable
    dh.record("A_props_offer_keyframes",
              all(offered.values()) and not rod_offered,
              {"shrink": offered, "length_factor": rod_offered})

    sheet, ball = rnas["Sheet"], rnas["Ball"]
    sheet.shrink_x = 1.0
    keyed = sheet.keyframe_insert("shrink_x", frame=1)
    sheet.shrink_x = SHEET_END
    keyed = sheet.keyframe_insert("shrink_x", frame=KEY_END) and keyed
    ball.shrink = 1.0
    keyed = ball.keyframe_insert("shrink", frame=1) and keyed
    ball.shrink = BALL_END
    keyed = ball.keyframe_insert("shrink", frame=KEY_END) and keyed
    scene.frame_set(1)

    # B: the schedules on the wire.
    data_bytes, param_bytes = dh.encode_payload()
    sx = ((group_params(param_bytes, "Sheet") or {}).get("param-anim") or {}).get("shrink-x")
    sb = ((group_params(param_bytes, "Ball") or {}).get("param-anim") or {}).get("shrink")
    dh.record("B_payload_carries_schedule",
              bool(keyed) and sx is not None and sb is not None
              and abs(float(sx[0]) - 1.0) < 1e-6 and abs(float(sx[-1]) - SHEET_END) < 1e-6
              and abs(float(sb[0]) - 1.0) < 1e-6 and abs(float(sb[-1]) - BALL_END) < 1e-6,
              {"keyed": bool(keyed),
               "shrink-x": None if sx is None else [float(sx[0]), float(sx[-1]), len(sx)],
               "shrink": None if sb is None else [float(sb[0]), float(sb[-1]), len(sb)]})

    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root.state.project_name)
    built, text = True, ""
    try:
        dh.build_and_wait(data_bytes, param_bytes, message="shrink keyframes",
                          timeout=240.0)
    except RuntimeError as error:
        built, text = False, str(error)
    phase("build")

    # C: the schedules in the session, sized as the solver reads them.
    layout = {}
    if built:
        session = session_dir()
        anim = os.path.join(session, "bin", "param_anim")
        param = os.path.join(session, "bin", "param")
        layout = {
            "files": sorted(os.listdir(anim)) if os.path.isdir(anim) else [],
            "n_times": file_floats(os.path.join(anim, "times.bin"), np.float64),
            "n_tri": file_floats(os.path.join(param, "tri-shrink-x.bin")),
            "n_tet": file_floats(os.path.join(param, "tet-shrink.bin")),
            "tri_values": file_floats(os.path.join(anim, "tri-shrink-x.bin")),
            "tet_values": file_floats(os.path.join(anim, "tet-shrink.bin")),
        }
    dh.record("C_session_carries_schedule",
              built and layout.get("n_times", 0) >= 2
              and layout["n_tri"] > 0 and layout["n_tet"] > 0
              and layout["tri_values"] == layout["n_times"] * layout["n_tri"]
              and layout["tet_values"] == layout["n_times"] * layout["n_tet"],
              {"error": text[-600:], **layout})

    frame = None
    ext = {}
    if built:
        dh.run_and_wait(timeout=600.0)
        phase("run")
        dh.force_frame_query(expected_frames=FRAMES - 1, timeout=60.0)
        state = dh.facade.engine.state
        frame = int(state.frame) if state.solver.name != "FAILED" else -1
        samples = fetch_frames(FRAMES - 1)
        phase("fetch %s samples at frame %s" % (samples, frame))
        ext = {name: extents(obj) for name, obj in bodies.items()}

    def ratio(name, axis, k):
        e = ext[name]
        return float(e[min(k, len(e) - 1), axis] / e[0, axis])

    ok_run = built and frame is not None and frame >= FRAMES - 1 and bool(ext)
    sheet_info, ball_info, steady_info = {}, {}, {}
    if ok_run:
        sheet_info = {"x_mid": ratio("Sheet", 0, MID), "x_end": ratio("Sheet", 0, FRAMES),
                      "y_end": ratio("Sheet", 1, FRAMES)}
        ball_info = {"mid": ratio("Ball", 0, MID), "end": ratio("Ball", 0, FRAMES)}
        steady_info = {"x_end": ratio("Steady", 0, FRAMES), "y_end": ratio("Steady", 1, FRAMES)}
    dh.record("D_sheet_shrinks_gradually",
              ok_run and abs(sheet_info["x_end"] - SHEET_END) < 0.08
              and sheet_info["x_end"] + 0.05 < sheet_info["x_mid"] < 0.98
              and abs(sheet_info["y_end"] - 1.0) < 0.08,
              {"frame": frame, "expected_x_end": SHEET_END, **sheet_info})
    dh.record("E_ball_shrinks_gradually",
              ok_run and abs(ball_info["end"] - BALL_END) < 0.06
              and ball_info["end"] + 0.03 < ball_info["mid"] < 0.99,
              {"frame": frame, "expected_end": BALL_END, **ball_info})
    dh.record("F_control_holds_its_size",
              ok_run and abs(steady_info["x_end"] - 1.0) < 0.03
              and abs(steady_info["y_end"] - 1.0) < 0.03,
              {"frame": frame, **steady_info})
except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
        .replace("<<PROJECT_NAME>>", ctx.project_name)
        .replace("<<PROJECT_ROOT>>", ctx.project_root.replace("\\", "/"))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 900.0))
    if err is not None:
        return err
    verdict = r.report_named_checks(result.get("checks", {}))
    verdict.setdefault("notes", []).append(
        "phases: " + ", ".join("%s %.1f" % (n, t) for n, t in result.get("phases", [])))
    return verdict
