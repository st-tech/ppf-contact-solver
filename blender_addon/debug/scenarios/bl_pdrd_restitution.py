# File: scenarios/bl_pdrd_restitution.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# A PDRD group's Restitution, from the Blender panel to the solver's step.
#
# THE SCENE is three PDRD groups side by side, each holding one icosphere
# 0.15 m above an invisible wall, frictionless, under gravity: "Bounce" at 0.8,
# the control, "Control", at 0, and "Fade", keyframed from 0.8 down to 0. Nothing else touches either ball, so the Restitution
# field alone decides how high each comes back. One build and one run serve
# every check, which keeps the scenario as quick as its neighbours.
#
# THE WITNESSES are the encoded param payload, the solver's own body table
# (`bin/pdrd_body.bin`, 24 floats per body with the restitution last), which
# is what the solver reads the coefficient from, and the fetched animation.
#
# Subtests:
#   A_rna_registered_with_default  the field exists, 0 by default, [0, 1].
#                                  New RNA needs a full Blender restart.
#   B_encoded_for_pdrd_only        the payload carries the field's value as
#                                  `restitution` on a PDRD group and carries
#                                  none once the group is a SHELL.
#   C_material_lock_covers_it      the field is in the material-lock list, so
#                                  a locked group cannot be edited behind it.
#   D_build_writes_body_row        the solver's body table holds 0.8, 0 and
#                                  (at the start frame) 0.8 in the bodies'
#                                  restitution slots.
#   E_bounces_when_set             at 0.8 the ball comes back up well above the
#                                  floor after its first impact.
#   F_stops_when_zero              at 0 (the control) it stays on the floor.
#   G_keyframes_reach_solver       a third ball's Restitution keyframed from
#                                  0.8 to 0 before it lands rides the payload
#                                  as a schedule and reaches the solver's
#                                  per-body table (`param_anim/
#                                  pdrd-restitution.bin`).
#   H_keyframed_zero_stops_it      that ball lands after its value reached 0
#                                  and stays down, although the value read at
#                                  the start frame is 0.8: the solver follows
#                                  the schedule, not the build-time value.

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
FRAMES = 36
DROP = 0.15
RADIUS = 0.1
ROW = 24


def session_dir():
    hits = [os.path.dirname(p) for p in glob.glob(
        os.path.join(PROJECT_ROOT, "**", "session", "info.toml"),
        recursive=True)]
    if not hits:
        raise RuntimeError("no session/info.toml under %s" % PROJECT_ROOT)
    hits.sort(key=os.path.getmtime, reverse=True)
    return hits[0]


def body_rows():
    path = os.path.join(session_dir(), "bin", "pdrd_body.bin")
    if not os.path.isfile(path):
        return None
    return np.fromfile(path, dtype=np.float32)


def group_params(param_bytes, object_name):
    for params, objects, _uuids in dh.decode_addon_blob(param_bytes)["group"]:
        if object_name in objects:
            return params
    return None


def phase(name):
    result["phases"].append([name, round(time.time() - START, 2)])


def pc2_samples(obj):
    path = dh.find_pc2_for(obj)
    if not path or not os.path.isfile(path):
        return 0
    try:
        return len(dh.read_pc2(path))
    except (OSError, ValueError):
        return 0


def fetch_frames(expected, timeout=60.0):
    # The frames the run's live fetch has not delivered yet, fetched and
    # applied, finished when every ball's cache holds every frame: those caches
    # are what the checks read. The frames already fetched are kept, so this
    # costs only the stragglers. (`fetch_and_drain` clears them and fetches the
    # whole run again, then waits on the runner's frame total, which stays 0
    # here, so it sits out its drain timeout.)
    dh.com.fetch()
    deadline = time.time() + timeout
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        dh.client.apply_animation()
        state = dh.facade.engine.state
        samples = min(pc2_samples(ball) for ball in balls.values())
        if samples >= expected and state.activity.name == "IDLE":
            return samples
        time.sleep(0.1)
    return min(pc2_samples(ball) for ball in balls.values())


def rebound_of(obj):
    # The first impact is the first frame the lowest point stops falling;
    # the rebound is the highest it climbs after that, over the drop.
    pc2 = np.asarray(dh.read_pc2(dh.find_pc2_for(obj)))
    low = pc2[:, :, 2].min(axis=1)
    first = next((k for k in range(1, len(low) - 1)
                  if low[k] <= low[k - 1] and low[k] < low[k + 1]), len(low) - 1)
    return float(low[first:].max() / low[0]) if first < len(low) - 1 else 0.0


try:
    dh = DriverHelpers(pkg, result)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = FRAMES

    balls = {}
    for name, x in (("Ball", -0.4), ("Control", 0.0), ("Fade", 0.4)):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=RADIUS,
                                              location=(x, 0.0, RADIUS + DROP))
        balls[name] = bpy.context.object
        balls[name].name = name
        bpy.context.view_layer.update()
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    dh.save_blend(PROBE_DIR, "pdrd_restitution.blend")
    root = dh.configure_state(project_name=PROJECT_NAME, frame_count=FRAMES,
                              gravity=(0.0, 0.0, -9.8))
    root.state.disable_contact = False
    dh.api.solver.add_wall((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    groups_mod = __import__(pkg + ".models.groups",
                            fromlist=["get_group_slot_index"])
    addon_root = dh.groups.get_addon_data(scene)
    rnas = {}
    for label, name in (("Bounce", "Ball"), ("Held", "Control"), ("Fading", "Fade")):
        group = dh.api.solver.create_group(label, "PDRD")
        group.add(name)
        rnas[name] = getattr(addon_root, "object_group_%d"
                             % groups_mod.get_group_slot_index(scene, group.uuid))
        rnas[name].friction = 0.0
    rna = rnas["Ball"]
    phase("scene")

    # A: the field, 0 by default, bounded to [0, 1].
    prop = rna.bl_rna.properties.get("pdrd_restitution")
    dh.record(
        "A_rna_registered_with_default",
        prop is not None and prop.type == "FLOAT"
        and rna.pdrd_restitution == 0.0
        and prop.hard_min == 0.0 and prop.hard_max == 1.0,
        {"type": None if prop is None else prop.type,
         "value": None if prop is None else rna.pdrd_restitution,
         "range": None if prop is None else [prop.hard_min, prop.hard_max],
         "note": "new RNA needs a full Blender restart, not a reload"},
    )

    # B: on the wire for PDRD, and not for another type.
    rna.pdrd_restitution = 0.8
    _data, param_bytes = dh.encode_payload()
    as_pdrd = (group_params(param_bytes, "Ball") or {}).get("restitution")
    rna.object_type = "SHELL"
    _data, param_bytes = dh.encode_payload()
    as_shell = (group_params(param_bytes, "Ball") or {}).get("restitution", "absent")
    rna.object_type = "PDRD"
    dh.record(
        "B_encoded_for_pdrd_only",
        as_pdrd is not None and abs(float(as_pdrd) - 0.8) < 1e-6
        and (as_shell == "absent" or float(as_shell) == 0.0),
        {"pdrd": None if as_pdrd is None else float(as_pdrd),
         "shell": as_shell if as_shell == "absent" else float(as_shell)},
    )

    # C: the material lock covers the field.
    locks = __import__(pkg + ".models.material_locks", fromlist=["x"])
    names = set()
    for value in vars(locks).values():
        if isinstance(value, (set, frozenset, list, tuple)):
            names.update(v for v in value if isinstance(v, str))
    dh.record("C_material_lock_covers_it", "pdrd_restitution" in names,
              {"found": "pdrd_restitution" in names})

    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root.state.project_name)

    # D, E, F: one build and one run, the bouncing ball and the control.
    rnas["Ball"].pdrd_restitution = 0.8
    rnas["Control"].pdrd_restitution = 0.0
    # G, H: keyed 0.8 at the first frame and 0 a few frames later, long before
    # the ball lands (a 0.15 m drop takes 0.17 s, frame 18 at 100 fps).
    fade = rnas["Fade"]
    fade.pdrd_restitution = 0.8
    keyed = fade.keyframe_insert("pdrd_restitution", frame=1)
    fade.pdrd_restitution = 0.0
    keyed = fade.keyframe_insert("pdrd_restitution", frame=6) and keyed
    scene.frame_set(1)
    _data, param_bytes = dh.encode_payload()
    series = ((group_params(param_bytes, "Fade") or {}).get("param-anim") or {}).get("restitution")
    data_bytes, param_bytes = dh.encode_payload()
    built, text = True, ""
    try:
        dh.build_and_wait(data_bytes, param_bytes,
                          message="restitution", timeout=240.0)
    except RuntimeError as error:
        built, text = False, str(error)
    phase("build")
    rows = body_rows() if built else None
    frame, bounce, control, faded = None, None, None, None
    if built:
        dh.run_and_wait(timeout=600.0)
        phase("run")
        # The last frame's index is FRAMES - 1; waiting for FRAMES would sit
        # out the whole timeout.
        dh.force_frame_query(expected_frames=FRAMES - 1, timeout=60.0)
        state = dh.facade.engine.state
        frame = int(state.frame) if state.solver.name != "FAILED" else -1
        phase("query to frame %d" % frame)
        samples = fetch_frames(FRAMES - 1)
        phase("fetch %s samples" % samples)
        bounce = rebound_of(balls["Ball"])
        control = rebound_of(balls["Control"])
        faded = rebound_of(balls["Fade"])
    slots = (None if rows is None or rows.size % ROW
             else sorted(float(v) for v in rows[ROW - 1::ROW]))
    dh.record("D_build_writes_body_row",
              built and slots is not None and len(slots) == 3
              and abs(slots[0]) < 1e-6 and abs(slots[1] - 0.8) < 1e-6
              and abs(slots[2] - 0.8) < 1e-6,
              {"error": text[-600:],
               "floats": None if rows is None else int(rows.size),
               "restitution_slots": slots})
    dh.record("E_bounces_when_set",
              built and frame >= FRAMES - 1 and bounce is not None and bounce > 0.3,
              {"frame": frame, "rebound_over_drop": bounce})
    dh.record("F_stops_when_zero",
              built and frame >= FRAMES - 1 and control is not None and control < 0.05,
              {"frame": frame, "rebound_over_drop": control})
    table = None
    if built:
        path = os.path.join(session_dir(), "bin", "param_anim", "pdrd-restitution.bin")
        if os.path.isfile(path):
            table = np.fromfile(path, dtype=np.float32)
    dh.record("G_keyframes_reach_solver",
              bool(keyed) and series is not None and len(series) > 1
              and abs(float(series[0]) - 0.8) < 1e-6 and abs(float(series[-1])) < 1e-6
              and table is not None and table.size % 3 == 0
              and abs(float(table.max()) - 0.8) < 1e-6 and abs(float(table[-3:].min())) < 1e-6,
              {"keyed": bool(keyed),
               "series": None if series is None else [float(series[0]), float(series[-1]), len(series)],
               "table": None if table is None else [int(table.size), float(table.max()),
                                                    [float(v) for v in table[-3:]]]})
    dh.record("H_keyframed_zero_stops_it",
              built and frame >= FRAMES - 1 and faded is not None and faded < 0.05,
              {"frame": frame, "rebound_over_drop": faded})
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
    # Where the time went, seconds since the driver started.
    verdict.setdefault("notes", []).append(
        "phases: " + ", ".join("%s %.1f" % (n, t) for n, t in result.get("phases", [])))
    return verdict
