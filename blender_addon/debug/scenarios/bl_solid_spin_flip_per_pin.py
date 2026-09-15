# File: scenarios/bl_solid_spin_flip_per_pin.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Two SPIN pins on ONE object must each rotate in their OWN direction, for a
# SOLID exactly as for a SHELL.
#
# The scene is the one reported in discussion #148: a bar along X with two
# pin vertex groups, "pinLeft" on the +X side and "pinRight" on the -X side.
# Both spin about +X at the same rate; only pinRight has Flip Direction set,
# so the two sides must turn in opposite directions. Three bars are solved in
# one run:
#
#   * a SHELL bar with its two end caps pinned (the control);
#   * a SOLID bar with its two end caps pinned, which decodes through the
#     partial-pin path (diffused pull weights, split into hard and soft
#     sub-holders);
#   * a SOLID bar whose two halves together pin EVERY vertex, which decodes
#     through the full-pin path (harmonic interior). Its spin rate is low so
#     the twist where the halves meet stays mild.
#
# What is measured: the HARD (pull == 0) pin vertices of each bar, read from
# the solver session (``info.toml`` pin blocks, ``bin/pin-ind-<k>.bin``, and
# the per-frame ``output/vert_<N>.bin``). A hard pin is an exact kinematic
# target on both backends, so every hard vertex must have turned by the
# authored angle of the side it lies on. The side is read off the vertex's
# rest X, which is independent of the pin block the decoder put it in;
# vertices within MARGIN of X = 0 are not assigned, since the full-pin bar's
# two groups meet there.
#
# Soft (pull > 0) pin vertices are not measured in the solver session: the
# emulated backend does not move them. A SOLID's unpinned Blender vertices are
# reconstructed from its tetrahedral surface, so at the Blender level only the
# pinned ones are asserted (subtest E). The median PC2 angle of each group is
# recorded in the details for a reader, not asserted.
#
# Direction convention: a spin rotates by the right-hand rule about its axis
# (``spin_step`` in crates/ppf-cts-core/src/datamodel/pin_apply.rs), and the
# encoder's Blender-to-solver axis swap (x, y, z) -> (x, z, -y) is a proper
# rotation that keeps X, so a positive angle about +X means the same in the
# solver output and in the PC2. The SHELL control verifies that convention
# and the measurement in the same run, so a SOLID failure cannot come from
# either.
#
# Subtests:
#   A. build_run_completes:
#         every bar builds and runs (solver not FAILED), and each has a
#         finite PC2 with at least frame_count - 1 samples.
#   B. shell_hard_pins_follow_own_group:
#         control. Every hard vertex of the SHELL bar turns by the authored
#         angle of its side, within ANGLE_TOL_DEG.
#   C. solid_partial_pin_hard_pins_follow_own_group:
#         the same check on the SOLID bar pinned at its caps.
#   D. solid_full_pin_hard_pins_follow_own_group:
#         the same check on the SOLID bar pinned by its two halves.
#   E. <bar>_blender_pins_follow_script, one per bar:
#         every pinned Blender vertex, as the bar's PC2 holds it, sits on its
#         scripted rotation within BLENDER_TOL_M. A SOLID's pinned Blender
#         vertices are placed where the solver evaluated their script
#         (display_pin_<N>.bin) rather than reconstructed from the
#         tetrahedral surface, so this holds for the SOLID bars as exactly as
#         for the SHELL.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# Hard pins are exact kinematic targets on the emulated backend and on the
# CUDA solver alike, and the session is read through the same
# ``remote_root`` lookup ``bl_fetch_frame_discovery`` uses on both.
BACKENDS = ("emulated", "real")


_FRAME_COUNT = 11
_FPS = 100
_SUBDIV = 3          # cube edge cuts; each cap holds (cuts + 2)^2 vertices
# Every bar has a 1 x 1 cross section and its own length. The lengths differ
# so no two bars share local geometry: the encoder deduplicates meshes by
# local geometry alone, and a SHELL instance of a SOLID's canonical mesh does
# not build. The spin angle does not depend on the bar length.
_SHELL_HALF_LENGTH = 1.5
_SOLID_PARTIAL_HALF_LENGTH = 2.0
_SOLID_FULL_HALF_LENGTH = 1.25
_PARTIAL_OMEGA_DEG = 360.0   # deg/s; 36 degrees by the last frame
_FULL_OMEGA_DEG = 30.0       # deg/s; 3 degrees by the last frame
# A hard vertex's angle differs from the authored one only by float32 output
# rounding. A vertex driven by the other side's spin is off by twice the
# authored angle.
_ANGLE_TOL_DEG = 0.05
# A pinned Blender vertex is placed where the solver evaluated its script
# (display_pin_<N>.bin), so it differs from the scripted rotation only by
# float32 rounding through the solver output and the PC2.
_BLENDER_TOL_M = 1e-5


_DRIVER_BODY = r"""
import os
import sys
import tomllib
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>
FPS = <<FPS>>
SUBDIV = <<SUBDIV>>
ANGLE_TOL_DEG = <<ANGLE_TOL_DEG>>
BLENDER_TOL_M = <<BLENDER_TOL_M>>

# pinRight alone is flipped, so the two sides must turn in opposite directions.
FLIP = {"pinLeft": False, "pinRight": True}
LAST_FRAME = FRAME_COUNT - 1

# (label, check name, group type, half length, location, pin layout, deg/s)
BARS = [
    ("shell", "B_shell_hard_pins_follow_own_group", "SHELL",
     <<SHELL_HALF_LENGTH>>, (0.0, 3.0, 0.0), "caps", <<PARTIAL_OMEGA_DEG>>),
    ("solid_partial", "C_solid_partial_pin_hard_pins_follow_own_group",
     "SOLID", <<SOLID_PARTIAL_HALF_LENGTH>>, (0.0, 0.0, 0.0), "caps",
     <<PARTIAL_OMEGA_DEG>>),
    ("solid_full", "D_solid_full_pin_hard_pins_follow_own_group",
     "SOLID", <<SOLID_FULL_HALF_LENGTH>>, (0.0, -3.0, 0.0), "halves",
     <<FULL_OMEGA_DEG>>),
]


def _make_bar(name, location, half_length, layout):
    # A subdivided unit cube stretched along X to [-half_length,
    # half_length]. "caps" puts the +X face in pinLeft and the -X face in
    # pinRight; "halves" puts every vertex with X >= 0 in pinLeft and the
    # rest in pinRight, so the two groups pin the whole mesh. The location
    # moves the bar off X only, so X = 0 separates the two sides in world
    # space as well as in the solver frame.
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=SUBDIV)
    bpy.ops.object.mode_set(mode="OBJECT")
    for v in obj.data.vertices:
        v.co.x *= 2.0 * half_length
    xs = [v.co.x for v in obj.data.vertices]
    xmax, xmin = max(xs), min(xs)
    if layout == "caps":
        members = {
            "pinLeft": [i for i, x in enumerate(xs) if abs(x - xmax) < 1e-4],
            "pinRight": [i for i, x in enumerate(xs) if abs(x - xmin) < 1e-4],
        }
    elif layout == "halves":
        members = {
            "pinLeft": [i for i, x in enumerate(xs) if x >= 0.0],
            "pinRight": [i for i, x in enumerate(xs) if x < 0.0],
        }
    else:
        raise ValueError(f"unknown pin layout {layout!r}")
    for vg_name, idx in members.items():
        obj.vertex_groups.new(name=vg_name).add(idx, 1.0, "REPLACE")
    return obj, members


def _author_spins(group, obj, omega_deg):
    for vg_name, flip in FLIP.items():
        group.create_pin(obj.name, vg_name).spin(
            axis=(1.0, 0.0, 0.0), angular_velocity=omega_deg, flip=flip,
            frame_start=1, frame_end=FRAME_COUNT, transition="LINEAR",
        )


def _angles_about_x_deg(p0, pn):
    # Signed rotation about +X of each point from p0 to pn, in the plane
    # normal to X about the point set's own centroid. A rotation about any
    # line parallel to X turns these offsets by the same angle, so the
    # measure does not depend on the spin center. Points near the centroid
    # are dropped: their angle is undefined at zero radius.
    u = p0[:, 1:] - p0[:, 1:].mean(axis=0)
    v = pn[:, 1:] - pn[:, 1:].mean(axis=0)
    radius = np.linalg.norm(u, axis=1)
    keep = radius > 0.25 * radius.max()
    cross = u[keep, 0] * v[keep, 1] - u[keep, 1] * v[keep, 0]
    dot = np.sum(u[keep] * v[keep], axis=1)
    return np.degrees(np.arctan2(cross, dot))


def _read_bin(path, dtype, rows, width):
    # A size that does not match the layout is an error, not a reshape:
    # reading a float64 bin as float32 (or the reverse) gives plausible
    # garbage.
    itemsize = np.dtype(dtype).itemsize
    size = os.path.getsize(path)
    if size != rows * width * itemsize:
        raise RuntimeError(
            f"{path}: {size} bytes, expected {rows} x {width} x {itemsize}")
    return np.fromfile(path, dtype=dtype).reshape(rows, width)


def _frame_time(session_dir, frame):
    path = os.path.join(session_dir, "output", "data", "frame_to_time.out")
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and int(parts[0]) == frame:
                return float(parts[1])
    raise RuntimeError(f"{path}: no time recorded for frame {frame}")


def _load_session(session_dir):
    with open(os.path.join(session_dir, "info.toml"), "rb") as f:
        info = tomllib.load(f)
    n_vert = int(info["count"]["vert"])
    x0 = _read_bin(os.path.join(session_dir, "output", "vert_0.bin"),
                   "<f4", n_vert, 3)
    xn = _read_bin(
        os.path.join(session_dir, "output", f"vert_{LAST_FRAME}.bin"),
        "<f4", n_vert, 3)
    return info, x0, xn, _frame_time(session_dir, LAST_FRAME)


def _hard_pins_follow_own_group(session_dir, session, obj, omega_deg,
                                margin):
    info, x0, xn, t = session
    obj_uuid = dh_uuid.get_object_uuid(obj)
    if not obj_uuid:
        raise RuntimeError(f"{obj.name} has no solver uuid after encoding")

    blocks = []
    hard = {vg_name: [] for vg_name in FLIP}
    for k in range(int(info["count"]["pin_block"])):
        block = info[f"pin-{k}"]
        owner, _, vg_name = block["pin_group_id"].partition(":")
        if owner != obj_uuid:
            continue
        blocks.append({"block": k, "pin_group": vg_name,
                       "pull": block["pull"], "n_verts": block["pin"]})
        if block["pull"] != 0.0:
            continue
        idx = _read_bin(
            os.path.join(session_dir, "bin", f"pin-ind-{k}.bin"),
            "<u8", int(block["pin"]), 1)[:, 0].astype(np.int64)
        for vi in idx:
            if x0[vi, 0] > margin:
                hard["pinLeft"].append(vi)
            elif x0[vi, 0] < -margin:
                hard["pinRight"].append(vi)

    # The spin runs from frame 1 (time 0) to frame FRAME_COUNT, and holds
    # its angle past its end time.
    swept = omega_deg * min(t, (FRAME_COUNT - 1) / FPS)
    ok = True
    report = {"time": t, "margin": margin, "pin_blocks": blocks}
    for vg_name, idx in hard.items():
        expected = -swept if FLIP[vg_name] else swept
        if len(idx) < 3:
            ok = False
            report[vg_name] = {"n_hard_verts": len(idx),
                               "reason": "fewer than 3 hard vertices on side"}
            continue
        angles = _angles_about_x_deg(x0[idx], xn[idx])
        worst = float(np.max(np.abs(angles - expected)))
        side_ok = worst <= ANGLE_TOL_DEG
        ok = ok and side_ok
        report[vg_name] = {
            "flip": FLIP[vg_name],
            "n_hard_verts": len(idx),
            "expected_deg": round(expected, 4),
            "measured_min_deg": round(float(angles.min()), 4),
            "measured_max_deg": round(float(angles.max()), 4),
            "worst_error_deg": round(worst, 4),
            "ok": side_ok,
        }
    report["angle_tol_deg"] = ANGLE_TOL_DEG
    return ok, report


def _rot_x(p, c, deg):
    a = np.radians(deg)
    ca, sa = np.cos(a), np.sin(a)
    q = p - c
    return np.column_stack([
        q[:, 0], ca * q[:, 1] - sa * q[:, 2], sa * q[:, 1] + ca * q[:, 2],
    ]) + c


def _blender_pins_follow_script(arr, members, omega_deg, t):
    # Every pinned Blender vertex of the bar, as its PC2 holds it, against its
    # scripted rotation about the X axis through its group's rest centroid.
    # PC2 sample 0 is the rest pose, and the bars carry no rotation or scale,
    # so the object-local frame of the PC2 is the frame the script runs in.
    last = min(int(arr.shape[0]) - 1, LAST_FRAME)
    swept = omega_deg * min(t, (FRAME_COUNT - 1) / FPS)
    ok = last >= 1
    report = {"last_sample": last, "tol_m": BLENDER_TOL_M}
    for vg_name, idx in members.items():
        idx = np.asarray(idx)
        p0 = arr[0, idx].astype(np.float64)
        deg = -swept if FLIP[vg_name] else swept
        target = _rot_x(p0, p0.mean(axis=0), deg)
        err = np.linalg.norm(arr[last, idx].astype(np.float64) - target, axis=1)
        side_ok = bool(err.max() <= BLENDER_TOL_M)
        ok = ok and side_ok
        report[vg_name] = {"n": int(idx.size),
                           "max_error_m": float(err.max()), "ok": side_ok}
    return ok, report


def _pc2_group_angles_deg(arr, members):
    last = min(int(arr.shape[0]) - 1, LAST_FRAME)
    return {
        vg_name: round(float(np.median(_angles_about_x_deg(
            arr[0, np.asarray(idx)], arr[last, np.asarray(idx)]))), 3)
        for vg_name, idx in members.items()
    }


try:
    dh = DriverHelpers(pkg, result)
    dh_uuid = __import__(pkg + ".core.uuid_registry",
                         fromlist=["get_object_uuid"])
    dh.log(f"addon={os.path.realpath(sys.modules[pkg].__file__)}")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bars = {}
    for label, check, kind, half_length, location, layout, omega in BARS:
        obj, members = _make_bar(label, location, half_length, layout)
        bars[label] = {"check": check, "kind": kind, "obj": obj,
                       "members": members, "omega": omega,
                       "margin": 0.5 * half_length}
        dh.log(f"{label} n_verts={len(obj.data.vertices)} "
               f"pinLeft={len(members['pinLeft'])} "
               f"pinRight={len(members['pinRight'])}")

    dh.save_blend(PROBE_DIR, "solid_spin_flip_per_pin.blend")
    root = dh.configure_state(
        project_name="solid_spin_flip_per_pin",
        frame_count=FRAME_COUNT,
        frame_rate=FPS,
        step_size=1.0 / FPS,
        gravity=(0.0, 0.0, 0.0),
    )
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = FRAME_COUNT

    for label, bar in bars.items():
        group = dh.api.solver.create_group(label, bar["kind"])
        group.add(bar["obj"].name)
        _author_spins(group, bar["obj"], bar["omega"])

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect(
        local_path=LOCAL_PATH,
        server_port=SERVER_PORT,
        project_name=root.state.project_name,
    )
    dh.log("connected")
    # fTetWild runs during build, so give it a generous window.
    dh.build_and_wait(data_bytes, param_bytes,
                      message="solid_spin_flip:build", timeout=240.0)
    dh.log("built")
    dh.run_and_wait(timeout=120.0)
    solver_state = dh.facade.engine.state.solver.name
    dh.log(f"ran solver={solver_state}")
    dh.force_frame_query(expected_frames=LAST_FRAME, timeout=30.0)
    dh.settle_idle(timeout=15.0)
    applied, total = dh.fetch_and_drain()
    dh.log(f"fetch.drained applied={applied}/{total}")
    session_dir = os.path.join(dh.facade.engine.state.remote_root, "session")

    for bar in bars.values():
        path = dh.find_pc2_for(bar["obj"])
        arr = dh.read_pc2(path) if path else None
        bar["path"] = path
        bar["arr"] = arr
        bar["samples"] = int(arr.shape[0]) if arr is not None else 0
        bar["finite"] = bool(arr is not None and np.all(np.isfinite(arr)))

    # ----- A: every bar builds, runs and produces a finite PC2 ----------
    dh.record(
        "A_build_run_completes",
        solver_state != "FAILED"
        and all(bar["samples"] >= FRAME_COUNT - 1 and bar["finite"]
                for bar in bars.values()),
        {
            "solver_state": solver_state,
            "error": dh.facade.engine.state.error,
            "session_dir": session_dir,
            "expected_min_samples": FRAME_COUNT - 1,
            **{label: {k: bar[k] for k in ("path", "samples", "finite")}
               for label, bar in bars.items()},
        },
    )

    # ----- B, C, D: hard pins follow the group of their side ------------
    session = _load_session(session_dir)
    for label, bar in bars.items():
        ok, report = _hard_pins_follow_own_group(
            session_dir, session, bar["obj"], bar["omega"], bar["margin"])
        if bar["arr"] is not None and bar["finite"]:
            report["pc2_group_angles_deg_not_asserted"] = (
                _pc2_group_angles_deg(bar["arr"], bar["members"]))
        dh.record(bar["check"], ok, report)

    # ----- E: every pinned Blender vertex follows its script -------------
    for label, bar in bars.items():
        check = f"E_{label}_blender_pins_follow_script"
        if bar["arr"] is None or not bar["finite"]:
            dh.record(check, False, {"reason": "no finite PC2"})
            continue
        ok, report = _blender_pins_follow_script(
            bar["arr"], bar["members"], bar["omega"], session[3])
        dh.record(check, ok, report)

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
        .replace("<<FRAME_COUNT>>", str(_FRAME_COUNT))
        .replace("<<FPS>>", str(_FPS))
        .replace("<<SUBDIV>>", str(_SUBDIV))
        .replace("<<SHELL_HALF_LENGTH>>", repr(_SHELL_HALF_LENGTH))
        .replace("<<SOLID_PARTIAL_HALF_LENGTH>>",
                 repr(_SOLID_PARTIAL_HALF_LENGTH))
        .replace("<<SOLID_FULL_HALF_LENGTH>>", repr(_SOLID_FULL_HALF_LENGTH))
        .replace("<<PARTIAL_OMEGA_DEG>>", repr(_PARTIAL_OMEGA_DEG))
        .replace("<<FULL_OMEGA_DEG>>", repr(_FULL_OMEGA_DEG))
        .replace("<<ANGLE_TOL_DEG>>", repr(_ANGLE_TOL_DEG))
        .replace("<<BLENDER_TOL_M>>", repr(_BLENDER_TOL_M))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 360.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
