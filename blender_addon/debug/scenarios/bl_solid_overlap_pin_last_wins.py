# File: scenarios/bl_solid_overlap_pin_last_wins.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Overlapping SOLID pins, per-vertex last-wins, captured end to end.
#
# A SOLID cube is driven by an Armature: a soft PULL pin ("pin", every
# vertex) carries the captured per-frame bone deformation, and a hard
# static pin ("pin-root", the bottom face) is added LAST. For the shared
# bottom verts the last pin wins, so they must be HARD-fixed (held at rest)
# while the rest of the body is pulled toward the captured bone pose.
#
# The capture keeps the fully-pinned SOLID DYNAMIC (without it, an
# all-pinned SOLID collapses to a static collider with no dynamic verts).
# The full-pin union decodes to a single harmonic holder; the split helper
# must (1) extract the hard bottom SURFACE verts into a FixPair holder so
# they are rigid (an interior fix pin is a CG nan), and (2) re-point the
# surviving pull holder's cfg lookup at a pull vert so pull + the captured
# move still attach (its first stored blender vert is in fact a bottom/root
# vert). The decoded scene is therefore two pin blocks: a PULL holder that
# carries the captured move ops, and a bottom FixPair.
#
# Subtests:
#   A. build_run_completes_no_nan:
#         the captured overlapping pull + hard pins build and run to
#         completion (solver not FAILED, finite PC2, >= frame_count - 1
#         samples, the cube stays DYNAMIC). Catches the static collapse,
#         the harmonic bytemuck panic, and the interior-fix-pin nan.
#   B. last_wins_split_structure:
#         the BUILT solver scene decodes to exactly the last-wins
#         structure: two pin blocks, one PULL holder (pull > 0) carrying
#         the captured move ops, and one bottom FixPair (pull == 0).
#
# Why B checks the built scene rather than runtime motion: the emulated
# (CPU) solver the rig runs does not execute the soft-pull *follow* (a
# move_by on a PULL holder produces no displacement; a fully-pinned SOLID
# then stays bit-for-bit at rest). The runtime follow is exercised on the
# live CUDA solver. What this scenario proves end to end is that the full
# encode -> build -> decode pipeline turns the overlapping captured pins
# into the correct solver scene, which is exactly what the fix changes.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True


_FRAME_COUNT = 10
_SUBDIV = 1
_BEND_ANGLE = 0.6        # rad; rotate the cube about X so the capture moves
_PULL_STRENGTH = 1000.0  # soft follow of the captured bone target


_DRIVER_BODY = r"""
import glob
import os
import re
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>
SUBDIV = <<SUBDIV>>
BEND_ANGLE = <<BEND_ANGLE>>
PULL_STRENGTH = <<PULL_STRENGTH>>


def _sample_pin_world_solver(obj, pin_indices, frame_start, frame_end):
    # Depsgraph-evaluate obj at each frame and return (n_frames, n_pin, 3)
    # float32 in solver world space, matching pin_capture_ops.
    transform_mod = __import__(pkg + ".core.transform", fromlist=["zup_to_yup"])
    zup_to_yup = transform_mod.zup_to_yup
    scene = bpy.context.scene
    saved = scene.frame_current
    n_frames = frame_end - frame_start + 1
    n_pin = len(pin_indices)
    out = np.empty((n_frames, n_pin, 3), dtype=np.float32)
    z2y = np.array(zup_to_yup(), dtype=np.float64).reshape(4, 4)
    pin_arr = np.asarray(pin_indices, dtype=np.int64)
    try:
        for i, f in enumerate(range(frame_start, frame_end + 1)):
            scene.frame_set(int(f))
            dg = bpy.context.evaluated_depsgraph_get()
            eo = obj.evaluated_get(dg)
            em = eo.to_mesh()
            try:
                n_total = len(em.vertices)
                co = np.empty((n_total, 3), dtype=np.float64)
                em.vertices.foreach_get("co", co.ravel())
                co_sub = co[pin_arr]
                mw = np.array(eo.matrix_world, dtype=np.float64).reshape(4, 4)
                m = z2y @ mw
                homog = np.concatenate(
                    [co_sub, np.ones((n_pin, 1), dtype=np.float64)], axis=1,
                )
                out[i] = (homog @ m.T)[:, :3].astype(np.float32, copy=False)
            finally:
                eo.to_mesh_clear()
    finally:
        scene.frame_set(saved)
    return out


def _parse_pin_blocks(pc2_path):
    # Locate the built scene's info.toml from the PC2 output dir and parse
    # the pin blocks. Returns (pin_block_count, [(pull, op_count), ...]).
    if not pc2_path:
        return -1, [], ""
    worker = os.path.dirname(os.path.dirname(os.path.dirname(pc2_path)))
    hits = glob.glob(os.path.join(worker, "project", "*", "session", "info.toml"))
    if not hits:
        return -1, [], ""
    txt = open(hits[0]).read()
    m = re.search(r"pin_block\s*=\s*(\d+)", txt)
    pin_block = int(m.group(1)) if m else -1
    blocks = []
    for body in re.findall(r"\[pin-\d+\]\n(.*?)(?=\n\[|\Z)", txt, re.S):
        pm = re.search(r"pull\s*=\s*([0-9.eE+-]+)", body)
        if pm is None:
            continue  # an op sub-block, not a holder header
        om = re.search(r"operation_count\s*=\s*(\d+)", body)
        blocks.append((float(pm.group(1)), int(om.group(1)) if om else 0))
    return pin_block, blocks, hits[0]


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Single-bone armature at the cube center; rotating the bone swings the
    # whole cube about the X axis through the origin.
    bpy.ops.object.armature_add(location=(0.0, 0.0, 0.0))
    arm = bpy.context.active_object
    arm.name = "CapArm"
    bone_name = arm.data.bones[0].name

    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 0.0))
    cube = bpy.context.active_object
    cube.name = "SolidBox"
    cube.parent = arm
    bpy.context.view_layer.objects.active = cube
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=SUBDIV)
    bpy.ops.object.mode_set(mode="OBJECT")

    n_verts = len(cube.data.vertices)
    zmin = min(v.co.z for v in cube.data.vertices)
    bottom = [v.index for v in cube.data.vertices if abs(v.co.z - zmin) < 1e-4]

    # Skin every vert to the bone, plus an Armature modifier, so the
    # depsgraph deforms the whole cube when the bone rotates.
    cube.vertex_groups.new(name=bone_name).add(
        list(range(n_verts)), 1.0, "REPLACE")
    amod = cube.modifiers.new(name="ArmatureMod", type="ARMATURE")
    amod.object = arm

    # Pins: "pin" (all verts, soft pull, captured) + "pin-root" (bottom,
    # hard, added LAST so it wins the shared bottom verts).
    cube.vertex_groups.new(name="pin").add(list(range(n_verts)), 1.0, "REPLACE")
    cube.vertex_groups.new(name="pin-root").add(bottom, 1.0, "REPLACE")
    dh.log(f"cube n_verts={n_verts} bottom={len(bottom)}")

    # Animate the bone rotation about X over the timeline.
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    pbone = arm.pose.bones[bone_name]
    pbone.rotation_mode = "XYZ"
    pbone.rotation_euler = (0.0, 0.0, 0.0)
    pbone.keyframe_insert(data_path="rotation_euler", frame=1)
    pbone.rotation_euler = (BEND_ANGLE, 0.0, 0.0)
    pbone.keyframe_insert(data_path="rotation_euler", frame=FRAME_COUNT)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = cube

    dh.save_blend(PROBE_DIR, "solid_overlap_pin.blend")
    root = dh.configure_state(
        project_name="solid_overlap_pin_last_wins",
        frame_count=FRAME_COUNT,
        frame_rate=100,
        step_size=0.01,
        gravity=(0.0, 0.0, -9.8),
    )
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = FRAME_COUNT

    solid = dh.api.solver.create_group("Solid", "SOLID")
    solid.add(cube.name)
    solid.create_pin(cube.name, "pin")        # index 0: pull (captured)
    solid.create_pin(cube.name, "pin-root")   # index 1: hard, LAST -> wins

    addon_root = dh.groups.get_addon_data(bpy.context.scene)
    group = addon_root.object_group_0
    group.pin_vertex_groups[0].use_pull = True
    group.pin_vertex_groups[0].pull_strength = PULL_STRENGTH
    group.pin_vertex_groups[1].use_pull = False   # hard (default)

    # Capture the bone deformation onto "pin" (headless: bypass the modal
    # operator, mirror its finalize). pin_anim is keyed by sorted vg verts.
    pc2 = __import__(pkg + ".core.pc2", fromlist=["write_pin_anim_pc2"])
    pin_ops = __import__(pkg + ".ui.dynamics.pin_ops",
                         fromlist=["_ensure_embedded_move_op"])
    pin_indices = list(range(n_verts))   # "pin" covers every vert
    captured = _sample_pin_world_solver(cube, pin_indices, 1, FRAME_COUNT)
    pc2.write_pin_anim_pc2(cube, "pin", captured)
    group.pin_vertex_groups[0].has_captured_anim = True
    pin_ops._ensure_embedded_move_op(group.pin_vertex_groups[0])
    cap_disp = float(np.max(np.linalg.norm(captured[-1] - captured[0], axis=1)))
    dh.log(f"pins captured: body swings ~{cap_disp:.2f}")

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes,
                      message="overlap_pin:build", timeout=240.0)
    dh.log("built")
    dh.run_and_wait(timeout=120.0)
    solver_state = dh.facade.engine.state.solver.name
    dh.log(f"ran solver={solver_state}")
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=30.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()

    pc2_path = dh.find_pc2_for(cube)
    arr = dh.read_pc2(pc2_path) if pc2_path else None
    samples = int(arr.shape[0]) if arr is not None else 0
    finite = bool(arr is not None and np.all(np.isfinite(arr)))

    dh.record(
        "A_build_run_completes_no_nan",
        solver_state != "FAILED" and arr is not None
        and samples >= FRAME_COUNT - 1 and finite,
        {"solver_state": solver_state, "samples": samples,
         "expected_min_samples": FRAME_COUNT - 1, "all_finite": finite,
         "error": dh.facade.engine.state.error},
    )

    # ----- B: decoded last-wins structure from the built scene ---------
    pin_block, blocks, info_path = _parse_pin_blocks(pc2_path)
    pull_holders = [(p, o) for (p, o) in blocks if p > 0.0]
    fix_holders = [(p, o) for (p, o) in blocks if p == 0.0]
    pull_carries_move = any(o > 0 for (_p, o) in pull_holders)
    dh.record(
        "B_last_wins_split_structure",
        pin_block == 2 and len(pull_holders) == 1 and len(fix_holders) == 1
        and pull_carries_move,
        {"pin_block": pin_block,
         "pull_holders": pull_holders, "fix_holders": fix_holders,
         "pull_holder_carries_captured_move": pull_carries_move,
         "info_toml_found": bool(info_path),
         "note": "pull holder (captured move) + bottom FixPair; emulated "
                 "solver does not run soft-pull follow, so the decoded "
                 "structure is verified instead of runtime motion"},
    )

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
        .replace("<<SUBDIV>>", str(_SUBDIV))
        .replace("<<BEND_ANGLE>>", repr(_BEND_ANGLE))
        .replace("<<PULL_STRENGTH>>", repr(_PULL_STRENGTH))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 360.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
