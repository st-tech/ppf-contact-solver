# File: scenarios/bl_time_scale_kinematic_invariance.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Time Scale end-to-end invariance. Time Scale stretches every schedule the
# solver consumes (pin ops, transform keyframes, captured-deformation times)
# by 1/time_scale AND slows the output cadence by the same factor, so solver
# frame k must land on the IDENTICAL pose regardless of time_scale: the
# kinematic result per Blender frame is invariant. Any channel that misses
# the scaling desyncs from the others and shifts its per-frame poses, so
# this one comparison validates the whole chain through the real pipeline
# (encode -> build -> run -> fetch -> PC2).
#
# The scene carries all three kinematic channels and is run twice
# (time_scale 1.0 then 0.5); the fetched PC2 arrays must match frame by
# frame:
#   * a fully pinned SHELL driven by a move_by pin op
#   * a fully pinned SHELL spun by a rate-parameterized SPIN op
#   * a STATIC cube moving rigidly via object location fcurves
#   * a STATIC cube driven by a captured (synthesized) deformation cache
#
# Gravity is zero and every vertex is prescribed, so the emulated backend's
# kinematic-constraint replay is exact and the invariance is tight.
#
# Subtests:
#   A. pin_motion_invariant
#   B. static_rigid_motion_invariant
#   C. static_captured_deform_invariant
#   D. sample_counts_match
#   E. spin_rate_motion_invariant (SPIN rate scaled by time_scale at encode)

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import os
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>

FRAME_COUNT = 6
FPS = 100
TOL = 5e-4


def _solver_world_rows(obj, n_rows, dy_per_row):
    # Synthesized capture rows in solver world space (zup_to_yup @
    # matrix_world @ co), row k translated k * dy_per_row along solver Y.
    transform = __import__(pkg + ".core.transform", fromlist=["zup_to_yup"])
    z2y = np.array(transform.zup_to_yup(), dtype=np.float64).reshape(4, 4)
    mw = np.array(obj.matrix_world, dtype=np.float64).reshape(4, 4)
    m = z2y @ mw
    n = len(obj.data.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", co)
    co = co.reshape(n, 3)
    homog = np.concatenate([co, np.ones((n, 1))], axis=1)
    base = (homog @ m.T)[:, :3]
    rows = np.empty((n_rows, n, 3), dtype=np.float32)
    for k in range(n_rows):
        rows[k] = base
        rows[k, :, 1] += dy_per_row * k
    return rows


def _fetch_arrays(dh, objs):
    dh.fetch_and_drain()
    out = {}
    for o in objs:
        p = dh.find_pc2_for(o)
        out[o.name] = (
            dh.read_pc2(p) if p and os.path.isfile(p) else None
        )
    return out


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    root = dh.configure_state(
        project_name="time_scale_invariance",
        frame_count=FRAME_COUNT,
        frame_rate=FPS,
        gravity=(0.0, 0.0, 0.0),
    )
    state = root.state

    # Fully pinned SHELL with a move_by op.
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "PinPlane"
    vg = plane.vertex_groups.new(name="AllPin")
    vg.add(list(range(len(plane.data.vertices))), 1.0, "REPLACE")
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.3, 0.0, 0.2), frame_start=1, frame_end=5)

    # Fully pinned SHELL spun by a rate-parameterized SPIN op. With the
    # rate scaled by time_scale at encode, rotation per FRAME is invariant,
    # so this object must land on identical per-frame poses too.
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(-3.0, 0.0, 0.0))
    spin_plane = bpy.context.active_object
    spin_plane.name = "SpinPlane"
    vg_s = spin_plane.vertex_groups.new(name="AllPin")
    vg_s.add(list(range(len(spin_plane.data.vertices))), 1.0, "REPLACE")
    cloth_s = dh.api.solver.create_group("ClothSpin", "SHELL")
    cloth_s.add(spin_plane.name)
    pin_s = cloth_s.create_pin(spin_plane.name, "AllPin")
    pin_s.spin(axis=(0.0, 0.0, 1.0), angular_velocity=120.0,
               center_mode="CENTROID", frame_start=1, frame_end=5)

    # STATIC rigid motion via location fcurves.
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(3.0, 0.0, 1.0))
    cube_rigid = bpy.context.active_object
    cube_rigid.name = "RigidCube"
    cube_rigid.keyframe_insert(data_path="location", frame=1)
    cube_rigid.location = (3.0, 0.0, 1.5)
    cube_rigid.keyframe_insert(data_path="location", frame=5)
    g_rigid = dh.api.solver.create_group("StatRigid", "STATIC")
    g_rigid.add(cube_rigid.name)

    # STATIC captured deformation (synthesized dense cache).
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(6.0, 0.0, 1.0))
    cube_def = bpy.context.active_object
    cube_def.name = "DeformCube"
    pc2 = __import__(pkg + ".core.pc2", fromlist=["write_static_deform_pc2"])
    pc2.write_static_deform_pc2(
        cube_def, _solver_world_rows(cube_def, FRAME_COUNT, 0.05))
    g_def = dh.api.solver.create_group("StatDeform", "STATIC")
    g_def.add(cube_def.name)

    bpy.context.view_layer.update()
    objs = [plane, spin_plane, cube_rigid, cube_def]
    dh.save_blend(PROBE_DIR, "time_scale_invariance.blend")

    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    animation_mod = __import__(pkg + ".core.animation",
                               fromlist=["clear_animation_data"])

    def _run_once(tag):
        data_b, param_b = dh.encode_payload()
        dh.build_and_wait(data_b, param_b, message="ts:" + tag, timeout=120.0)
        dh.run_and_wait(timeout=120.0)
        dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=15.0)
        dh.settle_idle(timeout=15.0)
        arrays = _fetch_arrays(dh, objs)
        dh.log("run_%s solver=%s" % (tag, dh.facade.engine.state.solver.name))
        return arrays

    state.time_scale = 1.0
    arrays_1 = _run_once("ts1")
    animation_mod.clear_animation_data(bpy.context)
    state.time_scale = 0.5
    arrays_05 = _run_once("ts05")
    state.time_scale = 1.0

    # Compare per object, from the first simulated sample onward (sample 0
    # is the Blender rest geometry written before any solve).
    def _cmp(name):
        a, b = arrays_1.get(name), arrays_05.get(name)
        if a is None or b is None or a.shape != b.shape or a.shape[0] < 2:
            return False, None, (None if a is None else a.shape,
                                 None if b is None else b.shape)
        err = float(np.max(np.abs(
            np.asarray(a[1:], dtype=np.float64)
            - np.asarray(b[1:], dtype=np.float64))))
        return err < TOL, err, (a.shape, b.shape)

    ok_a, err_a, shapes_a = _cmp("PinPlane")
    dh.record("A_pin_motion_invariant", ok_a,
              {"max_err": err_a, "tol": TOL, "shapes": shapes_a})

    ok_b, err_b, shapes_b = _cmp("RigidCube")
    dh.record("B_static_rigid_motion_invariant", ok_b,
              {"max_err": err_b, "tol": TOL, "shapes": shapes_b})

    ok_c, err_c, shapes_c = _cmp("DeformCube")
    dh.record("C_static_captured_deform_invariant", ok_c,
              {"max_err": err_c, "tol": TOL, "shapes": shapes_c})

    ok_e, err_e, shapes_e = _cmp("SpinPlane")
    dh.record("E_spin_rate_motion_invariant", ok_e,
              {"max_err": err_e, "tol": TOL, "shapes": shapes_e})

    counts_ok = all(
        arrays_1.get(o.name) is not None
        and arrays_05.get(o.name) is not None
        and arrays_1[o.name].shape == arrays_05[o.name].shape
        and arrays_1[o.name].shape[0] >= 2
        for o in objs
    )
    dh.record("D_sample_counts_match", counts_ok, {
        o.name: (
            None if arrays_1.get(o.name) is None else arrays_1[o.name].shape,
            None if arrays_05.get(o.name) is None else arrays_05[o.name].shape,
        ) for o in objs
    })

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    repo_root = REPO_ROOT_POSIX
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", repo_root)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
