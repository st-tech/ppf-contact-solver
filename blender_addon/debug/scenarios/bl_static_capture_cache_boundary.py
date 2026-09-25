# File: scenarios/bl_static_capture_cache_boundary.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Capture Deformation on a STATIC collider reads the stack CUT at the cache
# boundary, the same pose the encoder and the pin capture read.
#
# The collider is a grid with an Armature modifier followed by a Subdivision
# Surface. The solver builds a collider from the base mesh, so the capture
# must record the armature-deformed BASE CAGE and leave the Subdivision to run
# on top of the ContactSolverCache on display. Evaluating the whole stack gives
# the subdivided vertex count, which the capture refused with "vertex count
# changed", so such a collider could not be captured and therefore could not
# be transferred at all.
#
# Subtests:
#   A. capture_succeeds_on_the_base_cage: the shared capture job (the path the
#         Capture Deformation button runs) finishes without aborting and writes
#         one row per frame at the BASE vertex count.
#   B. capture_is_the_cut_pose: every row equals an independent evaluation with
#         the Subdivision hidden, mapped to solver world space, and the armature
#         really moves the cage over the range, or equality proves nothing.
#   C. display_stack_restored: after the capture the Subdivision is drawn again
#         and the evaluated mesh has more vertices than the base.
#   D. transfer_and_solve: the scene encodes, builds, runs and returns frames.
#   E. cache_sits_at_the_cut_and_replays_the_capture: the collider's
#         ContactSolverCache sits in front of the Subdivision, its PC2 is at the
#         base vertex count, and each sample, placed in world space by the
#         object's matrix at its own frame, lands on the captured row.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
BACKENDS = ("real",)

_FRAME_COUNT = 8
# How far the bone carries the cage over the range, in scene units. Large
# against _TOL so a capture of the undeformed cage fails B by far.
_TRAVEL = 0.4
# A capture row is a copy of an evaluation, not a solve: float32 round-off.
_TOL = 1e-5
# A PC2 sample of a STATIC collider is the solver's output for it, which
# tracks the capture to within the soft residual the other STATIC capture
# scenarios allow.
_REPLAY_BOUND = 0.02


_DRIVER_BODY = r"""
import traceback
import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>
TRAVEL = <<TRAVEL>>
TOL = <<TOL>>
REPLAY_BOUND = <<REPLAY_BOUND>>


def _drive_to_completion(advance):
    # Pump the capture job's tick wrapper until it reports done or aborted,
    # which is what the modal operator does on each timer tick.
    guard = 0
    aborted, err = False, ""
    while True:
        more, aborted, err = advance(bpy.context)
        guard += 1
        if aborted or not more or guard > 100000:
            break
    return aborted, err


def _cut_pose_world(obj, subsurf, frame):
    # The armature-deformed base cage at *frame*, in solver world space,
    # evaluated here with the Subdivision hidden by hand.
    transform_mod = __import__(pkg + ".core.transform", fromlist=["zup_to_yup"])
    scene = bpy.context.scene
    scene.frame_set(int(frame))
    subsurf.show_viewport = False
    try:
        dg = bpy.context.evaluated_depsgraph_get()
        eo = obj.evaluated_get(dg)
        em = eo.to_mesh()
        try:
            n = len(em.vertices)
            co = np.empty((n, 3), dtype=np.float64)
            em.vertices.foreach_get("co", co.ravel())
            mw = np.array(eo.matrix_world, dtype=np.float64).reshape(4, 4)
        finally:
            eo.to_mesh_clear()
    finally:
        subsurf.show_viewport = True
    z2y = np.array(transform_mod.zup_to_yup(), dtype=np.float64).reshape(4, 4)
    h = np.concatenate([co, np.ones((n, 1))], axis=1)
    return (h @ (z2y @ mw).T)[:, :3]


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    sd = __import__(pkg + ".ui.dynamics.static_deform_ops",
                    fromlist=["start_capture_for_objects", "advance_capture",
                              "finalize_capture", "cleanup_capture"])
    pc2 = __import__(pkg + ".core.pc2",
                     fromlist=["get_static_deform_cache", "MODIFIER_NAME"])
    transform_mod = __import__(pkg + ".core.transform",
                               fromlist=["world_matrix", "zup_to_yup"])

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = FRAME_COUNT

    # One bone; the whole cage is weighted to it.
    bpy.ops.object.armature_add(enter_editmode=True, location=(0.0, 0.0, 0.0))
    arm = bpy.context.active_object
    arm.name = "CutRig"
    bones = arm.data.edit_bones
    while bones:
        bones.remove(bones[0])
    b = bones.new("Bone")
    b.head = (0.0, 0.0, 0.0)
    b.tail = (0.0, 0.0, 1.0)
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
    floor = bpy.context.active_object
    floor.name = "CutCollider"
    bpy.context.view_layer.objects.active = floor
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=2)
    bpy.ops.object.mode_set(mode="OBJECT")
    n_base = len(floor.data.vertices)
    floor.vertex_groups.new(name="Bone").add(
        list(range(n_base)), 1.0, "REPLACE")
    arm_mod = floor.modifiers.new(name="ArmatureMod", type="ARMATURE")
    arm_mod.object = arm
    subsurf = floor.modifiers.new(name="Subdivision", type="SUBSURF")
    subsurf.levels = 1

    # The bone slides the cage sideways over the whole range.
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="POSE")
    pb = arm.pose.bones["Bone"]
    pb.location = (0.0, 0.0, 0.0)
    pb.keyframe_insert(data_path="location", frame=1)
    pb.location = (TRAVEL, 0.0, 0.0)
    pb.keyframe_insert(data_path="location", frame=FRAME_COUNT)
    bpy.ops.object.mode_set(mode="OBJECT")

    # Something dynamic to solve against the collider.
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.3))
    cloth = bpy.context.active_object
    cloth.name = "CutCloth"
    bpy.context.view_layer.objects.active = cloth
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=3)
    bpy.ops.object.mode_set(mode="OBJECT")

    dh.save_blend(PROBE_DIR, "static_capture_cache_boundary.blend")
    root = dh.configure_state(project_name="static_capture_cache_boundary",
                              frame_count=FRAME_COUNT,
                              frame_rate=24,
                              step_size=1.0 / 24.0)
    g_static = dh.api.solver.create_group("CutStatic", "STATIC")
    g_static.add(floor.name)
    g_cloth = dh.api.solver.create_group("CutShell", "SHELL")
    g_cloth.add(cloth.name)

    # ---- A: the capture job finishes on the base cage ------------------
    ctx = bpy.context
    ok, err = sd.start_capture_for_objects(ctx, [floor])
    aborted, abort_err = False, ""
    if ok:
        aborted, abort_err = _drive_to_completion(sd.advance_capture)
    n_objs, n_frames = sd.finalize_capture(ctx) if ok else (0, 0)
    sd.cleanup_capture(ctx)
    cache = pc2.get_static_deform_cache(floor)
    shape = tuple(cache.shape) if cache is not None else None
    dh.record(
        "A_capture_succeeds_on_the_base_cage",
        ok and not aborted and n_objs == 1
        and shape == (FRAME_COUNT, n_base, 3),
        {"ok": ok, "err": err, "aborted": aborted, "abort_err": abort_err,
         "objects": n_objs, "frames": n_frames, "cache_shape": shape,
         "base_verts": n_base, "expected_frames": FRAME_COUNT},
    )
    if cache is None:
        raise RuntimeError("no capture to check; A says why")
    captured = np.asarray(cache, dtype=np.float64)

    # ---- B: every row is the cut pose -----------------------------------
    saved = scene.frame_current
    try:
        expected = np.stack([_cut_pose_world(floor, subsurf, f)
                             for f in range(1, FRAME_COUNT + 1)])
    finally:
        scene.frame_set(saved)
    worst = float(np.max(np.abs(expected - captured)))
    moved = float(np.max(np.abs(captured[-1] - captured[0])))
    dh.record(
        "B_capture_is_the_cut_pose",
        expected.shape == captured.shape and worst < TOL
        and moved > 0.5 * TRAVEL,
        {"worst_err": worst, "tol": TOL, "cage_travel": moved,
         "required_travel": 0.5 * TRAVEL},
    )

    # ---- C: the display stack is back ------------------------------------
    dg = bpy.context.evaluated_depsgraph_get()
    eo = floor.evaluated_get(dg)
    em = eo.to_mesh()
    n_drawn = len(em.vertices)
    eo.to_mesh_clear()
    dh.record(
        "C_display_stack_restored",
        subsurf.show_viewport and arm_mod.show_viewport and n_drawn > n_base,
        {"subsurf_shown": subsurf.show_viewport,
         "armature_shown": arm_mod.show_viewport,
         "drawn_verts": n_drawn, "base_verts": n_base},
    )

    # ---- D: transfer and solve -------------------------------------------
    data_bytes, param_bytes = dh.encode_payload()
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root.state.project_name)
    dh.build_and_wait(data_bytes, param_bytes,
                      message="static_capture_cache_boundary:build")
    dh.run_and_wait(timeout=180.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=30.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain()
    path = dh.find_pc2_for(floor)
    arr = dh.read_pc2(path) if path else None
    n_samples = int(arr.shape[0]) if arr is not None else 0
    dh.record(
        "D_transfer_and_solve",
        n_samples >= FRAME_COUNT - 1,
        {"pc2": path, "samples": n_samples, "expected_at_least": FRAME_COUNT - 1},
    )

    # ---- E: the cache sits at the cut and replays the capture ------------
    names = [m.name for m in floor.modifiers]
    cache_i = names.index(pc2.MODIFIER_NAME) if pc2.MODIFIER_NAME in names else -1
    sub_i = names.index("Subdivision")
    arm_i = names.index("ArmatureMod")
    errs = []
    if arr is not None and arr.shape[1] == n_base:
        z2y = np.array(transform_mod.zup_to_yup(), dtype=np.float64).reshape(4, 4)
        saved = scene.frame_current
        try:
            for i in range(min(n_samples, FRAME_COUNT)):
                scene.frame_set(i + 1)
                wm = np.array(transform_mod.world_matrix(floor),
                              dtype=np.float64).reshape(4, 4)
                h = np.concatenate([arr[i].astype(np.float64),
                                    np.ones((n_base, 1))], axis=1)
                world = (h @ wm.T)[:, :3]
                errs.append(float(np.max(np.abs(world - captured[i]))))
        finally:
            scene.frame_set(saved)
    worst_replay = max(errs) if errs else -1.0
    dh.record(
        "E_cache_sits_at_the_cut_and_replays_the_capture",
        arm_i < cache_i < sub_i
        and arr is not None and arr.shape[1] == n_base
        and len(errs) > 0 and worst_replay < REPLAY_BOUND,
        {"stack": names, "pc2_verts": None if arr is None else int(arr.shape[1]),
         "base_verts": n_base, "checked": len(errs),
         "worst_err": worst_replay, "bound": REPLAY_BOUND},
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
        .replace("<<TRAVEL>>", repr(_TRAVEL))
        .replace("<<TOL>>", repr(_TOL))
        .replace("<<REPLAY_BOUND>>", repr(_REPLAY_BOUND))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
