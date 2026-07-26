# File: scenarios/bl_pinned_anim_transform_display.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A dynamics object with an ANIMATED OBJECT TRANSFORM must display the
# solver's output at the pose of each frame's own matrix_world.
#
# The ContactSolverCache (MESH_CACHE) modifier reads object-LOCAL positions
# and Blender re-applies the object's animated transform on top, so every
# PC2 frame written by the apply loop has to be divided by the matrix_world
# of ITS OWN frame. The per-frame matrix path existed but was gated on
# ``object_type == "STATIC"``: a SHELL / SOLID / ROD whose matrix_world
# varies across frames (fcurves, parent chain, drivers, constraints) fell
# back to ONE inverse matrix, snapshotted at whatever frame the playhead
# happened to sit on when each apply tick started. During a fetch the apply
# loop parks the playhead on the frame it just applied, so every frame was
# localized with the PREVIOUS frame's matrix: the displayed mesh trailed
# the solver's output by exactly one frame of object travel.
#
# The shipped case: a fully pinned SOLID collider parented to an animated
# armature (root motion in the object transform). The cloth hugged the
# solver's collider, but the DISPLAYED collider was one frame of root
# motion away, so the cloth appeared to penetrate it by up to the
# per-frame root travel (~55 mm at the scene's fastest frames) while the
# solver state was penetration-free.
#
# PC2 index 0 has its own variant of the same defect: fetch never
# downloads solver frame 0 (the initial state), so index 0 always comes
# from the leading gap-fill, whose fallback pose used to be a single
# deform-evaluated snapshot taken at whatever frame the scene was parked
# on. The same collider then showed the parked frame's pose on Blender
# frame 1 while every later frame was correct. The gap-fill now
# re-evaluates the pose at each gap frame's own frame, so this scenario
# keys a shape key on top of the moving transform: sample 0 is only
# correct if the gap-fill evaluated the mesh AT frame 1.
#
# Subtests:
#   A. parked_away_from_frame_one: the scene is parked away from frame 1
#         when the frames arrive, so the snapshot matrix and snapshot pose
#         are both wrong unless the apply loop re-reads them per frame.
#   B. display_matches_own_frame_matrix: every PC2 sample, mapped to world
#         through the matrix_world of ITS OWN frame, lands on the captured
#         kinematic pose for that frame. Sample 0 covers the gap-fill
#         pose; samples 1+ cover the apply loop's matrix. This is the
#         fixes' shared invariant.
#   C. stale_matrix_would_miss: mapping the same samples through the
#         PREVIOUS frame's matrix misses by roughly the per-frame travel.
#         Proves the object transform genuinely animates at bug-visible
#         amplitude, so B is not vacuously green.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
BACKENDS = ("emulated", "real")

_FRAME_COUNT = 10
# Object-level travel across the timeline. ~0.1 units/frame: the bug's
# one-frame lag displaces the display by that much, three orders above TOL.
_TRAVEL_Y = -0.9
# Shape-key lift across the timeline. Makes the LOCAL pose frame-dependent
# so the gap-filled sample 0 is wrong by the pose delta between the parked
# frame and frame 1 unless the gap-fill evaluates at frame 1 itself.
_LIFT_Z = 0.5
# Parked well away from frame 1 when the frames arrive.
_PARK_FRAME = 5
# The pins are exact kinematic targets and frame_rate * step_size == 1 puts
# every output frame ON a substep, so the display should match the captured
# pose to fp32 round-off.
_TOL = 1e-4


_DRIVER_BODY = r"""
import traceback
import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>
TRAVEL_Y = <<TRAVEL_Y>>
LIFT_Z = <<LIFT_Z>>
PARK_FRAME = <<PARK_FRAME>>
TOL = <<TOL>>


# Depsgraph-evaluate obj per frame -> (n_frames, n_verts, 3) in solver world
# space: the kinematic ground truth the pins prescribe and the solver tracks.
def capture_world_solver_frames(obj, frame_start, frame_end):
    transform_mod = __import__(pkg + ".core.transform", fromlist=["zup_to_yup"])
    scene = bpy.context.scene
    saved = scene.frame_current
    n_frames = frame_end - frame_start + 1
    n_verts = len(obj.data.vertices)
    out = np.empty((n_frames, n_verts, 3), dtype=np.float32)
    z2y = np.array(transform_mod.zup_to_yup(), dtype=np.float64).reshape(4, 4)
    try:
        for i, f in enumerate(range(frame_start, frame_end + 1)):
            scene.frame_set(int(f))
            dg = bpy.context.evaluated_depsgraph_get()
            eo = obj.evaluated_get(dg)
            em = eo.to_mesh()
            try:
                co = np.empty((n_verts, 3), dtype=np.float64)
                em.vertices.foreach_get("co", co.ravel())
                mw = np.array(eo.matrix_world, dtype=np.float64).reshape(4, 4)
                m = z2y @ mw
                h = np.concatenate([co, np.ones((n_verts, 1))], axis=1)
                out[i] = (h @ m.T)[:, :3].astype(np.float32, copy=False)
            finally:
                eo.to_mesh_clear()
    finally:
        scene.frame_set(saved)
    return out


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    plane = dh.reset_scene_to_pinned_plane(name="MovingCloth",
                                           pin_group="AllPin")
    n_verts = len(plane.data.vertices)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = FRAME_COUNT

    # An object-level location fcurve makes matrix_world vary per frame.
    # This is the animation source the display path must divide out frame
    # by frame; the mesh itself never deforms locally.
    plane.location = (0.0, 0.0, 0.0)
    plane.keyframe_insert(data_path="location", frame=1)
    plane.location = (0.0, TRAVEL_Y, 0.0)
    plane.keyframe_insert(data_path="location", frame=FRAME_COUNT)
    # LINEAR so every frame step travels the same distance; C's bound is
    # then exact instead of depending on bezier easing at the endpoints.
    utils_mod = __import__(pkg + ".core.utils", fromlist=["_get_fcurves"])
    for fc in utils_mod._get_fcurves(plane.animation_data.action):
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"

    # A keyed shape key makes the object-LOCAL pose frame-dependent, so
    # the gap-filled PC2 sample 0 (fetch never downloads solver frame 0)
    # is only right if the gap-fill evaluates the mesh AT frame 1 instead
    # of writing a rest-cage or parked-frame snapshot. The lift starts at
    # FULL value on frame 1 and decays, so frame 1's evaluated pose
    # differs from both the rest cage and the parked frame's pose by a
    # bug-visible amount.
    plane.shape_key_add(name="Basis", from_mix=False)
    lift = plane.shape_key_add(name="Lift", from_mix=False)
    for v in lift.data:
        v.co.z += LIFT_Z
    lift.value = 1.0
    lift.keyframe_insert(data_path="value", frame=1)
    lift.value = 0.0
    lift.keyframe_insert(data_path="value", frame=FRAME_COUNT)

    dh.save_blend(PROBE_DIR, "pinned_anim_transform_display.blend")
    root = dh.configure_state(project_name="pinned_anim_transform_display",
                              frame_count=FRAME_COUNT,
                              frame_rate=100,
                              step_size=0.01)

    group = dh.api.solver.create_group("Cloth", "SHELL")
    group.add(plane.name)
    group.create_pin(plane.name, "AllPin")
    addon_root = dh.groups.get_addon_data(bpy.context.scene)
    grp = addon_root.object_group_0
    grp.pin_vertex_groups_index = 0
    pin_item = grp.pin_vertex_groups[0]

    # Capture straight into the pin-anim cache (the modal Capture
    # Deformation operator needs event-loop ticks the rig's Blender does
    # not run). All verts are pinned, so the pin rows are the vertex rows.
    captured = capture_world_solver_frames(plane, 1, FRAME_COUNT)
    pc2 = __import__(pkg + ".core.pc2", fromlist=["write_pin_anim_pc2"])
    pin_ops = __import__(pkg + ".ui.dynamics.pin_ops",
                         fromlist=["_ensure_embedded_move_op"])
    pc2.write_pin_anim_pc2(plane, "AllPin", captured)
    pin_item.has_captured_anim = True
    pin_ops._ensure_embedded_move_op(pin_item)
    dh.log(f"captured {captured.shape}")

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes,
                      message="pinned_anim_transform_display:build")
    dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=15.0)
    dh.settle_idle(timeout=15.0)

    # THE POINT: park away from frame 1 before the frames land. The stale
    # snapshot matrix is then visibly wrong for every applied frame unless
    # the apply loop re-reads matrix_world per frame.
    scene.frame_set(PARK_FRAME)
    parked = int(scene.frame_current)
    dh.fetch_and_drain()

    dh.record(
        "A_parked_away_from_frame_one",
        parked != 1,
        {"parked_frame": parked},
    )

    # ---- B / C: map every PC2 sample back to solver world ------------
    # ``world_matrix`` is already the solver-space matrix
    # (zup_to_yup @ matrix_world), the same one the apply loop inverts.
    transform_mod = __import__(pkg + ".core.transform",
                               fromlist=["world_matrix"])
    arr = dh.read_pc2(dh.find_pc2_for(plane))
    n_samples = int(arr.shape[0]) if arr is not None else 0

    def world_err(sample_idx, matrix_frame):
        saved = scene.frame_current
        try:
            scene.frame_set(int(matrix_frame))
            wm = np.array(transform_mod.world_matrix(plane),
                          dtype=np.float64).reshape(4, 4)
        finally:
            scene.frame_set(saved)
        local = arr[sample_idx].astype(np.float64)
        h = np.concatenate([local, np.ones((n_verts, 1))], axis=1)
        world = (h @ wm.T)[:, :3]
        return float(np.max(np.abs(world - captured[sample_idx].astype(np.float64))))

    # PC2 index i is Blender frame i + start; this scenario leaves the
    # Starting Frame at its default of 1, so the offset is +1 here.
    own = [world_err(i, i + 1) for i in range(n_samples)]
    worst_own = max(own) if own else -1.0
    dh.record(
        "B_display_matches_own_frame_matrix",
        n_samples >= FRAME_COUNT - 1 and 0.0 <= worst_own < TOL,
        {"pc2_samples": n_samples, "worst_err": worst_own, "tol": TOL,
         "per_sample_err": [round(e, 6) for e in own],
         "parked_frame": parked,
         "note": "the bug puts one frame of object travel into each of "
                 "these numbers"},
    )

    # Same samples through the PREVIOUS frame's matrix must miss by about
    # the per-frame travel, or the object transform is not really animated
    # and B proves nothing. Sample 0 has no previous frame; skip it.
    stale = [world_err(i, i) for i in range(1, n_samples)]
    worst_stale = min(stale) if stale else -1.0
    expected_step = abs(TRAVEL_Y) / (FRAME_COUNT - 1)
    dh.record(
        "C_stale_matrix_would_miss",
        len(stale) > 0 and worst_stale > 0.5 * expected_step,
        {"n_checked": len(stale), "min_err": worst_stale,
         "expected_step": expected_step},
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
        .replace("<<TRAVEL_Y>>", repr(_TRAVEL_Y))
        .replace("<<LIFT_Z>>", repr(_LIFT_Z))
        .replace("<<PARK_FRAME>>", str(_PARK_FRAME))
        .replace("<<TOL>>", repr(_TOL))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
