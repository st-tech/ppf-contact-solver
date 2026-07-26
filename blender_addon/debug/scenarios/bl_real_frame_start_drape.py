# File: scenarios/bl_real_frame_start_drape.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Real-backend end-to-end check of a LATE start (Starting Frame > 1).
#
# Everything else about the Starting Frame feature is covered on the emulated
# backend (bl_frame_start, bl_frame_start_leadin): the resolver, the frame-to-
# time conversion, the playback placement, and the lead-in visibility keying,
# all against a hand-written PC2. What no emulated test can show is a REAL solve
# consuming the new time-shifted schedules and producing physics, then that
# real output being placed on a late timeline with the lead-in preserved. The
# emulated advance() is a no-op, so the free vertices never move; only the CUDA
# solver drapes them. This scenario is that missing end-to-end link, so it is
# real-only.
#
# The scene: a square cloth pinned along its top edge. A shape key lifts the
# FREE region and is animated 0 -> 1 across frames 1..START, so the frames
# before the solve carry a distinct, artist-authored pose. Starting Frame is
# START; the solve begins from the fully lifted shape and drapes it down under
# gravity.
#
# Subtests:
#   A. real_solve_runs_and_moves: encode -> build -> solve -> fetch completed
#      on the real solver, the PC2 is finite with >= frame_count-1 samples, the
#      pinned edge held to round-off, and the free region traveled DOWNWARD
#      (genuine dynamics the emulator freezes), all driven through the
#      frame_start encoder. Direction IS asserted: the encoder captures the
#      starting-frame deform pose as the solver's initial state, so the solve
#      begins from the lifted lead-in shape at rest and gravity leaves it
#      nowhere to go but down. Travel is measured across the SOLVER rows (PC2
#      index 1 onward); index 0 is the display gap-fill, not solver output, so
#      differencing against it would report a display artifact as physics.
#   B. output_lands_on_late_frames: after apply, the cache modifier carries the
#      START offset and the timeline spans [START, START+samples-1]. Evaluating
#      the mesh at Blender frame START reproduces solver row 0, so the real
#      output is placed where the late start puts it, not at frame 1.
#   C. leadin_preserved_on_real_output: deep in the lead-in the evaluated mesh
#      is the artist's partial-lift pose, NOT the solver's row-0 (full-lift)
#      pose, and the modifier visibility fcurves are the two CONSTANT keys the
#      fix writes (off at START-1, on at START). The regression that shipped
#      would show the frozen solver start pose here instead.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# Real-only: the drape is genuine gravity dynamics that the kinematic emulator
# freezes. Selected by the AWS Linux / Windows GPU jobs via
# ``runtests --backend real``.
BACKENDS = ("real",)


_FRAME_COUNT = 24
_START_FRAME = 20
# +Z lift applied to the free region at full shape-key value. Large relative to
# the gravity sag over the solve window, so "lifted" vs "sagged" and "partial"
# vs "full" lift are never a tolerance question.
_LEADIN_LIFT = 1.5
# A frame deep in the lead-in. At (3-1)/(START-1) of the ramp the free region
# sits near a tenth of the full lift, unmistakably distinct from the solver's
# row-0 full-lift pose that a broken (always-on) cache would show.
_LEADIN_PROBE_FRAME = 3
# Floor on the free region's DOWNWARD travel across the solve window. Its job is
# to separate a real solve from the emulated backend's frozen no-op, which leaves
# the free vertices at exactly zero travel; it is NOT a prediction of sag depth.
# The rig runs at 100 fps (``_driver_lib.configure_state``), so 23 solved frames
# span 0.23 s and free fall alone caps the mean drop at 0.5*9.8*0.23^2 = 0.259.
# Any floor at or above that is unsatisfiable no matter how the scene is posed,
# which is what the previous 0.3 was. This matches the floor
# ``bl_real_shell_drape`` already uses for the same grid, frame count and rig
# settings, so the two real drape checks stay calibrated together.
_MIN_DOWNWARD_TRAVEL = 0.05


_DRIVER_BODY = r"""
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>
START_FRAME = <<START_FRAME>>
LEADIN_LIFT = <<LEADIN_LIFT>>
LEADIN_PROBE_FRAME = <<LEADIN_PROBE_FRAME>>
MIN_DOWNWARD_TRAVEL = <<MIN_DOWNWARD_TRAVEL>>

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    pc2mod = __import__(pkg + ".core.pc2", fromlist=["MODIFIER_NAME"])
    utils_mod = __import__(pkg + ".core.utils", fromlist=["_get_fcurves"])

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=10, y_subdivisions=10, size=2.0, location=(0, 0, 0),
    )
    sheet = bpy.context.object
    sheet.name = "LateDrapeSheet"
    # SHELL meshes are not remeshed, so PC2 vertex order matches the input.
    pinned_idx = [i for i, v in enumerate(sheet.data.vertices) if v.co.y > 0.99]
    free_idx = [i for i, v in enumerate(sheet.data.vertices) if v.co.y <= 0.99]
    vg = sheet.vertex_groups.new(name="TopEdge")
    vg.add(pinned_idx, 1.0, "REPLACE")

    # Lead-in deformer: a shape key that lifts ONLY the free region in +Z,
    # animated 0 -> 1 over frames 1..START. Shape keys evaluate before the
    # modifier stack, so an always-on MESH_CACHE in OVERWRITE mode would hide
    # this: exactly the collision the visibility keying resolves. Pinned verts
    # stay at rest so the pin holds a clean edge.
    sheet.shape_key_add(name="Basis", from_mix=False)
    lift = sheet.shape_key_add(name="Lift", from_mix=False)
    for i in free_idx:
        lift.data[i].co.z += LEADIN_LIFT
    lift.value = 0.0
    lift.keyframe_insert("value", frame=1)
    lift.value = 1.0
    lift.keyframe_insert("value", frame=START_FRAME)
    # Linear ramp so the partial-lift pose at the probe frame is predictable.
    ad = sheet.data.shape_keys.animation_data
    if ad and ad.action:
        for fc in utils_mod._get_fcurves(ad.action):
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
    dh.log(f"grid verts={len(sheet.data.vertices)} pinned={len(pinned_idx)}")

    dh.save_blend(PROBE_DIR, "real_frame_start_drape.blend")
    root = dh.configure_state(
        project_name="real_frame_start_drape",
        frame_count=FRAME_COUNT,
        gravity=(0.0, 0.0, -9.8),
    )
    state = root.state
    # The feature under test: an addon-owned late start, field mode (not the
    # scene override), so resolve_start_frame returns exactly START_FRAME.
    state.use_scene_frame_start = False
    state.frame_start = START_FRAME
    scene = bpy.context.scene
    scene.frame_start = START_FRAME
    scene.frame_end = START_FRAME + FRAME_COUNT - 1

    enc = __import__(pkg + ".core.encoder", fromlist=["resolve_start_frame"])
    resolved_start = int(enc.resolve_start_frame(state))

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)
    cloth.create_pin(sheet.name, "TopEdge")

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect(
        local_path=LOCAL_PATH,
        server_port=SERVER_PORT,
        project_name=root.state.project_name,
    )
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes,
                      message="real_frame_start_drape:build", timeout=300.0)
    dh.run_and_wait(timeout=300.0)
    solver_state = dh.facade.engine.state.solver.name
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=120.0)
    dh.settle_idle(timeout=15.0)
    applied, total = dh.fetch_and_drain()
    dh.log(f"fetch.drained applied={applied}/{total} solver={solver_state}")

    pc2_path = dh.find_pc2_for(sheet)
    arr = dh.read_pc2(pc2_path) if pc2_path else None
    samples = int(arr.shape[0]) if arr is not None else 0
    finite = bool(arr is not None and np.all(np.isfinite(arr)))

    pinned = np.asarray(pinned_idx, dtype=np.int64)
    free = np.asarray(free_idx, dtype=np.int64)

    def eval_free_mean_z(frame):
        scene.frame_set(frame)
        dg = bpy.context.evaluated_depsgraph_get()
        ev = sheet.evaluated_get(dg)
        me = ev.to_mesh()
        a = np.empty(len(me.vertices) * 3, dtype=np.float64)
        me.vertices.foreach_get("co", a)
        ev.to_mesh_clear()
        return float(a.reshape(-1, 3)[free][:, 2].mean())

    # ----- A: the real solver ran the late-start scene and produced
    # genuine dynamics ------------------------------------------------
    # The frame_start encoder emitted a scene the real CUDA solver accepted
    # and simulated: >= frame_count-1 finite frames, the pinned edge held to
    # round-off, and the free region traveled DOWNWARD (the emulator would
    # freeze it). Direction IS asserted, because the encoder hands the solver
    # the starting-frame deform-evaluated mesh as its initial state
    # (``encoder/mesh.py:_start_frame_eval_local_verts``, which frame_sets to
    # ``resolve_start_frame`` first), and shape keys are part of that
    # evaluation. So the solve starts from the fully lifted lead-in pose at
    # rest, with that same pose as its rest metric, and gravity is the only
    # thing acting: the sheet cannot end above where it started.
    #
    # The travel is measured from the first SOLVER row, and index 0 must stay
    # out of it. Index 0 is the display gap-fill, written from the
    # starting-frame depsgraph pose, which is the very pose the encoder ships
    # as the initial state. A row-0-to-last difference therefore reports how
    # far those two paths agree rather than how far gravity moved the cloth,
    # and when they disagree it reports the disagreement at whatever sign and
    # magnitude that happens to be, which is why this subtest once read +1.26
    # on a sheet that was sagging. The pinned check still starts at index 0, so
    # the hold keeps covering the display gap frame.
    pin_disp = -1.0
    mean_free_dz = 0.0
    if arr is not None and samples >= 2 and finite:
        first_solver = arr[1]
        last = arr[-1]
        pin_disp = float(np.max(np.linalg.norm(last[pinned] - arr[0][pinned], axis=1)))
        mean_free_dz = float(np.mean((last[free] - first_solver[free])[:, 2]))
    dh.record(
        "A_real_solve_runs_and_moves",
        solver_state != "FAILED"
        and arr is not None and samples >= FRAME_COUNT - 1 and finite
        and pin_disp >= 0.0 and pin_disp < 0.1
        and mean_free_dz < -MIN_DOWNWARD_TRAVEL,
        {"solver_state": solver_state, "samples": samples,
         "expected_min_samples": FRAME_COUNT - 1, "all_finite": finite,
         "max_pinned_disp": round(pin_disp, 5),
         "mean_free_travel_z": round(mean_free_dz, 5),
         "required_below": -MIN_DOWNWARD_TRAVEL,
         "error": dh.facade.engine.state.error},
    )

    # ----- B: real output is placed on the late timeline --------------
    mod = sheet.modifiers.get(pc2mod.MODIFIER_NAME)
    row0_free_z = float(arr[0][free][:, 2].mean()) if arr is not None else None
    z_at_start = eval_free_mean_z(START_FRAME) if arr is not None else None
    place_ok = (
        mod is not None
        and abs(float(mod.frame_start) - START_FRAME) < 1e-6
        and int(scene.frame_start) == START_FRAME
        and int(scene.frame_end) == START_FRAME + samples - 1
        and row0_free_z is not None
        and abs(z_at_start - row0_free_z) < 1e-3
    )
    dh.record(
        "B_output_lands_on_late_frames",
        bool(place_ok),
        {"mod_frame_start": (float(mod.frame_start) if mod else None),
         "scene_start": int(scene.frame_start),
         "scene_end": int(scene.frame_end),
         "expected_end": START_FRAME + samples - 1,
         "mesh_z_at_start": (round(z_at_start, 5) if z_at_start is not None else None),
         "pc2_row0_free_z": (round(row0_free_z, 5) if row0_free_z is not None else None)},
    )

    # ----- C: the lead-in survives on real output ---------------------
    z_leadin = eval_free_mean_z(LEADIN_PROBE_FRAME) if arr is not None else None
    partial_expected = LEADIN_LIFT * (LEADIN_PROBE_FRAME - 1) / (START_FRAME - 1)
    # Visibility keys written by the REAL fetch path (apply_animation ->
    # setup_mesh_cache_modifier -> sync_cache_visibility_keys). Read through the
    # SLOT-ROBUST iterator: the sheet owns a shape-key action, so the keys land
    # in its own slot, which the first-channelbag utils._get_fcurves would miss
    # (that mismatch is exactly what this real run first surfaced).
    vis = {}
    act = sheet.animation_data.action if sheet.animation_data else None
    for _, fc in pc2mod._iter_cache_visibility_fcurves(act):
        vis[fc.data_path.rsplit(".", 1)[1]] = [
            (kp.co[0], kp.co[1], kp.interpolation) for kp in fc.keyframe_points
        ]
    expected_keys = [(float(START_FRAME - 1), 0.0, "CONSTANT"),
                     (float(START_FRAME), 1.0, "CONSTANT")]
    keys_ok = (set(vis) == {"show_viewport", "show_render"}
               and all(vis[k] == expected_keys for k in vis))
    # Behavioral: deep in the lead-in the mesh is the deformer's partial-lift
    # pose to round-off. A broken (always-on) cache would clamp to solver row 0
    # here instead, which is a different value, so the exact match is the
    # discriminator.
    leadin_ok = z_leadin is not None and abs(z_leadin - partial_expected) < 0.03
    dh.record(
        "C_leadin_preserved_on_real_output",
        bool(keys_ok and leadin_ok),
        {"z_at_leadin_probe": (round(z_leadin, 5) if z_leadin is not None else None),
         "partial_lift_expected": round(partial_expected, 5),
         "solver_row0_free_z": (round(row0_free_z, 5) if row0_free_z is not None else None),
         "probe_frame": LEADIN_PROBE_FRAME,
         "visibility_keys": vis, "expected_keys_each": expected_keys},
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
        .replace("<<START_FRAME>>", str(_START_FRAME))
        .replace("<<LEADIN_LIFT>>", repr(_LEADIN_LIFT))
        .replace("<<LEADIN_PROBE_FRAME>>", str(_LEADIN_PROBE_FRAME))
        .replace("<<MIN_DOWNWARD_TRAVEL>>", repr(_MIN_DOWNWARD_TRAVEL))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 420.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
