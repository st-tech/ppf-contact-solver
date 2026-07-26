# File: scenarios/bl_frame_start_leadin.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A solve that starts at frame S must leave frames 1..S-1 to the artist.
#
# MESH_CACHE clamps a cache index below zero to row 0, and its OVERWRITE deform
# mode then stamps that row onto the mesh. So a cache modifier left enabled
# through the lead-in replaces whatever the artist animated there with the
# simulation's STARTING pose, frozen. That defeats the entire point of a late
# start, and it is invisible to any test that only checks where the simulated
# frames land: the offset arithmetic is correct, the modifier is bound
# correctly, and the cached frames all display exactly where they should. It is
# the frames OUTSIDE the cache's range that are wrong.
#
# ``sync_cache_visibility_keys`` (core/pc2.py) writes two CONSTANT keys, off at
# S-1 and on at S, so the mesh belongs to its own deformers until the solve
# begins. This scenario is the gate on that behavior, driven through a shape
# key so the lead-in pose is deterministic and obviously distinct from the
# cached pose.
#
# Subtests:
#   A. keys_written_constant: two CONSTANT keys per visibility path, off at
#         S-1 and on at S. CONSTANT because this is a switch: a default BEZIER
#         handle would put fractional visibility on the frames between.
#   B. leadin_shows_deformer: BEFORE S the evaluated mesh is the shape-keyed
#         pose, not the cached row 0. This is the regression that shipped.
#   C. cache_wins_from_start: AT S the evaluated mesh is cached row 0, so the
#         handoff happens on exactly the right frame and not one either side.
#   D. frame_one_writes_no_keys: a solve starting at frame 1 has no lead-in, so
#         it stays plain always-on with no keys at all.
#   E. moving_back_to_one_clears: lowering the Starting Frame back to 1 removes
#         the keys and leaves the modifier visible, rather than stranding it
#         switched off for the first frames.
#   F. cleanup_removes_keys: tearing down the cache drops the keys too, so they
#         do not linger as channels pointing at a modifier that is gone.
#   G. missing_keys_detected: ``needs_cache_visibility_keys`` reports a cache
#         bound before this keying existed, which is what lets the heal pass
#         adopt it without waiting for a re-fetch.
#   H. static_leadin_shows_deformer: a STATIC collider binds its output cache
#         with place_after_deformers=True, a different stack position and a
#         different call path, so the same lead-in handoff is asserted there,
#         with a Subsurf above the cache to make the placement meaningful.
#   I. static_deform_sidecar_untouched: the captured-deformation sidecar (the
#         INPUT the solver consumes) is a separate cache from the output PC2,
#         and the visibility keying must not disturb it.
#   J. keys_survive_coexisting_action: when the object already owns an action,
#         Blender 5.x puts the visibility keys in its own slot (not the first
#         channelbag), so sync/needs/remove must walk every slot. A real GPU
#         run surfaced this; it is pinned here so the emulated suite guards it.
#
# Pure Blender scenario: no server, no solver, no transfer. The PC2 is written
# directly so the assertions are about playback placement, not about physics.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True

# Distinct, and distinct from 1, so an off-by-one or a stray hardcoded origin
# is unmistakable.
_START_FRAME = 24
# The lead-in pose (shape key fully on) and the cached pose, far apart so no
# tolerance question arises.
_LEADIN_Z = 5.0
_CACHED_Z = -3.0
_N_CACHE_FRAMES = 6


_DRIVER_TEMPLATE = r"""
import os, time, traceback
import bpy
import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

START_FRAME = <<START_FRAME>>
LEADIN_Z = <<LEADIN_Z>>
CACHED_Z = <<CACHED_Z>>
N_CACHE_FRAMES = <<N_CACHE_FRAMES>>


def log(msg):
    result["phases"].append((round(time.time(), 3), msg))


def record(name, ok, details):
    result["checks"][name] = {"ok": bool(ok), "details": details}


try:
    pkg = [m for m in __import__("sys").modules
           if m.endswith("ppf_contact_solver")][0]
    import sys
    pc2m = sys.modules[pkg + ".core.pc2"]
    utils = sys.modules[pkg + ".core.utils"]

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(size=2.0, x_subdivisions=2,
                                    y_subdivisions=2, location=(0, 0, 0))
    obj = bpy.context.active_object
    obj.name = "LeadInCloth"
    n_verts = len(obj.data.vertices)

    # Lead-in motion via a shape key held fully on. Shape keys evaluate before
    # the modifier stack, so an enabled MESH_CACHE in OVERWRITE mode wins over
    # this: exactly the collision the fix has to resolve.
    obj.shape_key_add(name="Basis", from_mix=False)
    lift = obj.shape_key_add(name="Lift", from_mix=False)
    for i in range(n_verts):
        lift.data[i].co.z = LEADIN_Z
    lift.value = 1.0

    # A cache whose every row sits at CACHED_Z, so "which pose is showing" is
    # a single unambiguous number.
    key = pc2m.object_pc2_key(obj)
    pc2_path = pc2m.get_pc2_path(key)
    os.makedirs(os.path.dirname(pc2_path), exist_ok=True)
    base = np.zeros((n_verts, 3), dtype=np.float32)
    for i, v in enumerate(obj.data.vertices):
        base[i] = (v.co.x, v.co.y, CACHED_Z)
    pc2m.write_pc2(pc2_path, [base.copy() for _ in range(N_CACHE_FRAMES)])
    log("pc2 written frames=" + str(N_CACHE_FRAMES))

    pc2m.setup_mesh_cache_modifier(obj, pc2_path, frame_start=float(START_FRAME))
    mod = obj.modifiers[pc2m.MODIFIER_NAME]

    def vis_fcurves():
        act = obj.animation_data.action if obj.animation_data else None
        if act is None:
            return {}
        out = {}
        for fc in utils._get_fcurves(act):
            if pc2m.MODIFIER_NAME in fc.data_path:
                out[fc.data_path.rsplit(".", 1)[1]] = [
                    (kp.co[0], kp.co[1], kp.interpolation)
                    for kp in fc.keyframe_points
                ]
        return out

    def mean_z_at(frame):
        scene = bpy.context.scene
        scene.frame_set(frame)
        dg = bpy.context.evaluated_depsgraph_get()
        ev = obj.evaluated_get(dg)
        me = ev.to_mesh()
        a = np.empty(len(me.vertices) * 3, dtype=np.float64)
        me.vertices.foreach_get("co", a)
        ev.to_mesh_clear()
        return float(a.reshape(-1, 3)[:, 2].mean())

    # ---- A: two CONSTANT keys per path, off at S-1, on at S ----------
    curves = vis_fcurves()
    expected = [(float(START_FRAME - 1), 0.0, "CONSTANT"),
                (float(START_FRAME), 1.0, "CONSTANT")]
    a_ok = (
        set(curves) == {"show_viewport", "show_render"}
        and all(curves[k] == expected for k in curves)
    )
    record("A_keys_written_constant", a_ok,
           {"curves": curves, "expected_each": expected})

    # ---- B: the lead-in belongs to the deformer ----------------------
    z_before = mean_z_at(START_FRAME - 3)
    record("B_leadin_shows_deformer",
           abs(z_before - LEADIN_Z) < 1e-4,
           {"mean_z": z_before, "expected": LEADIN_Z,
            "cached_pose_would_be": CACHED_Z,
            "frame": START_FRAME - 3,
            "cache_on": bool(mod.show_viewport)})

    # ---- C: the cache takes over exactly at S ------------------------
    z_at = mean_z_at(START_FRAME)
    z_prev = mean_z_at(START_FRAME - 1)
    record("C_cache_wins_from_start",
           abs(z_at - CACHED_Z) < 1e-4 and abs(z_prev - LEADIN_Z) < 1e-4,
           {"mean_z_at_start": z_at, "expected": CACHED_Z,
            "mean_z_one_before": z_prev, "expected_one_before": LEADIN_Z})

    # ---- D: a frame-1 solve stays plain always-on --------------------
    pc2m.sync_cache_visibility_keys(obj, 1)
    d_curves = vis_fcurves()
    record("D_frame_one_writes_no_keys",
           not d_curves and mod.show_viewport and mod.show_render,
           {"curves": d_curves, "show_viewport": bool(mod.show_viewport),
            "show_render": bool(mod.show_render)})

    # ---- E: lowering the start back to 1 clears and re-enables -------
    pc2m.sync_cache_visibility_keys(obj, START_FRAME)
    had = len(vis_fcurves())
    pc2m.sync_cache_visibility_keys(obj, 1)
    z_leadin_after = mean_z_at(START_FRAME - 3)
    record("E_moving_back_to_one_clears",
           had == 2 and not vis_fcurves()
           and abs(z_leadin_after - CACHED_Z) < 1e-4,
           {"keys_before": had, "keys_after": len(vis_fcurves()),
            "mean_z_in_old_leadin": z_leadin_after,
            "expected": CACHED_Z,
            "why": "at S=1 the cache drives every frame again"})

    # ---- F: teardown takes the keys with it --------------------------
    pc2m.sync_cache_visibility_keys(obj, START_FRAME)
    before_cleanup = len(vis_fcurves())
    pc2m.cleanup_mesh_cache(obj)
    record("F_cleanup_removes_keys",
           before_cleanup == 2 and not vis_fcurves()
           and obj.modifiers.get(pc2m.MODIFIER_NAME) is None,
           {"keys_before_cleanup": before_cleanup,
            "keys_after": len(vis_fcurves()),
            "modifier_gone": obj.modifiers.get(pc2m.MODIFIER_NAME) is None})

    # ---- G: a key-less cache is detected for healing -----------------
    pc2m.setup_mesh_cache_modifier(obj, pc2_path, frame_start=float(START_FRAME))
    fresh = not pc2m.needs_cache_visibility_keys(obj, START_FRAME)
    pc2m.remove_cache_visibility_keys(obj)
    stripped = pc2m.needs_cache_visibility_keys(obj, START_FRAME)
    at_one = pc2m.needs_cache_visibility_keys(obj, 1)
    record("G_missing_keys_detected",
           fresh and stripped and not at_one,
           {"fresh_cache_needs_nothing": bool(fresh),
            "stripped_cache_needs_keys": bool(stripped),
            "frame_one_never_needs_keys": bool(at_one)})

    # ---- H/I: the STATIC collider shape of the same problem ----------
    # A STATIC binds its output cache with place_after_deformers=True, so the
    # modifier sits AFTER the deformers that feed the simulation rather than
    # at index 0. That is a different stack position and a different call
    # path, and the lead-in has to behave the same way: the collider's own
    # deformers own the frames before the solve. Its captured-deformation
    # sidecar is a separate cache from the output PC2 checked above, and it
    # must survive the visibility keying untouched.
    bpy.ops.mesh.primitive_grid_add(size=2.0, x_subdivisions=2,
                                    y_subdivisions=2, location=(0, 0, 0))
    coll = bpy.context.active_object
    coll.name = "LeadInCollider"
    nv2 = len(coll.data.vertices)
    coll.shape_key_add(name="Basis", from_mix=False)
    lift2 = coll.shape_key_add(name="Lift", from_mix=False)
    for i in range(nv2):
        lift2.data[i].co.z = LEADIN_Z
    lift2.value = 1.0
    # A SUBSURF above the cache makes the placement meaningful: it is a
    # topology changer, so the cache must land before it and after the
    # deformers rather than simply at the end of the stack.
    coll.modifiers.new(name="Subsurf", type="SUBSURF")

    coll_key = pc2m.object_pc2_key(coll)
    coll_pc2 = pc2m.get_pc2_path(coll_key)
    base2 = np.zeros((nv2, 3), dtype=np.float32)
    for i, v in enumerate(coll.data.vertices):
        base2[i] = (v.co.x, v.co.y, CACHED_Z)
    pc2m.write_pc2(coll_pc2, [base2.copy() for _ in range(N_CACHE_FRAMES)])

    # Captured-deformation sidecar, the input the solver consumes. Distinct
    # key from the output PC2; nothing here should disturb it.
    sd_rows = np.zeros((N_CACHE_FRAMES, nv2, 3), dtype=np.float32)
    for k in range(N_CACHE_FRAMES):
        sd_rows[k, :, 2] = LEADIN_Z + k
    pc2m.write_static_deform_pc2(coll, sd_rows)

    pc2m.setup_mesh_cache_modifier(coll, coll_pc2,
                                   frame_start=float(START_FRAME),
                                   place_after_deformers=True)
    cmod = coll.modifiers[pc2m.MODIFIER_NAME]
    order = [m.type for m in coll.modifiers]
    cache_i = coll.modifiers.find(pc2m.MODIFIER_NAME)
    subsurf_i = coll.modifiers.find("Subsurf")

    def coll_mean_z(frame):
        scene = bpy.context.scene
        scene.frame_set(frame)
        dg = bpy.context.evaluated_depsgraph_get()
        ev = coll.evaluated_get(dg)
        me = ev.to_mesh()
        a = np.empty(len(me.vertices) * 3, dtype=np.float64)
        me.vertices.foreach_get("co", a)
        ev.to_mesh_clear()
        return float(a.reshape(-1, 3)[:, 2].mean())

    z_lead = coll_mean_z(START_FRAME - 3)
    z_start = coll_mean_z(START_FRAME)
    # Subsurf averages toward the interior, so an exact equality would be
    # wrong here. What matters is which pose the stack is built on, and the
    # two are far enough apart that "nearer to" is unambiguous.
    record("H_static_leadin_shows_deformer",
           abs(z_lead - LEADIN_Z) < abs(z_lead - CACHED_Z)
           and abs(z_start - CACHED_Z) < abs(z_start - LEADIN_Z)
           and cache_i < subsurf_i,
           {"mean_z_leadin": z_lead, "mean_z_at_start": z_start,
            "leadin_pose": LEADIN_Z, "cached_pose": CACHED_Z,
            "modifier_order": order,
            "cache_before_subsurf": cache_i < subsurf_i})

    sd_after = pc2m.get_static_deform_cache(coll)
    record("I_static_deform_sidecar_untouched",
           pc2m.has_static_deform_animation(coll)
           and sd_after is not None
           and tuple(sd_after.shape) == (N_CACHE_FRAMES, nv2, 3)
           and abs(float(sd_after[0][:, 2].mean()) - LEADIN_Z) < 1e-4,
           {"has_cache": bool(pc2m.has_static_deform_animation(coll)),
            "shape": list(sd_after.shape) if sd_after is not None else None,
            "expected_shape": [N_CACHE_FRAMES, nv2, 3],
            "row0_mean_z": (float(sd_after[0][:, 2].mean())
                            if sd_after is not None else None),
            "expected_row0": LEADIN_Z})

    # ---- J: keys survive a coexisting action on the object ----------
    # When the mesh already owns an action (a shape-key-animated cloth, an
    # object-level transform track), Blender 5.x puts a freshly inserted
    # modifier key in the object's OWN slot, which is often not the first
    # channelbag. Reading through the first-channelbag helper then misses the
    # keys and reports them absent, which drove the heal pass to rewrite them
    # every tick and left them BEZIER. sync/needs/remove must walk every slot.
    # This is the case the emulated suite could not see until a real run
    # surfaced it, so it is pinned here.
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=2, y_subdivisions=2, size=1.0)
    coexist = bpy.context.object
    coexist.name = "CoexistAction"
    coexist.modifiers.new(name=pc2m.MODIFIER_NAME, type="MESH_CACHE")
    # A separate animated channel that forces its own action slot.
    coexist.location.z = 0.0
    coexist.keyframe_insert("location", index=2, frame=1)
    coexist.location.z = 1.0
    coexist.keyframe_insert("location", index=2, frame=10)

    def coexist_vis_keys():
        act2 = coexist.animation_data.action if coexist.animation_data else None
        out = {}
        for _, fc in pc2m._iter_cache_visibility_fcurves(act2):
            out[fc.data_path.rsplit(".", 1)[1]] = [
                (kp.co[0], kp.co[1], kp.interpolation) for kp in fc.keyframe_points
            ]
        return out

    pc2m.sync_cache_visibility_keys(coexist, START_FRAME)
    jk = coexist_vis_keys()
    j_expected = [(float(START_FRAME - 1), 0.0, "CONSTANT"),
                  (float(START_FRAME), 1.0, "CONSTANT")]
    written_ok = (set(jk) == {"show_viewport", "show_render"}
                  and all(jk[k] == j_expected for k in jk))
    needs_false = pc2m.needs_cache_visibility_keys(coexist, START_FRAME) is False
    n_removed = pc2m.remove_cache_visibility_keys(coexist)
    needs_true = pc2m.needs_cache_visibility_keys(coexist, START_FRAME) is True
    record("J_keys_survive_coexisting_action",
           written_ok and needs_false and n_removed == 2 and needs_true,
           {"visibility_keys": jk, "expected_each": j_expected,
            "needs_when_present": not needs_false,
            "removed": n_removed, "needs_after_remove": needs_true})

    log("checks=" + str(len(result["checks"])) + " done")
except Exception as exc:
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<START_FRAME>>", str(_START_FRAME))
        .replace("<<LEADIN_Z>>", repr(_LEADIN_Z))
        .replace("<<CACHED_Z>>", repr(_CACHED_Z))
        .replace("<<N_CACHE_FRAMES>>", str(_N_CACHE_FRAMES))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
