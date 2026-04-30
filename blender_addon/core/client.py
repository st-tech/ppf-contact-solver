# File: client.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# This module re-exports the event-driven CommunicatorFacade as the
# ``communicator`` singleton.  All operator and UI code imports from here
# and sees the same API as before — no import changes needed anywhere.
#
# The old 1200-line Communicator class with its _process_task() loop,
# _update_status() polling, and manual lock management has been removed.
# State transitions now go through the pure transition() function
# (core/transitions.py), tested with 49 unit tests.

import os

import bpy  # pyright: ignore
import numpy

from ..models.console import console
from ..models.groups import get_addon_data, iterate_active_object_groups
from .facade import communicator, engine
from .pc2 import (
    append_pc2_frame,
    clear_gap_tracking,
    create_pc2_file,
    ensure_curve_handler,
    fill_gap_frames,
    get_pc2_path,
    load_curve_cache,
    mark_real_frame,
    object_pc2_key,
    overwrite_pc2_frame,
    read_pc2_frame_count,
    read_pc2_n_verts,
    setup_mesh_cache_modifier,
)
from .status import (
    CommunicatorInfo,
    ConnectionInfo,
    RemoteStatus,
)
from .utils import inv_world_matrix


def _apply_matrix_np(mat, positions):
    """Apply a 4x4 mathutils.Matrix to an (N,3) numpy array. Returns (N,3) float64."""
    m = numpy.array(mat, dtype=numpy.float64)
    return positions @ m[:3, :3].T + m[:3, 3]


# ---------------------------------------------------------------------------
# Animation helpers (main-thread only)
# ---------------------------------------------------------------------------

def apply_stitch_constraints(obj, map_vert, context):
    """Apply stitch constraints by averaging positions of stitched vertices.

    *map_vert* is a numpy ndarray (N, 3).  Returns an ndarray of the same shape.
    """
    try:
        from .encoder import detect_stitch_edges

        # Match the obj to its group by UUID to avoid picking the wrong
        # group if two objects share a name after a rename/duplicate.
        from .uuid_registry import get_object_uuid
        obj_uuid = get_object_uuid(obj)
        obj_group = None
        if obj_uuid:
            for group in iterate_active_object_groups(context.scene):
                for assigned in group.assigned_objects:
                    if assigned.uuid and assigned.uuid == obj_uuid:
                        obj_group = group
                        break
                if obj_group:
                    break

        if not obj_group:
            return map_vert

        # ROD meshes have no faces by design, so every edge is "loose"
        # and detect_stitch_edges flags them all. Averaging endpoints of
        # short rod edges is exactly the bug: pin animation surfaces it
        # as sudden vertex jumps when neighboring rod verts get close.
        if obj_group.object_type == "ROD":
            return map_vert

        stitch_data = detect_stitch_edges(obj.data)
        if not stitch_data:
            return map_vert

        computed_contact_gap = obj_group.computed_contact_gap
        computed_contact_offset = obj_group.computed_contact_offset
        stitch_threshold = 2 * computed_contact_gap + computed_contact_offset

        Ind, _ = stitch_data
        v1_idx = numpy.asarray(Ind[:, 0], dtype=numpy.intp)
        v2_idx = numpy.asarray(Ind[:, 1], dtype=numpy.intp)
        n = len(map_vert)
        valid = (v1_idx < n) & (v2_idx < n)
        v1_idx = v1_idx[valid]
        v2_idx = v2_idx[valid]

        stitched = map_vert.copy()
        diffs = stitched[v1_idx] - stitched[v2_idx]
        dists = numpy.linalg.norm(diffs, axis=1)
        mask = dists < stitch_threshold
        v1_m = v1_idx[mask]
        v2_m = v2_idx[mask]
        avg = (stitched[v1_m] + stitched[v2_m]) / 2.0
        stitched[v1_m] = avg
        stitched[v2_m] = avg
        return stitched

    except Exception as e:
        console.write(f"Failed to apply stitch constraints to {obj.name}: {e}")
        return map_vert


def _validate_curve_mapping(display_name, obj, map_indices, vert, spline_meta):
    # display_name is used ONLY for human-readable error text. Internal
    # routing is UUID-based via the tuples constructed in apply_animation.
    if len(map_indices) == 0:
        raise ValueError(f"Missing animation mapping for curve '{display_name}'.")
    mi = numpy.asarray(map_indices)
    if numpy.any((mi < 0) | (mi >= len(vert))):
        raise ValueError(f"Curve mapping out of range for '{display_name}'.")
    required = 0
    for meta in spline_meta:
        required += len(meta.get("params", []))
    if len(map_indices) < required:
        raise ValueError(
            f"Curve mapping too short for '{display_name}': need {required}, got {len(map_indices)}."
        )


def _validate_mesh_mapping(display_name, obj, map_indices, surface_map, vert):
    # display_name is used ONLY for human-readable error text.
    n_verts = len(obj.data.vertices)
    mi = numpy.asarray(map_indices)
    if surface_map is None:
        if len(mi) < n_verts:
            raise ValueError(
                f"Mesh mapping too short for '{display_name}': need {n_verts}, got {len(mi)}."
            )
        used = mi[:n_verts]
        if numpy.any((used < 0) | (used >= len(vert))):
            raise ValueError(f"Mesh mapping out of range for '{display_name}'.")
        return

    tri_indices, coefs, surf_tri = surface_map
    if len(tri_indices) < n_verts or len(coefs) < n_verts:
        raise ValueError(
            f"Surface mapping too short for '{display_name}': need {n_verts} entries."
        )
    ti_arr = numpy.asarray(tri_indices[:n_verts])
    if numpy.any((ti_arr < 0) | (ti_arr >= len(surf_tri))):
        raise ValueError(f"Surface triangle index out of range for '{display_name}'.")
    st_arr = numpy.asarray(surf_tri)
    tris_used = st_arr[ti_arr]
    if tris_used.shape[1] != 3:
        raise ValueError(f"Surface triangle shape is invalid for '{display_name}'.")
    if numpy.any((tris_used < 0) | (tris_used >= len(mi))):
        raise ValueError(f"Surface map index out of range for '{display_name}'.")
    mapped = mi[tris_used]
    if numpy.any((mapped < 0) | (mapped >= len(vert))):
        raise ValueError(f"Surface vertex mapping out of range for '{display_name}'.")


def _write_mesh_frame_to_pc2(obj, map_vert, blender_frame, n_verts_override=None):
    """Write a single frame to the object's PC2 file.

    Works for both mesh objects (vertex positions) and curve objects
    (CV positions in Blender's modifier layout).

    The PC2 file is keyed on the object's UUID so it survives renames;
    ``object_pc2_key`` migrates a legacy ``{obj.name}.pc2`` to the new
    location the first time it's accessed.

    Creates the file + MESH_CACHE modifier on the first arriving frame
    (regardless of frame index).  Fills gap frames when frames are
    skipped during live simulation preview.
    """
    n_verts = n_verts_override if n_verts_override is not None else len(obj.data.vertices)
    key = object_pc2_key(obj)
    pc2_path = get_pc2_path(key)
    frame_idx = blender_frame - 1  # PC2 is 0-indexed, Blender is 1-indexed

    # If the cache on disk was recorded with a different vertex count
    # (e.g. the user edited the mesh and re-transferred), discard it and
    # start fresh — appending/overwriting against a stale header produces
    # a corrupt PC2 that MESH_CACHE silently ignores.
    if os.path.exists(pc2_path):
        try:
            cached_n_verts = read_pc2_n_verts(pc2_path)
        except Exception:
            cached_n_verts = -1
        if cached_n_verts != n_verts:
            try:
                os.remove(pc2_path)
            except OSError:
                pass
            clear_gap_tracking(key)

    if not os.path.exists(pc2_path):
        # First frame to arrive — create PC2, gap-fill up to this frame,
        # and write the real data.
        os.makedirs(os.path.dirname(pc2_path), exist_ok=True)
        if obj.type == "CURVE":
            from .curve_rod import get_curve_rest_cvs
            rest_co = get_curve_rest_cvs(obj)
        else:
            rest_co = numpy.empty(n_verts * 3, dtype=numpy.float64)
            obj.data.vertices.foreach_get("co", rest_co)
            rest_co = rest_co.reshape(n_verts, 3)
        create_pc2_file(pc2_path, n_verts, start=0.0, sampling=1.0)
        # Fill gap frames [0, frame_idx) with rest pose if needed
        for _ in range(frame_idx):
            append_pc2_frame(pc2_path, rest_co, n_verts)
        # Write the actual simulation frame
        append_pc2_frame(pc2_path, map_vert, n_verts)
        mark_real_frame(key, frame_idx)
        if obj.type == "CURVE":
            # Curves use persistent handlers (MESH_CACHE doesn't work with curves)
            load_curve_cache(key)
            ensure_curve_handler()
        else:
            # Defensive try/except: if ID-write is momentarily denied,
            # the next PPF_OT_FramePump tick heals the modifier.
            try:
                setup_mesh_cache_modifier(obj, pc2_path, frame_start=1.0)
            except Exception:
                pass
    else:
        current_frame_count = read_pc2_frame_count(pc2_path)
        if frame_idx >= current_frame_count:
            if frame_idx > current_frame_count:
                fill_gap_frames(
                    pc2_path, current_frame_count - 1, frame_idx, n_verts,
                    obj_key=key,
                )
            append_pc2_frame(pc2_path, map_vert, n_verts)
        else:
            overwrite_pc2_frame(pc2_path, frame_idx, map_vert, n_verts)
        if obj.type == "CURVE":
            load_curve_cache(key)
            ensure_curve_handler()
        mark_real_frame(key, frame_idx)


# ---------------------------------------------------------------------------
# apply_animation — called from main-thread timer
# ---------------------------------------------------------------------------

def _get_curve_cv_count(obj):
    """Return total CV count for a curve in Blender's modifier layout."""
    total = 0
    for spline in obj.data.splines:
        if spline.type == "BEZIER":
            total += 3 * len(spline.bezier_points)
        else:
            total += len(spline.points)
    return total


def _apply_single_frame(context, n, vert, map_by_uuid, surface_map_by_uuid,
                        target_objects):
    """Process one simulation frame: write vertex/CV data to PC2 files.

    Both mesh and curve objects are written to PC2 without calling
    frame_set() — pure file I/O, no depsgraph evaluation.
    """
    blender_frame = n + 1

    for uid, obj, map in target_objects:
        mat = inv_world_matrix(obj)

        if obj.type == "CURVE":
            from .curve_rod import apply_fit, compute_params

            params_data = compute_params(obj)
            if not params_data.get("splines"):
                continue
            _validate_curve_mapping(uid, obj, map, vert, params_data["splines"])
            sim_pos = _apply_matrix_np(mat, vert[map])

            # Fit each spline and collect CVs in Blender's layout order
            all_cvs = []
            vi_offset = 0
            for si, spline_meta in enumerate(params_data["splines"]):
                if si >= len(obj.data.splines):
                    continue
                n_sp_verts = len(spline_meta["params"])
                sp_sim = sim_pos[vi_offset : vi_offset + n_sp_verts]
                cvs = apply_fit(obj.data.splines[si], sp_sim, spline_meta)
                all_cvs.append(cvs)
                vi_offset += n_sp_verts

            if all_cvs:
                curve_cvs = numpy.concatenate(all_cvs, axis=0)
                n_cvs = _get_curve_cv_count(obj)
                _write_mesh_frame_to_pc2(obj, curve_cvs[:n_cvs], blender_frame,
                                         n_verts_override=n_cvs)
            continue

        # --- Mesh path: write simulation output directly to PC2 ---
        surface_map = surface_map_by_uuid.get(uid)
        _validate_mesh_mapping(uid, obj, map, surface_map, vert)
        if surface_map is not None:
            # Frame-embedding reconstruction: p' = x0' + c1*b1' + c2*b2' + c3*n̂'.
            # Preserves the original vertex's absolute normal offset from the
            # tet triangle — the pure-barycentric combination (c1*v1+...) drops
            # that offset and causes subtle shrinkage after simulation.
            tri_indices_arr, coefs_arr, surf_tri_arr = surface_map
            n_verts = len(obj.data.vertices)
            ti = numpy.asarray(tri_indices_arr[:n_verts])
            c = numpy.asarray(coefs_arr[:n_verts], dtype=numpy.float64)
            tris = numpy.asarray(surf_tri_arr)[ti]
            v0 = vert[map[tris[:, 0]]]
            v1 = vert[map[tris[:, 1]]]
            v2 = vert[map[tris[:, 2]]]
            b1 = v1 - v0
            b2 = v2 - v0
            n = numpy.cross(b1, b2)
            n_sq = numpy.einsum("ij,ij->i", n, n)
            # Guard against deformed degenerate triangles: drop the normal
            # term instead of dividing by ~0. Matches the kernel's fallback.
            safe = n_sq > 1e-20
            inv_nlen = numpy.zeros_like(n_sq)
            inv_nlen[safe] = 1.0 / numpy.sqrt(n_sq[safe])
            n_hat = n * inv_nlen[:, None]
            interpolated = (
                v0
                + c[:, 0:1] * b1
                + c[:, 1:2] * b2
                + c[:, 2:3] * n_hat
            )
            map_vert = _apply_matrix_np(mat, interpolated)
        else:
            n_verts = len(obj.data.vertices)
            map_vert = _apply_matrix_np(mat, vert[map[:n_verts]])
        map_vert = apply_stitch_constraints(obj, map_vert, context)
        _write_mesh_frame_to_pc2(obj, map_vert, blender_frame)

    return blender_frame


_heal_logged: set = set()


def heal_mesh_caches_if_stale():
    """For every assigned object in an active group, guarantee that a
    correctly-configured `ContactSolverCache` MESH_CACHE modifier exists
    whenever a matching PC2 file is on disk:

      * no modifier    → create it and point it at the PC2
      * broken modifier (cache_format != PC2 or empty filepath) → rebind
      * vertex count mismatch between live mesh and PC2 → skip (the next
        fresh frame arrival will delete the stale PC2 and start over)

    Runs from the main-thread timer where ID writes are permitted. Each
    rebind is wrapped in try/except because certain Blender states briefly
    deny ID writes — the next tick will retry. Fast early-out when
    everything is already fine.
    """
    try:
        scene = bpy.context.scene
        if scene is None:
            return
        from .uuid_registry import resolve_assigned
        from ..models.groups import iterate_active_object_groups
        for g in iterate_active_object_groups(scene):
            if g.object_type == "STATIC":
                continue
            for assigned in g.assigned_objects:
                if not assigned.included:
                    continue
                obj = resolve_assigned(assigned)
                if obj is None or obj.type != "MESH":
                    continue
                pc2 = get_pc2_path(object_pc2_key(obj))
                if not os.path.exists(pc2):
                    continue
                try:
                    if read_pc2_n_verts(pc2) != len(obj.data.vertices):
                        continue
                except Exception as e:
                    # Log once per (pc2, error-kind) so a broken file doesn't
                    # spam the console on every heal tick.
                    key = (pc2, type(e).__name__)
                    if key not in _heal_logged:
                        _heal_logged.add(key)
                        from ..models.console import console
                        console.write(
                            f"[heal_mesh_caches] skipping {obj.name}: could "
                            f"not read {pc2}: {e}"
                        )
                    continue
                mod = obj.modifiers.get("ContactSolverCache")
                needs_setup = (
                    mod is None
                    or mod.cache_format != "PC2"
                    or not mod.filepath
                )
                if not needs_setup:
                    continue
                try:
                    setup_mesh_cache_modifier(obj, pc2, frame_start=1.0)
                except Exception:
                    # ID-write may be briefly blocked; next tick retries.
                    pass
    except Exception:
        pass


def apply_animation():
    """Drain every queued animation frame, writing vertex/CV data to PC2.

    One call handles both live simulation (``total==0``) and batch fetch
    (``total>0``) — the only difference is which events get dispatched
    at the end:

    - During batch fetch: dispatch ``ProgressUpdated(0.5 + 0.5*applied/total)``
      so the progress bar fills 0.5 → 1.0 across ticks, and
      ``AllFramesApplied`` on full drain.
    - During live sim: no events dispatched; frames flow as they arrive.

    Must be driven from PPF_OT_FramePump.modal (a modal-operator timer
    context) — Blender 5.x denies the State / modifier / scene.frame_*
    ID writes this performs when called from bpy.app.timers callbacks.
    """
    context = bpy.context
    com = communicator
    state = get_addon_data(context.scene).state

    try:
        target_objects = None
        max_blender_frame = 0
        last_applied = 0
        last_total = 0
        # Snapshot before we start filling the set: if it was empty,
        # this is the first apply call in the current run and we should
        # OVERWRITE scene.frame_end (which still carries Blender's 250
        # default). On subsequent calls we clamp non-decreasing so an
        # out-of-order chunk (lower max_blender_frame than a previous
        # call) cannot make the timeline go backward, which surfaced as
        # ``advanced_first10: [3, 2, 4, ...]`` on macOS CI.
        first_apply_in_run = len(state.fetched_frame) == 0

        while True:
            frame, map_by_uuid, surface_map_by_uuid, applied, total = (
                com.take_one_animation_frame()
            )
            last_applied, last_total = applied, total
            if frame is None:
                break

            n, vert = frame
            if n < 0:
                continue

            if target_objects is None:
                if getattr(context, "screen", None) and context.screen.is_animation_playing:
                    bpy.ops.screen.animation_cancel(restore_frame=False)
                target_objects = []
                # PC2 eligibility: anything the frontend put in
                # map_by_uuid. Fcurve-driven static objects are excluded
                # upstream (_exclude_from_output), so they never appear.
                for group in iterate_active_object_groups(context.scene):
                    for assigned in group.assigned_objects:
                        if not assigned.uuid:
                            raise ValueError(
                                f"client: assigned object has empty UUID"
                                f" (name={getattr(assigned, 'name', '?')})"
                            )
                        if assigned.uuid in map_by_uuid:
                            from .uuid_registry import get_object_by_uuid
                            obj = get_object_by_uuid(assigned.uuid)
                            if obj is None or obj.type not in ("MESH", "CURVE"):
                                continue
                            # ROD sims fold faces freely, so the Wireframe
                            # modifier's even-offset compensation produces
                            # huge frame-to-frame extrusion jumps. Force it
                            # off on every fetch so existing scenes (and any
                            # user-added Wireframe modifier) stay stable.
                            if (
                                group.object_type == "ROD"
                                and obj.type == "MESH"
                            ):
                                for m in obj.modifiers:
                                    if m.type == "WIREFRAME" and m.use_even_offset:
                                        m.use_even_offset = False
                            target_objects.append(
                                (assigned.uuid, obj, map_by_uuid[assigned.uuid])
                            )

            bf = _apply_single_frame(
                context, n, vert, map_by_uuid, surface_map_by_uuid,
                target_objects,
            )
            max_blender_frame = max(max_blender_frame, bf)
            state.add_fetched_frame(n)

        if max_blender_frame > 0:
            context.scene.frame_start = 1
            # Track the latest fetched frame; first apply in a run
            # overwrites Blender's 250 default, later calls clamp
            # non-decreasing so async out-of-order chunks can't
            # walk the timeline backward.
            if first_apply_in_run:
                context.scene.frame_end = max_blender_frame
            else:
                context.scene.frame_end = max(
                    context.scene.frame_end, max_blender_frame
                )
            context.scene.frame_set(max_blender_frame)

        if last_total > 0:
            if last_applied >= last_total:
                from .events import AllFramesApplied
                engine.dispatch(AllFramesApplied())
            else:
                # Each phase (FETCHING, APPLYING) owns its own 0→1 bar.
                # The apply side must only drive progress while activity
                # is APPLYING — dispatching during FETCHING would fight
                # the download-side value and flicker the bar.
                from .state import Activity
                if engine.state.activity == Activity.APPLYING:
                    from .events import ProgressUpdated
                    engine.dispatch(ProgressUpdated(
                        progress=last_applied / last_total
                    ))
    except Exception as e:
        console.write(f"Error applying animation frame: {e}")
        from .events import FetchFailed
        engine.dispatch(FetchFailed(reason=f"apply: {e}"))
