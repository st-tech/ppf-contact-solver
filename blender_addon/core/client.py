# File: client.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Re-exports the event-driven ``CommunicatorFacade`` as the ``communicator``
# singleton, plus the helpers operators and UI code call directly. State
# transitions go through the pure ``transition()`` function in
# ``core/transitions.py``.
#
# The ``communicator`` re-export here is a backward-compatible surface for
# legacy UI and MCP call sites; new core code should import the singleton
# directly from ``core/facade.py`` (its definition site, which is import-cycle
# free) rather than going through this module.

import collections
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
    has_static_deform_animation,
    get_static_deform_cache,
    load_curve_cache,
    mark_real_frame,
    object_pc2_key,
    object_pc2_key_readonly,
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
from .transform import inv_world_matrix


# Per-target apply state. ``needs_per_frame`` is True for STATIC meshes
# whose matrix_world can vary across frames (fcurves, NLA, drivers,
# parent chain, constraints). For those, ``constant_inv`` is None and
# the apply loop refreshes the inverse from a per-tick ``scene.frame_set``
# before each write. For everyone else ``constant_inv`` carries a
# pre-snapshotted ``inv_world_matrix(obj)`` numpy (4,4) and the loop
# skips the per-frame depsgraph eval.
_ApplyTarget = collections.namedtuple(
    "_ApplyTarget",
    ["uuid", "obj", "map", "object_type", "needs_per_frame", "constant_inv"],
)


def _static_needs_per_frame_matrix(obj):
    """True if a STATIC obj's matrix_world can vary across frames.

    Conservative: any frame-dependent source of motion makes us pay the
    extra per-frame ``scene.frame_set`` cost. We use Blender's own
    depsgraph eval (rather than re-implementing fcurve/NLA/driver math)
    so this also covers parent chains, constraints, and drivers.
    """
    from .utils import _get_fcurves
    ad = obj.animation_data
    if ad:
        if ad.action and any(_get_fcurves(ad.action)):
            return True
        if list(ad.nla_tracks):
            return True
        if list(ad.drivers):
            return True
    if obj.parent is not None:
        return True
    if obj.constraints:
        return True
    return False


def _apply_matrix_np(mat, positions):
    """Apply a 4x4 mathutils.Matrix to an (N,3) numpy array. Returns (N,3) float64."""
    m = numpy.array(mat, dtype=numpy.float64)
    return positions @ m[:3, :3].T + m[:3, 3]


def _needs_after_deformers(object_type, obj):
    """True when the object's MESH_CACHE must sit after upstream deformers.

    STATIC colliders always need it (captured-deformation or fcurve
    animation flows into the simulator). SHELL/SOLID/ROD cloths inherit
    the same need whenever they carry a deforming modifier stack
    (Armature, Lattice, ...) that would otherwise re-deform the solver's
    PC2 output on top of itself.
    """
    from .utils import has_deforming_modifier_stack
    return object_type == "STATIC" or has_deforming_modifier_stack(obj)


# ---------------------------------------------------------------------------
# Animation helpers (main-thread only)
# ---------------------------------------------------------------------------

def _apply_post_snap_closure(context, world_by_uuid):
    """Snap every stitched source vertex exactly onto its target, in world
    space, across the per-object reconstructed position arrays.

    ``world_by_uuid`` maps object uuid -> ndarray (N, 3) of that object's
    deformed positions for the frame, indexed by the object's local vertex
    (mesh) or sample (rod) index, which is the same index space the stitch
    data uses. Arrays are mutated in place. Handles both the cross-object
    snap/merge stitches (every supported type pair, solids reconstructed via
    surface embedding upstream) and the intra-object loose-edge stitches.
    The caller gates this on the global ``post_snap_exactly`` toggle.
    """
    try:
        import json

        # Snapshot so chained stitches read original (pre-move) targets.
        snap = {uid: arr.copy() for uid, arr in world_by_uuid.items()}

        # Cross-object stitches (snap/merge pairs, all supported type pairs).
        state = get_addon_data(context.scene).state
        for pair in state.merge_pairs:
            if not pair.cross_stitch_json:
                continue
            try:
                data = json.loads(pair.cross_stitch_json)
            except Exception:
                continue
            src_uid = data.get("source_uuid")
            tgt_uid = data.get("target_uuid")
            if src_uid not in world_by_uuid or tgt_uid not in snap:
                continue
            src_arr = world_by_uuid[src_uid]
            tgt = snap[tgt_uid]
            n_src, n_tgt = len(src_arr), len(tgt)
            for row, wt in zip(data.get("ind", []), data.get("w", [])):
                # The snap-time JSON keeps the source as a single Blender
                # vertex (the SOLID-source tet re-projection is solver-only),
                # so this Blender-space closure snaps that source vertex onto
                # the target barycentric point. New rows are 6-wide
                # [s, s, s, t0, t1, t2] / [1, 0, 0, a, b, c] (target bary in
                # slots 3..5); legacy rows are 4-wide [s, t0, t1, t2] /
                # [1, a, b, c]. Accept both so pre-migration scenes still
                # snap until they are re-snapped.
                if len(row) >= 6 and len(wt) >= 6:
                    si = int(row[0])
                    t0, t1, t2 = int(row[3]), int(row[4]), int(row[5])
                    a, b, c = float(wt[3]), float(wt[4]), float(wt[5])
                elif len(row) >= 4 and len(wt) >= 4:
                    si = int(row[0])
                    t0, t1, t2 = int(row[1]), int(row[2]), int(row[3])
                    a, b, c = float(wt[1]), float(wt[2]), float(wt[3])
                else:
                    continue
                if si >= n_src or t0 >= n_tgt or t1 >= n_tgt or t2 >= n_tgt:
                    continue
                src_arr[si] = a * tgt[t0] + b * tgt[t1] + c * tgt[t2]

        # Intra-object loose-edge stitches. Skip ROD: every rod edge is
        # "loose", so averaging endpoints would corrupt the rod.
        from .encoder import detect_stitch_edges
        from .uuid_registry import get_object_uuid, resolve_assigned
        for group in iterate_active_object_groups(context.scene):
            if group.object_type == "ROD":
                continue
            thr = 2 * group.computed_contact_gap + group.computed_contact_offset
            for obj_ref in group.assigned_objects:
                if not obj_ref.included:
                    continue
                obj = resolve_assigned(obj_ref)
                if not obj or obj.type != "MESH":
                    continue
                uid = get_object_uuid(obj)
                if uid is None or uid not in world_by_uuid:
                    continue
                sd = detect_stitch_edges(obj.data)
                if not sd:
                    continue
                arr = world_by_uuid[uid]
                base = snap[uid]
                Ind, _ = sd
                v1 = numpy.asarray(Ind[:, 0], dtype=numpy.intp)
                v2 = numpy.asarray(Ind[:, 1], dtype=numpy.intp)
                ok = (v1 < len(arr)) & (v2 < len(arr))
                v1, v2 = v1[ok], v2[ok]
                close = numpy.linalg.norm(base[v1] - base[v2], axis=1) < thr
                v1, v2 = v1[close], v2[close]
                mid = 0.5 * (base[v1] + base[v2])
                arr[v1] = mid
                arr[v2] = mid

    except Exception as e:
        console.write(f"post-snap closure failed: {e}")


def _validate_curve_mapping(display_name, map_indices, vert, spline_meta):
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


def _gap_fill_poses(obj, n_verts):
    """Compute the per-frame poses used to fill frames before the first
    real simulation frame arrives.

    Returns ``(rest_co, sd_cache_local)``:

    - ``rest_co``: ``(n_verts, 3)`` fallback pose. The object's rest cage,
      replaced by the deform-evaluated pose (Geometry Nodes, Armature, ...)
      when the stack deforms vertices, and by the curve rest CVs for a
      CURVE. The MESH_CACHE modifier is excluded from the eval so we don't
      read prior solver output back in.
    - ``sd_cache_local``: ``(n_frames, n_verts, 3)`` in object-local space
      for a Case-3 STATIC collider that carries a captured-deformation
      cache, else ``None``. When present, gap frame ``i`` should use
      ``sd_cache_local[i]`` (depsgraph-baked pose) in preference to
      ``rest_co``.
    """
    if obj.type == "CURVE":
        from .curve_rod import get_curve_rest_cvs
        return get_curve_rest_cvs(obj), None

    rest_co = numpy.empty(n_verts * 3, dtype=numpy.float64)
    obj.data.vertices.foreach_get("co", rest_co)
    rest_co = rest_co.reshape(n_verts, 3)
    # A deform-only modifier stack (Geometry Nodes, Armature, ...) makes
    # the visible pose diverge from the rest cage. Gap-fill from the
    # deform-evaluated pose so frames before the first sim arrival (notably
    # PC2 frame 0 / scene frame 1) match the shape the solver started from,
    # instead of showing the flat rest mesh for one frame.
    from .utils import (
        eval_deform_local_positions,
        has_deforming_modifier_stack,
    )
    if has_deforming_modifier_stack(obj):
        from .pc2 import MODIFIER_NAME
        deform_co = eval_deform_local_positions(
            obj, exclude_modifier_name=MODIFIER_NAME,
        )
        if deform_co is not None and len(deform_co) == n_verts:
            rest_co = deform_co.astype(numpy.float64)
    # Case 3 STATIC: gap-fill from the captured-deformation cache instead
    # of the undeformed rest mesh, so frames before the first sim arrival
    # still show the depsgraph-baked pose at each gap frame.
    sd_cache_local = None
    if has_static_deform_animation(obj):
        sd_cache = get_static_deform_cache(obj)
        if sd_cache is not None and sd_cache.shape[1] == n_verts:
            inv = numpy.array(inv_world_matrix(obj), dtype=numpy.float64)
            sd_world = sd_cache.astype(numpy.float64)
            sd_cache_local = sd_world @ inv[:3, :3].T + inv[:3, 3]
    return rest_co, sd_cache_local


def _append_leading_gap_frames(pc2_path, frame_idx, n_verts, obj):
    """Append gap frames covering PC2 indices ``[0, frame_idx)`` so the
    next appended real frame lands at PC2 index ``frame_idx``.

    Uses the Case-3 captured-deformation pose per frame when available,
    falling back to the (deform-evaluated) rest pose. Shared by the
    file-create path and the recovery path that finds an existing but
    0-frame (header-only) PC2 file on disk.
    """
    if frame_idx <= 0:
        return
    rest_co, sd_cache_local = _gap_fill_poses(obj, n_verts)
    for gap_i in range(frame_idx):
        if sd_cache_local is not None and gap_i < sd_cache_local.shape[0]:
            append_pc2_frame(pc2_path, sd_cache_local[gap_i], n_verts)
        else:
            append_pc2_frame(pc2_path, rest_co, n_verts)


def _write_mesh_frame_to_pc2(obj, map_vert, blender_frame, n_verts_override=None,
                              place_after_deformers=False):
    """Write a single frame to the object's PC2 file.

    Works for both mesh objects (vertex positions) and curve objects
    (CV positions in Blender's modifier layout).

    The PC2 file is keyed on the object's UUID so it survives renames;
    ``object_pc2_key`` migrates a legacy ``{obj.name}.pc2`` to the new
    location the first time it's accessed.

    Creates the file + MESH_CACHE modifier on the first arriving frame
    (regardless of frame index).  Fills gap frames when frames are
    skipped during live simulation preview. For STATIC meshes with a
    captured-deformation cache, the gap-fill draws from the cache so
    pre-real-frame display matches the depsgraph-baked pose at each
    gap frame, not the undeformed rest mesh.
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

    # An existing file whose header reports 0 frames (header-only: an
    # interrupted write between create and the first append, a disk-full
    # error mid-sequence, or external truncation) has no frame 0 to
    # duplicate, so it cannot go through the fill_gap_frames path below.
    # Treat it like a missing file and re-run the create/gap-fill flow so
    # the appended real frame still lands at PC2 index ``frame_idx``.
    file_present = os.path.exists(pc2_path)
    if file_present:
        try:
            file_present = read_pc2_frame_count(pc2_path) >= 1
        except Exception:
            file_present = False

    if not file_present:
        # First frame to arrive (or a 0-frame file we discard): create
        # PC2, gap-fill up to this frame, and write the real data.
        os.makedirs(os.path.dirname(pc2_path), exist_ok=True)
        create_pc2_file(pc2_path, n_verts, start=0.0, sampling=1.0)
        # Fill gap frames [0, frame_idx).
        _append_leading_gap_frames(pc2_path, frame_idx, n_verts, obj)
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
                setup_mesh_cache_modifier(
                    obj, pc2_path, frame_start=1.0,
                    place_after_deformers=place_after_deformers,
                )
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

# Per-tick cap on apply work during a "Fetch All" batch. The FramePump
# modal fires every 0.1s; processing one frame per tick lets the main
# thread yield between frames so tag_redraw can paint and the progress
# bar advances visibly. Without a cap the worker thread refills the
# queue as fast as the consumer drains it and the loop never exits,
# blocking redraws for the entire fetch. Live sim (total==0) bypasses
# this cap to drain everything inline and avoid viewport flicker.
_BATCH_FRAMES_PER_TICK = 1


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
                        target_objects, world_inv_by_uuid,
                        curve_fit_cache=None):
    """Process one simulation frame: write vertex/CV data to PC2 files.

    Both mesh and curve objects are written to PC2 without calling
    frame_set() here — the caller has already done so for frame N when
    any target needs a per-frame matrix. World-to-local conversion picks
    the per-frame inverse from ``world_inv_by_uuid`` for animated STATIC
    targets, falling back to the target's pre-snapshotted
    ``constant_inv`` for everyone else.

    ``curve_fit_cache`` is an optional ``{uuid: (cache_list, params_data)}``
    dict pre-populated by :func:`apply_animation` so each curve's
    pseudo-inverse is computed once per fetch session instead of once per
    frame. The fit matrix only depends on the spline parameterisation
    (segment indices, ``t``, weights, cyclic flag), which is frame-
    invariant.
    """
    blender_frame = n + 1

    from .curve_rod import apply_fit_cached, build_fit_cache

    # Phase 1: reconstruct every object's deformed positions for this frame,
    # carried in WORLD space so the post-snap closure can move stitched
    # vertices across objects in one common frame. Solid surfaces are
    # reconstructed in LOCAL space first (their embedding normal coef is in
    # local units) then converted to world; the world->local matrix is
    # reapplied at write time.
    world_by_uuid = {}
    records = []
    for target in target_objects:
        uid = target.uuid
        obj = target.obj
        map = target.map
        if target.needs_per_frame:
            mat = world_inv_by_uuid[uid]
        else:
            mat = target.constant_inv

        if obj.type == "CURVE":
            cached = curve_fit_cache.get(uid) if curve_fit_cache is not None else None
            if cached is None:
                cached = build_fit_cache(obj)
                if curve_fit_cache is not None:
                    curve_fit_cache[uid] = cached
            cache_list, params_data = cached
            if not params_data.get("splines"):
                continue
            _validate_curve_mapping(uid, map, vert, params_data["splines"])
            # Rod samples are already world-space solver output.
            world_sim = numpy.array(vert[map], dtype=numpy.float64)
            world_by_uuid[uid] = world_sim
            records.append({
                "kind": "curve", "obj": obj, "mat": mat, "world": world_sim,
                "cache_list": cache_list, "params": params_data,
            })
            continue

        # STATIC colliders (and any cloth with a deforming modifier stack)
        # need their MESH_CACHE after the upstream deformers so PC2 wins.
        place_after_deformers = _needs_after_deformers(target.object_type, obj)

        # --- Mesh path ---
        surface_map = surface_map_by_uuid.get(uid)
        _validate_mesh_mapping(uid, obj, map, surface_map, vert)
        if surface_map is not None:
            # Frame-embedding reconstruction p' = x0' + c1*b1' + c2*b2' + c3*n.
            # Done in LOCAL space (corners converted via mat first) because c3
            # carries local-space length units; mixing it with world-space
            # corners would inject a scale-dependent rest-pose offset.
            tri_indices_arr, coefs_arr, surf_tri_arr = surface_map
            n_verts = len(obj.data.vertices)
            ti = numpy.asarray(tri_indices_arr[:n_verts])
            c = numpy.asarray(coefs_arr[:n_verts], dtype=numpy.float64)
            tris = numpy.asarray(surf_tri_arr)[ti]
            v0 = _apply_matrix_np(mat, vert[map[tris[:, 0]]])
            v1 = _apply_matrix_np(mat, vert[map[tris[:, 1]]])
            v2 = _apply_matrix_np(mat, vert[map[tris[:, 2]]])
            b1 = v1 - v0
            b2 = v2 - v0
            nrm = numpy.cross(b1, b2)
            n_sq = numpy.einsum("ij,ij->i", nrm, nrm)
            # Guard against deformed degenerate triangles: drop the normal
            # term instead of dividing by ~0. Matches the kernel's fallback.
            safe = n_sq > 1e-20
            inv_nlen = numpy.zeros_like(n_sq)
            inv_nlen[safe] = 1.0 / numpy.sqrt(n_sq[safe])
            n_hat = nrm * inv_nlen[:, None]
            map_vert_local = (
                v0 + c[:, 0:1] * b1 + c[:, 1:2] * b2 + c[:, 2:3] * n_hat
            )
            mat_to_world = numpy.linalg.inv(numpy.array(mat, dtype=numpy.float64))
            world_vert = _apply_matrix_np(mat_to_world, map_vert_local)
        else:
            n_verts = len(obj.data.vertices)
            world_vert = numpy.array(vert[map[:n_verts]], dtype=numpy.float64)
        world_by_uuid[uid] = world_vert
        records.append({
            "kind": "mesh", "obj": obj, "mat": mat, "world": world_vert,
            "after": place_after_deformers,
        })

    # Phase 2: snap every stitched source vertex exactly onto its target
    # (world space), across all object types, when the global toggle is on.
    if get_addon_data(context.scene).state.post_snap_exactly:
        _apply_post_snap_closure(context, world_by_uuid)

    # Phase 3: convert each object's world positions back to local, write PC2.
    for rec in records:
        obj = rec["obj"]
        mat = rec["mat"]
        if rec["kind"] == "curve":
            sim_pos = _apply_matrix_np(mat, rec["world"])
            params_data = rec["params"]
            cache_list = rec["cache_list"]
            all_cvs = []
            vi_offset = 0
            for si, spline_meta in enumerate(params_data["splines"]):
                if si >= len(obj.data.splines):
                    continue
                n_sp_verts = len(spline_meta["params"])
                sp_sim = sim_pos[vi_offset : vi_offset + n_sp_verts]
                cvs = apply_fit_cached(sp_sim, cache_list[si])
                all_cvs.append(cvs)
                vi_offset += n_sp_verts
            if all_cvs:
                curve_cvs = numpy.concatenate(all_cvs, axis=0)
                n_cvs = _get_curve_cv_count(obj)
                _write_mesh_frame_to_pc2(obj, curve_cvs[:n_cvs], blender_frame,
                                         n_verts_override=n_cvs)
        else:
            map_vert = _apply_matrix_np(mat, rec["world"])
            _write_mesh_frame_to_pc2(obj, map_vert, blender_frame,
                                     place_after_deformers=rec["after"])

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
            for assigned in g.assigned_objects:
                if not assigned.included:
                    continue
                obj = resolve_assigned(assigned)
                if obj is None or obj.type != "MESH":
                    continue
                # Read-only key: this timer fires every 0.1s, and the
                # side-effecting variant runs an O(N) duplicate scan per
                # object (O(N^2) across the scene). An object that owns a
                # cache already has a UUID, so an empty key means "no
                # cache" and there is nothing to heal.
                key = object_pc2_key_readonly(obj)
                if not key:
                    continue
                pc2 = get_pc2_path(key)
                if not os.path.exists(pc2):
                    continue
                # Decide whether the modifier needs (re)binding BEFORE
                # touching the PC2 file. In steady state every modifier is
                # already healthy, so this skips the per-object PC2 header
                # read (1.6k file reads/tick on a large scene) that only
                # exists to validate vertex count before a rebind.
                mod = obj.modifiers.get("ContactSolverCache")
                needs_setup = (
                    mod is None
                    or mod.cache_format != "PC2"
                    or not mod.filepath
                )
                if not needs_setup:
                    continue
                try:
                    if read_pc2_n_verts(pc2) != len(obj.data.vertices):
                        continue
                except Exception as e:
                    # Log once per (pc2, error-kind) so a broken file doesn't
                    # spam the console on every heal tick.
                    log_key = (pc2, type(e).__name__)
                    if log_key not in _heal_logged:
                        _heal_logged.add(log_key)
                        from ..models.console import console
                        console.write(
                            f"[heal_mesh_caches] skipping {obj.name}: could "
                            f"not read {pc2}: {e}"
                        )
                    continue
                try:
                    setup_mesh_cache_modifier(
                        obj, pc2, frame_start=1.0,
                        place_after_deformers=_needs_after_deformers(
                            g.object_type, obj
                        ),
                    )
                except Exception:
                    # ID-write may be briefly blocked; next tick retries.
                    pass
    except Exception:
        pass


def apply_animation():
    """Drain queued animation frames, writing vertex/CV data to PC2.

    One call handles both live simulation (``total==0``) and batch fetch
    (``total>0``):

    - During batch fetch: apply at most ``_BATCH_FRAMES_PER_TICK`` frames
      then yield, so the FramePump modal can return PASS_THROUGH and the
      progress bar repaint between ticks. Without this cap the worker
      thread refills the queue as fast as the consumer drains it and
      the main thread stays inside this function for the entire fetch,
      blocking redraws. Dispatches
      ``ProgressUpdated(0.5 + 0.5*applied/total)`` so the bar fills
      0.5 → 1.0 across ticks, and ``AllFramesApplied`` on full drain.
    - During live sim: drain everything in one call to avoid viewport
      flicker; no progress events dispatched (frames flow as they arrive).

    Must be driven from PPF_OT_FramePump.modal (a modal-operator timer
    context) — Blender 5.x denies the State / modifier / scene.frame_*
    ID writes this performs when called from bpy.app.timers callbacks.
    """
    context = bpy.context
    com = communicator
    state = get_addon_data(context.scene).state
    # Snapshotted so a mid-fetch error can return the user to their
    # original frame instead of stranding the playhead wherever the
    # exception landed. Success leaves frame_current at the latest
    # written frame (existing intentional behavior for live preview).
    original_frame = context.scene.frame_current

    try:
        target_objects = None
        # Per-fetch-session cache of {uuid: (per_spline_cache_list, params)}.
        # build_fit_cache runs once per curve and is reused across every
        # frame in the loop below.
        curve_fit_cache = {}
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
        applied_this_tick = 0
        any_per_frame = False

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
                            needs_per_frame = (
                                group.object_type == "STATIC"
                                and obj.type == "MESH"
                                and _static_needs_per_frame_matrix(obj)
                            )
                            constant_inv = (
                                None if needs_per_frame
                                else numpy.array(
                                    inv_world_matrix(obj), dtype=numpy.float64,
                                )
                            )
                            target_objects.append(_ApplyTarget(
                                uuid=assigned.uuid,
                                obj=obj,
                                map=map_by_uuid[assigned.uuid],
                                object_type=group.object_type,
                                needs_per_frame=needs_per_frame,
                                constant_inv=constant_inv,
                            ))
                any_per_frame = any(t.needs_per_frame for t in target_objects)

            # For STATIC targets whose matrix_world varies per frame, we
            # must evaluate Blender's depsgraph at frame N to read the
            # true playback matrix_world before computing PC2_local. This
            # is the cost of letting PC2 + MESH_CACHE display the
            # simulator's soft-projected static collider positions
            # against the user's preserved fcurves/parents/constraints
            # without drift.
            world_inv_by_uuid = {}
            if any_per_frame:
                context.scene.frame_set(n + 1)
                for t in target_objects:
                    if t.needs_per_frame:
                        world_inv_by_uuid[t.uuid] = numpy.array(
                            inv_world_matrix(t.obj), dtype=numpy.float64,
                        )

            bf = _apply_single_frame(
                context, n, vert, map_by_uuid, surface_map_by_uuid,
                target_objects, world_inv_by_uuid,
                curve_fit_cache=curve_fit_cache,
            )
            max_blender_frame = max(max_blender_frame, bf)
            state.add_fetched_frame(n)
            applied_this_tick += 1
            # Yield the main thread between batches during a Fetch All
            # so the progress bar can repaint. Live sim (total==0) keeps
            # draining to avoid viewport flicker.
            if total > 0 and applied_this_tick >= _BATCH_FRAMES_PER_TICK:
                break

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
            # Rods play back via a frame_change_post handler, not MESH_CACHE.
            # A per-frame frame_set earlier in this tick (STATIC per-frame
            # matrices) can run that handler before this tick's curve cache
            # was loaded, leaving the rod a frame behind the meshes during
            # live fetch. Force a fresh curve apply at the final frame so
            # rods stay in lockstep.
            from .pc2 import refresh_curves_at_current_frame
            refresh_curves_at_current_frame()

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
        # Restore the playhead so the user isn't stranded on whatever
        # frame the per-frame scene.frame_set last set before the error.
        try:
            context.scene.frame_set(original_frame)
        except Exception:
            pass
        console.write(f"Error applying animation frame: {e}")
        from .events import FetchFailed
        engine.dispatch(FetchFailed(reason=f"apply: {e}"))
