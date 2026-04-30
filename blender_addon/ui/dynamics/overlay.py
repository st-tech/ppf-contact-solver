# File: overlay.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import bpy  # pyright: ignore
import gpu  # pyright: ignore

from gpu_extras.batch import batch_for_shader  # pyright: ignore
from mathutils import Vector  # pyright: ignore

from ..state import iterate_active_object_groups
from ...models.groups import get_addon_data

from .overlay_geometry import (
    DirectionPreviewManager,
    _build_collider_batches,
    _build_operation_batches,
    _build_pin_data,
    _build_rod_batches,
    _build_snap_batches,
    _build_violation_batches,
    _build_velocity_arrow_batches,
    _resolve_scene_dyn_params,
)
from .overlay_labels import draw_overlay_labels

# Global overlay handler
overlay_handler = None

# Module-level cache for overlay draw data
_overlay_cache = {
    "version": -1,
    "frame": -1,
    # Keys used by view-distance-dependent builders. Stored separately so a
    # zoom-only change only rebuilds the view-scaled batches, not the
    # scene-topology batches.
    "view_distance": 0.0,
    "view_version": -1,
    "view_frame": -1,
    "rod_batches": [],
    "snap_batches": [],
    "snap_points": [],
    "pin_data": [],
    "direction_labels": [],
    "direction_batches": [],
    "op_batches": [],
    "op_labels": [],
    "collider_batches": [],
    "velocity_batches": [],
    "violation_batches": [],
    "violation_labels": [],
    "violation_version": -1,
}

# Zoom delta that triggers rebuild of view-scaled batches.
_VIEW_DISTANCE_TOL = 0.05

_direction_manager = DirectionPreviewManager()


def draw_overlay_callback():
    """Draw overlay for ROD objects and pin vertices"""
    context = bpy.context
    scene = context.scene
    if context.screen is None:
        return
    view_3d = None

    for area in context.screen.areas:
        if area.type == "VIEW_3D":
            view_3d = area
            break

    if not view_3d:
        return

    try:
        vis_state = get_addon_data(scene).state
    except Exception:
        vis_state = None
    hide_pins = bool(getattr(vis_state, "hide_pins", False))
    hide_arrows = bool(getattr(vis_state, "hide_arrows", False))
    hide_overlay_colors = bool(getattr(vis_state, "hide_overlay_colors", False))
    hide_snaps = bool(getattr(vis_state, "hide_snaps", False))
    hide_pin_operations = bool(getattr(vis_state, "hide_pin_operations", False))

    # Ensure object color mode is active when groups have overlay colors
    has_overlay = (not hide_overlay_colors) and any(
        group.show_overlay_color
        for group in iterate_active_object_groups(scene)
    )
    if has_overlay:
        space = view_3d.spaces.active
        if hasattr(space, "shading"):
            shading = space.shading
            if shading.type == "SOLID" and shading.color_type != "OBJECT":
                shading.color_type = "OBJECT"

    # Get matrices
    region = view_3d.spaces.active.region_3d

    # Get view and projection matrices
    view_matrix = region.view_matrix
    projection_matrix = region.window_matrix

    # --- Cache invalidation ---
    try:
        version = get_addon_data(scene).state.overlay_version
    except Exception:
        version = 0
    frame = scene.frame_current
    view_distance = region.view_distance
    cached_view_distance = _overlay_cache["view_distance"]
    rod_needs_rebuild = (
        version != _overlay_cache["version"]
        or frame != _overlay_cache["frame"]
    )
    # View-scaled batches also need to refresh on zoom.
    view_needs_rebuild = (
        rod_needs_rebuild
        or version != _overlay_cache["view_version"]
        or frame != _overlay_cache["view_frame"]
        or cached_view_distance <= 0.0
        or abs(view_distance - cached_view_distance)
           > _VIEW_DISTANCE_TOL * cached_view_distance
    )
    if rod_needs_rebuild:
        depsgraph = context.evaluated_depsgraph_get()
        rebuilt_rod = False
        rebuilt_pin = False
        rebuilt_snap = False
        try:
            _overlay_cache["rod_batches"] = _build_rod_batches(scene, depsgraph)
            rebuilt_rod = True
        except Exception as exc:
            print(f"[ppf] _build_rod_batches failed: {exc!r}")
        try:
            _overlay_cache["pin_data"] = _build_pin_data(scene, depsgraph)
            rebuilt_pin = True
        except Exception as exc:
            print(f"[ppf] _build_pin_data failed: {exc!r}")
        # Snap batches: reads PC2 files directly (no depsgraph), independent
        # of view_distance — gated on the same scene-topology key.
        try:
            (
                _overlay_cache["snap_batches"],
                _overlay_cache["snap_points"],
            ) = _build_snap_batches(scene)
            rebuilt_snap = True
        except Exception as exc:
            print(f"[ppf] _build_snap_batches failed: {exc!r}")
        # Only promote the cache key if all builds succeeded; otherwise we
        # want to retry next frame rather than freezing a stale/empty result.
        if rebuilt_rod and rebuilt_pin and rebuilt_snap:
            _overlay_cache["version"] = version
            _overlay_cache["frame"] = frame

    rod_batches = _overlay_cache["rod_batches"]
    snap_batches = _overlay_cache["snap_batches"]
    snap_points = _overlay_cache["snap_points"]

    if rod_batches:
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("LESS_EQUAL")

        for batch, color in rod_batches:
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)

        gpu.state.blend_set("NONE")

    if snap_batches and not hide_snaps:
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("LESS_EQUAL")

        for batch, color in snap_batches:
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)

        gpu.state.blend_set("NONE")

    if snap_points and not hide_snaps:
        # Group by (size, color) so a scene with many snap points issues
        # one GPU batch per unique style rather than one batch per point.
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(False)

        grouped: dict[tuple, list] = {}
        for vertex, size, color in snap_points:
            grouped.setdefault((float(size), tuple(color)), []).append(vertex)
        for (size, color), verts in grouped.items():
            gpu.state.point_size_set(size)
            batch = batch_for_shader(shader, "POINTS", {"pos": verts})
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)

        gpu.state.point_size_set(1.0)
        gpu.state.depth_mask_set(True)
        gpu.state.blend_set("NONE")

    pin_data = _overlay_cache["pin_data"]

    if pin_data and not hide_pins:
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(False)

        grouped: dict[tuple, list] = {}
        for vertex, size, color in pin_data:
            # Clip only points behind the camera (ndc.w <= 0 in OpenGL convention).
            # The GPU handles frustum clipping for points that are in front but
            # outside the XY viewport, so we intentionally do NOT reject on
            # |ndc.xy| > 1 or on ndc.z — Blender's window_matrix is OpenGL-style
            # with ndc.z in [-1, 1], and a [0, 1] range would silently drop any
            # pin in the front half of the frustum.
            clip = (
                projection_matrix
                @ view_matrix
                @ Vector((vertex[0], vertex[1], vertex[2], 1.0))
            )
            if clip.w <= 0:
                continue
            grouped.setdefault((float(size), tuple(color)), []).append(vertex)

        for (size, color), verts in grouped.items():
            gpu.state.point_size_set(size)
            batch = batch_for_shader(shader, "POINTS", {"pos": verts})
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)

        gpu.state.point_size_set(1.0)
        gpu.state.depth_mask_set(True)
        gpu.state.blend_set("NONE")

    # --- Direction previews (gravity, wind, etc.) ---
    try:
        if view_needs_rebuild:
            state = get_addon_data(scene).state
            _direction_manager.clear()
            current_frame = scene.frame_current

            gravity, wind_dir, wind_strength = _resolve_scene_dyn_params(
                state, current_frame,
            )

            if state.preview_gravity_direction and not hide_arrows:
                vec = Vector(gravity)
                if vec.length > 1e-6:
                    _direction_manager.add(
                        "Gravity", vec.normalized(), (0.3, 0.5, 1.0),
                        label="Gravity", strength=vec.length, unit="m/s\u00b2",
                        raw_direction=vec.normalized(),
                    )
            if state.preview_wind_direction and not hide_arrows:
                vec = Vector(wind_dir)
                if vec.length > 1e-6 and wind_strength > 1e-6:
                    _direction_manager.add(
                        "Wind", vec.normalized(), (0.3, 1.0, 0.5),
                        label="Wind", strength=wind_strength, unit="m/s",
                        raw_direction=vec.normalized(),
                    )
            # Scale sphere to a fixed fraction of the view distance
            view_radius = view_distance * 0.08
            _overlay_cache["direction_batches"] = (
                _direction_manager.build_batches(radius=view_radius) or []
            )
        preview_batches = _overlay_cache["direction_batches"]
        if preview_batches:
            shader = gpu.shader.from_builtin("UNIFORM_COLOR")
            gpu.state.blend_set("ALPHA")
            gpu.state.depth_test_set("NONE")
            gpu.state.depth_mask_set(False)

            for data in preview_batches:
                # Filled sphere background (semi-transparent)
                shader.bind()
                shader.uniform_float("color", data["fill_color"])
                data["fill_batch"].draw(shader)

                # Sphere wireframe (triangle-based thickness)
                shader.bind()
                shader.uniform_float("color", data["sphere_color"])
                data["sphere_batch"].draw(shader)

                # Arrow shaft (triangle-based thickness)
                shader.bind()
                shader.uniform_float("color", data["arrow_color"])
                data["shaft_batch"].draw(shader)

                # Arrow cone head
                shader.bind()
                shader.uniform_float("color", data["arrow_color"])
                data["cone_batch"].draw(shader)

            gpu.state.depth_test_set("LESS_EQUAL")
            gpu.state.depth_mask_set(True)
            gpu.state.blend_set("NONE")

        # Store label data for 2D text handler
        _overlay_cache["direction_labels"] = preview_batches or []

    except Exception:
        _overlay_cache["direction_labels"] = []

    # --- Operation overlays (spin circle, move_by/scale trajectories) ---
    if view_needs_rebuild:
        try:
            if hide_pin_operations:
                _overlay_cache["op_batches"] = []
                _overlay_cache["op_labels"] = []
            else:
                depsgraph = context.evaluated_depsgraph_get()
                op_batches, op_labels = _build_operation_batches(
                    scene, depsgraph, view_distance,
                )
                _overlay_cache["op_batches"] = op_batches
                _overlay_cache["op_labels"] = op_labels
        except Exception:
            _overlay_cache["op_batches"] = []
            _overlay_cache["op_labels"] = []

    op_batches = _overlay_cache["op_batches"]
    if op_batches:
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("NONE")
        gpu.state.depth_mask_set(False)
        gpu.state.point_size_set(8.0)

        for batch, color in op_batches:
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)

        gpu.state.point_size_set(1.0)
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(True)
        gpu.state.blend_set("NONE")

    # --- Invisible collider previews ---
    if view_needs_rebuild:
        try:
            _overlay_cache["collider_batches"] = _build_collider_batches(view_distance)
        except Exception:
            _overlay_cache["collider_batches"] = []

    collider_batches = _overlay_cache["collider_batches"]
    if collider_batches:
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(False)

        for batch, color in collider_batches:
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)

        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(True)
        gpu.state.blend_set("NONE")

    # --- Violation highlights ---
    try:
        from ...core.facade import communicator as _com
        v_list = _com.info.violations
        v_ver = id(v_list) if v_list else -1
        if v_ver != _overlay_cache["violation_version"]:
            _overlay_cache["violation_version"] = v_ver
            if v_list:
                depsgraph = context.evaluated_depsgraph_get()
                vb, vl = _build_violation_batches(scene, depsgraph, v_list)
                _overlay_cache["violation_batches"] = vb
                _overlay_cache["violation_labels"] = vl
            else:
                _overlay_cache["violation_batches"] = []
                _overlay_cache["violation_labels"] = []
    except Exception:
        pass

    violation_batches = _overlay_cache["violation_batches"]
    if violation_batches:
        shader = gpu.shader.from_builtin("UNIFORM_COLOR")
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("NONE")
        gpu.state.depth_mask_set(False)

        for batch, prim_type, color in violation_batches:
            if prim_type == "POINTS":
                gpu.state.point_size_set(16.0)
            shader.bind()
            shader.uniform_float("color", color)
            batch.draw(shader)

        gpu.state.point_size_set(1.0)
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.depth_mask_set(True)
        gpu.state.blend_set("NONE")

    # --- Per-object initial velocity arrows ---
    if view_needs_rebuild:
        try:
            if hide_arrows:
                vel_batches, vel_labels = [], []
            else:
                vel_batches, vel_labels = _build_velocity_arrow_batches(
                    scene, view_distance,
                )
            _overlay_cache["velocity_batches"] = vel_batches
            _overlay_cache["velocity_labels"] = vel_labels
        except Exception:
            _overlay_cache["velocity_batches"] = []
            _overlay_cache["velocity_labels"] = []
        # Promote the view-cache key last, so a failed builder retries next
        # frame rather than freezing stale/empty cached batches.
        _overlay_cache["view_distance"] = view_distance
        _overlay_cache["view_version"] = version
        _overlay_cache["view_frame"] = frame
    try:
        vel_batches = _overlay_cache["velocity_batches"]
        if vel_batches:
            shader = gpu.shader.from_builtin("UNIFORM_COLOR")
            gpu.state.blend_set("ALPHA")
            gpu.state.depth_test_set("NONE")
            gpu.state.depth_mask_set(False)
            for batch, color in vel_batches:
                shader.bind()
                shader.uniform_float("color", color)
                batch.draw(shader)
            gpu.state.depth_test_set("LESS_EQUAL")
            gpu.state.depth_mask_set(True)
            gpu.state.blend_set("NONE")
    except Exception:
        _overlay_cache["velocity_labels"] = []


def _draw_overlay_labels():
    """Draw text labels for direction previews, operations, and velocity (POST_PIXEL handler)."""
    draw_overlay_labels(_overlay_cache)


# Second handler for 2D text labels
_text_handler = None


def register_overlay():
    """Register the overlay drawing handlers"""
    global overlay_handler, _text_handler
    if overlay_handler is None:
        overlay_handler = bpy.types.SpaceView3D.draw_handler_add(
            draw_overlay_callback, (), "WINDOW", "POST_VIEW"
        )
    if _text_handler is None:
        _text_handler = bpy.types.SpaceView3D.draw_handler_add(
            _draw_overlay_labels, (), "WINDOW", "POST_PIXEL"
        )


def unregister_overlay():
    """Unregister the overlay drawing handlers"""
    global overlay_handler, _text_handler
    if overlay_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(overlay_handler, "WINDOW")
        overlay_handler = None
    if _text_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_text_handler, "WINDOW")
        _text_handler = None


def apply_object_overlays():
    """Update object colors and invalidate overlay cache."""
    from ...models.groups import invalidate_overlays

    scene = bpy.context.scene
    try:
        hide_overlay_colors = bool(get_addon_data(scene).state.hide_overlay_colors)
    except Exception:
        hide_overlay_colors = False
    # Reset all object colors to default
    for obj in bpy.data.objects:
        if obj.type in ("MESH", "CURVE"):
            obj.color = (1.0, 1.0, 1.0, 1.0)
    # Apply group colors
    if not hide_overlay_colors:
        for group in iterate_active_object_groups(scene):
            if group.show_overlay_color:
                for obj_ref in group.assigned_objects:
                    if not obj_ref.included:
                        continue
                    from ...core.uuid_registry import resolve_assigned
                    obj = resolve_assigned(obj_ref)
                    if obj and obj.type in ("MESH", "CURVE"):
                        obj.color = group.color
    # Ensure solid shading uses object colors
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                space = area.spaces.active
                if hasattr(space, "shading"):
                    if space.shading.type == "SOLID":
                        space.shading.color_type = "OBJECT"

    invalidate_overlays()


def _on_undo_redo(scene):
    apply_object_overlays()


def register():
    register_overlay()
    # Drop any stale copies left by a previous reload (function identity
    # changes across reloads so a plain `in` check wouldn't find them).
    for lst in (bpy.app.handlers.undo_post, bpy.app.handlers.redo_post):
        for h in list(lst):
            if getattr(h, "__name__", "") == "_on_undo_redo":
                lst.remove(h)
    bpy.app.handlers.undo_post.append(_on_undo_redo)
    bpy.app.handlers.redo_post.append(_on_undo_redo)


def unregister():
    for h in list(bpy.app.handlers.redo_post):
        if getattr(h, "__name__", "") == "_on_undo_redo":
            bpy.app.handlers.redo_post.remove(h)
    for h in list(bpy.app.handlers.undo_post):
        if getattr(h, "__name__", "") == "_on_undo_redo":
            bpy.app.handlers.undo_post.remove(h)
    unregister_overlay()
