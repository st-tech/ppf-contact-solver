# File: panels.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import os

import bpy  # pyright: ignore

from bpy.types import Panel  # pyright: ignore

from ...core.utils import get_category_name
from ...models.groups import get_addon_data, has_addon_data, pair_supports_cross_stitch
from ..state import decode_vertex_group_identifier, iterate_active_object_groups
from .utils import get_assigned_by_selection_uuid

# Icons by ObjectGroup.object_type — hoisted out of DYNAMICS_PT_Groups.draw
# so the dict isn't rebuilt for every group on every redraw.
_GROUP_TYPE_ICONS = {
    "SOLID": "MESH_CUBE",
    "SHELL": "SURFACE_DATA",
    "ROD": "CURVE_DATA",
    "STATIC": "FREEZE",
}


# Cache `os.path.isfile(bpy.path.abspath(path))` per unique path string.
# The panel draw consults this up to 3 paths per group per redraw (scene /
# pin / material profile). We never invalidate — the profile operators
# write to disk and set the path property, and a miss here only delays the
# "profile is present" UI state by one redraw which any user-triggered
# event will flush anyway.
_profile_path_cache: dict[str, bool] = {}


def _profile_path_exists(path: str) -> bool:
    if not path:
        return False
    cached = _profile_path_cache.get(path)
    if cached is None:
        cached = os.path.isfile(bpy.path.abspath(path))
        _profile_path_cache[path] = cached
    return cached


def _draw_velocity_keyframes(param_box, group, actual_index):
    """Draw the per-object velocity keyframes UIList."""
    vel_box = param_box.box()
    row = vel_box.row()
    row.label(text="Velocity Overwrite", icon="FORCE_FORCE")
    assigned = get_assigned_by_selection_uuid(group, "velocity_object_selection")
    copy_sub = row.row(align=True)
    copy_sub.enabled = assigned is not None and len(assigned.velocity_keyframes) > 0
    op = copy_sub.operator("object.copy_velocity_keyframes", text="", icon="COPYDOWN")
    op.group_index = actual_index
    op = row.operator("object.paste_velocity_keyframes", text="", icon="PASTEDOWN")
    op.group_index = actual_index
    row.prop(group, "preview_velocity", text="", icon="HIDE_OFF" if group.preview_velocity else "HIDE_ON")
    row.prop(group, "velocity_object_selection", text="")

    if assigned is None:
        return

    row = vel_box.row()
    row.template_list(
        "OBJECT_UL_VelocityKeyframesList", "",
        assigned, "velocity_keyframes",
        assigned, "velocity_keyframes_index",
        rows=2,
    )
    col = row.column(align=True)
    add_op = col.operator("object.add_velocity_keyframe", icon="ADD", text="")
    add_op.group_index = actual_index
    rm_op = col.operator("object.remove_velocity_keyframe", icon="REMOVE", text="")
    rm_op.group_index = actual_index

    if assigned.velocity_keyframes and 0 <= assigned.velocity_keyframes_index < len(assigned.velocity_keyframes):
        kf = assigned.velocity_keyframes[assigned.velocity_keyframes_index]
        vel_box.prop(kf, "frame")
        vel_box.prop(kf, "direction")
        vel_box.prop(kf, "speed")


def _draw_collision_windows(param_box, group, actual_index):
    """Draw the collision active duration windows UI."""
    cw_box = param_box.box()
    cw_box.prop(group, "use_collision_windows")
    if not group.use_collision_windows:
        return

    row = cw_box.row()
    row.label(text="Active Windows", icon="TIME")
    row.prop(group, "collision_window_object_selection", text="")

    assigned = get_assigned_by_selection_uuid(group, "collision_window_object_selection")
    if assigned is None:
        return

    row = cw_box.row()
    row.template_list(
        "OBJECT_UL_CollisionWindowsList", "",
        assigned, "collision_windows",
        assigned, "collision_windows_index",
        rows=2,
    )
    col = row.column(align=True)
    add_op = col.operator("object.add_collision_window", icon="ADD", text="")
    add_op.group_index = actual_index
    rm_op = col.operator("object.remove_collision_window", icon="REMOVE", text="")
    rm_op.group_index = actual_index

    if assigned.collision_windows and 0 <= assigned.collision_windows_index < len(assigned.collision_windows):
        entry = assigned.collision_windows[assigned.collision_windows_index]
        row = cw_box.row(align=True)
        row.prop(entry, "frame_start")
        row.prop(entry, "frame_end")

    cw_box.label(text="Invisible colliders always active", icon="INFO")
    cw_box.label(text="No entries = contact always active", icon="INFO")


def _draw_static_ops(pin_box, group, actual_index):
    """Draw the per-object move/spin/scale ops UI for a STATIC group.

    The outer list (assigned_objects) is already shown in the group box
    above, so this draws the inner ops list plus per-op editor for
    whichever assigned object is active. Controls stay enabled even if
    the selected object has Blender fcurves — a warning label explains
    that fcurves will take precedence at simulate time.
    """
    idx = group.assigned_objects_index
    if idx < 0 or idx >= len(group.assigned_objects):
        pin_box.label(text="Select an assigned object above", icon="INFO")
        return
    assigned = group.assigned_objects[idx]

    from ...core.uuid_registry import get_object_by_uuid
    from ...core.utils import has_transform_fcurves
    obj = get_object_by_uuid(assigned.uuid) if assigned.uuid else None
    has_fcurves = obj is not None and has_transform_fcurves(obj)

    if obj is None:
        pin_box.label(text="Object unresolved", icon="ERROR")
    elif has_fcurves:
        pin_box.label(
            text="Object has Blender keyframes — these ops will be ignored",
            icon="ERROR",
        )

    row = pin_box.row()
    row.template_list(
        "OBJECT_UL_StaticOpsList", "",
        assigned, "static_ops",
        assigned, "static_ops_index",
        rows=3,
    )
    btn_col = row.column(align=True)
    add_op = btn_col.operator_menu_enum(
        "object.add_static_op", "op_type", icon="ADD", text="",
    )
    add_op.group_index = actual_index
    rm_op = btn_col.operator("object.remove_static_op", icon="REMOVE", text="")
    rm_op.group_index = actual_index
    up_op = btn_col.operator("object.move_static_op", icon="TRIA_UP", text="")
    up_op.group_index = actual_index
    up_op.direction = -1
    down_op = btn_col.operator("object.move_static_op", icon="TRIA_DOWN", text="")
    down_op.group_index = actual_index
    down_op.direction = 1

    # Per-op editor for the active row
    op_idx = assigned.static_ops_index
    if 0 <= op_idx < len(assigned.static_ops):
        op = assigned.static_ops[op_idx]
        editor = pin_box.box()
        row = editor.row(align=True)
        row.prop(op, "frame_start")
        row.prop(op, "frame_end")
        editor.prop(op, "transition")
        if op.op_type == "MOVE_BY":
            editor.prop(op, "delta")
        elif op.op_type == "SPIN":
            editor.prop(op, "spin_axis")
            editor.prop(op, "spin_angular_velocity")
            editor.label(text="Center: object origin", icon="OBJECT_ORIGIN")
        elif op.op_type == "SCALE":
            editor.prop(op, "scale_factor")
            editor.label(text="Center: object origin", icon="OBJECT_ORIGIN")


_FTETWILD_FIELDS = (
    ("edge_length_fac", False),
    ("epsilon", False),
    ("stop_energy", False),
    ("num_opt_iter", False),
    ("optimize", True),
    ("simplify", True),
    ("coarsen", True),
)


def _draw_ftetwild(param_box, group):
    """Draw per-group fTetWild overrides inside an expandable box.

    SOLID groups only. When a sub-override flag is off, tetrahedralize()
    receives no kwarg for that field and fTetWild's default applies, so
    leaving the whole box untouched yields pure defaults.
    """
    ft_box = param_box.box()
    row = ft_box.row()
    row.prop(
        group,
        "show_ftetwild",
        icon="TRIA_DOWN" if group.show_ftetwild else "TRIA_RIGHT",
        emboss=False,
        icon_only=True,
    )
    row.label(text="fTetWild", icon="MESH_ICOSPHERE")
    if not group.show_ftetwild:
        return
    for field, _is_bool in _FTETWILD_FIELDS:
        row = ft_box.row(align=True)
        row.prop(group, f"ftetwild_override_{field}", text="")
        sub = row.row(align=True)
        sub.enabled = getattr(group, f"ftetwild_override_{field}")
        sub.prop(group, f"ftetwild_{field}")


# Per-mesh cache: mesh.as_pointer() -> (topology_key, has_loose_edge).
# topology_key is (n_verts, n_edges, n_polys, n_loops); any add/remove/merge
# that matters to loose-edge detection changes at least one count.
# Runs inside DYNAMICS_PT_Groups.draw() — redraws fire on every mouse move.
_stitch_cache: dict[int, tuple[tuple[int, int, int, int], bool]] = {}


def _mesh_has_loose_edge(mesh) -> bool:
    """Return True if `mesh` has any edge not referenced by a polygon loop.

    Result is cached per-mesh and invalidated whenever vertex/edge/poly/loop
    counts change.
    """
    import numpy as np
    n_verts = len(mesh.vertices)
    n_edges = len(mesh.edges)
    n_polys = len(mesh.polygons)
    n_loops = len(mesh.loops)
    key = (n_verts, n_edges, n_polys, n_loops)
    ptr = mesh.as_pointer()
    cached = _stitch_cache.get(ptr)
    if cached is not None and cached[0] == key:
        return cached[1]

    if n_edges == 0:
        has = False
    elif n_loops == 0:
        has = True
    else:
        loop_edges = np.empty(n_loops, dtype=np.int32)
        mesh.loops.foreach_get("edge_index", loop_edges)
        hits = np.zeros(n_edges, dtype=np.uint8)
        hits[loop_edges] = 1
        has = bool(int(hits.sum()) < n_edges)

    _stitch_cache[ptr] = (key, has)
    return has


def _group_has_stitch(group) -> bool:
    """Check if any assigned object in the group has stitch (loose) edges."""
    from ...core.uuid_registry import resolve_assigned
    for obj_ref in group.assigned_objects:
        if not obj_ref.included:
            continue
        obj = resolve_assigned(obj_ref)
        if not obj or obj.type != "MESH":
            continue
        if _mesh_has_loose_edge(obj.data):
            return True
    return False


def get_active_groups_with_indices(scene):
    """Get active groups with their actual property indices (for object_group_N)."""
    from ..state import N_MAX_GROUPS

    addon_data = get_addon_data(scene)
    for i in range(N_MAX_GROUPS):
        prop_name = f"object_group_{i}"
        group = getattr(addon_data, prop_name, None)
        if group and group.active:
            yield i, group


class MAIN_PT_SceneConfiguration(Panel):
    bl_label = "Scene Configuration"
    bl_idname = "SSH_PT_ObjectGroupsManager"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = get_category_name()

    @classmethod
    def poll(cls, context):
        return has_addon_data(context.scene)

    def draw(self, context):
        layout = self.layout

        params = get_addon_data(context.scene).state

        # Scene param profile
        profile_row = layout.row(align=True)
        if _profile_path_exists(params.scene_profile_path):
            profile_row.prop(params, "scene_profile_selection", text="Profile")
            profile_row.operator(
                "scene.open_scene_profile", text="", icon="FILEBROWSER"
            )
            profile_row.operator(
                "scene.clear_scene_profile", text="", icon="X"
            )
            profile_row.operator(
                "scene.reload_scene_profile", text="", icon="FILE_REFRESH"
            )
            profile_row.operator(
                "scene.save_scene_profile", text="", icon="FILE_TICK"
            )
        else:
            profile_row.operator(
                "scene.open_scene_profile", text="Open Profile", icon="FILEBROWSER"
            )
            profile_row.operator(
                "scene.save_scene_profile", text="", icon="FILE_TICK"
            )

        fps_box = layout.box()
        fps_box.prop(params, "use_frame_rate_in_output")
        if params.use_frame_rate_in_output:
            fps = context.scene.render.fps
            row = fps_box.row()
            row.label(text="FPS:")
            row.label(text=str(fps))
        else:
            fps_box.prop(params, "frame_rate")
        layout.prop(params, "frame_count")
        layout.prop(params, "step_size")
        layout.prop(params, "min_newton_steps")
        layout.prop(params, "air_density")
        layout.prop(params, "air_friction")
        layout.prop(params, "gravity_3d")
        layout.prop(params, "preview_gravity_direction")

        has_shell_type = any(
            group.object_type == "SHELL"
            for group in iterate_active_object_groups(context.scene)
        )
        row = layout.row()
        row.enabled = has_shell_type
        row.prop(params, "inactive_momentum_frames")

        wind_box = layout.box()
        row = wind_box.row()
        row.prop(
            params,
            "show_wind",
            icon="TRIA_DOWN" if params.show_wind else "TRIA_RIGHT",
            emboss=False,
            icon_only=True,
        )
        row.label(text="Wind", icon="FORCE_WIND")
        if params.show_wind:
            wind_box.prop(params, "wind_direction")
            wind_box.prop(params, "preview_wind_direction")
            wind_box.prop(params, "wind_strength")

        advanced_box = layout.box()
        row = advanced_box.row()
        row.prop(
            params,
            "show_advanced_parameters",
            icon="TRIA_DOWN" if params.show_advanced_parameters else "TRIA_RIGHT",
            emboss=False,
            icon_only=True,
        )
        row.label(text="Advanced Params", icon="PREFERENCES")
        if params.show_advanced_parameters:
            advanced_box.prop(params, "contact_nnz")
            advanced_box.prop(params, "vertex_air_damp")
            advanced_box.prop(params, "auto_save")
            if params.auto_save:
                advanced_box.prop(params, "auto_save_interval")
            advanced_box.prop(params, "line_search_max_t")
            advanced_box.prop(params, "constraint_ghat")
            advanced_box.prop(params, "cg_max_iter")
            advanced_box.prop(params, "cg_tol")
            advanced_box.prop(params, "include_face_mass")
            advanced_box.prop(params, "friction_mode")
            advanced_box.prop(params, "disable_contact")

        # Dynamic Scene Parameters
        dyn_box = layout.box()
        dyn_row = dyn_box.row()
        dyn_row.prop(
            params,
            "show_dyn_params",
            icon="TRIA_DOWN" if params.show_dyn_params else "TRIA_RIGHT",
            emboss=False,
            icon_only=True,
        )
        dyn_row.label(text="Dynamic Parameters", icon="TIME")
        if params.show_dyn_params:
            dyn_box.template_list(
                "SCENE_UL_DynParamsList", "",
                params, "dyn_params",
                params, "dyn_params_index",
                rows=2,
            )
            row = dyn_box.row(align=True)
            row.operator_menu_enum(
                "scene.add_dyn_param", "param_type",
                text="Add", icon="ADD",
            )
            rm_row = row.row()
            rm_row.enabled = 0 <= params.dyn_params_index < len(params.dyn_params)
            rm_row.operator("scene.remove_dyn_param", text="Remove", icon="REMOVE")

            dyn_idx = params.dyn_params_index
            if 0 <= dyn_idx < len(params.dyn_params):
                dyn_item = params.dyn_params[dyn_idx]
                dyn_box.separator()
                dyn_box.template_list(
                    "SCENE_UL_DynParamKeyframesList", "",
                    dyn_item, "keyframes",
                    dyn_item, "keyframes_index",
                    rows=2,
                )
                kf_row = dyn_box.row(align=True)
                kf_row.operator(
                    "scene.add_dyn_param_keyframe",
                    text="Add Keyframe", icon="ADD",
                )
                kf_rm = kf_row.row()
                kf_rm.enabled = (
                    0 <= dyn_item.keyframes_index < len(dyn_item.keyframes)
                    and dyn_item.keyframes_index > 0
                )
                kf_rm.operator(
                    "scene.remove_dyn_param_keyframe",
                    text="Remove", icon="REMOVE",
                )

                kf_idx = dyn_item.keyframes_index
                if 0 <= kf_idx < len(dyn_item.keyframes):
                    kf = dyn_item.keyframes[kf_idx]
                    kf_box = dyn_box.box()
                    if kf_idx == 0:
                        kf_box.label(text="Uses global parameter values", icon="INFO")
                    else:
                        kf_box.prop(kf, "frame")
                        kf_box.prop(kf, "use_hold")
                        if not kf.use_hold:
                            if dyn_item.param_type == "GRAVITY":
                                kf_box.prop(kf, "gravity_value")
                            elif dyn_item.param_type == "WIND":
                                kf_box.prop(kf, "wind_direction_value")
                                kf_box.prop(kf, "wind_strength_value")
                            else:
                                kf_box.prop(kf, "scalar_value")

        # Invisible Colliders (enclosed box)
        ic_box = layout.box()
        ic_row = ic_box.row()
        ic_row.prop(
            params,
            "show_invisible_colliders",
            icon="TRIA_DOWN" if params.show_invisible_colliders else "TRIA_RIGHT",
            emboss=False,
            icon_only=True,
        )
        ic_row.label(text="Invisible Colliders", icon="GHOST_ENABLED")
        if params.show_invisible_colliders:
            ic_box.template_list(
                "SCENE_UL_InvisibleCollidersList", "",
                params, "invisible_colliders",
                params, "invisible_colliders_index",
                rows=2,
            )
            row = ic_box.row(align=True)
            row.operator_menu_enum(
                "scene.add_invisible_collider", "collider_type",
                text="Add", icon="ADD",
            )
            rm_row = row.row()
            rm_row.enabled = 0 <= params.invisible_colliders_index < len(params.invisible_colliders)
            rm_row.operator("scene.remove_invisible_collider", text="Remove", icon="REMOVE")

            ic_idx = params.invisible_colliders_index
            if 0 <= ic_idx < len(params.invisible_colliders):
                ic_item = params.invisible_colliders[ic_idx]
                prop_box = ic_box.box()
                prop_box.prop(ic_item, "name", text="Name")
                if ic_item.collider_type == "WALL":
                    prop_box.prop(ic_item, "position")
                    prop_box.prop(ic_item, "normal")
                else:
                    prop_box.prop(ic_item, "position")
                    prop_box.prop(ic_item, "radius")
                    row = prop_box.row()
                    row.prop(ic_item, "invert")
                    row.prop(ic_item, "hemisphere")
                prop_box.prop(ic_item, "contact_gap")
                prop_box.prop(ic_item, "friction")
                prop_box.prop(ic_item, "thickness")
                prop_box.prop(ic_item, "enable_active_duration")
                if ic_item.enable_active_duration:
                    prop_box.prop(ic_item, "active_duration")

                ic_box.separator()
                ic_box.template_list(
                    "SCENE_UL_ColliderKeyframesList", "",
                    ic_item, "keyframes",
                    ic_item, "keyframes_index",
                    rows=2,
                )
                kf_row = ic_box.row(align=True)
                kf_row.operator("scene.add_collider_keyframe", text="Add Keyframe", icon="ADD")
                kf_rm = kf_row.row()
                kf_rm.enabled = (
                    0 <= ic_item.keyframes_index < len(ic_item.keyframes)
                    and ic_item.keyframes_index > 0
                )
                kf_rm.operator("scene.remove_collider_keyframe", text="Remove", icon="REMOVE")

                kf_idx = ic_item.keyframes_index
                if 0 <= kf_idx < len(ic_item.keyframes):
                    kf = ic_item.keyframes[kf_idx]
                    kf_box = ic_box.box()
                    if kf_idx == 0:
                        kf_box.label(text="Uses base properties above", icon="INFO")
                    else:
                        kf_box.prop(kf, "frame")
                        kf_box.prop(kf, "use_hold")
                        if not kf.use_hold:
                            kf_box.prop(kf, "position")
                            if ic_item.collider_type == "SPHERE":
                                kf_box.prop(kf, "radius")


class DYNAMICS_PT_Groups(Panel):
    bl_label = "Dynamics Groups"
    bl_idname = "DYNAMICS_PT_Groups"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = get_category_name()

    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.operator("object.create_group", icon="PLUS")
        row.operator("object.delete_all_groups", icon="TRASH")

        for display_index, (actual_index, group) in enumerate(
            get_active_groups_with_indices(context.scene)
        ):
            box = layout.box()
            # Group header with collapse toggle
            row = box.row()
            # Use custom name if provided, otherwise default to "Group N"
            group_label = (
                group.name if group.name.strip() else f"Group {display_index + 1}"
            )
            group_icon = _GROUP_TYPE_ICONS.get(group.object_type, "OBJECT_DATA")
            row.prop(
                group,
                "show_group",
                icon="TRIA_DOWN" if group.show_group else "TRIA_RIGHT",
                emboss=False,
                icon_only=True,
            )
            row.label(text=group_label, icon=group_icon)
            op = row.operator("object.duplicate_group", text="", icon="DUPLICATE")
            op.group_index = actual_index

            if group.show_group:
                # Group name field and type selection on same row
                row = box.row(align=True)
                row.prop(group, "name", text="Name")
                row.prop(group, "object_type", text="")

                row = box.row(align=True)
                row.prop(group, "show_overlay_color")
                if group.show_overlay_color:
                    row.prop(group, "color", text="")
                box.template_list(
                    "OBJECT_UL_AssignedObjectsList",
                    "",
                    group,
                    "assigned_objects",
                    group,
                    "assigned_objects_index",
                )
                # First row: Add Selected + Remove Object
                row = box.row()
                add_op = row.operator(
                    "object.add_objects_to_group",
                    icon="ADD",
                )
                add_op.group_index = actual_index
                col = row.column()
                col.enabled = (
                    0 <= group.assigned_objects_index < len(group.assigned_objects)
                )
                remove_op = col.operator(
                    "object.remove_object_from_group", icon="REMOVE"
                )
                remove_op.group_index = actual_index

                # Second row: Bake Animation + Bake Single Frame
                row = box.row()
                has_selection = 0 <= group.assigned_objects_index < len(group.assigned_objects)
                has_anim = False
                if has_selection:
                    from ...core.uuid_registry import resolve_assigned as _resolve
                    from ...core.pc2 import has_mesh_cache
                    _sel_obj = _resolve(group.assigned_objects[group.assigned_objects_index])
                    has_anim = _sel_obj is not None and has_mesh_cache(_sel_obj)
                from .bake_ops import is_bake_running as _is_bake_running
                _bake_active = _is_bake_running()
                col = row.column()
                col.enabled = has_anim and not _bake_active
                bake_op = col.operator("object.bake_animation", icon="ACTION")
                bake_op.group_index = actual_index
                col = row.column()
                col.enabled = has_anim and not _bake_active
                bake_frame_op = col.operator("object.bake_single_frame", icon="KEYFRAME")
                bake_frame_op.group_index = actual_index

                # Third row: Delete Group button
                row = box.row()
                delete_op = row.operator("object.delete_group", icon="TRASH")
                delete_op.group_index = actual_index

                pin_box = box.box()
                row = pin_box.row()
                row.prop(
                    group,
                    "show_pin",
                    icon="TRIA_DOWN" if group.show_pin else "TRIA_RIGHT",
                    emboss=False,
                    icon_only=True,
                )
                if group.object_type == "STATIC":
                    row.label(text="Transform", icon="DRIVER")
                else:
                    row.label(text="Pins", icon="PINNED")
                if group.show_pin and group.object_type == "STATIC":
                    _draw_static_ops(pin_box, group, actual_index)
                if group.show_pin and group.object_type != "STATIC":
                    col = pin_box.column()
                    row = col.row()
                    row.prop(group, "pin_vertex_group_items", text="")
                    add_pin_op = row.operator(
                        "object.add_pin_vertex_group",
                        icon="ADD",
                        text="Add",
                    )
                    add_pin_op.group_index = actual_index
                    row = col.row()
                    remove_enabled = (
                        0
                        <= group.pin_vertex_groups_index
                        < len(group.pin_vertex_groups)
                    )
                    sub = row.row()
                    sub.enabled = remove_enabled
                    remove_pin_op = sub.operator(
                        "object.remove_pin_vertex_group",
                        icon="REMOVE",
                    )
                    remove_pin_op.group_index = actual_index
                    sub = row.row()
                    sub.enabled = remove_enabled
                    rename_pin_op = sub.operator(
                        "object.rename_pin_vertex_group",
                        icon="SORTALPHA",
                        text="Rename",
                    )
                    rename_pin_op.group_index = actual_index
                    create_pin_op = row.operator(
                        "object.create_pin_vertex_group",
                        icon="ADD",
                        text="Create",
                    )
                    create_pin_op.group_index = actual_index
                    col.template_list(
                        "OBJECT_UL_PinVertexGroupsList",
                        "",
                        group,
                        "pin_vertex_groups",
                        group,
                        "pin_vertex_groups_index",
                    )
                    row = col.row(align=True)
                    row.prop(group, "show_pin_overlay")
                    if group.show_pin_overlay:
                        row.prop(group, "pin_overlay_size", text="Size")
                    # Resolve selected pin's object UUID for edit-mode gating.
                    from ...core.uuid_registry import get_object_uuid
                    _pin_idx = group.pin_vertex_groups_index
                    _pin_obj_uuid = ""
                    if 0 <= _pin_idx < len(group.pin_vertex_groups):
                        _pin_obj_uuid = group.pin_vertex_groups[_pin_idx].object_uuid or ""
                    _editing_pin_obj = False
                    if context.mode in ("EDIT_MESH", "EDIT_CURVE") and context.edit_object:
                        _editing_pin_obj = (
                            _pin_obj_uuid != ""
                            and get_object_uuid(context.edit_object) == _pin_obj_uuid
                        )
                    if _editing_pin_obj:
                        row = col.row(align=True)
                        sel_op = row.operator("object.select_pin_vertices", icon="RESTRICT_SELECT_OFF")
                        sel_op.group_index = actual_index
                        desel_op = row.operator("object.deselect_pin_vertices", icon="RESTRICT_SELECT_ON")
                        desel_op.group_index = actual_index
                        row = col.row(align=True)
                        mk_op = row.operator("object.make_pin_keyframe", icon="KEYFRAME")
                        mk_op.group_index = actual_index
                        dk_op = row.operator("object.delete_pin_keyframes", icon="KEYFRAME_HLT")
                        dk_op.group_index = actual_index
                    pin_idx = group.pin_vertex_groups_index
                    if 0 <= pin_idx < len(group.pin_vertex_groups):
                        pin_item = group.pin_vertex_groups[pin_idx]
                        row = col.row(align=True)
                        row.prop(pin_item, "use_pin_duration")
                        sub = row.row()
                        sub.enabled = pin_item.use_pin_duration
                        sub.prop(pin_item, "pin_duration")
                        row = col.row(align=True)
                        row.prop(pin_item, "use_pull")
                        sub = row.row()
                        sub.enabled = pin_item.use_pull
                        sub.prop(pin_item, "pull_strength")

                        # Operations list
                        col.separator()
                        ops_row = col.row()
                        ops_row.label(text="Operations:")
                        op = ops_row.operator("object.copy_pin_ops", text="", icon="COPYDOWN")
                        op.group_index = actual_index
                        op = ops_row.operator("object.paste_pin_ops", text="", icon="PASTEDOWN")
                        op.group_index = actual_index
                        # Pin operations profile
                        pin_prof_row = col.row(align=True)
                        if _profile_path_exists(group.pin_profile_path):
                            pin_prof_row.prop(
                                group, "pin_profile_selection", text="Profile"
                            )
                            op = pin_prof_row.operator(
                                "object.open_pin_profile", text="", icon="FILEBROWSER"
                            )
                            op.group_index = actual_index
                            op = pin_prof_row.operator(
                                "object.clear_pin_profile", text="", icon="X"
                            )
                            op.group_index = actual_index
                            op = pin_prof_row.operator(
                                "object.reload_pin_profile", text="", icon="FILE_REFRESH"
                            )
                            op.group_index = actual_index
                            op = pin_prof_row.operator(
                                "object.save_pin_profile", text="", icon="FILE_TICK"
                            )
                            op.group_index = actual_index
                        else:
                            op = pin_prof_row.operator(
                                "object.open_pin_profile",
                                text="Open Profile",
                                icon="FILEBROWSER",
                            )
                            op.group_index = actual_index
                            op = pin_prof_row.operator(
                                "object.save_pin_profile", text="", icon="FILE_TICK"
                            )
                            op.group_index = actual_index
                        col.template_list(
                            "OBJECT_UL_PinOperationsList", "",
                            pin_item, "operations",
                            pin_item, "operations_index",
                            rows=2,
                        )
                        row = col.row(align=True)
                        add_op = row.operator_menu_enum(
                            "object.add_pin_operation", "op_type",
                            text="Add", icon="ADD",
                        )
                        add_op.group_index = actual_index
                        rm_op = row.operator("object.remove_pin_operation", text="Remove", icon="REMOVE")
                        rm_op.group_index = actual_index
                        up_op = row.operator("object.move_pin_operation", text="", icon="TRIA_UP")
                        up_op.group_index = actual_index
                        up_op.direction = -1
                        dn_op = row.operator("object.move_pin_operation", text="", icon="TRIA_DOWN")
                        dn_op.group_index = actual_index
                        dn_op.direction = 1

                        # Show selected operation properties
                        op_idx = pin_item.operations_index
                        if 0 <= op_idx < len(pin_item.operations):
                            op = pin_item.operations[op_idx]
                            op_box = col.box()
                            if op.op_type == "MOVE_BY":
                                op_box.prop(op, "delta")
                                row = op_box.row(align=True)
                                row.prop(op, "frame_start")
                                row.prop(op, "frame_end")
                                op_box.prop(op, "transition")
                            elif op.op_type == "SPIN":
                                op_box.prop(op, "spin_center_mode")
                                if op.spin_center_mode == "ABSOLUTE":
                                    op_box.prop(op, "spin_center")
                                    pick = op_box.operator(
                                        "object.pick_center_from_selected",
                                        icon="EYEDROPPER",
                                    )
                                    pick.group_index = actual_index
                                    pick.target = "spin"
                                elif op.spin_center_mode == "MAX_TOWARDS":
                                    op_box.prop(op, "spin_center_direction")
                                    op_box.prop(op, "show_max_towards_spin")
                                elif op.spin_center_mode == "VERTEX":
                                    row = op_box.row(align=True)
                                    row.prop(op, "spin_center_vertex")
                                    pick = row.operator(
                                        "object.pick_vertex_center",
                                        icon="EYEDROPPER",
                                        text="",
                                    )
                                    pick.group_index = actual_index
                                    pick.target = "spin"
                                    op_box.prop(op, "show_vertex_spin")
                                op_box.label(text="Rotation:")
                                op_box.prop(op, "spin_axis")
                                op_box.prop(op, "spin_angular_velocity")
                                op_box.prop(op, "spin_flip")
                                row = op_box.row(align=True)
                                row.prop(op, "frame_start")
                                row.prop(op, "frame_end")
                                op_box.prop(op, "transition")
                            elif op.op_type == "SCALE":
                                op_box.prop(op, "scale_center_mode")
                                if op.scale_center_mode == "ABSOLUTE":
                                    op_box.prop(op, "scale_center")
                                    pick = op_box.operator(
                                        "object.pick_center_from_selected",
                                        icon="EYEDROPPER",
                                    )
                                    pick.group_index = actual_index
                                    pick.target = "scale"
                                elif op.scale_center_mode == "MAX_TOWARDS":
                                    op_box.prop(op, "scale_center_direction")
                                    op_box.prop(op, "show_max_towards_scale")
                                elif op.scale_center_mode == "VERTEX":
                                    row = op_box.row(align=True)
                                    row.prop(op, "scale_center_vertex")
                                    pick = row.operator(
                                        "object.pick_vertex_center",
                                        icon="EYEDROPPER",
                                        text="",
                                    )
                                    pick.group_index = actual_index
                                    pick.target = "scale"
                                    op_box.prop(op, "show_vertex_scale")
                                op_box.label(text="Scaling:")
                                op_box.prop(op, "scale_factor")
                                row = op_box.row(align=True)
                                row.prop(op, "frame_start")
                                row.prop(op, "frame_end")
                                op_box.prop(op, "transition")
                            elif op.op_type == "TORQUE":
                                op_box.prop(op, "torque_axis_component")
                                op_box.prop(op, "torque_magnitude")
                                op_box.prop(op, "torque_flip")
                                row = op_box.row(align=True)
                                row.prop(op, "frame_start")
                                row.prop(op, "frame_end")

                info_box = box.box()
                row = info_box.row()
                row.prop(
                    group,
                    "show_stats",
                    icon="TRIA_DOWN" if group.show_stats else "TRIA_RIGHT",
                    emboss=False,
                    icon_only=True,
                )
                row.label(text="Stats", icon="INFO")
                if group.show_stats:
                    from ...core.uuid_registry import get_object_by_uuid
                    col = info_box.column()
                    for obj in group.assigned_objects:
                        if not obj.included:
                            continue
                        obj_data = get_object_by_uuid(obj.uuid) if obj.uuid else None
                        if obj_data and obj_data.type == "MESH":
                            row = col.row()
                            row.label(text=obj_data.name)
                            row.label(text=f"#Vert: {len(obj_data.data.vertices)}")
                            row.label(text=f"#Face: {len(obj_data.data.polygons)}")

                param_box = box.box()
                row = param_box.row()
                row.prop(
                    group,
                    "show_parameters",
                    icon="TRIA_DOWN" if group.show_parameters else "TRIA_RIGHT",
                    emboss=False,
                    icon_only=True,
                )
                row.label(text="Material Params", icon="MATERIAL")
                op = row.operator("object.copy_material_params", text="", icon="COPYDOWN")
                op.group_index = actual_index
                op = row.operator("object.paste_material_params", text="", icon="PASTEDOWN")
                op.group_index = actual_index
                # Material param profile
                mat_profile_row = param_box.row(align=True)
                if _profile_path_exists(group.material_profile_path):
                    mat_profile_row.prop(
                        group, "material_profile_selection", text="Profile"
                    )
                    op = mat_profile_row.operator(
                        "object.open_material_profile", text="", icon="FILEBROWSER"
                    )
                    op.group_index = actual_index
                    op = mat_profile_row.operator(
                        "object.clear_material_profile", text="", icon="X"
                    )
                    op.group_index = actual_index
                    op = mat_profile_row.operator(
                        "object.reload_material_profile", text="", icon="FILE_REFRESH"
                    )
                    op.group_index = actual_index
                    op = mat_profile_row.operator(
                        "object.save_material_profile", text="", icon="FILE_TICK"
                    )
                    op.group_index = actual_index
                else:
                    op = mat_profile_row.operator(
                        "object.open_material_profile",
                        text="Open Profile",
                        icon="FILEBROWSER",
                    )
                    op.group_index = actual_index
                    op = mat_profile_row.operator(
                        "object.save_material_profile", text="", icon="FILE_TICK"
                    )
                    op.group_index = actual_index
                if group.show_parameters:
                    if group.object_type == "SOLID":
                        param_box.prop(group, "solid_model")
                        param_box.prop(group, "solid_density")
                        param_box.prop(group, "solid_young_modulus")
                        param_box.prop(group, "solid_poisson_ratio")
                        param_box.prop(group, "shrink")
                        param_box.prop(group, "friction")

                        # Contact Gap Settings Box
                        contact_box = param_box.box()
                        contact_box.prop(group, "use_group_bounding_box_diagonal")
                        if group.use_group_bounding_box_diagonal:
                            contact_box.prop(group, "contact_gap_rat")
                            contact_box.prop(group, "contact_offset_rat")
                        else:
                            contact_box.prop(group, "contact_gap")
                            contact_box.prop(group, "contact_offset")
                        _draw_collision_windows(param_box, group, actual_index)
                        plast_box = param_box.box()
                        plast_box.prop(group, "enable_plasticity")
                        if group.enable_plasticity:
                            plast_box.prop(group, "plasticity")
                            plast_box.prop(group, "plasticity_threshold")
                        _draw_velocity_keyframes(param_box, group, actual_index)
                        _draw_ftetwild(param_box, group)
                        if _group_has_stitch(group):
                            param_box.prop(group, "stitch_stiffness")
                    elif group.object_type == "SHELL":
                        param_box.prop(group, "shell_model")
                        param_box.prop(group, "shell_density")
                        param_box.prop(group, "shell_young_modulus")
                        param_box.prop(group, "shell_poisson_ratio")
                        param_box.prop(group, "friction")

                        # Contact Gap Settings Box
                        contact_box = param_box.box()
                        contact_box.prop(group, "use_group_bounding_box_diagonal")
                        if group.use_group_bounding_box_diagonal:
                            contact_box.prop(group, "contact_gap_rat")
                            contact_box.prop(group, "contact_offset_rat")
                        else:
                            contact_box.prop(group, "contact_gap")
                            contact_box.prop(group, "contact_offset")
                        _draw_collision_windows(param_box, group, actual_index)
                        param_box.prop(group, "bend")
                        row = param_box.row(align=True)
                        row.prop(group, "shrink_x")
                        row.prop(group, "shrink_y")
                        sl_box = param_box.box()
                        sl_box.prop(group, "enable_strain_limit")
                        if group.enable_strain_limit:
                            if group.shrink_x != 1.0 or group.shrink_y != 1.0:
                                sl_box.label(
                                    text="Shrink/extend disables strain limiting",
                                    icon="ERROR",
                                )
                            sl_box.prop(group, "strain_limit")
                        inf_box = param_box.box()
                        inf_box.prop(group, "enable_inflate")
                        if group.enable_inflate:
                            inf_box.prop(group, "inflate_pressure")
                        plast_box = param_box.box()
                        plast_box.prop(group, "enable_plasticity")
                        if group.enable_plasticity:
                            plast_box.prop(group, "plasticity")
                            plast_box.prop(group, "plasticity_threshold")
                        bplast_box = param_box.box()
                        bplast_box.prop(group, "bend_rest_angle_source")
                        bplast_box.prop(group, "enable_bend_plasticity")
                        if group.enable_bend_plasticity:
                            bplast_box.prop(group, "bend_plasticity")
                            bplast_box.prop(group, "bend_plasticity_threshold")
                        _draw_velocity_keyframes(param_box, group, actual_index)
                        if _group_has_stitch(group):
                            param_box.prop(group, "stitch_stiffness")
                    elif group.object_type == "ROD":
                        # Rod model is always ARAP, no need to show selection
                        param_box.prop(group, "rod_density")
                        param_box.prop(group, "rod_young_modulus")
                        param_box.prop(group, "friction")

                        contact_box = param_box.box()
                        contact_box.prop(group, "contact_gap")
                        contact_box.prop(group, "contact_offset")
                        _draw_collision_windows(param_box, group, actual_index)
                        bend_box = param_box.box()
                        bend_box.label(text="Bend")
                        bend_box.prop(group, "bend")
                        bend_box.prop(group, "bend_rest_angle_source")
                        sl_box = param_box.box()
                        sl_box.prop(group, "enable_strain_limit")
                        if group.enable_strain_limit:
                            sl_box.prop(group, "strain_limit")
                        bplast_box = param_box.box()
                        bplast_box.prop(group, "enable_bend_plasticity")
                        if group.enable_bend_plasticity:
                            bplast_box.prop(group, "bend_plasticity")
                            bplast_box.prop(group, "bend_plasticity_threshold")
                        _draw_velocity_keyframes(param_box, group, actual_index)
                    else:  # STATIC
                        param_box.prop(group, "friction")

                        contact_box = param_box.box()
                        contact_box.prop(group, "use_group_bounding_box_diagonal")
                        if group.use_group_bounding_box_diagonal:
                            contact_box.prop(group, "contact_gap_rat")
                            contact_box.prop(group, "contact_offset_rat")
                        else:
                            contact_box.prop(group, "contact_gap")
                            contact_box.prop(group, "contact_offset")


class SNAPMERGE_PT_SnapAndMerge(Panel):
    bl_label = "Snap and Merge"
    bl_idname = "SNAPMERGE_PT_SnapAndMerge"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = get_category_name()
    bl_options = {"DEFAULT_CLOSED"}

    @classmethod
    def poll(cls, context):
        return has_addon_data(context.scene)

    def draw(self, context):
        layout = self.layout

        # Snap to Vertices box
        snap_box = layout.box()
        snap_box.label(text="Snap To Nearby Vertices")

        # Object A dropdown
        col = snap_box.column(align=True)
        col.label(text="Object A (moves):")
        row = col.row(align=True)
        row.prop(get_addon_data(context.scene).state, "snap_object_a", text="")
        op = row.operator("object.pick_snap_object", text="", icon="EYEDROPPER")
        op.target = "A"

        # Object B dropdown
        col.label(text="Object B (target):")
        row = col.row(align=True)
        row.prop(get_addon_data(context.scene).state, "snap_object_b", text="")
        op = row.operator("object.pick_snap_object", text="", icon="EYEDROPPER")
        op.target = "B"

        # Snap button
        row = snap_box.row()
        row.operator("object.snap_to_vertices", icon="SNAP_ON")

        # Merge pairs and explicit snap-authored stitch anchors
        state = get_addon_data(context.scene).state
        if len(state.merge_pairs) > 0:
            merge_box = layout.box()
            merge_box.label(text="Merge Pairs")

            merge_box.template_list(
                "OBJECT_UL_MergePairsList",
                "",
                state,
                "merge_pairs",
                state,
                "merge_pairs_index",
            )

            row = merge_box.row()
            row.enabled = 0 <= state.merge_pairs_index < len(state.merge_pairs)
            row.operator("object.remove_merge_pair", icon="REMOVE")

            # Show stitch stiffness only when a SOLID object is involved
            # (sheet-sheet and rod-rod merge vertices exactly).
            idx = state.merge_pairs_index
            if 0 <= idx < len(state.merge_pairs):
                pair = state.merge_pairs[idx]
                type_a = type_b = ""
                for group in iterate_active_object_groups(context.scene):
                    for assigned in group.assigned_objects:
                        if pair.object_a_uuid and assigned.uuid == pair.object_a_uuid:
                            type_a = group.object_type
                        elif pair.object_b_uuid and assigned.uuid == pair.object_b_uuid:
                            type_b = group.object_type
                if pair_supports_cross_stitch(type_a, type_b) and "SOLID" in (type_a, type_b):
                    merge_box.prop(pair, "stitch_stiffness")


class VISUALIZATION_PT_Visualization(Panel):
    bl_label = "Visualization"
    bl_idname = "VISUALIZATION_PT_Visualization"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = get_category_name()
    bl_options = {"DEFAULT_CLOSED"}
    bl_order = 100

    @classmethod
    def poll(cls, context):
        return has_addon_data(context.scene)

    def draw(self, context):
        layout = self.layout
        state = get_addon_data(context.scene).state
        layout.prop(state, "hide_pins")
        layout.prop(state, "hide_arrows")
        layout.prop(state, "hide_overlay_colors")
        layout.prop(state, "hide_snaps")
        layout.prop(state, "hide_pin_operations")


classes = (
    MAIN_PT_SceneConfiguration,
    DYNAMICS_PT_Groups,
    SNAPMERGE_PT_SnapAndMerge,
    VISUALIZATION_PT_Visualization,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
