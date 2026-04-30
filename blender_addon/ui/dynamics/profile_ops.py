# File: profile_ops.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Profile operators for scene and material parameter profiles.

import bpy  # pyright: ignore
from bpy.props import BoolProperty, IntProperty, PointerProperty, StringProperty  # pyright: ignore
from bpy.types import Operator  # pyright: ignore

from ...core.param_introspect import (
    MATERIAL_CLIPBOARD_EXCLUDE,
    PIN_OP_CLIPBOARD_EXCLUDE,
    copy_scalar_props,
    material_param_applies,
)
from ...core.utils import redraw_all_areas
from ...models.groups import get_addon_data, invalidate_overlays
from .utils import get_group_from_index


class SCENE_OT_OpenSceneProfile(Operator):
    """Open a TOML scene parameter profile file."""

    bl_idname = "scene.open_scene_profile"
    bl_label = "Open Scene Profile"

    filepath: StringProperty(subtype="FILE_PATH")  # pyright: ignore
    filter_glob: StringProperty(default="*.toml", options={"HIDDEN"})  # pyright: ignore

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = "scene_profile.toml"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        from ...core.profile import get_profile_names

        state = get_addon_data(context.scene).state
        state.scene_profile_path = self.filepath
        abs_path = bpy.path.abspath(self.filepath)
        names = get_profile_names(abs_path)
        if names:
            state.scene_profile_selection = names[0]
        redraw_all_areas(context)
        return {"FINISHED"}


class SCENE_OT_ClearSceneProfile(Operator):
    """Clear the loaded scene parameter profile."""

    bl_idname = "scene.clear_scene_profile"
    bl_label = "Clear Scene Profile"

    def execute(self, context):
        get_addon_data(context.scene).state.scene_profile_path = ""
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_OpenMaterialProfile(Operator):
    """Open a TOML material parameter profile file."""

    bl_idname = "object.open_material_profile"
    bl_label = "Open Material Profile"

    filepath: StringProperty(subtype="FILE_PATH")  # pyright: ignore
    filter_glob: StringProperty(default="*.toml", options={"HIDDEN"})  # pyright: ignore
    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = "material_profile.toml"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        from ...core.profile import get_profile_names

        addon_data = get_addon_data(context.scene)
        group = getattr(addon_data, f"object_group_{self.group_index}", None)
        if group is None:
            return {"CANCELLED"}
        group.material_profile_path = self.filepath
        abs_path = bpy.path.abspath(self.filepath)
        names = get_profile_names(abs_path)
        if names:
            group.material_profile_selection = names[0]
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_ClearMaterialProfile(Operator):
    """Clear the loaded material parameter profile."""

    bl_idname = "object.clear_material_profile"
    bl_label = "Clear Material Profile"

    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    def execute(self, context):
        addon_data = get_addon_data(context.scene)
        group = getattr(addon_data, f"object_group_{self.group_index}", None)
        if group is None:
            return {"CANCELLED"}
        group.material_profile_path = ""
        redraw_all_areas(context)
        return {"FINISHED"}


class SCENE_OT_ReloadSceneProfile(Operator):
    """Re-apply the currently selected scene profile."""

    bl_idname = "scene.reload_scene_profile"
    bl_label = "Reload Scene Profile"

    def execute(self, context):
        from ...core.profile import apply_scene_profile, load_profiles

        state = get_addon_data(context.scene).state
        if not state.scene_profile_path or state.scene_profile_selection == "NONE":
            return {"CANCELLED"}
        abs_path = bpy.path.abspath(state.scene_profile_path)
        profiles = load_profiles(abs_path)
        profile = profiles.get(state.scene_profile_selection)
        if profile is None:
            return {"CANCELLED"}
        apply_scene_profile(profile, state)
        from ...models.groups import invalidate_overlays
        invalidate_overlays()
        redraw_all_areas(context)
        return {"FINISHED"}


class SCENE_OT_SaveSceneProfile(Operator):
    """Save current scene parameters to a profile."""

    bl_idname = "scene.save_scene_profile"
    bl_label = "Save Scene Profile"

    filepath: StringProperty(subtype="FILE_PATH")  # pyright: ignore
    filter_glob: StringProperty(default="*.toml", options={"HIDDEN"})  # pyright: ignore
    entry_name: StringProperty(name="Entry Name", default="Default")  # pyright: ignore

    def invoke(self, context, event):
        state = get_addon_data(context.scene).state
        if state.scene_profile_path and state.scene_profile_selection != "NONE":
            return self.execute(context)
        if not self.filepath:
            self.filepath = "scene_profile.toml"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        from ...core.profile import read_scene_profile, save_profile_entry

        state = get_addon_data(context.scene).state
        data = read_scene_profile(state)
        if state.scene_profile_path and state.scene_profile_selection != "NONE":
            abs_path = bpy.path.abspath(state.scene_profile_path)
            save_profile_entry(abs_path, state.scene_profile_selection, data)
            self.report({"INFO"}, f"Saved to '{state.scene_profile_selection}'")
        else:
            save_profile_entry(self.filepath, self.entry_name, data)
            state.scene_profile_path = self.filepath
            state.scene_profile_selection = self.entry_name
            self.report({"INFO"}, f"Saved to '{self.entry_name}'")
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_ReloadMaterialProfile(Operator):
    """Re-apply the currently selected material profile."""

    bl_idname = "object.reload_material_profile"
    bl_label = "Reload Material Profile"

    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    def execute(self, context):
        from ...core.profile import apply_material_profile, load_profiles

        group = get_group_from_index(context.scene, self.group_index)
        if group is None:
            return {"CANCELLED"}
        if not group.material_profile_path or group.material_profile_selection == "NONE":
            return {"CANCELLED"}
        abs_path = bpy.path.abspath(group.material_profile_path)
        profiles = load_profiles(abs_path)
        profile = profiles.get(group.material_profile_selection)
        if profile is None:
            return {"CANCELLED"}
        apply_material_profile(profile, group)
        from ...models.groups import invalidate_overlays
        invalidate_overlays()
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_SaveMaterialProfile(Operator):
    """Save current material parameters to a profile."""

    bl_idname = "object.save_material_profile"
    bl_label = "Save Material Profile"

    filepath: StringProperty(subtype="FILE_PATH")  # pyright: ignore
    filter_glob: StringProperty(default="*.toml", options={"HIDDEN"})  # pyright: ignore
    entry_name: StringProperty(name="Entry Name", default="Default")  # pyright: ignore
    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    def invoke(self, context, event):
        addon_data = get_addon_data(context.scene)
        group = getattr(addon_data, f"object_group_{self.group_index}", None)
        if group and group.material_profile_path and group.material_profile_selection != "NONE":
            return self.execute(context)
        if not self.filepath:
            self.filepath = "material_profile.toml"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        from ...core.profile import read_material_profile, save_profile_entry

        group = get_group_from_index(context.scene, self.group_index)
        if group is None:
            return {"CANCELLED"}
        data = read_material_profile(group)
        if group.material_profile_path and group.material_profile_selection != "NONE":
            abs_path = bpy.path.abspath(group.material_profile_path)
            save_profile_entry(abs_path, group.material_profile_selection, data)
            self.report({"INFO"}, f"Saved to '{group.material_profile_selection}'")
        else:
            save_profile_entry(self.filepath, self.entry_name, data)
            group.material_profile_path = self.filepath
            group.material_profile_selection = self.entry_name
            self.report({"INFO"}, f"Saved to '{self.entry_name}'")
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_CopyMaterialParams(Operator):
    """Copy material parameters to clipboard."""

    bl_idname = "object.copy_material_params"
    bl_label = "Copy Material Params"

    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    def execute(self, context):
        group = get_group_from_index(context.scene, self.group_index)
        if group is None:
            return {"CANCELLED"}
        wm = context.window_manager
        copy_scalar_props(group, wm.material_clipboard, exclude=MATERIAL_CLIPBOARD_EXCLUDE)
        wm.material_clipboard_src_type = group.object_type
        wm.material_clipboard_valid = True
        self.report({"INFO"}, "Material params copied")
        return {"FINISHED"}


class OBJECT_OT_PasteMaterialParams(Operator):
    """Paste material parameters from clipboard."""

    bl_idname = "object.paste_material_params"
    bl_label = "Paste Material Params"

    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return getattr(context.window_manager, "material_clipboard_valid", False)

    def execute(self, context):
        group = get_group_from_index(context.scene, self.group_index)
        if group is None:
            return {"CANCELLED"}
        wm = context.window_manager
        # Filter model-specific params by the source's object_type so
        # cross-type pastes don't overwrite the target's unused
        # shell_*/solid_*/rod_* fields with the source's defaults.
        src_type = wm.material_clipboard_src_type or group.object_type
        copy_scalar_props(
            wm.material_clipboard,
            group,
            exclude=MATERIAL_CLIPBOARD_EXCLUDE,
            filter_fn=lambda n: material_param_applies(n, src_type),
        )
        redraw_all_areas(context)
        self.report({"INFO"}, "Material params pasted")
        return {"FINISHED"}


def _get_selected_pin(context, group_index):
    """Get the selected pin vertex group item for a given object group."""
    group = get_group_from_index(context.scene, group_index)
    if group is None or group.pin_vertex_groups_index < 0:
        return None, None
    idx = group.pin_vertex_groups_index
    if idx >= len(group.pin_vertex_groups):
        return None, None
    return group, group.pin_vertex_groups[idx]


class OBJECT_OT_OpenPinProfile(Operator):
    """Open a TOML pin operations profile file."""

    bl_idname = "object.open_pin_profile"
    bl_label = "Open Pin Operations Profile"

    filepath: StringProperty(subtype="FILE_PATH")  # pyright: ignore
    filter_glob: StringProperty(default="*.toml", options={"HIDDEN"})  # pyright: ignore
    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    def invoke(self, context, event):
        if not self.filepath:
            self.filepath = "pin_profile.toml"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        from ...core.profile import get_profile_names

        addon_data = get_addon_data(context.scene)
        group = getattr(addon_data, f"object_group_{self.group_index}", None)
        if group is None:
            return {"CANCELLED"}
        group.pin_profile_path = self.filepath
        abs_path = bpy.path.abspath(self.filepath)
        names = get_profile_names(abs_path)
        if names:
            group.pin_profile_selection = names[0]
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_ClearPinProfile(Operator):
    """Clear the loaded pin operations profile."""

    bl_idname = "object.clear_pin_profile"
    bl_label = "Clear Pin Profile"

    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    def execute(self, context):
        addon_data = get_addon_data(context.scene)
        group = getattr(addon_data, f"object_group_{self.group_index}", None)
        if group is None:
            return {"CANCELLED"}
        group.pin_profile_path = ""
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_ReloadPinProfile(Operator):
    """Re-apply the currently selected pin operations profile."""

    bl_idname = "object.reload_pin_profile"
    bl_label = "Reload Pin Profile"

    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    def execute(self, context):
        from ...core.profile import apply_pin_operations, load_profiles

        group, pin_item = _get_selected_pin(context, self.group_index)
        if pin_item is None:
            return {"CANCELLED"}
        if not group.pin_profile_path or group.pin_profile_selection == "NONE":
            return {"CANCELLED"}
        abs_path = bpy.path.abspath(group.pin_profile_path)
        profiles = load_profiles(abs_path)
        profile = profiles.get(group.pin_profile_selection)
        if profile is None:
            return {"CANCELLED"}
        apply_pin_operations(profile, pin_item)
        from ...models.groups import invalidate_overlays
        invalidate_overlays()
        redraw_all_areas(context)
        return {"FINISHED"}


class OBJECT_OT_SavePinProfile(Operator):
    """Save current pin operations to a profile."""

    bl_idname = "object.save_pin_profile"
    bl_label = "Save Pin Operations Profile"

    filepath: StringProperty(subtype="FILE_PATH")  # pyright: ignore
    filter_glob: StringProperty(default="*.toml", options={"HIDDEN"})  # pyright: ignore
    entry_name: StringProperty(name="Entry Name", default="Default")  # pyright: ignore
    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    def invoke(self, context, event):
        addon_data = get_addon_data(context.scene)
        group = getattr(addon_data, f"object_group_{self.group_index}", None)
        if group and group.pin_profile_path and group.pin_profile_selection != "NONE":
            return self.execute(context)
        if not self.filepath:
            self.filepath = "pin_profile.toml"
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        from ...core.profile import read_pin_operations, save_profile_entry

        group, pin_item = _get_selected_pin(context, self.group_index)
        if pin_item is None:
            self.report({"WARNING"}, "No pin vertex group selected")
            return {"CANCELLED"}
        data = read_pin_operations(pin_item)
        if group.pin_profile_path and group.pin_profile_selection != "NONE":
            abs_path = bpy.path.abspath(group.pin_profile_path)
            save_profile_entry(abs_path, group.pin_profile_selection, data)
            self.report({"INFO"}, f"Saved operations to '{group.pin_profile_selection}'")
        else:
            save_profile_entry(self.filepath, self.entry_name, data)
            group.pin_profile_path = self.filepath
            group.pin_profile_selection = self.entry_name
            self.report({"INFO"}, f"Saved operations to '{self.entry_name}'")
        redraw_all_areas(context)
        return {"FINISHED"}


def _copy_operations_collection(src_ops, dst_ops):
    """Replace *dst_ops* with a deep copy of *src_ops*. Field set is
    discovered from ``PinOperation.bl_rna.properties`` — new fields
    added to PinOperation flow through automatically.
    """
    dst_ops.clear()
    for src_op in src_ops:
        new_op = dst_ops.add()
        copy_scalar_props(src_op, new_op, exclude=PIN_OP_CLIPBOARD_EXCLUDE)


class OBJECT_OT_CopyPinOps(Operator):
    """Copy pin operations to clipboard."""

    bl_idname = "object.copy_pin_ops"
    bl_label = "Copy Pin Operations"

    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    def execute(self, context):
        _, pin_item = _get_selected_pin(context, self.group_index)
        if pin_item is None:
            self.report({"WARNING"}, "No pin vertex group selected")
            return {"CANCELLED"}
        wm = context.window_manager
        _copy_operations_collection(pin_item.operations, wm.pin_ops_clipboard.operations)
        wm.pin_ops_clipboard_valid = True
        self.report({"INFO"}, f"Copied {len(pin_item.operations)} operations")
        return {"FINISHED"}


class OBJECT_OT_PastePinOps(Operator):
    """Paste pin operations from clipboard."""

    bl_idname = "object.paste_pin_ops"
    bl_label = "Paste Pin Operations"

    group_index: IntProperty(options={"HIDDEN"})  # pyright: ignore

    @classmethod
    def poll(cls, context):
        return getattr(context.window_manager, "pin_ops_clipboard_valid", False)

    def execute(self, context):
        _, pin_item = _get_selected_pin(context, self.group_index)
        if pin_item is None:
            self.report({"WARNING"}, "No pin vertex group selected")
            return {"CANCELLED"}
        wm = context.window_manager
        _copy_operations_collection(wm.pin_ops_clipboard.operations, pin_item.operations)
        invalidate_overlays()
        redraw_all_areas(context)
        self.report({"INFO"}, "Pin operations pasted")
        return {"FINISHED"}


classes = [
    SCENE_OT_OpenSceneProfile,
    SCENE_OT_ClearSceneProfile,
    SCENE_OT_ReloadSceneProfile,
    SCENE_OT_SaveSceneProfile,
    OBJECT_OT_OpenMaterialProfile,
    OBJECT_OT_ClearMaterialProfile,
    OBJECT_OT_ReloadMaterialProfile,
    OBJECT_OT_SaveMaterialProfile,
    OBJECT_OT_CopyMaterialParams,
    OBJECT_OT_PasteMaterialParams,
    OBJECT_OT_OpenPinProfile,
    OBJECT_OT_ClearPinProfile,
    OBJECT_OT_ReloadPinProfile,
    OBJECT_OT_SavePinProfile,
    OBJECT_OT_CopyPinOps,
    OBJECT_OT_PastePinOps,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    # Clipboard storage: PointerProperty to the same PropertyGroup
    # classes the source uses. Field-by-field copy is driven by
    # bl_rna introspection at copy/paste time, so the clipboard
    # schema auto-tracks any new ObjectGroup / PinOperation params.
    # WindowManager scope = session-only (not saved to .blend),
    # which matches clipboard semantics.
    from ..state_types import PinVertexGroupItem
    from ..object_group import ObjectGroup
    bpy.types.WindowManager.material_clipboard = PointerProperty(type=ObjectGroup)
    bpy.types.WindowManager.material_clipboard_valid = BoolProperty(default=False)
    bpy.types.WindowManager.material_clipboard_src_type = StringProperty(default="")
    bpy.types.WindowManager.pin_ops_clipboard = PointerProperty(type=PinVertexGroupItem)
    bpy.types.WindowManager.pin_ops_clipboard_valid = BoolProperty(default=False)


def unregister():
    for attr in (
        "material_clipboard",
        "material_clipboard_valid",
        "material_clipboard_src_type",
        "pin_ops_clipboard",
        "pin_ops_clipboard_valid",
    ):
        if hasattr(bpy.types.WindowManager, attr):
            delattr(bpy.types.WindowManager, attr)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
