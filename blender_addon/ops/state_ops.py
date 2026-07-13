"""State manipulation operators for bpy.ops.zozo_contact_solver.set().

Provides a unified `set` operator that writes to scene.state, scene.ssh_state,
or ObjectGroup properties based on the key (and optional group_uuid).
"""

import bpy  # pyright: ignore
from bpy.props import StringProperty  # pyright: ignore
from bpy.app.translations import pgettext_iface as iface_, pgettext_tip as tip_

from ..models.defaults import SCENE_PARAM_ALIASES
from ..models.groups import get_addon_data


def _coerce_value(prop_group, key, value_str):
    """Convert a string value to the property's actual type using bl_rna introspection."""
    bl_props = prop_group.bl_rna.properties
    if key not in bl_props:
        raise KeyError(f"Property '{key}' not found")

    prop_info = bl_props[key]
    prop_type = prop_info.type

    is_array = getattr(prop_info, "is_array", False)

    if prop_type == "FLOAT" and is_array:
        import ast
        parsed = ast.literal_eval(value_str)
        return tuple(float(x) for x in parsed)
    elif prop_type == "FLOAT":
        return float(value_str)
    elif prop_type == "INT":
        return int(value_str)
    elif prop_type == "BOOLEAN":
        return value_str.lower() in ("true", "1", "yes")
    elif prop_type == "ENUM":
        return value_str
    elif prop_type == "STRING":
        return value_str
    else:
        return value_str


class ZOZO_CTS_OT_Set(bpy.types.Operator):
    """Set a scene/SSH/group property by key name with auto type conversion."""

    bl_idname = "zozo_contact_solver.set"
    bl_label = "Set Property"
    bl_options = {"REGISTER", "UNDO"}

    key: StringProperty(name="Key", description="Property name to set")  # pyright: ignore
    value: StringProperty(name="Value", description="Value as string (auto-converted)")  # pyright: ignore
    group_uuid: StringProperty(
        name="Group UUID",
        default="",
        description="UUID of the object group (optional, for group properties)",
    )  # pyright: ignore

    def _refresh_ui(self, context):
        """Tag all areas for redraw so UI reflects the change."""
        from ..core.utils import redraw_all_areas
        redraw_all_areas(context)

    def execute(self, context):
        scene = context.scene
        key = self.key
        value_str = self.value

        if not key:
            self.report({"ERROR"}, iface_("key is required"))
            return {"CANCELLED"}

        # If group_uuid is provided, target ObjectGroup
        if self.group_uuid:
            from ..models.groups import get_group_by_uuid

            group = get_group_by_uuid(scene, self.group_uuid)
            if not group:
                self.report({"ERROR"}, iface_("Group with UUID '{uuid}' not found").format(uuid=self.group_uuid))
                return {"CANCELLED"}
            try:
                converted = _coerce_value(group, key, value_str)
                setattr(group, key, converted)
                self._refresh_ui(context)
                return {"FINISHED"}
            except KeyError:
                self.report({"ERROR"}, iface_("Property '{key}' not found on ObjectGroup").format(key=key))
                return {"CANCELLED"}
            except Exception as e:
                self.report({"ERROR"}, iface_("Failed to set group property '{key}': {error}").format(key=key, error=e))
                return {"CANCELLED"}

        # Scene/SSH parameter aliases for backward compatibility. Applied only
        # on the scene path so group-targeted keys keep the caller's name in
        # any error message.
        key = SCENE_PARAM_ALIASES.get(key, key)

        # Try addon_data.state first
        addon_data = get_addon_data(scene)
        if addon_data and hasattr(addon_data, "state"):
            state = addon_data.state
            bl_props = state.bl_rna.properties
            if key in bl_props:
                try:
                    converted = _coerce_value(state, key, value_str)
                    setattr(state, key, converted)
                    self._refresh_ui(context)
                    return {"FINISHED"}
                except Exception as e:
                    self.report({"ERROR"}, iface_("Failed to set state.{key}: {error}").format(key=key, error=e))
                    return {"CANCELLED"}

        # Try addon_data.ssh_state
        if addon_data and hasattr(addon_data, "ssh_state"):
            ssh_state = addon_data.ssh_state
            bl_props = ssh_state.bl_rna.properties
            if key in bl_props:
                try:
                    converted = _coerce_value(ssh_state, key, value_str)
                    setattr(ssh_state, key, converted)
                    self._refresh_ui(context)
                    return {"FINISHED"}
                except Exception as e:
                    self.report({"ERROR"}, iface_("Failed to set ssh_state.{key}: {error}").format(key=key, error=e))
                    return {"CANCELLED"}

        self.report(
            {"ERROR"},
            iface_("Property '{key}' not found in addon state or ssh_state").format(key=key),
        )
        return {"CANCELLED"}


_classes = [
    ZOZO_CTS_OT_Set,
]


def register():
    for cls in _classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(_classes):
        bpy.utils.unregister_class(cls)
