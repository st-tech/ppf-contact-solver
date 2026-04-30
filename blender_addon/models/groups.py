# File: groups.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Pure data-access functions for object groups.
# Extracted from ui/state.py to break core/ → ui/ dependency.

import re

import bpy  # pyright: ignore

N_MAX_GROUPS = 32
_ADDON_NAMESPACE = "zozo_contact_solver"


def get_addon_data(scene=None):
    """Return the addon's root PropertyGroup for the given scene.

    This is the single place that knows the attribute name on ``bpy.types.Scene``.
    All other code should call this instead of hard-coding the namespace.

    Args:
        scene: A Blender scene.  Defaults to ``bpy.context.scene``.
    """
    if scene is None:
        scene = bpy.context.scene
    return getattr(scene, _ADDON_NAMESPACE)


def has_addon_data(scene=None):
    """Return True when the addon's root PropertyGroup is registered on *scene*.

    Panel ``poll()`` methods use this to avoid dereferencing the state
    during transitions (initial addon register, fresh .blend load_post
    before handlers have wired up).
    """
    if scene is None:
        scene = bpy.context.scene
    if scene is None:
        return False
    return hasattr(scene, _ADDON_NAMESPACE)


def invalidate_overlays():
    """Bump overlay version and tag all viewports for redraw."""
    try:
        root = get_addon_data()
        root.state.overlay_version += 1
    except Exception:
        pass
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


# Centralized default values for ObjectGroup properties
# This ensures single source of truth for all defaults
OBJECT_GROUP_DEFAULTS = {
    # Core group properties
    "active": False,
    "name": "",  # Empty name shows "Group N" in UI
    # Object type and models
    "object_type": "SOLID",
    "solid_model": "ARAP",
    "shell_model": "BARAFF_WITKIN",
    "rod_model": "ARAP",
    # Density values
    "solid_density": 1000.0,
    "shell_density": 1.0,
    "rod_density": 1.0,
    # Young's modulus values
    "solid_young_modulus": 500.0,
    "shell_young_modulus": 1000.0,
    "rod_young_modulus": 10000.0,
    # Poisson's ratio values
    "solid_poisson_ratio": 0.35,
    "shell_poisson_ratio": 0.35,
    # Material properties
    "friction": 0.5,
    # Contact parameters
    "contact_gap": 0.001,
    "contact_offset": 0.0,
    "use_group_bounding_box_diagonal": True,
    "contact_gap_rat": 0.001,
    "contact_offset_rat": 0.0,
    "computed_contact_gap": 0.001,
    "computed_contact_offset": 0.0,
    # Solid-specific parameters
    "shrink": 1.0,
    # fTetWild overrides (SOLID groups). When no sub-override is set, no
    # kwargs are forwarded and pure fTetWild defaults apply.
    "show_ftetwild": False,
    "ftetwild_override_edge_length_fac": False,
    "ftetwild_edge_length_fac": 0.05,
    "ftetwild_override_epsilon": False,
    "ftetwild_epsilon": 1e-3,
    "ftetwild_override_stop_energy": False,
    "ftetwild_stop_energy": 10.0,
    "ftetwild_override_num_opt_iter": False,
    "ftetwild_num_opt_iter": 80,
    "ftetwild_override_optimize": False,
    "ftetwild_optimize": True,
    "ftetwild_override_simplify": False,
    "ftetwild_simplify": True,
    "ftetwild_override_coarsen": False,
    "ftetwild_coarsen": False,
    # Shell-specific parameters
    "enable_strain_limit": False,
    "strain_limit": 0.05,
    "bend": 10.0,
    "shrink_x": 1.0,
    "shrink_y": 1.0,
    "enable_inflate": False,
    "inflate_pressure": 0.0,
    "enable_plasticity": False,
    "plasticity": 0.5,
    "plasticity_threshold": 0.0,
    "enable_bend_plasticity": False,
    "bend_plasticity": 0.5,
    "bend_plasticity_threshold": 0.0,
    "bend_rest_angle_source": "FLAT",
    "stitch_stiffness": 1.0,
    # UI visibility flags
    "show_parameters": False,
    "show_overlay_color": True,
    "show_stats": False,
    "show_pin": False,
    "show_group": True,
    # Pin overlay settings
    "show_pin_overlay": True,
    "pin_overlay_size": 12.0,
    # Collection indices
    "assigned_objects_index": -1,
    "pin_vertex_groups_index": -1,
}


def get_object_type(object_type):
    intensity = 0.75
    alpha = 0.75
    if object_type == "STATIC":
        color = (0.0, 0.0, intensity, alpha)
    elif object_type == "SHELL":
        color = (0.0, intensity, 0.0, alpha)
    elif object_type == "SOLID":
        color = (intensity, 0.0, 0.0, alpha)
    elif object_type == "ROD":
        color = (intensity, intensity, 0.0, alpha)  # Yellow for rod
    else:
        color = (0.0, 0.0, 0.0, alpha)
    return color


def get_vertex_group_items(self, _):
    from ..core.uuid_registry import get_object_by_uuid

    items = []
    names_seen = set()
    for obj_ref in self.assigned_objects:
        obj = get_object_by_uuid(obj_ref.uuid) if obj_ref.uuid else None
        if obj and obj.type == "MESH":
            for vg in obj.vertex_groups:
                name = f"[{obj.name}][{vg.name}]"
                identifier = encode_vertex_group_identifier(obj.name, vg.name)
                description = f"Vertex group {vg.name} of {obj.name}"
                if identifier not in names_seen:
                    items.append((identifier, name, description))
                    names_seen.add(identifier)
    if not items:
        items.append(("NONE", "None", "No vertex groups found"))
    return items


def encode_vertex_group_identifier(obj_name: str, vg_name: str) -> str:
    """Canonical `[obj][vg]` identifier builder.

    All producers should go through this so the decoder's regex stays
    the single source of truth for the format.
    """
    return f"[{obj_name}][{vg_name}]"


def decode_vertex_group_identifier(identifier: str):
    """Decode identifier in the format [NAME1][NAME2] to (NAME1, NAME2), allowing [ or ] in names."""
    match = re.fullmatch(r"\[(.*)]\[(.*)]", identifier)
    if match:
        return match.group(1), match.group(2)
    return None, None


def parse_pin_identifier(
    identifier: str, error_cls,
) -> tuple[str, str]:
    """Parse a pin identifier in either `object::vgroup` or `[object][vgroup]` form.

    Raises `error_cls` on malformed input. Callers plug in their layer's
    exception type (MCPError, ValidationError, ...).
    """
    if "::" in identifier:
        obj_name, vg_name = identifier.split("::", 1)
        return obj_name, vg_name
    if identifier.startswith("["):
        obj_name, vg_name = decode_vertex_group_identifier(identifier)
        if obj_name is None:
            raise error_cls(
                "Invalid identifier format. Use 'object_name::vertex_group_name'"
            )
        return obj_name, vg_name
    raise error_cls(
        "vertex_group_identifier must be in format 'object_name::vertex_group_name'"
    )


def iterate_object_groups(scene):
    """Iterate over all object groups in the scene."""
    for i in range(N_MAX_GROUPS):
        prop_name = f"object_group_{i}"
        group = getattr(get_addon_data(scene), prop_name, None)
        if group:
            yield group


def iterate_active_object_groups(scene):
    """Iterate over active object groups in the scene."""
    for group in iterate_object_groups(scene):
        if group.active:
            yield group


def get_group_by_uuid(scene, group_uuid: str):
    """Get a group by its UUID."""
    for group in iterate_object_groups(scene):
        if group.uuid == group_uuid:
            return group
    return None


def get_active_group_by_uuid(scene, group_uuid: str):
    """Get an active group by its UUID."""
    group = get_group_by_uuid(scene, group_uuid)
    if group and group.active:
        return group
    return None



def find_available_group_slot(scene):
    """Find the first available object_group_N slot."""
    for i in range(N_MAX_GROUPS):
        prop_name = f"object_group_{i}"
        group = getattr(get_addon_data(scene), prop_name, None)
        if not group or not group.active:
            return i
    return None


def pair_supports_cross_stitch(type_a, type_b):
    """Check whether the given pair of group types supports cross-stitch."""
    pair = {type_a, type_b}
    return (
        pair == {"SHELL"}
        or pair == {"SHELL", "SOLID"}
        or pair == {"ROD", "SHELL"}
        or pair == {"ROD", "SOLID"}
        or pair == {"ROD"}
    )


def assign_display_indices(scene):
    """Assign display indices to active groups sequentially."""
    active_groups = list(iterate_active_object_groups(scene))
    for idx, group in enumerate(active_groups):
        group.index = idx
