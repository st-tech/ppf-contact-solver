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
    from ..core.utils import redraw_all_windows
    redraw_all_windows("VIEW_3D")


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
    "solid_density": 100.0,
    "shell_density": 1.0,
    "rod_density": 1.0,
    # Young's modulus values
    "solid_young_modulus": 500.0,
    "shell_young_modulus": 1000.0,
    "rod_young_modulus": 10000.0,
    # Whether the Young's modulus field is a density-normalized value (Pa/rho,
    # the solver's native convention). True (default) sends it unchanged; False
    # treats it as a true modulus in pascals and the encoder divides by density.
    "young_mod_density_normalized": True,
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
    # PDRD-specific parameter: volumetric density. The body moves
    # exactly rigidly; its asset stays a Tri surface mesh and no
    # tetrahedralization is performed for PDRD bodies.
    "pdrd_density": 100.0,
    # SAND-specific parameters. A SAND group is a faceless mesh of loose
    # vertices (grain centers); the grains share one radius and a
    # volumetric density, plus an inter-grain friction coefficient.
    "sand_grain_radius": 0.02,
    "sand_particle_mass": 1.0,
    "sand_friction": 0.0,
    # Rod-specific parameters (1D analog of shell shrink_x / solid shrink)
    "length_factor": 1.0,
    # The tetrahedralizer backend (fTetWild / TetGen) and its per-field
    # overrides are per-object now; their defaults live on the
    # AssignedObject property definitions (see ui/state_types.py).
    # Shell-specific parameters
    "enable_strain_limit": False,
    # Percentage of rest length (5.0 == 5% allowed stretch). Encoder divides
    # by 100 for the solver wire value.
    "strain_limit_percent": 5.0,
    "bend": 10.0,
    # Stiffness-proportional Rayleigh damping coefficients (seconds). 0.0
    # disables damping. deformation_damping damps stretch/membrane/solid
    # motion; bending_damping damps shell/rod bending (usually smaller).
    "deformation_damping": 0.0,
    "bending_damping": 0.0,
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
    "bend_rest_from_reference": False,
    # Direct stiffness factor for the loose-edge stitch force: the solver
    # scales the stitch gradient and Hessian by this value, with no mass or
    # dt normalization. Raise it to hold seams harder under gravity/collision
    # loads; lower it to let light (rod) vertices articulate more freely.
    "stitch_stiffness": 1.0,
    # UI visibility flags
    "show_parameters": False,
    "show_overlay_color": True,
    "show_stats": False,
    "show_pin": False,
    "show_group": True,
    "show_pdrd_hinge": False,
    "pdrd_hinge_visualize": True,
    # Pin overlay settings
    "pin_overlay_size": 12.0,
    # Collection indices
    "assigned_objects_index": -1,
    "pin_vertex_groups_index": -1,
}


# Icons by ObjectGroup.object_type, the single source for the type set.
# Each entry carries two role-specific icons: "header" is the group-type
# badge drawn once on the group header (panels.py); "object" is the
# per-object glyph drawn beside each assigned object name (ui_lists.py).
# The two vocabularies are intentionally distinct; do not collapse them.
GROUP_TYPE_ICONS = {
    "SOLID": {"header": "MESH_CUBE", "object": "MESH_CUBE"},
    "SHELL": {"header": "SURFACE_DATA", "object": "OUTLINER_OB_SURFACE"},
    "ROD": {"header": "CURVE_DATA", "object": "VIEW_ORTHO"},
    "STATIC": {"header": "FREEZE", "object": "OBJECT_ORIGIN"},
    "PDRD": {"header": "MESH_ICOSPHERE", "object": "MESH_ICOSPHERE"},
    "SAND": {"header": "PARTICLE_DATA", "object": "PARTICLE_DATA"},
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
    elif object_type == "PDRD":
        color = (intensity, 0.0, intensity, alpha)  # Magenta for PDRD
    elif object_type == "SAND":
        color = (intensity, 0.5 * intensity, 0.0, alpha)  # Tan/orange for sand
    else:
        color = (0.0, 0.0, 0.0, alpha)
    return color


def sand_radius_source_object(group):
    """First included SAND object carrying a stamped ``ppf_grain_radius``.

    The Convert op stamps the user-chosen radius on each object it converts;
    that stamped value is the locked source of truth for both the rendered
    sphere and the contact skin. Returns the Blender object, or None if no
    included object has been converted yet (so the group's
    ``sand_grain_radius`` fallback applies).
    """
    from ..core.uuid_registry import get_object_by_uuid

    for obj_ref in group.assigned_objects:
        if not obj_ref.included or not obj_ref.uuid:
            continue
        obj = get_object_by_uuid(obj_ref.uuid)
        if obj is not None and obj.get("ppf_grain_radius"):
            return obj
    return None


def sand_seeded_radius(group):
    """Locked grain radius for a SAND group.

    The Convert op stamps the user-chosen radius on the object as
    ``ppf_grain_radius`` and derives the non-overlapping seeding spacing from
    it, so that value is the single source of truth for both the rendered
    sphere and the contact skin. Returns the first included converted object's
    stamped radius, falling back to the group's ``sand_grain_radius`` for an
    object converted before the radius was stamped. Returns a float.
    """
    obj = sand_radius_source_object(group)
    if obj is not None:
        return float(obj["ppf_grain_radius"])
    return float(group.sand_grain_radius)


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


# Object types that participate in the dynamics solve (STATIC is excluded).
DYNAMIC_OBJECT_TYPES = ("SOLID", "SHELL", "ROD", "PDRD", "SAND")


def has_simulatable_dynamics(scene) -> bool:
    """Return True if any active group holds a dynamic object with assignments."""
    return any(
        group.object_type in DYNAMIC_OBJECT_TYPES
        and len(group.assigned_objects) > 0
        for group in iterate_active_object_groups(scene)
    )


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
    """Check whether the given pair of group types supports cross-stitch.

    A STATIC collider may be a stitch TARGET for any dynamic source type
    (SOLID/SHELL/ROD). The STATIC side is treated like a SHELL target: its
    surface is a plain triangle mesh whose vertex indices map 1:1 to the
    solver (a STATIC is never re-tetrahedralized), so it needs no
    re-projection. A non-moving STATIC that participates in a stitch is
    promoted into the dynamic pinned namespace at decode time so the stitch
    index can reach it (see frontend ``_populate_static``).
    """
    pair = {type_a, type_b}
    return (
        pair == {"SHELL"}
        or pair == {"SOLID"}
        or pair == {"SHELL", "SOLID"}
        or pair == {"ROD", "SHELL"}
        or pair == {"ROD", "SOLID"}
        or pair == {"ROD"}
        or pair == {"SOLID", "STATIC"}
        or pair == {"SHELL", "STATIC"}
        or pair == {"ROD", "STATIC"}
    )


def assign_display_indices(scene):
    """Assign display indices to active groups sequentially."""
    active_groups = list(iterate_active_object_groups(scene))
    for idx, group in enumerate(active_groups):
        group.index = idx
