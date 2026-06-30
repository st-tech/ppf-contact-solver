# File: object_group.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import uuid

import bpy  # pyright: ignore
from bpy.props import (  # pyright: ignore
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import PropertyGroup  # pyright: ignore

from ..models.groups import OBJECT_GROUP_DEFAULTS, get_object_type, get_vertex_group_items
from .state_types import AssignedObject, PinVertexGroupItem

# Blender requires Python to keep a reference to EnumProperty `items` strings
# returned by a callable; otherwise the C side reads freed memory and the
# dropdown shows random characters. Module-level caches below hold the latest
# items list per callback so the strings outlive the callback return.
_velocity_items_ref: list = []
_collision_window_items_ref: list = []
_tet_items_ref: list = []
_pdrd_hinge_items_ref: list = []
_bend_ref_items_ref: list = []
_material_preset_items_ref: list = []


def _build_assigned_object_enum_items(group) -> list:
    """Build EnumProperty items for the included assigned objects of a group.

    Returns a list of ``(uuid, label, "")`` tuples (label is the live object
    name when resolvable, else the stored name or a uuid prefix), with a single
    ``NONE`` fallback when no objects are assigned. The three object-selection
    callbacks share this body; each keeps its own module-level ref cache so the
    returned strings outlive the callback return (see the note above).
    """
    from ..core.uuid_registry import get_object_by_uuid

    items = []
    for assigned in group.assigned_objects:
        if assigned.included and assigned.uuid:
            obj = get_object_by_uuid(assigned.uuid)
            label = obj.name if obj else (assigned.name or assigned.uuid[:8])
            items.append((assigned.uuid, label, ""))
    if not items:
        items.append(("NONE", "None", "No objects assigned"))
    return items


def _invalidate_overlay_from_group(self=None, ctx=None):
    """Bump overlay version so velocity/group-scoped batches rebuild.

    Signature matches Blender PropertyGroup `update=` callbacks, so the
    function can be passed directly without a wrapper lambda.
    """
    from ..models.groups import invalidate_overlays
    invalidate_overlays()


def _get_material_profile_items(self, context):
    """Dynamic callback for material_profile_selection EnumProperty."""
    from ..core.profile import get_profile_names

    path = self.material_profile_path
    if not path:
        return [("NONE", "(No Profile)", "")]

    abs_path = bpy.path.abspath(path)
    names = get_profile_names(abs_path)
    if not names:
        return [("NONE", "(No Profile)", "")]
    return [(n, n, f"Material profile: {n}") for n in names]


def _on_material_profile_selected(self, context):
    """Update callback when user picks a material param profile."""
    from ..core.profile import apply_material_profile, load_profiles
    from ..core.utils import redraw_all_areas

    if self.material_profile_selection == "NONE":
        return

    abs_path = bpy.path.abspath(self.material_profile_path)
    profiles = load_profiles(abs_path)
    profile = profiles.get(self.material_profile_selection)
    if profile is None:
        return
    apply_material_profile(profile, self)
    from ..models.groups import invalidate_overlays
    invalidate_overlays()
    redraw_all_areas(context)


def _get_material_preset_items(self, context):
    """Dynamic callback for the material_preset_selection EnumProperty.

    Lists only presets whose ``object_type`` matches the group's current Type,
    so a SHELL group sees only fabric presets and a SOLID group only soft-solid
    presets; a PDRD group has no presets, so its list is just the NONE entry
    (decision D2). The returned list is stashed in a module-level
    ref so Blender does not free the strings before reading them (see the note
    by the ref caches above).
    """
    global _material_preset_items_ref
    from ..core.material_presets import get_preset_items

    _material_preset_items_ref = get_preset_items(self.object_type)
    return _material_preset_items_ref


def _on_material_preset_selected(self, context):
    """Update callback when the user picks a bundled material preset.

    Applies the preset's parameters, then resets the picker back to ``NONE`` so
    the same preset can be re-applied and the dropdown does not show a sticky
    selection. Setting the value here re-enters this callback, but the ``NONE``
    guard returns immediately, so there is no recursion.
    """
    from ..core.material_presets import apply_material_preset
    from ..core.utils import redraw_all_areas
    from ..models.groups import invalidate_overlays

    if self.material_preset_selection == "NONE":
        return
    apply_material_preset(self.material_preset_selection, self)
    self.material_preset_selection = "NONE"
    invalidate_overlays()
    redraw_all_areas(context)


def _get_pin_profile_items(self, context):
    """Dynamic callback for pin_profile_selection EnumProperty."""
    from ..core.profile import get_profile_names

    path = self.pin_profile_path
    if not path:
        return [("NONE", "(No Profile)", "")]

    abs_path = bpy.path.abspath(path)
    names = get_profile_names(abs_path)
    if not names:
        return [("NONE", "(No Profile)", "")]
    return [(n, n, f"Pin profile: {n}") for n in names]


def _on_pin_profile_selected(self, context):
    """Update callback when user picks a pin profile.

    Pin profile files store per-pin operation sets (saved by
    ``SavePinProfile``), NOT the bulk ``{pins: ...}`` format.
    Apply to the currently selected pin item via
    ``apply_pin_operations``, matching ``ReloadPinProfile``.
    """
    from ..core.profile import apply_pin_operations, load_profiles
    from ..core.utils import redraw_all_areas
    from ..models.groups import invalidate_overlays

    if self.pin_profile_selection == "NONE":
        return

    abs_path = bpy.path.abspath(self.pin_profile_path)
    profiles = load_profiles(abs_path)
    profile = profiles.get(self.pin_profile_selection)
    if profile is None:
        return
    # Apply to the currently selected pin item (not bulk apply).
    idx = self.pin_vertex_groups_index
    if idx < 0 or idx >= len(self.pin_vertex_groups):
        return
    apply_pin_operations(profile, self.pin_vertex_groups[idx])
    invalidate_overlays()
    redraw_all_areas(context)


class ObjectGroup(PropertyGroup):
    name: StringProperty(default="", description="Group name")  # pyright: ignore
    material_profile_path: StringProperty(
        name="Material Profile",
        subtype="FILE_PATH",
        default="",
        description="Path to a TOML material parameter profile file",
    )  # pyright: ignore
    material_profile_selection: EnumProperty(
        name="Material Profile",
        items=_get_material_profile_items,
        update=_on_material_profile_selected,
        description="Select a material parameter profile",
    )  # pyright: ignore
    material_preset_selection: EnumProperty(
        name="Material Preset",
        items=_get_material_preset_items,
        update=_on_material_preset_selected,
        description="Apply a bundled, physically-grounded material preset "
        "(filtered by this group's Type; applying it never changes the Type)",
        options={"SKIP_SAVE"},
    )  # pyright: ignore
    pin_profile_path: StringProperty(
        name="Pin Profile",
        subtype="FILE_PATH",
        default="",
        description="Path to a TOML pin configuration profile file",
    )  # pyright: ignore
    pin_profile_selection: EnumProperty(
        name="Pin Profile",
        items=_get_pin_profile_items,
        update=_on_pin_profile_selected,
        description="Select a pin configuration profile",
    )  # pyright: ignore
    assigned_objects: CollectionProperty(type=AssignedObject)  # pyright: ignore
    assigned_objects_index: IntProperty(default=-1)  # pyright: ignore  # Selected item in assigned objects list
    index: IntProperty(default=-1)  # pyright: ignore  # Display index for UI
    uuid: StringProperty(default="")  # pyright: ignore  # Unique identifier
    pin_vertex_groups: CollectionProperty(type=PinVertexGroupItem)  # pyright: ignore
    pin_vertex_groups_index: IntProperty(default=-1)  # pyright: ignore

    def _get_velocity_object_items(self, context):
        global _velocity_items_ref
        _velocity_items_ref = _build_assigned_object_enum_items(self)
        return _velocity_items_ref

    velocity_object_selection: EnumProperty(
        name="Object",
        items=_get_velocity_object_items,
        description="Select object for velocity keyframes",
        options={"SKIP_SAVE"},
    )  # pyright: ignore
    preview_velocity: BoolProperty(
        name="Preview Direction",
        default=False,
        description="Show velocity directions in viewport for all objects in this group",
        update=_invalidate_overlay_from_group,
    )  # pyright: ignore

    use_collision_windows: BoolProperty(
        name="Collision Active Duration Windows",
        default=False,
        description="Restrict collision detection to specific time windows per object",
    )  # pyright: ignore
    def _get_collision_window_object_items(self, context):
        global _collision_window_items_ref
        _collision_window_items_ref = _build_assigned_object_enum_items(self)
        return _collision_window_items_ref

    collision_window_object_selection: EnumProperty(
        name="Object",
        items=_get_collision_window_object_items,
        description="Select object for collision window editing",
        options={"SKIP_SAVE"},
    )  # pyright: ignore

    def _get_tet_object_items(self, context):
        global _tet_items_ref
        _tet_items_ref = _build_assigned_object_enum_items(self)
        return _tet_items_ref

    tet_object_selection: EnumProperty(
        name="Object",
        items=_get_tet_object_items,
        description="Select object whose tetrahedralizer settings to edit",
        options={"SKIP_SAVE"},
    )  # pyright: ignore

    def ensure_uuid(self):
        """Ensure this group has a UUID assigned."""
        if not self.uuid:
            self.uuid = str(uuid.uuid4())
        return self.uuid

    def reset_to_defaults(self):
        """Reset all parameters to their PropertyGroup defaults.

        This method uses the centralized OBJECT_GROUP_DEFAULTS dictionary
        to ensure single source of truth for all default values.
        """
        # Reset all properties using the centralized defaults
        for property_name, default_value in OBJECT_GROUP_DEFAULTS.items():
            if hasattr(self, property_name):
                try:
                    setattr(self, property_name, default_value)
                except Exception:
                    # Some properties might be read-only or have constraints
                    continue

        # Set color based on default object type
        self.color = get_object_type(OBJECT_GROUP_DEFAULTS["object_type"])

        # Clear collections
        self.assigned_objects.clear()
        self.pin_vertex_groups.clear()

    def update_object_type(self, _):
        self.color = get_object_type(self.object_type)
        # Ensure rod_model is always ARAP for ROD objects
        if self.object_type == "ROD":
            self.rod_model = "ARAP"
            self.use_group_bounding_box_diagonal = False
            self.contact_gap = 0.001
            self.contact_offset = 0.005
            self.computed_contact_gap = 0.001
            self.computed_contact_offset = 0.005
            self.bend = 1.0
        from .dynamics.overlay import apply_object_overlays
        apply_object_overlays()

    def update_overlay_color(self, _):
        """Update object colors when overlay color setting changes"""
        # Import here to avoid circular imports
        from .dynamics.overlay import apply_object_overlays

        apply_object_overlays()

    def _invalidate_overlay(self, context):
        from .dynamics.overlay import apply_object_overlays
        apply_object_overlays()

    object_type: EnumProperty(  # pyright: ignore
        name="Type",
        items=[
            ("SOLID", "Solid", "Solid object"),
            ("SHELL", "Shell", "Shell object"),
            ("ROD", "Rod", "Rod object"),
            ("STATIC", "Static", "Static object"),
            ("PDRD", "PDRD", "Painless Differentiable Rotation Dynamics: exactly-rigid body whose surface mesh moves as a single best-fit rigid transform (no tetrahedralization)"),
            ("SAND", "Sand", "Granular / sand body"),
        ],
        default=OBJECT_GROUP_DEFAULTS["object_type"],
        update=update_object_type,
    )
    color: FloatVectorProperty(  # pyright: ignore
        name="Color",
        subtype="COLOR",
        size=4,
        min=0.0,
        max=1.0,
        default=get_object_type("SOLID"),
        description="Color for the group",
        update=update_overlay_color,
    )
    solid_model: EnumProperty(  # pyright: ignore
        name="Model",
        items=[
            ("STABLE_NEOHOOKEAN", "Stable NeoHookean", "Stable NeoHookean model"),
            ("ARAP", "ARAP", "As-Rigid-As-Possible model"),
        ],
        default=OBJECT_GROUP_DEFAULTS["solid_model"],
    )
    shell_model: EnumProperty(  # pyright: ignore
        name="Model",
        # Blender stores ``EnumProperty`` values as the integer ``number``
        # field. Explicit numbers below freeze each identifier to its slot
        # so older ``.blend`` files keep loading as the same identifier
        # even though ``STABLE_NEOHOOKEAN`` is no longer a valid choice
        # for shell groups. The empty visible name suppresses the entry
        # from the dropdown; ``core/encoder/params.py`` coerces any
        # remaining ``STABLE_NEOHOOKEAN`` upload to ``ARAP``.
        items=[
            ("STABLE_NEOHOOKEAN", "", "", "NONE", 0),
            ("ARAP", "ARAP", "As-Rigid-As-Possible model", "NONE", 1),
            ("BARAFF_WITKIN", "Baraff-Witkin", "Baraff-Witkin model", "NONE", 2),
        ],
        default=OBJECT_GROUP_DEFAULTS["shell_model"],
    )
    rod_model: EnumProperty(  # pyright: ignore
        name="Model",
        items=[
            ("ARAP", "ARAP", "As-Rigid-As-Possible model"),
        ],
        default=OBJECT_GROUP_DEFAULTS["rod_model"],
    )
    rod_density: FloatProperty(
        name="Density (kg/m)",
        default=OBJECT_GROUP_DEFAULTS["rod_density"],
        min=0.01,
        max=10000,
        precision=2,
        description="Linear density of the rod material",
    )  # pyright: ignore
    shell_density: FloatProperty(
        name="Density (kg/m\u00b2)",
        default=OBJECT_GROUP_DEFAULTS["shell_density"],
        min=0.01,
        max=10000,
        precision=2,
        description="Area density of the shell material",
    )  # pyright: ignore
    solid_density: FloatProperty(
        name="Density (kg/m\u00b3)",
        default=OBJECT_GROUP_DEFAULTS["solid_density"],
        min=0.01,
        max=10000,
        precision=2,
        description="Volume density of the solid material",
    )  # pyright: ignore
    pdrd_density: FloatProperty(
        name="Density (kg/m³)",
        default=OBJECT_GROUP_DEFAULTS["pdrd_density"],
        min=0.01,
        max=10000,
        precision=2,
        description="Volume density of the PDRD body; mass is the density times the enclosed volume of the surface mesh",
    )  # pyright: ignore
    sand_grain_radius: FloatProperty(
        name="Grain Radius (m)",
        default=OBJECT_GROUP_DEFAULTS["sand_grain_radius"],
        min=1e-4,
        precision=4,
        description="Per-grain radius of the granular (sand) particles",
    )  # pyright: ignore
    sand_particle_mass: FloatProperty(
        name="Particle Mass (g)",
        default=OBJECT_GROUP_DEFAULTS["sand_particle_mass"],
        min=1e-6,
        max=1e6,
        precision=4,
        description="Mass of a single sand particle, in grams (sent to the solver in kilograms)",
    )  # pyright: ignore
    sand_friction: FloatProperty(
        name="Friction",
        default=OBJECT_GROUP_DEFAULTS["sand_friction"],
        min=0.0,
        precision=2,
        description="Inter-grain friction coefficient of the granular (sand) body",
    )  # pyright: ignore

    def _get_pdrd_hinge_object_items(self, context):
        global _pdrd_hinge_items_ref
        _pdrd_hinge_items_ref = _build_assigned_object_enum_items(self)
        return _pdrd_hinge_items_ref

    pdrd_hinge_object_selection: EnumProperty(
        name="Object",
        items=_get_pdrd_hinge_object_items,
        description="Select which PDRD body's hinge to edit",
        options={"SKIP_SAVE"},
    )  # pyright: ignore
    pdrd_hinge_visualize: BoolProperty(
        name="Visualize",
        default=OBJECT_GROUP_DEFAULTS["pdrd_hinge_visualize"],
        description="Draw the hinge axle gizmo in the viewport for hinged bodies in this group",
        update=_invalidate_overlay_from_group,
    )  # pyright: ignore

    def _get_bend_ref_object_items(self, context):
        global _bend_ref_items_ref
        _bend_ref_items_ref = _build_assigned_object_enum_items(self)
        return _bend_ref_items_ref

    bend_ref_object_selection: EnumProperty(
        name="Object",
        items=_get_bend_ref_object_items,
        description="Select which object's reference rest angle to edit",
        options={"SKIP_SAVE"},
    )  # pyright: ignore
    rod_young_modulus: FloatProperty(
        name="Young's Modulus (Pa/\u03c1)",
        default=OBJECT_GROUP_DEFAULTS["rod_young_modulus"],
        min=0.01,
        soft_max=10000000.0,
        max=1000000000.0,
        precision=2,
        description="Young's modulus for the rod material",
    )  # pyright: ignore
    shell_young_modulus: FloatProperty(
        name="Young's Modulus (Pa/\u03c1)",
        default=OBJECT_GROUP_DEFAULTS["shell_young_modulus"],
        min=0.01,
        soft_max=10000000.0,
        max=1000000000.0,
        precision=2,
        description="Young's modulus for the shell material",
    )  # pyright: ignore
    solid_young_modulus: FloatProperty(
        name="Young's Modulus (Pa/\u03c1)",
        default=OBJECT_GROUP_DEFAULTS["solid_young_modulus"],
        min=0.01,
        soft_max=10000000.0,
        max=1000000000.0,
        precision=2,
        description="Young's modulus for the solid material",
    )  # pyright: ignore
    young_mod_density_normalized: BoolProperty(
        name="Density-Normalized (Pa/\u03c1)",
        default=OBJECT_GROUP_DEFAULTS["young_mod_density_normalized"],
        description=(
            "Sets what the Young's Modulus field above means. When enabled "
            "(default) it is a density-normalized value in Pa/\u03c1, the solver's "
            "native convention: changing a body's density alone leaves its "
            "motion unchanged. Keep it on to match existing scenes. Disable it "
            "to enter a true Young's modulus in pascals (for example a value "
            "from a material reference table); the addon then divides it by "
            "this group's density before sending it to the solver, so a denser "
            "body of the same material is correspondingly stiffer to move"
        ),
    )  # pyright: ignore
    shell_poisson_ratio: FloatProperty(
        name="Poisson's Ratio",
        default=OBJECT_GROUP_DEFAULTS["shell_poisson_ratio"],
        min=0.0,
        max=0.4999,
        precision=4,
        description="Poisson's ratio for the shell material",
    )  # pyright: ignore
    solid_poisson_ratio: FloatProperty(
        name="Poisson's Ratio",
        default=OBJECT_GROUP_DEFAULTS["solid_poisson_ratio"],
        min=0.0,
        max=0.4999,
        precision=4,
        description="Poisson's ratio for the solid material",
    )  # pyright: ignore
    friction: FloatProperty(
        name="Friction",
        default=OBJECT_GROUP_DEFAULTS["friction"],
        min=0.0,
        max=1.0,
        precision=2,
        description="Friction coefficient",
    )  # pyright: ignore
    contact_gap: FloatProperty(
        name="Contact Gap",
        default=OBJECT_GROUP_DEFAULTS["contact_gap"],
        min=0.001,
        max=0.01,
        precision=3,
        description="Contact gap (scaled by World Scale before use)",
    )  # pyright: ignore
    use_group_bounding_box_diagonal: BoolProperty(
        name="Use Group Bounding Box Diagonal",
        default=OBJECT_GROUP_DEFAULTS["use_group_bounding_box_diagonal"],
        description="Use group bounding box diagonal to calculate contact gap",
    )  # pyright: ignore
    contact_gap_rat: FloatProperty(
        name="Contact Gap Ratio",
        default=OBJECT_GROUP_DEFAULTS["contact_gap_rat"],
        min=1e-5,
        max=1.0,
        precision=5,
        description="Contact gap as ratio of group bounding box diagonal",
    )  # pyright: ignore
    computed_contact_gap: FloatProperty(
        name="Computed Contact Gap",
        default=OBJECT_GROUP_DEFAULTS["computed_contact_gap"],
        description="Hidden property storing the absolute computed contact gap value",
        options={"HIDDEN"},
    )  # pyright: ignore
    contact_offset: FloatProperty(
        name="Contact Offset",
        default=OBJECT_GROUP_DEFAULTS["contact_offset"],
        min=0.0,
        max=0.03,
        precision=3,
        description="Contact offset (scaled by World Scale before use)",
    )  # pyright: ignore
    contact_offset_rat: FloatProperty(
        name="Contact Offset Ratio",
        default=OBJECT_GROUP_DEFAULTS["contact_offset_rat"],
        min=0.0,
        max=1.0,
        precision=5,
        description="Contact offset as ratio of group bounding box diagonal",
    )  # pyright: ignore
    computed_contact_offset: FloatProperty(
        name="Computed Contact Offset",
        default=OBJECT_GROUP_DEFAULTS["computed_contact_offset"],
        description="Hidden property storing the absolute computed contact offset value",
        options={"HIDDEN"},
    )  # pyright: ignore
    enable_strain_limit: BoolProperty(
        name="Enable Strain Limit",
        default=OBJECT_GROUP_DEFAULTS["enable_strain_limit"],
        description="Enable strain limit for the shell material",
    )  # pyright: ignore
    # Stored and displayed as a percentage (1% allows 1% stretch). The encoder
    # converts to the wire fraction (value / 100) at the solver boundary. Named
    # distinctly from the old fractional `strain_limit` so pre-existing .blend
    # files fall back to the default here instead of reading a misscaled value.
    strain_limit_percent: FloatProperty(
        name="Strain Limit",
        default=OBJECT_GROUP_DEFAULTS["strain_limit_percent"],
        min=0.0,
        max=100.0,
        precision=2,
        subtype="PERCENTAGE",
        description="Maximum stretch beyond rest length, as a percentage "
        "(1% allows 1% stretch)",
    )  # pyright: ignore
    enable_inflate: BoolProperty(
        name="Inflate",
        default=OBJECT_GROUP_DEFAULTS["enable_inflate"],
        description="Enable inflation pressure for the shell surface",
    )  # pyright: ignore
    inflate_pressure: FloatProperty(
        name="Pressure (Pa)",
        default=OBJECT_GROUP_DEFAULTS["inflate_pressure"],
        min=0.0,
        soft_max=100.0,
        precision=3,
        description="Inflation pressure applied uniformly along face normals (Pa)",
    )  # pyright: ignore
    enable_plasticity: BoolProperty(
        name="Plasticity",
        default=OBJECT_GROUP_DEFAULTS["enable_plasticity"],
        description="Enable plasticity (permanent deformation)",
    )  # pyright: ignore
    plasticity: FloatProperty(
        name="Theta",
        default=OBJECT_GROUP_DEFAULTS["plasticity"],
        min=0.0,
        soft_max=10.0,
        precision=2,
        description="Plasticity rate constant (0=elastic, 1=63%/s, 5=99%/s)",
    )  # pyright: ignore
    plasticity_threshold: FloatProperty(
        name="Threshold",
        default=OBJECT_GROUP_DEFAULTS["plasticity_threshold"],
        min=0.0,
        soft_max=1.0,
        precision=3,
        description="Dead zone around S=1. Plasticity only activates when |S-1| > threshold",
    )  # pyright: ignore
    enable_bend_plasticity: BoolProperty(
        name="Bend Plasticity",
        default=OBJECT_GROUP_DEFAULTS["enable_bend_plasticity"],
        description="Enable bending plasticity (hinge / rod-joint rest angle drifts)",
    )  # pyright: ignore
    bend_plasticity: FloatProperty(
        name="Bend Theta",
        default=OBJECT_GROUP_DEFAULTS["bend_plasticity"],
        min=0.0,
        soft_max=10.0,
        precision=2,
        description="Bending plasticity rate constant (0=elastic, higher=faster creep)",
    )  # pyright: ignore
    bend_plasticity_threshold: FloatProperty(
        name="Bend Threshold",
        default=OBJECT_GROUP_DEFAULTS["bend_plasticity_threshold"],
        min=0.0,
        soft_max=3.14159,
        precision=3,
        subtype="ANGLE",
        unit="ROTATION",
        description="Dead zone around rest angle. Plasticity activates when |θ-θ_rest| > threshold",
    )  # pyright: ignore
    bend_rest_angle_source: EnumProperty(
        name="Rest Angle",
        items=[
            ("FLAT", "Flat / Straight",
             "Start from flat (shell hinge θ₀=0) or straight (rod θ₀=π)"),
            ("FROM_GEOMETRY", "From Initial Geometry",
             "Use the angle of the initial pose as rest"),
        ],
        default=OBJECT_GROUP_DEFAULTS.get("bend_rest_angle_source", "FLAT"),
        description="How the initial bending rest angle is populated",
    )  # pyright: ignore
    bend_rest_from_reference: BoolProperty(
        name="From Reference Geometry",
        default=OBJECT_GROUP_DEFAULTS.get("bend_rest_from_reference", False),
        description=(
            "Enable per-object reference rest angles. Pick a reference object "
            "(a topological copy whose vertices were moved, e.g. by a modifier "
            "or geometry nodes) for individual shell or rod objects; each "
            "object's bending rest angle is then computed from its reference "
            "instead of its own initial pose, overriding the group Rest Angle "
            "source for that object"
        ),
    )  # pyright: ignore
    bend: FloatProperty(
        name="Bend Stiffness",
        default=OBJECT_GROUP_DEFAULTS["bend"],
        min=0.0,
        soft_max=100.0,
        precision=2,
        description="Bending stiffness for the shell material",
    )  # pyright: ignore
    deformation_damping: FloatProperty(
        name="Deformation Damping",
        default=OBJECT_GROUP_DEFAULTS["deformation_damping"],
        min=0.0,
        soft_max=0.1,
        precision=4,
        description=(
            "Stiffness-proportional Rayleigh damping for stretch/membrane/solid "
            "deformation (seconds). Damps high-frequency deformation; 0 disables it"
        ),
    )  # pyright: ignore
    bending_damping: FloatProperty(
        name="Bending Damping",
        default=OBJECT_GROUP_DEFAULTS["bending_damping"],
        min=0.0,
        soft_max=0.1,
        precision=4,
        description=(
            "Stiffness-proportional Rayleigh damping for shell and rod bending "
            "(seconds), usually smaller than deformation damping; 0 disables it"
        ),
    )  # pyright: ignore
    shrink_x: FloatProperty(
        name="Shrink X",
        default=OBJECT_GROUP_DEFAULTS["shrink_x"],
        min=0.1,
        soft_max=2.0,
        precision=2,
        description="Scale factor along the UV X direction; <1 shrinks, >1 extends",
    )  # pyright: ignore
    shrink_y: FloatProperty(
        name="Shrink Y",
        default=OBJECT_GROUP_DEFAULTS["shrink_y"],
        min=0.1,
        soft_max=2.0,
        precision=2,
        description="Scale factor along the UV Y direction; <1 shrinks, >1 extends",
    )  # pyright: ignore
    shrink: FloatProperty(
        name="Shrink",
        default=OBJECT_GROUP_DEFAULTS["shrink"],
        min=0.1,
        soft_max=2.0,
        precision=2,
        description="Uniform scale factor for solid material; <1 shrinks, >1 extends",
    )  # pyright: ignore
    length_factor: FloatProperty(
        name="Shrink",
        default=OBJECT_GROUP_DEFAULTS["length_factor"],
        min=0.1,
        soft_max=2.0,
        precision=2,
        description=(
            "Rest-length scale for rod edges; <1 shrinks the rod, "
            ">1 extends it. Maps to the solver's 'length-factor' "
            "rod parameter"
        ),
    )  # pyright: ignore

    # The tetrahedralizer backend picker and its overrides moved to a
    # per-object surface on AssignedObject (see state_types.py); the group
    # only keeps tet_object_selection above to choose which object to edit,
    # mirroring Velocity Overwrite.

    stitch_stiffness: FloatProperty(
        name="Stitch Stiffness",
        default=OBJECT_GROUP_DEFAULTS["stitch_stiffness"],
        min=0.0,
        soft_max=10000.0,
        precision=2,
        description="Stiffness factor for stitch constraints",
    )  # pyright: ignore
    show_parameters: BoolProperty(
        name="Material Params",
        default=OBJECT_GROUP_DEFAULTS["show_parameters"],
        description="Toggle visibility of group parameters",
    )  # pyright: ignore
    active: BoolProperty(
        name="Active",
        default=OBJECT_GROUP_DEFAULTS["active"],
        description="Indicates if the group is active",
    )  # pyright: ignore
    show_overlay_color: BoolProperty(  # pyright: ignore
        name="Overlay Color",
        default=OBJECT_GROUP_DEFAULTS["show_overlay_color"],
        description="Toggle visibility of overlay color",
        update=update_overlay_color,
    )
    show_stats: BoolProperty(
        name="Stats",
        default=OBJECT_GROUP_DEFAULTS["show_stats"],
        description="Toggle visibility of object information",
    )  # pyright: ignore
    show_pin: BoolProperty(
        name="Pins",
        default=OBJECT_GROUP_DEFAULTS["show_pin"],
        description="Toggle visibility of pinned vertex group",
    )  # pyright: ignore
    show_pdrd_hinge: BoolProperty(
        name="Hinge",
        default=OBJECT_GROUP_DEFAULTS["show_pdrd_hinge"],
        description="Expand the per-object PDRD hinge controls",
    )  # pyright: ignore
    pin_overlay_size: FloatProperty(
        name="Pin Size",
        default=OBJECT_GROUP_DEFAULTS["pin_overlay_size"],
        min=4.0,
        max=32.0,
        description="Size of pin overlay circles",
        update=_invalidate_overlay,
    )  # pyright: ignore
    show_group: BoolProperty(
        name="Show Group",
        default=OBJECT_GROUP_DEFAULTS["show_group"],
        description="Toggle visibility of group contents",
    )  # pyright: ignore
    pin_vertex_group_items: EnumProperty(
        name="Vertex Groups",
        items=get_vertex_group_items,
        description="Select a vertex group to pin",
    )  # pyright: ignore
