# File: state_types.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

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


def _invalidate_overlay(self=None, ctx=None):
    """Bump overlay version for any batch cache that tracks `overlay_version`.

    Suitable for use as a PropertyGroup `update=` callback.
    """
    from ..models.groups import invalidate_overlays
    invalidate_overlays()


class FetchedFrameItem(PropertyGroup):
    value: IntProperty(name="Frame", default=0)  # pyright: ignore


class CheckpointFrameItem(PropertyGroup):
    """A single saved-checkpoint frame for the Resume-From dialog UIList."""

    frame: IntProperty(name="Frame", default=0)  # pyright: ignore


class SaveCheckpointFrameItem(PropertyGroup):
    """A user-requested frame at which the solver saves a resumable state.

    ``frame`` is the Blender 1-based frame the artist enters in the "Save
    Checkpoints" UIList. The encoder converts it to the solver's 0-based
    frame index before it ships to the backend.
    """

    frame: IntProperty(name="Frame", default=1, min=1)  # pyright: ignore


class MergePairItem(PropertyGroup):
    object_a: StringProperty(name="Object A", default="")  # pyright: ignore
    object_b: StringProperty(name="Object B", default="")  # pyright: ignore
    object_a_uuid: StringProperty()  # pyright: ignore
    object_b_uuid: StringProperty()  # pyright: ignore
    cross_stitch_json: StringProperty(
        name="Cross Stitch JSON",
        default="",
        description="Explicit cross-object stitch data captured at snap time",
        options={"HIDDEN"},
    )  # pyright: ignore
    stitch_stiffness: FloatProperty(
        name="Stitch Stiffness",
        # Direct stiffness factor for the cross-object stitch force: the solver
        # scales the stitch gradient and Hessian by this value, with no mass or
        # dt normalization. Raise it to hold seams harder.
        default=1.0,
        min=0.0,
        soft_max=10000.0,
        precision=2,
        description="Stiffness for generated cross-object stitch constraints",
    )  # pyright: ignore
    show_stitch: BoolProperty(
        name="Show Stitch",
        default=True,
        description="Show stitch visualization for this merge pair",
        update=_invalidate_overlay,
    )  # pyright: ignore


class DynParamKeyframe(PropertyGroup):
    """A single keyframe for a dynamic scene parameter."""

    frame: IntProperty(
        name="Frame",
        default=1,
        min=1,
        description="Frame number for this keyframe",
        update=_invalidate_overlay,
    )  # pyright: ignore
    gravity_value: FloatVectorProperty(
        name="Gravity (m/s\u00b2)",
        subtype="XYZ",
        size=3,
        default=(0.0, 0.0, -9.8),
        precision=2,
        description="Gravity acceleration vector",
        update=_invalidate_overlay,
    )  # pyright: ignore
    wind_direction_value: FloatVectorProperty(
        name="Direction",
        subtype="XYZ",
        size=3,
        default=(0.0, 0.0, 0.0),
        description="Wind direction vector",
        update=_invalidate_overlay,
    )  # pyright: ignore
    wind_strength_value: FloatProperty(
        name="Strength (m/s)",
        default=0.0,
        min=0.0,
        max=1000.0,
        precision=2,
        description="Wind strength",
        update=_invalidate_overlay,
    )  # pyright: ignore
    scalar_value: FloatProperty(
        name="Value",
        default=0.0,
        min=0.0,
        precision=4,
        description="Scalar parameter value",
        update=_invalidate_overlay,
    )  # pyright: ignore
    use_hold: BoolProperty(
        name="Hold",
        default=False,
        description="Hold the previous keyframe value (step function)",
        update=_invalidate_overlay,
    )  # pyright: ignore


class DynParamItem(PropertyGroup):
    """A dynamic parameter entry with its keyframes."""

    param_type: EnumProperty(
        name="Parameter",
        items=[
            ("GRAVITY", "Gravity", "Dynamic gravity"),
            ("WIND", "Wind", "Dynamic wind"),
            ("AIR_DENSITY", "Air Density", "Dynamic air density"),
            ("AIR_FRICTION", "Air Friction", "Dynamic air friction"),
            ("VERTEX_AIR_DAMP", "Vertex Air Damping", "Dynamic vertex air damping"),
        ],
        update=_invalidate_overlay,
    )  # pyright: ignore
    keyframes: CollectionProperty(type=DynParamKeyframe)  # pyright: ignore
    keyframes_index: IntProperty(default=0)  # pyright: ignore


class InvisibleColliderKeyframe(PropertyGroup):
    """A single keyframe for an invisible collider."""

    frame: IntProperty(
        name="Frame",
        default=1,
        min=1,
        description="Frame number for this keyframe",
        update=lambda self, ctx: _invalidate_collider_overlay(),
    )  # pyright: ignore
    position: FloatVectorProperty(
        name="Position",
        subtype="XYZ",
        size=3,
        default=(0.0, 0.0, 0.0),
        precision=3,
        description="Collider position at this keyframe",
        update=lambda self, ctx: _invalidate_collider_overlay(),
    )  # pyright: ignore
    radius: FloatProperty(
        name="Radius",
        default=1.0,
        min=0.001,
        precision=3,
        description="Sphere radius at this keyframe",
        update=lambda self, ctx: _invalidate_collider_overlay(),
    )  # pyright: ignore
    use_hold: BoolProperty(
        name="Hold",
        default=False,
        description="Hold the previous keyframe value (step function)",
        update=lambda self, ctx: _invalidate_collider_overlay(),
    )  # pyright: ignore


def _redraw_viewports():
    """Tag all 3D viewports for redraw."""
    if bpy.context.screen is None:
        return
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            area.tag_redraw()


def _invalidate_collider_overlay():
    """Bump overlay version so the collider batch cache rebuilds.

    Plain tag_redraw isn't enough: overlay.py gates collider batch rebuilds
    on state.overlay_version, so the cache keeps the pre-toggle batches
    until the version bumps.
    """
    from ..models.groups import invalidate_overlays
    invalidate_overlays()


class InvisibleColliderItem(PropertyGroup):
    """An invisible collider (wall or sphere) with optional keyframes."""

    collider_type: EnumProperty(
        name="Type",
        items=[
            ("WALL", "Wall", "Infinite plane collider"),
            ("SPHERE", "Sphere", "Sphere collider"),
        ],
        update=lambda self, ctx: _invalidate_collider_overlay(),
    )  # pyright: ignore
    name: StringProperty(
        name="Name",
        default="",
        description="Display name for this collider",
    )  # pyright: ignore
    position: FloatVectorProperty(
        name="Position",
        subtype="XYZ",
        size=3,
        default=(0.0, 0.0, 0.0),
        precision=3,
        description="Point on plane (wall) or center (sphere)",
        update=lambda self, ctx: _invalidate_collider_overlay(),
    )  # pyright: ignore
    normal: FloatVectorProperty(
        name="Normal",
        subtype="XYZ",
        size=3,
        default=(0.0, 0.0, 1.0),
        precision=3,
        description="Outer normal direction (wall only)",
        update=lambda self, ctx: _invalidate_collider_overlay(),
    )  # pyright: ignore
    radius: FloatProperty(
        name="Radius",
        default=1.0,
        min=0.001,
        precision=3,
        description="Sphere radius",
        update=lambda self, ctx: _invalidate_collider_overlay(),
    )  # pyright: ignore
    hemisphere: BoolProperty(
        name="Hemisphere",
        default=False,
        description="Bowl shape (top half open)",
        update=lambda self, ctx: _invalidate_collider_overlay(),
    )  # pyright: ignore
    invert: BoolProperty(
        name="Invert",
        default=False,
        description="Collision on inside of sphere",
        update=lambda self, ctx: _invalidate_collider_overlay(),
    )  # pyright: ignore
    contact_gap: FloatProperty(
        name="Contact Gap",
        default=0.001,
        min=0.0,
        precision=4,
        description="Contact gap tolerance",
    )  # pyright: ignore
    friction: FloatProperty(
        name="Friction",
        default=0.0,
        min=0.0,
        max=1.0,
        precision=2,
        description="Friction coefficient",
    )  # pyright: ignore
    thickness: FloatProperty(
        name="Thickness",
        default=1.0,
        min=1e-5,
        precision=4,
        subtype="DISTANCE",
        unit="LENGTH",
        description=(
            "Maximum penetration depth. If a vertex is embedded deeper than "
            "this inside the collider, it passes through — contact and CCD "
            "are both ignored. Must be > 0."
        ),
    )  # pyright: ignore
    enable_active_duration: BoolProperty(
        name="Active Duration",
        default=False,
        description="If set, collider stops acting at the specified Blender frame",
        update=lambda self, ctx: _invalidate_collider_overlay(),
    )  # pyright: ignore
    active_duration: IntProperty(
        name="Active Until (frame)",
        default=60,
        min=1,
        description="First Blender frame at which this collider is no longer active. The cutoff is exclusive: frame < END_FRAME is active, frame >= END_FRAME is not. Substeps of the transition TO END_FRAME already run without the collider, so the displayed state carries no residual effect",
        update=lambda self, ctx: _invalidate_collider_overlay(),
    )  # pyright: ignore
    show_preview: BoolProperty(
        name="Preview",
        default=False,
        description="Show collider preview in viewport",
        update=lambda self, ctx: _invalidate_collider_overlay(),
    )  # pyright: ignore
    keyframes: CollectionProperty(type=InvisibleColliderKeyframe)  # pyright: ignore
    keyframes_index: IntProperty(default=0)  # pyright: ignore


class VelocityKeyframe(PropertyGroup):
    frame: IntProperty(
        name="Frame", default=1, min=1,
        description="Frame at which this velocity is applied",
        update=_invalidate_overlay,
    )  # pyright: ignore
    # Per-component overwrite gates. A keyframe overwrites the translational
    # velocity only when enable_translational is on, and the angular velocity
    # only when enable_angular is on (either or both). The two are
    # independent: a pure-spin keyframe (angular only) leaves the translation
    # untouched, and a pure-translation keyframe (linear only) emits no spin.
    enable_translational: BoolProperty(
        name="Enable Translational Velocity Overwrite",
        default=True,
        description="Overwrite the object's translational velocity at this frame",
        update=_invalidate_overlay,
    )  # pyright: ignore
    direction: FloatVectorProperty(
        name="Direction", subtype="XYZ", size=3,
        default=(0.0, 0.0, 0.0),
        description="Velocity direction (normalized before use)",
        update=_invalidate_overlay,
    )  # pyright: ignore
    speed: FloatProperty(
        name="Speed (m/s)", default=0.0, min=0.0, precision=2,
        description="Velocity magnitude",
        update=_invalidate_overlay,
    )  # pyright: ignore
    # Angular (spin) component, shown only for solid / shell / PDRD. The axis
    # is one of the body's principal axes (like the PDRD hinge axle); the
    # solver resolves the world axis from the live geometry each firing, so it
    # tracks the simulated pose. Speed is signed (right-hand rule).
    enable_angular: BoolProperty(
        name="Enable Angular Velocity Overwrite",
        default=False,
        description="Overwrite the object's angular (spin) velocity at this frame",
        update=_invalidate_overlay,
    )  # pyright: ignore
    angular_axis: EnumProperty(
        name="Spin Axis",
        items=[
            ("PC1", "Principal Axis 1", "Largest-extent principal axis, resolved dynamically from the simulated geometry"),
            ("PC2", "Principal Axis 2", "Middle-extent principal axis, resolved dynamically from the simulated geometry"),
            ("PC3", "Principal Axis 3", "Smallest-extent principal axis (the usual axle for a flat gear or disk), resolved dynamically"),
            ("X", "World X", "Fixed world X axis"),
            ("Y", "World Y", "Fixed world Y axis"),
            ("Z", "World Z", "Fixed world Z axis"),
            ("CUSTOM", "Custom Axis", "User-specified fixed world-space axis (set below)"),
        ],
        default="PC3",
        description=(
            "Axis to spin about. Principal axes (PC1-3) track the simulated "
            "geometry; World X/Y/Z and Custom are fixed world-space directions"
        ),
        update=_invalidate_overlay,
    )  # pyright: ignore
    angular_axis_custom: FloatVectorProperty(
        name="Custom Axis", subtype="XYZ", size=3,
        default=(0.0, 0.0, 1.0),
        description="Custom world-space spin axis (normalized before use); used when Spin Axis is Custom",
        update=_invalidate_overlay,
    )  # pyright: ignore
    angular_speed: FloatProperty(
        name="Angular Speed (°/s)", default=0.0, precision=2,
        description="Signed spin speed in degrees per second about the chosen axis (0 = no spin)",
        update=_invalidate_overlay,
    )  # pyright: ignore
    preview: BoolProperty(
        name="Preview Direction",
        default=False,
        description="Show velocity direction in viewport",
        update=_invalidate_overlay,
    )  # pyright: ignore


class CollisionWindowEntry(PropertyGroup):
    frame_start: IntProperty(
        name="Start Frame", default=1, min=1,
        description="Frame when collision becomes active",
    )  # pyright: ignore
    frame_end: IntProperty(
        name="End Frame", default=60, min=1,
        description="Frame when collision becomes inactive",
    )  # pyright: ignore


class StaticOpItem(PropertyGroup):
    """UI-assigned move/spin/scale op for a static moving object.

    The three supported ops mirror the subset of pin ops that make sense
    for whole-object motion. They compose in list order inside the
    pin-shell that drives an animated static object on the solver side.
    """

    def _on_overlay_changed(self, context):
        from ..models.groups import invalidate_overlays
        invalidate_overlays()

    show_overlay: BoolProperty(
        name="Show Overlay",
        default=False,
        description="Show this op's preview in the 3D viewport",
        update=_on_overlay_changed,
    )  # pyright: ignore
    op_type: EnumProperty(
        name="Type",
        items=[
            ("MOVE_BY", "Move By", "Translate by a delta over time"),
            ("SPIN", "Spin", "Rotate around an axis over time"),
            ("SCALE", "Scale", "Scale from a center over time"),
        ],
        default="MOVE_BY",
        update=_on_overlay_changed,
    )  # pyright: ignore
    # Move By
    delta: FloatVectorProperty(
        name="Delta (m)", subtype="XYZ", default=(0.0, 0.0, 0.0),
        update=_on_overlay_changed,
    )  # pyright: ignore
    # Spin, always around the object origin
    spin_axis: FloatVectorProperty(
        name="Axis", subtype="XYZ", default=(0.0, 0.0, 1.0),
        update=_on_overlay_changed,
    )  # pyright: ignore
    spin_angular_velocity: FloatProperty(
        name="Angular Velocity (\u00b0/s)",
        default=360.0,
        description="Degrees per second",
        update=_on_overlay_changed,
    )  # pyright: ignore
    # Scale, always around the object origin
    scale_factor: FloatProperty(name="Factor", default=1.0, min=0.01, update=_on_overlay_changed)  # pyright: ignore
    # Common time range (in frames; converted to seconds at encode time)
    frame_start: IntProperty(name="Start", default=1, min=1)  # pyright: ignore
    frame_end: IntProperty(name="End", default=60, min=1)  # pyright: ignore
    transition: EnumProperty(
        name="Transition",
        items=[
            ("LINEAR", "Linear", "Linear interpolation"),
            ("SMOOTH", "Smooth", "Smoothstep interpolation"),
        ],
        default="LINEAR",
    )  # pyright: ignore


class AssignedObject(PropertyGroup):
    name: StringProperty()  # pyright: ignore
    uuid: StringProperty()  # pyright: ignore

    def update_included(self, context):
        """Update overlays when included state changes"""
        from ..ui.dynamics.overlay import apply_object_overlays

        apply_object_overlays()

    included: BoolProperty(
        name="Include",
        default=True,
        description="Include this object in the simulation",
        update=update_included,
    )  # pyright: ignore
    velocity_keyframes: CollectionProperty(type=VelocityKeyframe)  # pyright: ignore
    velocity_keyframes_index: IntProperty(default=0)  # pyright: ignore
    collision_windows: CollectionProperty(type=CollisionWindowEntry)  # pyright: ignore
    collision_windows_index: IntProperty(default=0)  # pyright: ignore
    # PDRD hinge joint (per object): pin the body and lock its rotation to a
    # single principal (PCA) axis. Per-object because one PDRD group can hold
    # several bodies (e.g. a gear train), each on its own axle.
    pdrd_hinge_enable: BoolProperty(
        name="Hinge",
        default=False,
        description=(
            "Pin this PDRD body and lock its rotation to a single principal "
            "axis (a hinge / pin joint). Build gears by hinging each gear to "
            "its axle and letting tooth contact transmit the torque"
        ),
        update=_invalidate_overlay,
    )  # pyright: ignore
    pdrd_hinge_axis: EnumProperty(
        name="Axle",
        items=[
            ("0", "Principal Axis 1", "Spin about the largest-extent principal axis of the rest shape"),
            ("1", "Principal Axis 2", "Spin about the middle-extent principal axis of the rest shape"),
            ("2", "Principal Axis 3", "Spin about the smallest-extent principal axis (the usual axle for a flat gear or disk)"),
        ],
        default="2",
        description="Which principal (PCA) axis of the rest shape is the free hinge axle, like Blender's torque-axis dropdown",
        update=_invalidate_overlay,
    )  # pyright: ignore
    # Per-object bending reference rest angle (SHELL). When enabled with a
    # valid reference object set, this object's bending rest angle is computed
    # from the reference geometry (a topological copy whose vertices were
    # moved, e.g. by a modifier or geometry nodes) instead of from its own
    # initial pose. This overrides the group's Rest Angle source for just this
    # object. The reference is stored by UUID so it survives renames;
    # ``bend_ref_name`` is kept only as a fallback display label.
    bend_ref_enable: BoolProperty(
        name="Enable Reference Rest Angle",
        default=False,
        description=(
            "Compute this object's bending rest angle from a reference "
            "object's geometry instead of its own initial pose. Overrides the "
            "group's Rest Angle source for this object"
        ),
    )  # pyright: ignore
    bend_ref_uuid: StringProperty(default="")  # pyright: ignore
    bend_ref_name: StringProperty(default="")  # pyright: ignore
    static_ops: CollectionProperty(type=StaticOpItem)  # pyright: ignore
    static_ops_index: IntProperty(default=-1, options={"HIDDEN"})  # pyright: ignore

    # Per-object tetrahedralizer config (SOLID only; ignored otherwise).
    # The backend picker and its knobs live on the object, not the group,
    # so each SOLID mesh in a group can choose fTetWild or TetGen and its
    # own overrides (mirrors the per-object Velocity Overwrite surface).
    #
    # Numeric IDs are explicit: Blender stores EnumProperty values as the
    # item's integer when none is given, so renaming/reordering items
    # would silently remap the backend on existing .blend files.
    tet_backend: EnumProperty(
        name="Tetrahedralizer",
        items=[
            (
                "FTETWILD", "fTetWild",
                "Tolerant remesher: handles open, cracked, or non-manifold "
                "input, but resamples the surface so input vertices are "
                "reconstructed through a surface map",
                0,
            ),
            (
                "TETGEN", "TetGen",
                "Preserves the input surface exactly (1-1 vertex map); "
                "requires a clean, closed, manifold mesh",
                1,
            ),
        ],
        default="FTETWILD",
    )  # pyright: ignore
    # fTetWild per-field overrides. Each value is forwarded to pytetwild
    # only when its override flag is on; with none set, fTetWild defaults
    # apply.
    ftetwild_override_edge_length_fac: BoolProperty(name="Override", default=False)  # pyright: ignore
    ftetwild_edge_length_fac: FloatProperty(
        name="Edge Length Factor",
        default=0.05, min=1e-4, max=1.0, precision=4,
        description="Ideal tet edge length as fraction of bbox diagonal (fTetWild -l)",
    )  # pyright: ignore
    ftetwild_override_epsilon: BoolProperty(name="Override", default=False)  # pyright: ignore
    ftetwild_epsilon: FloatProperty(
        name="Epsilon",
        default=1e-3, min=1e-6, max=1.0, precision=6,
        description="Envelope size as fraction of bbox diagonal (fTetWild -e)",
    )  # pyright: ignore
    ftetwild_override_stop_energy: BoolProperty(name="Override", default=False)  # pyright: ignore
    ftetwild_stop_energy: FloatProperty(
        name="Stop Energy",
        default=10.0, min=3.0, max=1000.0,
        description="AMIPS energy threshold; larger = faster, lower quality",
    )  # pyright: ignore
    ftetwild_override_num_opt_iter: BoolProperty(name="Override", default=False)  # pyright: ignore
    ftetwild_num_opt_iter: IntProperty(
        name="Max Opt Iterations",
        default=80, min=1, max=1000,
        description="Maximum fTetWild optimization passes",
    )  # pyright: ignore
    ftetwild_override_optimize: BoolProperty(name="Override", default=False)  # pyright: ignore
    ftetwild_optimize: BoolProperty(
        name="Optimize", default=True,
        description="Improve cell quality (slower)",
    )  # pyright: ignore
    ftetwild_override_simplify: BoolProperty(name="Override", default=False)  # pyright: ignore
    ftetwild_simplify: BoolProperty(
        name="Simplify Input", default=True,
        description="Simplify the input surface before tetrahedralization",
    )  # pyright: ignore
    ftetwild_override_coarsen: BoolProperty(name="Override", default=False)  # pyright: ignore
    ftetwild_coarsen: BoolProperty(
        name="Coarsen Output", default=False,
        description="Coarsen output while preserving quality",
    )  # pyright: ignore
    # TetGen per-field overrides. The surface is always preserved (nobisect
    # / -Y); these only tune the interior refinement.
    tetgen_override_min_ratio: BoolProperty(name="Override", default=False)  # pyright: ignore
    tetgen_min_ratio: FloatProperty(
        name="Min Radius-Edge Ratio",
        default=2.0, min=1.0, soft_max=5.0, precision=2,
        description=(
            "TetGen quality bound (-q): smaller forces rounder cells via "
            "more interior Steiner points. The input surface is never "
            "touched"
        ),
    )  # pyright: ignore
    tetgen_override_max_volume: BoolProperty(name="Override", default=False)  # pyright: ignore
    tetgen_max_volume: FloatProperty(
        name="Max Tet Volume",
        default=0.0, min=0.0, precision=6,
        description=(
            "TetGen maximum tetrahedron volume (-a), in object units; caps "
            "interior cell size for a finer mesh. 0 leaves it uncapped"
        ),
    )  # pyright: ignore


class PinOperation(PropertyGroup):

    def _on_overlay_changed(self, context):
        from ..models.groups import invalidate_overlays
        invalidate_overlays()

    show_overlay: BoolProperty(
        name="Show Overlay",
        default=False,
        description="Show operation preview in viewport",
        update=_on_overlay_changed,
    )  # pyright: ignore
    # Numeric IDs are explicit because Blender's EnumProperty stores values
    # as integers when no number is given; auto-numbering by list order means
    # removing or reordering an item silently renames every saved op in
    # existing .blend files. Slot 0 is reserved for the legacy EMBEDDED_MOVE
    # op so files saved before that variant was retired keep decoding their
    # remaining ops to the right identifier. Filter `EMBEDDED_MOVE` out at
    # the encoder; never offer it from the Add menu.
    op_type: EnumProperty(
        name="Type",
        items=[
            ("EMBEDDED_MOVE", "Embedded Move", "Deprecated: kept only so legacy .blend files decode correctly", "NONE", 0),
            ("MOVE_BY", "Move By", "Move by delta over time range", "NONE", 1),
            ("SPIN", "Spin", "Rotate around axis", "NONE", 2),
            ("SCALE", "Scale", "Scale from center", "NONE", 3),
            ("TORQUE", "Torque", "Apply rotational force around PCA axis", "NONE", 4),
        ],
        default="MOVE_BY",
        update=_on_overlay_changed,
    )  # pyright: ignore
    # Move By params
    delta: FloatVectorProperty(name="Delta (m)", subtype="XYZ", default=(0, 0, 0), update=_on_overlay_changed)  # pyright: ignore
    # Spin params
    spin_axis: FloatVectorProperty(name="Axis", subtype="XYZ", default=(1, 0, 0), update=_on_overlay_changed)  # pyright: ignore
    spin_angular_velocity: FloatProperty(name="Angular Velocity (\u00b0/s)", default=360.0, description="Degrees per second", update=_on_overlay_changed)  # pyright: ignore
    spin_flip: BoolProperty(name="Flip Direction", default=False, description="Reverse the spin rotation direction", update=_on_overlay_changed)  # pyright: ignore
    spin_center: FloatVectorProperty(
        name="Center", subtype="XYZ", default=(0, 0, 0),
        description=(
            "Spin center in the op's local frame (object-rotated, "
            "pre-translation). Matches world coords only when the "
            "object transform is identity."
        ),
        update=_on_overlay_changed,
    )  # pyright: ignore
    spin_center_mode: EnumProperty(
        name="Center",
        items=[
            ("CENTROID", "Centroid", "Compute center from vertex positions at runtime"),
            ("ABSOLUTE", "Fixed", "User-entered center in the op's local frame (object-rotated, before the world translation is applied — equals world coords only when the object transform is identity)"),
            ("MAX_TOWARDS", "Max Towards", "Centroid of vertices furthest in a direction"),
            ("VERTEX", "Vertex", "Use a single vertex picked in Edit Mode"),
        ],
        default="CENTROID",
        update=_on_overlay_changed,
    )  # pyright: ignore
    spin_center_vertex: IntProperty(name="Vertex Index", default=-1, description="Vertex index for center", update=_on_overlay_changed)  # pyright: ignore
    show_vertex_spin: BoolProperty(
        name="Show Vertex",
        default=False,
        description="Highlight the center vertex in the viewport",
        update=_on_overlay_changed,
    )  # pyright: ignore
    spin_center_direction: FloatVectorProperty(
        name="Direction", subtype="XYZ", default=(0, 0, -1),
        description="Direction to select vertices furthest towards",
        update=_on_overlay_changed,
    )  # pyright: ignore
    show_max_towards_spin: BoolProperty(
        name="Show Max Towards",
        default=False,
        description="Visualize selected vertices and direction",
        update=_on_overlay_changed,
    )  # pyright: ignore
    # Scale params
    scale_factor: FloatProperty(name="Factor", default=1.0, min=0.01, update=_on_overlay_changed)  # pyright: ignore
    scale_center: FloatVectorProperty(
        name="Center", subtype="XYZ", default=(0, 0, 0),
        description=(
            "Scale center in the op's local frame (object-rotated, "
            "pre-translation). Matches world coords only when the "
            "object transform is identity."
        ),
        update=_on_overlay_changed,
    )  # pyright: ignore
    scale_center_mode: EnumProperty(
        name="Center",
        items=[
            ("CENTROID", "Centroid", "Compute center from vertex positions at runtime"),
            ("ABSOLUTE", "Fixed", "User-entered center in the op's local frame (object-rotated, before the world translation is applied — equals world coords only when the object transform is identity)"),
            ("MAX_TOWARDS", "Max Towards", "Centroid of vertices furthest in a direction"),
            ("VERTEX", "Vertex", "Use a single vertex picked in Edit Mode"),
        ],
        default="CENTROID",
        update=_on_overlay_changed,
    )  # pyright: ignore
    scale_center_vertex: IntProperty(name="Vertex Index", default=-1, description="Vertex index for center", update=_on_overlay_changed)  # pyright: ignore
    show_vertex_scale: BoolProperty(
        name="Show Vertex",
        default=False,
        description="Highlight the center vertex in the viewport",
        update=_on_overlay_changed,
    )  # pyright: ignore
    scale_center_direction: FloatVectorProperty(
        name="Direction", subtype="XYZ", default=(0, 0, -1),
        description="Direction to select vertices furthest towards",
        update=_on_overlay_changed,
    )  # pyright: ignore
    show_max_towards_scale: BoolProperty(
        name="Show Max Towards",
        default=False,
        description="Visualize selected vertices and direction",
        update=_on_overlay_changed,
    )  # pyright: ignore
    # Torque params
    torque_axis_component: EnumProperty(
        name="Axis",
        items=[
            ("PC1", "1st Component", "Major principal axis (greatest spread)"),
            ("PC2", "2nd Component", "Middle principal axis"),
            ("PC3", "3rd Component", "Minor principal axis (least spread, often surface normal)"),
        ],
        default="PC3",
        update=_on_overlay_changed,
    )  # pyright: ignore
    torque_magnitude: FloatProperty(name="Magnitude (N\u00b7m)", default=1.0, description="Torque in Newton-metres", update=_on_overlay_changed)  # pyright: ignore
    torque_flip: BoolProperty(name="Flip Direction", default=False, description="Reverse the torque rotation direction", update=_on_overlay_changed)  # pyright: ignore
    # Common time range (in frames)
    frame_start: IntProperty(name="Start", default=1, min=1)  # pyright: ignore
    frame_end: IntProperty(name="End", default=60, min=1)  # pyright: ignore
    transition: EnumProperty(
        name="Transition",
        items=[
            ("LINEAR", "Linear", "Linear interpolation"),
            ("SMOOTH", "Smooth", "Smoothstep interpolation"),
        ],
        default="LINEAR",
    )  # pyright: ignore


class PinVertexGroupItem(PropertyGroup):

    def _invalidate_pin_overlay(self, context):
        from ..models.groups import invalidate_overlays
        invalidate_overlays()

    name: StringProperty()  # pyright: ignore  # [ObjectName][VertexGroupName]
    object_uuid: StringProperty()  # pyright: ignore
    vg_hash: StringProperty()  # pyright: ignore  # 64-bit content hash of VG vertex indices
    included: BoolProperty(
        name="Include",
        default=True,
        description="Include this pin in the simulation",
        update=_invalidate_pin_overlay,
    )  # pyright: ignore
    show_overlay: BoolProperty(
        name="Show",
        default=True,
        description="Show this pin's vertices as overlay circles in the viewport",
        update=_invalidate_pin_overlay,
    )  # pyright: ignore
    use_pin_duration: BoolProperty(
        name="Duration",
        default=False,
        description="Limit how long this pin is active",
        update=_invalidate_pin_overlay,
    )  # pyright: ignore
    pin_duration: IntProperty(
        name="Active For",
        default=60,
        min=1,
        description="Number of frames this pin is active for",
        update=_invalidate_pin_overlay,
    )  # pyright: ignore
    use_pull: BoolProperty(
        name="Pull",
        default=False,
        description="Use pull force instead of hard constraint",
    )  # pyright: ignore
    pull_strength: FloatProperty(
        name="Strength",
        default=1.0,
        min=0.0,
        soft_max=1000.0,
        precision=2,
        description="Pull force strength (0=no pull, 1=default)",
    )  # pyright: ignore
    pin_stiffness: FloatProperty(
        name="Pin Stiffness",
        default=1.0,
        min=0.0,
        soft_max=10000.0,
        precision=2,
        description=(
            "Stiffness scale for this pin's moving (animated) hard "
            "constraint force. 1.0 is the default; raise it if an animated "
            "pin lags or wobbles away from its target, lower it for a softer "
            "hold. Has no effect on a stationary pin or a Pull pin"
        ),
    )  # pyright: ignore
    fix_weight_threshold: FloatProperty(
        name="Fix Weight Threshold",
        default=0.5,
        min=0.0,
        max=1.0,
        soft_min=0.0,
        soft_max=1.0,
        precision=3,
        description=(
            "SOLID hard pin only: tet vertices whose diffused pin weight "
            "reaches this threshold are held as hard kinematic fixes; "
            "lower-weight vertices stay soft-pulled. Lower it toward 0 to "
            "hold more of the pinned surface region rigidly, raise it to "
            "soften the skirt. Interior vertices are never hard-fixed (they "
            "would crash the solver). No effect on pull pins or non-SOLID "
            "groups"
        ),
    )  # pyright: ignore
    operations: CollectionProperty(type=PinOperation)  # pyright: ignore
    operations_index: IntProperty(default=-1)  # pyright: ignore
    # True after Capture Deformation has written a ``_pindeform.pc2``
    # cache for this pin. Source of truth for the UIList label
    # ("(Captured)" suffix), the Capture/Clear button enabled state,
    # and the encoder's PC2-wins branch. Reconciled against on-disk
    # cache presence on file load via a load_post handler.
    has_captured_anim: BoolProperty(default=False)  # pyright: ignore
    # Cached result of the O(N) full-pin coverage check
    # (``pin_covers_all_vertices``), refreshed on demand by the Refresh button
    # next to the rest-pose toggle rather than recomputed on every panel
    # redraw. ``full_pin_checked`` stays False until the user clicks Refresh;
    # once True, ``full_pin_cached`` holds whether the pin's vertex group
    # covered every vertex of the mesh. The cache can go stale if the group is
    # edited afterward (re-click Refresh to update it); the encoder
    # independently re-verifies coverage at encode time, so a stale UI cache
    # only gates the toggle's editability and never changes sim behavior.
    full_pin_checked: BoolProperty(default=False)  # pyright: ignore
    full_pin_cached: BoolProperty(default=False)  # pyright: ignore
    # User opt-in, drawn under "Capture Deformation" for SOLID pins. When
    # checked, a captured deformation also drives a time-varying rest pose (the
    # encoder sets ``rest_shape_track``) so the dynamic body settles into the
    # captured shape instead of straining against it. Requires a FULL pin
    # (every vertex in the group), so the whole captured mesh becomes the rest;
    # disabled for a partial pin. Off by default: the capture then only
    # pulls/fixes the pinned vertices and the rest pose stays the undeformed
    # shape.
    track_rest_pose_deformation: BoolProperty(
        name="Track Rest-Pose Deformation",
        default=False,
        description=(
            "Also drive a time-varying rest pose from the captured "
            "deformation, so the dynamic body settles into the captured shape "
            "instead of straining against it. Requires a SOLID pin that covers "
            "every vertex of the mesh (a full pin) with a captured deformation; "
            "disabled otherwise. When unchecked, the capture only pulls or "
            "fixes the pinned vertices and the rest pose is unchanged"
        ),
    )  # pyright: ignore
