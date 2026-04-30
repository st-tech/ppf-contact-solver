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


class SavedPinKeyframePoint(PropertyGroup):
    frame: IntProperty()  # pyright: ignore
    value: FloatProperty()  # pyright: ignore


class SavedPinFCurve(PropertyGroup):
    data_path: StringProperty()  # pyright: ignore
    array_index: IntProperty()  # pyright: ignore
    points: CollectionProperty(type=SavedPinKeyframePoint)  # pyright: ignore


class SavedPinGroup(PropertyGroup):
    object_name: StringProperty()  # pyright: ignore
    object_uuid: StringProperty()  # pyright: ignore
    vertex_group: StringProperty()  # pyright: ignore
    vg_hash: StringProperty()  # pyright: ignore  # 64-bit content hash of VG vertex indices
    fcurves: CollectionProperty(type=SavedPinFCurve)  # pyright: ignore


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
    static_ops: CollectionProperty(type=StaticOpItem)  # pyright: ignore
    static_ops_index: IntProperty(default=-1, options={"HIDDEN"})  # pyright: ignore


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
    op_type: EnumProperty(
        name="Type",
        items=[
            ("EMBEDDED_MOVE", "Embedded Move", "Keyframe-based vertex movement (auto-detected)"),
            ("MOVE_BY", "Move By", "Move by delta over time range"),
            ("SPIN", "Spin", "Rotate around axis"),
            ("SCALE", "Scale", "Scale from center"),
            ("TORQUE", "Torque", "Apply rotational force around PCA axis"),
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
    operations: CollectionProperty(type=PinOperation)  # pyright: ignore
    operations_index: IntProperty(default=-1)  # pyright: ignore
