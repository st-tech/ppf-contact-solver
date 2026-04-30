# File: state.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import json
import os

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

from ..models.defaults import DEFAULT_MCP_PORT, DEFAULT_RELOAD_PORT, DEFAULT_SERVER_PORT
from ..models.groups import (
    N_MAX_GROUPS,
    OBJECT_GROUP_DEFAULTS,
    assign_display_indices,
    decode_vertex_group_identifier,
    find_available_group_slot,
    get_group_by_uuid,
    get_object_type,
    get_vertex_group_items,
    iterate_active_object_groups,
)

# Re-exports for backward compatibility
from .state_types import (
    FetchedFrameItem,
    SavedPinKeyframePoint,
    SavedPinFCurve,
    SavedPinGroup,
    MergePairItem,
    DynParamKeyframe,
    DynParamItem,
    InvisibleColliderKeyframe,
    InvisibleColliderItem,
    AssignedObject,
    PinOperation,
    PinVertexGroupItem,
    StaticOpItem,
    VelocityKeyframe,
    CollisionWindowEntry,
)
from .object_group import ObjectGroup


def _get_profile_items(self, context):
    """Dynamic callback for profile_selection EnumProperty."""
    from ..core.profile import get_profile_names

    path = self.profile_path
    if not path:
        return [("NONE", "(No Profile)", "")]

    abs_path = bpy.path.abspath(path)
    names = get_profile_names(abs_path)
    if not names:
        return [("NONE", "(No Profile)", "")]
    return [(n, n, f"Profile: {n}") for n in names]


def _on_profile_selected(self, context):
    """Update callback when user picks a profile from the dropdown."""
    from ..core.profile import apply_profile, load_profiles
    from ..core.utils import redraw_all_areas

    if self.profile_selection == "NONE":
        return

    abs_path = bpy.path.abspath(self.profile_path)
    profiles = load_profiles(abs_path)
    profile = profiles.get(self.profile_selection)
    if profile is None:
        return
    apply_profile(profile, self)
    redraw_all_areas(context)


class SSHState(PropertyGroup):
    profile_path: StringProperty(
        name="Profile",
        subtype="FILE_PATH",
        default="",
        description="Path to a TOML connection profile file",
    )  # pyright: ignore
    profile_selection: EnumProperty(
        name="Profile",
        items=_get_profile_items,
        update=_on_profile_selected,
        description="Select a connection profile",
    )  # pyright: ignore
    host: StringProperty(name="Host", default="")  # pyright: ignore
    port: IntProperty(name="Port", default=22)  # pyright: ignore
    username: StringProperty(name="User", default="")  # pyright: ignore
    default_key_path = os.path.expanduser("~/.ssh/id_ed25519")
    if not os.path.exists(default_key_path):
        default_key_path = os.path.expanduser("~/.ssh/id_rsa")
    key_path: StringProperty(
        name="SSH Key",
        subtype="FILE_PATH",
        default=default_key_path,
    )  # pyright: ignore
    docker_path: StringProperty(
        name="Container Path", default="/root/ppf-contact-solver"
    )  # pyright: ignore
    local_path: StringProperty(
        name="Path", default=os.path.expanduser("~/ppf-contact-solver")
    )  # pyright: ignore
    server_type: EnumProperty(  # pyright: ignore
        name="Type",
        items=[
            ("LOCAL", "Local", "Use local directory"),
            ("CUSTOM", "SSH", "Use a custom ssh config"),
            ("COMMAND", "SSH Command", "Use ssh command"),
            ("DOCKER", "Docker", "Use docker"),
            ("DOCKER_SSH", "Docker over SSH", "Use docker over ssh"),
            (
                "DOCKER_SSH_COMMAND",
                "Docker over SSH Command",
                "Use docker over ssh command",
            ),
            ("WIN_NATIVE", "Windows Native", "Use local Windows native build with CUDA"),
        ],
        default="CUSTOM",
    )
    command: StringProperty(name="SSH Command", default="ssh -p xxx root@zzz")  # pyright: ignore
    container: StringProperty(name="Container", default="ppf-dev")  # pyright: ignore
    ssh_remote_path: StringProperty(
        name="Remote Path", default="/root/ppf-contact-solver"
    )  # pyright: ignore
    win_native_path: StringProperty(
        name="Solver Path",
        subtype="DIR_PATH",
        default="",
        description="Root directory where server.py is located",
    )  # pyright: ignore
    docker_port: IntProperty(
        name="Docker Port",
        default=DEFAULT_SERVER_PORT,
        min=1024,
        max=65535,
        description="Port for the remote server (must be exposed in Docker)",
    )  # pyright: ignore


_snap_objects_cache = []

def get_snap_objects(self=None, context=None):
    """Get all objects supported by the snap tool.

    EnumProperty items are (identifier, display_name, tooltip).
    The *identifier* is the object's UUID so the selection survives
    renames within the session.  The display name is the human-visible
    object name.
    """
    from ..core.uuid_registry import get_object_uuid

    global _snap_objects_cache
    items = [("NONE", "None", "No object selected")]
    rod_curve_uuids = set()
    if context is not None:
        for group in iterate_active_object_groups(context.scene):
            if group.object_type != "ROD":
                continue
            for assigned in group.assigned_objects:
                if assigned.uuid:
                    rod_curve_uuids.add(assigned.uuid)
    for obj in bpy.data.objects:
        uid = get_object_uuid(obj)
        if not uid:
            continue
        if obj.type == "MESH":
            items.append((uid, obj.name, f"Mesh object: {obj.name}"))
        elif obj.type == "CURVE" and uid in rod_curve_uuids:
            items.append((uid, obj.name, f"Curve object: {obj.name}"))
    _snap_objects_cache = items
    return items


def _get_scene_profile_items(self, context):
    """Dynamic callback for scene_profile_selection EnumProperty."""
    from ..core.profile import get_profile_names

    path = self.scene_profile_path
    if not path:
        return [("NONE", "(No Profile)", "")]

    abs_path = bpy.path.abspath(path)
    names = get_profile_names(abs_path)
    if not names:
        return [("NONE", "(No Profile)", "")]
    return [(n, n, f"Scene profile: {n}") for n in names]


def _on_scene_profile_selected(self, context):
    """Update callback when user picks a scene param profile."""
    from ..core.profile import apply_scene_profile, load_profiles
    from ..core.utils import redraw_all_areas
    from ..models.groups import invalidate_overlays

    if self.scene_profile_selection == "NONE":
        return

    abs_path = bpy.path.abspath(self.scene_profile_path)
    profiles = load_profiles(abs_path)
    profile = profiles.get(self.scene_profile_selection)
    if profile is None:
        return
    apply_scene_profile(profile, self)
    invalidate_overlays()
    redraw_all_areas(context)


def _on_direction_preview_changed(self, context):
    from ..models.groups import invalidate_overlays

    invalidate_overlays()


def _on_hide_overlay_colors_changed(self, context):
    from .dynamics.overlay import apply_object_overlays

    apply_object_overlays()


class State(PropertyGroup):
    uuid_migration_result: StringProperty(
        name="UUID Migration Result",
        default="",
    )  # pyright: ignore
    scene_profile_path: StringProperty(
        name="Scene Profile",
        subtype="FILE_PATH",
        default="",
        description="Path to a TOML scene parameter profile file",
    )  # pyright: ignore
    scene_profile_selection: EnumProperty(
        name="Scene Profile",
        items=_get_scene_profile_items,
        update=_on_scene_profile_selected,
        description="Select a scene parameter profile",
    )  # pyright: ignore
    step_size: FloatProperty(
        name="Step Size",
        default=0.01,
        min=0.001,
        max=0.01,
        precision=3,
        unit="TIME",
        soft_min=0.001,
        soft_max=0.01,
        description="Simulation step size",
    )  # pyright: ignore
    min_newton_steps: IntProperty(
        name="Min Newton Steps",
        default=1,
        min=1,
        max=64,
        description="Minimum number of Newton steps",
    )  # pyright: ignore
    air_density: FloatProperty(
        name="Air Density (kg/m\u00b3)",
        default=0.001,
        min=0.0,
        max=0.01,
        precision=3,
        soft_min=0.001,
        soft_max=0.01,
        description="Air density for the simulation",
    )  # pyright: ignore
    gravity_3d: FloatVectorProperty(
        name="Gravity (m/s\u00b2)",
        subtype="XYZ",
        size=3,
        default=(0.0, 0.0, -9.8),
        precision=2,
        description="Gravity acceleration vector (m/s\u00b2)",
        update=_on_direction_preview_changed,
    )  # pyright: ignore
    preview_gravity_direction: BoolProperty(
        name="Preview Direction",
        default=False,
        description="Show gravity direction preview in viewport",
        update=_on_direction_preview_changed,
    )  # pyright: ignore
    show_wind: BoolProperty(
        name="Wind",
        default=False,
        description="Toggle visibility of wind parameters",
    )  # pyright: ignore
    wind_direction: FloatVectorProperty(
        name="Direction",
        default=(0.0, 0.0, 0.0),
        subtype="XYZ",
        description="Wind direction vector (XYZ)",
        update=_on_direction_preview_changed,
    )  # pyright: ignore
    wind_strength: FloatProperty(
        name="Strength (m/s)",
        default=0.0,
        min=0.0,
        max=1000.0,
        precision=2,
        description="Wind strength (m/s)",
        update=_on_direction_preview_changed,
    )  # pyright: ignore
    preview_wind_direction: BoolProperty(
        name="Preview Direction",
        default=False,
        description="Show wind direction preview in viewport",
        update=_on_direction_preview_changed,
    )  # pyright: ignore
    air_friction: FloatProperty(
        name="Air Friction",
        default=0.2,
        min=0.0,
        max=1.0,
        precision=2,
        description="Ratio of tangential friction to normal friction for air drag/lift",
    )  # pyright: ignore
    friction_mode: EnumProperty(
        name="Friction Mode",
        items=[
            ("MIN", "Minimum", "Use min(friction_A, friction_B); the more slippery side wins"),
            ("MAX", "Maximum", "Use max(friction_A, friction_B); the grippier side wins"),
            ("MEAN", "Mean", "Use 0.5 * (friction_A + friction_B)"),
        ],
        default="MIN",
        description="How to combine friction coefficients of two contacting elements",
    )  # pyright: ignore
    inactive_momentum_frames: IntProperty(
        name="Inactive Momentum Frames",
        default=0,
        min=0,
        max=600,
        description="Number of frames with inactive momentum (0 to disable)",
    )  # pyright: ignore
    frame_count: IntProperty(
        name="Frame Count",
        default=180,
        min=10,
        description="Number of frames for simulation",
    )  # pyright: ignore
    frame_rate: IntProperty(
        name="FPS",
        default=60,
        min=24,
        description="Frame rate for simulation",
    )  # pyright: ignore
    use_frame_rate_in_output: BoolProperty(  # pyright: ignore
        name="Use Frame Rate in Output",
        default=False,
        description="Use frame rate in output. If unchecked, FPS field is shown.",
    )
    show_advanced_parameters: BoolProperty(
        name="Advanced Params",
        default=False,
        description="Toggle visibility of advanced parameters",
    )  # pyright: ignore
    contact_nnz: IntProperty(
        name="Max Contact",
        default=100000000,
        min=10000000,
        description="Number of non-zero entries in the contact matrix",
    )  # pyright: ignore
    line_search_max_t: FloatProperty(
        name="Line Search Max T",
        default=1.25,
        min=0.1,
        max=10.0,
        precision=2,
        description="Factor to extend TOI for CCD to avoid possible solver divergence",
    )  # pyright: ignore
    constraint_ghat: FloatProperty(
        name="Constraint Gap",
        default=0.001,
        min=0.0001,
        max=0.1,
        precision=4,
        description="Gap distance to activate boundary condition barriers",
    )  # pyright: ignore
    cg_max_iter: IntProperty(
        name="PCG Max Iterations",
        default=10000,
        min=100,
        max=100000,
        description="Maximum number of PCG iterations before divergence",
    )  # pyright: ignore
    cg_tol: FloatProperty(
        name="PCG Tolerance",
        default=0.001,
        min=0.0001,
        max=0.1,
        precision=4,
        description="Relative tolerance for PCG solver termination",
    )  # pyright: ignore
    include_face_mass: BoolProperty(
        name="Include Face Mass",
        default=False,
        description="Include shell mass for surface elements of volume solids",
    )  # pyright: ignore
    disable_contact: BoolProperty(
        name="Disable Contact",
        default=False,
        description="Disable all contact detection in the simulation",
    )  # pyright: ignore
    auto_save: BoolProperty(  # pyright: ignore
        name="Auto Save",
        default=False,
        description="Enable auto-saving of the simulation state",
    )  # pyright: ignore
    auto_save_interval: IntProperty(  # pyright: ignore
        name="Auto Save Interval",
        default=10,
        min=1,
        description="Interval for auto-saving the simulation state",
    )
    vertex_air_damp: FloatProperty(  # pyright: ignore
        name="Vertex Air Damping",
        default=0.0,
        min=0.0,
        max=1.0,
        precision=3,
        description="Damping factor for air resistance",
    )
    show_statistics: BoolProperty(
        name="Statistics",
        default=True,
        description="Toggle visibility of simulation statistics",
    )  # pyright: ignore
    show_scene_info: BoolProperty(
        name="Scene Info",
        default=True,
        description="Toggle visibility of scene information",
    )  # pyright: ignore
    show_hardware: BoolProperty(
        name="Remote Hardware",
        default=False,
        description="Toggle visibility of remote hardware info",
    )  # pyright: ignore
    show_usage: BoolProperty(
        name="Resource Usage",
        default=True,
        description="Toggle visibility of machine resources usage",
    )  # pyright: ignore
    debug_mode: BoolProperty(
        name="Debug Options",
        default=False,
        description="Enable or disable debug mode",
    )  # pyright: ignore
    server_script: StringProperty(
        name="Args",
        default="",
        description="Arguments to the server.py script",
    )  # pyright: ignore
    shell_command: StringProperty(
        name="Command",
        default="",
        description="Shell command to execute",
    )  # pyright: ignore
    mcp_port: IntProperty(
        name="MCP Port",
        default=DEFAULT_MCP_PORT,
        min=1024,
        max=65535,
        description="Port number for MCP server communication",
    )  # pyright: ignore
    reload_port: IntProperty(
        name="Reload Port",
        default=DEFAULT_RELOAD_PORT,
        min=1024,
        max=65535,
        description="UDP port for addon reload server",
    )  # pyright: ignore
    jupyter_port: IntProperty(
        name="JupyterLab Port",
        default=8080,
        min=1024,
        max=65535,
        description="Port number for JupyterLab server",
    )  # pyright: ignore
    show_connection: BoolProperty(
        name="Connection",
        default=True,
        description="Toggle visibility of connection settings",
    )  # pyright: ignore
    show_mcp: BoolProperty(
        name="MCP Settings",
        default=False,
        description="Toggle visibility of MCP settings panel",
    )  # pyright: ignore
    show_jupyter: BoolProperty(
        name="JupyterLab",
        default=False,
        description="Toggle visibility of JupyterLab export panel",
    )  # pyright: ignore
    jupyter_last_export: StringProperty(
        name="Last Export Path",
        default="",
        description="Path of the last exported JupyterLab notebook",
    )  # pyright: ignore
    max_console_lines: IntProperty(
        name="Max Console Lines",
        default=60,
        min=8,
        max=10000,
        description="Maximum number of lines to keep in the console",
    )  # pyright: ignore
    use_shell: BoolProperty(
        name="Run as Shell",
        default=True,
        description="Execute commands using a shell",
    )  # pyright: ignore
    data_size: IntProperty(
        name="Data Size (MB)",
        default=1,
        min=1,
        max=256,
        description="Size of data to transfer in MB",
    )  # pyright: ignore
    log_file_path: StringProperty(
        name="Log Path",
        default="",
        description=(
            "Path to export console log to. Leave empty to disable file "
            "logging; pick a destination via the file-browser button to "
            "enable it."
        ),
    )  # pyright: ignore
    project_name: StringProperty(
        name="Project Name",
        default="unnamed",
        description="Name of the current project",
    )  # pyright: ignore
    fetched_frame: CollectionProperty(
        type=FetchedFrameItem,
        name="Fetched Frame",
        description="A list of fetched frames",
    )  # pyright: ignore
    saved_pin_keyframes: CollectionProperty(
        type=SavedPinGroup,
        name="Saved Pin Keyframes",
        description="Pin keyframes saved before simulation for restore on clear",
    )  # pyright: ignore

    def convert_fetched_frames_to_list(self) -> list[int]:
        """Convert fetched frames to a list of integers."""
        return [item.value for item in self.fetched_frame]

    def fetch_frames_from_list(self, frames: list[int]):
        """Fetch frames from a list and add them to the fetched_frame collection."""
        for frame in frames:
            if not self.has_fetched_frame(frame):
                item = self.fetched_frame.add()
                item.value = frame

    def has_fetched_frame(self, frame: int) -> bool:
        """Check if a frame has been fetched."""
        return any(item.value == frame for item in self.fetched_frame)

    def clear_fetched_frames(self):
        """Clear the list of fetched frames."""
        self.fetched_frame.clear()

    def add_fetched_frame(self, frame: int):
        """Add a fetched frame to the list."""
        if not self.has_fetched_frame(frame):
            item = self.fetched_frame.add()
            item.value = frame

    # Group management
    current_group_uuid: StringProperty(
        name="Current Group UUID",
        default="",
        description="UUID of the currently selected group",
    )  # pyright: ignore
    # Snap to vertices properties
    snap_object_a: EnumProperty(
        name="Object A",
        items=get_snap_objects,
        description="Object to move (will snap to Object B)",
        options={"SKIP_SAVE"},
    )  # pyright: ignore
    snap_object_b: EnumProperty(
        name="Object B",
        items=get_snap_objects,
        description="Target object (stays in place)",
        options={"SKIP_SAVE"},
    )  # pyright: ignore

    # Merge pairs (auto-populated by Snap A to B)
    merge_pairs: CollectionProperty(type=MergePairItem)  # pyright: ignore
    merge_pairs_index: IntProperty(default=-1)  # pyright: ignore

    # Dynamic scene parameters
    dyn_params: CollectionProperty(type=DynParamItem)  # pyright: ignore
    dyn_params_index: IntProperty(default=-1)  # pyright: ignore
    show_dyn_params: BoolProperty(
        name="Dynamic Parameters",
        default=False,
        description="Toggle visibility of dynamic parameters",
    )  # pyright: ignore

    # Invisible colliders
    invisible_colliders: CollectionProperty(type=InvisibleColliderItem)  # pyright: ignore
    invisible_colliders_index: IntProperty(default=-1)  # pyright: ignore
    show_invisible_colliders: BoolProperty(
        name="Invisible Colliders",
        default=False,
        description="Toggle visibility of invisible colliders section",
    )  # pyright: ignore

    # Visualization master toggles
    hide_pins: BoolProperty(
        name="Hide all pins",
        default=False,
        description="Hide pin vertex overlays across all groups",
        update=_on_direction_preview_changed,
    )  # pyright: ignore
    hide_arrows: BoolProperty(
        name="Hide all directional arrows",
        default=False,
        description="Hide gravity, wind, and per-object velocity direction arrows",
        update=_on_direction_preview_changed,
    )  # pyright: ignore
    hide_overlay_colors: BoolProperty(
        name="Hide all overlaid colors",
        default=False,
        description="Suppress per-group object color tinting in the viewport",
        update=_on_hide_overlay_colors_changed,
    )  # pyright: ignore
    hide_snaps: BoolProperty(
        name="Hide all snaps",
        default=False,
        description="Hide snap correspondence lines and merge-pair markers",
        update=_on_direction_preview_changed,
    )  # pyright: ignore
    hide_pin_operations: BoolProperty(
        name="Hide all pin operations",
        default=False,
        description="Hide pin operation overlays (spin circles, move/scale trajectories, torque arcs)",
        update=_on_direction_preview_changed,
    )  # pyright: ignore


    overlay_version: IntProperty(default=0, options={"HIDDEN"})  # pyright: ignore

    # Last session id used by this .blend.  Written by the facade whenever
    # the engine's session_id changes; compared on reconnect to decide
    # whether a running remote sim belongs to this project or should be
    # flagged as orphan / adoptable.
    last_session_id: StringProperty(
        name="Last Session",
        default="",
        description="Session id stamped on artifacts produced by this project",
        options={"HIDDEN"},
    )  # pyright: ignore

    # Stored topology summary for validation across transfer/run/fetch.
    mesh_hash_json: StringProperty(
        name="Mesh Hash JSON",
        default="{}",
        description="JSON string storing mesh topology hash for validation",
    )  # pyright: ignore

    def set_mesh_hash(self, hash_data: dict):
        """Store mesh hash data as JSON string."""
        self.mesh_hash_json = json.dumps(hash_data)

    def get_mesh_hash(self) -> dict:
        """Retrieve mesh hash data from JSON string."""
        try:
            return json.loads(self.mesh_hash_json)
        except (json.JSONDecodeError, ValueError):
            return {}

    def validate_mesh_hash(self, context) -> str:
        """Compare the stored hash from the last transfer against a fresh
        snapshot of the current scene. Returns a user-facing warning
        message if the topology has diverged (vertex count / triangle
        count / pin membership changed) since transfer, or an empty
        string if the hash was never stored or matches."""
        stored = self.get_mesh_hash()
        if not stored:
            return ""
        try:
            from ..core.encoder.mesh import compute_mesh_hash
            current = compute_mesh_hash(context)
        except Exception:
            return ""
        if current == stored:
            return ""
        diverged = []
        for key in set(current) | set(stored):
            if current.get(key) != stored.get(key):
                diverged.append(key)
        return (
            "Mesh topology changed since last transfer "
            f"(groups differing: {', '.join(diverged)}). Re-transfer to sync."
        )


class SceneRoot(PropertyGroup):
    state: bpy.props.PointerProperty(type=State)  # pyright: ignore
    ssh_state: bpy.props.PointerProperty(type=SSHState)  # pyright: ignore


for _i in range(N_MAX_GROUPS):
    SceneRoot.__annotations__[f"object_group_{_i}"] = bpy.props.PointerProperty(type=ObjectGroup)


classes = [
    FetchedFrameItem,
    SavedPinKeyframePoint,
    SavedPinFCurve,
    SavedPinGroup,
    VelocityKeyframe,
    CollisionWindowEntry,
    StaticOpItem,
    AssignedObject,
    PinOperation,
    PinVertexGroupItem,
    ObjectGroup,
    MergePairItem,
    DynParamKeyframe,
    DynParamItem,
    InvisibleColliderKeyframe,
    InvisibleColliderItem,
    State,
    SSHState,
    SceneRoot,
]


def register():
    from ..models.groups import _ADDON_NAMESPACE

    for cls in classes:
        bpy.utils.register_class(cls)
    setattr(bpy.types.Scene, _ADDON_NAMESPACE, bpy.props.PointerProperty(type=SceneRoot))


def unregister():
    from ..models.groups import _ADDON_NAMESPACE

    delattr(bpy.types.Scene, _ADDON_NAMESPACE)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
