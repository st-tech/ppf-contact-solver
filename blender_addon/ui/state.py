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
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    StringProperty,
)
from bpy.types import PropertyGroup  # pyright: ignore
from bpy.app.translations import pgettext_iface as iface_, pgettext_tip as tip_  # pyright: ignore

from ..models.defaults import DEFAULT_MCP_PORT, DEFAULT_RELOAD_PORT, DEFAULT_SERVER_PORT
from ..models.enum_props import EnumProperty, dynamic_enum_items
# `decode_vertex_group_identifier`, `assign_display_indices`,
# `find_available_group_slot` are imported here because ui.dynamics
# submodules import them via ``from ..state import ...``.
from ..models.groups import (  # noqa: F401
    N_MAX_GROUPS,
    assign_display_indices,
    decode_vertex_group_identifier,
    find_available_group_slot,
    iterate_active_object_groups,
)

# Re-exports for backward compatibility
from .state_types import (
    CheckpointFrameItem,
    SaveCheckpointFrameItem,
    FetchedFrameItem,
    MergePairItem,
    DynParamKeyframe,
    DynParamItem,
    InvisibleColliderKeyframe,
    InvisibleColliderItem,
    AssignedObject,
    MaterialMapItem,
    MaterialMapSample,
    PinOperation,
    PinVertexGroupItem,
    StaticOpItem,
    VelocityKeyframe,
    CollisionWindowEntry,
)
from .object_group import ObjectGroup


def _sync_scene_timeline_to_sim(self, context):
    from ..core.animation import sync_scene_timeline_to_sim

    scene = context.scene if context is not None else bpy.context.scene
    sync_scene_timeline_to_sim(scene, self)


@dynamic_enum_items
def _get_profile_items(self, context):
    """Dynamic callback for profile_selection EnumProperty."""
    from ..core.profile import get_profile_names

    path = self.profile_path
    if not path:
        return [("NONE", iface_("(No Profile)"), "")]
    names = get_profile_names(bpy.path.abspath(path))
    if not names:
        return [("NONE", iface_("(No Profile)"), "")]
    return [(n, n, tip_("Profile: {name}").format(name=n)) for n in names]


@dynamic_enum_items
def _get_solver_gpu_items(self, context):
    """Dynamic callback for the ``solver_gpu`` dropdown.

    Each item's numeric ID is its CUDA index plus one, with 0 reserved for
    Automatic, so the number an item carries is fixed by the device it names
    rather than by where it sits in the list. A selection the solver host
    cannot satisfy gets its own entry so it stays visible and named instead of
    resolving to a different GPU.

    The devices offered are those of the machine that will run the server,
    which is not always the one Blender runs on.
    """
    from ..core.gpu_devices import (
        AUTOMATIC,
        STALE_SELECTION_ID,
        cached_gpu_devices,
    )

    items = [(
        "AUTO",
        iface_("Automatic"),
        tip_("Set no CUDA_VISIBLE_DEVICES, so the solver host's own choice stands"),
        "NONE",
        0,
    )]
    present = set()
    present_uuids = set()
    for device in cached_gpu_devices():
        present.add(device.index)
        present_uuids.add(device.uuid)
        items.append((
            str(device.index),
            f"{device.index}: {device.name}",
            tip_("Run the solver on GPU {index} ({name})").format(
                index=device.index, name=device.name
            ),
            "NONE",
            device.index + 1,
        ))
    stored = self.solver_gpu_index
    stored_uuid = self.solver_gpu_uuid
    missing = (
        (stored != AUTOMATIC or bool(stored_uuid))
        and (
            (bool(stored_uuid) and stored_uuid not in present_uuids)
            or (not stored_uuid and stored not in present)
        )
    )
    if missing:
        missing_name = stored_uuid or str(stored)
        items.append((
            "MISSING",
            iface_("{index}: not detected").format(index=missing_name),
            tip_("The solver host reports no GPU {index}").format(
                index=missing_name
            ),
            "ERROR",
            STALE_SELECTION_ID,
        ))
    return items


def _get_solver_gpu(self):
    """Map the saved GPU identity onto the dropdown's numeric ID."""
    from ..core.gpu_devices import (
        AUTOMATIC,
        STALE_SELECTION_ID,
        find_device,
        find_device_by_uuid,
        has_probed,
    )

    device = find_device_by_uuid(self.solver_gpu_uuid)
    if self.solver_gpu_uuid and device is None:
        return STALE_SELECTION_ID
    index = device.index if device is not None else self.solver_gpu_index
    if index == AUTOMATIC:
        return 0
    if not self.solver_gpu_uuid and has_probed() and find_device(index) is None:
        return STALE_SELECTION_ID
    return index + 1


def _set_solver_gpu(self, value):
    """Store the picked device's display index and stable UUID."""
    from ..core.gpu_devices import AUTOMATIC, STALE_SELECTION_ID, find_device

    if value == STALE_SELECTION_ID:
        return
    index = AUTOMATIC if value == 0 else value - 1
    device = find_device(index)
    self.solver_gpu_index = index
    self.solver_gpu_uuid = "" if device is None else device.uuid


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
    proxy_jump: StringProperty(
        name="Proxy Jump",
        default="",
        description=(
            "Jump host to tunnel the connection through, written as ssh -J "
            "takes it: [user@]host[:port], comma separated for a chain. "
            "Leave empty to use the ProxyJump entry ~/.ssh/config gives for "
            "the host, if it has one"
        ),
    )  # pyright: ignore
    # The published image builds the solver at /root/ppf-contact-solver
    # (Dockerfile: PROJ_NAME=ppf-contact-solver, WORKDIR /root/${PROJ_NAME}),
    # so the field is already right for a reader who followed the public
    # docker run. It carries a default rather than a blank because the path
    # exists only inside the container: no public instruction prints it, and
    # from outside it can be read only by entering the running container.
    docker_path: StringProperty(
        name="Container Path",
        default="/root/ppf-contact-solver",
        description=(
            "Directory INSIDE the container that holds the solver, i.e. the "
            "one with target/release/ppf-cts-server under it. For the "
            "published image this is /root/ppf-contact-solver, which is the "
            "default. Not a path on the machine Blender runs on"
        ),
    )  # pyright: ignore
    # WHERE THE SOLVER RUNS. Three of these reach a server on THIS machine and
    # start it themselves, one per platform; the rest reach one somewhere else.
    #
    # EXPLICIT NUMERIC IDs, as every saved enum in this file must have. Blender
    # stores the NUMBER in the `.blend`, so 3-tuples auto-numbered by list order
    # mean deleting or reordering an item silently repoints every saved file.
    # The numbers here are the ones the 3-tuple list had, so every `.blend`
    # written before they were spelled out keeps the connection it was saved
    # with.
    #
    # SLOT 0 IS RETIRED AND STAYS RETIRED. It was `LOCAL`, a connection to a
    # server the artist had started by hand on this machine, and the three
    # native types replaced it: they reach the same server and also know how to
    # start it, which build directory it came out of, and which device it runs
    # on. Nothing may take the number back, because a `.blend` saved with Local
    # still carries it; `core.migrate_renames.migrate_retired_connection`
    # moves such a file onto this platform's native type and carries its path
    # across, which is why no item is offered for it here.
    server_type: EnumProperty(  # pyright: ignore
        name="Type",
        items=[
            ("CUSTOM", "SSH", "Use a custom ssh config", "NONE", 1),
            ("COMMAND", "SSH Command", "Use ssh command", "NONE", 2),
            ("DOCKER", "Docker", "Use docker", "NONE", 3),
            ("DOCKER_SSH", "Docker over SSH", "Use docker over ssh", "NONE", 4),
            (
                "DOCKER_SSH_COMMAND",
                "Docker over SSH Command",
                "Use docker over ssh command",
                "NONE",
                5,
            ),
            (
                "WIN_NATIVE",
                "Windows Native",
                "Use a Windows build on this machine, and start its server",
                "NONE",
                6,
            ),
            (
                "MAC_NATIVE",
                "macOS Native",
                "Use a macOS build on this machine, and start its server",
                "NONE",
                7,
            ),
            (
                "LINUX_NATIVE",
                "Linux Native",
                "Use a Linux build on this machine, and start its server",
                "NONE",
                8,
            ),
        ],
        default="CUSTOM",
    )
    command: StringProperty(name="SSH Command", default="ssh -p xxx root@zzz")  # pyright: ignore
    # The default names the container the project's own public instructions
    # create (README's `docker run --name ppf-contact-solver`), so the field is
    # already right for a reader who followed them. The previous default,
    # `ppf-dev`, is the name of this project's internal development container
    # and appears in no public instruction, so it matched nothing a community
    # user had and every first connection failed on it.
    container: StringProperty(
        name="Container",
        default="ppf-contact-solver",
        description=(
            "Name of the Docker container running the solver, as 'docker ps' "
            "lists it. The container must already exist; the add-on starts a "
            "stopped one but does not create one"
        ),
    )  # pyright: ignore
    ssh_remote_path: StringProperty(
        name="Remote Path", default=""
    )  # pyright: ignore
    win_native_path: StringProperty(
        name="Solver Path",
        subtype="DIR_PATH",
        default="",
        description="Root directory where ppf-cts-server.exe is located",
    )  # pyright: ignore
    mac_native_path: StringProperty(
        name="Solver Path",
        subtype="DIR_PATH",
        default="",
        description="Root directory where ppf-cts-server is located",
    )  # pyright: ignore
    linux_native_path: StringProperty(
        name="Solver Path",
        subtype="DIR_PATH",
        default="",
        description="Root directory where ppf-cts-server is located",
    )  # pyright: ignore
    # WHICH COMPUTE DEVICE THE SERVER RUNS ON.
    #
    # ONE PROPERTY FOR EVERY CONNECTION, not one each. The three natives differ
    # in which accelerator the host has, CUDA or ROCm against Metal, and the
    # remote connections differ in which machine is asked; none of them differs
    # in the QUESTION, so a second property would be two names for one choice
    # and would let a `.blend` carry answers that disagree.
    #
    # EXPLICIT NUMERIC IDs, as every saved enum in this file must have. Blender
    # stores the NUMBER in the `.blend`, so 3-tuples auto-numbered by list order
    # mean deleting or reordering an item silently repoints every saved file.
    # Retire a value by keeping its slot, never by removing it.
    #
    # GPU IS SLOT 0 AND THE DEFAULT, so every `.blend` saved before this
    # property existed reads as GPU, which is what those files were built
    # against.
    native_device: EnumProperty(
        name="Compute Device",
        description=(
            "Which build of the solver to run. GPU uses the solver host's "
            "accelerator (CUDA or ROCm on Windows and Linux, Metal on macOS); "
            "CPU runs the portable backend, which needs no GPU and is "
            "substantially slower"
        ),
        items=[
            (
                "GPU",
                "GPU",
                "Run the accelerated build (CUDA or ROCm on Windows and Linux, "
                "Metal on macOS)",
                "NONE",
                0,
            ),
            (
                "CPU",
                "CPU",
                "Run the portable CPU build. No GPU required, and "
                "substantially slower than the accelerated one",
                "NONE",
                1,
            ),
        ],
        default="GPU",
        # READ ONCE, WHEN THE SERVER IS SPAWNED. Blender makes every property
        # keyframable unless told otherwise, and `options` REPLACES that default
        # rather than adding to it, so leaving this off would put a working
        # keyframe button next to a value nothing samples per frame.
        options=set(),
    )  # pyright: ignore
    # WHICH ACCELERATOR, once Compute Device says GPU and the solver host holds
    # more than one GPU build. A Windows x64 distribution carries CUDA and ROCm
    # together, a Linux x86_64 one carries both as well, and a machine can have
    # an NVIDIA and an AMD card at once, so the artist needs a way to say which
    # one runs; a notebook says the same thing with ``frontend.set_backend``
    # (``frontend/_backends_.py``).
    #
    # AUTOMATIC IS SLOT 0 AND THE DEFAULT, so a `.blend` saved before this
    # property existed, and every artist who does not care, gets the rule rather
    # than a name: the one GPU build present, or where several are, the first
    # whose solver reports a usable device, CUDA before ROCm.
    #
    # EXPLICIT NUMERIC IDs, as every saved enum in this file must have, and a
    # retired value keeps its slot.
    native_gpu_backend: EnumProperty(
        name="GPU Backend",
        description=(
            "Which accelerator to run on when Compute Device is GPU and the "
            "solver host holds more than one GPU build"
        ),
        items=[
            (
                "AUTO",
                "Automatic",
                "Use the only GPU build present, or where there are several, "
                "the first whose solver reports a usable device (CUDA, then "
                "ROCm)",
                "NONE",
                0,
            ),
            (
                "CUDA",
                "CUDA",
                "Run the CUDA build, on an NVIDIA GPU. Refused by name when "
                "this folder has no CUDA build",
                "NONE",
                1,
            ),
            (
                "ROCM",
                "ROCm",
                "Run the ROCm build, on an AMD GPU. Refused by name when this "
                "folder has no ROCm build",
                "NONE",
                2,
            ),
        ],
        default="AUTO",
        # Read when the server is spawned, like Compute Device above, so it
        # carries no keyframe button.
        options=set(),
    )  # pyright: ignore
    # The index is retained for display and backward compatibility. UUID is
    # the stable saved identity used for launch; ``solver_gpu`` is only the
    # dropdown view and carries no saved value of its own.
    solver_gpu_index: IntProperty(
        name="GPU Index",
        default=-1,
        min=-1,
        description="CUDA device index for the solver server, or -1 to set no CUDA_VISIBLE_DEVICES",
    )  # pyright: ignore
    solver_gpu_uuid: StringProperty(
        name="GPU UUID",
        default="",
        description="Stable UUID of the selected solver GPU",
    )  # pyright: ignore
    solver_gpu: EnumProperty(
        name="GPU",
        items=_get_solver_gpu_items,
        get=_get_solver_gpu,
        set=_set_solver_gpu,
        description="Which CUDA device on the solver host the server runs the solver on",
    )  # pyright: ignore
    docker_port: IntProperty(
        name="Docker Port",
        default=DEFAULT_SERVER_PORT,
        min=1024,
        max=65535,
        description="Port for the remote server (must be exposed in Docker)",
    )  # pyright: ignore


@dynamic_enum_items
def get_snap_objects(self=None, context=None):
    """Get all objects supported by the snap tool.

    EnumProperty items are (identifier, display_name, tooltip).
    The *identifier* is the object's UUID so the selection survives
    renames within the session.  The display name is the human-visible
    object name.
    """
    from ..core.uuid_registry import get_object_uuid

    items = [("NONE", iface_("None"), tip_("No object selected"))]
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
            items.append((uid, obj.name, tip_("Mesh object: {name}").format(name=obj.name)))
        elif obj.type == "CURVE" and uid in rod_curve_uuids:
            items.append((uid, obj.name, tip_("Curve object: {name}").format(name=obj.name)))
    return items


@dynamic_enum_items
def _get_scene_profile_items(self, context):
    """Dynamic callback for scene_profile_selection EnumProperty."""
    from ..core.profile import get_profile_names

    path = self.scene_profile_path
    if not path:
        return [("NONE", iface_("(No Profile)"), "")]
    names = get_profile_names(bpy.path.abspath(path))
    if not names:
        return [("NONE", iface_("(No Profile)"), "")]
    return [(n, n, tip_("Scene profile: {name}").format(name=n)) for n in names]


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
    world_scaling: FloatProperty(
        name="World Scaling",
        default=1.0,
        min=0.001,
        max=1000.0,
        soft_min=0.01,
        soft_max=100.0,
        precision=4,
        description=(
            "Uniform scale applied to all geometry before simulating; results are "
            "scaled back so the scene stays at its authored size. Use it to "
            "simulate an over- or under-sized scene at a sensible physical scale "
            "(e.g. 0.1 simulates a 15 m mesh at 1.5 m). Only geometry and relative "
            "contact gaps scale; gravity and absolute gaps do not"
        ),
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
    precond: EnumProperty(
        name="Preconditioner",
        # Explicit numeric IDs (4th=icon, 5th=number) freeze each identifier to
        # its slot so reordering or removing an item later cannot silently
        # corrupt the choice stored in pre-existing .blend files.
        items=[
            ("BLOCK_JACOBI", "Block Jacobi",
             "3x3 per-vertex diagonal preconditioner (default): fast with the "
             "device-resident PCG loop and does not run out of memory on "
             "heavy-contact scenes", "NONE", 0),
            ("SCHWARZ", "Schwarz",
             "Single-level additive aggregate-Schwarz; fewer iterations on "
             "systems mixing stiff and soft elements, but heavier per iteration "
             "and can OOM on large contact counts", "NONE", 1),
        ],
        default="BLOCK_JACOBI",
        description="Preconditioner for the PCG linear solver",
    )  # pyright: ignore
    schwarz_levels: EnumProperty(
        name="Schwarz Levels",
        # Explicit numeric IDs freeze each identifier to its slot so reordering
        # or removing an item later cannot silently corrupt the choice stored
        # in pre-existing .blend files.
        items=[
            ("LEVEL_1", "Level 1",
             "Single-level additive aggregate-Schwarz smoother", "NONE", 0),
            ("LEVEL_2", "Level 2",
             "Two-level additive Schwarz with a coarse correction over the "
             "connectivity partition; reduces the worst-case PCG iteration "
             "count on stiff multibody contact", "NONE", 1),
        ],
        default="LEVEL_2",
        description=(
            "Number of additive levels for the Schwarz preconditioner "
            "(only used when the preconditioner is Schwarz)"
        ),
    )  # pyright: ignore
    inactive_momentum_frames: IntProperty(
        name="Inactive Momentum Frames",
        default=0,
        min=0,
        max=600,
        description="Number of frames with inactive momentum (0 to disable)",
    )  # pyright: ignore
    save_state_on_finish: BoolProperty(
        name="Save State on Finish",
        default=False,
        description=(
            "Save the simulation state on the final frame before the solver "
            "exits, so the result stays resumable even when auto-save is off"
        ),
    )  # pyright: ignore
    # "Save and Checkpoints" UI box: a collapsible group above Wind that
    # collects Save State on Finish, the Auto Save sub-box, and the
    # per-frame Save Checkpoints list.
    show_save_and_checkpoints: BoolProperty(
        name="Save and Checkpoints",
        default=False,
        description="Toggle visibility of the save and checkpoint settings",
    )  # pyright: ignore
    show_auto_save: BoolProperty(
        name="Auto Save",
        default=False,
        description="Toggle visibility of the auto-save interval settings",
    )  # pyright: ignore
    show_checkpoints: BoolProperty(
        name="Save Checkpoints",
        default=False,
        description="Toggle visibility of the per-frame save checkpoints list",
    )  # pyright: ignore
    # Per-frame save checkpoints (input side). Each item is a Blender
    # 1-based frame at which the solver writes a resumable state. The
    # encoder converts these to solver 0-based indices. Distinct from
    # ``checkpoint_frames`` below, which lists states the solver has
    # already saved (the Resume-From dialog reads that one).
    save_checkpoint_frames: CollectionProperty(type=SaveCheckpointFrameItem)  # pyright: ignore
    save_checkpoint_frames_index: IntProperty(default=-1)  # pyright: ignore

    def convert_save_checkpoint_frames_to_remote(self) -> list[int]:
        """Sorted, de-duplicated solver 0-based frames for the encoder.

        The UIList stores Blender frames; the solver counts frames from 0 at
        the resolved starting frame (Blender N -> solver N - start). Solver
        frame 0 is the rest pose written before the step loop and is never a
        checkpoint, so frames at or before the starting frame are dropped.
        """
        # Local import: ``core.encoder`` pulls in the encoders, which read
        # this module's PropertyGroups.
        from ..core.encoder import resolve_start_frame
        start = resolve_start_frame(self)
        remote = {int(item.frame) - start for item in self.save_checkpoint_frames}
        return sorted(f for f in remote if f > 0)
    frame_start: IntProperty(
        name="Starting Frame",
        default=1,
        min=0,
        update=_sync_scene_timeline_to_sim,
        description=(
            "Blender frame the simulation's first output frame lands on, so a "
            "solve can be placed after a hand-animated lead-in. Simulated time "
            "zero is this frame. Ignored while Take Starting Frame from Scene "
            "is on"
        ),
    )  # pyright: ignore
    use_scene_frame_start: BoolProperty(  # pyright: ignore
        name="Take Starting Frame from Scene",
        default=False,
        update=_sync_scene_timeline_to_sim,
        description=(
            "Start the simulation at the Blender scene's start frame instead of "
            "the Starting Frame field, so the solve tracks the scene timeline"
        ),
    )
    frame_count: IntProperty(
        name="Frame Count",
        default=180,
        min=10,
        update=_sync_scene_timeline_to_sim,
        description="Number of frames for simulation",
    )  # pyright: ignore
    frame_rate: IntProperty(
        name="FPS",
        default=60,
        # Low rates (1, 2, 6, ...) are legitimate: each frame simply covers
        # more simulated time, and the solver already runs fractional rates
        # below 24 under Time Scale (substep dt is clamped to a frame
        # fraction solver-side, so a long frame just takes more substeps).
        min=1,
        soft_max=240,
        description=(
            "Frame rate the simulation runs at: how much simulated time one "
            "frame covers. Ignored while Take FPS from Scene is on"
        ),
    )  # pyright: ignore
    use_scene_fps: BoolProperty(  # pyright: ignore
        name="Take FPS from Scene",
        default=False,
        description=(
            "Run the simulation at the Blender scene's frame rate instead of "
            "the FPS field, so simulated time matches the scene timeline"
        ),
    )
    time_scale: FloatProperty(  # pyright: ignore
        name="Time Scale",
        default=1.0,
        min=0.01,
        max=10.0,
        soft_min=0.1,
        soft_max=1.0,
        description=(
            "Playback speed of the Blender animation in simulated time. "
            "1.0 runs the animation in real time; 0.5 re-interprets it at "
            "half speed, so fast keyframed motion (a combat move driving a "
            "collider) covers the same path over twice the simulated time "
            "and the cloth has time to respond. Gravity and materials stay "
            "physical, so cloth settles more per frame at lower values. "
            "Solve cost grows as 1 / Time Scale"
        ),
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
        min=0.00001,
        max=0.1,
        precision=5,
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
    keep_states: IntProperty(  # pyright: ignore
        name="Keep Saved States",
        default=0,
        min=0,
        description=(
            "Number of auto-saved checkpoints to retain. 0 keeps all "
            "(required for resuming from older frames)."
        ),
    )
    vertex_air_damp: FloatProperty(  # pyright: ignore
        name="Vertex Air Damping",
        default=0.0,
        min=0.0,
        max=1.0,
        precision=6,
        step=1,
        description="Damping factor for air resistance",
    )
    fix_xz: FloatProperty(  # pyright: ignore
        name="Fix XZ Above Height",
        default=0.0,
        min=0.0,
        precision=3,
        description=(
            "Height threshold (m) above which lateral (XY in Blender, "
            "XZ in solver Y-up) motion is constrained. 0 disables. "
            "Useful for hanging cloth/rods from above without an explicit pin."
        ),
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
    debug_mode: BoolProperty(
        name="Debug Options",
        default=False,
        description="Enable or disable debug mode",
    )  # pyright: ignore
    server_script: StringProperty(
        name="Args",
        default="",
        description="Arguments to the ppf-cts-server binary",
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

    # Saved-checkpoint frames offered in the Resume-From dialog. Populated
    # on the operator's invoke from ``com.saved_state_frames()`` and drawn
    # through ``SOLVER_UL_CheckpointFrames``.
    checkpoint_frames: CollectionProperty(type=CheckpointFrameItem)  # pyright: ignore
    checkpoint_frames_index: IntProperty(default=-1)  # pyright: ignore

    def convert_checkpoint_frames_to_list(self) -> list[int]:
        """Convert the checkpoint frames collection to a list of integers."""
        return [item.frame for item in self.checkpoint_frames]

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

    # Global (not per-pair) toggle: when on, fetched frames snap every
    # stitched source vertex exactly onto its target so seams read as
    # joined; when off, the raw simulated soft-stitch gap is shown.
    post_snap_exactly: BoolProperty(
        name="Post Snap Exactly",
        default=True,
        description=(
            "On fetch, move every stitched vertex exactly onto its stitch "
            "target so seams appear joined. Applies to all stitch pairs. "
            "Turn off to keep the raw simulated gap between stitched parts"
        ),
    )  # pyright: ignore

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

    show_linear_system_solver: BoolProperty(
        name="Linear System Solver",
        default=False,
        description="Toggle visibility of linear system solver settings",
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
    statistics_object_uuid: StringProperty(
        name="Statistics Object",
        default="",
        description="UUID of the object selected in the Statistics panel",
        options={"HIDDEN"},
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
    # Held from the geometry encode until the upload is accepted, so a
    # Transfer refused after that encode does not record its topology as
    # transferred and silence the stale-topology warning.
    pending_mesh_hash_json: StringProperty(
        name="Pending Mesh Hash JSON",
        default="",
        options={"HIDDEN"},
    )  # pyright: ignore

    def set_mesh_hash(self, hash_data: dict):
        """Store mesh hash data as JSON string."""
        self.mesh_hash_json = json.dumps(hash_data)

    def set_pending_mesh_hash(self, hash_data: dict):
        """Hold a topology summary until the upload it describes is accepted.

        The encoder computes it, but a Transfer can still be refused after the
        geometry encode: a parameter refusal, a failed connection. Recording it
        as transferred at compute time would suppress the stale-topology
        warning for a scene that never reached the wire.
        """
        self.pending_mesh_hash_json = json.dumps(hash_data)

    def commit_pending_mesh_hash(self):
        """Promote the held topology summary, if there is one."""
        if self.pending_mesh_hash_json:
            self.mesh_hash_json = self.pending_mesh_hash_json
            self.pending_mesh_hash_json = ""

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
        return iface_(
            "Mesh topology changed since last transfer "
            "(groups differing: {groups}). Re-transfer to sync."
        ).format(groups=", ".join(diverged))


class SceneRoot(PropertyGroup):
    state: bpy.props.PointerProperty(type=State)  # pyright: ignore
    ssh_state: bpy.props.PointerProperty(type=SSHState)  # pyright: ignore


for _i in range(N_MAX_GROUPS):
    SceneRoot.__annotations__[f"object_group_{_i}"] = bpy.props.PointerProperty(type=ObjectGroup)


classes = [
    FetchedFrameItem,
    CheckpointFrameItem,
    SaveCheckpointFrameItem,
    VelocityKeyframe,
    CollisionWindowEntry,
    StaticOpItem,
    AssignedObject,
    PinOperation,
    PinVertexGroupItem,
    MaterialMapSample,
    MaterialMapItem,
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
