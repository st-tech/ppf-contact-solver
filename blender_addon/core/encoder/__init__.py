# File: encoder/__init__.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import bpy  # pyright: ignore

from ..transform import _normalize_and_scale, _swap_axes, _to_solver


def resolve_fps(state) -> float:
    """The frame rate the simulation actually runs at, in one place.

    This is the ONLY fps the solver ever sees: every encoder converts frames to
    seconds through it, and it is what lands in the solver's ``param.toml``.
    ``use_scene_fps`` takes it from the Blender scene, so simulated time tracks
    the scene timeline; otherwise it is the explicit ``frame_rate`` field, which
    lets the simulation run on a different time base than playback.

    Anything reporting an fps to the user or to an API caller must report THIS,
    never ``state.frame_rate`` raw: the field keeps its last value while the
    scene override is on, so quoting it would name a rate the solver is not
    using.
    """
    return (
        bpy.context.scene.render.fps
        if state.use_scene_fps
        else int(state.frame_rate)
    )


def resolve_solver_fps(state) -> float:
    """The frame rate the SOLVER's time mapping runs at, in one place.

    ``resolve_fps(state) * state.time_scale``. Time Scale re-interprets how
    much simulated time one Blender frame covers: at 0.5 each frame lasts
    twice as long in solver seconds, so keyframed motion plays at half speed
    while gravity and materials stay physical. Every encoder that converts a
    Blender frame to solver seconds (pin schedules, captured-deformation
    times, transform keyframes, the ``fps`` param itself) must use THIS, so
    the whole schedule scales coherently and output frames stay one per
    Blender frame.

    Reporting surfaces (the panel, MCP ``effective_fps``) keep using
    ``resolve_fps``: Time Scale is a separate, explicitly-shown knob, not a
    change to the scene's frame rate.
    """
    return resolve_fps(state) * resolve_time_scale(state)


def resolve_time_scale(state) -> float:
    """The Time Scale factor, validated, in one place.

    Besides scaling the solver fps (``resolve_solver_fps``), every
    rate-parameterized quantity authored in ANIMATION seconds must be
    multiplied by this at encode so its per-frame effect is invariant under
    Time Scale and stays in sync with keyframed motion: a SPIN op's degrees
    per second, and the speed of EVERY velocity keyframe, translational and
    angular, the one in effect at the starting frame (the object's initial
    velocity) included. Nothing downstream of the encoder applies Time Scale
    to any of them. Physical settings (gravity, wind, materials) are NOT
    scaled: they stay physical in solver seconds.
    """
    scale = float(getattr(state, "time_scale", 1.0))
    if scale <= 0.0:
        raise ValueError(f"Time Scale must be positive, got {scale}")
    return scale


def resolve_world_scaling(state) -> float:
    """The World Scaling factor, validated, and the rule for what it scales.

    World Scaling changes the PHYSICAL size the scene is simulated at: 0.1
    simulates a 15 m mesh as a 1.5 m one, under the same gravity, and writes
    the result back at 15 m. The solver multiplies the scene it ingests by
    this factor and divides every output position back by it. A length
    authored in SCENE units is therefore multiplied by World Scaling exactly
    once, by whichever side reads it raw:

    * The solver scales what it ingests as geometry: vertex, rest-shape and
      collision-mesh positions, invisible-collider positions, radii and
      thicknesses, pin-operation deltas and centers, the initial velocity in
      vel.bin, and the fix-xz threshold.
    * The encoder scales every scene-unit length the solver reads as a plain
      parameter: contact gaps and offsets (a group's, an invisible
      collider's, and their keyframed samples), the velocity schedule, and
      constraint-ghat, static and animated alike.

    Physical constants are NOT scaled, because they describe the world the
    resized scene is simulated in, not the scene: gravity, wind (an air
    velocity), material parameters (air density included) and a pin's
    torque. Quantities with no length in them (angular velocity, friction,
    time) are not scaled either.
    """
    scale = float(state.world_scaling)
    if not scale > 0.0:
        raise ValueError(f"World Scaling must be positive, got {scale}")
    return scale


def solver_gravity(gravity_3d) -> list[float]:
    """Gravity as the solver reads it: solver axes, physical units.

    The one transform the static setting, a sampled keyframe and a legacy
    keyframe-list entry all take, so the three cannot disagree. Gravity is a
    physical constant, so World Scaling does not touch it
    (``resolve_world_scaling``).
    """
    return [float(g) for g in _swap_axes(gravity_3d)]


def solver_wind(direction, strength, what: str) -> list[float]:
    """Wind as the solver reads it: an air velocity in solver axes.

    ``_normalize_and_scale`` validates the strength, naming ``what`` in a
    refusal. Wind is a physical air velocity, so World Scaling does not
    touch it (``resolve_world_scaling``).
    """
    return _swap_axes(_normalize_and_scale(direction, strength, what))


def resolve_start_frame(state) -> int:
    """The Blender frame the simulation begins on, in one place.

    Simulated time zero IS this frame: every encoder converts a Blender frame
    to seconds as ``(frame - resolve_start_frame(state)) /
    resolve_solver_fps(state)`` (the Time-Scaled solver rate), and playback
    maps solver frame ``n`` back to Blender frame ``n + start``.
    That lets a solve sit after a hand-animated lead-in instead of always
    occupying frames 1..N.

    ``use_scene_frame_start`` takes it from the Blender scene's start frame, so
    the solve tracks the scene timeline; otherwise it is the explicit
    ``frame_start`` field, which lets the solve sit anywhere independent of the
    timeline the artist is scrubbing.

    Anything reporting a starting frame to the user or to an API caller must
    report THIS, never ``state.frame_start`` raw: like ``frame_rate`` under
    ``use_scene_fps``, the field keeps its last value while the scene override
    is on, so quoting it would name a frame the solver is not starting on.
    """
    return (
        int(bpy.context.scene.frame_start)
        if state.use_scene_frame_start
        else int(state.frame_start)
    )


def resolve_start_frame_or_default(scene, default: int = 1) -> int:
    """``resolve_start_frame`` for callers that may run before the addon's
    PropertyGroup is registered on the scene.

    ``get_addon_data`` dereferences the namespace with a bare ``getattr``, so it
    raises during addon register and during a fresh ``load_post`` before the
    handlers have wired up. Handlers, timers and panel helpers that can fire in
    those windows use this and get *default* (frame 1, the property default)
    instead. Anything running from an operator or the encoder has a live state
    and must call ``resolve_start_frame`` directly, so a genuinely missing state
    still surfaces there rather than being masked everywhere.
    """
    from ...models.groups import get_addon_data, has_addon_data
    if scene is None or not has_addon_data(scene):
        return default
    return resolve_start_frame(get_addon_data(scene).state)


def frame_to_time(frame, fps: float, start_frame: int) -> float:
    """Seconds of simulated time at Blender frame *frame*.

    The one conversion every encoder goes through, so the frame-to-seconds
    convention is stated once. The starting frame is simulated time zero, so a
    frame before it yields a negative time, which no solver schedule accepts.
    A caller that can be handed such a frame decides what it means: a window
    or a later keyframe placed there is refused by name
    (``check_frame_window``), while a value that is simply in effect when the
    solve begins, such as a velocity keyframe authored during a lead-in,
    lands on time zero.
    """
    return (float(frame) - start_frame) / fps


def op_type_label(op) -> str:
    """The name the panel shows for an operation's type, such as "Move By".

    Read off the op's own enum so a pin operation and a STATIC operation are
    both named the way their dropdown names them.
    """
    return op.bl_rna.properties["op_type"].enum_items[op.op_type].name


def seed_window_at_start(item, state) -> None:
    """Move a NEW item's default window to begin at the starting frame.

    Operations and collision windows default to frames 1 to 60, and Transfer
    refuses a window that starts before the starting frame. An item created
    from those defaults in a scene whose simulation starts later is therefore
    shifted, keeping its length, so adding one does not author a refusal. A
    window the artist or a caller sets explicitly is theirs and is not moved.
    """
    shift = resolve_start_frame(state) - int(item.frame_start)
    if shift > 0:
        item.frame_start = int(item.frame_start) + shift
        item.frame_end = int(item.frame_end) + shift


def check_frame_window(owner: str, frame_start, frame_end, start_frame: int):
    """Refuse a frame window the solve cannot honor, naming whose it is.

    A window is a Start and an End frame on the timeline: a pin operation, a
    STATIC operation or a collision window. Two shapes are refused by name
    rather than reshaped into something the artist did not author:

    * An End frame that is not after the Start frame. The window covers no
      time, and what reaches the solver depends on the operation: a solver
      error that names nothing, no motion at all, or a jump in one instant. A
      lone collision window of that shape switches contact off for the whole
      run.
    * A Start frame before the starting frame (``resolve_start_frame``).
      Simulated time begins at the starting frame, so the part of the window
      before it never runs, and cutting it off would change how fast the
      operation moves or how long contact lasts without saying so.

    A window that starts exactly on the starting frame begins at simulated
    time zero and is accepted.

    ``owner`` names the window in the artist's terms, for example
    ``"Pin 'Top' on 'Sheet': its Move By operation"``, and every refusal
    quotes both of its frames.
    """
    frame_start = int(frame_start)
    frame_end = int(frame_end)
    start_frame = int(start_frame)
    if frame_end <= frame_start:
        raise ValueError(
            f"{owner} runs from frame {frame_start} to frame {frame_end}, but "
            "it has to end after it starts: set its End frame after frame "
            f"{frame_start}."
        )
    if frame_start < start_frame:
        raise ValueError(
            f"{owner} runs from frame {frame_start} to frame {frame_end}, "
            f"which starts before the starting frame {start_frame}, where the "
            "simulation begins, so its first part would never run. Move its "
            f"Start frame to {start_frame} or later, or start the simulation "
            f"at frame {frame_start} or earlier."
        )


from .mesh import (  # noqa: E402
    compute_data_hash,
    compute_mesh_hash,
    detect_stitch_edges,
    encode_obj,
    encode_obj_with_hash,
)
from .params import (  # noqa: E402
    compute_param_hash,
    encode_param,
    encode_param_with_hash,
)

__all__ = [
    "_swap_axes",
    "_to_solver",
    "_normalize_and_scale",
    "resolve_fps",
    "resolve_solver_fps",
    "resolve_time_scale",
    "resolve_world_scaling",
    "solver_gravity",
    "solver_wind",
    "resolve_start_frame",
    "resolve_start_frame_or_default",
    "frame_to_time",
    "op_type_label",
    "check_frame_window",
    "encode_obj",
    "compute_data_hash",
    "compute_mesh_hash",
    "detect_stitch_edges",
    "encode_param",
    "compute_param_hash",
    "prepare_upload",
]


def prepare_upload(
    context,
    *,
    want_data: bool = True,
    want_param: bool = True,
) -> tuple[bytes, bytes, str, str]:
    """Single source of truth for what gets sent up the wire.

    Builds each payload tree once, encodes to CBOR, and hashes the
    encoded bytes so the upload-time hash and the click-time drift
    hash use the same algorithm. The server echoes the hashes on every
    status response; ``SOLVER_OT_Run`` and ``SOLVER_OT_UpdateParams``
    re-compute against the live scene to decide whether the user has
    drifted from the last upload.

    Returns ``(data, param, data_hash, param_hash)``. Either payload
    is ``b""`` (and its hash ``""``) when its ``want_*`` flag is False.
    """
    data, data_hash = encode_obj_with_hash(context) if want_data else (b"", "")
    param, param_hash = encode_param_with_hash(context) if want_param else (b"", "")
    return data, param, data_hash, param_hash
