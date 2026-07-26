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
    rate-parameterized quantity authored in ANIMATION seconds (a SPIN op's
    degrees per second, velocity-schedule speeds) must be multiplied by this
    at encode so its per-frame effect is invariant under Time Scale and stays
    in sync with keyframed motion. Physics initial conditions (the group
    initial velocity) and physical constants (gravity, materials) are NOT
    scaled.
    """
    scale = float(getattr(state, "time_scale", 1.0))
    if scale <= 0.0:
        raise ValueError(f"Time Scale must be positive, got {scale}")
    return scale


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
    frame before it yields a negative time; callers that can be handed such a
    frame (an op window authored ahead of the solve) must clamp, since the
    solver's schedules start at zero.
    """
    return (float(frame) - start_frame) / fps


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
    "resolve_start_frame",
    "resolve_start_frame_or_default",
    "frame_to_time",
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
