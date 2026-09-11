# File: handlers/statistics.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# MCP handlers for the per-object solver statistics cache: what the solver
# measured for each simulated object, frame by frame.  Thin adapters over
# ``core.statistics_cache``: resolve the caller's object name to the UUID the
# cache is keyed by, turn a Blender frame into a solver frame with
# ``resolve_start_frame``, and read.  Every handler here only reads; the cache
# is written by the frame-fetch path.

import bpy  # pyright: ignore

from ...core.encoder import resolve_start_frame
from ...core.statistics_cache import (
    CHANNEL_BY_ID,
    CHANNELS,
    iter_scalar_records,
    load_manifest,
    manifest_object,
    read_record,
    scalar_value,
)
from ...models.groups import get_addon_data
from ..decorators import MCPError, mcp_handler

# The panel's wording for a manifest that carries no entry, so an agent and an
# artist are told the same thing about the same state.
_UNAVAILABLE = "Statistics unavailable; rerun the simulation"


def _scene_state():
    """Addon state on the active scene."""
    scene = bpy.context.scene
    if scene is None:
        raise MCPError("No active Blender scene")
    return get_addon_data(scene).state


def _manifest():
    """The installed statistics manifest, or a refusal naming what is missing."""
    manifest = load_manifest()
    if manifest is None:
        raise MCPError(_UNAVAILABLE)
    return manifest


def _channel_ids(supported: int) -> list[str]:
    """Channel ids the supported-channel mask has set, in catalog order."""
    return [
        channel_id
        for channel_id, _label, _unit, bit in CHANNELS
        if supported & (1 << bit)
    ]


def _current_name(object_uuid: str) -> str | None:
    """The object's name in the scene now, or None if the UUID resolves to nothing."""
    from ...core.uuid_registry import get_object_by_uuid

    obj = get_object_by_uuid(object_uuid)
    return obj.name if obj is not None else None


def _resolve_object(object_name: str, manifest: dict) -> tuple[str, dict]:
    """Return (object_uuid, manifest entry) for a caller-supplied identifier.

    The identifier is a Blender object name, or the object_uuid from
    list_statistics_objects for statistics whose object is no longer in the
    scene.
    """
    from ...core.uuid_registry import get_object_uuid

    obj = bpy.data.objects.get(object_name)
    if obj is None:
        entry = manifest_object(object_name, manifest)
        if entry is None:
            raise MCPError(
                f"Object '{object_name}' is not in the scene and is not a "
                f"statistics object UUID; call list_statistics_objects for the "
                f"objects that have statistics"
            )
        return object_name, entry

    object_uuid = get_object_uuid(obj)
    if not object_uuid:
        raise MCPError(
            f"Object '{object_name}' has no UUID, so it was not part of a solve"
        )
    entry = manifest_object(object_uuid, manifest)
    if entry is None:
        raise MCPError(f"{_UNAVAILABLE} (object '{object_name}')")
    return object_uuid, entry


def _frame_argument(name: str, value) -> int | None:
    """Convert an optional frame argument to an int, or refuse it by name.

    A parameter typed ``int | None`` arrives exactly as the client sent it,
    because the decorator converts plain ``int`` hints only. Converting here
    keeps a frame that arrived as text out of the frame arithmetic.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise MCPError(f"{name} must be an integer frame, got {value!r}")
    try:
        number = float(value)
    except ValueError as exc:
        raise MCPError(f"{name} must be an integer frame, got {value!r}") from exc
    # A frame indexes a record, so a fractional one names no record. Truncating
    # it would read a different frame than the caller asked for and report that
    # as the answer.
    if number != int(number):
        raise MCPError(
            f"{name} must be a whole frame number, got {value!r}"
        )
    return int(number)


@mcp_handler
def list_statistics_objects():
    """List the objects the solver recorded statistics for, with their channels.

    The statistics are whatever is on disk from the last simulation whose
    frames were fetched, so this reports a past run, not what the scene would
    produce if it were run now.

    ``object_name`` is the object's name in the scene at this moment and is
    null when its UUID resolves to nothing, which happens once the object is
    deleted; pass ``object_uuid`` to the other statistics tools in that case.
    ``recorded_name`` and ``dynamics_type`` are what the solver stored at run
    time. ``channels`` holds the channel ids measured for that object, which is
    the set get_object_statistics_series accepts for it, and
    ``channel_catalog`` gives every channel's label and unit.

    ``start_frame`` is the Blender frame the solve starts on, which every frame
    number in these tools is expressed against.
    """
    manifest = _manifest()
    start_frame = resolve_start_frame(_scene_state())

    objects = []
    for entry in manifest["objects"]:
        object_uuid = entry["object_uuid"]
        objects.append(
            {
                "object_uuid": object_uuid,
                "object_name": _current_name(object_uuid),
                "recorded_name": entry.get("object_name"),
                "dynamics_type": entry.get("dynamics_type"),
                "channels": _channel_ids(entry["supported_channels"]),
            }
        )

    return {
        "objects": objects,
        "object_count": len(objects),
        "start_frame": start_frame,
        "channel_catalog": [
            {"id": channel_id, "label": label, "unit": unit}
            for channel_id, label, unit, _bit in CHANNELS
        ],
    }


@mcp_handler
def get_object_statistics(object_name: str, frame: int):
    """Read every channel the solver measured for one object at one frame.

    ``frame`` is a Blender timeline frame, the same number the statistics panel
    shows, and it is converted to the solver frame by subtracting the start
    frame reported as ``effective_start_frame`` by get_scene_parameters.

    Only the channels the object supports are returned, since which quantities
    exist depends on what the object is: a rod has a length, a solid has a
    volume. The remaining ids are listed under ``unsupported_channels``. A
    supported channel whose value the solver did not record for this frame
    comes back with a null ``value``.

    A frame the run never wrote is refused rather than reported as zero; call
    get_object_statistics_series for the frames that are present.

    Args:
        object_name: Blender object name, or the object_uuid from
            list_statistics_objects when the object is gone from the scene.
        frame: Blender timeline frame to read.
    """
    # A plain `int` hint does not keep a JSON boolean out: bool is a subclass
    # of int, so the decorator's type check passes it through and it would
    # reach the frame arithmetic as 0 or 1.
    frame = _frame_argument("frame", frame)
    manifest = _manifest()
    object_uuid, entry = _resolve_object(object_name, manifest)
    start_frame = resolve_start_frame(_scene_state())
    solver_frame = frame - start_frame

    record = read_record(object_uuid, solver_frame)
    if record is None:
        raise MCPError(
            f"No statistics recorded for '{object_name}' at frame {frame} "
            f"(solver frame {solver_frame}, start frame {start_frame}); call "
            f"get_object_statistics_series for the frames that are present"
        )

    supported = entry["supported_channels"]
    channels = [
        {
            "id": channel_id,
            "label": label,
            "unit": unit,
            "value": scalar_value(record, channel_id),
        }
        for channel_id, label, unit, bit in CHANNELS
        if supported & (1 << bit)
    ]

    return {
        "object_name": _current_name(object_uuid),
        "object_uuid": object_uuid,
        "dynamics_type": entry.get("dynamics_type"),
        "frame": frame,
        "solver_frame": solver_frame,
        "start_frame": start_frame,
        "time_s": record["time_seconds"],
        "channels": channels,
        "unsupported_channels": [
            channel_id
            for channel_id, _label, _unit, bit in CHANNELS
            if not supported & (1 << bit)
        ],
    }


@mcp_handler
def get_object_statistics_series(
    object_name: str,
    channel: str,
    frame_start: int | None = None,
    frame_end: int | None = None,
):
    """Read one channel of one object across frames, as the CSV export does.

    Returns one sample per recorded frame, each carrying the Blender frame, the
    simulated time in seconds, and the value. The channel is a single id from
    list_statistics_objects, so a vector is read one component at a time
    (LOCATION_X, LOCATION_Y, LOCATION_Z), and a channel the object does not
    support is refused instead of answered with nulls. A sample whose value the
    solver did not record for that frame carries a null ``value``.

    The window bounds are Blender frames and both ends are inclusive. Leaving
    one out extends the window to the recorded frames on that side, so leaving
    both out returns every frame in the cache. An empty ``samples`` list means
    no frame in the window has been recorded yet.

    Args:
        object_name: Blender object name, or the object_uuid from
            list_statistics_objects when the object is gone from the scene.
        channel: Channel id, for example SPEED or CONTACT_COUNT.
        frame_start: First Blender frame to include; omit for the earliest
            recorded frame.
        frame_end: Last Blender frame to include; omit for the latest recorded
            frame.
    """
    manifest = _manifest()
    object_uuid, entry = _resolve_object(object_name, manifest)

    channel_id = channel.strip().upper()
    definition = CHANNEL_BY_ID.get(channel_id)
    if definition is None:
        catalog = ", ".join(identifier for identifier, *_rest in CHANNELS)
        raise MCPError(
            f"Unknown statistics channel '{channel}'; valid ids are {catalog}"
        )

    supported = entry["supported_channels"]
    if not supported & (1 << definition[3]):
        available = ", ".join(_channel_ids(supported)) or "none"
        raise MCPError(
            f"Channel '{channel_id}' is not measured for '{object_name}'; "
            f"its channels are {available}"
        )

    first = _frame_argument("frame_start", frame_start)
    last = _frame_argument("frame_end", frame_end)
    if first is not None and last is not None and first > last:
        raise MCPError(f"frame_start {first} is after frame_end {last}")

    start_frame = resolve_start_frame(_scene_state())
    samples = []
    for solver_frame, time_seconds, value in iter_scalar_records(
        object_uuid, channel_id
    ):
        blender_frame = solver_frame + start_frame
        if first is not None and blender_frame < first:
            continue
        if last is not None and blender_frame > last:
            continue
        samples.append(
            {
                "frame": blender_frame,
                "time_s": time_seconds,
                "value": value,
            }
        )

    return {
        "object_name": _current_name(object_uuid),
        "object_uuid": object_uuid,
        "channel": {
            "id": channel_id,
            "label": definition[1],
            "unit": definition[2],
        },
        "start_frame": start_frame,
        "samples": samples,
        "sample_count": len(samples),
    }
