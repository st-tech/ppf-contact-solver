# File: blender_addon/core/statistics_cache.py
# Code: GitHub Copilot
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Local cache for solver-produced per-object timeline statistics."""

from __future__ import annotations

import hashlib
import os
import struct
import tempfile

from .pc2 import get_pc2_dir


STATISTICS_VERSION = 1
MANIFEST_KIND = "StatisticsManifest"
FRAME_KIND = "StatisticsFrame"
MANIFEST_FILENAME = "statistics_manifest.cbor"

CHANNELS = (
    ("LOCATION_X", "Location X", "m", 0),
    ("LOCATION_Y", "Location Y", "m", 1),
    ("LOCATION_Z", "Location Z", "m", 2),
    ("VOLUME", "Volume", "m\u00b3", 3),
    ("SURFACE_AREA", "Surface Area", "m\u00b2", 4),
    ("AREA_STRETCH", "Area Stretch", "ratio", 5),
    ("ROD_LENGTH", "Rod Length", "m", 6),
    ("LENGTH_STRETCH", "Length Stretch", "ratio", 7),
    ("VELOCITY_X", "Velocity X", "m/s", 8),
    ("VELOCITY_Y", "Velocity Y", "m/s", 9),
    ("VELOCITY_Z", "Velocity Z", "m/s", 10),
    ("SPEED", "Speed", "m/s", 11),
    ("ACCELERATION_X", "Acceleration X", "m/s\u00b2", 12),
    ("ACCELERATION_Y", "Acceleration Y", "m/s\u00b2", 13),
    ("ACCELERATION_Z", "Acceleration Z", "m/s\u00b2", 14),
    ("ACCELERATION_MAGNITUDE", "Acceleration Magnitude", "m/s\u00b2", 15),
    ("ANGULAR_VELOCITY_X", "Angular Velocity X", "rad/s", 16),
    ("ANGULAR_VELOCITY_Y", "Angular Velocity Y", "rad/s", 17),
    ("ANGULAR_VELOCITY_Z", "Angular Velocity Z", "rad/s", 18),
    ("ANGULAR_SPEED", "Angular Speed", "rad/s", 19),
    ("ANGULAR_AXIS_X", "Angular Axis X", "axis", 20),
    ("ANGULAR_AXIS_Y", "Angular Axis Y", "axis", 21),
    ("ANGULAR_AXIS_Z", "Angular Axis Z", "axis", 22),
    ("VOLUME_STRETCH", "Volume Stretch", "ratio", 23),
    ("CONTACT_COUNT", "Contact Count", "count", 24),
)
CHANNEL_BY_ID = {channel[0]: channel for channel in CHANNELS}

_MAGIC = b"PPFSTAT1"
_CACHE_VERSION = 2
_HEADER = struct.Struct("<8sII16s")
_RECORD = struct.Struct("<dQQ24fQ")
_ZERO_RECORD = _RECORD.pack(0.0, 0, 0, *([0.0] * 24), 0)
_RECORD_PRESENT = 1
# Widest value the record's trailing unsigned 64-bit field can hold.
_MAX_CONTACT_COUNT = (1 << 64) - 1
# Characters that would make a UUID address something other than one file
# inside the cache directory. NUL is included because a path carrying it
# cannot be opened at all.
_UUID_FORBIDDEN = ("/", "\\", ":", "\0")


class StatisticsCacheError(RuntimeError):
    pass


def _cbor2():
    from .module import get_cbor2

    return get_cbor2()


def _is_number(value) -> bool:
    """Report whether a decoded value is a plain number.

    A CBOR boolean decodes to a Python bool, which is an int subclass, so an
    isinstance test would let ``true`` through as the number 1. Every numeric
    field here comes off the wire, so the test is on the exact type.
    """
    return type(value) is int or type(value) is float


def _safe_uuid(object_uuid: str) -> str:
    """Return the UUID as the plain filename component that names its cache file.

    The value arrives from a decoded manifest and is used to build a path, so
    it must be a single component and the mapping from UUID to filename must
    be injective: one UUID names one file, and two UUIDs never name the same
    one. A value that is not such a component is a corrupt manifest and is
    rejected here, so no caller can be handed a path outside the cache
    directory or a file shared with another object.
    """
    if not object_uuid:
        raise StatisticsCacheError("statistics object UUID is empty")
    if object_uuid in (".", "..") or any(
        char in object_uuid for char in _UUID_FORBIDDEN
    ):
        raise StatisticsCacheError(
            f"statistics object UUID is not a filename: {object_uuid!r}"
        )
    return object_uuid


def _manifest_path(base_dir: str | None = None) -> str:
    return os.path.join(base_dir or get_pc2_dir(), MANIFEST_FILENAME)


def _object_path(object_uuid: str, base_dir: str | None = None) -> str:
    return os.path.join(base_dir or get_pc2_dir(), f"{_safe_uuid(object_uuid)}.stats")


def _uuid_digest(object_uuid: str) -> bytes:
    return hashlib.blake2s(object_uuid.encode("utf-8"), digest_size=16).digest()


def _decode_envelope(blob: bytes, expected_kind: str) -> dict:
    try:
        envelope = _cbor2().loads(blob)
    except Exception as exc:
        raise StatisticsCacheError(f"statistics CBOR decode failed: {exc}") from exc
    if not isinstance(envelope, dict):
        raise StatisticsCacheError("statistics envelope must be a map")
    version = envelope.get("version")
    if version != STATISTICS_VERSION:
        raise StatisticsCacheError(
            f"statistics version mismatch: got {version}, expected {STATISTICS_VERSION}"
        )
    kind = envelope.get("kind")
    if kind != expected_kind:
        raise StatisticsCacheError(
            f"statistics kind mismatch: got {kind!r}, expected {expected_kind!r}"
        )
    payload = envelope.get("payload")
    if not isinstance(payload, dict):
        raise StatisticsCacheError("statistics payload must be a map")
    return payload


def decode_manifest(blob: bytes) -> dict:
    manifest = _decode_envelope(blob, MANIFEST_KIND)
    objects = manifest.get("objects")
    if not isinstance(objects, list):
        raise StatisticsCacheError("statistics manifest objects must be a list")
    seen = set()
    for position, obj in enumerate(objects):
        if not isinstance(obj, dict):
            raise StatisticsCacheError("statistics manifest object must be a map")
        if obj.get("object_index") != position:
            raise StatisticsCacheError(
                f"statistics manifest object {position} has invalid index"
            )
        object_uuid = obj.get("object_uuid")
        if not isinstance(object_uuid, str) or not object_uuid:
            raise StatisticsCacheError(
                f"statistics manifest object {position} has an empty UUID"
            )
        if object_uuid in seen:
            raise StatisticsCacheError(
                f"statistics manifest has duplicate UUID {object_uuid!r}"
            )
        seen.add(object_uuid)
        supported = obj.get("supported_channels")
        if not isinstance(supported, int) or supported < 0 or supported >> len(CHANNELS):
            raise StatisticsCacheError(
                f"statistics manifest object {position} has invalid channel mask"
            )
    return manifest


def decode_frame(blob: bytes, manifest: dict) -> dict:
    frame = _decode_envelope(blob, FRAME_KIND)
    solver_frame = frame.get("solver_frame")
    time_seconds = frame.get("time_seconds")
    objects = frame.get("objects")
    if type(solver_frame) is not int or solver_frame < 0:
        raise StatisticsCacheError("statistics frame index is invalid")
    if (
        not _is_number(time_seconds)
        or not float(time_seconds) >= 0.0
        or not float(time_seconds) < float("inf")
    ):
        raise StatisticsCacheError("statistics frame time is invalid")
    manifest_objects = manifest["objects"]
    if not isinstance(objects, list) or len(objects) != len(manifest_objects):
        raise StatisticsCacheError("statistics frame object count does not match manifest")
    for position, (record, obj) in enumerate(zip(objects, manifest_objects, strict=True)):
        if not isinstance(record, dict) or record.get("object_index") != position:
            raise StatisticsCacheError(
                f"statistics frame object {position} has invalid index"
            )
        valid = record.get("valid_channels")
        supported = obj["supported_channels"]
        if (
            not isinstance(valid, int)
            or valid < 0
            or valid >> len(CHANNELS)
            or valid & ~supported
        ):
            raise StatisticsCacheError(
                f"statistics frame object {position} has invalid channel mask"
            )
    return frame


def install_manifest(blob: bytes) -> dict:
    manifest = decode_manifest(blob)
    path = _manifest_path()
    old = None
    try:
        with open(path, "rb") as file:
            old = file.read()
    except FileNotFoundError:
        pass
    if old != blob:
        if old is not None:
            clear_statistics_cache()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "wb") as file:
            file.write(blob)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
    return manifest


def load_manifest() -> dict | None:
    try:
        with open(_manifest_path(), "rb") as file:
            return decode_manifest(file.read())
    except FileNotFoundError:
        return None


def _scalar(field: str, value) -> float:
    """Return a decoded field as a float that the record struct can pack.

    Every value reaching this point came out of a CBOR map, so it may be of
    any type the format can carry. Only a plain number converts; a bool, a
    string, a container, or an integer too large for a float is a corrupt
    frame and is named as such instead of escaping as a bare ValueError,
    TypeError, or OverflowError.
    """
    if not _is_number(value):
        raise StatisticsCacheError(f"statistics {field} is not a number")
    try:
        return float(value)
    except OverflowError as exc:
        raise StatisticsCacheError(f"statistics {field} is out of range") from exc


def _flatten_record(record: dict) -> tuple[list[float], int]:
    def vector(name):
        value = record.get(name)
        if not isinstance(value, list) or len(value) != 3:
            raise StatisticsCacheError(f"statistics {name} must have three components")
        return [
            _scalar(f"{name}[{axis}]", component)
            for axis, component in enumerate(value)
        ]

    def scalar(name):
        return _scalar(name, record.get(name, 0.0))

    values = [
        *vector("location"),
        scalar("volume"),
        scalar("surface_area"),
        scalar("area_stretch"),
        scalar("rod_length"),
        scalar("length_stretch"),
        *vector("velocity"),
        scalar("speed"),
        *vector("acceleration"),
        scalar("acceleration_magnitude"),
        *vector("angular_velocity"),
        scalar("angular_speed"),
        *vector("angular_axis"),
        scalar("volume_stretch"),
    ]
    contact_count = record.get("contact_count", 0)
    if (
        type(contact_count) is not int
        or contact_count < 0
        or contact_count > _MAX_CONTACT_COUNT
    ):
        raise StatisticsCacheError("statistics contact count is invalid")
    return values, contact_count


def _ensure_object_file(object_uuid: str) -> str:
    path = _object_path(object_uuid)
    digest = _uuid_digest(object_uuid)
    if os.path.exists(path):
        _validate_header(path, object_uuid)
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        file.write(_HEADER.pack(_MAGIC, _CACHE_VERSION, _RECORD.size, digest))
        file.flush()
        os.fsync(file.fileno())
    return path


def _validate_header(path: str, object_uuid: str) -> None:
    with open(path, "rb") as file:
        header = file.read(_HEADER.size)
    if len(header) != _HEADER.size:
        raise StatisticsCacheError(f"statistics cache header is truncated: {path}")
    magic, version, record_size, found_digest = _HEADER.unpack(header)
    if (
        magic != _MAGIC
        or version != _CACHE_VERSION
        or record_size != _RECORD.size
        or found_digest != _uuid_digest(object_uuid)
    ):
        raise StatisticsCacheError(f"statistics cache header mismatch: {path}")


def write_frame_blob(
    blob: bytes, manifest: dict | None = None, *, max_solver_frame: int
) -> int:
    """Write one decoded statistics frame into the per-object cache files.

    The frame index decoded from the blob selects the byte offset the write
    zero-fills each file up to, so it is bounded at both ends before any
    file grows. Below by ``decode_frame``, which admits only a non-negative
    integer, so the offset cannot address behind the header. Above by
    ``max_solver_frame``, the highest frame the caller has fetched, so the
    cache cannot be extended past the run the caller is reading.
    """
    if manifest is None:
        manifest = load_manifest()
    if manifest is None:
        raise StatisticsCacheError("statistics manifest is not installed")
    frame = decode_frame(blob, manifest)
    solver_frame = frame["solver_frame"]
    if solver_frame > max_solver_frame:
        raise StatisticsCacheError(
            f"statistics frame {solver_frame} is past fetched frame {max_solver_frame}"
        )
    time_seconds = float(frame["time_seconds"])
    for obj, record in zip(manifest["objects"], frame["objects"], strict=True):
        values, contact_count = _flatten_record(record)
        valid = record["valid_channels"]
        for bit, value in enumerate(values):
            if valid & (1 << bit) and not float("-inf") < value < float("inf"):
                raise StatisticsCacheError(
                    f"statistics object {obj['object_index']} has a non-finite valid value"
                )
        path = _ensure_object_file(obj["object_uuid"])
        offset = _HEADER.size + solver_frame * _RECORD.size
        with open(path, "r+b") as file:
            file.seek(0, os.SEEK_END)
            while file.tell() < offset:
                file.write(_ZERO_RECORD)
            file.seek(offset)
            file.write(_RECORD.pack(
                time_seconds, _RECORD_PRESENT, valid, *values, contact_count,
            ))
            file.flush()
            os.fsync(file.fileno())
    return solver_frame


def manifest_object(object_uuid: str, manifest: dict | None = None) -> dict | None:
    manifest = manifest if manifest is not None else load_manifest()
    if manifest is None:
        return None
    return next(
        (obj for obj in manifest["objects"] if obj["object_uuid"] == object_uuid),
        None,
    )


def read_record(object_uuid: str, solver_frame: int) -> dict | None:
    if solver_frame < 0:
        return None
    path = _object_path(object_uuid)
    try:
        _validate_header(path, object_uuid)
        with open(path, "rb") as file:
            file.seek(_HEADER.size + solver_frame * _RECORD.size)
            data = file.read(_RECORD.size)
    except FileNotFoundError:
        return None
    if len(data) == 0:
        return None
    if len(data) != _RECORD.size:
        raise StatisticsCacheError(f"statistics cache record is truncated: {path}")
    unpacked = _RECORD.unpack(data)
    if unpacked[1] & _RECORD_PRESENT == 0:
        return None
    return {
        "solver_frame": solver_frame,
        "time_seconds": unpacked[0],
        "valid_channels": unpacked[2],
        "values": unpacked[3:27],
        "contact_count": unpacked[27],
    }


def scalar_value(record: dict | None, channel_id: str) -> float | int | None:
    if record is None:
        return None
    channel = CHANNEL_BY_ID.get(channel_id)
    if channel is None:
        raise StatisticsCacheError(f"unknown statistics channel {channel_id!r}")
    bit = channel[3]
    if record["valid_channels"] & (1 << bit) == 0:
        return None
    if channel_id == "CONTACT_COUNT":
        return record["contact_count"]
    return record["values"][bit]


def iter_scalar_records(object_uuid: str, channel_id: str):
    path = _object_path(object_uuid)
    channel = CHANNEL_BY_ID.get(channel_id)
    if channel is None:
        raise StatisticsCacheError(f"unknown statistics channel {channel_id!r}")
    try:
        _validate_header(path, object_uuid)
        size = os.path.getsize(path)
    except FileNotFoundError:
        return
    payload_size = size - _HEADER.size
    if payload_size < 0 or payload_size % _RECORD.size:
        raise StatisticsCacheError(f"statistics cache size is invalid: {path}")
    count = payload_size // _RECORD.size
    with open(path, "rb") as file:
        file.seek(_HEADER.size)
        for solver_frame in range(count):
            data = file.read(_RECORD.size)
            # The count is derived from the file size, so every one of these
            # reads must return a whole record. A short read means the file
            # shrank while it was being walked.
            if len(data) != _RECORD.size:
                raise StatisticsCacheError(
                    f"statistics cache record is truncated: {path}"
                )
            unpacked = _RECORD.unpack(data)
            if unpacked[1] & _RECORD_PRESENT == 0:
                continue
            record = {
                "valid_channels": unpacked[2],
                "values": unpacked[3:27],
                "contact_count": unpacked[27],
            }
            yield solver_frame, unpacked[0], scalar_value(record, channel_id)


def clear_statistics_cache() -> None:
    directories = {get_pc2_dir()}
    if not bpy_data_is_saved():
        directories.add(os.path.join(tempfile.gettempdir(), "data"))
    for directory in directories:
        try:
            names = os.listdir(directory)
        except FileNotFoundError:
            continue
        for name in names:
            if name == MANIFEST_FILENAME or name.endswith(".stats"):
                try:
                    os.remove(os.path.join(directory, name))
                except FileNotFoundError:
                    pass


def bpy_data_is_saved() -> bool:
    import bpy  # pyright: ignore

    return bool(bpy.data.filepath)


def migrate_statistics_on_save() -> None:
    import shutil

    target_dir = get_pc2_dir()
    source_dir = os.path.realpath(os.path.join(tempfile.gettempdir(), "data"))
    if os.path.realpath(target_dir) == source_dir or not os.path.isdir(source_dir):
        return
    for name in os.listdir(source_dir):
        if name != MANIFEST_FILENAME and not name.endswith(".stats"):
            continue
        source = os.path.join(source_dir, name)
        if not os.path.isfile(source):
            continue
        os.makedirs(target_dir, exist_ok=True)
        shutil.move(source, os.path.join(target_dir, name))
