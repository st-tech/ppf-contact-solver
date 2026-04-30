# File: manifest.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Project manifest: a single JSON document at the project root that
# describes what the addon knows about this project — addon version,
# profile schema version, last session id, and an inventory of PC2
# cache files with their sha256 and frame count.
#
# Purpose:
#   * Detect orphaned PC2 files (on disk but not referenced by any
#     active object UUID) so they can be cleaned up or flagged.
#   * Detect schema-version mismatches after an addon upgrade and
#     route through a migration function.
#   * Reconcile on reconnect: compare ``last_session_id`` with what
#     the remote still has running.
#
# The manifest is advisory — the addon still works if it's missing
# (lazily created on first save).  Its presence enables features that
# would otherwise require speculative scanning.

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

MANIFEST_FILENAME = ".ppf_manifest.json"
MANIFEST_SCHEMA_VERSION = 1


@dataclass
class PC2Entry:
    """One row per cached PC2 file."""
    object_uuid: str = ""
    filename: str = ""           # basename only
    sha256: str = ""
    frame_count: int = 0
    size_bytes: int = 0


@dataclass
class ProjectManifest:
    """The complete manifest document.

    Versions: ``schema_version`` tracks the shape of this document;
    ``addon_version`` records the addon release that last wrote it.
    """

    schema_version: int = MANIFEST_SCHEMA_VERSION
    addon_version: str = ""
    last_session_id: str = ""
    last_write_utc: float = 0.0
    pc2_entries: list[PC2Entry] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ProjectManifest":
        entries = [
            PC2Entry(**e) if isinstance(e, dict) else e
            for e in data.get("pc2_entries", [])
        ]
        return cls(
            schema_version=int(data.get("schema_version", 0)),
            addon_version=str(data.get("addon_version", "")),
            last_session_id=str(data.get("last_session_id", "")),
            last_write_utc=float(data.get("last_write_utc", 0.0)),
            pc2_entries=entries,
        )


def manifest_path_for(blend_path: str) -> Optional[str]:
    """Resolve the manifest path next to a .blend file.

    Returns ``None`` when ``blend_path`` is empty (unsaved scene).
    """
    if not blend_path:
        return None
    return os.path.join(os.path.dirname(blend_path), MANIFEST_FILENAME)


def load_manifest(blend_path: str) -> Optional[ProjectManifest]:
    """Read the manifest next to *blend_path*.  Returns None when
    absent or corrupt (silent fallback — manifest is advisory)."""
    path = manifest_path_for(blend_path)
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ProjectManifest.from_dict(data)
    except (OSError, ValueError, TypeError):
        return None


def save_manifest(blend_path: str, manifest: ProjectManifest) -> None:
    """Atomically write the manifest next to *blend_path*.  Silent
    no-op when the scene isn't saved yet."""
    import tempfile
    import time

    path = manifest_path_for(blend_path)
    if not path:
        return
    manifest.last_write_utc = time.time()
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
    # Temp-file + rename for atomicity — a partial write would leave
    # the old manifest intact and be ignored.
    dir_ = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_, prefix=".ppf_manifest.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path)
    except OSError:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def sha256_file(path: str, chunk_size: int = 65536) -> str:
    """Compute the sha256 of a file.  Returns empty string on error
    (file missing, permission denied)."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                buf = f.read(chunk_size)
                if not buf:
                    break
                h.update(buf)
        return h.hexdigest()
    except OSError:
        return ""


def build_pc2_inventory(pc2_dir: str, active_uuids: set[str]) -> list[PC2Entry]:
    """Scan *pc2_dir* and return one PC2Entry per UUID-named .pc2 file.

    ``active_uuids`` is used only to set ``PC2Entry.object_uuid`` when
    the filename matches; files present on disk but not in the set
    are still included so the caller can decide to prune or adopt.
    """
    if not pc2_dir or not os.path.isdir(pc2_dir):
        return []
    from .pc2 import read_pc2_frame_count

    entries: list[PC2Entry] = []
    for name in sorted(os.listdir(pc2_dir)):
        if not name.endswith(".pc2"):
            continue
        full = os.path.join(pc2_dir, name)
        if not os.path.isfile(full):
            continue
        stem = name[:-4]
        uuid_match = stem if stem in active_uuids else ""
        try:
            frame_count = int(read_pc2_frame_count(full))
        except (OSError, ValueError, TypeError):
            frame_count = 0
        try:
            size_bytes = os.path.getsize(full)
        except OSError:
            size_bytes = 0
        entries.append(PC2Entry(
            object_uuid=uuid_match,
            filename=name,
            sha256=sha256_file(full),
            frame_count=frame_count,
            size_bytes=size_bytes,
        ))
    return entries


def find_orphan_pc2(manifest: Optional[ProjectManifest], active_uuids: set[str]) -> list[str]:
    """Return filenames listed in the manifest (or inferred from disk)
    whose object_uuid is no longer in the active set."""
    if manifest is None:
        return []
    return [
        e.filename for e in manifest.pc2_entries
        if e.object_uuid and e.object_uuid not in active_uuids
    ]


def _collect_active_uuids() -> set[str]:
    """Union of every UUID referenced by an active group's assigned_objects."""
    import bpy  # pyright: ignore
    try:
        from ..models.groups import (
            get_addon_data,
            has_addon_data,
            iterate_active_object_groups,
        )
    except Exception:
        return set()
    uuids: set[str] = set()
    for scene in bpy.data.scenes:
        if not has_addon_data(scene):
            continue
        for group in iterate_active_object_groups(scene):
            for obj_ref in group.assigned_objects:
                if obj_ref.uuid:
                    uuids.add(obj_ref.uuid)
    return uuids


def _current_addon_version() -> str:
    """Addon version as a dotted string (e.g. '1.0.0'), or '' on failure.

    Reads the manifest TOML at the addon root. We don't fall back to the
    legacy bl_info dict, since the manifest is the source of truth in
    Blender 5+.
    """
    try:
        import tomllib  # Python 3.11+, ships with Blender 5
        from pathlib import Path
        manifest_path = Path(__file__).resolve().parent.parent / "blender_manifest.toml"
        with manifest_path.open("rb") as f:
            return str(tomllib.load(f).get("version", ""))
    except Exception:
        return ""


def write_manifest_now() -> Optional[ProjectManifest]:
    """Build and save a manifest for the currently-open .blend.

    Returns the manifest that was written, or None when the scene
    isn't saved yet (no anchor path).
    """
    import bpy  # pyright: ignore

    blend_path = bpy.data.filepath
    if not blend_path:
        return None

    from .pc2 import get_pc2_dir
    pc2_dir = get_pc2_dir()
    active_uuids = _collect_active_uuids()
    entries = build_pc2_inventory(pc2_dir, active_uuids)

    sid = ""
    try:
        from .facade import communicator
        sid = communicator.session_id or communicator.last_saved_session_id() or ""
    except Exception:
        pass

    manifest = ProjectManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        addon_version=_current_addon_version(),
        last_session_id=sid,
        pc2_entries=entries,
    )
    save_manifest(blend_path, manifest)
    return manifest


def reconcile_on_load() -> None:
    """Called from load_post.  Logs orphan-PC2 and version drift info.

    Never raises — manifest is advisory, not load-blocking.
    """
    import bpy  # pyright: ignore
    try:
        from ..models.console import console
    except Exception:
        return

    blend_path = bpy.data.filepath
    if not blend_path:
        return

    manifest = load_manifest(blend_path)
    if manifest is None:
        return

    addon_version = _current_addon_version()
    if manifest.addon_version and manifest.addon_version != addon_version:
        console.write(
            f"[manifest] opened with addon {addon_version}, was last "
            f"written by {manifest.addon_version}"
        )

    active = _collect_active_uuids()
    orphans = find_orphan_pc2(manifest, active)
    if orphans:
        console.write(
            f"[manifest] {len(orphans)} orphan PC2 file(s) on disk: "
            f"{', '.join(orphans[:5])}"
            + ("..." if len(orphans) > 5 else "")
        )
