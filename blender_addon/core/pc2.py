# File: pc2.py
# License: Apache v2.0
#
# PC2 (PointCache2) file writer and Blender MESH_CACHE modifier management.
# Used to store per-frame vertex positions for simulation animation.

import os
import struct
import tempfile

import numpy

try:
    import bpy  # pyright: ignore
except ImportError:
    bpy = None

PC2_HEADER = b"POINTCACHE2\0"
PC2_VERSION = 1
PC2_HEADER_SIZE = 32
FRAME_VERTEX_SIZE = 12  # 3 * sizeof(float32)
MODIFIER_NAME = "ContactSolverCache"

# Gap tracking: {obj_uuid: set of real (non-gap-filled) frame indices}
_real_frames: dict[str, set[int]] = {}


# ---------------------------------------------------------------------------
# PC2 file I/O
# ---------------------------------------------------------------------------


def write_pc2(filepath, frames, start=0.0, sampling=1.0):
    """Write a complete PC2 file from a list of per-frame vertex positions.

    Args:
        filepath: Output .pc2 file path.
        frames: List of numpy arrays, each shape (n_verts, 3).
        start: Start frame (float).
        sampling: Frame sampling rate (float).
    """
    n_frames = len(frames)
    n_verts = frames[0].shape[0]
    with open(filepath, "wb") as f:
        f.write(PC2_HEADER)
        f.write(struct.pack("<i", PC2_VERSION))
        f.write(struct.pack("<i", n_verts))
        f.write(struct.pack("<f", start))
        f.write(struct.pack("<f", sampling))
        f.write(struct.pack("<i", n_frames))
        for positions in frames:
            f.write(numpy.asarray(positions, dtype="<f").tobytes())


def create_pc2_file(filepath, n_verts, start=0.0, sampling=1.0):
    """Create an empty PC2 file with header only (0 frames)."""
    with open(filepath, "wb") as f:
        f.write(PC2_HEADER)
        f.write(struct.pack("<i", PC2_VERSION))
        f.write(struct.pack("<i", n_verts))
        f.write(struct.pack("<f", start))
        f.write(struct.pack("<f", sampling))
        f.write(struct.pack("<i", 0))


def append_pc2_frame(filepath, positions, n_verts):
    """Append a single frame to an existing PC2 file and update the header.

    Ordering is important: we write the frame data at EOF and fsync
    before bumping the header's frame count.  If MESH_CACHE or another
    reader observes the file mid-write, it sees either the old count
    with the old data (stale but consistent) or the new count with the
    new data on disk — never "new count, data missing".
    """
    with open(filepath, "r+b") as f:
        f.seek(28)
        frame_tot = struct.unpack("<i", f.read(4))[0]
        # Data first: append to EOF and flush to disk.
        f.seek(0, 2)
        f.write(numpy.asarray(positions, dtype="<f").tobytes())
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, AttributeError):
            pass
        # Header update publishes the new frame to readers.
        f.seek(28)
        f.write(struct.pack("<i", frame_tot + 1))
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, AttributeError):
            pass


def read_pc2_frame_count(filepath):
    """Read the frame_tot from a PC2 file header."""
    with open(filepath, "rb") as f:
        f.seek(28)
        return struct.unpack("<i", f.read(4))[0]


def read_pc2_n_verts(filepath):
    """Read the vertex count from a PC2 file header (offset 16)."""
    with open(filepath, "rb") as f:
        f.seek(16)
        return struct.unpack("<i", f.read(4))[0]


def read_pc2_frame(filepath, frame_idx, n_verts):
    """Read a specific frame from a PC2 file."""
    offset = PC2_HEADER_SIZE + frame_idx * n_verts * FRAME_VERTEX_SIZE
    with open(filepath, "rb") as f:
        f.seek(offset)
        data = f.read(n_verts * FRAME_VERTEX_SIZE)
        return numpy.frombuffer(data, dtype="<f").reshape(n_verts, 3).copy()


def overwrite_pc2_frame(filepath, frame_idx, positions, n_verts):
    """Overwrite a specific frame in an existing PC2 file."""
    offset = PC2_HEADER_SIZE + frame_idx * n_verts * FRAME_VERTEX_SIZE
    with open(filepath, "r+b") as f:
        f.seek(offset)
        f.write(numpy.asarray(positions, dtype="<f").tobytes())


def fill_gap_frames(filepath, from_frame_idx, to_frame_idx, n_verts, obj_key=None):
    """Fill gap frames by duplicating the nearest real frame across (from, to).

    If ``obj_key`` (a UUID string) is supplied and the object has any real
    frames tracked in ``_real_frames``, the largest real frame <=
    ``from_frame_idx`` is used as the source. This avoids cascading a
    previously-gap-filled pose into new gaps. When no real frame is available
    yet, the function falls back to reading ``from_frame_idx`` (old behavior)
    so the first-ever gap-fill still produces a plausible pose.
    """
    src_idx = from_frame_idx
    if obj_key is not None:
        reals = _real_frames.get(obj_key, set())
        candidates = [i for i in reals if i <= from_frame_idx]
        if candidates:
            src_idx = max(candidates)
    src_data = read_pc2_frame(filepath, src_idx, n_verts)
    src_bytes = numpy.asarray(src_data, dtype="<f").tobytes()
    with open(filepath, "r+b") as f:
        # Append gap frames at end of file
        f.seek(0, 2)
        for _ in range(from_frame_idx + 1, to_frame_idx):
            f.write(src_bytes)
        # Update frame_tot
        f.seek(28)
        old_tot = struct.unpack("<i", f.read(4))[0]
        new_tot = max(old_tot, to_frame_idx)
        f.seek(28)
        f.write(struct.pack("<i", new_tot))


# ---------------------------------------------------------------------------
# Gap tracking
# ---------------------------------------------------------------------------


def mark_real_frame(obj_key, frame_idx):
    """Record that a frame contains real (non-gap-filled) data."""
    _real_frames.setdefault(obj_key, set()).add(frame_idx)


def get_gap_frame_indices(obj_key, total_frames):
    """Return frame indices that are gap-filled (not real)."""
    real = _real_frames.get(obj_key, set())
    return [i for i in range(total_frames) if i not in real]


def clear_gap_tracking(obj_key=None):
    """Clear gap tracking for one or all objects."""
    if obj_key is None:
        _real_frames.clear()
    else:
        _real_frames.pop(obj_key, None)


# ---------------------------------------------------------------------------
# File path helpers
# ---------------------------------------------------------------------------


def get_pc2_dir():
    """Get directory for PC2 cache files.

    When the .blend is saved, files are stored under a per-blend subfolder so
    multiple .blend files sharing the same directory don't collide:
        <blend_dir>/data/<blend_basename_noext>/<name>.pc2
    Before first save, a flat temp dir is used; migrate_pc2_on_save moves
    those files into the per-blend layout on save_post.
    """
    blend_path = bpy.data.filepath if bpy else ""
    if blend_path:
        basename = os.path.splitext(os.path.basename(blend_path))[0]
        return os.path.join(os.path.dirname(blend_path), "data", basename)
    return os.path.join(tempfile.gettempdir(), "data")


def get_pc2_path(key):
    """Get the PC2 file path for a given stable key string.

    The key is normally an object UUID (see ``object_pc2_key``). For
    legacy data that was written under the object's Blender name, the
    same sanitisation still produces the old filename so
    ``object_pc2_key`` can detect and migrate it.
    """
    safe = key.replace(" ", "_").replace("/", "_")
    return os.path.join(get_pc2_dir(), f"{safe}.pc2")


def object_pc2_key(obj) -> str:
    """Return the UUID-backed PC2 key for ``obj``.

    Ensures the object carries a UUID, and migrates a legacy
    ``<name>.pc2`` file in place so the UUID-keyed path is the only
    one on disk afterwards. Return value is safe to pass to
    ``get_pc2_path`` and to use as a ``_real_frames`` / ``_curve_cache``
    dict key.
    """
    from .uuid_registry import get_or_create_object_uuid
    uid = get_or_create_object_uuid(obj)
    target = get_pc2_path(uid)
    if not os.path.exists(target):
        legacy = get_pc2_path(obj.name)
        if legacy != target and os.path.exists(legacy):
            try:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                os.rename(legacy, target)
            except OSError:
                pass
    return uid


# ---------------------------------------------------------------------------
# MESH_CACHE modifier management
# ---------------------------------------------------------------------------


def setup_mesh_cache_modifier(obj, pc2_path, frame_start=0.0):
    """Add or update a MESH_CACHE modifier on *obj* pointing to *pc2_path*.

    Also stamps the current session id on the object as ``_solver_session``
    so reconcile-on-reconnect can tell whether the modifier's cached data
    was produced by the currently-connected run.
    """
    mod = obj.modifiers.get(MODIFIER_NAME)
    if mod is None:
        mod = obj.modifiers.new(name=MODIFIER_NAME, type="MESH_CACHE")
        # Move to first so it applies before other modifiers
        idx = obj.modifiers.find(MODIFIER_NAME)
        if idx > 0:
            obj.modifiers.move(idx, 0)
    mod.cache_format = "PC2"
    mod.filepath = bpy.path.relpath(pc2_path) if bpy.data.filepath else pc2_path
    mod.interpolation = "LINEAR"
    mod.deform_mode = "OVERWRITE"
    mod.play_mode = "SCENE"
    mod.time_mode = "FRAME"
    mod.frame_start = frame_start
    mod.frame_scale = 1.0
    mod.factor = 1.0
    mod.forward_axis = "POS_Y"
    mod.up_axis = "POS_Z"
    # Session stamp: record which connected run bound this cache so
    # a later reopen can detect stale bindings.
    try:
        from .facade import communicator as _com
        sid = _com.session_id or ""
        if sid:
            obj["_solver_session"] = sid
    except Exception:
        pass
    return mod


def remove_mesh_cache_modifier(obj):
    """Remove the MESH_CACHE modifier from *obj* if present."""
    mod = obj.modifiers.get(MODIFIER_NAME)
    if mod is not None:
        obj.modifiers.remove(mod)


def has_mesh_cache(obj):
    """Check if *obj* has simulation animation (modifier, curve cache, or PC2 file)."""
    if obj.modifiers.get(MODIFIER_NAME) is not None:
        return True
    key = object_pc2_key(obj)
    if obj.type == "CURVE" and key in _curve_cache:
        return True
    # Also check for PC2 file on disk (survives addon reload)
    if os.path.exists(get_pc2_path(key)):
        return True
    return False


def remove_pc2_file(key):
    """Delete the PC2 file for the given UUID key if it exists."""
    path = get_pc2_path(key)
    if os.path.exists(path):
        os.remove(path)
    clear_gap_tracking(key)


def cleanup_mesh_cache(obj, *, keep_baked_pose: bool = False):
    """Remove modifier/cache and delete PC2 file for *obj*.

    For curves, restores rest-pose CVs (frame 0 from PC2) before deleting,
    since the handler modifies curve data directly. Pass
    ``keep_baked_pose=True`` when the caller has already written baked
    fcurves/positions and wants neither the rest-pose restore nor the
    bezier handle_type restore (AUTO handles would silently overwrite
    the baked handle positions).
    """
    remove_mesh_cache_modifier(obj)
    key = object_pc2_key(obj)
    if obj.type == "CURVE":
        if not keep_baked_pose:
            cache = _curve_cache.get(key)
            if cache is not None and len(cache) > 0:
                _apply_curve_cvs(obj, 0)
            snapshot = _saved_handle_types.pop(key, None)
            _handles_freed.discard(key)
            if snapshot is not None:
                for spline, per_spline in zip(obj.data.splines, snapshot):
                    if spline.type != "BEZIER":
                        continue
                    for bp, (left, right) in zip(spline.bezier_points, per_spline):
                        bp.handle_left_type = left
                        bp.handle_right_type = right
        else:
            _saved_handle_types.pop(key, None)
            _handles_freed.discard(key)
        unload_curve_cache(key)
    remove_pc2_file(key)


# ---------------------------------------------------------------------------
# Curve CV playback via persistent handlers
# ---------------------------------------------------------------------------

# In-memory cache keyed by object UUID: {uuid: numpy array shape (n_frames, n_cvs, 3)}
_curve_cache: dict[str, numpy.ndarray] = {}


def load_curve_cache(key):
    """Load a curve's PC2 file into the in-memory cache (keyed by UUID)."""
    path = get_pc2_path(key)
    if not os.path.exists(path):
        return
    with open(path, "rb") as f:
        f.seek(16)
        n_verts = struct.unpack("<i", f.read(4))[0]
        f.seek(28)
        n_frames = struct.unpack("<i", f.read(4))[0]
        if n_frames < 1 or n_verts < 1:
            return
        data = numpy.frombuffer(f.read(), dtype="<f")
    _curve_cache[key] = data.reshape(n_frames, n_verts, 3).copy()


def unload_curve_cache(key=None):
    """Remove one or all curves from the in-memory cache."""
    global _curve_last_frame
    if key is None:
        _curve_cache.clear()
        _handles_freed.clear()
    else:
        _curve_cache.pop(key, None)
        _handles_freed.discard(key)
    _curve_last_frame = -1


def has_curve_cache(key):
    """Check if a curve UUID has cached animation data."""
    return key in _curve_cache


# Tracks objects whose handle types have been set to FREE and remembers
# the original (left_type, right_type) per spline→point so
# ``cleanup_mesh_cache`` can restore the user's intent when playback
# ends. Keyed by object UUID.
_handles_freed: set[str] = set()
_saved_handle_types: dict[str, list] = {}


def _apply_curve_cvs(obj, frame_idx):
    """Set curve CVs from cached data for the given frame index."""
    key = object_pc2_key(obj)
    cache = _curve_cache.get(key)
    if cache is None:
        return
    n_frames = cache.shape[0]
    idx = max(0, min(frame_idx, n_frames - 1))
    cv_data = cache[idx]
    # Set handle types to FREE once (writing properties triggers updates)
    if key not in _handles_freed:
        snapshot = []
        for spline in obj.data.splines:
            per_spline = []
            if spline.type == "BEZIER":
                for bp in spline.bezier_points:
                    per_spline.append((bp.handle_left_type, bp.handle_right_type))
                    bp.handle_left_type = "FREE"
                    bp.handle_right_type = "FREE"
            snapshot.append(per_spline)
        _saved_handle_types[key] = snapshot
        _handles_freed.add(key)
    cv_i = 0
    for spline in obj.data.splines:
        if spline.type == "BEZIER":
            for bp in spline.bezier_points:
                bp.handle_left = cv_data[cv_i]
                cv_i += 1
                bp.co = cv_data[cv_i]
                cv_i += 1
                bp.handle_right = cv_data[cv_i]
                cv_i += 1
        else:
            for pt in spline.points:
                c = cv_data[cv_i]
                pt.co[0] = c[0]
                pt.co[1] = c[1]
                pt.co[2] = c[2]
                cv_i += 1


_curve_last_frame: int = -1


def _ensure_curve_caches():
    """Lazy-load curve caches from PC2 files on disk.

    Handles file reopen / addon reload: the PC2 files persist but the
    in-memory cache is lost.  Scans curve objects and loads any PC2
    files that aren't already cached.
    """
    for obj in bpy.data.objects:
        if obj.type != "CURVE":
            continue
        key = object_pc2_key(obj)
        if key in _curve_cache:
            continue
        if os.path.exists(get_pc2_path(key)):
            load_curve_cache(key)


def _apply_curves_at_current_frame():
    """Apply cached curve CVs for the current scene frame.

    Lazy-loads caches from PC2 files if needed (file reopen / addon reload).
    Tracks last-applied frame to avoid redundant work.
    """
    global _curve_last_frame
    try:
        if not _curve_cache:
            _ensure_curve_caches()
        if not _curve_cache:
            return
        current = bpy.context.scene.frame_current
        if current == _curve_last_frame:
            return
        _curve_last_frame = current
        frame_idx = current - 1
        for obj in bpy.data.objects:
            if obj.type != "CURVE":
                continue
            if object_pc2_key(obj) in _curve_cache:
                _apply_curve_cvs(obj, frame_idx)
    except Exception as e:
        # frame_change handlers run VERY frequently — log once per
        # message rather than every tick, but don't silence entirely.
        _log_curve_handler_error(str(e))


_curve_handler_error_reported: set[str] = set()


def _log_curve_handler_error(msg: str) -> None:
    if msg in _curve_handler_error_reported:
        return
    _curve_handler_error_reported.add(msg)
    try:
        from ..models.console import console
        console.write(f"[curve playback] {msg}")
    except Exception:
        pass


@bpy.app.handlers.persistent
def curve_frame_change_handler(*_args):
    """``frame_change_post`` handler for curve CV playback."""
    _apply_curves_at_current_frame()


def ensure_curve_handler():
    """Register frame_change_post handler for curve CV playback.

    Detects stale handlers by __name__ (not identity) so a reloaded
    module generation isn't double-added next to the old function
    object from the previous generation.
    """
    fc = bpy.app.handlers.frame_change_post
    if not any(
        getattr(h, "__name__", "") == "curve_frame_change_handler"
        for h in fc
    ):
        fc.append(curve_frame_change_handler)


def remove_curve_handler():
    """Unregister every handler named ``curve_frame_change_handler``
    from ``frame_change_post``, regardless of module generation."""
    fc = bpy.app.handlers.frame_change_post
    for h in list(fc):
        if getattr(h, "__name__", "") == "curve_frame_change_handler":
            fc.remove(h)


# ---------------------------------------------------------------------------
# Save-post migration (temp → data/)
# ---------------------------------------------------------------------------


@bpy.app.handlers.persistent
def migrate_pc2_on_save(*_args):
    """Move PC2 files from temp dir to 'data/' next to .blend after first save.

    Called from a ``bpy.app.handlers.save_post`` handler.
    """
    import shutil

    blend_path = bpy.data.filepath
    if not blend_path:
        return
    target_dir = get_pc2_dir()
    tmp_dir = os.path.realpath(os.path.join(tempfile.gettempdir(), "data"))
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            mod = obj.modifiers.get(MODIFIER_NAME)
            if mod is None:
                continue
            abs_path = os.path.realpath(bpy.path.abspath(mod.filepath))
            if not abs_path.startswith(tmp_dir + os.sep) and abs_path != tmp_dir:
                continue
            if not os.path.exists(abs_path):
                continue
            os.makedirs(target_dir, exist_ok=True)
            filename = os.path.basename(abs_path)
            new_path = os.path.join(target_dir, filename)
            shutil.move(abs_path, new_path)
            mod.filepath = bpy.path.relpath(new_path)
        elif obj.type == "CURVE":
            key = object_pc2_key(obj)
            if key not in _curve_cache:
                continue
            safe_key = key.replace(" ", "_").replace("/", "_")
            old_path = os.path.realpath(
                os.path.join(tmp_dir, f"{safe_key}.pc2")
            )
            if not os.path.exists(old_path):
                continue
            os.makedirs(target_dir, exist_ok=True)
            new_path = os.path.join(target_dir, f"{safe_key}.pc2")
            shutil.move(old_path, new_path)
            # Reload cache from new location
            load_curve_cache(key)


# ---------------------------------------------------------------------------
# Render-animation warning (gentle, non-blocking)
#
# Blender doesn't pass an ``is_animation`` flag to render handlers, and
# ``wm.operators`` doesn't record RENDER_OT_render, so we detect
# animation renders by counting ``render_pre`` calls: a single-image
# render reaches frame 1 only; an animation reaches frame 2+.
# ---------------------------------------------------------------------------


_render_job: dict[str, object] = {"frame_count": 0, "warned": False}


@bpy.app.handlers.persistent
def reset_render_counter(_scene, *_args):
    """``render_init`` handler: reset per-job state."""
    _render_job["frame_count"] = 0
    _render_job["warned"] = False


@bpy.app.handlers.persistent
def warn_missing_frames_on_render(scene, *_args):
    """``render_pre`` handler: on the 2nd frame of a render job we know
    it's an animation. If remote frames are still unfetched, show a
    one-shot non-blocking popup."""
    try:
        _render_job["frame_count"] = int(_render_job["frame_count"]) + 1
        if _render_job["frame_count"] < 2 or _render_job["warned"]:
            return

        from .client import communicator as com
        from .derived import is_server_busy_from_response as is_running
        from ..models.groups import get_addon_data

        response = com.info.response
        remote_frames = int(response.get("frame", 0)) if response else 0
        if remote_frames <= 0 or is_running(response):
            _render_job["warned"] = True
            return
        fetched = get_addon_data(scene).state.convert_fetched_frames_to_list()
        n_missing = remote_frames - len(fetched)
        _render_job["warned"] = True
        if n_missing <= 0:
            return

        msg = (f"{n_missing} frames unfetched — rendered animation may be "
               "incomplete. Press \"Fetch All Animation\" first.")

        def _popup():
            def _draw(self, _ctx):
                self.layout.label(text=msg, icon="ERROR")
            bpy.context.window_manager.popup_menu(
                _draw, title="Unfetched Frames", icon="ERROR")
            return None
        bpy.app.timers.register(_popup, first_interval=0.0)
    except Exception:
        pass
