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
# Byte offsets into the PC2 header (see write_pc2): 12-byte magic +
# version(i) + n_verts(i) + start(f) + sampling(f) + n_frames(i).
PC2_NVERTS_OFFSET = 16
PC2_FRAMECOUNT_OFFSET = 28
MODIFIER_NAME = "ContactSolverCache"
# Cache namespace suffixes embedded in PC2 keys / on-disk filenames.
STATIC_DEFORM_SUFFIX = "_staticdeform"
PIN_DEFORM_SUFFIX = "__pindeform"

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
        f.seek(PC2_FRAMECOUNT_OFFSET)
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
        f.seek(PC2_FRAMECOUNT_OFFSET)
        f.write(struct.pack("<i", frame_tot + 1))
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, AttributeError):
            pass


def read_pc2_frame_count(filepath):
    """Read the frame_tot from a PC2 file header."""
    with open(filepath, "rb") as f:
        f.seek(PC2_FRAMECOUNT_OFFSET)
        return struct.unpack("<i", f.read(4))[0]


def read_pc2_n_verts(filepath):
    """Read the vertex count from a PC2 file header."""
    with open(filepath, "rb") as f:
        f.seek(PC2_NVERTS_OFFSET)
        return struct.unpack("<i", f.read(4))[0]


def read_pc2_frame(filepath, frame_idx, n_verts):
    """Read a specific frame from a PC2 file."""
    if frame_idx < 0:
        raise ValueError(
            f"read_pc2_frame: negative frame index {frame_idx} for {filepath}"
        )
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


def _load_pc2_array(filepath):
    """Read a whole PC2 file into an ``(n_frames, n_verts, 3)`` array.

    Returns ``None`` (and leaves no side effect) when the file is
    missing or carries an empty (0-frame / 0-vert) header. Reads the
    header and the float body from a single open handle so the bulk
    payload streams without reopening the file.
    """
    if not os.path.exists(filepath):
        return None
    with open(filepath, "rb") as f:
        f.seek(PC2_NVERTS_OFFSET)
        n_verts = struct.unpack("<i", f.read(4))[0]
        f.seek(PC2_FRAMECOUNT_OFFSET)
        n_frames = struct.unpack("<i", f.read(4))[0]
        if n_frames < 1 or n_verts < 1:
            return None
        data = numpy.frombuffer(f.read(), dtype="<f")
    return data.reshape(n_frames, n_verts, 3).copy()


def fill_gap_frames(filepath, from_frame_idx, to_frame_idx, n_verts, obj_key=None):
    """Fill gap frames by duplicating the nearest real frame across (from, to).

    If ``obj_key`` (a UUID string) is supplied and the object has any real
    frames tracked in ``_real_frames``, the largest real frame <=
    ``from_frame_idx`` is used as the source. This avoids cascading a
    previously-gap-filled pose into new gaps. When no real frame is available
    yet, the function falls back to reading ``from_frame_idx`` (old behavior)
    so the first-ever gap-fill still produces a plausible pose.

    Raises ``ValueError`` when ``from_frame_idx < 0``: there is no source
    frame to duplicate (a 0-frame header has no frame 0 either), so the
    caller must instead seed the leading gap from a known pose (rest /
    captured-deformation), as the file-create path does.
    """
    if from_frame_idx < 0:
        raise ValueError(
            f"fill_gap_frames: no source frame for from_frame_idx="
            f"{from_frame_idx} ({filepath})"
        )
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
        f.seek(PC2_FRAMECOUNT_OFFSET)
        old_tot = struct.unpack("<i", f.read(4))[0]
        new_tot = max(old_tot, to_frame_idx)
        f.seek(PC2_FRAMECOUNT_OFFSET)
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


def object_pc2_key_readonly(obj) -> str:
    """Return the existing PC2 key for ``obj`` with no side effects.

    Unlike :func:`object_pc2_key`, this never assigns a UUID, never
    reassigns another object's UUID on duplicate detection, and never
    migrates a legacy ``<name>.pc2`` file. It returns the empty string
    when ``obj`` has no UUID yet. Use it from read-only contexts (draw,
    operator poll, frame_change handlers): any object that actually owns
    a cache was assigned a UUID by the write-context capture path or by
    load_post migration, so an empty key here is behaviorally "no cache".
    """
    from .uuid_registry import get_object_uuid
    return get_object_uuid(obj)


# ---------------------------------------------------------------------------
# MESH_CACHE modifier management
# ---------------------------------------------------------------------------


# Modifier types that change vertex count or topology. MESH_CACHE
# must come BEFORE any of these in the stack so its per-frame data
# matches the vertex count baked into the PC2 file. Deform-only
# modifiers (Armature, Lattice, Hook, Shape Keys, ...) preserve
# vertex count and can sit above MESH_CACHE; their output is then
# overridden by the PC2 data on display. Geometry Nodes can do
# anything topology-wise, so we treat them as generative.
_GENERATIVE_MODIFIER_TYPES = frozenset({
    "ARRAY", "BEVEL", "BOOLEAN", "BUILD", "DECIMATE", "EDGE_SPLIT",
    "MASK", "MIRROR", "MULTIRES", "REMESH", "SCREW", "SKIN",
    "SOLIDIFY", "SUBSURF", "TRIANGULATE", "WELD", "WIREFRAME",
    "VOLUME_TO_MESH", "MESH_TO_VOLUME",
    "EXPLODE", "PARTICLE_INSTANCE",
    "NODES",
})


def _stack_preserves_vertex_count(obj) -> bool:
    """True if *obj*'s evaluated mesh keeps the base vertex count.

    Used to disambiguate a Geometry Nodes modifier, which lives in both
    the deforming and the generative type sets. When the whole stack
    preserves the vertex count, no modifier changes topology, so any GN
    present only displaces existing vertices (deform-only) and the
    cache must sit AFTER it. When the count differs, something is
    generative and we keep the conservative before-the-boundary
    placement. Returns False if evaluation isn't available, which keeps
    the safe (treat GN as generative) behavior.
    """
    try:
        import bpy
        base = len(obj.data.vertices)
        deps = bpy.context.evaluated_depsgraph_get()
        eval_obj = obj.evaluated_get(deps)
        mesh = eval_obj.to_mesh()
        try:
            return len(mesh.vertices) == base
        finally:
            eval_obj.to_mesh_clear()
    except Exception:
        return False


def _cache_insertion_index(obj) -> int:
    """Index where ContactSolverCache should sit so it lands AFTER every
    position-preserving deformer and BEFORE the first topology-changing
    modifier. Computed over the stack with the cache excluded, so the
    returned value is the cache's desired final index and can be passed
    straight to ``obj.modifiers.move(current_idx, here)``.

    A NODES (Geometry Nodes) modifier is treated as a topology boundary
    only when it actually changes the vertex count; a deform-only GN
    (e.g. a Set Position wave) is treated as a deformer so PC2 wins on
    display instead of the GN re-deforming the cache output on top of
    itself.
    """
    nodes_are_deformers = _stack_preserves_vertex_count(obj)
    pos = 0
    for m in obj.modifiers:
        if m.name == MODIFIER_NAME:
            continue
        is_boundary = m.type in _GENERATIVE_MODIFIER_TYPES
        if m.type == "NODES" and nodes_are_deformers:
            is_boundary = False
        if is_boundary:
            return pos
        pos += 1
    return pos


def setup_mesh_cache_modifier(obj, pc2_path, frame_start=0.0,
                              place_after_deformers=False):
    """Add or update a MESH_CACHE modifier on *obj* pointing to *pc2_path*.

    Also stamps the current session id on the object as ``_solver_session``
    so reconcile-on-reconnect can tell whether the modifier's cached data
    was produced by the currently-connected run.

    ``place_after_deformers``: when True, position the new modifier so it
    sits AFTER all position-preserving deformers (Armature, Lattice,
    Hook, Shape Keys, ...) and BEFORE the first topology-changing
    modifier (Subsurf, Mirror, Solidify, ...). Used for STATIC colliders
    whose captured-deformation cache feeds into the simulator: the
    simulator-projected positions in PC2 must win over the deformers
    that produced the input, while the user's downstream visual
    decorators (Subsurf, etc.) keep applying on top. The default
    (False) puts the modifier first, which is the right semantics for
    dynamic objects (sim output is the base; decorators run after).
    """
    mod = obj.modifiers.get(MODIFIER_NAME)
    if mod is None:
        mod = obj.modifiers.new(name=MODIFIER_NAME, type="MESH_CACHE")
    # Enforce placement for both freshly-created AND already-present
    # caches: a stale order saved by an older session (or a file made
    # before a deformer was added) is corrected on the next setup,
    # rather than leaving the cache wherever it happened to land.
    idx = obj.modifiers.find(MODIFIER_NAME)
    if place_after_deformers:
        target = _cache_insertion_index(obj)
        if target != idx:
            obj.modifiers.move(idx, target)
    else:
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


def suspend_mesh_cache_display(obj):
    """Disable the ContactSolverCache modifier's viewport evaluation and
    return its prior ``show_viewport`` flag (``None`` if absent).

    Capture Deformation records the pure deformer (Armature, Lattice,
    Mesh Deform, shape-key, ...) output by sampling
    ``obj.evaluated_get(depsgraph).to_mesh()``. The ContactSolverCache
    MESH_CACHE modifier runs with ``deform_mode='OVERWRITE'``, so while it
    is enabled the evaluated mesh is the solver's *previous* output, not
    the deformer result. Re-capturing after a run would then record stale
    (often gap-filled) solver positions and feed them back as the next
    input. Capture operators suspend it for the duration of the job and
    restore it via :func:`resume_mesh_cache_display`. The flag is the only
    state touched, and the modifier is addon-owned, so this stays within
    the addon's own state.
    """
    if obj is None:
        return None
    mod = obj.modifiers.get(MODIFIER_NAME)
    if mod is None:
        return None
    prior = bool(mod.show_viewport)
    try:
        mod.show_viewport = False
    except (AttributeError, RuntimeError):
        # Restricted context (e.g. a UI draw() handler): Blender forbids
        # writing to ID data. Callers that might run there gate the heavy
        # path off separately; returning None here means "nothing to
        # restore" so a stray call can't crash.
        return None
    return prior


def resume_mesh_cache_display(obj, prior):
    """Restore the ContactSolverCache ``show_viewport`` flag saved by
    :func:`suspend_mesh_cache_display`. No-op when *prior* is ``None``
    (modifier was absent) or the modifier has since been removed."""
    if prior is None or obj is None:
        return
    mod = obj.modifiers.get(MODIFIER_NAME)
    if mod is not None:
        mod.show_viewport = bool(prior)


def modifiers_above_cache(obj) -> list:
    """Names of modifiers sitting above ContactSolverCache in *obj*'s
    stack. This is exactly the set :func:`strip_modifiers_above_cache`
    would remove on bake finalize; the bake-confirm dialog renders
    this list so the user can see what they'd lose before clicking
    OK. Empty when the cache modifier is at index 0, absent, or
    *obj* is not a MESH (curves and other types are never struck).
    """
    if obj is None or getattr(obj, "type", None) != "MESH":
        return []
    names: list[str] = []
    for m in obj.modifiers:
        if m.name == MODIFIER_NAME:
            return names
        names.append(m.name)
    return []


def strip_modifiers_above_cache(obj):
    """Remove every modifier that sits above ContactSolverCache in the
    stack. Used by Bake operators on STATIC objects whose Case-3
    deformation feeders (Armature, Lattice, MeshDeform, Shape Keys
    via a Shape Keys modifier, ...) would otherwise re-deform the
    baked shape-key data once ContactSolverCache itself is removed.

    No-op when the cache modifier is absent (e.g. already removed)
    or sits at index 0 with nothing above it (dynamics, Case 1, and
    Case 2 stacks all land here, so the helper is safe to call
    unconditionally for any baked MESH).
    """
    for name in modifiers_above_cache(obj):
        mod = obj.modifiers.get(name)
        if mod is not None:
            obj.modifiers.remove(mod)


def has_mesh_cache(obj):
    """Check if *obj* has simulation animation (modifier, curve cache, or PC2 file).

    Reached from read-only contexts (panel draw, operator poll), so it
    uses the side-effect-free :func:`object_pc2_key_readonly`. An object
    that owns a cache always carries a UUID (assigned by the capture path
    or load_post migration), so an empty key means no cache.
    """
    if obj.modifiers.get(MODIFIER_NAME) is not None:
        return True
    key = object_pc2_key_readonly(obj)
    if not key:
        return False
    if obj.type == "CURVE" and key in _curve_cache:
        return True
    # Also check for PC2 file on disk (survives addon reload)
    if os.path.exists(get_pc2_path(key)):
        return True
    return False


def scene_has_solver_cache() -> bool:
    """Fast, stateless: True when any object carries solver animation.

    A ``ContactSolverCache`` MESH_CACHE modifier is only ever placed on a
    solver-managed (assigned) object, and removing an object from a group
    strips it (see ``cleanup_mesh_cache``), so scanning every object's
    modifiers is equivalent to a per-active-group scan in practice. Curves
    keep their cache in the in-memory ``_curve_cache`` instead of a modifier.

    Used by panel ``poll()`` methods that run on every UI redraw. Reading
    ``obj.modifiers`` is a C-level collection lookup, so this whole scan is
    ~100x cheaper than resolving each assigned object by UUID (which dominated
    redraw time on large scenes) -- and it recomputes every call, so it never
    goes stale the way a memoized result can.
    """
    for obj in bpy.data.objects:
        if obj.modifiers.get(MODIFIER_NAME) is not None:
            return True
    return bool(_curve_cache)


def scene_has_static_deform_cache() -> bool:
    """Fast, stateless: True when any STATIC-deform cache exists (captured in
    memory, or a PC2 on disk after a fresh reload).

    Used by the "Clear All Deformations" poll on every redraw. The in-memory
    dict is an O(1) check; the on-disk fallback is a single directory scan for
    ``*_staticdeform.pc2`` -- both avoid the per-object UUID resolve + per-
    object ``os.path.exists`` that made the old scan ~10ms on large scenes.
    """
    if _static_deform_cache:
        return True
    suffix = STATIC_DEFORM_SUFFIX + ".pc2"
    try:
        with os.scandir(get_pc2_dir()) as it:
            for entry in it:
                if entry.name.endswith(suffix):
                    return True
    except OSError:
        # No pc2 dir yet (nothing ever written) -> no cache.
        pass
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
    global _curve_last_frame
    arr = _load_pc2_array(get_pc2_path(key))
    if arr is not None:
        _curve_cache[key] = arr
        # Invalidate the frame-dedup guard. A live append grows this
        # cache mid-tick; if an earlier per-frame scene.frame_set this
        # tick (STATIC per-frame matrices, client.apply_animation) already
        # ran the playback handler against the shorter cache, it stamped
        # _curve_last_frame and clamped the rod a frame back. Clearing the
        # guard lets the tick's final frame_set re-apply with the just-
        # loaded frame so rods stay in lockstep with the meshes' MESH_CACHE.
        _curve_last_frame = -1


def unload_curve_cache(key=None):
    """Remove one or all curves from the in-memory cache."""
    global _curve_last_frame
    if key is None:
        _curve_cache.clear()
        _handles_freed.clear()
        # Reset the log-once dedup so "log once per message" resets
        # between playback sessions rather than per process.
        _curve_handler_error_reported.clear()
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


def _apply_curve_cvs(obj, frame_idx, key=None):
    """Set curve CVs from cached data for the given frame index.

    When ``key`` is given (the playback handler already resolved it via
    the read-only accessor), reuse it. When ``None`` (the
    ``cleanup_mesh_cache`` operator path), fall back to the side-effecting
    :func:`object_pc2_key` so the legacy ``<name>.pc2`` migration still
    runs in that write-permitted context.
    """
    if key is None:
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
            # Read-only key: this runs from frame_change_post, so it must
            # not stamp a UUID on an uncached curve or migrate any file.
            key = object_pc2_key_readonly(obj)
            if key and key in _curve_cache:
                _apply_curve_cvs(obj, frame_idx, key=key)
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


def refresh_curves_at_current_frame():
    """Force curve CV playback to the current frame, bypassing the
    per-frame dedup guard.

    Called at the end of an apply tick (``client.apply_animation``): the
    frame_change handler may have already run earlier in the tick (e.g. a
    per-frame ``scene.frame_set`` for STATIC matrices) against a curve
    cache that was still a frame short, and its ``_curve_last_frame``
    guard would otherwise suppress the corrected re-apply when the tick's
    final ``frame_set`` lands on the same frame. This guarantees rods sit
    on the same frame as the meshes' MESH_CACHE.
    """
    global _curve_last_frame
    _curve_last_frame = -1
    _apply_curves_at_current_frame()


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
    # Runs on addon reload (see __init__.py), giving a clean per-reload
    # reset of the curve-handler log-once dedup.
    _curve_handler_error_reported.clear()



# ---------------------------------------------------------------------------
# Static-deform mesh cache (deforming STATIC collider input)
#
# A STATIC mesh whose modifier stack deforms vertices (Armature,
# MeshDeform, Lattice, shape keys, ...) carries a per-frame absolute
# vertex buffer captured from Blender's depsgraph. Cache layout matches
# the pin-input PC2 (PointCache2 binary, dict[key]->ndarray), but the
# data is solver-input only: the depsgraph already paints the deformed
# mesh in the viewport, so this namespace registers NO frame_change
# handler. Suffix ``_staticdeform`` keeps these keys isolated from
# pin-input PC2 (``_pininput``) so an object moving between groups
# can't inherit a stale animation from the wrong subsystem.
# ---------------------------------------------------------------------------

_static_deform_cache: dict[str, numpy.ndarray] = {}


def static_deform_pc2_key(obj) -> str:
    """Return the UUID-backed key for an object's static-deform PC2 file."""
    from .uuid_registry import get_or_create_object_uuid
    return get_or_create_object_uuid(obj) + STATIC_DEFORM_SUFFIX


def write_static_deform_pc2(obj, frames):
    """Write an object's static-deform cache to PC2 and stash it in memory.

    Args:
        obj: The mesh object.
        frames: ndarray ``(n_frames, n_verts, 3)`` of solver-world-space
            positions; index 0 maps to ``scene.frame_start``.
    """
    arr = numpy.ascontiguousarray(frames, dtype="<f")
    key = static_deform_pc2_key(obj)
    path = get_pc2_path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_pc2(path, [arr[i] for i in range(arr.shape[0])])
    _static_deform_cache[key] = arr.copy()


def load_static_deform_cache(key):
    """Load a static-deform PC2 file into the in-memory cache."""
    arr = _load_pc2_array(get_pc2_path(key))
    if arr is not None:
        _static_deform_cache[key] = arr


def unload_static_deform_cache(key=None):
    """Remove one or all objects from the static-deform cache."""
    if key is None:
        _static_deform_cache.clear()
    else:
        _static_deform_cache.pop(key, None)


def has_static_deform_cache_key(key):
    """Check if a static-deform key has cached data in memory."""
    return key in _static_deform_cache


def has_static_deform_animation(obj):
    """True if *obj* has a static-deform PC2 (in memory or on disk)."""
    if obj is None or getattr(obj, "type", None) != "MESH":
        return False
    key = static_deform_pc2_key(obj)
    if key in _static_deform_cache:
        return True
    return os.path.exists(get_pc2_path(key))


def get_static_deform_cache(obj):
    """Return the cached ``(n_frames, n_verts, 3)`` array for *obj*.

    Lazy-loads from the PC2 file on disk (file reopen / addon reload).
    Returns ``None`` when the object has no static-deform cache.
    """
    key = static_deform_pc2_key(obj)
    if key not in _static_deform_cache:
        load_static_deform_cache(key)
    return _static_deform_cache.get(key)


def remove_static_deform_pc2(obj):
    """Delete an object's static-deform PC2 file and drop the cache entry."""
    key = static_deform_pc2_key(obj)
    path = get_pc2_path(key)
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
    unload_static_deform_cache(key)


def _clear_all_by_suffix(file_suffix):
    """Delete every PC2 file in ``get_pc2_dir`` whose name ends with
    *file_suffix*. Walks the directory rather than live objects so caches
    whose owning object was already removed from the scene are still
    deleted (no orphan files left behind)."""
    pc2_dir = get_pc2_dir()
    if not os.path.isdir(pc2_dir):
        return
    for fname in os.listdir(pc2_dir):
        if fname.endswith(file_suffix):
            try:
                os.remove(os.path.join(pc2_dir, fname))
            except OSError:
                pass


def clear_all_static_deform_animation():
    """Delete every static-deform PC2 file and clear the in-memory cache."""
    _clear_all_by_suffix(STATIC_DEFORM_SUFFIX + ".pc2")
    unload_static_deform_cache()


# ---------------------------------------------------------------------------
# Pin-deform animation cache (Capture Deformation on SHELL/SOLID/ROD pins)
#
# A pin whose vertices move because of a deformer on the cloth mesh
# (Armature, Lattice, MeshDeform, Hook, shape keys, drivers, ...)
# carries a per-frame absolute vertex buffer captured from the
# depsgraph. The encoder reads this in preference to the manual
# vertex-co fcurves the Make Keyframe button writes (implicit
# PC2-wins). Suffix ``_pindeform`` keeps these keys isolated from
# the STATIC ``_staticdeform`` namespace.
#
# Keyed per (object, vertex-group) so two pins on the same mesh
# capture independently. The cache stores ONLY the pin's vertices,
# in the same order as ``_get_pin_indices`` returns at capture
# time, in solver world space (z2y @ matrix_world @ co_local). The
# decoder consumes consecutive frames as MoveBy deltas; absolute
# space cancels out, but world space matches what the cloth mesh's
# initial-upload coords use, so frame-0 alignment is exact.
# ---------------------------------------------------------------------------

_pin_anim_cache: dict[str, numpy.ndarray] = {}


def _safe_vg_for_key(vg_name: str) -> str:
    """Make a vertex-group name filesystem-safe for use in a PC2 key.

    Mirrors the substitution ``get_pc2_path`` applies to keys before
    they hit disk. Applied here so the lookup key in
    ``_pin_anim_cache`` matches the eventual filename one-to-one,
    which makes the load_post reconciler's "is this cache present?"
    check identical to ``os.path.exists`` on the same key.
    """
    return vg_name.replace(" ", "_").replace("/", "_")


def pin_anim_pc2_key(obj, vg_name: str) -> str:
    """Return the UUID-backed key for a pin's depsgraph-captured cache."""
    from .uuid_registry import get_or_create_object_uuid
    return (f"{get_or_create_object_uuid(obj)}__"
            f"{_safe_vg_for_key(vg_name)}{PIN_DEFORM_SUFFIX}")


def write_pin_anim_pc2(obj, vg_name: str, frames):
    """Write a pin's captured per-frame positions to PC2 and the in-memory cache.

    Args:
        obj: The mesh object the pin belongs to.
        vg_name: The vertex group name identifying the pin.
        frames: ndarray ``(n_frames, n_pin_verts, 3)`` of solver-world-space
            positions; index 0 maps to ``scene.frame_start``. Vertex
            order matches ``_get_pin_indices(obj, vg_name)`` at capture
            time.
    """
    arr = numpy.ascontiguousarray(frames, dtype="<f")
    key = pin_anim_pc2_key(obj, vg_name)
    path = get_pc2_path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_pc2(path, [arr[i] for i in range(arr.shape[0])])
    _pin_anim_cache[key] = arr.copy()


def load_pin_anim_cache(key):
    """Load a pin-deform PC2 file into the in-memory cache (no-op if missing)."""
    arr = _load_pc2_array(get_pc2_path(key))
    if arr is not None:
        _pin_anim_cache[key] = arr


def unload_pin_anim_cache(key=None):
    """Drop one or all pin-deform entries from the in-memory cache."""
    if key is None:
        _pin_anim_cache.clear()
    else:
        _pin_anim_cache.pop(key, None)


def has_pin_anim_pc2(obj, vg_name: str) -> bool:
    """True if a pin-deform PC2 exists (in memory or on disk)."""
    if obj is None or getattr(obj, "type", None) != "MESH":
        return False
    key = pin_anim_pc2_key(obj, vg_name)
    if key in _pin_anim_cache:
        return True
    return os.path.exists(get_pc2_path(key))


def get_pin_anim_cache(obj, vg_name: str):
    """Return ``(n_frames, n_pin_verts, 3)`` for a pin, lazy-loading from disk."""
    key = pin_anim_pc2_key(obj, vg_name)
    if key not in _pin_anim_cache:
        load_pin_anim_cache(key)
    return _pin_anim_cache.get(key)


def remove_pin_anim_pc2(obj, vg_name: str):
    """Delete a pin's PC2 file and drop its in-memory cache entry."""
    key = pin_anim_pc2_key(obj, vg_name)
    path = get_pc2_path(key)
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
    unload_pin_anim_cache(key)


def clear_all_pin_anim_animation():
    """Delete every pin-deform PC2 file and clear the in-memory cache.

    Pin caches use a per-pin filename suffix, not a per-object key, so
    matching by suffix on disk catches every cache regardless of which
    pin authored it.
    """
    _clear_all_by_suffix(PIN_DEFORM_SUFFIX + ".pc2")
    unload_pin_anim_cache()


# ---------------------------------------------------------------------------
# Save-post migration (temp → data/)
# ---------------------------------------------------------------------------


@bpy.app.handlers.persistent
def migrate_pc2_on_save(*_args):
    """Move PC2 files from temp dir to 'data/' next to .blend after first save.

    Called from a ``bpy.app.handlers.save_post`` handler.
    """
    import shutil

    from .uuid_registry import get_object_uuid

    blend_path = bpy.data.filepath
    if not blend_path:
        return
    target_dir = get_pc2_dir()
    tmp_dir = os.path.realpath(os.path.join(tempfile.gettempdir(), "data"))
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            # Static-deform PC2: temp -> data/ migration on first save.
            # Derive the on-disk filename through get_pc2_path so it can
            # never diverge from the name every reader/existence check uses.
            sd_key = static_deform_pc2_key(obj)
            sd_name = os.path.basename(get_pc2_path(sd_key))
            sd_old = os.path.realpath(os.path.join(tmp_dir, sd_name))
            if os.path.exists(sd_old):
                os.makedirs(target_dir, exist_ok=True)
                sd_new = os.path.join(target_dir, sd_name)
                if os.path.realpath(sd_new) != sd_old:
                    shutil.move(sd_old, sd_new)
                    load_static_deform_cache(sd_key)
            # Pin-deform PC2: per-pin caches share the object's UUID
            # prefix and end with ``__pindeform.pc2``. Migrate every
            # match for this object.
            uid = get_object_uuid(obj)
            if uid and os.path.isdir(tmp_dir):
                prefix = f"{uid}__"
                for fname in os.listdir(tmp_dir):
                    if (not fname.startswith(prefix)
                            or not fname.endswith(PIN_DEFORM_SUFFIX + ".pc2")):
                        continue
                    pd_old = os.path.realpath(os.path.join(tmp_dir, fname))
                    if not os.path.exists(pd_old):
                        continue
                    os.makedirs(target_dir, exist_ok=True)
                    pd_new = os.path.join(target_dir, fname)
                    if os.path.realpath(pd_new) != pd_old:
                        shutil.move(pd_old, pd_new)
                        load_pin_anim_cache(fname[:-len(".pc2")])
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
            # Derive the filename through get_pc2_path so the migrate
            # name can never diverge from what every reader computes.
            curve_name = os.path.basename(get_pc2_path(key))
            old_path = os.path.realpath(
                os.path.join(tmp_dir, curve_name)
            )
            if not os.path.exists(old_path):
                continue
            os.makedirs(target_dir, exist_ok=True)
            new_path = os.path.join(target_dir, curve_name)
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

        from .facade import communicator as com
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
