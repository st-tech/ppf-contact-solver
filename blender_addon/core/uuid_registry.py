# File: core/uuid_registry.py
# License: Apache v2.0
#
# Stable UUID-based identification for Blender objects.
# UUIDs are stored as custom properties on objects and persist across renames.
# Vertex groups are identified by a content hash of their vertex indices,
# enabling recovery after renames.

import hashlib
import struct
import uuid as _uuid

import bpy  # pyright: ignore

_OBJ_UUID_KEY = "_solver_uuid"


def _is_writable(obj: bpy.types.Object) -> bool:
    """True when a custom property can be safely written to obj.

    Library-linked objects (obj.library is not None) are read-only;
    attempting to write raises `RuntimeError: Writing to ID classes in
    this context is not allowed`.  Library overrides (obj.override_library)
    DO allow custom-prop writes because overrides have local storage for
    ID-properties, so we allow those.
    """
    return obj.library is None


def get_or_create_object_uuid(obj: bpy.types.Object) -> str:
    """Get or assign a stable UUID for a Blender object.

    If another object shares the same UUID (duplicate), the object that is
    referenced by a group keeps its UUID; the other gets a fresh one.

    For library-linked objects we cannot write custom props — we return
    any existing UUID but never create one.  Callers that receive an
    empty string MUST treat the object as unidentifiable.
    """
    uid = obj.get(_OBJ_UUID_KEY)
    if uid:
        uid = str(uid)
        # Only run duplicate-resolution for writable objects.  A linked
        # object cannot participate in reassignment anyway.
        if not _is_writable(obj):
            return uid
        for other in bpy.data.objects:
            if other != obj and str(other.get(_OBJ_UUID_KEY, "")) == uid:
                # Duplicate found — decide who keeps the UUID by
                # consulting the rename-tracker's last-known mapping.
                # _prev_names[uid] records which object instance was
                # observed with this UUID in the previous depsgraph
                # tick, so it IS the original; the one we're handling
                # here may be the copy produced by e.g. duplicate-obj.
                # Names are NOT used — they may have diverged since.
                prev_name = _prev_names.get(uid)
                # Original is whichever object is currently tracked
                # under this UUID. If obj matches the tracked name
                # it's the original; otherwise obj is the copy.
                obj_is_original = (prev_name is not None and prev_name == obj.name)
                if obj_is_original and _is_writable(other):
                    # obj is the original — reassign the other
                    new_uid = str(_uuid.uuid4())
                    other[_OBJ_UUID_KEY] = new_uid
                    _cache[new_uid] = other.name
                elif _is_writable(obj):
                    # obj is the copy (or tie-broken as such) — reassign it
                    uid = str(_uuid.uuid4())
                    obj[_OBJ_UUID_KEY] = uid
                    _cache[uid] = obj.name
                # else: both sides unwritable — nothing we can do; the
                # duplicate UUID persists harmlessly until a writable
                # copy is made.
                break
        return uid
    if not _is_writable(obj):
        return ""
    uid = str(_uuid.uuid4())
    obj[_OBJ_UUID_KEY] = uid
    return uid


def get_object_uuid(obj: bpy.types.Object) -> str:
    """Get UUID of an object, or empty string if none assigned."""
    uid = obj.get(_OBJ_UUID_KEY)
    return str(uid) if uid else ""


_cache: dict[str, str] = {}  # uuid -> object name (lazy, warm)


def get_object_by_uuid(uid: str) -> "bpy.types.Object | None":
    """Find a Blender object by its UUID. O(1) when cached."""
    if not uid:
        raise ValueError("get_object_by_uuid called with empty/None UUID")
    # Try cache
    name = _cache.get(uid)
    if name:
        obj = bpy.data.objects.get(name)
        if obj and obj.get(_OBJ_UUID_KEY) == uid:
            return obj
    # Cache miss — scan once
    for obj in bpy.data.objects:
        if obj.get(_OBJ_UUID_KEY) == uid:
            _cache[uid] = obj.name
            return obj
    return None


def resolve_object(uid: str) -> "bpy.types.Object | None":
    """Resolve an object by UUID."""
    return get_object_by_uuid(uid) if uid else None


def resolve_assigned(assigned) -> "bpy.types.Object | None":
    """Resolve an AssignedObject by UUID.

    Syncs display-cache names if the object was renamed.
    Auto-migration at load_post (core.migrate.migrate_legacy_data) populates
    UUIDs on every AssignedObject before this is reached — a missing UUID
    here means the record was never migrated (e.g. programmatic insertion
    without going through the proper API), and we return None to surface
    the failure rather than silently pick by name.
    """
    uid = getattr(assigned, "uuid", "")
    if not uid:
        raise ValueError(
            f"resolve_assigned: AssignedObject has empty UUID"
            f" (name={getattr(assigned, 'name', '?')})"
        )
    obj = get_object_by_uuid(uid)
    if obj is None:
        return None
    if hasattr(assigned, "name") and assigned.name != obj.name:
        _sync_all_names(uid, obj.name)
    return obj


def _sync_all_names(uid: str, new_name: str):
    """Sync every stored display name that references *uid* across the addon.

    Identity-carrying strings that get reconciled:
      - AssignedObject.name                  (group membership display)
      - PinVertexGroupItem.name              (composite [obj][vg])
      - SavedPinGroup.object_name            (keyframe cache display)
      - MergePairItem.object_a / object_b    (snap-pair display)

    Not synced here: vertex-group renames.  Those don't change an object's
    name; ``resolve_pin`` handles VG-rename reconciliation via content hash.
    cross_stitch_json is UUID-only since the latest migration.
    """
    try:
        from ..models.groups import (
            decode_vertex_group_identifier,
            encode_vertex_group_identifier,
            get_addon_data,
            iterate_active_object_groups,
        )
        scene = bpy.context.scene
        if not scene:
            return
        for group in iterate_active_object_groups(scene):
            for assigned in group.assigned_objects:
                if assigned.uuid == uid and assigned.name != new_name:
                    assigned.name = new_name
            for pin_item in group.pin_vertex_groups:
                if pin_item.object_uuid == uid:
                    obj_name, vg_name = decode_vertex_group_identifier(pin_item.name)
                    if obj_name and obj_name != new_name:
                        pin_item.name = encode_vertex_group_identifier(new_name, vg_name)
        root = get_addon_data(scene)
        if hasattr(root, "state"):
            for grp_entry in root.state.saved_pin_keyframes:
                if grp_entry.object_uuid == uid and grp_entry.object_name != new_name:
                    grp_entry.object_name = new_name
            for pair in root.state.merge_pairs:
                if pair.object_a_uuid == uid and pair.object_a != new_name:
                    pair.object_a = new_name
                if pair.object_b_uuid == uid and pair.object_b != new_name:
                    pair.object_b = new_name
    except Exception:
        pass


def _sync_vg_name(uid: str, old_vg: str, new_vg: str):
    """Sync SavedPinGroup.vertex_group after a VG rename on object *uid*.

    ``encoder/pin.py`` keys saved keyframes by (object_uuid, vertex_group
    name), so a VG rename without this sync orphans the keyframe cache
    and silently produces empty pin_anim at encode time.
    """
    if old_vg == new_vg:
        return
    try:
        from ..models.groups import get_addon_data
        scene = bpy.context.scene
        if not scene:
            return
        root = get_addon_data(scene)
        if not hasattr(root, "state"):
            return
        for grp_entry in root.state.saved_pin_keyframes:
            if grp_entry.object_uuid == uid and grp_entry.vertex_group == old_vg:
                grp_entry.vertex_group = new_vg
    except Exception:
        pass


def resolve_pin(pin_item) -> "bpy.types.Object | None":
    """Resolve a PinVertexGroupItem's object by UUID, syncing stale names."""
    uid = getattr(pin_item, "object_uuid", "")
    if not uid:
        raise ValueError(
            f"resolve_pin: PinVertexGroupItem has empty object_uuid"
            f" (name={getattr(pin_item, 'name', '?')})"
        )
    obj = get_object_by_uuid(uid)
    if not obj:
        return None
    # Sync stale pin identifier for object and vertex group renames
    try:
        from ..models.groups import decode_vertex_group_identifier
        obj_name, vg_name = decode_vertex_group_identifier(pin_item.name)
        if obj_name and vg_name:
            updated_obj = obj_name != obj.name
            # Resolve VG rename via content hash
            stored_hash = int(pin_item.vg_hash) if pin_item.vg_hash else 0
            old_vg_name = vg_name
            if stored_hash:
                current_vg = resolve_vg_name(obj, vg_name, stored_hash)
                if current_vg != vg_name:
                    vg_name = current_vg
                    updated_obj = True  # force name update
            if updated_obj:
                from ..models.groups import encode_vertex_group_identifier
                pin_item.name = encode_vertex_group_identifier(obj.name, vg_name)
                # VG rename: migrate custom property keys on the object
                # so _pin_{old} / _embedded_move_{old} don't orphan.
                if old_vg_name != vg_name:
                    for prefix in ("_pin_", "_embedded_move_"):
                        old_key = f"{prefix}{old_vg_name}"
                        new_key = f"{prefix}{vg_name}"
                        if old_key in obj and new_key not in obj:
                            obj[new_key] = obj[old_key]
                            del obj[old_key]
                    _sync_vg_name(uid, old_vg_name, vg_name)
    except Exception:
        pass
    return obj


# ---------------------------------------------------------------------------
# Vertex group identification by content hash
# ---------------------------------------------------------------------------

def _get_vg_indices(obj: bpy.types.Object, vg_name: str) -> list[int]:
    """Get sorted vertex indices belonging to a vertex group."""
    vg = obj.vertex_groups.get(vg_name)
    if not vg:
        return []
    idx = vg.index
    indices = []
    for v in obj.data.vertices:
        for g in v.groups:
            if g.group == idx:
                indices.append(v.index)
                break
    return sorted(indices)


def _hash_indices(indices: list[int]) -> int:
    """Compute a 64-bit blake2b hash from sorted vertex indices."""
    if not indices:
        return 0
    data = struct.pack(f"<{len(indices)}I", *sorted(indices))
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return struct.unpack("<Q", digest)[0]


def _get_curve_pin_indices(obj: bpy.types.Object, vg_name: str) -> list[int]:
    """Get pin indices from a curve's custom property."""
    import json as _json
    raw = obj.get(f"_pin_{vg_name}")
    if raw:
        return sorted(_json.loads(raw))
    return []


def compute_vg_hash(obj: bpy.types.Object, vg_name: str) -> int:
    """Compute a 64-bit content hash from vertex/pin indices."""
    if obj.type == "CURVE":
        return _hash_indices(_get_curve_pin_indices(obj, vg_name))
    return _hash_indices(_get_vg_indices(obj, vg_name))


def _iter_vg_candidates(obj: bpy.types.Object):
    """Yield (name, hash) for all vertex group / curve pin slots on obj."""
    if obj.type == "MESH":
        for vg in obj.vertex_groups:
            yield vg.name, _hash_indices(_get_vg_indices(obj, vg.name))
    elif obj.type == "CURVE":
        for key in obj.keys():
            if key.startswith("_pin_"):
                name = key[5:]
                yield name, _hash_indices(_get_curve_pin_indices(obj, name))


def resolve_vg_name(obj: bpy.types.Object, stored_name: str, stored_hash: int) -> str:
    """Resolve the current vertex group name.

    If ``stored_name`` still exists and its hash matches, return it.
    Otherwise scan all VGs/pins on the object for a hash match (renamed).
    Returns the current name, or ``stored_name`` if no match found.
    """
    if not obj:
        return stored_name
    # Quick check: stored name still valid
    if compute_vg_hash(obj, stored_name) == stored_hash:
        return stored_name
    # Scan all candidates for a hash match (renamed)
    for name, h in _iter_vg_candidates(obj):
        if h == stored_hash:
            return name
    return stored_name


# ---------------------------------------------------------------------------
# Depsgraph handler for rename detection
# ---------------------------------------------------------------------------

_prev_names: dict[str, str] = {}  # uuid -> last known name
_duplicate_detected: list[bool] = [False]  # set by handler, consumed after loop


@bpy.app.handlers.persistent
def _on_depsgraph_update(scene, depsgraph):
    """Detect object renames and keep display names in sync.

    Also strips duplicate UUIDs from copied objects immediately.
    Duplicate detection scans by UUID (never by name) — names are
    unstable across renames and can collide independently of identity.

    The UUID index is built once per invocation — O(M) — so that
    per-update duplicate checks are O(1) instead of O(M), giving
    O(N+M) total instead of the previous O(N*M).
    """
    # Fast path: skip the O(M) index build when no object was updated.
    has_object_update = False
    for update in depsgraph.updates:
        if isinstance(update.id, bpy.types.Object):
            has_object_update = True
            break
    if not has_object_update:
        return

    # Build a UUID -> owner index once: O(M).
    # When multiple objects share a UUID (duplication copies custom
    # props), iteration order cannot be trusted to pick the original —
    # Blender's internal ID list may place the copy first. Resolve
    # collisions via _prev_names[uid], which records the last-known
    # name of the object that held this UUID before the duplicate
    # appeared. Without this, a copy iterated first would become the
    # "owner", and its rename-vs-prev_name mismatch would fire
    # _sync_all_names, swapping the group's AssignedObject name from
    # the original to the duplicate.
    by_uuid: dict[str, list[bpy.types.Object]] = {}
    for obj in bpy.data.objects:
        uid = obj.get(_OBJ_UUID_KEY)
        if uid:
            by_uuid.setdefault(str(uid), []).append(obj)

    uuid_index: dict[str, bpy.types.Object] = {}
    for uid, objs in by_uuid.items():
        if len(objs) == 1:
            uuid_index[uid] = objs[0]
            continue
        prev_name = _prev_names.get(uid)
        owner = None
        if prev_name:
            for o in objs:
                if o.name == prev_name:
                    owner = o
                    break
        if owner is None:
            # No memory of who held this UUID (first tick after load,
            # or the original was renamed in the same tick it was
            # duplicated). Blender appends ".NNN" to copies, so the
            # shortest name is the most likely original.
            owner = min(objs, key=lambda o: (len(o.name), o.name))
        uuid_index[uid] = owner
        # Strip UUID from every non-owner now — the copy may not
        # appear in depsgraph.updates (only the new object's update
        # fires for a duplicate; the original is unchanged), so the
        # per-update loop alone would miss it.
        for o in objs:
            if o is owner:
                continue
            if _is_writable(o):
                del o[_OBJ_UUID_KEY]
                _duplicate_detected[0] = True

    # Prune _prev_names for deleted objects.  The index already contains
    # every live UUID, so anything absent is stale.
    stale = [u for u in _prev_names if u not in uuid_index]
    for u in stale:
        del _prev_names[u]
        _cache.pop(u, None)

    renamed = False
    for update in depsgraph.updates:
        if not isinstance(update.id, bpy.types.Object):
            continue
        obj = update.id
        uid = obj.get(_OBJ_UUID_KEY)
        if not uid:
            continue
        uid = str(uid)
        # Duplicate check: O(1) lookup.  uuid_index's owner was
        # already resolved above via _prev_names; anything else is a
        # copy whose UUID the collision pass should have stripped.
        # This branch catches the residual case where _is_writable
        # refused the strip (library-linked).
        owner = uuid_index.get(uid)
        if owner is not None and owner is not obj:
            if _is_writable(obj):
                del obj[_OBJ_UUID_KEY]
                _duplicate_detected[0] = True
            continue
        # Genuine rename or first encounter
        prev = _prev_names.get(uid)
        if prev != obj.name:
            _cache[uid] = obj.name
            _sync_all_names(uid, obj.name)
            if prev is not None:
                renamed = True
        _prev_names[uid] = obj.name

    if _duplicate_detected[0]:
        _duplicate_detected[0] = False
        try:
            from ..ui.dynamics.overlay import apply_object_overlays
            apply_object_overlays()
        except Exception:
            pass
    elif renamed:
        # A pure rename updated name fields in group-membership /
        # pin-list / merge-pair PropertyGroups. Those panels otherwise
        # stay stuck on the old name until the next unrelated redraw.
        try:
            for window in bpy.context.window_manager.windows:
                for area in window.screen.areas:
                    area.tag_redraw()
        except Exception:
            pass


def register():
    """Register the rename detection handler."""
    handlers = bpy.app.handlers.depsgraph_update_post
    # Drop any stale copies left by a previous reload (function identity
    # changes across reloads, so `in` alone wouldn't find them).
    for h in list(handlers):
        if getattr(h, "__name__", "") == "_on_depsgraph_update":
            handlers.remove(h)
    handlers.append(_on_depsgraph_update)


def unregister():
    """Unregister the rename detection handler."""
    handlers = bpy.app.handlers.depsgraph_update_post
    for h in list(handlers):
        if getattr(h, "__name__", "") == "_on_depsgraph_update":
            handlers.remove(h)
    _prev_names.clear()
    _cache.clear()
