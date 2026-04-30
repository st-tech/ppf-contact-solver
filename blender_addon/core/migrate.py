# File: core/migrate.py
# License: Apache v2.0
#
# UUID migration for old .blend files.
# Assigns UUIDs to all name-based references so the UUID system works.
# Triggered manually via the "Run UUID Migration" button in Debug Options.

import json

import bpy  # pyright: ignore

from .uuid_registry import compute_vg_hash, get_or_create_object_uuid


def needs_migration() -> str:
    """Check if UUID migration is needed without modifying anything.

    Returns an error message if migration is needed, or empty string if OK.
    """
    if not hasattr(bpy.context, "scene") or not bpy.context.scene:
        return ""
    from ..models.groups import (
        iterate_active_object_groups,
        get_addon_data,
    )

    scene = bpy.context.scene
    counts = {"groups": 0, "objects": 0, "pins": 0, "pairs": 0, "keyframes": 0}

    for group in iterate_active_object_groups(scene):
        if not group.uuid:
            counts["groups"] += 1
        for assigned in group.assigned_objects:
            if assigned.name and not assigned.uuid:
                counts["objects"] += 1
        for pin_item in group.pin_vertex_groups:
            if pin_item.name and (not pin_item.object_uuid or not pin_item.vg_hash):
                counts["pins"] += 1

    root = get_addon_data(scene)
    if hasattr(root, "state"):
        state = root.state
        for pair in state.merge_pairs:
            if (pair.object_a and not pair.object_a_uuid) or \
               (pair.object_b and not pair.object_b_uuid):
                counts["pairs"] += 1
            elif pair.cross_stitch_json:
                try:
                    cs = json.loads(pair.cross_stitch_json)
                    if "source_name" in cs or "target_name" in cs:
                        counts["pairs"] += 1
                except (json.JSONDecodeError, ValueError):
                    pass
        from .uuid_registry import get_object_by_uuid
        orphaned_keyframe_idx = []
        for i, grp_entry in enumerate(state.saved_pin_keyframes):
            # Drop orphaned entries whose UUID no longer resolves — likely
            # a leftover from a deleted/renamed object in a previous file.
            if grp_entry.object_uuid and not get_object_by_uuid(grp_entry.object_uuid):
                orphaned_keyframe_idx.append(i)
                continue
            # Legacy shapes: no UUID, or UUID set but vg_hash missing.
            if grp_entry.object_name and (
                not grp_entry.object_uuid or not grp_entry.vg_hash
            ):
                counts["keyframes"] += 1
        for i in reversed(orphaned_keyframe_idx):
            state.saved_pin_keyframes.remove(i)

    total = sum(counts.values())
    if total == 0:
        return ""
    parts = [f"{v} {k}" for k, v in counts.items() if v > 0]
    return f"UUID migration needed ({', '.join(parts)}). Run UUID Migration first."


def migrate_legacy_data() -> str:
    """Migrate old name-based data to UUID-based.

    Assigns UUIDs to:
    - AssignedObjects (group membership)
    - PinVertexGroupItems (pin references)
    - MergePairItems (snap/stitch pairs)
    - Cross-stitch JSON embedded in merge pairs
    - SavedPinGroups (keyframe data)

    Returns a summary string.
    """
    try:
        if not hasattr(bpy.context, "scene") or not bpy.context.scene:
            return ""
        from ..models.groups import (
            iterate_active_object_groups,
            get_addon_data,
            decode_vertex_group_identifier,
        )

        scene = bpy.context.scene

        migrated_objects = []
        migrated_pins = []
        migrated_pairs = []
        migrated_keyframes = []
        migrated_groups = []

        for group in iterate_active_object_groups(scene):
            # Group-level UUID (safe to write here — operator context)
            if not group.uuid:
                group.ensure_uuid()
                migrated_groups.append(group.name or "unnamed")

            # Assigned objects
            for assigned in group.assigned_objects:
                if assigned.name and not assigned.uuid:
                    obj = bpy.data.objects.get(assigned.name)
                    if obj:
                        assigned.uuid = get_or_create_object_uuid(obj)
                        migrated_objects.append(assigned.name)

            # Pin vertex groups
            for pin_item in group.pin_vertex_groups:
                obj_name, vg_name = decode_vertex_group_identifier(pin_item.name) if pin_item.name else (None, None)
                if not obj_name:
                    continue
                obj = bpy.data.objects.get(obj_name)
                if not obj:
                    continue
                if not pin_item.object_uuid:
                    pin_item.object_uuid = get_or_create_object_uuid(obj)
                    migrated_pins.append(f"{obj_name}:{vg_name}")
                if not pin_item.vg_hash and vg_name:
                    h = compute_vg_hash(obj, vg_name)
                    if h:
                        pin_item.vg_hash = str(h)
                        migrated_pins.append(f"{obj_name}:{vg_name}:hash")

        root = get_addon_data(scene)
        if hasattr(root, "state"):
            state = root.state

            # Merge pairs
            for pair in state.merge_pairs:
                if pair.object_a and not pair.object_a_uuid:
                    obj = bpy.data.objects.get(pair.object_a)
                    if obj:
                        pair.object_a_uuid = get_or_create_object_uuid(obj)
                        migrated_pairs.append(pair.object_a)
                if pair.object_b and not pair.object_b_uuid:
                    obj = bpy.data.objects.get(pair.object_b)
                    if obj:
                        pair.object_b_uuid = get_or_create_object_uuid(obj)
                        migrated_pairs.append(pair.object_b)

                # Inject UUIDs into cross_stitch_json and strip legacy name keys.
                # Run unconditionally (even if one UUID is empty) so stale
                # name keys never persist in the JSON.
                if pair.cross_stitch_json:
                    try:
                        cs = json.loads(pair.cross_stitch_json)
                        changed = False
                        if not cs.get("source_uuid") and pair.object_a_uuid:
                            sn = cs.get("source_name", "")
                            if sn == pair.object_a:
                                cs["source_uuid"] = pair.object_a_uuid
                                changed = True
                            elif sn == pair.object_b:
                                cs["source_uuid"] = pair.object_b_uuid
                                changed = True
                            # else: name was changed before migration — we
                            # cannot reliably guess which object is the
                            # source.  Leave source_uuid empty so the
                            # encoder skips this stitch instead of binding
                            # to the wrong object.
                        if not cs.get("target_uuid") and pair.object_b_uuid:
                            tn = cs.get("target_name", "")
                            if tn == pair.object_a:
                                cs["target_uuid"] = pair.object_a_uuid
                                changed = True
                            elif tn == pair.object_b:
                                cs["target_uuid"] = pair.object_b_uuid
                                changed = True
                            # else: same — cannot determine target after
                            # rename.  Leave target_uuid empty.
                        # Strip name keys unconditionally — they're no longer
                        # used for lookup anywhere.
                        if "source_name" in cs:
                            del cs["source_name"]
                            changed = True
                        if "target_name" in cs:
                            del cs["target_name"]
                            changed = True
                        if changed:
                            pair.cross_stitch_json = json.dumps(
                                cs, separators=(",", ":")
                            )
                            migrated_pairs.append("cross_stitch_json")
                    except (json.JSONDecodeError, Exception):
                        pass

            # Saved pin keyframes — populate object_uuid and vg_hash so the
            # rename/hash resolution machinery works for restored data too.
            for grp_entry in state.saved_pin_keyframes:
                if not grp_entry.object_name:
                    continue
                obj = bpy.data.objects.get(grp_entry.object_name)
                if not obj:
                    continue
                if not grp_entry.object_uuid:
                    grp_entry.object_uuid = get_or_create_object_uuid(obj)
                    migrated_keyframes.append(grp_entry.object_name)
                if not grp_entry.vg_hash and grp_entry.vertex_group:
                    h = compute_vg_hash(obj, grp_entry.vertex_group)
                    if h:
                        grp_entry.vg_hash = str(h)
                        migrated_keyframes.append(
                            f"{grp_entry.object_name}:{grp_entry.vertex_group}:hash"
                        )

        total = (
            len(migrated_groups)
            + len(migrated_objects)
            + len(migrated_pins)
            + len(migrated_pairs)
            + len(migrated_keyframes)
        )
        if total > 0:
            parts = []
            if migrated_groups:
                parts.append(f"groups: {', '.join(migrated_groups)}")
            if migrated_objects:
                parts.append(f"objects: {', '.join(migrated_objects)}")
            if migrated_pins:
                parts.append(f"pins: {', '.join(migrated_pins)}")
            if migrated_pairs:
                parts.append(f"pairs: {', '.join(migrated_pairs)}")
            if migrated_keyframes:
                parts.append(f"keyframes: {', '.join(migrated_keyframes)}")
            return f"Migrated {total} items ({'; '.join(parts)}). Save to persist."
        return "No migration needed — all items already have UUIDs."

    except Exception as e:
        return f"Migration error: {e}"
