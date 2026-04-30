# File: param_introspect.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Runtime introspection helpers for parameter copy/paste. Walks
# ``bl_rna.properties`` so clipboard operations pick up newly added
# PropertyGroup fields automatically — no hardcoded per-param list.

# RNA scalar types we can round-trip with plain ``setattr``. POINTER
# (nested PropertyGroup) and COLLECTION need structural handling, so
# the walker skips them — callers deal with collections explicitly.
_SCALAR_RNA_TYPES = frozenset({"BOOLEAN", "INT", "FLOAT", "STRING", "ENUM"})

# ObjectGroup fields that must NEVER be copied: identity (uuid/name/
# index), TOML profile binding, UI toggles, dynamic enums whose items
# depend on per-group context, computed caches, and per-assigned-
# object collections. Anything not in this set is treated as a
# material parameter and flows through copy/paste automatically.
MATERIAL_CLIPBOARD_EXCLUDE = frozenset({
    "name",
    "uuid",
    "index",
    "active",
    "material_profile_path",
    "material_profile_selection",
    "pin_profile_path",
    "pin_profile_selection",
    "velocity_object_selection",
    "collision_window_object_selection",
    "preview_velocity",
    "computed_contact_gap",
    "computed_contact_offset",
    "show_parameters",
    "show_overlay_color",
    "show_stats",
    "show_pin",
    "show_pin_overlay",
    "pin_overlay_size",
    "show_group",
    "pin_vertex_group_items",
})

# PinOperation ``show_*`` flags are viewport-only preview toggles; the
# remaining fields are the operation's semantic payload.
PIN_OP_CLIPBOARD_EXCLUDE = frozenset({
    "show_overlay",
    "show_vertex_spin",
    "show_max_towards_spin",
    "show_vertex_scale",
    "show_max_towards_scale",
})


def list_copyable_params(pg_instance_or_class, exclude=frozenset()):
    """Yield property identifiers on *pg_instance_or_class* that can be
    shuttled via ``setattr`` — scalar RNA types, minus anything in
    *exclude*. POINTER/COLLECTION properties are skipped because they
    need structural handling that plain attribute copy can't do.
    """
    bl_rna = getattr(pg_instance_or_class, "bl_rna", None)
    if bl_rna is None:
        return
    for prop in bl_rna.properties:
        name = prop.identifier
        if name in exclude:
            continue
        if prop.type not in _SCALAR_RNA_TYPES:
            continue
        if getattr(prop, "is_readonly", False):
            continue
        yield name


def material_param_applies(name: str, object_type: str) -> bool:
    """Paste-time filter: skip model-specific params whose prefix
    doesn't match the source's object_type so a SOLID→SHELL paste
    doesn't overwrite the shell_* side with SOLID's unused defaults.
    Shared params (no solid_/shell_/rod_ prefix) always apply.
    """
    if name.startswith("solid_"):
        return object_type == "SOLID"
    if name.startswith("shell_"):
        return object_type == "SHELL"
    if name.startswith("rod_"):
        return object_type == "ROD"
    return True


def copy_scalar_props(src, dst, exclude=frozenset(), filter_fn=None):
    """Copy all scalar RNA props from *src* to *dst*. *filter_fn*, if
    given, is called with the property name and must return True for
    the prop to be copied. Float vectors (``array_length > 0``) are
    read as a tuple and assigned back as a tuple so Blender accepts
    them on both sides.
    """
    bl_rna = getattr(src, "bl_rna", None)
    if bl_rna is None:
        return
    for prop in bl_rna.properties:
        name = prop.identifier
        if name in exclude:
            continue
        if prop.type not in _SCALAR_RNA_TYPES:
            continue
        if getattr(prop, "is_readonly", False):
            continue
        if filter_fn is not None and not filter_fn(name):
            continue
        try:
            value = getattr(src, name)
            if prop.type == "FLOAT" and getattr(prop, "array_length", 0) > 0:
                value = tuple(value)
            setattr(dst, name, value)
        except Exception:
            # Some enums may have dynamic items that don't contain the
            # source's current value on the destination; skip rather
            # than abort the whole copy.
            continue
