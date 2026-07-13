# File: material_presets.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Bundled material-preset library loaded from ``presets/materials.toml``.
# Presets set physically-grounded material parameters on a dynamics group
# with one click. The preset menu is FILTERED by the group's current Type
# (object_type below), and applying a preset never changes the Type.
#
# Adding a material is a TOML edit only; this module needs no change. The
# whole addon tree is auto-bundled by Blender extensions, so the new
# ``presets/`` directory requires no blender_manifest.toml entry.

import os
import tomllib

from bpy.app.translations import pgettext_iface as iface_, pgettext_tip as tip_

from .param_introspect import MATERIAL_CLIPBOARD_EXCLUDE
from .profile import _VECTOR_PROPERTIES

# <addon>/presets/materials.toml, resolved relative to this file
# (<addon>/core/material_presets.py -> <addon>/presets/materials.toml).
_PRESETS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "presets", "materials.toml"
)

# Parsed-once cache; the bundled file never changes at runtime.
_cache = None

# Keys a preset table may carry that are NOT applied to the group:
#  - ``description`` is the dropdown tooltip;
#  - everything in MATERIAL_CLIPBOARD_EXCLUDE is identity (name/uuid/index),
#    classification (object_type, whose skip enforces D2: apply never changes
#    Type), profile bindings, and UI toggles. Reusing that set keeps a stray or
#    typo'd preset key from silently overwriting group identity or UI state,
#    matching how copy/paste and material profiles already protect those fields.
# None of these are real material params, so excluding them never drops a value
# a preset legitimately ships.
_SKIP_KEYS = MATERIAL_CLIPBOARD_EXCLUDE | {"description"}


def load_material_presets() -> dict:
    """Parse and cache ``presets/materials.toml``.

    Returns a dict mapping preset display name -> preset table. Returns an
    empty dict on any error (missing/unparsable file) so the UI degrades to
    "no presets" rather than raising during a panel draw.
    """
    global _cache
    if _cache is None:
        try:
            with open(_PRESETS_PATH, "rb") as f:
                data = tomllib.load(f)
            # Keep only table entries (one [Table] per preset). A stray
            # top-level scalar (e.g. a future ``version = 1`` line) would
            # otherwise be iterated as a preset and make ``.get()`` raise on a
            # non-dict during a panel draw, breaking the graceful-degrade
            # contract this function promises.
            _cache = {k: v for k, v in data.items() if isinstance(v, dict)}
        except Exception:
            _cache = {}
    return _cache


def get_preset_items(object_type: str) -> list:
    """EnumProperty items for presets matching *object_type*.

    Returns ``[("NONE", "Select Preset...", ""), (name, name, description), ...]``
    in file order. The caller (the EnumProperty ``items=`` callback) MUST stash
    the returned list in a module-level reference, or Blender frees the strings
    and the dropdown renders garbage (see ``ui/object_group.py``).
    """
    items = [("NONE", iface_("Select Preset..."), "")]
    for name, preset in load_material_presets().items():
        if preset.get("object_type") == object_type:
            items.append((name, name, preset.get("description", "")))
    return items


def has_presets_for(object_type: str) -> bool:
    """True if any bundled preset targets *object_type* (drives UI row visibility)."""
    return any(
        preset.get("object_type") == object_type
        for preset in load_material_presets().values()
    )


def apply_material_preset(name: str, group) -> bool:
    """Apply the named preset's parameters onto *group*.

    Iterates the preset's own keys (NOT ``profile._MATERIAL_PARAM_FIELDS``,
    which omits fields such as the ``sand_*`` group, so reusing it would
    silently drop them). Keys in ``_SKIP_KEYS`` (the description
    tooltip plus identity/classification/UI fields from
    ``MATERIAL_CLIPBOARD_EXCLUDE``) are never applied; skipping ``object_type``
    there is what enforces D2 (apply never changes the group's Type), and
    skipping the rest stops a stray key from overwriting group identity or UI
    state. Every remaining key is validated against the group's RNA before
    ``setattr`` so a typo'd TOML key is dropped rather than raising. Vector
    fields are coerced to tuples like ``profile.apply_material_profile``.

    Returns True if the preset exists, False otherwise.
    """
    preset = load_material_presets().get(name)
    if not preset:
        return False
    rna_props = group.bl_rna.properties
    for key, value in preset.items():
        if key in _SKIP_KEYS:
            continue
        if key not in rna_props:
            continue
        if key in _VECTOR_PROPERTIES and isinstance(value, list):
            value = tuple(value)
        try:
            setattr(group, key, value)
        except Exception:
            # Range/enum guard, mirroring copy_scalar_props: skip a single
            # field rather than abort the whole apply.
            continue
    return True
