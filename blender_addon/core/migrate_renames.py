# File: core/migrate_renames.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Carry renamed scene properties forward when an older .blend is opened.

A PropertyGroup field is stored in the .blend under its identifier, so renaming
one strands the saved value: the file still carries the old key as a raw
ID-property, while the new field reads its default. For a flag that selects the
simulation's time base, that is a silent change to the file's physics, which is
exactly the kind of quiet wrong answer this project refuses. Each entry below
moves the stored value onto the new field and drops the stale key, so opening
and re-saving an old scene preserves what it was set to.

Unlike ``migrate.py`` (the one-shot UUID cutover, which is slated for deletion),
this module is permanent: it is where any future property rename registers its
legacy key.
"""

import bpy  # pyright: ignore

from ..models.groups import get_addon_data

# Legacy ID-property key -> current State field.
#
# `use_frame_rate_in_output` was renamed because it read as the opposite of what
# it did: True meant "ignore the FPS field and take the Blender scene's frame
# rate". The value carries over unchanged; only the name is honest now.
_STATE_RENAMES = {
    "use_frame_rate_in_output": "use_scene_fps",
}


def migrate_renamed_state_props(scene=None) -> str:
    """Move any legacy state keys in `scene` onto their current fields.

    Returns a summary of what moved, or an empty string when there was nothing
    to do (the common case: a scene saved by a current build).
    """
    scene = scene or getattr(bpy.context, "scene", None)
    if scene is None:
        return ""
    root = get_addon_data(scene)
    if root is None:
        return ""
    state = root.state

    moved = []
    for old_key, new_field in _STATE_RENAMES.items():
        # `in state.keys()` inspects the raw ID-properties the .blend carries,
        # which is where a field of a since-renamed identifier survives. A field
        # never written to has no ID-property at all, and its absence correctly
        # means "was the default".
        if old_key not in state.keys():
            continue
        value = state[old_key]
        try:
            setattr(state, new_field, bool(value))
        except (TypeError, ValueError):
            # A key we cannot land on its field is left in place rather than
            # dropped, so the value is still recoverable from the file.
            continue
        del state[old_key]
        moved.append(f"{old_key}={bool(value)} -> {new_field}")

    return "; ".join(moved)
