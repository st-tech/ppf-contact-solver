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

A RETIRED ENUM VALUE IS THE SAME PROBLEM WEARING A NUMBER. Blender stores an
EnumProperty's ITEM NUMBER, so a value whose item is gone leaves the field
reading its default, and for a connection that means a file opens pointed at
somewhere other than where its solver is. ``migrate_retired_connection`` is that
case for the retired Local connection type.

Unlike ``migrate.py`` (the one-shot UUID cutover, which is slated for deletion),
this module is permanent: it is where any future property rename registers its
legacy key.
"""

import sys

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

# The `server_type` item number the retired Local connection was saved under,
# and the field that held its directory. `ui/state.py` says why the number can
# never be reused.
_RETIRED_LOCAL_ID = 0
_RETIRED_LOCAL_PATH_KEY = "local_path"

# The native connection this platform's Local connections become, and the field
# that takes the directory. Local reached a server on THIS machine, and so does
# the native, so the platform is the whole mapping.
_PLATFORM_NATIVE = {
    "win32": ("WIN_NATIVE", "win_native_path"),
    "darwin": ("MAC_NATIVE", "mac_native_path"),
}
_DEFAULT_NATIVE = ("LINUX_NATIVE", "linux_native_path")


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


def migrate_retired_connection(scene=None) -> str:
    """Move a scene saved with the retired Local connection onto this platform's native.

    WHAT IS BROKEN WITHOUT IT. The `.blend` carries `server_type` as the ITEM
    NUMBER Local occupied, and no item carries that number any more, so the
    field reads its default (SSH) and the saved directory has no field left to
    live in. The artist opens a scene that was connecting to a solver on this
    machine and finds it pointed at a host it was never given, with the path
    they chose gone from the file.

    The native connection is where Local's case went: it reaches a server on
    THIS machine on the same port, and also knows how to start it, which build
    directory it came out of, and which device it runs on. So the directory
    carries over unchanged and only the type is different.

    A native path already set is NOT overwritten. A scene that carries both was
    connected some other way since, and that later answer is the one to keep.

    Returns a summary of what moved, or an empty string when there was nothing
    to do (the common case: a scene saved by a current build).
    """
    scene = scene or getattr(bpy.context, "scene", None)
    if scene is None:
        return ""
    root = get_addon_data(scene)
    if root is None:
        return ""
    props = root.ssh_state

    # `in props.keys()` reads the raw stored value rather than the field, which
    # is the only way to see a number whose item is gone: reading the field
    # would report the default and say nothing about what the file holds.
    if "server_type" not in props.keys():
        return ""
    if props["server_type"] != _RETIRED_LOCAL_ID:
        return ""

    native, path_field = _PLATFORM_NATIVE.get(sys.platform, _DEFAULT_NATIVE)
    moved = [f"server_type=Local -> {native}"]

    saved_path = props.get(_RETIRED_LOCAL_PATH_KEY, "")
    if saved_path and not getattr(props, path_field, ""):
        setattr(props, path_field, saved_path)
        moved.append(f"{_RETIRED_LOCAL_PATH_KEY} -> {path_field}")
        del props[_RETIRED_LOCAL_PATH_KEY]
    # A KEY THAT DID NOT LAND IS LEFT IN PLACE, never dropped, which is the rule
    # the rename migration above states: the value is then still recoverable
    # from the file. It does not land when the native path already holds one,
    # and that directory is the later answer of the two.

    # Written last, so a failure above leaves the file still saying Local and
    # the migration runs again on the next open rather than stranding the path.
    props.server_type = native
    return "; ".join(moved)
