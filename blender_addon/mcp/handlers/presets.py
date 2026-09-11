# File: handlers/presets.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""MCP handlers for material presets, parameter profiles, and the clipboards.

A PRESET is bundled with the addon and read only. ``presets/materials.toml``
holds one table per material, carrying physically grounded parameters for a
fabric or a soft solid, plus an ``object_type`` that decides which groups may
take it and a ``description`` for the dropdown. Applying one writes material
parameters onto a group and nothing else: it never changes the group's Type,
and it never overwrites a parameter the artist has locked.

A PROFILE is the artist's own TOML file, written by the addon from the current
settings and applied back later. There are four kinds, and each binds a file
path plus a selected entry onto a different PropertyGroup:

- SCENE reads the solver parameters, the dynamic parameter schedules and the
  invisible colliders off the scene state.
- MATERIAL reads one dynamics group's material parameters.
- PIN reads the operations of one pin.
- CONNECTION reads the solver host connection settings.

They differ only in what they read and where the binding lives, so one set of
tools takes the kind as an argument. The file format is shared: a TOML table
per entry, several entries per file.

Removing an entry from a profile file has no path in the addon, so no tool
here does it; ``clear_profile_path`` unbinds the file from the scene, which is
what the panel's Clear button does, and leaves the file untouched.
"""

import math
import os

import bpy  # pyright: ignore

from ...core.material_presets import (
    # The same set the apply skips, imported so this module's report of what a
    # preset writes cannot disagree with what it actually wrote.
    _SKIP_KEYS,
    load_material_presets,
)
from ...core.material_presets import (
    apply_material_preset as apply_preset_to_group,
)
from ...core.param_introspect import (
    MATERIAL_CLIPBOARD_EXCLUDE,
    list_copyable_params,
    material_param_applies,
)
from ...core.profile import (
    _LEGACY_SCENE_PARAM_KEYS,
    _MATERIAL_PARAM_FIELDS,
    _SCENE_PARAM_FIELDS,
    _SSH_STATE_FIELDS,
    PROFILE_TYPE_MAP,
    apply_material_profile,
    apply_pin_operations,
    apply_scene_profile,
    load_profiles,
    read_connection_profile,
    read_material_profile,
    read_pin_operations,
    read_scene_profile,
    save_profile_entry,
)
from ...core.profile import (
    apply_profile as apply_connection_profile,
)
from ...core.uuid_registry import get_object_uuid
from ...models.groups import (
    decode_vertex_group_identifier,
    get_addon_data,
    parse_pin_identifier,
)
from ...models.material_locks import locked_props
from ..decorators import MCPError, group_handler, mcp_handler
from .group import get_active_group_by_uuid_helper, get_group_index_by_uuid

# The bundled library, named in every refusal that involves it so a caller
# knows which file to look at.
_PRESET_FILE = "presets/materials.toml"

# Each profile kind: where its binding lives, which operator clears it, what
# it reads, and which entry keys its apply function understands.
#
# ``recognized`` is built from the field maps in core/profile.py rather than
# restated, so a key this module reports as ignored really is one the apply
# drops. The extra keys are the structures an apply handles outside its map:
# a scene entry's dynamic parameters and invisible colliders, a material
# entry's embedded pin operations, and a connection entry's server type.
#
# ``fields`` is the same map unreduced, TOML key to property name, and holds
# exactly the keys an apply writes with a plain assignment. A load reads each
# of them back off the property it names, so the property name is what it
# needs. The structures listed above are absent from it because an apply
# rebuilds them rather than assigning them, and a PIN entry's operations are
# the whole of that kind, which is why its map is empty.
_PROFILE_KINDS = {
    "SCENE": {
        "path_prop": "scene_profile_path",
        "selection_prop": "scene_profile_selection",
        "clear_operator": "scene.clear_scene_profile",
        "needs_group": False,
        "needs_pin": False,
        "contents": (
            "the scene solver parameters, the dynamic parameter schedules "
            "and the invisible colliders"
        ),
        "recognized": (
            frozenset(_SCENE_PARAM_FIELDS)
            | frozenset(_LEGACY_SCENE_PARAM_KEYS)
            | frozenset({"dyn_params", "colliders"})
        ),
        "fields": _SCENE_PARAM_FIELDS,
    },
    "MATERIAL": {
        "path_prop": "material_profile_path",
        "selection_prop": "material_profile_selection",
        "clear_operator": "object.clear_material_profile",
        "needs_group": True,
        "needs_pin": False,
        "contents": "one dynamics group's material parameters",
        "recognized": frozenset(_MATERIAL_PARAM_FIELDS) | frozenset({"pins"}),
        "fields": _MATERIAL_PARAM_FIELDS,
    },
    "PIN": {
        "path_prop": "pin_profile_path",
        "selection_prop": "pin_profile_selection",
        "clear_operator": "object.clear_pin_profile",
        "needs_group": True,
        "needs_pin": True,
        "contents": "the operations of one pin",
        "recognized": frozenset({"operations"}),
        "fields": {},
    },
    "CONNECTION": {
        "path_prop": "profile_path",
        "selection_prop": "profile_selection",
        "clear_operator": "ssh.clear_profile",
        "needs_group": False,
        "needs_pin": False,
        "contents": "the solver host connection settings",
        "recognized": frozenset(_SSH_STATE_FIELDS) | frozenset({"type"}),
        "fields": _SSH_STATE_FIELDS,
    },
}

# The dropdowns use this identifier for "no profile selected", so an entry
# carrying it as a name could not be selected once saved.
_NO_SELECTION = "NONE"


# ---------------------------------------------------------------------------
# Scene and RNA access
# ---------------------------------------------------------------------------


def _scene():
    """The active scene, or a refusal."""
    scene = bpy.context.scene
    if scene is None:
        raise MCPError("No active Blender scene")
    return scene


def _object_group_properties():
    """The RNA properties of a dynamics group.

    Read off the registered class, so what a preset key is checked against is
    what Blender will accept when the value is written.
    """
    from ...ui.object_group import ObjectGroup

    return ObjectGroup.bl_rna.properties


def _plain(value):
    """An RNA value as JSON can carry it.

    A vector property reads back as a Blender array rather than a list, and a
    result carrying one cannot be serialized.
    """
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if hasattr(value, "__len__"):
        return [_plain(item) for item in value]
    return value


def _matches(current, wanted) -> bool:
    """Whether a property now holds the value a preset asked for.

    Float properties are 32 bit, so a decimal from the TOML file is stored as
    the nearest float32 and an exact comparison would report every float as
    refused.
    """
    if isinstance(wanted, list | tuple):
        current = list(current) if hasattr(current, "__len__") else [current]
        if len(current) != len(wanted):
            return False
        return all(
            _matches(now, asked)
            for now, asked in zip(current, wanted, strict=True)
        )
    if isinstance(wanted, bool) or isinstance(current, bool):
        return bool(current) is bool(wanted)
    if isinstance(wanted, int | float) and isinstance(current, int | float):
        return math.isclose(float(current), float(wanted), rel_tol=1e-6, abs_tol=1e-9)
    return current == wanted


def _describe_refused(refused: list[tuple[str, object, object]]) -> str:
    """Name each property that did not take a value, and both values.

    A property that will not take a value stores something else instead, a
    number clamped into its range or the value it already held, and reports
    nothing. What it holds now is the measurement a caller can act on, so it
    is named beside what was asked for.
    """
    return ", ".join(
        f"{name} asked for {asked!r} and holds {current!r}"
        for name, asked, current in sorted(refused, key=lambda item: item[0])
    )


def _run_operator(operator, name: str, **properties):
    """Run a UI operator and refuse unless it reports FINISHED.

    ``EXEC_DEFAULT`` skips the confirm and file select flow, which needs a
    window an MCP call does not have. An operator that refuses its own
    preconditions returns CANCELLED, and reporting that as success would hide
    a scene that was never changed.
    """
    result = operator("EXEC_DEFAULT", **properties)
    if "FINISHED" not in result:
        raise MCPError(
            f"Operator {name} returned {sorted(result)} rather than FINISHED, "
            "so nothing was changed."
        )
    return result


def _refresh_viewport(overlays: bool):
    """Redraw the panels, and rebuild the overlays when values behind them moved."""
    from ...core.utils import redraw_all_areas

    if overlays:
        from ...models.groups import invalidate_overlays

        invalidate_overlays()
    redraw_all_areas(bpy.context)


# ---------------------------------------------------------------------------
# Material presets
# ---------------------------------------------------------------------------


def _preset_library() -> dict:
    """Every bundled preset, or a refusal naming the file."""
    presets = load_material_presets()
    if not presets:
        raise MCPError(
            f"The bundled preset library {_PRESET_FILE} holds no preset. It "
            "ships with the addon, so an empty result means the file is "
            "missing from the installed extension or is not valid TOML."
        )
    return presets


def _object_type_identifiers() -> list[str]:
    """The group Type values, read from the RNA rather than restated here."""
    return [
        item.identifier
        for item in _object_group_properties()["object_type"].enum_items
    ]


def _preset_parameters(preset: dict) -> tuple[dict, list[str]]:
    """Split a preset table into the parameters it writes and its stray keys.

    ``object_type`` and ``description`` classify the preset, and the rest of
    the skip set is group identity and UI state, so none of them is a material
    parameter. A key that names no group property is written by nothing and is
    reported separately: the apply drops it in silence.
    """
    properties = _object_group_properties()
    parameters = {}
    unknown = []
    for key, value in preset.items():
        if key in _SKIP_KEYS:
            continue
        if key not in properties:
            unknown.append(key)
            continue
        parameters[key] = value
    return parameters, unknown


@mcp_handler
def list_material_presets(object_type: str | None = None):
    """List the bundled material presets and the parameters each one writes.

    A preset carries an object_type that decides which groups may take it: a
    SHELL group is offered the fabrics and a SOLID group the soft solids, and
    apply_material_preset refuses a mismatch. A Type the library ships no
    preset for gives an empty list rather than an error, so an empty result is
    an answer and not a failure.

    'parameters' is what applying the preset writes, keyed by group property
    name. 'unknown_keys' names any key in the preset table that matches no
    group property; those are written by nothing, and a non-empty list is a
    defect in the bundled file rather than something a caller can act on.

    Args:
        object_type: Group Type to filter by, one of SOLID, SHELL, ROD,
            STATIC, PDRD, SAND. Omit to list every preset.
    """
    if object_type is not None:
        valid = _object_type_identifiers()
        if object_type not in valid:
            raise MCPError(
                f"Unknown object_type '{object_type}'. Valid types are "
                f"{', '.join(valid)}."
            )
    presets = []
    for name, preset in _preset_library().items():
        preset_type = preset.get("object_type", "")
        if object_type is not None and preset_type != object_type:
            continue
        parameters, unknown = _preset_parameters(preset)
        presets.append(
            {
                "name": name,
                "object_type": preset_type,
                "description": preset.get("description", ""),
                "parameters": {key: _plain(v) for key, v in parameters.items()},
                "parameter_count": len(parameters),
                "unknown_keys": unknown,
            }
        )
    return {
        "presets": presets,
        "preset_count": len(presets),
        "object_type_filter": object_type,
        "source": _PRESET_FILE,
    }


@group_handler
def apply_material_preset(group_uuid: str, preset_name: str):
    """Write a bundled material preset's parameters onto a dynamics group.

    The preset's object_type has to match the group's Type. Applying one never
    changes the Type, so a fabric preset on a SOLID group would write shell
    parameters that group's elements never read, and it is refused instead.
    Use set_group_type first, or pick a preset for the Type the group has.

    A parameter the group has locked keeps its value, which is what the
    padlock beside it promises against the tools that overwrite a whole group
    at once. Locked parameters are reported under 'kept_locked'.

    'written' reports every parameter that now carries the preset's value,
    including any that already did. A parameter whose property did not take
    the preset's value leaves the group holding part of the preset and raises,
    naming the value asked for and the value the group now holds.

    Args:
        group_uuid: UUID of the group to write the preset onto.
        preset_name: Preset name as reported by list_material_presets.
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    library = _preset_library()
    preset = library.get(preset_name)
    if preset is None:
        available = [
            name
            for name, entry in library.items()
            if entry.get("object_type") == group.object_type
        ]
        offer = (
            f"Presets for a {group.object_type} group: {', '.join(available)}."
            if available
            else f"No bundled preset targets a {group.object_type} group."
        )
        raise MCPError(f"Unknown material preset '{preset_name}'. {offer}")

    preset_type = preset.get("object_type", "")
    if preset_type != group.object_type:
        raise MCPError(
            f"Preset '{preset_name}' targets a {preset_type} group, and group "
            f"'{group.name}' is {group.object_type}. Applying a preset never "
            "changes the Type: call set_group_type first, or pick a preset "
            "list_material_presets reports for this Type."
        )

    parameters, unknown = _preset_parameters(preset)
    locked = locked_props(group)
    if not apply_preset_to_group(preset_name, group):
        raise MCPError(
            f"The preset library no longer holds '{preset_name}'; it was read "
            "back as missing between the lookup and the apply."
        )

    written = {}
    kept_locked = []
    refused = []
    for key, value in parameters.items():
        if key in locked:
            kept_locked.append(key)
            continue
        current = getattr(group, key)
        if _matches(current, value):
            written[key] = _plain(current)
        else:
            refused.append((key, _plain(value), _plain(current)))

    _refresh_viewport(overlays=True)

    if refused:
        raise MCPError(
            f"Preset '{preset_name}' wrote {len(written)} of its "
            f"parameters onto group '{group.name}' and {len(refused)} did "
            "not take the value the preset names, so the group holds part "
            f"of the preset: {_describe_refused(refused)}. The property "
            f"behind each one did not accept the value {_PRESET_FILE} "
            "carries, which can be a value outside the range the property "
            "enforces, an enum identifier it does not offer, or a value of "
            "the wrong type. Nothing in the apply distinguishes the three, "
            "so this refusal does not pick one."
        )

    return {
        "message": (
            f"Applied preset '{preset_name}' to group '{group.name}', writing "
            f"{len(written)} parameters"
        ),
        "group_uuid": group_uuid,
        "preset_name": preset_name,
        "object_type": group.object_type,
        "written": written,
        "written_count": len(written),
        "kept_locked": sorted(kept_locked),
        "unknown_keys": unknown,
    }


# ---------------------------------------------------------------------------
# Profile resolution
# ---------------------------------------------------------------------------


def _check_kind(kind: str) -> str:
    """The profile kind as this module spells it, or a refusal."""
    normalized = kind.strip().upper()
    if normalized not in _PROFILE_KINDS:
        raise MCPError(
            f"Unknown profile kind '{kind}'. Valid kinds are "
            f"{', '.join(_PROFILE_KINDS)}."
        )
    return normalized


def _binder(kind: str) -> str:
    """What a refusal calls the thing this kind's file path is bound to.

    MATERIAL and PIN store their path on one dynamics group and the other two
    on the scene, so a message that says "the scene" for all four names the
    wrong place for two of them.
    """
    return "this group" if _PROFILE_KINDS[kind]["needs_group"] else "this scene"


def _resolve_owner(kind: str, group_uuid: str | None):
    """The PropertyGroup holding this kind's file path, and its group.

    MATERIAL and PIN bind their file to one dynamics group, so both need a
    group_uuid and the other two kinds refuse one: a group named on a SCENE or
    CONNECTION call is a caller mistake, and ignoring it would report success
    for a call that addressed something else.
    """
    spec = _PROFILE_KINDS[kind]
    if spec["needs_group"]:
        if not group_uuid:
            raise MCPError(
                f"A {kind} profile belongs to one dynamics group, so "
                "group_uuid is required. Call get_active_groups for the "
                "groups in the scene."
            )
        group = get_active_group_by_uuid_helper(group_uuid)
        return group, group
    if group_uuid:
        raise MCPError(
            f"A {kind} profile is not per group, so group_uuid is not "
            "accepted. Drop it, or pass kind MATERIAL or PIN."
        )
    addon_data = get_addon_data(_scene())
    if kind == "SCENE":
        return addon_data.state, None
    return addon_data.ssh_state, None


def _resolve_pin_item(group, vertex_group_identifier: str):
    """(pin item, row index) for a pin of `group`, matched by object UUID.

    Identity is the object's UUID plus the vertex group name, never the
    display name, so a renamed object still resolves to its own pin.
    """
    obj_name, vg_name = parse_pin_identifier(vertex_group_identifier, MCPError)
    obj = bpy.data.objects.get(obj_name)
    if obj is None:
        raise MCPError(f"Object '{obj_name}' not found in the scene")
    obj_uuid = get_object_uuid(obj)
    if not obj_uuid:
        raise MCPError(
            f"Object '{obj_name}' carries no UUID, so no pin can be matched "
            "to it"
        )
    for index, item in enumerate(group.pin_vertex_groups):
        if item.object_uuid != obj_uuid:
            continue
        _, item_vg = decode_vertex_group_identifier(item.name)
        if item_vg == vg_name:
            return item, index
    raise MCPError(
        f"Pin '{obj_name}::{vg_name}' not found in group '{group.name}'. Call "
        "list_pins for the pins it carries."
    )


def _resolve_pin_argument(kind: str, group, vertex_group_identifier: str | None):
    """The pin a PIN profile reads or writes, or (None, None) for the rest."""
    if not _PROFILE_KINDS[kind]["needs_pin"]:
        if vertex_group_identifier:
            raise MCPError(
                f"A {kind} profile does not address a pin, so "
                "vertex_group_identifier is not accepted. Drop it, or pass "
                "kind PIN."
            )
        return None, None
    if not vertex_group_identifier:
        raise MCPError(
            "A PIN profile holds the operations of one pin, so "
            "vertex_group_identifier is required, in "
            "'object_name::vertex_group_name' form. Call list_pins for the "
            "pins the group carries."
        )
    return _resolve_pin_item(group, vertex_group_identifier)


def _check_entry_name(name: str) -> str:
    """The entry name to write, or a refusal."""
    if not isinstance(name, str) or not name.strip():
        raise MCPError(
            "name must be a non-empty string naming the entry to write. Each "
            "profile file holds several entries, one TOML table per name."
        )
    stripped = name.strip()
    if stripped == _NO_SELECTION:
        raise MCPError(
            f"'{_NO_SELECTION}' is the identifier the profile dropdowns use "
            "for no selection, so an entry saved under that name could never "
            "be selected. Pick another name."
        )
    return stripped


def _resolve_path(kind: str, owner, path: str | None) -> tuple[str, str]:
    """(path as the binding stores it, path on disk) for this call.

    The addon stores the path the artist chose, which may be a '//' path
    relative to the .blend or carry a '~'. Both are resolved here, and a path
    that still is not absolute afterwards is refused: it would be read
    relative to whatever directory Blender was started in, which is not
    something a caller can predict.
    """
    spec = _PROFILE_KINDS[kind]
    stored = path if path else getattr(owner, spec["path_prop"])
    if not stored:
        raise MCPError(
            f"No {kind} profile file is bound to {_binder(kind)}, so there is "
            "nothing to read. Pass 'path' to name a file; saving to it binds "
            "it for later calls."
        )
    absolute = os.path.expanduser(bpy.path.abspath(stored))
    if not os.path.isabs(absolute):
        raise MCPError(
            f"'{stored}' does not resolve to an absolute path. A '//' path is "
            "relative to the .blend file and needs one that has been saved; "
            "otherwise pass an absolute path."
        )
    return stored, absolute


def _read_file(absolute: str) -> tuple[dict, list[str]]:
    """(entries, keys that are not entries) of a profile file, or a refusal.

    An entry is a TOML table. A scalar written at the top level of the file is
    not one, and is reported rather than offered as an entry that would fail
    on use.
    """
    if not os.path.isfile(absolute):
        raise MCPError(f"No profile file at {absolute}")
    raw = load_profiles(absolute)
    if not raw:
        raise MCPError(
            f"{absolute} yielded no profile entry. The addon reads a file it "
            "cannot parse as an empty one, so check that the file is valid "
            "TOML and holds at least one [entry] table."
        )
    entries = {key: value for key, value in raw.items() if isinstance(value, dict)}
    others = [key for key in raw if key not in entries]
    if not entries:
        raise MCPError(
            f"{absolute} holds no [entry] table; its top level carries only "
            f"{', '.join(others)}."
        )
    return entries, others


def _select_entry(owner, spec: dict, stored_path: str, name: str):
    """Point the panel's file path and dropdown at the entry just used.

    The path is set first: the dropdown reads its items out of the bound file,
    so a name set before the path would not be among them. Setting the
    selection re-applies the entry through the property's own update callback,
    which is the same work the caller just asked for.
    """
    setattr(owner, spec["path_prop"], stored_path)
    setattr(owner, spec["selection_prop"], name)


def _current_selection(owner, spec: dict) -> str | None:
    """The entry the panel has selected, or None when it has none."""
    selection = getattr(owner, spec["selection_prop"])
    return None if selection == _NO_SELECTION else selection


def _read_entry(kind: str, owner, group, pin_item) -> dict:
    """The current settings this kind of profile saves.

    A material entry carries the group's own parameters and no pins. The read
    that also embeds the operations of every pin in the group nests a table
    inside the entry, and the addon's TOML writer has no rule for a nested
    table: it emits the Python repr, which does not parse, so the whole file
    stops being readable. A PIN profile is how a pin's operations are saved.
    """
    if kind == "SCENE":
        return read_scene_profile(owner)
    if kind == "MATERIAL":
        return read_material_profile(group)
    if kind == "PIN":
        return read_pin_operations(pin_item)
    return read_connection_profile(owner)


def _assigned_values(kind: str, entry: dict) -> list[tuple[str, str, object]]:
    """(entry key, property name, value) for every key the apply assigns.

    A key outside this kind's field map addresses a structure the apply
    rebuilds rather than assigns, so it names no single property and is left
    out. A SCENE entry may spell a key by a retired name, which the apply
    reads under the current one when the current one is absent; the same
    renaming is done here, and the pair keeps the name the entry uses so a
    refusal names the key the caller would edit.
    """
    fields = _PROFILE_KINDS[kind]["fields"]
    retired = _LEGACY_SCENE_PARAM_KEYS if kind == "SCENE" else {}
    assigned = []
    for key, value in entry.items():
        current_key = key
        if key in retired and retired[key] not in entry:
            current_key = retired[key]
        prop_name = fields.get(current_key)
        if prop_name is not None:
            assigned.append((key, prop_name, value))
    return assigned


def _apply_target(kind: str, owner, group, pin_item):
    """The PropertyGroup this kind's field map is written onto."""
    if kind == "MATERIAL":
        return group
    if kind == "PIN":
        return pin_item
    return owner


def _apply_entry(kind: str, entry: dict, owner, group, pin_item):
    """Write a profile entry back onto the scene."""
    if kind == "SCENE":
        apply_scene_profile(entry, owner)
        return
    if kind == "MATERIAL":
        apply_material_profile(entry, group)
        return
    if kind == "PIN":
        apply_pin_operations(entry, pin_item)
        return
    if not apply_connection_profile(entry, owner):
        raise MCPError(
            "The entry names no server type this addon knows, so no "
            "connection setting was written. Its 'type' key has to be one of "
            f"{', '.join(PROFILE_TYPE_MAP)}."
        )


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


@mcp_handler
def list_profiles(
    kind: str,
    group_uuid: str | None = None,
    path: str | None = None,
):
    """List the entries of a profile file, for one of the four profile kinds.

    A profile file holds several entries, one TOML table per name, and the
    scene binds one file and one selected entry per kind. With no 'path' the
    bound file is read, and a kind with no file bound is refused rather than
    reported as empty.

    MATERIAL and PIN bind their file to a dynamics group, so both need
    group_uuid; SCENE and CONNECTION refuse one.

    'unrecognized_keys' names keys of an entry that this kind's apply drops,
    which is what an entry saved under a different kind looks like from here.

    Args:
        kind: SCENE, MATERIAL, PIN or CONNECTION.
        group_uuid: UUID of the group the file is bound to, for MATERIAL and
            PIN.
        path: Profile file to read instead of the bound one. Absolute, or
            '//' relative to a saved .blend.
    """
    kind = _check_kind(kind)
    spec = _PROFILE_KINDS[kind]
    owner, _group = _resolve_owner(kind, group_uuid)
    stored, absolute = _resolve_path(kind, owner, path)
    entries, non_entry_keys = _read_file(absolute)
    recognized = spec["recognized"]
    listed = []
    for name, entry in entries.items():
        unrecognized = sorted(key for key in entry if key not in recognized)
        listed.append(
            {
                "name": name,
                "key_count": len(entry),
                "recognized_key_count": len(entry) - len(unrecognized),
                "unrecognized_keys": unrecognized,
            }
        )
    return {
        "kind": kind,
        "path": stored,
        "absolute_path": absolute,
        "bound_path": getattr(owner, spec["path_prop"]),
        "selected_entry": _current_selection(owner, spec),
        "profiles": listed,
        "profile_count": len(listed),
        "non_entry_keys": non_entry_keys,
        "contents": spec["contents"],
    }


@mcp_handler
def save_profile(
    kind: str,
    name: str,
    group_uuid: str | None = None,
    vertex_group_identifier: str | None = None,
    path: str | None = None,
):
    """Save current settings as a named entry in a profile file.

    Each kind reads a different part of the scene: SCENE the solver
    parameters, the dynamic parameter schedules and the invisible colliders;
    MATERIAL one group's material parameters, and no pins; PIN the operations
    of one pin; CONNECTION the solver host settings. MATERIAL and PIN need
    group_uuid, and PIN also needs vertex_group_identifier.

    An entry that already carries this name is replaced, and the result says
    so under 'replaced_existing_entry'. Every other entry in the file is kept.
    A file whose top level carries a key outside an [entry] table is refused
    before anything is written, because the writer emits entry tables only
    and would destroy the rest of the file; edit such a file by hand first.
    With no 'path' the file already bound is written, which for MATERIAL and
    PIN is the file bound to the group and for SCENE and CONNECTION the file
    bound to the scene; passing one writes that file and binds it, which is
    what the panel's Save button does with a file it was just given. The file
    and the entry become the selection the panel shows.

    Args:
        kind: SCENE, MATERIAL, PIN or CONNECTION.
        name: Entry name to write. An entry named NONE is refused, since that
            identifier means "no profile" in the dropdowns.
        group_uuid: UUID of the group to read, for MATERIAL and PIN.
        vertex_group_identifier: Pin to read, in
            'object_name::vertex_group_name' form, for PIN. The pin also
            becomes the one selected in the panel, which is how the pin
            profile picker addresses a pin.
        path: Profile file to write instead of the bound one. Absolute, or
            '//' relative to a saved .blend.
    """
    kind = _check_kind(kind)
    spec = _PROFILE_KINDS[kind]
    entry_name = _check_entry_name(name)
    owner, group = _resolve_owner(kind, group_uuid)
    pin_item, pin_index = _resolve_pin_argument(kind, group, vertex_group_identifier)
    stored, absolute = _resolve_path(kind, owner, path)

    existing = {}
    # A file that exists and holds no bytes is a first save into it, not a
    # file that cannot be read: load_profiles answers {} for both, so the
    # size on disk is what separates them. A file of zero bytes carries no
    # entry to keep and no key to refuse, so neither check runs on one.
    if os.path.isfile(absolute) and os.path.getsize(absolute) > 0:
        existing, non_entry_keys = _read_file(absolute)
        if non_entry_keys:
            held = f"{len(existing)} " + (
                "entry" if len(existing) == 1 else "entries"
            )
            raise MCPError(
                f"{absolute} carries {', '.join(non_entry_keys)} at its top "
                "level, outside any [entry] table, and nothing was written. "
                "The addon's TOML writer emits one [entry] table per profile "
                "and has no rule for a top-level key: it truncates the file "
                "before writing and then stops on that key, leaving what it "
                f"had emitted so far in place of the {held} the file holds "
                "now. Move those keys into an entry table or "
                "remove them by hand first, or pass 'path' to save into "
                "another file."
            )
    entry = _read_entry(kind, owner, group, pin_item)
    save_profile_entry(absolute, entry_name, entry)

    # Read the file back before reporting a save. The addon's TOML writer
    # emits a value it has no rule for as a Python repr, which parses as
    # nothing, so a write that raised no exception is not on its own evidence
    # that the file can still be read.
    verified = load_profiles(absolute)
    if entry_name not in verified:
        also_lost = (
            f" The {len(existing)} entries the file already held are "
            "unreadable now as well."
            if existing
            else ""
        )
        raise MCPError(
            f"Wrote profile '{entry_name}' to {absolute}, but reading the "
            "file back does not find it, so what was written does not parse "
            f"as TOML.{also_lost}"
        )

    if pin_index is not None:
        # The pin dropdown applies to the selected pin, so the pin this call
        # named becomes the selected one before the selection is set.
        group.pin_vertex_groups_index = pin_index
    _select_entry(owner, spec, stored, entry_name)
    _refresh_viewport(overlays=kind != "CONNECTION")

    return {
        "message": f"Saved {kind} profile '{entry_name}' to {absolute}",
        "kind": kind,
        "name": entry_name,
        "path": stored,
        "absolute_path": absolute,
        "entry_keys": sorted(entry),
        "entry_key_count": len(entry),
        "replaced_existing_entry": entry_name in existing,
        "profile_count": len(verified),
    }


@mcp_handler
def load_profile(
    kind: str,
    name: str,
    group_uuid: str | None = None,
    vertex_group_identifier: str | None = None,
    path: str | None = None,
):
    """Apply a named entry from a profile file onto the scene.

    The entry overwrites every setting its kind covers, so a MATERIAL entry
    replaces the group's material parameters, including its Type and, when the
    entry embeds pins, the operations of the pins it names. A material lock
    does not hold against a profile load; it guards against the presets and
    the clipboard.

    An entry whose keys this kind's apply understands none of is refused
    rather than applied as nothing, which is what loading an entry saved under
    a different kind would otherwise look like. Keys the apply does drop are
    reported under 'ignored_keys', for an entry written by an older build.

    Every value the entry names for one of this kind's own parameters is read
    back off its property afterwards and compared with what the entry asked
    for, and a value the property did not take raises, naming both. Blender
    clamps a number outside a property's range and reports nothing, so a value
    the entry names is otherwise reported as applied while the property holds
    something else. A PIN entry's operations are rebuilt rather than assigned
    to properties this way, so they are reported as applied without that
    comparison.

    The file and the entry become the selection the panel shows.

    Args:
        kind: SCENE, MATERIAL, PIN or CONNECTION.
        name: Entry name, as reported by list_profiles.
        group_uuid: UUID of the group to write, for MATERIAL and PIN.
        vertex_group_identifier: Pin to write, in
            'object_name::vertex_group_name' form, for PIN. The pin also
            becomes the one selected in the panel, which is how the pin
            profile picker addresses a pin.
        path: Profile file to read instead of the bound one. Absolute, or
            '//' relative to a saved .blend.
    """
    kind = _check_kind(kind)
    spec = _PROFILE_KINDS[kind]
    owner, group = _resolve_owner(kind, group_uuid)
    pin_item, pin_index = _resolve_pin_argument(kind, group, vertex_group_identifier)
    stored, absolute = _resolve_path(kind, owner, path)
    entries, _non_entry_keys = _read_file(absolute)
    entry = entries.get(name)
    if entry is None:
        raise MCPError(
            f"{absolute} holds no profile named '{name}'. It holds "
            f"{', '.join(entries)}."
        )

    recognized = spec["recognized"]
    applied_keys = sorted(key for key in entry if key in recognized)
    ignored_keys = sorted(key for key in entry if key not in recognized)
    if not applied_keys:
        raise MCPError(
            f"Profile '{name}' carries none of the keys a {kind} profile is "
            f"made of, which are {spec['contents']}. Its keys are "
            f"{', '.join(sorted(entry))}, so it was probably saved under a "
            "different kind."
        )

    _apply_entry(kind, entry, owner, group, pin_item)
    if pin_index is not None:
        # The pin dropdown applies to the selected pin, so the pin this call
        # named becomes the selected one before the selection is set.
        group.pin_vertex_groups_index = pin_index
    _select_entry(owner, spec, stored, name)
    _refresh_viewport(overlays=kind != "CONNECTION")

    # The apply assigns each mapped value and reads nothing back, and Blender
    # clamps a number outside a property's range without raising, so a key
    # this handler recognizes is not on its own evidence that the property
    # took the entry's value. Read them back after the selection callback,
    # which is the last writer.
    target = _apply_target(kind, owner, group, pin_item)
    refused = []
    for key, prop_name, value in _assigned_values(kind, entry):
        current = getattr(target, prop_name)
        if not _matches(current, value):
            refused.append((key, _plain(value), _plain(current)))
    if refused:
        raise MCPError(
            f"{kind} profile '{name}' from {absolute} was applied, but not "
            "every value it names is what the property now holds: "
            f"{_describe_refused(refused)}. Everything else the entry names "
            "was written. Edit the entry to values the properties take before "
            "loading it again."
        )

    return {
        "message": f"Applied {kind} profile '{name}' from {absolute}",
        "kind": kind,
        "name": name,
        "path": stored,
        "absolute_path": absolute,
        "applied_keys": applied_keys,
        "ignored_keys": ignored_keys,
    }


@mcp_handler
def clear_profile_path(kind: str, group_uuid: str | None = None):
    """Unbind a profile file from the scene, leaving the file untouched.

    This is the panel's Clear button: the scene stops pointing at the file,
    and the dropdown for that kind goes empty. Nothing on disk changes, and
    the settings the last load applied stay as they are. Removing an entry
    from a profile file has no path in the addon, so no tool here does it.

    Args:
        kind: SCENE, MATERIAL, PIN or CONNECTION.
        group_uuid: UUID of the group the file is bound to, for MATERIAL and
            PIN.
    """
    kind = _check_kind(kind)
    spec = _PROFILE_KINDS[kind]
    owner, _group = _resolve_owner(kind, group_uuid)
    previous = getattr(owner, spec["path_prop"])

    operator_id = spec["clear_operator"]
    module, _, function = operator_id.partition(".")
    operator = getattr(getattr(bpy.ops, module), function)
    properties = {}
    if spec["needs_group"]:
        properties["group_index"] = get_group_index_by_uuid(group_uuid)
    _run_operator(operator, operator_id, **properties)

    return {
        "message": (
            f"Unbound the {kind} profile file {previous}"
            if previous
            else f"No {kind} profile file was bound"
        ),
        "kind": kind,
        "cleared_path": previous,
    }


# ---------------------------------------------------------------------------
# Clipboards
# ---------------------------------------------------------------------------


def _window_manager():
    """The WindowManager the clipboards live on."""
    wm = bpy.context.window_manager
    if wm is None:
        raise MCPError("No window manager, so the clipboards are unreachable")
    return wm


def _material_clipboard_names(clipboard, source_type: str) -> list[str]:
    """The clipboard parameters a group of `source_type` would paste.

    A model specific parameter only applies to the type that reads it, so a
    SHELL to SOLID paste carries the shared parameters and leaves the target's
    own shell fields alone.
    """
    return [
        name
        for name in list_copyable_params(
            clipboard, exclude=MATERIAL_CLIPBOARD_EXCLUDE
        )
        if material_param_applies(name, source_type)
    ]


@group_handler
def copy_material_parameters(group_uuid: str):
    """Copy a group's material parameters into the addon's material clipboard.

    The clipboard holds one set of parameters at a time and lives on the
    window manager, so it is not saved in the .blend and is empty again after
    a Blender restart. Copying records the source group's Type as well, which
    decides which parameters a later paste applies.

    This copies parameters only. Identity, the group's Type, its overlay
    color, its profile bindings, its per parameter locks and everything owned
    by an assigned object stay with their own group.

    Args:
        group_uuid: UUID of the group to copy from.
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    slot = get_group_index_by_uuid(group_uuid)
    _run_operator(
        bpy.ops.object.copy_material_params,
        "object.copy_material_params",
        group_index=slot,
    )
    wm = _window_manager()
    if not getattr(wm, "material_clipboard_valid", False):
        raise MCPError(
            "The copy operator reported success but left the material "
            "clipboard marked empty, so a paste would refuse."
        )
    copied = sorted(
        list_copyable_params(group, exclude=MATERIAL_CLIPBOARD_EXCLUDE)
    )
    return {
        "message": (
            f"Copied {len(copied)} material parameters from group "
            f"'{group.name}'"
        ),
        "group_uuid": group_uuid,
        "source_object_type": group.object_type,
        "parameters": copied,
        "parameter_count": len(copied),
    }


@group_handler
def paste_material_parameters(group_uuid: str):
    """Paste the material clipboard onto a group, keeping its locked values.

    Call copy_material_parameters first: the clipboard lives on the window
    manager, so a paste is refused after a restart, and after a session that
    never copied.

    A parameter the target has locked keeps its value, which is what the
    padlock beside it promises; those are reported under 'kept_locked'. A
    parameter only the source's Type reads is not pasted at all, so a paste
    between two Types carries the shared parameters and leaves the target's
    own model fields alone. The target's Type never changes.

    Args:
        group_uuid: UUID of the group to paste onto.
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    wm = _window_manager()
    if not getattr(wm, "material_clipboard_valid", False):
        raise MCPError(
            "The material clipboard is empty. Call copy_material_parameters "
            "on the source group first; the clipboard lives on the window "
            "manager, so it does not survive a Blender restart."
        )
    source_type = wm.material_clipboard_src_type or group.object_type
    locked = locked_props(group)
    candidates = _material_clipboard_names(wm.material_clipboard, source_type)
    pasted = sorted(name for name in candidates if name not in locked)
    kept_locked = sorted(name for name in candidates if name in locked)

    slot = get_group_index_by_uuid(group_uuid)
    _run_operator(
        bpy.ops.object.paste_material_params,
        "object.paste_material_params",
        group_index=slot,
    )
    _refresh_viewport(overlays=True)

    return {
        "message": (
            f"Pasted {len(pasted)} material parameters onto group "
            f"'{group.name}', keeping {len(kept_locked)} locked"
        ),
        "group_uuid": group_uuid,
        "source_object_type": source_type,
        "target_object_type": group.object_type,
        "pasted": pasted,
        "pasted_count": len(pasted),
        "kept_locked": kept_locked,
    }


@group_handler
def copy_pin_operations(group_uuid: str, vertex_group_identifier: str):
    """Copy one pin's operations into the addon's pin operation clipboard.

    The clipboard holds the operations of one pin at a time and lives on the
    window manager, so it is not saved in the .blend and is empty again after
    a Blender restart. The pin named here also becomes the one selected in the
    panel, which is how the pin clipboard addresses a pin.

    Args:
        group_uuid: UUID of the group that owns the pin.
        vertex_group_identifier: Pin in 'object_name::vertex_group_name' form,
            as reported by list_pins.
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    pin_item, pin_index = _resolve_pin_item(group, vertex_group_identifier)
    group.pin_vertex_groups_index = pin_index
    slot = get_group_index_by_uuid(group_uuid)
    _run_operator(
        bpy.ops.object.copy_pin_ops,
        "object.copy_pin_ops",
        group_index=slot,
    )
    wm = _window_manager()
    if not getattr(wm, "pin_ops_clipboard_valid", False):
        raise MCPError(
            "The copy operator reported success but left the pin operation "
            "clipboard marked empty, so a paste would refuse."
        )
    return {
        "message": (
            f"Copied {len(pin_item.operations)} operations from pin "
            f"'{vertex_group_identifier}'"
        ),
        "group_uuid": group_uuid,
        "vertex_group_identifier": vertex_group_identifier,
        "operation_count": len(wm.pin_ops_clipboard.operations),
    }


@group_handler
def paste_pin_operations(group_uuid: str, vertex_group_identifier: str):
    """Paste the pin operation clipboard onto a pin, replacing its operations.

    Every operation the target pin carries is discarded and replaced by the
    clipboard's, so this is not an append. Call copy_pin_operations first: the
    clipboard lives on the window manager, so a paste is refused after a
    restart, and after a session that never copied.

    The pin named here also becomes the one selected in the panel, which is
    how the pin clipboard addresses a pin.

    Args:
        group_uuid: UUID of the group that owns the pin.
        vertex_group_identifier: Pin in 'object_name::vertex_group_name' form,
            as reported by list_pins.
    """
    group = get_active_group_by_uuid_helper(group_uuid)
    pin_item, pin_index = _resolve_pin_item(group, vertex_group_identifier)
    wm = _window_manager()
    if not getattr(wm, "pin_ops_clipboard_valid", False):
        raise MCPError(
            "The pin operation clipboard is empty. Call copy_pin_operations "
            "on the source pin first; the clipboard lives on the window "
            "manager, so it does not survive a Blender restart."
        )
    replaced = len(pin_item.operations)
    group.pin_vertex_groups_index = pin_index
    slot = get_group_index_by_uuid(group_uuid)
    _run_operator(
        bpy.ops.object.paste_pin_ops,
        "object.paste_pin_ops",
        group_index=slot,
    )
    return {
        "message": (
            f"Pasted {len(pin_item.operations)} operations onto pin "
            f"'{vertex_group_identifier}', replacing {replaced}"
        ),
        "group_uuid": group_uuid,
        "vertex_group_identifier": vertex_group_identifier,
        "operation_count": len(pin_item.operations),
        "replaced_operation_count": replaced,
    }
