# File: profile.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Connection profile loading from TOML files.

import os
import tomllib

# TOML "type" string → SSHState.server_type enum value
PROFILE_TYPE_MAP = {
    "Local": "LOCAL",
    "SSH": "CUSTOM",
    "SSH Command": "COMMAND",
    "Docker": "DOCKER",
    "Docker over SSH": "DOCKER_SSH",
    "Docker over SSH Command": "DOCKER_SSH_COMMAND",
    "Windows Native": "WIN_NATIVE",
}

# TOML key → SSHState property name (keys not listed here are ignored)
_SSH_STATE_FIELDS = {
    "host": "host",
    "port": "port",
    "username": "username",
    "key_path": "key_path",
    "command": "command",
    "container": "container",
    "remote_path": "ssh_remote_path",
    "docker_path": "docker_path",
    "local_path": "local_path",
    "win_native_path": "win_native_path",
    "docker_port": "docker_port",
}


def load_profiles(path: str) -> dict[str, dict]:
    """Parse a TOML profile file and return all profiles.

    Args:
        path: Absolute path to the .toml file.

    Returns:
        dict mapping profile name → profile data dict.
        Empty dict on any error (file not found, parse error, etc.).
    """
    try:
        path = os.path.expanduser(path)
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def get_profile_names(path: str) -> list[str]:
    """Return sorted list of profile names from a TOML file."""
    profiles = load_profiles(path)
    return sorted(profiles.keys())


def apply_profile(profile: dict, ssh_state) -> bool:
    """Apply a profile dict to SSHState properties.

    Args:
        profile: Single profile dict from the TOML file.
        ssh_state: The SSHState PropertyGroup instance.

    Returns:
        True if the profile type was valid, False otherwise.
    """
    # Set server_type from "type" key
    type_str = profile.get("type", "")
    server_type = PROFILE_TYPE_MAP.get(type_str)
    if server_type is None:
        return False
    ssh_state.server_type = server_type

    # Set SSHState fields
    for toml_key, prop_name in _SSH_STATE_FIELDS.items():
        if toml_key in profile:
            setattr(ssh_state, prop_name, profile[toml_key])

    return True


# Reverse map: server_type enum → TOML type string
_REVERSE_TYPE_MAP = {v: k for k, v in PROFILE_TYPE_MAP.items()}


def read_connection_profile(ssh_state) -> dict:
    """Read current SSHState properties into a profile dict."""
    result = {}
    type_str = _REVERSE_TYPE_MAP.get(ssh_state.server_type)
    if type_str:
        result["type"] = type_str
    for toml_key, prop_name in _SSH_STATE_FIELDS.items():
        val = getattr(ssh_state, prop_name)
        if val is not None:
            result[toml_key] = val
    return result


# TOML key → State property name for scene params
_SCENE_PARAM_FIELDS = {
    "step_size": "step_size",
    "min_newton_steps": "min_newton_steps",
    "air_density": "air_density",
    "air_friction": "air_friction",
    "gravity": "gravity_3d",
    "frame_count": "frame_count",
    "frame_rate": "frame_rate",
    "use_frame_rate_in_output": "use_frame_rate_in_output",
    "inactive_momentum_frames": "inactive_momentum_frames",
    "wind_direction": "wind_direction",
    "wind_strength": "wind_strength",
    "contact_nnz": "contact_nnz",
    "vertex_air_damp": "vertex_air_damp",
    "auto_save": "auto_save",
    "auto_save_interval": "auto_save_interval",
    "line_search_max_t": "line_search_max_t",
    "constraint_ghat": "constraint_ghat",
    "cg_max_iter": "cg_max_iter",
    "cg_tol": "cg_tol",
    "include_face_mass": "include_face_mass",
    "disable_contact": "disable_contact",
}

# TOML key → ObjectGroup property name for material params.
# Everything here is group-level. Per-assigned-object state (velocity
# overrides, collision windows, pins) lives on `AssignedObject` and is
# therefore already outside this set — do NOT mirror those into the
# profile: they are tied to a specific assigned object's identity and
# don't make sense to copy across groups.
_MATERIAL_PARAM_FIELDS = {
    "object_type": "object_type",
    "color": "color",
    "solid_model": "solid_model",
    "shell_model": "shell_model",
    "rod_model": "rod_model",
    "solid_density": "solid_density",
    "shell_density": "shell_density",
    "rod_density": "rod_density",
    "solid_young_modulus": "solid_young_modulus",
    "shell_young_modulus": "shell_young_modulus",
    "rod_young_modulus": "rod_young_modulus",
    "solid_poisson_ratio": "solid_poisson_ratio",
    "shell_poisson_ratio": "shell_poisson_ratio",
    "friction": "friction",
    "contact_gap": "contact_gap",
    "contact_offset": "contact_offset",
    "use_group_bounding_box_diagonal": "use_group_bounding_box_diagonal",
    "contact_gap_rat": "contact_gap_rat",
    "contact_offset_rat": "contact_offset_rat",
    "bend": "bend",
    "shrink": "shrink",
    "shrink_x": "shrink_x",
    "shrink_y": "shrink_y",
    "enable_strain_limit": "enable_strain_limit",
    "strain_limit": "strain_limit",
    "enable_inflate": "enable_inflate",
    "inflate_pressure": "inflate_pressure",
    "stitch_stiffness": "stitch_stiffness",
    "enable_plasticity": "enable_plasticity",
    "plasticity": "plasticity",
    "plasticity_threshold": "plasticity_threshold",
    "enable_bend_plasticity": "enable_bend_plasticity",
    "bend_plasticity": "bend_plasticity",
    "bend_plasticity_threshold": "bend_plasticity_threshold",
    "bend_rest_angle_source": "bend_rest_angle_source",
    "use_collision_windows": "use_collision_windows",
}

_VECTOR_PROPERTIES = {"gravity_3d", "wind_direction", "color"}


def apply_scene_profile(profile: dict, state) -> bool:
    """Apply a scene param profile dict to State properties."""
    for toml_key, prop_name in _SCENE_PARAM_FIELDS.items():
        if toml_key in profile:
            value = profile[toml_key]
            if prop_name in _VECTOR_PROPERTIES and isinstance(value, list):
                value = tuple(value)
            setattr(state, prop_name, value)
    # Restore dynamic parameters
    if "dyn_params" in profile:
        state.dyn_params.clear()
        for param_entry in profile["dyn_params"]:
            param_type = param_entry.get("param_type", "")
            keyframes = param_entry.get("keyframes", [])
            dyn_item = state.dyn_params.add()
            dyn_item.param_type = param_type
            for i, kf_entry in enumerate(keyframes):
                kf = dyn_item.keyframes.add()
                kf.frame = kf_entry.get("frame", 1)
                if i > 0:
                    kf.use_hold = kf_entry.get("use_hold", False)
                    if not kf.use_hold:
                        if param_type == "GRAVITY" and "gravity_value" in kf_entry:
                            kf.gravity_value = tuple(kf_entry["gravity_value"])
                        elif param_type == "WIND":
                            if "wind_direction_value" in kf_entry:
                                kf.wind_direction_value = tuple(kf_entry["wind_direction_value"])
                            if "wind_strength_value" in kf_entry:
                                kf.wind_strength_value = kf_entry["wind_strength_value"]
                        elif "scalar_value" in kf_entry:
                            kf.scalar_value = kf_entry["scalar_value"]
        state.dyn_params_index = 0 if len(state.dyn_params) > 0 else -1
    # Restore invisible colliders
    if "colliders" in profile:
        apply_collider_profile(profile, state)
    return True


def apply_material_profile(profile: dict, object_group) -> bool:
    """Apply a material param profile dict to ObjectGroup properties."""
    for toml_key, prop_name in _MATERIAL_PARAM_FIELDS.items():
        if toml_key in profile:
            value = profile[toml_key]
            if prop_name in _VECTOR_PROPERTIES and isinstance(value, list):
                value = tuple(value)
            setattr(object_group, prop_name, value)
    # Restore embedded pin profiles when present (UUID-keyed apply).
    if "pins" in profile:
        apply_pin_profiles(profile, object_group)
    return True


_TOML_STRING_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _toml_escape_string(s: str) -> str:
    """Escape a Python string for a basic TOML string literal.

    TOML basic strings use backslash escapes for: backslash, double-quote,
    \\b \\t \\n \\f \\r, plus \\uXXXX for other control characters. Anything
    else is emitted as-is (TOML is UTF-8, so non-ASCII text passes through).
    """
    out = []
    for ch in s:
        esc = _TOML_STRING_ESCAPES.get(ch)
        if esc is not None:
            out.append(esc)
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return "".join(out)


def _toml_is_bare_key(s: str) -> bool:
    """TOML bare keys: [A-Za-z0-9_-]+."""
    if not s:
        return False
    return all(c.isalnum() or c in "_-" for c in s)


def _toml_key(s: str) -> str:
    """Format a TOML key: bare if simple, quoted otherwise."""
    if _toml_is_bare_key(s):
        return s
    return f'"{_toml_escape_string(s)}"'


def _toml_value(val) -> str:
    """Format a Python value as a TOML value string."""
    if val is None:
        # TOML has no null; emit an empty string so loader round-trips produce
        # an empty value the caller's coerce step can handle, instead of the
        # bare literal "None" which would make the whole file unparsable.
        return '""'
    if isinstance(val, bool):
        return "true" if val else "false"
    elif isinstance(val, (list, tuple)):
        items = ", ".join(_toml_value(x) for x in val)
        return f"[{items}]"
    elif isinstance(val, float):
        return str(val)
    elif isinstance(val, int):
        return str(val)
    elif isinstance(val, str):
        return f'"{_toml_escape_string(val)}"'
    return str(val)


def read_scene_profile(state) -> dict:
    """Read current State properties into a profile dict."""
    result = {}
    for toml_key, prop_name in _SCENE_PARAM_FIELDS.items():
        val = getattr(state, prop_name)
        if prop_name in _VECTOR_PROPERTIES:
            val = list(val)
        result[toml_key] = val
    # Dynamic parameters: grouped by param_type with nested keyframes
    dyn_entries = []
    for dyn_item in state.dyn_params:
        param_entry = {"param_type": dyn_item.param_type, "keyframes": []}
        for i, kf in enumerate(dyn_item.keyframes):
            kf_entry = {"frame": kf.frame}
            if i > 0:
                kf_entry["use_hold"] = kf.use_hold
                if not kf.use_hold:
                    if dyn_item.param_type == "GRAVITY":
                        kf_entry["gravity_value"] = list(kf.gravity_value)
                    elif dyn_item.param_type == "WIND":
                        kf_entry["wind_direction_value"] = list(kf.wind_direction_value)
                        kf_entry["wind_strength_value"] = kf.wind_strength_value
                    else:
                        kf_entry["scalar_value"] = kf.scalar_value
            param_entry["keyframes"].append(kf_entry)
        dyn_entries.append(param_entry)
    if dyn_entries:
        result["dyn_params"] = dyn_entries
    # Invisible colliders
    collider_data = read_collider_profile(state)
    if "colliders" in collider_data:
        result["colliders"] = collider_data["colliders"]
    return result


def read_material_profile(object_group, include_pins=False) -> dict:
    """Read current ObjectGroup properties into a profile dict."""
    result = {}
    for toml_key, prop_name in _MATERIAL_PARAM_FIELDS.items():
        val = getattr(object_group, prop_name)
        if prop_name in _VECTOR_PROPERTIES:
            val = list(val)
        result[toml_key] = val
    if include_pins:
        pin_data = read_pin_profiles(object_group)
        result.update(pin_data)
    return result


_OP_FIELDS = {
    "op_type": "op_type",
    "delta": "delta",
    "spin_axis": "spin_axis",
    "spin_angular_velocity": "spin_angular_velocity",
    "spin_flip": "spin_flip",
    "spin_center": "spin_center",
    "spin_center_mode": "spin_center_mode",
    "spin_center_vertex": "spin_center_vertex",
    "spin_center_direction": "spin_center_direction",
    "scale_factor": "scale_factor",
    "scale_center": "scale_center",
    "scale_center_mode": "scale_center_mode",
    "scale_center_vertex": "scale_center_vertex",
    "scale_center_direction": "scale_center_direction",
    "torque_axis_component": "torque_axis_component",
    "torque_magnitude": "torque_magnitude",
    "torque_flip": "torque_flip",
    "frame_start": "frame_start",
    "frame_end": "frame_end",
    "transition": "transition",
}

_OP_VECTOR_FIELDS = {"delta", "spin_axis", "spin_center", "scale_center", "spin_center_direction", "scale_center_direction"}


def read_collider_profile(state) -> dict:
    """Read current invisible colliders into a profile dict."""
    colliders = []
    for item in state.invisible_colliders:
        entry = {
            "collider_type": item.collider_type,
            "name": item.name,
            "position": list(item.position),
            "contact_gap": item.contact_gap,
            "friction": item.friction,
        }
        if item.collider_type == "WALL":
            entry["normal"] = list(item.normal)
        else:
            entry["radius"] = item.radius
            entry["hemisphere"] = item.hemisphere
            entry["invert"] = item.invert
        kf_list = []
        for i, kf in enumerate(item.keyframes):
            kf_entry = {"frame": kf.frame}
            if i > 0:
                kf_entry["use_hold"] = kf.use_hold
                if not kf.use_hold:
                    kf_entry["position"] = list(kf.position)
                    if item.collider_type == "SPHERE":
                        kf_entry["radius"] = kf.radius
            kf_list.append(kf_entry)
        entry["keyframes"] = kf_list
        colliders.append(entry)
    result = {}
    if colliders:
        result["colliders"] = colliders
    return result


def apply_collider_profile(profile: dict, state) -> bool:
    """Apply a collider profile dict to State invisible_colliders."""
    if "colliders" not in profile:
        return True
    state.invisible_colliders.clear()
    for entry in profile["colliders"]:
        item = state.invisible_colliders.add()
        item.collider_type = entry.get("collider_type", "WALL")
        item.name = entry.get("name", "")
        if "position" in entry:
            item.position = tuple(entry["position"])
        if item.collider_type == "WALL":
            if "normal" in entry:
                item.normal = tuple(entry["normal"])
        else:
            if "radius" in entry:
                item.radius = entry["radius"]
            item.hemisphere = entry.get("hemisphere", False)
            item.invert = entry.get("invert", False)
        item.contact_gap = entry.get("contact_gap", 0.001)
        item.friction = entry.get("friction", 0.0)
        for i, kf_entry in enumerate(entry.get("keyframes", [])):
            kf = item.keyframes.add()
            kf.frame = kf_entry.get("frame", 1)
            if i > 0:
                kf.use_hold = kf_entry.get("use_hold", False)
                if not kf.use_hold:
                    if "position" in kf_entry:
                        kf.position = tuple(kf_entry["position"])
                    if "radius" in kf_entry:
                        kf.radius = kf_entry["radius"]
    state.invisible_colliders_index = 0 if len(state.invisible_colliders) > 0 else -1
    return True


def read_pin_operations(pin_item) -> dict:
    """Read operations from a PinVertexGroupItem into a profile dict."""
    ops = []
    for op in pin_item.operations:
        op_dict = {}
        for toml_key, prop_name in _OP_FIELDS.items():
            val = getattr(op, prop_name)
            if prop_name in _OP_VECTOR_FIELDS:
                val = list(val)
            op_dict[toml_key] = val
        ops.append(op_dict)
    return {"operations": ops}


def apply_pin_operations(profile: dict, pin_item):
    """Apply operations from a profile dict to a PinVertexGroupItem."""
    if "operations" not in profile:
        return
    pin_item.operations.clear()
    for op_data in profile["operations"]:
        op = pin_item.operations.add()
        for toml_key, prop_name in _OP_FIELDS.items():
            if toml_key in op_data:
                val = op_data[toml_key]
                if prop_name in _OP_VECTOR_FIELDS and isinstance(val, list):
                    val = tuple(val)
                setattr(op, prop_name, val)


def read_pin_profiles(object_group) -> dict:
    """Read all pin vertex groups on *object_group* into a profile dict.

    Per project-wide UUID-only identity rule, internal identity is
    ``(object_uuid, vg_name)``. The on-disk TOML table key uses the
    display identifier ``[obj_name][vg_name]`` purely for user
    readability; ``object_uuid`` and the vertex group name are embedded
    in the entry itself so :func:`apply_pin_profiles` can match by UUID
    even after object or VG renames.
    """
    from ..models.groups import decode_vertex_group_identifier

    result = {}
    for pin_item in object_group.pin_vertex_groups:
        entry = read_pin_operations(pin_item)
        # Embed stable identity (UUID + VG name) so apply-back is
        # robust to object/VG renames. Readable display key stays for
        # the TOML table name.
        entry["object_uuid"] = pin_item.object_uuid
        _, vg_name = decode_vertex_group_identifier(pin_item.name)
        if vg_name:
            entry["vg_name"] = vg_name
        if pin_item.vg_hash:
            entry["vg_hash"] = pin_item.vg_hash
        result[pin_item.name] = entry
    return {"pins": result} if result else {}


def apply_pin_profiles(profile: dict, object_group):
    """Apply all pin profiles back to *object_group*'s pin_vertex_groups.

    Match by ``(object_uuid, vg_name)`` — NEVER by display name —
    so renames never break profile apply. Falls back to display-name
    match only when the profile entry lacks a stored UUID (legacy
    profiles written before UUID embedding).
    """
    from ..models.groups import decode_vertex_group_identifier

    pins = profile.get("pins")
    if not pins:
        return
    # Build UUID-keyed index: (object_uuid, vg_name) -> pin_item.
    # vg_name is decoded from the display identifier; object_uuid is
    # the stable identity we trust.
    uuid_index = {}
    name_index = {}
    for item in object_group.pin_vertex_groups:
        _, vg_name = decode_vertex_group_identifier(item.name)
        if item.object_uuid and vg_name:
            uuid_index[(item.object_uuid, vg_name)] = item
        name_index[item.name] = item
    for pin_key, pin_data in pins.items():
        target = None
        obj_uuid = pin_data.get("object_uuid", "") if isinstance(pin_data, dict) else ""
        vg_name = pin_data.get("vg_name", "") if isinstance(pin_data, dict) else ""
        if obj_uuid and vg_name:
            target = uuid_index.get((obj_uuid, vg_name))
        # Legacy fallback: profile without embedded UUID
        if target is None:
            target = name_index.get(pin_key)
        if target is None:
            continue
        apply_pin_operations(pin_data, target)


def save_profile_entry(path: str, entry_name: str, data: dict):
    """Save a profile entry to a TOML file, preserving other entries.

    If the file exists, updates/adds the entry. Otherwise creates a new file.
    """
    path = os.path.expanduser(path)
    profiles = load_profiles(path) if os.path.exists(path) else {}
    profiles[entry_name] = data

    with open(path, "w", encoding="utf-8") as f:
        for name, profile in profiles.items():
            name_tok = _toml_key(name)
            f.write(f"[{name_tok}]\n")
            for key, val in profile.items():
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    continue  # Write arrays of tables after scalar fields
                f.write(f"{_toml_key(key)} = {_toml_value(val)}\n")
            # Write arrays of tables (e.g., operations, dyn_params)
            for key, val in profile.items():
                if isinstance(val, list) and val and isinstance(val[0], dict):
                    key_tok = _toml_key(key)
                    for item in val:
                        f.write(f"\n[[{name_tok}.{key_tok}]]\n")
                        for k, v in item.items():
                            if isinstance(v, list) and v and isinstance(v[0], dict):
                                continue  # Write nested arrays after scalar fields
                            f.write(f"{_toml_key(k)} = {_toml_value(v)}\n")
                        # Write nested arrays of tables (e.g., keyframes)
                        for k, v in item.items():
                            if isinstance(v, list) and v and isinstance(v[0], dict):
                                k_tok = _toml_key(k)
                                for sub_item in v:
                                    f.write(f"\n[[{name_tok}.{key_tok}.{k_tok}]]\n")
                                    for sk, sv in sub_item.items():
                                        f.write(f"{_toml_key(sk)} = {_toml_value(sv)}\n")
            f.write("\n")
