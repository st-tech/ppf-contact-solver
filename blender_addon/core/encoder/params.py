# File: encoder/params.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import hashlib
import json

import bpy  # pyright: ignore
import numpy as np

from ...models.groups import get_addon_data, iterate_object_groups
from . import _normalize_and_scale, _swap_axes, _to_solver
from .dyn import _encode_dyn_params, _encode_invisible_colliders
from .mesh import compute_group_bounding_box_diagonal
from .pin import _encode_pin_config


_FTETWILD_FLOAT_FIELDS = ("edge_length_fac", "epsilon", "stop_energy")
_FTETWILD_INT_FIELDS = ("num_opt_iter",)
_FTETWILD_BOOL_FIELDS = ("optimize", "simplify", "coarsen")


def _encode_ftetwild_kwargs(group) -> dict:
    """Build the pytetwild kwargs dict for a group.

    Returns {} when no sub-override is set, so the decoder treats an
    empty/missing dict as "use fTetWild defaults".
    """
    kwargs: dict = {}
    for field in _FTETWILD_FLOAT_FIELDS:
        if getattr(group, f"ftetwild_override_{field}", False):
            kwargs[field] = float(getattr(group, f"ftetwild_{field}"))
    for field in _FTETWILD_INT_FIELDS:
        if getattr(group, f"ftetwild_override_{field}", False):
            kwargs[field] = int(getattr(group, f"ftetwild_{field}"))
    for field in _FTETWILD_BOOL_FIELDS:
        if getattr(group, f"ftetwild_override_{field}", False):
            kwargs[field] = bool(getattr(group, f"ftetwild_{field}"))
    return kwargs


def _encode_scene_params(context, state):
    """Build the scene-level parameter dict."""
    scene = context.scene
    fps = (
        bpy.context.scene.render.fps
        if state.use_frame_rate_in_output
        else int(state.frame_rate)
    )
    frame_count = int(state.frame_count)
    auto_save = int(state.auto_save_interval) if bool(state.auto_save) else 0
    has_shell_type = any(
        group.object_type == "SHELL"
        for group in iterate_object_groups(scene)
        if group.active
    )
    use_inactive_momentum = has_shell_type and int(state.inactive_momentum_frames) > 0

    wind_dir = np.array([w for w in state.wind_direction], dtype=np.float64)
    wind_norm = np.linalg.norm(wind_dir)
    if wind_norm > 0:
        wind_dir = wind_dir / wind_norm
    wind_blender = wind_dir * float(state.wind_strength)
    # Z-up (Blender) -> Y-up (solver): (x, y, z) -> (x, z, -y)
    wind_force = _swap_axes(wind_blender)

    scene_params = {
        "dt": np.float32(state.step_size),
        "min-newton-steps": int(state.min_newton_steps),
        "air-density": np.float32(state.air_density),
        "air-friction": np.float32(state.air_friction),
        "friction-mode": str(state.friction_mode).lower(),
        "gravity": _swap_axes(state.gravity_3d),
        "wind": wind_force,
        "frames": frame_count - 1,  # Blender 1..N → remote 0..N-1
        "fps": fps,
        "csrmat-max-nnz": int(state.contact_nnz),
        "isotropic-air-friction": np.float32(state.vertex_air_damp),
        "auto-save": auto_save,
        "line-search-max-t": np.float32(state.line_search_max_t),
        "constraint-ghat": np.float32(state.constraint_ghat),
        "cg-max-iter": int(state.cg_max_iter),
        "cg-tol": np.float32(state.cg_tol),
        "include-face-mass": bool(state.include_face_mass),
        "disable-contact": bool(state.disable_contact),
    }

    if use_inactive_momentum:
        scene_params["inactive-momentum"] = float(state.inactive_momentum_frames) / fps

    # Stitch stiffness: take max across all active groups and merge pairs
    group_stiffness = max(
        (
            float(group.stitch_stiffness)
            for group in iterate_object_groups(scene)
            if group.active
        ),
        default=1.0,
    )
    pair_stiffness = max(
        (
            float(p.stitch_stiffness)
            for p in state.merge_pairs
            if p.object_a and p.object_b
        ),
        default=0.0,
    )
    stitch_stiffness = max(group_stiffness, pair_stiffness)
    scene_params["stitch-stiffness"] = np.float32(stitch_stiffness)

    return scene_params


def _encode_group_params(context, groups, state, fps):
    """Encode per-group material parameters."""
    from ..uuid_registry import resolve_assigned
    group_params = []
    for group in groups:
        for assigned in group.assigned_objects:
            resolve_assigned(assigned)
        for assigned in group.assigned_objects:
            if assigned.included and not assigned.uuid:
                raise RuntimeError(
                    f"Assigned object '{assigned.name}' in group '{group.name}' "
                    "has no UUID after resolve. All objects must have UUIDs before encoding."
                )
        objects = [assigned.name for assigned in group.assigned_objects if assigned.included]
        object_uuids = [assigned.uuid for assigned in group.assigned_objects if assigned.included]
        active_entries = {
            "SOLID": [
                "density",
                "young-mod",
                "poiss-rat",
                "shrink",
                "friction",
                "contact-gap",
                "contact-offset",
                "plasticity",
                "plasticity-threshold",
                "model",
                "velocity",
                "velocity-schedule",
                "collision-windows",
                "ftetwild",
            ],
            "SHELL": [
                "density",
                "young-mod",
                "poiss-rat",
                "friction",
                "contact-gap",
                "contact-offset",
                "strain-limit",
                "bend",
                "shrink-x",
                "shrink-y",
                "pressure",
                "plasticity",
                "plasticity-threshold",
                "bend-plasticity",
                "bend-plasticity-threshold",
                "bend-rest-from-geometry",
                "model",
                "velocity",
                "velocity-schedule",
                "collision-windows",
            ],
            "ROD": [
                "density",
                "young-mod",
                "friction",
                "contact-gap",
                "contact-offset",
                "bend",
                "strain-limit",
                "bend-plasticity",
                "bend-plasticity-threshold",
                "bend-rest-from-geometry",
                "model",
                "length-factor",
                "velocity",
                "velocity-schedule",
                "collision-windows",
            ],
            "STATIC": ["contact-gap", "contact-offset", "friction"],
        }
        model_map = {
            "ARAP": "arap",
            "STABLE_NEOHOOKEAN": "snhk",
            "BARAFF_WITKIN": "baraff-witkin",
        }
        if group.object_type == "SOLID":
            model = group.solid_model
        elif group.object_type == "SHELL":
            model = group.shell_model
            # Stable NeoHookean is a volumetric model; the UI dropped
            # it from the SHELL picker. A `.blend` saved before that
            # change can still hold the old enum value, in which case
            # Blender keeps it on the property even though the items
            # list no longer offers it. Coerce to ARAP so the upload
            # carries a model the solver can build a SHELL group with.
            if model == "STABLE_NEOHOOKEAN":
                model = "ARAP"
        elif group.object_type == "ROD":
            model = group.rod_model
        else:
            model = "N/A"
        strain_limit = (
            np.float32(group.strain_limit) if group.enable_strain_limit else 0.0
        )
        # Shrink/extend invalidates strain limiting: the strain-limit solver
        # bakes the rest shape assuming unit scaling, so any shrink_x/y != 1
        # makes the limit ill-defined. Silently disable to match the UI warning.
        if group.object_type == "SHELL" and (
            group.shrink_x != 1.0 or group.shrink_y != 1.0
        ):
            strain_limit = np.float32(0.0)

        if group.object_type == "SOLID":
            density = np.float32(group.solid_density)
            young_modulus = np.float32(group.solid_young_modulus)
            poisson_ratio = np.float32(group.solid_poisson_ratio)
        elif group.object_type == "SHELL":
            density = np.float32(group.shell_density)
            young_modulus = np.float32(group.shell_young_modulus)
            poisson_ratio = np.float32(group.shell_poisson_ratio)
        elif group.object_type == "ROD":
            density = np.float32(group.rod_density)
            young_modulus = np.float32(group.rod_young_modulus)
            poisson_ratio = np.float32(0.0)  # Rod objects don't use Poisson's ratio
        else:  # STATIC
            density = np.float32(1000.0)  # Default density for static
            young_modulus = np.float32(100000.0)  # Default young modulus for static
            poisson_ratio = np.float32(0.35)  # Default Poisson ratio for static

        if group.use_group_bounding_box_diagonal:
            bbox_diagonal = compute_group_bounding_box_diagonal(group)
            contact_gap_value = bbox_diagonal * group.contact_gap_rat
            contact_offset_value = bbox_diagonal * group.contact_offset_rat
        else:
            contact_gap_value = group.contact_gap
            contact_offset_value = group.contact_offset
        group.computed_contact_gap = contact_gap_value
        group.computed_contact_offset = contact_offset_value

        params = {
            "model": str(model_map.get(model, "unknown")),
            "density": density,
            "young-mod": young_modulus,
            "poiss-rat": poisson_ratio,
            "friction": np.float32(group.friction),
            "contact-gap": np.float32(contact_gap_value),
            "contact-offset": np.float32(contact_offset_value),
            "bend": np.float32(group.bend),
            "shrink": np.float32(group.shrink),
            "shrink-x": np.float32(group.shrink_x),
            "shrink-y": np.float32(group.shrink_y),
            "strain-limit": strain_limit,
            "pressure": np.float32(group.inflate_pressure) if group.enable_inflate else np.float32(0.0),
            "plasticity": np.float32(group.plasticity) if group.enable_plasticity else np.float32(0.0),
            "plasticity-threshold": np.float32(group.plasticity_threshold) if group.enable_plasticity else np.float32(0.0),
            "bend-plasticity": np.float32(group.bend_plasticity) if group.enable_bend_plasticity else np.float32(0.0),
            "bend-plasticity-threshold": np.float32(group.bend_plasticity_threshold) if group.enable_bend_plasticity else np.float32(0.0),
            "bend-rest-from-geometry": np.float32(1.0 if group.bend_rest_angle_source == "FROM_GEOMETRY" else 0.0),
            "length-factor": np.float32(1.0),
            # Per-object dicts key on UUID so the decoder looks up the
            # right entry even if the Blender object was renamed between
            # transfer and simulation. pin_config already uses UUID.
            "velocity": {
                assigned.uuid: next(
                    (_swap_axes(_normalize_and_scale(kf.direction, kf.speed))
                     for kf in assigned.velocity_keyframes if kf.frame == 1),
                    np.array([0.0, 0.0, 0.0], dtype=np.float32),
                )
                for assigned in group.assigned_objects
                if assigned.included
            },
            "velocity-schedule": {
                assigned.uuid: [
                    (float(kf.frame - 1) / fps, _swap_axes(_normalize_and_scale(kf.direction, kf.speed)))
                    for kf in assigned.velocity_keyframes if kf.frame > 1
                ]
                for assigned in group.assigned_objects
                if assigned.included
            },
            "ftetwild": _encode_ftetwild_kwargs(group),
            "collision-windows": {
                assigned.uuid: [
                    (
                        float(cw.frame_start - 1) / fps,
                        float(max(cw.frame_end, cw.frame_start) - 1) / fps,
                    )
                    for cw in assigned.collision_windows
                ]
                for assigned in group.assigned_objects
                if assigned.included
            } if group.use_collision_windows else {},
        }
        obj_type = group.object_type
        del_keys = []
        for key in params:
            if key not in active_entries.get(obj_type, []):
                del_keys.append(key)
        for key in del_keys:
            del params[key]
        group_params.append((params, objects, object_uuids))

    return group_params


def _encode_merge_pairs(context, groups):
    """Encode non-explicit merge pairs that still use alias merging."""
    scene = context.scene

    from ...mesh_ops.merge_ops import cleanup_stale_merge_pairs

    cleanup_stale_merge_pairs(scene)
    state = get_addon_data(scene).state
    merge_pairs_list = [
        (p.object_a, p.object_b, float(p.stitch_stiffness),
         p.object_a_uuid, p.object_b_uuid)
        for p in state.merge_pairs
        if p.object_a and p.object_b and not p.cross_stitch_json
    ]

    return merge_pairs_list


def _object_type_by_uuid(groups):
    from ..uuid_registry import resolve_assigned
    mapping = {}
    for group in groups:
        for assigned in group.assigned_objects:
            if not assigned.included:
                continue
            resolve_assigned(assigned)
            if not assigned.uuid:
                raise RuntimeError(
                    f"Assigned object '{assigned.name}' in group '{group.name}' "
                    "has no UUID after resolve. All objects must have UUIDs before encoding."
                )
            mapping[assigned.uuid] = group.object_type
    return mapping


def _encode_explicit_merge_pairs(context, groups):
    scene = context.scene
    state = get_addon_data(scene).state
    type_by_uuid = _object_type_by_uuid(groups)
    result = []
    for pair in state.merge_pairs:
        if not pair.cross_stitch_json:
            continue
        if not pair.object_a_uuid or not pair.object_b_uuid:
            continue
        pair_types = {
            type_by_uuid.get(pair.object_a_uuid),
            type_by_uuid.get(pair.object_b_uuid),
        }
        if pair_types not in ({"ROD"}, {"SHELL"}):
            continue
        try:
            data = json.loads(pair.cross_stitch_json)
        except json.JSONDecodeError:
            continue
        ind = data.get("ind", [])
        if not ind:
            continue
        w_list = data.get("w", [])
        pairs = []
        for row, weight in zip(ind, w_list):
            if len(row) < 4 or len(weight) < 4:
                continue
            # Pick the target vertex with the highest barycentric weight
            bary = [float(weight[1]), float(weight[2]), float(weight[3])]
            best = max(range(3), key=lambda k: bary[k])
            pairs.append([int(row[0]), int(row[1 + best])])
        result.append({
            "source_uuid": pair.object_a_uuid,
            "target_uuid": pair.object_b_uuid,
            "pairs": pairs,
        })
    return result


def _encode_cross_stitch(context):
    scene = context.scene
    state = get_addon_data(scene).state
    groups = [group for group in iterate_object_groups(scene) if group.active]
    type_by_uuid = _object_type_by_uuid(groups)
    result = []
    for pair in state.merge_pairs:
        if not pair.cross_stitch_json:
            continue
        # Skip entries whose UUIDs could not be resolved (e.g. legacy
        # names renamed before migration).  The frontend raises ValueError
        # on empty source_uuid/target_uuid, so silently dropping them here
        # lets the rest of the simulation proceed.
        if not pair.object_a_uuid or not pair.object_b_uuid:
            continue
        pair_types = {
            type_by_uuid.get(pair.object_a_uuid),
            type_by_uuid.get(pair.object_b_uuid),
        }
        if pair_types in ({"ROD"}, {"SHELL"}):
            continue
        try:
            data = json.loads(pair.cross_stitch_json)
        except json.JSONDecodeError:
            continue
        if data:
            if not data.get("source_uuid") or not data.get("target_uuid"):
                continue
            target_points = data.get("target_points")
            if target_points:
                data["target_points"] = [_to_solver(point) for point in target_points]
            data["stitch_stiffness"] = float(pair.stitch_stiffness)
            result.append(data)
    return result


def _build_param_dict(context) -> dict:
    """Assemble the parameter dict that ``encode_param`` serializes.

    Factored out so ``compute_param_hash`` and ``encode_param_with_hash``
    can derive a stable fingerprint from the same source-of-truth dict.
    Both call sites must see identical content; the "Update Params"
    button's enabled state depends on the fingerprint matching what the
    server computed on the last upload.
    """
    scene = context.scene
    state = get_addon_data(scene).state
    groups = [group for group in iterate_object_groups(scene) if group.active]

    fps = (
        bpy.context.scene.render.fps
        if state.use_frame_rate_in_output
        else int(state.frame_rate)
    )

    scene_params = _encode_scene_params(context, state)
    group_params = _encode_group_params(context, groups, state, fps)
    pin_config = _encode_pin_config(context, groups, state)
    merge_pairs_list = _encode_merge_pairs(context, groups)
    explicit_merge_pairs = _encode_explicit_merge_pairs(context, groups)
    cross_stitch = _encode_cross_stitch(context)
    dyn_param = _encode_dyn_params(state, fps)

    result = {
        "scene": scene_params,
        "group": group_params,
        "pin_config": pin_config,
        "merge_pairs": merge_pairs_list,
    }
    if explicit_merge_pairs:
        result["explicit_merge_pairs"] = explicit_merge_pairs
    if cross_stitch:
        result["cross_stitch"] = cross_stitch
    if dyn_param:
        result["dyn_param"] = dyn_param
    ic = _encode_invisible_colliders(state, fps)
    if ic:
        result["invisible_colliders"] = ic
    return result


def encode_param(context) -> bytes:
    # CBOR envelope on the wire. See blender_addon/core/encoder/cbor_encode.py.
    from .cbor_encode import dumps_envelope
    return dumps_envelope("Param", _build_param_dict(context))


def encode_param_with_hash(context) -> tuple[bytes, str]:
    """Encode the param tree and hash the encoded bytes in one pass.

    Avoids the build-twice cost of ``encode_param`` + ``compute_param_hash``
    and the canonical-JSON walk that dominated runtime on scenes with
    large keyframe streams. ``_build_param_dict`` produces a fixed
    dict-insertion order and ``cbor2.dumps`` preserves it, so the
    SHA-256 of the bytes is stable across runs.
    """
    from .cbor_encode import dumps_envelope
    tree = _build_param_dict(context)
    encoded = dumps_envelope("Param", tree)
    return encoded, hashlib.sha256(encoded).hexdigest()


def compute_param_hash(context) -> str:
    """Stable hash of the current parameter set.

    Hashes the same CBOR bytes that ``encode_param`` produces, so the
    upload-time hash and the click-time drift hash always agree.
    """
    from .cbor_encode import dumps_envelope
    return hashlib.sha256(
        dumps_envelope("Param", _build_param_dict(context))
    ).hexdigest()
