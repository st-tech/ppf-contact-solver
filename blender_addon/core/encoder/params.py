# File: encoder/params.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import hashlib
import json

import numpy as np

from ...models.groups import get_addon_data, iterate_object_groups
from . import _normalize_and_scale, _swap_axes, _to_solver, resolve_fps
from .dyn import _encode_dyn_params, _encode_invisible_colliders
from .mesh import compute_group_bounding_box_diagonal, evaluate_at_frame_one
from .pin import _encode_pin_config


_FTETWILD_FLOAT_FIELDS = ("edge_length_fac", "epsilon", "stop_energy")
_FTETWILD_INT_FIELDS = ("num_opt_iter",)
_FTETWILD_BOOL_FIELDS = ("optimize", "simplify", "coarsen")


def _encode_obj_tet_kwargs(assigned) -> dict:
    """Build the per-object ``tetrahedralize()`` kwargs dict.

    The backend picker and its overrides live on the AssignedObject, so
    each SOLID mesh chooses its own tetrahedralizer. Returns ``{}`` for a
    plain fTetWild object with no overrides (no ``backend`` key), so the
    decoder treats it as "use fTetWild defaults" and the param hash for
    untouched scenes is unchanged. TetGen objects always carry
    ``backend="tetgen"`` so the choice survives even with no overrides.
    """
    kwargs: dict = {}
    backend = getattr(assigned, "tet_backend", "FTETWILD")
    if backend == "TETGEN":
        kwargs["backend"] = "tetgen"
        if getattr(assigned, "tetgen_override_min_ratio", False):
            kwargs["min_ratio"] = float(assigned.tetgen_min_ratio)
        if getattr(assigned, "tetgen_override_max_volume", False):
            max_vol = float(assigned.tetgen_max_volume)
            # 0 means "uncapped"; never forward a zero max volume.
            if max_vol > 0.0:
                kwargs["max_volume"] = max_vol
        return kwargs
    for field in _FTETWILD_FLOAT_FIELDS:
        if getattr(assigned, f"ftetwild_override_{field}", False):
            kwargs[field] = float(getattr(assigned, f"ftetwild_{field}"))
    for field in _FTETWILD_INT_FIELDS:
        if getattr(assigned, f"ftetwild_override_{field}", False):
            kwargs[field] = int(getattr(assigned, f"ftetwild_{field}"))
    for field in _FTETWILD_BOOL_FIELDS:
        if getattr(assigned, f"ftetwild_override_{field}", False):
            kwargs[field] = bool(getattr(assigned, f"ftetwild_{field}"))
    return kwargs


def _encode_scene_params(context, state, fps):
    """Build the scene-level parameter dict."""
    scene = context.scene
    frame_count = int(state.frame_count)
    auto_save = int(state.auto_save_interval) if bool(state.auto_save) else 0
    # Checkpoint retention. 0 keeps all (the solver's remove_old_files
    # early-returns on keep_number <= 0), which is required so any listed
    # checkpoint stays resumable. The state property defaults to 0.
    keep_states = int(getattr(state, "keep_states", 0))
    # Explicit per-frame save checkpoints, comma-separated solver 0-based
    # frame indices. The solver's SimArgs.checkpoints parses this string
    # and writes a resumable state at each listed frame, independent of the
    # auto-save cadence. Empty string when the artist listed no frames.
    checkpoints = ",".join(
        str(f) for f in state.convert_save_checkpoint_frames_to_remote()
    )
    has_shell_type = any(
        group.object_type == "SHELL"
        for group in iterate_object_groups(scene)
        if group.active
    )
    use_inactive_momentum = has_shell_type and int(state.inactive_momentum_frames) > 0

    # Z-up (Blender) -> Y-up (solver): (x, y, z) -> (x, z, -y)
    wind_force = _swap_axes(_normalize_and_scale(state.wind_direction, state.wind_strength))

    scene_params = {
        "dt": np.float32(state.step_size),
        "min-newton-steps": int(state.min_newton_steps),
        "air-density": np.float32(state.air_density),
        "air-friction": np.float32(state.air_friction),
        "friction-mode": str(state.friction_mode).lower(),
        "precond": "schwarz" if state.precond == "SCHWARZ" else "block-jacobi",
        "schwarz-levels": 1 if state.schwarz_levels == "LEVEL_1" else 2,
        "gravity": _swap_axes(state.gravity_3d),
        "wind": wind_force,
        "frames": frame_count - 1,  # Blender 1..N → remote 0..N-1
        "fps": fps,
        "csrmat-max-nnz": int(state.contact_nnz),
        "isotropic-air-friction": np.float32(state.vertex_air_damp),
        "fix-xz": np.float32(state.fix_xz),
        "world-scaling": np.float32(state.world_scaling),
        "auto-save": auto_save,
        "keep-states": keep_states,
        "checkpoints": checkpoints,
        "line-search-max-t": np.float32(state.line_search_max_t),
        "constraint-ghat": np.float32(state.constraint_ghat),
        "cg-max-iter": int(state.cg_max_iter),
        "cg-tol": np.float32(state.cg_tol),
        "include-face-mass": bool(state.include_face_mass),
        "disable-contact": bool(state.disable_contact),
        "save-state-on-finish": bool(state.save_state_on_finish),
    }

    if use_inactive_momentum:
        scene_params["inactive-momentum"] = float(state.inactive_momentum_frames) / fps

    # Stitch stiffness is per-object now: each group emits its own
    # "stitch-stiffness" in _encode_group_params (applied to that object's
    # loose-edge stitches), and each merge pair carries its own stiffness in
    # the cross_stitch payload. No global scene-level stitch stiffness.

    return scene_params


_ANGULAR_PCA_INDEX = {"PC1": 0, "PC2": 1, "PC3": 2}
_ANGULAR_WORLD_VECTOR = {
    "X": (1.0, 0.0, 0.0),
    "Y": (0.0, 1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
}


def _angular_axis_blender_vector(kf):
    """Blender-space spin axis for a fixed-axis (World X/Y/Z or Custom)
    velocity keyframe. Principal-axis modes (PC1-3) are resolved by the
    solver instead and never reach here."""
    return _ANGULAR_WORLD_VECTOR.get(kf.angular_axis, tuple(kf.angular_axis_custom))


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
                "stitch-stiffness",
                "deformation-damping",
                "contact-gap",
                "contact-offset",
                "plasticity",
                "plasticity-threshold",
                "model",
                "velocity",
                "velocity-schedule",
                "angular-velocity-schedule",
                "angular-velocity-world-schedule",
                "collision-windows",
                "ftetwild",
            ],
            "SHELL": [
                "density",
                "young-mod",
                "poiss-rat",
                "friction",
                "stitch-stiffness",
                "deformation-damping",
                "bending-damping",
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
                "angular-velocity-schedule",
                "angular-velocity-world-schedule",
                "collision-windows",
            ],
            "ROD": [
                "density",
                "young-mod",
                "friction",
                "stitch-stiffness",
                "deformation-damping",
                "bending-damping",
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
            "SAND": [
                "sand-particle-mass",
                "sand-friction",
                # grain radius is sent via contact-offset (the grain's skin),
                # so there is no separate sand-grain-radius solver param.
                "contact-gap",
                "contact-offset",
                "velocity",
                "velocity-schedule",
                "collision-windows",
            ],
            "PDRD": [
                "density",
                "friction",
                "contact-gap",
                "contact-offset",
                "model",
                "hinge",
                "velocity",
                "velocity-schedule",
                "angular-velocity-schedule",
                "angular-velocity-world-schedule",
                "collision-windows",
            ],
        }
        model_map = {
            "ARAP": "arap",
            "STABLE_NEOHOOKEAN": "snhk",
            "BARAFF_WITKIN": "baraff-witkin",
            "PDRD": "pdrd",
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
        elif group.object_type == "PDRD":
            model = "PDRD"
        else:
            model = "N/A"
        # strain_limit_percent is a percentage (UI); the solver wants the fraction.
        strain_limit = (
            np.float32(group.strain_limit_percent / 100.0)
            if group.enable_strain_limit
            else 0.0
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
        elif group.object_type == "PDRD":
            # Volumetric density; mass is density times enclosed
            # mesh volume, distributed over surface vertices by area
            # weighting downstream. Young/Poisson are placeholders so
            # the per-face param expansion stays key-compatible with
            # shells; the solver gates on `Model::Pdrd` and never
            # consumes them.
            density = np.float32(group.pdrd_density)
            young_modulus = np.float32(0.0)
            poisson_ratio = np.float32(0.0)
        else:  # STATIC
            density = np.float32(1000.0)  # Default density for static
            young_modulus = np.float32(100000.0)  # Default young modulus for static
            poisson_ratio = np.float32(0.35)  # Default Poisson ratio for static

        # The solver consumes "young-mod" as a density-normalized value (Pa/rho).
        # When the group's field is instead a true Young's modulus in pascals
        # (young_mod_density_normalized off), normalize it here by dividing by
        # density. When on (default), the field is already Pa/rho and is sent
        # unchanged. density is guaranteed > 0 (UI min 0.01), so the division is
        # safe. STATIC uses a fixed placeholder stiffness, so the flag does not
        # apply there.
        if group.object_type in ("SOLID", "SHELL", "ROD") and not group.young_mod_density_normalized:
            young_modulus = np.float32(young_modulus / density)

        if group.object_type == "SAND":
            # A sand grain's physical radius IS its contact skin, so the grain
            # radius is sent as the contact-offset. The contact gap (barrier
            # activation distance) is the user-set value on top of that skin.
            # The radius is the locked, seeding-derived value (sand_seeded_radius)
            # so the contact skin matches the non-overlapping seed spacing.
            from ...models.groups import sand_seeded_radius

            # The grain radius (sent as the contact skin/offset) is a geometric
            # length tied to the seed spacing, and the grain seed positions are
            # scaled by world_scaling in the solver, so scale the radius to match
            # (otherwise grains overlap). The contact gap on top is a world-space
            # distance too, so scale it by world_scaling for the same reason.
            contact_gap_value = group.contact_gap * state.world_scaling
            contact_offset_value = sand_seeded_radius(group) * state.world_scaling
        elif group.use_group_bounding_box_diagonal:
            # Relative gaps/offsets are a fraction of the mesh size, so they scale
            # with world_scaling (the solver shrinks the mesh by the same factor).
            bbox_diagonal = compute_group_bounding_box_diagonal(group) * state.world_scaling
            contact_gap_value = bbox_diagonal * group.contact_gap_rat
            contact_offset_value = bbox_diagonal * group.contact_offset_rat
        else:
            # Absolute gaps/offsets are authored as world-space distances. The
            # solver shrinks the mesh by world_scaling on ingest, so scale these
            # by the same factor to keep them a fixed physical size relative to
            # the geometry (consistent with collider thickness, which the solver
            # also scales by world_scaling).
            contact_gap_value = group.contact_gap * state.world_scaling
            contact_offset_value = group.contact_offset * state.world_scaling
        group.computed_contact_gap = contact_gap_value
        group.computed_contact_offset = contact_offset_value

        # A captured pull-pin deformation drives a time-varying rest shape
        # (decoded as per-frame inv_rest). Plasticity also mutates the rest
        # shape, so the two cannot coexist: the streamed rest shape would
        # clobber the plastic creep every frame. Drop plasticity for such a
        # group and warn, so the rest-shape capture (the explicit user action)
        # wins deterministically.
        has_rest_shape_capture = (
            group.object_type in ("SOLID", "SHELL")
            and any(
                p.use_pull and getattr(p, "has_captured_anim", False)
                for p in group.pin_vertex_groups
            )
        )
        plasticity_on = group.enable_plasticity and not has_rest_shape_capture
        bend_plasticity_on = group.enable_bend_plasticity and not has_rest_shape_capture
        if has_rest_shape_capture and (
            group.enable_plasticity or group.enable_bend_plasticity
        ):
            print(
                f"[ppf-cts] warning: group '{group.name}' has both a captured "
                "pull-pin deformation and plasticity enabled. Plasticity is "
                "ignored for this group; the captured rest shape takes over."
            )

        params = {
            "model": str(model_map.get(model, "unknown")),
            "density": density,
            "young-mod": young_modulus,
            "poiss-rat": poisson_ratio,
            "friction": np.float32(group.friction),
            # SAND (granular) material params; the strip loop below keeps
            # them only for SAND groups via the "SAND" allowlist entry.
            # Particle mass is authored in grams on the UI and shipped to the
            # solver in kilograms (SI), like the rest of the solver's units.
            "sand-particle-mass": np.float32(group.sand_particle_mass * 1e-3),
            "sand-friction": np.float32(group.sand_friction),
            "stitch-stiffness": np.float32(group.stitch_stiffness),
            "deformation-damping": np.float32(group.deformation_damping),
            "bending-damping": np.float32(group.bending_damping),
            "contact-gap": np.float32(contact_gap_value),
            "contact-offset": np.float32(contact_offset_value),
            "bend": np.float32(group.bend),
            "shrink": np.float32(group.shrink),
            "shrink-x": np.float32(group.shrink_x),
            "shrink-y": np.float32(group.shrink_y),
            "strain-limit": strain_limit,
            "pressure": np.float32(group.inflate_pressure) if group.enable_inflate else np.float32(0.0),
            "plasticity": np.float32(group.plasticity) if plasticity_on else np.float32(0.0),
            "plasticity-threshold": np.float32(group.plasticity_threshold) if plasticity_on else np.float32(0.0),
            "bend-plasticity": np.float32(group.bend_plasticity) if bend_plasticity_on else np.float32(0.0),
            "bend-plasticity-threshold": np.float32(group.bend_plasticity_threshold) if bend_plasticity_on else np.float32(0.0),
            "bend-rest-from-geometry": np.float32(1.0 if group.bend_rest_angle_source == "FROM_GEOMETRY" else 0.0),
            "length-factor": np.float32(group.length_factor),
            # Per-object dicts key on UUID so the decoder looks up the
            # right entry even if the Blender object was renamed between
            # transfer and simulation. pin_config already uses UUID.
            # Translational overwrite is emitted only for keyframes whose
            # "Enable Translational Velocity Overwrite" box is checked, so a
            # pure-spin keyframe does not zero the translation.
            "velocity": {
                assigned.uuid: next(
                    # Initial velocity (frame 1) is written to vel.bin, which the
                    # solver scales by world_scaling on ingest (like geometry), so
                    # it must NOT be scaled here too (that double-scales to ws^2).
                    # The frame>1 schedule below is a dyn_param the solver does
                    # not scale, so it IS scaled here.
                    (_swap_axes(_normalize_and_scale(kf.direction, kf.speed))
                     for kf in assigned.velocity_keyframes
                     if kf.frame == 1 and kf.enable_translational),
                    np.array([0.0, 0.0, 0.0], dtype=np.float32),
                )
                for assigned in group.assigned_objects
                if assigned.included
            },
            "velocity-schedule": {
                assigned.uuid: [
                    (float(kf.frame - 1) / fps, _swap_axes(_normalize_and_scale(kf.direction, kf.speed * state.world_scaling)))
                    for kf in assigned.velocity_keyframes
                    if kf.frame > 1 and kf.enable_translational
                ]
                for assigned in group.assigned_objects
                if assigned.included
            },
            # Principal-axis angular (spin) overwrite. ALL keyframes (incl.
            # frame 1 -> t=0) go through the schedule so the spin axis is
            # resolved dynamically by the solver from the live geometry; each
            # entry is (t, pca_index, speed_rad). No axis-swap: a pca_index
            # carries no frame, and the axis is resolved in solver space.
            # Angular overwrite splits by axis mode. Principal axes (PC1-3)
            # carry a pca_index that the solver resolves to a world axis from
            # the live geometry each firing. World X/Y/Z and Custom are fixed
            # directions, pre-resolved here into a world-space ω vector (axis
            # swapped Blender->solver, scaled by the speed in rad/s).
            "angular-velocity-schedule": {
                assigned.uuid: [
                    (
                        float(kf.frame - 1) / fps,
                        _ANGULAR_PCA_INDEX[kf.angular_axis],
                        float(np.radians(kf.angular_speed)),
                    )
                    for kf in assigned.velocity_keyframes
                    if kf.enable_angular and kf.angular_speed != 0.0
                    and kf.angular_axis in _ANGULAR_PCA_INDEX
                ]
                for assigned in group.assigned_objects
                if assigned.included
            },
            "angular-velocity-world-schedule": {
                assigned.uuid: [
                    (
                        float(kf.frame - 1) / fps,
                        _swap_axes(_normalize_and_scale(
                            _angular_axis_blender_vector(kf),
                            np.radians(kf.angular_speed),
                        )),
                    )
                    for kf in assigned.velocity_keyframes
                    if kf.enable_angular and kf.angular_speed != 0.0
                    and kf.angular_axis not in _ANGULAR_PCA_INDEX
                ]
                for assigned in group.assigned_objects
                if assigned.included
            },
            # Per-object tetrahedralizer kwargs keyed by UUID (mirrors
            # "velocity"). Empty for default fTetWild objects; the decoder
            # peeks this at populate time to pick each mesh's backend.
            "ftetwild": {
                assigned.uuid: kw
                for assigned in group.assigned_objects
                if assigned.included and (kw := _encode_obj_tet_kwargs(assigned))
            },
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
            # PDRD hinge: per-UUID principal-axis index (0/1/2), set per
            # assigned object (each body can be hinged on its own axle).
            # Only hinge-enabled objects appear; the decoder turns each entry
            # into an Object.hinge(axis) call. Empty = no hinged bodies.
            "hinge": {
                assigned.uuid: int(assigned.pdrd_hinge_axis)
                for assigned in group.assigned_objects
                if assigned.included
                and getattr(assigned, "pdrd_hinge_enable", False)
            } if group.object_type == "PDRD" else {},
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


def _encode_cross_stitch(context):
    scene = context.scene
    from ...mesh_ops.merge_ops import cleanup_stale_merge_pairs
    cleanup_stale_merge_pairs(scene)
    state = get_addon_data(scene).state
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
        # Every supported pair is a soft, mass-scaled stitch (including
        # shell-shell and rod-rod, which previously hard-merged DOFs).
        try:
            data = json.loads(pair.cross_stitch_json)
        except json.JSONDecodeError:
            continue
        if data:
            if not data.get("source_uuid") or not data.get("target_uuid"):
                continue
            # Upgrade legacy 4-wide rows [si, t0, t1, t2] / [1, a, b, c] to
            # the 6-wide barycentric-barycentric layout (degenerate source
            # [si, si, si] / [1, 0, 0]) so pre-migration scenes still build.
            # Legacy rows carry no source_points, so a SOLID source is
            # dropped by the decoder until re-snapped, which is safer than
            # emitting a mis-mapped stitch.
            ind = data.get("ind")
            w = data.get("w")
            if ind and w and len(ind[0]) == 4:
                data["ind"] = [[r[0], r[0], r[0], r[1], r[2], r[3]] for r in ind]
                data["w"] = [[1.0, 0.0, 0.0, x[1], x[2], x[3]] for x in w]
            source_points = data.get("source_points")
            if source_points:
                data["source_points"] = [_to_solver(point) for point in source_points]
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

    fps = resolve_fps(state)

    # Evaluate the whole param tree at frame 1, matching the data encoder
    # (_build_obj_data). The per-group bounding-box diagonal that scales
    # contact-gap / contact-offset reads live mesh state, so without this
    # the fingerprint would track the artist's current timeline frame and
    # drift from what the server stored at upload. The inner keyframe
    # samplers (_encode_pin_config / _encode_dyn_params) save and restore
    # their own frame, so nesting them here is safe.
    with evaluate_at_frame_one(context):
        scene_params = _encode_scene_params(context, state, fps)
        group_params = _encode_group_params(context, groups, state, fps)
        pin_config = _encode_pin_config(context, groups, state, fps)
        cross_stitch = _encode_cross_stitch(context)
        dyn_param = _encode_dyn_params(state, fps)

    result = {
        "scene": scene_params,
        "group": group_params,
        "pin_config": pin_config,
    }
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

    Avoids the build-twice cost of calling ``encode_param`` then
    ``compute_param_hash`` separately. ``_build_param_dict`` produces a fixed
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
