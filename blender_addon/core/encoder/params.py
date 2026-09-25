# File: encoder/params.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import hashlib
import json

import numpy as np

from ...models.groups import get_addon_data, iterate_object_groups
from ...models.intersection_allowances import (
    EXISTING_ALLOWANCE,
    INTER_GROUP_ALLOWANCE,
    INTER_OBJECT_ALLOWANCE,
    SELF_ALLOWANCE,
    allowance_applies_to_all,
    allowance_enabled,
    allowed_object_uuids,
)
from ...models.material_maps import rest_shape_plasticity_conflict, to_solver_value
from . import (
    _normalize_and_scale,
    _swap_axes,
    _to_solver,
    check_frame_window,
    frame_to_time,
    resolve_solver_fps,
    resolve_start_frame,
    resolve_time_scale,
    resolve_world_scaling,
    solver_gravity,
    solver_wind,
)
from .curve_refusal import refuse_unsampled_curves
from .dyn import _encode_dyn_params, _encode_invisible_colliders
from .scene_anim import encode_scene_param_anim
from .param_anim import encode_param_anim
from .material_maps import encode_material_maps
from .mesh import compute_group_bounding_box_diagonal, evaluate_at_start_frame
from .pin import _encode_pin_config


_FTETWILD_FLOAT_FIELDS = ("edge_length_fac", "epsilon", "stop_energy")
_FTETWILD_INT_FIELDS = ("num_opt_iter",)
_FTETWILD_BOOL_FIELDS = ("optimize", "simplify", "coarsen")


def _solver_value_or_zero(group, key: str, ui_value) -> float:
    """`key` in solver units, or 0.0 when the group's feature toggle is off.

    Zero is what a closed gate means for a static parameter: the solver builds
    the group with the term switched off. A map target has no such reading, so
    `encode_material_maps` refuses one instead of substituting a value.
    """
    value = to_solver_value(group, key, ui_value)
    return 0.0 if value is None else value


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


def _encode_soft_constraint(group) -> dict:
    """Map each included object of a STATIC group to its pin spring stiffness.

    Empty when the group holds its collider exactly, which is the default and
    keeps the param hash of untouched scenes unchanged.
    """
    if not getattr(group, "enable_soft_constraint", False):
        return {}
    stiffness = float(group.soft_constraint_stiffness)
    if not stiffness > 0.0:
        # The solver reads a pull weight of zero as "this pin is a hard fix",
        # so forwarding it would hand back the exact constraint the group asked
        # to drop, with nothing in the UI or the log to say so. The UI minimum
        # does not cover the MCP or material-profile paths that also write it.
        raise ValueError(
            f"STATIC group '{group.name}' has soft constraints enabled with "
            f"stiffness {stiffness}, which must be strictly positive. Zero is "
            "how the solver spells an exact pin, so it would silently keep the "
            "collider rigid. Raise Stiffness, or uncheck Apply Soft "
            "Constraints."
        )
    return {
        assigned.uuid: stiffness
        for assigned in group.assigned_objects
        if assigned.included
    }


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

    # constraint-ghat is a scene-unit length the solver reads raw, so World
    # Scaling is applied to it here; gravity and wind are physical constants
    # and are not scaled (see resolve_world_scaling).
    world_scaling = resolve_world_scaling(state)

    scene_params = {
        "dt": np.float32(state.step_size),
        "min-newton-steps": int(state.min_newton_steps),
        "air-density": np.float32(state.air_density),
        "air-friction": np.float32(state.air_friction),
        "friction-mode": str(state.friction_mode).lower(),
        "precond": "schwarz" if state.precond == "SCHWARZ" else "block-jacobi",
        "schwarz-levels": 1 if state.schwarz_levels == "LEVEL_1" else 2,
        "gravity": solver_gravity(state.gravity_3d),
        "wind": solver_wind(
            state.wind_direction, state.wind_strength, "Wind",
        ),
        # A count, so the starting frame does not enter: the solve always
        # produces remote frames 0..N-1, which playback places on Blender
        # frames start..start+N-1.
        "frames": frame_count - 1,
        "fps": fps,
        "csrmat-max-nnz": int(state.contact_nnz),
        "isotropic-air-friction": np.float32(state.vertex_air_damp),
        "fix-xz": np.float32(state.fix_xz),
        "world-scaling": np.float32(state.world_scaling),
        "auto-save": auto_save,
        "keep-states": keep_states,
        "checkpoints": checkpoints,
        "line-search-max-t": np.float32(state.line_search_max_t),
        "constraint-ghat": np.float32(state.constraint_ghat * world_scaling),
        "cg-max-iter": int(state.cg_max_iter),
        "cg-tol": np.float32(state.cg_tol),
        "include-face-mass": bool(state.include_face_mass),
        "disable-contact": bool(state.disable_contact),
        "save-state-on-finish": bool(state.save_state_on_finish),
    }

    if use_inactive_momentum:
        scene_params["inactive-momentum"] = float(state.inactive_momentum_frames) / fps

    # Stitch stiffness is per object: each SOLID and SHELL group emits its own
    # "stitch-stiffness" in _encode_group_params (applied to that object's
    # loose-edge stitches), and each merge pair carries its own stiffness in
    # the cross_stitch payload. There is no scene-level stitch stiffness.

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


def _initial_translational_velocity(assigned, start_frame, time_scale):
    """The translational velocity an object enters the solve with.

    The LAST translational keyframe at or before the starting frame wins: that
    is the value in effect at simulated time zero. Keys strictly after it go to
    "velocity-schedule" instead. Matching on the starting frame exactly would
    drop every key authored during a lead-in the solve does not cover, leaving
    the object at rest with no warning while a spin key on the same row (which
    clamps to t=0) still applied. At the default starting frame of 1 this
    selects the frame-1 key, since ``VelocityKeyframe.frame`` has ``min=1``.

    The speed is authored per ANIMATION second like every later key, so it is
    multiplied by *time_scale* here exactly as the schedule's keys are (see
    ``resolve_time_scale``); nothing downstream applies Time Scale to it.

    Written to vel.bin, which the solver scales by world_scaling on ingest
    (like geometry), so it must NOT be scaled here too (that double-scales to
    ws^2). The schedule is a dyn_param the solver does not scale, so it IS
    scaled there.
    """
    chosen = None
    for kf in assigned.velocity_keyframes:
        if not kf.enable_translational or kf.frame > start_frame:
            continue
        if chosen is None or kf.frame >= chosen.frame:
            chosen = kf
    if chosen is None:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return _swap_axes(_normalize_and_scale(
        chosen.direction, chosen.speed * time_scale,
        f"The velocity keyframe at frame {chosen.frame} of '{assigned.name}'",
    ))


def _encode_collision_windows(group, assigned, fps, start_frame):
    """One object's collision windows as ``[(t_start, t_end), ...]`` seconds.

    Each window is checked by ``check_frame_window`` first: one that ends
    before it starts, or starts before the starting frame, is refused naming
    the object and both frames. Shipped as authored, a zero-length window
    never admits contact (the solver's interval is half-open), and one cut
    off at the starting frame would end contact earlier than the artist set
    it to.
    """
    windows = []
    for cw in assigned.collision_windows:
        check_frame_window(
            f"Object '{assigned.name}' in group '{group.name}': its "
            "collision window",
            cw.frame_start, cw.frame_end, start_frame,
        )
        windows.append((
            frame_to_time(cw.frame_start, fps, start_frame),
            frame_to_time(cw.frame_end, fps, start_frame),
        ))
    return windows


def _encode_lock_translation_axis(assigned) -> list[float]:
    """Solver-space, unit-length Lock Translation axis for one object.

    A world-space direction only (no world scaling: a line direction has
    no length to scale). Validated finite and non-zero BEFORE the axis
    swap and normalize, so a degenerate axis fails loudly here rather
    than reaching the solver as NaN or a silently-disabled zero vector.
    Raises fail loud: an enabled Lock Translation with a zero or non-finite
    axis is a scene-authoring error, not a "just disable it" case, since the
    UI already warns and defaults to a non-zero axis.
    """
    axis = np.array(
        [float(assigned.lock_translation_axis[i]) for i in range(3)],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(axis)):
        raise ValueError(
            f"Object '{assigned.name}': Lock Translation axis must be finite; "
            f"got {tuple(assigned.lock_translation_axis)!r}"
        )
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise ValueError(
            f"Object '{assigned.name}': Lock Translation is enabled with a "
            "zero-length axis. Set a non-zero direction or disable Lock "
            "Translation."
        )
    return _swap_axes((axis / norm).tolist())


def _encode_lock_rotation_axis(assigned) -> list[float]:
    """Solver-space, unit-length Lock Rotation axis for one object.

    A world-space direction only (no world scaling: a rotation axis has
    no length to scale). Validated finite and non-zero BEFORE the axis
    swap and normalize, so a degenerate axis fails loudly here rather
    than reaching the solver as NaN or a silently-disabled zero vector.
    Raises fail loud: an enabled Lock Rotation with a zero or non-finite
    axis is a scene-authoring error, not a "just disable it" case, since
    the UI already warns and defaults to a non-zero axis. Mirrors
    `_encode_lock_translation_axis` above; the two features are encoded
    independently of each other.
    """
    axis = np.array(
        [float(assigned.lock_rotation_axis[i]) for i in range(3)],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(axis)):
        raise ValueError(
            f"Object '{assigned.name}': Lock Rotation axis must be finite; "
            f"got {tuple(assigned.lock_rotation_axis)!r}"
        )
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise ValueError(
            f"Object '{assigned.name}': Lock Rotation is enabled with a "
            "zero-length axis. Set a non-zero direction or disable Lock "
            "Rotation."
        )
    return _swap_axes((axis / norm).tolist())


def _encode_force_field(context, groups, state, start_frame, frame_count, fps):
    """The scene's force fields and exact script, or ``None`` for neither.

    Sampled into grids by ``core.force_field.encode``, over the box the
    simulated objects occupy at the starting frame unless a Domain is set.
    """
    from ..force_field import encode
    from ..uuid_registry import resolve_assigned

    dynamic = []
    for group in groups:
        if str(group.object_type) == "STATIC":
            continue
        for assigned in group.assigned_objects:
            if assigned.included:
                obj = resolve_assigned(assigned)
                if obj is not None:
                    dynamic.append(obj)
    position_by_uuid = {group.uuid: i for i, group in enumerate(groups)}
    return encode(context, state, dynamic, start_frame, frame_count, fps,
                  position_by_uuid)


def _encode_intersection_allowance(group, spec, object_uuids):
    """One intersection allowance, as the decoder takes it.

    The allowance is per OBJECT in the solver: the frontend reads each
    object's own material and resolves one policy byte per vertex. So a
    narrowed allowance needs no new mechanism below this line, only a value
    the decoder can apply per object.

    Returns a plain float while the allowance reaches every object of the
    group, which keeps the payload of a scene that never narrows one exactly
    what it was before narrowing existed. Once narrowed it returns
    ``{uuid: 1.0 | 0.0}`` over every INCLUDED object, stating both answers
    rather than leaving the unnamed objects to a default, so the payload says
    what each object was given instead of what it was not.
    """
    if not allowance_enabled(group, spec):
        return np.float32(0.0)
    if allowance_applies_to_all(group, spec):
        return np.float32(1.0)
    allowed = allowed_object_uuids(group, spec)
    return {
        obj_uuid: np.float32(1.0 if obj_uuid in allowed else 0.0)
        for obj_uuid in object_uuids
    }


def group_contact_lengths(group, scale: float = 1.0) -> tuple[float, float]:
    """``(contact gap, contact offset)`` of ``group``, times ``scale``.

    The one resolver for these two lengths: the encoder asks with the world
    scaling (the solver shrinks the mesh by it on ingest, so every branch here
    is a world-space length scaled by the same factor), and the snap operator
    and the fetch-time stitch closure ask with 1.0, in Blender units. Nothing
    stores a copy, so no reader can see a value an earlier encode left behind.

    A SAND grain's physical radius IS its contact skin, so its offset is the
    locked seeding radius (``sand_seeded_radius``); a group sized by its
    bounding box takes both as fractions of the diagonal.
    """
    if group.object_type == "SAND":
        from ...models.groups import sand_seeded_radius

        return group.contact_gap * scale, sand_seeded_radius(group) * scale
    if group.use_group_bounding_box_diagonal:
        bbox_diagonal = compute_group_bounding_box_diagonal(group) * scale
        return (bbox_diagonal * group.contact_gap_rat,
                bbox_diagonal * group.contact_offset_rat)
    return group.contact_gap * scale, group.contact_offset * scale


def _encode_group_params(context, groups, state, fps, start_frame):
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
                "lock-translation",
                "lock-rotation",
                "lock-rotation-prohibit-axis",
                "lock-all-translations",
                "lock-all-rotations",
                "allow-self-intersection",
                "allow-inter-object-intersection",
                "allow-inter-group-intersection",
                "allow-existing-intersection",
                "force-field-weight",
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
                "bend-warp",
                "bend-weft",
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
                "lock-translation",
                "lock-rotation",
                "lock-rotation-prohibit-axis",
                "lock-all-translations",
                "lock-all-rotations",
                "allow-self-intersection",
                "allow-inter-object-intersection",
                "allow-inter-group-intersection",
                "allow-existing-intersection",
                "force-field-weight",
            ],
            "ROD": [
                # No stitch-stiffness: a rod carries no loose-edge stitch (its
                # edges are the rod), and a merge pair ships its own.
                "density",
                "young-mod",
                "friction",
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
                "lock-translation",
                "lock-rotation",
                "lock-rotation-prohibit-axis",
                "lock-all-translations",
                "lock-all-rotations",
                "allow-self-intersection",
                "allow-inter-object-intersection",
                "allow-inter-group-intersection",
                "allow-existing-intersection",
                "force-field-weight",
            ],
            "STATIC": [
                "contact-gap",
                "contact-offset",
                "friction",
                "soft-constraint",
                "allow-self-intersection",
                "allow-inter-object-intersection",
                "allow-inter-group-intersection",
                "allow-existing-intersection",
            ],
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
                "lock-translation",
                "lock-rotation",
                "lock-rotation-prohibit-axis",
                "lock-all-translations",
                "lock-all-rotations",
                "allow-self-intersection",
                "allow-inter-object-intersection",
                "allow-inter-group-intersection",
                "force-field-weight",
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
                "lock-translation",
                "lock-rotation",
                "lock-rotation-prohibit-axis",
                "lock-all-translations",
                "lock-all-rotations",
                "allow-self-intersection",
                "allow-inter-object-intersection",
                "allow-inter-group-intersection",
                "allow-existing-intersection",
                "force-field-weight",
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
            # The SHELL picker does not offer Stable NeoHookean, yet a
            # `.blend` or profile saved with it still loads the identifier.
            # Running any other model in its place would ship something the
            # artist did not author, so it is refused by name.
            from ...models.groups import withdrawn_shell_model_refusal

            refusal = withdrawn_shell_model_refusal(group)
            if refusal is not None:
                raise ValueError(refusal)
        elif group.object_type == "ROD":
            model = group.rod_model
        elif group.object_type == "PDRD":
            model = "PDRD"
        else:
            model = "N/A"
        # `to_solver_value` carries the percent-to-fraction conversion and both
        # conditions that switch strain limiting off, so the static value, a
        # sampled keyframe and a map target agree on all three.
        strain_limit = np.float32(
            _solver_value_or_zero(group, "strain-limit", group.strain_limit_percent)
        )

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
        # `to_solver_value` owns that conversion, so a map target and a sampled
        # keyframe reach the solver in the same units this value does. PDRD and
        # STATIC carry a placeholder stiffness and no density property, so the
        # conversion leaves them alone.
        young_modulus = np.float32(
            _solver_value_or_zero(group, "young-mod", young_modulus)
        )

        if group.object_type == "SAND" and allowance_enabled(
                group, EXISTING_ALLOWANCE):
            # Refused rather than dropped: a SAND group is never offered this
            # allowance, so one that carries it came from a script or an older
            # file, and shipping without it would start a scene the artist
            # expects to tolerate an overlap that the solver will refuse.
            raise ValueError(
                f"Group '{group.name}': Allow Existing Intersections is not "
                "supported on sand, whose grains overlapping at the start are "
                "not exempted from contact. Turn it off on this group."
            )
        if group.object_type == "SAND":
            # The group ships one grain radius as its contact offset
            # (`group_contact_lengths`), so every grain must share it.
            from ...models.groups import sand_radius_conflict

            radii = sand_radius_conflict(group)
            if radii is not None:
                listed = ", ".join(f"'{name}' at {radius:g}" for name, radius in radii)
                raise ValueError(
                    f"Group '{group.name}': its grains were converted at "
                    f"different radii ({listed}), and a SAND group is solved "
                    "at one. Convert them at one radius or split the group."
                )

        # World-space lengths, scaled like the geometry: the solver shrinks
        # the mesh by world_scaling on ingest (and collider thickness with it).
        contact_gap_value, contact_offset_value = group_contact_lengths(
            group, state.world_scaling,
        )

        # A tracked captured deformation streams the rest shape every frame and
        # plasticity creeps it every step, so a group cannot carry both:
        # refused here by name rather than one of them dropped in silence.
        conflict = rest_shape_plasticity_conflict(group)
        if conflict is not None:
            raise ValueError(
                f"Group '{group.name}': pin '{conflict.name}' tracks its "
                "captured rest shape (Track Rest-Pose Deformation) and the "
                "group has Plasticity on. Both rewrite the rest shape, so turn "
                "one of them off."
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
            # Directional bending stiffnesses added on top of `bend`. Both
            # default to 0.0, which adds nothing, so a group that leaves them
            # alone reaches the solver exactly as it did before.
            "bend-warp": np.float32(group.bend_warp),
            "bend-weft": np.float32(group.bend_weft),
            # Intersection allowances (issue #138). Float-encoded booleans,
            # like bend-rest-from-geometry. The pairs they name get no contact
            # force, no CCD filter and no report at the scene-build check or
            # at any solver intersection scan, so they pass through freely.
            #
            # A scalar when the allowance reaches every object of the group,
            # a per-uuid dict when it has been narrowed to a subset. See
            # `_encode_intersection_allowance`.
            "allow-self-intersection": _encode_intersection_allowance(
                group, SELF_ALLOWANCE, object_uuids),
            "allow-inter-object-intersection": _encode_intersection_allowance(
                group, INTER_OBJECT_ALLOWANCE, object_uuids),
            "allow-inter-group-intersection": _encode_intersection_allowance(
                group, INTER_GROUP_ALLOWANCE, object_uuids),
            # Allow Existing Intersections: the pairs this object STARTS
            # intersecting with are linked at the scene-build check and stay
            # out of contact for the run; nothing else is exempted.
            "allow-existing-intersection": _encode_intersection_allowance(
                group, EXISTING_ALLOWANCE, object_uuids),
            # The scene force field's scale on this group's objects (issues
            # #151 and #114): 1.0 applies it as authored, 0.0 opts out.
            "force-field-weight": np.float32(group.force_field_weight),
            "shrink": np.float32(group.shrink),
            "shrink-x": np.float32(group.shrink_x),
            "shrink-y": np.float32(group.shrink_y),
            "strain-limit": strain_limit,
            "pressure": np.float32(_solver_value_or_zero(group, "pressure", group.inflate_pressure)),
            "plasticity": np.float32(_solver_value_or_zero(group, "plasticity", group.plasticity)),
            "plasticity-threshold": np.float32(_solver_value_or_zero(group, "plasticity-threshold", group.plasticity_threshold)),
            "bend-plasticity": np.float32(_solver_value_or_zero(group, "bend-plasticity", group.bend_plasticity)),
            "bend-plasticity-threshold": np.float32(_solver_value_or_zero(group, "bend-plasticity-threshold", group.bend_plasticity_threshold)),
            "bend-rest-from-geometry": np.float32(1.0 if group.bend_rest_angle_source == "FROM_GEOMETRY" else 0.0),
            "length-factor": np.float32(group.length_factor),
            # Per-object dicts key on UUID so the decoder looks up the
            # right entry even if the Blender object was renamed between
            # transfer and simulation. pin_config already uses UUID.
            # Translational overwrite is emitted only for keyframes whose
            # "Enable Translational Velocity Overwrite" box is checked, so a
            # pure-spin keyframe does not zero the translation.
            "velocity": {
                assigned.uuid: _initial_translational_velocity(
                    assigned, start_frame, resolve_time_scale(state),
                )
                for assigned in group.assigned_objects
                if assigned.included
            },
            "velocity-schedule": {
                assigned.uuid: [
                    (
                        frame_to_time(kf.frame, fps, start_frame),
                        _swap_axes(_normalize_and_scale(
                            kf.direction,
                            # Animation m/s -> physical (see resolve_time_scale).
                            kf.speed * state.world_scaling * resolve_time_scale(state),
                            f"The velocity keyframe at frame {kf.frame} of "
                            f"'{assigned.name}'",
                        )),
                    )
                    for kf in assigned.velocity_keyframes
                    if kf.frame > start_frame and kf.enable_translational
                ]
                for assigned in group.assigned_objects
                if assigned.included
            },
            # Principal-axis angular (spin) overwrite. ALL keyframes (incl.
            # the starting frame -> t=0) go through the schedule so the spin
            # axis is resolved dynamically by the solver from the live
            # geometry; each
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
                        max(0.0, frame_to_time(kf.frame, fps, start_frame)),
                        _ANGULAR_PCA_INDEX[kf.angular_axis],
                        # Animation rad/s -> physical (see resolve_time_scale).
                        float(np.radians(kf.angular_speed)) * resolve_time_scale(state),
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
                        max(0.0, frame_to_time(kf.frame, fps, start_frame)),
                        # Animation rad/s -> physical (see resolve_time_scale),
                        # as the principal-axis schedule above.
                        _swap_axes(_normalize_and_scale(
                            _angular_axis_blender_vector(kf),
                            np.radians(kf.angular_speed) * resolve_time_scale(state),
                            f"The angular velocity keyframe at frame {kf.frame} "
                            f"of '{assigned.name}'",
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
            # Spring stiffness holding each collider vertex to its animated
            # position, keyed by UUID (mirrors "ftetwild": the decoder peeks it
            # at populate time, since a pin's pull weight is not an object
            # param and cannot go through param.set). The group-level value is
            # broadcast per object. Absent means exact pins, so an unchecked
            # group sends nothing. Riding the PARAM payload rather than DATA is
            # deliberate: retuning the stiffness then re-sends kilobytes of
            # parameters instead of the whole mesh dataset.
            "soft-constraint": _encode_soft_constraint(group),
            "collision-windows": {
                assigned.uuid: _encode_collision_windows(
                    group, assigned, fps, start_frame,
                )
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
            # Lock Translation: per-UUID normalized world-space axis (solver
            # space, direction only, no world scaling), set per assigned
            # object. Only enabled objects appear; the decoder turns each
            # entry into an Object.lock_translation(*axis) call. An
            # all-axes lock names no axis and the axis encoder refuses a
            # zero-length one, so such an object is OMITTED here and
            # carried by "lock-all-translations" below: one lock state, one
            # spelling on the wire. Empty therefore means no object is
            # LINE-locked, which is not the same as no object being locked.
            "lock-translation": {
                assigned.uuid: _encode_lock_translation_axis(assigned)
                for assigned in group.assigned_objects
                if assigned.included
                and getattr(assigned, "lock_translation_enable", False)
                and not getattr(assigned, "lock_translation_all", False)
            },
            # Lock All Translations: per-UUID bool, set for every
            # translation-lock-enabled object. True pins the center of mass
            # to its initial POINT (three rows) and is the only entry such
            # an object has, since it appears in no axis dict; False is the
            # line lock whose axis sits in "lock-translation" above. A mode
            # rather than a direction, so it goes through neither the axis
            # swap nor world scaling. Empty = no object in this group has
            # Lock Translation enabled at all.
            "lock-all-translations": {
                assigned.uuid: bool(getattr(assigned, "lock_translation_all", False))
                for assigned in group.assigned_objects
                if assigned.included
                and getattr(assigned, "lock_translation_enable", False)
            },
            # Lock Rotation: per-UUID normalized world-space axis (solver
            # space, direction only, no world scaling), set per assigned
            # object. Independent of "lock-translation" above: an object
            # may appear in either dict, both, or neither. Only enabled
            # objects appear; the decoder turns each entry into an
            # Object.lock_rotation(*axis) call. An all-axes lock is
            # OMITTED here for the same reason it is omitted from
            # "lock-translation", so empty means no object is locked to a
            # single AXIS, not that every object rotates freely.
            "lock-rotation": {
                assigned.uuid: _encode_lock_rotation_axis(assigned)
                for assigned in group.assigned_objects
                if assigned.included
                and getattr(assigned, "lock_rotation_enable", False)
                and not getattr(assigned, "lock_rotation_all", False)
            },
            # Lock Rotation mode: per-UUID bool, set only for the objects
            # that appear in "lock-rotation" above. False (default) keeps
            # the axis a whitelist (only rotation about it is allowed);
            # True flips it to a blacklist (rotation about it is
            # forbidden, the perpendicular plane stays free instead).
            # The decoder passes this to Object.lock_rotation alongside
            # the axis from "lock-rotation", so an all-axes lock, which
            # has no axis to pair with, is omitted from both.
            "lock-rotation-prohibit-axis": {
                assigned.uuid: bool(assigned.lock_rotation_prohibit_axis)
                for assigned in group.assigned_objects
                if assigned.included
                and getattr(assigned, "lock_rotation_enable", False)
                and not getattr(assigned, "lock_rotation_all", False)
            },
            # Lock All Rotations: per-UUID bool, set for every
            # rotation-lock-enabled object. True forbids net rotation about
            # every axis (three rows) and is the only entry such an object
            # has, appearing in neither "lock-rotation" nor its mode dict;
            # False is one of the two per-axis modes those two carry. A mode
            # rather than a direction, so it goes through neither the axis
            # swap nor world scaling. Empty = no object in this group has
            # Lock Rotation enabled at all.
            "lock-all-rotations": {
                assigned.uuid: bool(getattr(assigned, "lock_rotation_all", False))
                for assigned in group.assigned_objects
                if assigned.included
                and getattr(assigned, "lock_rotation_enable", False)
            },
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
    """Every merge pair's stitch, or a refusal naming the pair that cannot ship.

    A pair ``merge_pair_problem`` finds fault with is refused, never skipped:
    a skipped pair is a seam the artist authored that silently fails to form.
    """
    from ...mesh_ops.merge_ops import merge_pair_problem, pair_label

    scene = context.scene
    state = get_addon_data(scene).state
    result = []
    for pair in state.merge_pairs:
        problem = merge_pair_problem(scene, pair)
        if problem is not None:
            raise ValueError(f"Merge pair {pair_label(pair)}: {problem}.")
        data = json.loads(pair.cross_stitch_json)
        # Upgrade 4-wide rows [si, t0, t1, t2] / [ws, a, b, c] to the 6-wide
        # barycentric-barycentric layout with a degenerate source
        # [si, si, si] / [1, 0, 0]. The validator refuses a 4-wide pair with a
        # SOLID side, which needs the points 4-wide rows never carried.
        ind = data["ind"]
        w = data["w"]
        if len(ind[0]) == 4:
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

    # A curve on an add-on property the encoder does not sample is refused
    # before anything is read. That is also what makes every setting below a
    # value that cannot move with the playhead unless it is sampled.
    refuse_unsampled_curves(scene)

    # Evaluate the whole param tree at the starting frame, matching the data
    # encoder (_build_obj_data). EVERYTHING the payload reads belongs inside
    # this block, not only the geometry-derived encodings (the per-group
    # bounding-box diagonal that scales contact-gap / contact-offset, a
    # material map's ATTRIBUTE source read off the evaluated mesh): the frame
    # rate, the starting frame, Time Scale, the invisible colliders and every
    # static setting are read here too, so the payload and its hash are the
    # same wherever the artist parked the playhead. Outside it, a value would
    # track the current timeline frame and drift from what the server stored at
    # upload. The inner keyframe samplers (_encode_pin_config /
    # _encode_dyn_params / the F-curve samplers) save and restore their own
    # frame, so nesting them here is safe.
    with evaluate_at_start_frame(context, state):
        # Solver-fps (Time Scale applied): every frame->seconds conversion
        # below, and the "fps" param itself, must use the scaled rate so the
        # whole schedule re-interprets time coherently.
        fps = resolve_solver_fps(state)
        time_scale = resolve_time_scale(state)
        start_frame = resolve_start_frame(state)
        # The block moved the playhead to the starting frame it resolved on
        # entry; resolving it again here must name the same frame, or the
        # starting frame itself depends on the playhead.
        assert scene.frame_current == start_frame, (
            f"the param encode is evaluating frame {scene.frame_current}, "
            f"but the starting frame resolves to {start_frame}"
        )
        scene_params = _encode_scene_params(context, state, fps)
        group_params = _encode_group_params(context, groups, state, fps, start_frame)
        pin_config = _encode_pin_config(context, groups, state, fps, start_frame)
        cross_stitch = _encode_cross_stitch(context)
        # Scene settings the artist keyframed, sampled from their own
        # F-curves. This is the authoring path; `_encode_dyn_params` reads the
        # addon's legacy keyframe list and stays only until every saved scene
        # has been migrated off it. A key present in both is taken from the
        # F-curve, because that is the one the artist can see on the timeline.
        dyn_param = _encode_dyn_params(state, fps, start_frame)
        dyn_param.update(
            encode_scene_param_anim(state, fps, start_frame, int(state.frame_count))
        )
        # Material sliders the artist keyframed, sampled across the solve's
        # frame range. Times are shared by the whole scene so the solver reads
        # one times.bin; the per-group values ride along with that group's
        # params. The maps are encoded first: a map keyed over time needs its
        # own times represented on the shared keyframe axis, which
        # `encode_param_anim` picks.
        spatial, map_schedules = encode_material_maps(
            context, groups, fps, start_frame
        )
        anim_times, anim_by_group = encode_param_anim(
            state, groups, fps, start_frame, int(state.frame_count), map_schedules
        )
        ic = _encode_invisible_colliders(state, fps, start_frame)
        force_field = _encode_force_field(
            context, groups, state, start_frame, int(state.frame_count), fps
        )

    result = {
        "scene": scene_params,
        # TOP-LEVEL on purpose, not inside scene_params: apply_to_session
        # forwards every scene key to session.param.set, which would demand
        # a params.rs whitelist entry and pollute param.toml. The decoder
        # reads this to convert authored animation rates (a SPIN op's
        # degrees per animation second in the DATA payload) to solver rates.
        "time_scale": time_scale,
        "group": group_params,
        "pin_config": pin_config,
    }
    if cross_stitch:
        result["cross_stitch"] = cross_stitch
    if dyn_param:
        result["dyn_param"] = dyn_param
    if force_field is not None:
        result["force_field"] = force_field

    if anim_times:
        result["param_anim_times"] = anim_times
        # `_encode_group_params` appends exactly one entry per group with no
        # skips, so the two lists are positionally aligned; zip rather than a
        # uuid lookup keeps that assumption in one place, and the assert makes
        # it fail loudly if a skip is ever added.
        assert len(group_params) == len(groups), (
            f"group_params has {len(group_params)} entries for {len(groups)} "
            "groups; the animated-material attach below assumes one each"
        )
        for group, (params_dict, _objects, _uuids) in zip(groups, group_params):
            series = anim_by_group.get(group.uuid)
            if series:
                params_dict["param-anim"] = series
    # Spatial material maps, attached to their group the same way the animated
    # schedules are. Per object inside, because the weights are per vertex and
    # a group can hold several objects.
    if spatial:
        for group, (params_dict, _objects, _uuids) in zip(groups, group_params):
            series = spatial.get(group.uuid)
            if series:
                params_dict["material-maps"] = series

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
