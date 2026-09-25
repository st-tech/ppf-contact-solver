# File: encoder/param_anim.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Sampling of Blender F-curves on a group's material sliders into the
# per-object material schedules the solver reads.
#
# The artist keyframes the slider itself; there is no separate keyframe list to
# fill in. That is the whole point of the feature: a property that offers a
# keyframe button has to honor the curve drawn on it, and the ones that cannot
# carry `options=NOT_ANIMATABLE` so Blender never offers the button.

import bpy  # pyright: ignore

from ...models.material_maps import gate_open, to_solver_value
from ..utils import get_id_fcurves


# Solver key -> how to read it off a group, per object type.
#
# `prop` is the addon property holding the value, chosen by object type where
# the UI splits it. Every sample then goes through `to_solver_value`, which is
# the same conversion the static value takes, so a curve on a percent slider
# ships a fraction and a curve on a pascal slider ships Pa/rho. A key whose
# feature toggle is off is skipped: the toggle is not animatable, so the value
# is zero for the whole solve and a schedule of zeros is the same result at the
# cost of a per-element array per frame.
#
# Only keys the solver can actually animate appear here. The rest feed the
# hinge, edge and vertex tables that are derived from the faces at build, and
# the solver refuses a schedule for them rather than animating the membrane
# while the hinges stay put.
ANIMATABLE_MATERIAL_KEYS = {
    "pressure": {
        "prop": {"SHELL": "inflate_pressure", "SOLID": "inflate_pressure"},
    },
    "young-mod": {
        "prop": {
            "SHELL": "shell_young_modulus",
            "SOLID": "solid_young_modulus",
            "ROD": "rod_young_modulus",
        },
    },
    "poiss-rat": {
        "prop": {"SHELL": "shell_poisson_ratio", "SOLID": "solid_poisson_ratio"},
    },
    "deformation-damping": {
        "prop": {"SHELL": "deformation_damping", "SOLID": "deformation_damping",
                 "ROD": "deformation_damping"},
    },
    "strain-limit": {
        "prop": {"SHELL": "strain_limit_percent", "ROD": "strain_limit_percent"},
    },
    # The rest reach the hinge, edge and vertex tables too. The solver
    # re-derives those every frame from the animated faces, so they stay in
    # step instead of holding their build-time values.
    "bend": {
        "prop": {"SHELL": "bend", "SOLID": "bend", "ROD": "bend"},
    },
    "bend-warp": {
        "prop": {"SHELL": "bend_warp"},
    },
    "bend-weft": {
        "prop": {"SHELL": "bend_weft"},
    },
    "bending-damping": {
        "prop": {"SHELL": "bending_damping", "SOLID": "bending_damping",
                 "ROD": "bending_damping"},
    },
    "friction": {
        "prop": {"SHELL": "friction", "SOLID": "friction", "ROD": "friction",
                 "PDRD": "friction"},
    },
    "plasticity": {
        "prop": {"SHELL": "plasticity", "SOLID": "plasticity"},
    },
    "plasticity-threshold": {
        "prop": {"SHELL": "plasticity_threshold", "SOLID": "plasticity_threshold"},
    },
    "bend-plasticity": {
        "prop": {"SHELL": "bend_plasticity"},
    },
    "bend-plasticity-threshold": {
        "prop": {"SHELL": "bend_plasticity_threshold"},
    },
    # Contact geometry. Their value is not a single slider: a group either
    # authors an absolute distance or a fraction of its own bounding-box
    # diagonal, and either way the result is multiplied by world scaling. The
    # branch is resolved per group by `_contact_resolution` below, because
    # sampling one of the two sliders blindly would ship the wrong number
    # whenever the artist used the other.
    #
    # These are the parameters that can ABORT a run when tightened mid-solve:
    # raising an offset can place two already-touching surfaces inside each
    # other's shell, which the solver reports as a non-separated contact rather
    # than resolving. That is correct and is not to be smoothed over with a
    # tolerance; author the change over enough frames for the contact to open.
    "contact-gap": {"resolve": "gap"},
    "contact-offset": {"resolve": "offset"},
}


def _contact_resolution(group, state, which):
    """`(property, scale)` for a group's contact gap or offset, or `(None, 0)`.

    Mirrors the branch `_encode_group_params` takes for the static value, so an
    animated contact distance means the same thing as a static one. The branch
    itself cannot change over the solve: `use_group_bounding_box_diagonal` is
    not animatable, and a SAND group's offset is its seeded grain radius rather
    than a slider at all.
    """
    if group.object_type == "SAND":
        return None, 0.0
    if group.use_group_bounding_box_diagonal:
        from .mesh import compute_group_bounding_box_diagonal
        diagonal = compute_group_bounding_box_diagonal(group) * state.world_scaling
        return f"contact_{which}_rat", diagonal
    return f"contact_{which}", float(state.world_scaling)


def _slot_of(scene, group):
    """The ``object_group_N`` slot holding *group*, which is what an F-curve's
    data path names. This is not ``ObjectGroup.index``."""
    from ...models.groups import get_group_slot_index
    return get_group_slot_index(scene, group.uuid)


def _drop_collinear(times, series):
    """Drop samples that lie on the straight line between their neighbors.

    The solver interpolates linearly between samples, so a sample equal to the
    interpolation of the two around it carries no information: removing it
    reproduces the same curve exactly. A slider that is keyframed over a short
    window and flat elsewhere collapses to a handful of samples, which matters
    because every retained sample costs one value per element per key on disk.

    `series` maps key -> list of values aligned to `times`; a sample is kept
    when ANY key needs it.
    """
    n = len(times)
    if n <= 2:
        return list(range(n))
    keep = [0]
    for i in range(1, n - 1):
        t0, t1, t2 = times[i - 1], times[i], times[i + 1]
        span = t2 - t0
        w = (t1 - t0) / span if span > 0 else 0.0
        needed = False
        for values in series.values():
            lerp = values[i - 1] + (values[i + 1] - values[i - 1]) * w
            # Relative tolerance: stiffness runs to 1e5 while a damping
            # coefficient sits near 1e-3, so a single absolute epsilon would
            # either keep every stiffness sample or erase every damping one.
            if abs(values[i] - lerp) > 1e-6 * max(1.0, abs(values[i])):
                needed = True
                break
        if needed:
            keep.append(i)
    keep.append(n - 1)
    return keep


def _track_at_frames(sample_frames, values, frames):
    """`values`, authored at `sample_frames`, evaluated at every entry of `frames`.

    Linear between two samples and constant outside them, which is how both the
    frontend and the solver read a keyed sequence.
    """
    out = []
    for f in frames:
        if f <= sample_frames[0]:
            out.append(values[0])
            continue
        if f >= sample_frames[-1]:
            out.append(values[-1])
            continue
        hi = next(i for i, s in enumerate(sample_frames) if s > f)
        span = sample_frames[hi] - sample_frames[hi - 1]
        alpha = (f - sample_frames[hi - 1]) / span
        out.append(values[hi - 1] + (values[hi] - values[hi - 1]) * alpha)
    return out


# The keys a SOLID can animate. A SOLID's per-frame values reach only its
# SURFACE triangles, which carry its contact material (the tetrahedra have no
# contact fields), while its elastic material lives on its tetrahedra, which
# have no per-frame table. So friction and the contact distances animate, and
# every elastic key is refused rather than shipped to a table nothing reads.
_SOLID_ANIMATED_KEYS = frozenset({"friction", "contact-gap", "contact-offset"})


def encode_param_anim(
    state, groups, fps, start_frame, frame_count, map_schedules=None
):
    """Sample every keyframed material slider across the solve's frame range.

    Returns ``(times, per_group)`` where ``times`` is a list of seconds and
    ``per_group`` maps a group's uuid to ``{solver_key: [value per time]}``.
    Both are empty when nothing in the scene varies a material over time.

    `map_schedules` carries what a spatial map keyed over time needs from the
    shared time axis. Its series vote on which times survive decimation and are
    never emitted: the map itself is resolved per element by the frontend.
    """
    scene = bpy.context.scene
    # A curve on any group field outside the sampled set (a material map's
    # own fields included) has already been refused by
    # `curve_refusal.refuse_unsampled_curves`.
    curves = get_id_fcurves(scene)
    if not curves and not map_schedules:
        return [], {}
    by_path = {fc.data_path: fc for fc in curves}

    frames = [start_frame + i for i in range(max(1, int(frame_count)))]
    per_group = {}
    witness = {}
    for group in groups:
        slot = _slot_of(scene, group)
        if slot is None:
            continue
        obj_type = group.object_type
        series = {}
        for key, spec in ANIMATABLE_MATERIAL_KEYS.items():
            if "resolve" in spec:
                prop, scale = _contact_resolution(group, state, spec["resolve"])
            else:
                prop, scale = spec["prop"].get(obj_type), None
            if prop is None:
                # The offer has to match the delivery. Blender draws a keyframe
                # control on every material slider, so a group type this key is
                # not sampled for has to say so rather than drop the curve
                # before the lookup and simulate the slider's static value.
                candidates = (
                    set(spec["prop"].values())
                    if "prop" in spec
                    else {f"contact_{spec['resolve']}",
                          f"contact_{spec['resolve']}_rat"}
                )
                drawn = next(
                    (
                        name
                        for name in sorted(candidates)
                        if f"zozo_contact_solver.object_group_{slot}.{name}"
                        in by_path
                    ),
                    None,
                )
                if drawn is not None:
                    raise ValueError(
                        f"group '{group.name}' keyframes '{drawn}', but a "
                        f"{obj_type} group's '{key}' is not animated: the "
                        "solver carries a per-frame material table for "
                        "triangles only. Remove the keyframes, or split the "
                        "change into separate solves."
                    )
                continue
            fcurve = by_path.get(f"zozo_contact_solver.object_group_{slot}.{prop}")
            if fcurve is None:
                continue
            if obj_type == "ROD":
                # The per-frame material tables are written for TRIANGLES only.
                # A rod contributes none, so a sampled schedule would be
                # accepted here, shipped, and then reach no element table.
                raise ValueError(
                    f"group '{group.name}' keyframes '{prop}', but a ROD "
                    "group's material values are not animated: the solver "
                    "carries a per-frame table for triangles only. Remove the "
                    "keyframes, or split the change into separate solves."
                )
            if obj_type == "SOLID" and key not in _SOLID_ANIMATED_KEYS:
                raise ValueError(
                    f"group '{group.name}' keyframes '{prop}', but a SOLID "
                    f"group's '{key}' is not animated: its elastic material "
                    "lives on its tetrahedra, which carry no per-frame table "
                    "(only friction and the contact distances, which a SOLID "
                    "keeps on its surface, animate). Remove the keyframes, or "
                    "split the change into separate solves."
                )
            if scale is not None:
                # A contact distance is a length. It is resolved against the
                # group's own authoring mode and world scaling by
                # `_contact_resolution`, not by the material conversion.
                series[key] = [float(fcurve.evaluate(f)) * scale for f in frames]
                continue
            if not gate_open(group, key):
                continue
            series[key] = [
                to_solver_value(group, key, fcurve.evaluate(f)) for f in frames
            ]
        if series:
            per_group[group.uuid] = series

        schedule = (map_schedules or {}).get(group.uuid)
        for key, entry in (schedule or {}).items():
            spec = ANIMATABLE_MATERIAL_KEYS.get(key)
            prop = spec["prop"].get(obj_type) if spec and "prop" in spec else None
            if prop is None:
                raise ValueError(
                    f"group '{group.name}' keys a '{key}' map over time, but "
                    f"the solver carries no schedule for '{key}' on a "
                    f"{obj_type} group"
                )
            base = series.get(key)
            if base is None:
                value = to_solver_value(group, key, getattr(group, prop))
                base = [value] * len(frames)
            target = entry["target"]
            for i, track in enumerate(entry["tracks"]):
                weights = _track_at_frames(entry["frames"], track, frames)
                witness[f"{group.uuid}:{key}:map{i}"] = [
                    b + (target - b) * w for b, w in zip(base, weights)
                ]

    if not per_group and not witness:
        return [], {}

    # One shared time list for the whole scene, so the solver reads a single
    # times.bin. A sample is kept when any group's any key needs it.
    from . import frame_to_time
    times = [max(0.0, frame_to_time(f, fps, start_frame)) for f in frames]
    merged = {}
    for uuid, series in per_group.items():
        for key, values in series.items():
            merged[f"{uuid}:{key}"] = values
    merged.update(witness)
    keep = _drop_collinear(times, merged)
    times = [times[i] for i in keep]
    per_group = {
        uuid: {key: [values[i] for i in keep] for key, values in series.items()}
        for uuid, series in per_group.items()
    }
    return times, per_group
