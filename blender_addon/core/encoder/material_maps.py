# File: encoder/material_maps.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Reads a group's spatial material maps into per-vertex weight arrays.
#
# A map varies one parameter across the surface, or along a rod: the effective
# value at a vertex is lerp(base, target, weight), with `base` the group's own
# slider (or that frame's animated value, so a parameter can be animated and
# mapped at once). Weight 0 reproduces the unmapped result exactly, which is
# what keeps the calibrated presets meaningful.

import math

from ...models.material_maps import (
    MAP_OBJECT_TYPES,
    base_property,
    gate_reason,
    to_solver_value,
)


def _weights_from_vertex_group(obj, name):
    """Per-vertex weights from a vertex group, 0 where a vertex is not in it.

    `vg.weight(i)` raises for a vertex the group does not contain, which is the
    normal case for a painted region rather than an error, so absence reads as
    zero: unpainted geometry keeps the group's own value.
    """
    vg = obj.vertex_groups.get(name)
    if vg is None:
        raise ValueError(
            f"object '{obj.name}' has no vertex group '{name}' for its material map"
        )
    out = []
    for i in range(len(obj.data.vertices)):
        try:
            out.append(float(vg.weight(i)))
        except RuntimeError:
            out.append(0.0)
    return out


def _weights_from_attribute(obj, name, context):
    """Per-vertex weights from a float attribute on the POINT domain.

    The attribute is read off the EVALUATED object, which is where a Store
    Named Attribute node in a Geometry Nodes modifier writes it. The evaluated
    vertex count must equal the base mesh's, because the weights are shipped
    against the base mesh's vertex order.
    """
    eval_obj = obj.evaluated_get(context.evaluated_depsgraph_get())
    if eval_obj == obj:
        # Not a Geometry Nodes problem: the object is not in the depsgraph at
        # all, so no modifier ran and no attribute a modifier writes exists.
        raise ValueError(
            f"object '{obj.name}' is excluded from evaluation, so its modifier "
            f"stack did not run and attribute '{name}' cannot be read from it. "
            "Disable in Viewports (the monitor icon) and excluding the "
            "collection from the view layer both do this; Hide in Viewport "
            "(the eye icon) does not."
        )
    eval_mesh = eval_obj.to_mesh()
    try:
        attr = eval_mesh.attributes.get(name)
        if attr is None:
            raise ValueError(
                f"object '{obj.name}' has no attribute '{name}' for its "
                "material map. A vertex group is read by name from the object; "
                "an attribute is read from the evaluated mesh, so a Geometry "
                "Nodes output has to reach the modifier stack's result."
            )
        if attr.domain != "POINT":
            raise ValueError(
                f"attribute '{name}' on '{obj.name}' is on the {attr.domain} "
                "domain; a material map needs per-vertex (POINT) weights"
            )
        if attr.data_type not in ("FLOAT", "INT"):
            raise ValueError(
                f"attribute '{name}' on '{obj.name}' has type {attr.data_type}; "
                "a material map needs a scalar float"
            )
        n_base = len(obj.data.vertices)
        if len(eval_mesh.vertices) != n_base:
            raise ValueError(
                f"object '{obj.name}' evaluates to {len(eval_mesh.vertices)} "
                f"vertices but its base mesh has {n_base}, so the weights in "
                f"attribute '{name}' cannot be matched to the vertices the "
                "solver receives. Read the map from a vertex group, or move "
                "the modifier that changes the vertex count."
            )
        return [float(d.value) for d in attr.data]
    finally:
        eval_obj.to_mesh_clear()


def _clamped_weights(w, object_name, source_name):
    """`w` clamped into [0, 1], refusing a weight that is not a finite number.

    The solver blends in [0, 1]. A painted weight is already in range, but an
    attribute holds whatever wrote it, and a value outside the interval would
    push the parameter past its target rather than toward it. A non-finite
    weight is refused rather than clamped: `min(1.0, max(0.0, nan))` is `0.0`,
    which reads as "use the base value" and hides the source of the NaN.
    """
    out = []
    for i, x in enumerate(w):
        if not math.isfinite(x):
            raise ValueError(
                f"weight {x} at vertex {i} of '{object_name}' comes from "
                f"'{source_name}' and is not a finite number, so it names no "
                "point between the group's value and the map target"
            )
        out.append(min(1.0, max(0.0, x)))
    return out


def _witness_tracks(frames, per_object):
    """One weight sequence per authored TIME, chosen by curvature at that time.

    The decimation keeps a sample only when some series deviates from the
    straight line through its two neighbors, so the question it asks is about
    CURVATURE, not about magnitude of change. A vertex that moves a long way
    but affinely needs no interior sample at all, while one that moves a short
    way with a kink needs exactly the authored time where it bends. Selecting
    by largest change answers the wrong question and lets an authored
    excursion be interpolated away.

    The chosen vertex's own sequence is used rather than a synthetic ramp, so
    the decimation's relative tolerance is anchored to a value the solver will
    actually carry.
    """
    tracks = []
    for k in range(1, len(frames)):
        best_dev, best_track = 0.0, None
        for weights in per_object.values():
            if not weights[k]:
                continue
            for vertex in range(len(weights[k])):
                here = float(weights[k][vertex])
                before = float(weights[k - 1][vertex])
                if k + 1 < len(frames):
                    after = float(weights[k + 1][vertex])
                    span = frames[k + 1] - frames[k - 1]
                    alpha = (frames[k] - frames[k - 1]) / span if span else 0.0
                    dev = abs(here - (before + (after - before) * alpha))
                else:
                    # The track holds after the last authored frame, so the
                    # kink there is the last segment's own change.
                    dev = abs(here - before)
                if dev > best_dev:
                    best_dev = dev
                    best_track = [float(w[vertex]) for w in weights]
        # A time where no vertex bends needs no sample of its own: the base's
        # own series already governs the composed value there.
        if best_track is not None and best_dev > 0.0:
            tracks.append(best_track)
    return tracks


def _sample_sources(group, entry, start_frame):
    """`[(frame, source_type, source_name)]` for one map, earliest first.

    The row's own source is the map at the start frame; each sample names a
    different source reached at its own frame.
    """
    sources = [(int(start_frame), entry.source_type, entry.source_name)]
    seen = {int(start_frame)}
    for sample in sorted(entry.samples, key=lambda s: int(s.frame)):
        frame = int(sample.frame)
        if frame <= int(start_frame):
            raise ValueError(
                f"group '{group.name}' has a '{entry.parameter}' map sample at "
                f"frame {frame}, at or before the start frame {int(start_frame)}"
                ". The map's own source is already the weights at the start "
                "frame."
            )
        if frame in seen:
            raise ValueError(
                f"group '{group.name}' has two '{entry.parameter}' map samples "
                f"at frame {frame}"
            )
        if not sample.source_name:
            raise ValueError(
                f"group '{group.name}' has a '{entry.parameter}' map sample at "
                f"frame {frame} with no source name"
            )
        seen.add(frame)
        sources.append((frame, sample.source_type, sample.source_name))
    return sources


def _weights_from_hair(obj, source_type, name, context):
    """Per-point weights of a Hair Curves object, whose points are its rod
    vertices.

    Hair Curves have no vertex groups, so a map on one reads a point
    attribute. It is read where the shipped pose is read
    (`hair_curves.eval_point_weights`): the groom's own attribute, or one a
    Geometry Nodes modifier in front of the solver's cache stores.
    """
    if source_type == "VERTEX_GROUP":
        raise ValueError(
            f"Hair Curves '{obj.name}' has no vertex groups, so its material "
            f"map cannot read vertex group '{name}'. Set the map's Source to "
            "Attribute and name a float point attribute."
        )
    from ..hair_curves import eval_point_weights

    return eval_point_weights(obj, name, context).tolist()


def _read_weights(obj, source_type, source_name, context):
    """One weight per vertex of `obj`, clamped into [0, 1]."""
    if obj.type == "CURVES":
        w = _weights_from_hair(obj, source_type, source_name, context)
    elif source_type == "VERTEX_GROUP":
        w = _weights_from_vertex_group(obj, source_name)
    else:
        w = _weights_from_attribute(obj, source_name, context)
    return _clamped_weights(w, obj.name, source_name)


def _refuse_unmappable_object(group, key, assigned, obj):
    """Raise unless `obj` is an object a map on `group` can read weights from."""
    if obj is not None and obj.type in MAP_OBJECT_TYPES.get(group.object_type, ()):
        return
    if obj is not None and obj.type == "CURVE":
        raise ValueError(
            f"group '{group.name}' has a '{key}' map but its object "
            f"'{assigned.name}' is a Curve, whose rod vertices are sampled "
            "along the curve and are not points a weight can be authored "
            "on. Convert it to a mesh or to Hair Curves, or remove it from "
            "the group."
        )
    raise ValueError(
        f"group '{group.name}' has a '{key}' map but its object "
        f"'{assigned.name}' is missing or is not an object a map can read "
        "weights from. Re-pick the object, or remove it from the group."
    )


def encode_material_maps(context, groups, fps, start_frame):
    """Per-group spatial maps and the series their decimation needs.

    Returns ``(payload, schedules)``.

    The payload is ``{group_uuid: {solver_key: row}}``. A static map's row is
    ``{"target": float, "weights": {object_uuid: [w per vertex]}}``. A map with
    samples ships ``{"target": float, "times": [seconds],
    "weight_frames": {object_uuid: [[w per vertex] per time]}}`` instead. A row
    carries one of the two, never both.

    The schedules are not shipped. They carry, per authored segment, the
    composed value of the vertex whose weight moves most on that segment, so
    the shared keyframe axis keeps the times a moving map needs.
    """
    from . import frame_to_time
    from ..uuid_registry import get_object_by_uuid

    out = {}
    schedules = {}
    for group in groups:
        maps = [m for m in getattr(group, "material_maps", []) if m.enabled]
        if not maps:
            continue
        obj_type = group.object_type
        per_key = {}
        for entry in maps:
            key = entry.parameter
            if base_property(key, obj_type) is None:
                raise ValueError(
                    f"group '{group.name}' maps '{key}', which is not "
                    f"available as a map on a {obj_type} group. A map is "
                    "reduced to one coefficient per element, so the parameter "
                    "has to be one this group's elements read."
                )
            if not entry.source_name:
                raise ValueError(
                    f"group '{group.name}' has a '{key}' map with no source name"
                )
            if key in per_key:
                # Two maps on one parameter have no defined composition: the
                # blend is base-to-target, and a second target is a different
                # answer for the same value, not a refinement of it.
                raise ValueError(
                    f"group '{group.name}' has two maps driving '{key}'; "
                    "each parameter takes at most one"
                )
            # Validate the target against the mapped property's OWN floor, not
            # against zero: `young-mod` has a positive minimum, so a target of
            # 0.0 clears a non-negative test and then aborts the build.
            prop_name = base_property(key, obj_type)
            floor = 0.0
            try:
                floor = float(group.bl_rna.properties[prop_name].hard_min)
            except (KeyError, AttributeError, TypeError):
                floor = 0.0
            if not math.isfinite(entry.target_value):
                raise ValueError(
                    f"group '{group.name}' has a '{key}' map whose target is "
                    f"{entry.target_value}, which is not a finite number"
                )
            if entry.target_value < floor:
                extra = (
                    " A negative target does not tighten the parameter: it "
                    "switches the term off on exactly the painted elements."
                    if floor == 0.0 else ""
                )
                raise ValueError(
                    f"group '{group.name}' has a '{key}' map with a target of "
                    f"{entry.target_value}, below the {floor} minimum the "
                    f"'{prop_name}' slider itself enforces.{extra}"
                )
            target = to_solver_value(group, key, entry.target_value)
            if target is None:
                raise ValueError(
                    f"group '{group.name}' has a '{key}' map, but "
                    f"{gate_reason(group, key)}, so the parameter is zero for "
                    "the whole solve. A map target cannot reintroduce a value "
                    "the group switched off; turn the parameter on, or remove "
                    "the map."
                )
            sources = _sample_sources(group, entry, start_frame)
            if len(sources) > 1 and obj_type != "SHELL":
                raise ValueError(
                    f"group '{group.name}' has a '{key}' map keyed over time, "
                    f"which a {obj_type} group does not carry: the solver has "
                    "no per-element material schedule for it."
                )
            per_object = {}
            for assigned in group.assigned_objects:
                if not assigned.included:
                    continue
                # The weights ship keyed by uuid, so they are resolved by uuid
                # too. A name lookup would lose the map of a renamed object
                # while still shipping its uuid as the key.
                obj = get_object_by_uuid(assigned.uuid)
                _refuse_unmappable_object(group, key, assigned, obj)
                per_object[assigned.uuid] = [
                    _read_weights(obj, source_type, source_name, context)
                    for _frame, source_type, source_name in sources
                ]
            if not per_object:
                continue
            if len(sources) == 1:
                per_key[key] = {
                    "target": target,
                    "weights": {u: w[0] for u, w in per_object.items()},
                }
            else:
                frames = [f for f, _t, _n in sources]
                per_key[key] = {
                    "target": target,
                    "times": [
                        max(0.0, frame_to_time(f, fps, start_frame)) for f in frames
                    ],
                    "weight_frames": per_object,
                }
                schedules.setdefault(group.uuid, {})[key] = {
                    "target": target,
                    "frames": frames,
                    "tracks": _witness_tracks(frames, per_object),
                }
        if per_key:
            out[group.uuid] = per_key
    return out, schedules
