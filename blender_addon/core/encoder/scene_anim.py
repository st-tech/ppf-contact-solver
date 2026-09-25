# File: encoder/scene_anim.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Sampling of Blender F-curves on the SCENE-level solver settings into the
# dyn_param schedule the solver already reads.
#
# This replaces the addon's own keyframe list. The artist keyframes the slider
# itself, the same gesture that drives a material parameter, and there is no
# second place to learn. The wire format is unchanged: the solver still reads
# `dyn_param.txt` and the frontend still builds it through `session.param.dyn`,
# so only the AUTHORING moved.
#
# Every value goes through the same transform the static encoder applies, per
# sample (`solver_gravity`, `solver_wind`). Gravity and wind are swapped into
# solver axes (they are physical, so World Scaling does not scale them), and
# wind is a direction times a strength. Sampling the raw slider and skipping that would ship a plausible
# number that means something else.
#
# Only the properties named in `SCENE_ANIM_KEYS` are sampled, and they are the
# only animatable State properties. Every other State property is read once,
# at the starting frame (Inactive Momentum Frames, for one, counts frames from
# the start of the solve and the solver reads its key as an on/off flag). A
# curve on any of them is refused before this sampler runs, by
# `curve_refusal.refuse_unsampled_curves`.

from ..utils import get_id_fcurves
from . import frame_to_time, solver_gravity, solver_wind
from .param_anim import _drop_collinear


_STATE_PATH = "zozo_contact_solver.state."

# solver key -> the state properties it reads, and how they combine.
#
# `props` is what must be watched for an F-curve: a key is animated when ANY of
# its properties carries one, because wind is a direction and a strength and
# keyframing either one animates the result.
SCENE_ANIM_KEYS = {
    "gravity": {"props": ("gravity_3d",), "kind": "gravity"},
    "wind": {"props": ("wind_direction", "wind_strength"), "kind": "wind"},
    "air-density": {"props": ("air_density",), "kind": "scalar"},
    "air-friction": {"props": ("air_friction",), "kind": "scalar"},
    "isotropic-air-friction": {"props": ("vertex_air_damp",), "kind": "scalar"},
    "dt": {"props": ("step_size",), "kind": "scalar"},
}


def _sample(curves_by_key, state, prop, frame, length):
    """`prop`'s value at `frame`: its F-curve where one exists, else the slider.

    A vector property keyframed on one component only is normal (gravity down
    a single axis), so each component falls back independently rather than the
    whole property falling back together.
    """
    static = getattr(state, prop)
    if length == 0:
        fc = curves_by_key.get((prop, 0))
        return float(fc.evaluate(frame)) if fc is not None else float(static)
    out = []
    for i in range(length):
        fc = curves_by_key.get((prop, i))
        out.append(float(fc.evaluate(frame)) if fc is not None else float(static[i]))
    return out


def encode_scene_param_anim(state, fps, start_frame, frame_count):
    """Sample every keyframed scene setting across the solve's frame range.

    Returns the `dyn_param` dict the frontend decoder already consumes:
    ``{solver_key: [(time_seconds, value_list, is_hold), ...]}``. `is_hold` is
    always False here, because a sampled curve carries its own shape and has
    nothing to hold.

    Empty when no scene setting is keyframed, which leaves the run on its
    static values.
    """
    import bpy  # pyright: ignore

    curves = get_id_fcurves(bpy.context.scene)
    if not curves:
        return {}
    by_prop = {}
    for fc in curves:
        if not fc.data_path.startswith(_STATE_PATH):
            continue
        prop = fc.data_path[len(_STATE_PATH):]
        by_prop[(prop, fc.array_index)] = fc
    if not by_prop:
        return {}

    frames = [start_frame + i for i in range(max(1, int(frame_count)))]
    lengths = {
        p: getattr(state.bl_rna.properties[p], "array_length", 0)
        for spec in SCENE_ANIM_KEYS.values()
        for p in spec["props"]
    }

    series = {}
    for key, spec in SCENE_ANIM_KEYS.items():
        watched = spec["props"]
        if not any((p, i) in by_prop for p in watched for i in range(4)):
            continue
        values = []
        for frame in frames:
            read = {p: _sample(by_prop, state, p, frame, lengths[p]) for p in watched}
            kind = spec["kind"]
            if kind == "gravity":
                values.append(solver_gravity(read["gravity_3d"]))
            elif kind == "wind":
                values.append(solver_wind(
                    read["wind_direction"], read["wind_strength"],
                    f"The keyframed wind at frame {frame}"))
            else:
                values.append([read[watched[0]]])
        series[key] = values

    if not series:
        return {}

    # One shared sample set, so a scene animating gravity and wind together
    # ships one time list rather than two that drift apart. A sample is kept
    # when any key needs it.
    times = [max(0.0, frame_to_time(f, fps, start_frame)) for f in frames]
    flat = {}
    for key, values in series.items():
        width = len(values[0])
        for c in range(width):
            flat[f"{key}:{c}"] = [v[c] for v in values]
    keep = _drop_collinear(times, flat)

    out = {}
    for key, values in series.items():
        out[key] = [(times[i], values[i], False) for i in keep]
    return out
