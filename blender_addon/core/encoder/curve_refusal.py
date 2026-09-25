# File: encoder/curve_refusal.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# One encode-time check for animation the encoder does not sample.
#
# The encoder samples exactly two families of add-on properties over the solve:
# the scene settings in `SCENE_ANIM_KEYS` and a group's material sliders in
# `ANIMATABLE_MATERIAL_PROPS`, both from the scene's active action. Every other
# add-on property is read once, at the starting frame, and carries
# `options=NOT_ANIMATABLE` so Blender offers no keyframe on it. A .blend saved
# before a property became non-animatable can still carry a curve on it, and a
# driver can still be attached from Python. Blender then evaluates neither, yet
# the timeline keeps drawing the keys, so the artist believes the setting
# animates while the solve holds it constant. Every such curve is refused here,
# naming its data path, so it is deleted rather than trusted.
#
# Drivers and NLA strips are refused on EVERY add-on path, the sampled ones
# included: the samplers read the active action only, so a driven or NLA-held
# setting would reach the solve as its starting-frame value while the viewport
# shows it changing.
#
# The check reads data paths only and evaluates nothing, so it answers the same
# wherever the playhead sits.

import re

from ...models.material_locks import ANIMATABLE_MATERIAL_PROPS
from ..utils import get_id_fcurves
from .scene_anim import SCENE_ANIM_KEYS


# Every add-on property lives under this namespace on the scene
# (`ui/state.py` registers it); no other ID carries add-on RNA.
_ROOT = "zozo_contact_solver"

_STATE_PROP = re.compile(r"zozo_contact_solver\.state\.([A-Za-z_]\w*)$")
_GROUP_PROP = re.compile(r"zozo_contact_solver\.object_group_\d+\.([A-Za-z_]\w*)$")

SAMPLED_STATE_PROPS = frozenset(
    prop for spec in SCENE_ANIM_KEYS.values() for prop in spec["props"]
)
SAMPLED_GROUP_PROPS = frozenset(ANIMATABLE_MATERIAL_PROPS)

# Paths whose reason is more specific than "read once". Matched by substring.
_SPECIFIC_REASONS = (
    (
        "zozo_contact_solver.state.inactive_momentum_frames",
        "Inactive Momentum Frames counts frames from the start of the solve "
        "and cannot change over it; delete its keyframes",
    ),
    (
        ".material_maps[",
        "a material map's own fields are not animated. Key the map's weights "
        "by adding map samples, and delete this curve",
    ),
)

_READ_ONCE_REASON = (
    "nothing samples this setting over time (the solver reads it once, at "
    "the starting frame, if at all), so the curve would be ignored for the "
    "whole solve; delete its keyframes"
)


def is_sampled_path(data_path: str) -> bool:
    """Whether the encoder samples an active-action curve on *data_path*."""
    match = _STATE_PROP.match(data_path)
    if match is not None:
        return match.group(1) in SAMPLED_STATE_PROPS
    match = _GROUP_PROP.match(data_path)
    if match is not None:
        return match.group(1) in SAMPLED_GROUP_PROPS
    return False


def _is_addon_path(data_path: str) -> bool:
    return data_path == _ROOT or data_path.startswith((_ROOT + ".", _ROOT + "["))


def _reason(data_path: str) -> str:
    for needle, reason in _SPECIFIC_REASONS:
        if needle in data_path:
            return reason
    return _READ_ONCE_REASON


def _nla_curves(scene):
    """``(track, strip, fcurve)`` for every curve an NLA strip on *scene* holds,
    read from the slot each strip is assigned."""
    ad = scene.animation_data
    out = []
    for track in ad.nla_tracks:
        for strip in track.strips:
            action = strip.action
            if action is None:
                continue
            slot = strip.action_slot
            for layer in action.layers:
                for action_strip in layer.strips:
                    bag = action_strip.channelbag(slot) if slot is not None else None
                    if bag is not None:
                        out.extend((track, strip, fc) for fc in bag.fcurves)
            out.extend(
                (track, strip, fc) for fc in (getattr(action, "fcurves", None) or [])
            )
    return out


def refuse_unsampled_curves(scene) -> None:
    """Raise ``ValueError`` naming the first add-on curve the encoder ignores.

    Refused: an active-action curve on an add-on path outside the sampled set,
    and every driver or NLA-strip curve on an add-on path. A path outside the
    add-on namespace is not the encoder's business and passes untouched.
    """
    ad = getattr(scene, "animation_data", None)
    if ad is None:
        return
    for fc in get_id_fcurves(scene):
        path = fc.data_path
        if _is_addon_path(path) and not is_sampled_path(path):
            raise ValueError(f"The scene keyframes '{path}': {_reason(path)}.")
    for fc in ad.drivers:
        path = fc.data_path
        if _is_addon_path(path):
            raise ValueError(
                f"The scene drives '{path}' with a driver, but the solver never "
                "evaluates a driver: it reads the value once, at the starting "
                "frame, and holds it for the whole solve. Remove the driver"
                + (
                    " and keyframe the setting instead."
                    if is_sampled_path(path) else "."
                )
            )
    for track, strip, fc in _nla_curves(scene):
        path = fc.data_path
        if _is_addon_path(path):
            reason = (
                "the solver samples the scene's active action only, so this "
                "curve would be ignored; move its keys into the active action "
                "or delete them"
                if is_sampled_path(path) else _reason(path)
            )
            raise ValueError(
                f"The scene keyframes '{path}' in NLA strip '{strip.name}' of "
                f"track '{track.name}': {reason}."
            )
