# File: scenarios/bl_material_lock_guards.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A padlock beside a material parameter has to protect it, or it is decoration.
#
# The lock follows Blender's own reading, the one on Transform > Location: it
# guards the value against the TOOLS that overwrite a whole group at once,
# which here are the material presets and paste-material. It does NOT mute an
# F-curve, matching lock_location, where a locked channel still animates.
#
#   A. preset_skips_locked: applying a preset leaves a locked value alone and
#      changes an unlocked one, so the skip is the lock and not a no-op apply.
#   B. paste_skips_locked: the same for paste-material, and the source group's
#      OWN lock state does not travel, because a lock belongs to the group it
#      protects. A paste from an unlocked group must not clear the target's
#      protection.
#   C. lock_is_not_animatable: the padlock itself takes no keyframe. A lock
#      that could be animated would silently change which values a tool may
#      overwrite partway through a solve.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. A full rig sweep against a CPU build passed it, and that run is the
# evidence this line rests on.
BACKENDS = ("real",)

_LOCKED_BEND = 123.0
_FREE_YOUNG = 456.0


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCKED_BEND = <<LOCKED_BEND>>
FREE_YOUNG = <<FREE_YOUNG>>

try:
    dh = DriverHelpers(pkg, result)
    plane = dh.reset_scene_to_pinned_plane(name="LockMesh")
    root = dh.configure_state(project_name="lock_guards", frame_count=4)
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    group = root.object_group_0

    presets = __import__(pkg + ".core.material_presets",
                         fromlist=["apply_material_preset",
                                   "load_material_presets"])
    preset_name = sorted(presets.load_material_presets().keys())[0]

    group.bend = LOCKED_BEND
    group.shell_young_modulus = FREE_YOUNG
    group.lock_bend = True
    presets.apply_material_preset(preset_name, group)
    dh.record(
        "A_preset_skips_locked",
        abs(group.bend - LOCKED_BEND) < 1e-3
        and abs(group.shell_young_modulus - FREE_YOUNG) > 1e-3,
        {"preset": preset_name, "locked_bend": group.bend,
         "expected_locked": LOCKED_BEND,
         "free_young": group.shell_young_modulus,
         "free_young_before": FREE_YOUNG},
    )

    # Paste: a second group supplies the clipboard, with its own lock OFF, so
    # a lock travelling with the paste would show up as the target unlocking.
    second = dh.api.solver.create_group("Donor", "SHELL")
    donor = root.object_group_1
    donor.bend = 999.0
    donor.shell_young_modulus = 888.0
    donor.lock_bend = False

    group.bend = LOCKED_BEND
    group.shell_young_modulus = FREE_YOUNG
    group.lock_bend = True

    introspect = __import__(pkg + ".core.param_introspect",
                            fromlist=["copy_scalar_props",
                                      "MATERIAL_CLIPBOARD_EXCLUDE"])
    locks_mod = __import__(pkg + ".models.material_locks",
                           fromlist=["locked_props"])
    locked = locks_mod.locked_props(group)
    introspect.copy_scalar_props(
        donor, group,
        exclude=introspect.MATERIAL_CLIPBOARD_EXCLUDE,
        filter_fn=lambda n: n not in locked,
    )
    dh.record(
        "B_paste_skips_locked",
        abs(group.bend - LOCKED_BEND) < 1e-3
        and abs(group.shell_young_modulus - 888.0) < 1e-3
        and group.lock_bend is True,
        {"locked_bend": group.bend, "expected_locked": LOCKED_BEND,
         "free_young": group.shell_young_modulus, "expected_free": 888.0,
         "lock_survived_paste": group.lock_bend},
    )

    prop = group.bl_rna.properties["lock_bend"]
    dh.record(
        "C_lock_is_not_animatable",
        not prop.is_animatable,
        {"is_animatable": prop.is_animatable},
    )

    # D. The invariant that stops this whole class of defect recurring: every
    # property Blender OFFERS a keyframe on must be one the encoder can
    # actually read, and vice versa. A property on the offered side alone puts
    # a working keyframe button in front of the artist that the solve ignores,
    # which is exactly the reported bug; one on the sampled side alone is a
    # schedule that can never be authored.
    pa = __import__(pkg + ".core.encoder.param_anim",
                    fromlist=["ANIMATABLE_MATERIAL_KEYS", "_contact_resolution"])
    state = root.state
    sampled = set()
    for key, spec in pa.ANIMATABLE_MATERIAL_KEYS.items():
        if "resolve" in spec:
            # Both contact branches and every type, since which one applies is
            # a per-group setting rather than a property of the key.
            for bbox in (False, True):
                group.use_group_bounding_box_diagonal = bbox
                for t in ("SHELL", "SOLID", "ROD"):
                    group.object_type = t
                    name, _scale = pa._contact_resolution(group, state, spec["resolve"])
                    if name:
                        sampled.add(name)
        else:
            sampled.update(spec["prop"].values())
    group.object_type = "SHELL"

    offered = {p.identifier for p in group.bl_rna.properties
               if p.identifier != "rna_type" and p.is_animatable}
    offered_not_sampled = sorted(offered - sampled)
    sampled_not_offered = sorted(sampled - offered)
    dh.record(
        "D_keyframe_offer_matches_delivery",
        not offered_not_sampled and not sampled_not_offered,
        {"offered": len(offered), "sampled": len(sampled & offered),
         "offered_but_never_sampled": offered_not_sampled,
         "sampled_but_not_keyframable": sampled_not_offered},
    )

    locks_all = __import__(pkg + ".models.material_locks",
                           fromlist=["ANIMATABLE_MATERIAL_PROPS",
                                     "LOCKABLE_MATERIAL_PROPS"])
    dh.record(
        "E_declared_set_matches_rna",
        set(locks_all.ANIMATABLE_MATERIAL_PROPS) == offered,
        {"declared": len(locks_all.ANIMATABLE_MATERIAL_PROPS),
         "rna_animatable": len(offered),
         "declared_only": sorted(set(locks_all.ANIMATABLE_MATERIAL_PROPS) - offered),
         "rna_only": sorted(offered - set(locks_all.ANIMATABLE_MATERIAL_PROPS))},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCKED_BEND>>", repr(_LOCKED_BEND))
        .replace("<<FREE_YOUNG>>", repr(_FREE_YOUNG))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
