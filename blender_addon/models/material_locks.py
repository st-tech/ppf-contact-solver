# File: models/material_locks.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# The material properties that carry a lock, and what a lock means.
#
# This is exactly the set the solver can vary per frame, which is the same set
# Blender offers a keyframe on (everything else carries options=NOT_ANIMATABLE).
# One list, so the padlock cannot appear beside a property that no schedule can
# drive, and cannot be missing from one that can.
#
# A lock follows Blender's own reading, the one on Transform > Location: it
# guards the value against being changed by a TOOL. Here the tools are the
# material presets and the copy-paste-material operators, which overwrite a
# whole group at once and are exactly what an artist wants a hand-tuned value
# protected from. It does NOT mute an F-curve, matching lock_location, where a
# locked channel still animates; muting is what the curve's own controls are
# for.

# Locks are named after their property (`lock_bend`), not indexed. An index
# would be positional, so inserting a property would silently repoint every
# saved lock, the same hazard the material-map enum ids avoid by being
# permanent. A name has no order to get wrong.
LOCKABLE_MATERIAL_PROPS = (
    "bend",
    "bend_plasticity",
    "bend_plasticity_threshold",
    "bend_warp",
    "bend_weft",
    "bending_damping",
    "contact_gap",
    "contact_gap_rat",
    "contact_offset",
    "contact_offset_rat",
    "deformation_damping",
    "friction",
    "inflate_pressure",
    "length_factor",
    "pdrd_density",
    "plasticity",
    "plasticity_threshold",
    "rod_density",
    "rod_young_modulus",
    "sand_friction",
    "sand_particle_mass",
    "shell_density",
    "shell_poisson_ratio",
    "shell_young_modulus",
    "shrink",
    "shrink_x",
    "shrink_y",
    "solid_density",
    "solid_poisson_ratio",
    "solid_young_modulus",
    "stitch_stiffness",
    "strain_limit_percent",
)


# The properties Blender may offer a KEYFRAME on. Strictly the set the encoder
# samples, so the offer and the delivery are the same thing.
#
# Everything in LOCKABLE_MATERIAL_PROPS but not here is a material parameter
# with a padlock and no keyframe, because no transport exists for it:
#
#   density (rod/shell/solid/pdrd) and sand_particle_mass change MASS, which is
#     fixed at build and feeds the inertia term plus the PDRD and grain
#     inertia. Animating it needs a decision about momentum first.
#   shrink, shrink_x, shrink_y, length_factor change the REST SHAPE, which is
#     inv_rest and belongs to the streamed rest-shape path, not a material
#     table.
#   sand_friction is a SAND material and the schedules are per-triangle, so a
#     faceless grain cloud has nothing to carry it.
#   stitch_stiffness is resolved per object and applied per stitch ROW, not
#     through a per-element parameter array.
#
# A lock still makes sense for all of those: protecting a value from a preset
# is independent of animating it.
#
# `bl_material_lock_guards` asserts this equals what the encoder can actually
# read. Adding a property here without giving it a sampler entry puts a
# keyframe button in front of the artist that the solve ignores, which is the
# defect this whole branch exists to remove.
ANIMATABLE_MATERIAL_PROPS = (
    "bend",
    "bend_plasticity",
    "bend_plasticity_threshold",
    "bend_warp",
    "bend_weft",
    "bending_damping",
    "contact_gap",
    "contact_gap_rat",
    "contact_offset",
    "contact_offset_rat",
    "deformation_damping",
    "friction",
    "inflate_pressure",
    "plasticity",
    "plasticity_threshold",
    "rod_young_modulus",
    "shell_poisson_ratio",
    "shell_young_modulus",
    "solid_poisson_ratio",
    "solid_young_modulus",
    "strain_limit_percent",
)


def lock_name(prop: str) -> str:
    """The lock boolean guarding `prop`."""
    return f"lock_{prop}"


def is_locked(group, prop: str) -> bool:
    """Whether `prop` is locked on `group`.

    False for a property with no lock, so a caller can ask about any property
    without first checking whether it is lockable.
    """
    return bool(getattr(group, lock_name(prop), False))


def locked_props(group) -> set:
    """Every locked property on `group`."""
    return {p for p in LOCKABLE_MATERIAL_PROPS if is_locked(group, p)}
