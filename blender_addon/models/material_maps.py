# File: models/material_maps.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The material parameters a spatial map may drive, in one place.
#
# Two consumers read this: the EnumProperty on MaterialMapItem, which needs
# explicit numeric ids because it lives on a SAVED PropertyGroup, and the
# encoder, which needs the addon property behind each solver key. Keeping them
# in one table is what stops a map from offering a parameter the encoder cannot
# resolve, which would present as a map the artist can author and the solve
# ignores.

# (solver key, label, description, numeric id)
#
# NUMERIC IDS ARE PERMANENT. Blender stores the number, not the identifier, in
# the .blend, so renumbering silently repoints every saved map at a different
# parameter. Append new entries with the next unused number; never reuse one.
MATERIAL_MAP_KEYS = [
    ("young-mod", "Young's Modulus", "Membrane stiffness", 1),
    ("bend", "Bending Stiffness", "Isotropic hinge bending stiffness", 2),
    ("friction", "Friction", "Coulomb friction coefficient at contacts", 3),
    ("deformation-damping", "Deformation Damping", "Rayleigh damping on stretch", 4),
    ("bending-damping", "Bending Damping", "Rayleigh damping on bending", 5),
    # Offered by the enum but deliberately WITHOUT a base property below, so a
    # map on it is refused at encode. See MATERIAL_MAP_BASE_PROP for why. The
    # id stays reserved: Blender stores the number, so reusing 6 would repoint
    # every saved map that names it.
    ("pressure", "Inflation Pressure", "Outward pressure along face normals", 6),
    ("strain-limit", "Strain Limit", "Upper bound on tensile strain", 7),
    ("plasticity", "Plasticity Rate", "Stretch plasticity creep rate", 8),
    ("bend-plasticity", "Bend Plasticity Rate", "Bending plasticity creep rate", 9),
    ("bend-warp", "Bending (Warp)", "Extra bending stiffness along UV X", 10),
    ("bend-weft", "Bending (Weft)", "Extra bending stiffness along UV Y", 11),
]

# The addon property each key blends away from, per object type. The map's
# BASE is whatever that property holds (or that frame's animated value), so a
# weight of 0 reproduces the unmapped result exactly.
#
# Only SHELL and SOLID appear. A map is reduced to one coefficient per element
# by averaging that element's own vertices, and a rod, a sand cloud and a PDRD
# body carry no element table to reduce over.
#
# `pressure` is absent on purpose. Its per-face potential is
# E_f = -(P_f / 6) * x0 . (x1 x x2), whose gradient is translation-variant: only
# the SUM over a closed surface is invariant, and that sum telescopes to an
# origin-independent force per vertex only while P is uniform. A map makes P
# vary, and the cancellation stops. Measured on a 5 cm icosphere with P painted
# 20 to 100: per-vertex forces are identical under translation with a uniform P,
# and change by 275% of their peak when the object is moved 1 m, 2202% at 8 m.
# The same painted map would mean something different depending on where the
# artist put the object. Re-offering it needs the pressure force assembled as a
# surface traction (-P_f A_f n_f / 3 per vertex, origin-independent, and equal
# to today's result for uniform P) with a PSD-projected Hessian, not a change
# here.
MATERIAL_MAP_BASE_PROP = {
    "young-mod": {
        "SHELL": "shell_young_modulus",
        "SOLID": "solid_young_modulus",
    },
    # A solid's surface hinges carry `type.hinge & 1` and the bending energy
    # skips exactly those, which is why the SOLID panel draws no bend slider.
    # A bend map on a solid would name a value no solid element reads.
    "bend": {"SHELL": "bend"},
    "friction": {"SHELL": "friction", "SOLID": "friction"},
    "deformation-damping": {
        "SHELL": "deformation_damping",
        "SOLID": "deformation_damping",
    },
    "bending-damping": {"SHELL": "bending_damping"},
    "strain-limit": {"SHELL": "strain_limit_percent"},
    "plasticity": {"SHELL": "plasticity", "SOLID": "plasticity"},
    "bend-plasticity": {"SHELL": "bend_plasticity"},
    "bend-warp": {"SHELL": "bend_warp"},
    "bend-weft": {"SHELL": "bend_weft"},
}

# Keys whose UI value is not the solver's value.
MATERIAL_SCALE = {
    "strain-limit": 0.01,  # authored in percent
}

# Keys a feature checkbox switches off. A closed gate means the solver value is
# zero for the whole solve, whatever the slider, the keyframe or a map target
# holds.
MATERIAL_GATE = {
    "pressure": "enable_inflate",
    "strain-limit": "enable_strain_limit",
    "plasticity": "enable_plasticity",
    "plasticity-threshold": "enable_plasticity",
    "bend-plasticity": "enable_bend_plasticity",
    "bend-plasticity-threshold": "enable_bend_plasticity",
}

# The addon property holding a group's density, for the types that carry one.
_DENSITY_PROP = {
    "SOLID": "solid_density",
    "SHELL": "shell_density",
    "ROD": "rod_density",
}

def pin_tracks_rest_shape(group, pin_item) -> bool:
    """Whether `pin_item` streams its captured deformation as `group`'s rest shape.

    The ONE predicate for it, which the encoder, the panel and the refusal
    below all ask: Track Rest-Pose Deformation on a captured pin of a SOLID
    group. The encoder refuses the toggle on a pin that does not hold every
    vertex, so these flags are the whole answer once an encode succeeds.
    """
    return (
        group.object_type == "SOLID"
        and bool(getattr(pin_item, "track_rest_pose_deformation", False))
        and bool(getattr(pin_item, "has_captured_anim", False))
    )


def rest_shape_plasticity_conflict(group):
    """The pin whose tracked rest shape collides with `group`'s plasticity, or None.

    Plasticity creeps the rest shape each step and a tracked capture replaces
    it each frame, so the two cannot both hold it: the solver overwrites the
    rest shape wholesale on the assumption the frontend never ships both.
    """
    if not getattr(group, "enable_plasticity", False):
        return None
    for pin_item in group.pin_vertex_groups:
        if pin_tracks_rest_shape(group, pin_item):
            return pin_item
    return None


# Where a map's weights are read from. NUMERIC IDS ARE PERMANENT, for the same
# reason `MATERIAL_MAP_KEYS`' ids are: this enum lives on a SAVED PropertyGroup.
SOURCE_TYPE_ITEMS = [
    ("VERTEX_GROUP", "Vertex Group", "Read weights from a vertex group", 1),
    ("ATTRIBUTE", "Attribute", "Read weights from a float attribute", 2),
]


def enum_items():
    """EnumProperty items for the parameter a map drives."""
    return [(key, label, desc, num) for key, label, desc, num in MATERIAL_MAP_KEYS]


def base_property(key: str, object_type: str):
    """The addon property holding `key`'s base value, or None if the object
    type has no such parameter (a rod has no inflation pressure)."""
    return MATERIAL_MAP_BASE_PROP.get(key, {}).get(object_type)


def density_prop(object_type: str):
    """The addon property holding `object_type`'s density, or None."""
    return _DENSITY_PROP.get(object_type)


def gate_open(group, key: str) -> bool:
    """Whether `key` reaches the solver as an authored value on `group`."""
    gate = MATERIAL_GATE.get(key)
    if gate is not None and not getattr(group, gate):
        return False
    if key == "strain-limit" and group.object_type == "SHELL":
        # The strain-limit solver bakes its rest shape assuming unit scaling,
        # so a shrink factor other than 1 leaves the limit undefined.
        if group.shrink_x != 1.0 or group.shrink_y != 1.0:
            return False
    return True


def wants_anisotropic_bending(group) -> bool:
    """Whether `group` asks for directional bending anywhere.

    The sliders are not the whole answer. The natural way to paint anisotropy
    into a region is to leave the slider at zero and give a `bend-warp` or
    `bend-weft` map a positive target, and the solver refuses directional
    bending on a mesh with no UV direction however it was authored. A predicate
    that watched only the sliders would leave that case unwarned until the
    build refused it.
    """
    if group.bend_warp > 0.0 or group.bend_weft > 0.0:
        return True
    return any(
        entry.enabled
        and entry.parameter in ("bend-warp", "bend-weft")
        and entry.target_value > 0.0
        for entry in getattr(group, "material_maps", [])
    )


def gate_reason(group, key: str):
    """Why `key` is switched off on `group`, or None when it is not.

    Names the condition an artist can act on, which is not always the key's own
    checkbox: a shrunk shell and a captured rest shape each close a gate that
    no single boolean describes. Plasticity against a tracked rest shape is
    not a closed gate but a refused encode (`rest_shape_plasticity_conflict`).
    """
    gate = MATERIAL_GATE.get(key)
    if gate is not None and not getattr(group, gate):
        return f"'{gate}' is off"
    if key == "strain-limit" and group.object_type == "SHELL":
        if group.shrink_x != 1.0 or group.shrink_y != 1.0:
            return "the shell is shrunk, which leaves the strain limit undefined"
    return None


def to_solver_value(group, key: str, ui_value: float):
    """`key`'s value in solver units on `group`, or None when its gate is closed.

    One definition for three callers: the group's static parameter, a sampled
    keyframe, and a spatial map's target. A map blends between the first (or the
    second) and the third, so a conversion applied to one end and not the other
    would blend across two different units.
    """
    if not gate_open(group, key):
        return None
    value = float(ui_value) * MATERIAL_SCALE.get(key, 1.0)
    if key == "young-mod":
        prop = density_prop(group.object_type)
        # The solver consumes young-mod density-normalized, as Pa/rho. When the
        # group's field holds a true Young's modulus in pascals instead, divide
        # it here. The density UI minimum is 0.01, so the division is defined.
        if prop is not None and not group.young_mod_density_normalized:
            value = value / float(getattr(group, prop))
    return value
