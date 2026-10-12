"""Seed a small Hair Curves scene for documentation screenshots.

Passed to ``blender_addon/capture.sh --pre-python``, this runs inside
Blender after the add-on has registered and before the sidebar is set up::

    bash blender_addon/capture.sh -o docs/blender_addon/images/hair \\
        --panel-only --pre-python docs/tools/capture_hair.py \\
        --all DYNAMICS_PT_Groups

Why this exists: the Pins section and the Spatial Material Maps box draw
lines for hair that no mesh scene reaches. A weighted pin shows what its
weights do under the Pull row, and a Rod group draws the map box at all only
since it takes maps.

The scene is one Hair Curves object of a few straight strands in a Rod
group, with the two pins and the one map the Hair Curves page describes:

- ``Hold``, a weighted pin that pulls, listed above
- ``Roots``, the derived pin that holds the first segment of every strand, and
- a Bending Stiffness map read from the attribute ``stiff``.

``PPF_CAPTURE_HAIR_SECTION`` picks what the group box shows expanded:
``pins`` (the default) or ``material``.
"""

import os

import bpy
import numpy as np

from bl_ext.user_default.ppf_contact_solver.core import hair_curves
from bl_ext.user_default.ppf_contact_solver.models.groups import get_addon_data
from bl_ext.user_default.ppf_contact_solver.ops.api import solver

HAIR = "Fur"
STRANDS = 12
POINTS = 8
SECTION = os.environ.get("PPF_CAPTURE_HAIR_SECTION", "pins")


def _clear_default_scene() -> None:
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _build_hair():
    curves = bpy.data.hair_curves.new(HAIR)
    curves.add_curves([POINTS] * STRANDS)
    points = [
        (0.05 * strand, 0.0, 0.03 * point)
        for strand in range(STRANDS)
        for point in range(POINTS)
    ]
    curves.attributes["position"].data.foreach_set(
        "vector", np.array(points, dtype=np.float32).ravel()
    )
    obj = bpy.data.objects.new(HAIR, curves)
    bpy.context.collection.objects.link(obj)
    return obj


def _weights(falloff_points: int) -> np.ndarray:
    """One weight per point: 1 on the first segment of each strand, falling
    in equal steps to 0 after ``falloff_points`` points."""
    along = np.array(
        [
            1.0 - max(0, k - 1) / max(falloff_points - 1, 1) if k < falloff_points else 0.0
            for k in range(POINTS)
        ],
        dtype=np.float32,
    )
    return np.tile(along, STRANDS)


def main() -> None:
    _clear_default_scene()
    hair = _build_hair()

    group = solver.create_group(HAIR, type="ROD")
    group.add(HAIR)

    # The weighted pin first and the roots after it: the lower pin in the
    # list wins a point two pins share.
    hair_curves.store_pin_weights(hair, "Hold", _weights(POINTS // 2))
    group.create_pin(HAIR, "Hold").pull(strength=300.0)
    group.create_pin(HAIR, hair_curves.ROOTS_PIN_NAME)

    stiff = hair.data.attributes.new("stiff", "FLOAT", "POINT")
    stiff.data.foreach_set("value", _weights(POINTS))

    slot = get_addon_data(bpy.context.scene).object_group_0
    row = slot.material_maps.add()
    row.parameter = "bend"
    row.source_type = "ATTRIBUTE"
    row.source_name = "stiff"
    row.target_value = 40.0
    slot.material_maps_index = 0

    slot.show_group = True
    slot.show_pin = SECTION == "pins"
    slot.show_parameters = SECTION == "material"
    slot.pin_vertex_groups_index = 0

    bpy.context.view_layer.objects.active = hair
    hair.select_set(True)
    print(f"capture_hair: {HAIR} (ROD), pins "
          f"{[p.name for p in slot.pin_vertex_groups]}, section {SECTION}")


main()
