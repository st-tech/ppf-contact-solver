"""Seed a small, representative scene for documentation screenshots.

Passed to ``blender_addon/capture.sh --pre-python``, this runs inside
Blender after the add-on has registered and before the sidebar is set up::

    bash blender_addon/capture.sh -o docs/blender_addon/images/tour \\
        --panel-only --open-closed-panels \\
        --pre-python docs/tools/capture_scene.py \\
        --all SSH_PT_ObjectGroupsManager --all DYNAMICS_PT_Groups

Why this exists: several sidebar panels draw nothing useful in a fresh
Blender.  Scene Configuration lists groups, Dynamics Groups lists the
objects assigned to them, and Object Statistics says "No dynamics
objects" until something is assigned.  Screenshotting the empty state
documents the add-on as if it did nothing, so the pages that show those
panels need a scene behind them.

The scene is deliberately tiny and generic — a draped sheet over a
sphere, the smallest setup that produces one deformable group and one
collider — so the screenshots stay legible at documentation width and do
not date as fast as a scene tied to a specific example would.

Two environment variables let one fixture serve pages that name their
objects differently, rather than each growing its own near-copy of this
file:

``PPF_CAPTURE_CLOTH``
    Name of the deformable object and its SHELL group (default ``Cloth``).

``PPF_CAPTURE_PIN``
    When set, the cloth's top edge is put in a vertex group of this name
    and registered as a pin, which is what the Pins section needs in
    order to draw a populated list.

``PPF_CAPTURE_COLLIDER``
    ``0`` builds the cloth alone.  Some figures are captioned as showing
    one group, and a second group below it would contradict the caption.
"""

import os

import bpy

from bl_ext.user_default.ppf_contact_solver.ops.api import solver

CLOTH = os.environ.get("PPF_CAPTURE_CLOTH", "Cloth")
PIN = os.environ.get("PPF_CAPTURE_PIN", "")
COLLIDER = "Sphere"
WITH_COLLIDER = os.environ.get("PPF_CAPTURE_COLLIDER", "1") != "0"


def _clear_default_scene() -> None:
    """Remove the startup cube, camera and light.

    The cube in particular is a trap: it is selected on startup, so a
    panel that reports on "selected mesh objects" counts it and the
    screenshot then disagrees with the scene the caption describes.
    """
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _build_meshes() -> None:
    # A grid rather than a plane: the solver reads obj.data.vertices, so a
    # single quad would show up in the panels as a 4-vertex object and make
    # the vertex counts in the screenshots look wrong.
    bpy.ops.mesh.primitive_grid_add(
        size=2.0, x_subdivisions=24, y_subdivisions=24, location=(0.0, 0.0, 1.2)
    )
    bpy.context.active_object.name = CLOTH

    if WITH_COLLIDER:
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=0.6, segments=32, ring_count=16, location=(0.0, 0.0, 0.0)
        )
        bpy.context.active_object.name = COLLIDER


def _add_top_edge_vertex_group(name: str) -> None:
    """Put the cloth's +Y edge in a vertex group called *name*.

    A pin has to name a vertex group that already exists on the mesh, and
    an edge is the shape a reader recognises as "hung from the top".
    """
    obj = bpy.data.objects[CLOTH]
    max_y = max(v.co.y for v in obj.data.vertices)
    edge = [v.index for v in obj.data.vertices if abs(v.co.y - max_y) < 1e-5]
    obj.vertex_groups.new(name=name).add(edge, 1.0, "REPLACE")


def _assign_groups() -> None:
    shell = solver.create_group(CLOTH, type="SHELL")
    shell.add(CLOTH)
    if PIN:
        _add_top_edge_vertex_group(PIN)
        shell.create_pin(CLOTH, PIN)

    if WITH_COLLIDER:
        static = solver.create_group(COLLIDER, type="STATIC")
        static.add(COLLIDER)


def main() -> None:
    _clear_default_scene()
    _build_meshes()
    _assign_groups()
    pinned = f", pinned at {PIN}" if PIN else ""
    over = f" over {COLLIDER} (STATIC)" if WITH_COLLIDER else ""
    print(f"capture_scene: {CLOTH} (SHELL){over}{pinned}")


main()
