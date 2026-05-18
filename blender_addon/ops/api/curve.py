"""``_CurveBuilder``: build a multi-spline Bezier curve object.

See :mod:`blender_addon.ops.api` for the package overview.
"""

import bpy  # pyright: ignore

from .._api_markers import blender_api


def _remove_if_exists(name: str) -> None:
    obj = bpy.data.objects.get(name)
    if obj is None:
        return
    data = obj.data
    bpy.data.objects.remove(obj, do_unlink=True)
    if isinstance(data, bpy.types.Curve) and data.users == 0:
        bpy.data.curves.remove(data)


@blender_api
class _CurveBuilder:
    """Builder for a multi-spline Bezier curve object.

    Created via :meth:`Solver.create_curve`.  Each :meth:`add_spline`
    appends one Bezier spline to the underlying curve datablock;
    :meth:`finalize` links the resulting object into the active scene
    and returns it.

    Pin definition is *not* part of this builder.  Pass the
    control-point indices to :meth:`Group.create_pin` instead, which
    writes the ``_pin_<name>`` custom property and registers the pin
    in one call.

    Example::

        curve = solver.create_curve("WovenCylinder", bevel_depth=3e-3)
        for points, closed in strands:
            curve.add_spline(points, closed=closed)
        obj = curve.finalize()

        rod = solver.create_group("Strands", type="ROD")
        rod.add(obj.name)
        rod.create_pin(obj.name, "left", indices=left_indices)
    """

    def __init__(self, name: str, *, bevel_depth: float = 0.0,
                 bevel_resolution: int = 2, resolution_u: int = 4,
                 dimensions: str = "3D", clear_existing: bool = True):
        if clear_existing:
            _remove_if_exists(name)
        curve = bpy.data.curves.new(name=name, type="CURVE")
        curve.dimensions = dimensions
        curve.resolution_u = resolution_u
        curve.bevel_depth = bevel_depth
        curve.bevel_resolution = bevel_resolution
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_curve", curve)
        object.__setattr__(self, "_finalized", False)

    @property
    @blender_api
    def name(self) -> str:
        """Object name this builder will create on :meth:`finalize`."""
        return object.__getattribute__(self, "_name")

    @blender_api
    def add_spline(self, points, *, closed: bool = False) -> int:
        """Append a Bezier spline with AUTO handles.

        Args:
            points: Iterable of ``(x, y, z)`` control-point coordinates
                (a NumPy array of shape ``(n, 3)`` works).
            closed: Set ``True`` to make the spline cyclic.

        Returns:
            Zero-based index of the new spline within this curve.  Use
            it with :meth:`set_material`.

        Raises:
            ValueError: If ``points`` has fewer than two coordinates.
        """
        curve = object.__getattribute__(self, "_curve")
        n = len(points)
        if n < 2:
            raise ValueError("add_spline requires at least 2 points")
        spline = curve.splines.new(type="BEZIER")
        spline.bezier_points.add(n - 1)
        for bp, p in zip(spline.bezier_points, points):
            bp.co = (float(p[0]), float(p[1]), float(p[2]))
            bp.handle_left_type = "AUTO"
            bp.handle_right_type = "AUTO"
        spline.use_cyclic_u = bool(closed)
        return len(curve.splines) - 1

    @blender_api
    def set_material(self, spline_index: int,
                     material: "bpy.types.Material") -> "_CurveBuilder":
        """Bind a material to a spline by index.

        The material is appended to the curve's slots if it isn't
        already present.  Pre-existing slots are reused so repeated
        calls with the same material don't grow the slot list.

        Args:
            spline_index: Index returned by :meth:`add_spline`.
            material: An existing ``bpy.types.Material``.  Create it
                with ``bpy.data.materials.new(...)`` before calling.

        Returns:
            ``self`` for chaining.

        Raises:
            IndexError: If ``spline_index`` is out of range.
        """
        curve = object.__getattribute__(self, "_curve")
        if spline_index < 0 or spline_index >= len(curve.splines):
            raise IndexError(
                f"spline index {spline_index} out of range "
                f"(curve has {len(curve.splines)} splines)"
            )
        slot = curve.materials.find(material.name)
        if slot < 0:
            curve.materials.append(material)
            slot = len(curve.materials) - 1
        curve.splines[spline_index].material_index = slot
        return self

    @blender_api
    def finalize(self) -> "bpy.types.Object":
        """Create the ``bpy.types.Object``, link it to the scene, and
        return it.

        Raises:
            RuntimeError: If called more than once on the same builder.
        """
        if object.__getattribute__(self, "_finalized"):
            raise RuntimeError(
                f"curve '{object.__getattribute__(self, '_name')}' "
                "has already been finalized"
            )
        name = object.__getattribute__(self, "_name")
        curve = object.__getattribute__(self, "_curve")
        obj = bpy.data.objects.new(name, curve)
        bpy.context.scene.collection.objects.link(obj)
        object.__setattr__(self, "_finalized", True)
        return obj

    def __repr__(self):
        return f"_CurveBuilder({self._name!r})"
