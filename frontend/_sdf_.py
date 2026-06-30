# File: _sdf_.py
# SDF (Signed Distance Field) implementation backed by the Rust kernel.

import numpy as np

from . import _rust  # type: ignore[attr-defined]


# SDF type constants. These integer tags must match the kernel-side
# tags in crates/ppf-cts-core/src/kernels/sdf.rs; keep both in sync.
SDF_SPHERE = 0
SDF_CAPSULE = 1


def eval_sdf_grid(xs, ys, zs, sdf_types, sdf_params):
    """Evaluate the SDF union on the full (xs, ys, zs) grid.

    Output shape is ``(nx, ny, nz)`` float64 ndarray.
    """
    return _rust.eval_sdf_grid(
        np.ascontiguousarray(xs, dtype=np.float64),
        np.ascontiguousarray(ys, dtype=np.float64),
        np.ascontiguousarray(zs, dtype=np.float64),
        np.ascontiguousarray(sdf_types, dtype=np.int32),
        np.ascontiguousarray(sdf_params, dtype=np.float64),
    )


def marching_cubes(sdf_func, bounds, step):
    """Extract a triangle mesh from an SDF using marching cubes.

    Args:
        sdf_func: SDF object providing a get_kernel_primitives() method.
        bounds: ((min_x, min_y, min_z), (max_x, max_y, max_z)).
        step: Grid step size.

    Returns:
        (vertices, faces) as numpy arrays. When no surface is found,
        empty arrays of shapes (0, 3) and (0, 3) are returned.
    """
    (min_x, min_y, min_z), (max_x, max_y, max_z) = bounds

    xs = np.arange(min_x, max_x + step, step)
    ys = np.arange(min_y, max_y + step, step)
    zs = np.arange(min_z, max_z + step, step)

    try:
        sdf_types, sdf_params = sdf_func.get_kernel_primitives()
    except NotImplementedError as exc:
        # Intersection (``&``) and difference (``-``) composites have no
        # kernel meshing path: the evaluator only aggregates a union of
        # primitives. Re-raise with an actionable message instead of the
        # bare "not yet supported" from get_kernel_primitives().
        raise NotImplementedError(
            f"{type(sdf_func).__name__} cannot be meshed: only union (|) of "
            "spheres and capsules is supported by the kernel evaluator. "
            "Intersection (&) and difference (-) have no meshing path yet."
        ) from exc

    return _rust.marching_cubes(
        np.ascontiguousarray(xs, dtype=np.float64),
        np.ascontiguousarray(ys, dtype=np.float64),
        np.ascontiguousarray(zs, dtype=np.float64),
        float(step),
        np.ascontiguousarray(sdf_types, dtype=np.int32),
        np.ascontiguousarray(sdf_params, dtype=np.float64),
    )


class SDF:
    """Base class for signed distance functions.

    Primitives (``SphereSDF``, ``CapsuleSDF``) and composite operators
    (``UnionSDF``, ``IntersectionSDF``, ``DifferenceSDF``) all inherit
    from ``SDF``. Composite shapes are built with the operator overloads
    ``|`` (union), ``&`` (intersection), and ``-`` (difference). Use
    :meth:`save` to mesh the field and write it to disk.

    Example:
        Build a squishy ball with capsule spikes and save it as a mesh
        (pattern from ``examples/trapped.ipynb``)::

            from frontend import sdf

            shape = sdf.sphere(1.1)
            shape = shape | sdf.capsule([-1, 0, 0], [1, 0, 0], 0.05)
            shape.save("squishy.ply", step=0.03)
    """

    def __or__(self, other):
        """Return the union of this SDF with another."""
        return UnionSDF(self, other)

    def __and__(self, other):
        """Return the intersection of this SDF with another."""
        return IntersectionSDF(self, other)

    def __sub__(self, other):
        """Return the difference of this SDF with another (self minus other)."""
        return DifferenceSDF(self, other)

    def bounds(self):
        """Return the bounding box as ((min_x, min_y, min_z), (max_x, max_y, max_z))."""
        raise NotImplementedError

    def get_kernel_primitives(self):
        """Return (sdf_types, sdf_params) arrays for the kernel evaluator."""
        raise NotImplementedError

    def save(self, path, step=0.01):
        """Mesh this SDF with marching cubes and export it to a file."""
        import trimesh

        b = self.bounds()
        verts, faces = marching_cubes(self, b, step)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.export(path)


class SphereSDF(SDF):
    """Signed distance field of a sphere centered at ``center`` with the given radius."""

    def __init__(self, radius, center=(0, 0, 0)):
        self.radius = float(radius)
        self.cx, self.cy, self.cz = float(center[0]), float(center[1]), float(center[2])

    def bounds(self):
        return _rust.sphere_bounds(self.cx, self.cy, self.cz, self.radius)

    def get_kernel_primitives(self):
        return _rust.pack_sdf_primitive(
            SDF_SPHERE,
            (self.cx, self.cy, self.cz, self.radius, 0.0, 0.0, 0.0, 0.0),
        )


class CapsuleSDF(SDF):
    """Signed distance field of a capsule (swept sphere along a line segment)."""

    def __init__(self, p0, p1, radius):
        self.p0x, self.p0y, self.p0z = float(p0[0]), float(p0[1]), float(p0[2])
        self.p1x, self.p1y, self.p1z = float(p1[0]), float(p1[1]), float(p1[2])
        self.bax = self.p1x - self.p0x
        self.bay = self.p1y - self.p0y
        self.baz = self.p1z - self.p0z
        self.ba_dot_ba = self.bax * self.bax + self.bay * self.bay + self.baz * self.baz
        self.radius = float(radius)

    def bounds(self):
        return _rust.capsule_bounds(
            self.p0x, self.p0y, self.p0z,
            self.p1x, self.p1y, self.p1z,
            self.radius,
        )

    def get_kernel_primitives(self):
        return _rust.pack_sdf_primitive(
            SDF_CAPSULE,
            (
                self.p0x, self.p0y, self.p0z,
                self.bax, self.bay, self.baz,
                self.ba_dot_ba, self.radius,
            ),
        )


class UnionSDF(SDF):
    """Union of two or more SDFs (the ``|`` operator).

    Nested unions are automatically flattened so the kernel evaluator can
    scan all primitives in a single pass.
    """

    def __init__(self, a, b):
        # Flatten via list concatenation rather than a per-operand
        # ``append/extend`` loop: each ``children`` slice exists fully
        # before the join, so no list grows incrementally.
        a_children = list(a.children) if isinstance(a, UnionSDF) else [a]
        b_children = list(b.children) if isinstance(b, UnionSDF) else [b]
        self.children = a_children + b_children

    def bounds(self):
        return _rust.reduce_bounds_from_children(self.children, "union")

    def get_kernel_primitives(self):
        # The full (types, params) walk + concatenation runs inside
        # Rust: it calls ``get_kernel_primitives()`` on each child via PyO3
        # and accumulates into a single flat buffer, so no
        # intermediate Python list grows on the way to the kernel
        # evaluator.
        return _rust.concat_sdf_primitives_from_children(self.children)


class IntersectionSDF(SDF):
    """Intersection of two or more SDFs (the ``&`` operator).

    The kernel evaluator only aggregates a union of primitives, so
    intersections have no meshing path yet: :meth:`get_kernel_primitives`
    raises ``NotImplementedError`` and :meth:`save` therefore cannot
    export an intersection. The ``bounds`` query still works.
    """

    def __init__(self, a, b):
        # Flatten via list concatenation rather than a per-operand
        # ``append/extend`` loop: each ``children`` slice exists fully
        # before the join, so no list grows incrementally.
        a_children = list(a.children) if isinstance(a, IntersectionSDF) else [a]
        b_children = list(b.children) if isinstance(b, IntersectionSDF) else [b]
        self.children = a_children + b_children

    def bounds(self):
        return _rust.reduce_bounds_from_children(self.children, "intersect")

    def get_kernel_primitives(self):
        raise NotImplementedError("Intersection not yet supported with the kernel evaluator")


class DifferenceSDF(SDF):
    """Difference ``a - b`` of two SDFs (the ``-`` operator)."""

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def bounds(self):
        return self.a.bounds()

    def get_kernel_primitives(self):
        raise NotImplementedError("Difference not yet supported with the kernel evaluator")


# Convenience functions.

def sphere(radius, center=(0, 0, 0)):
    """Create a sphere SDF with the given radius and center."""
    return SphereSDF(radius, center)


def capsule(p0, p1, radius):
    """Create a capsule SDF with axis endpoints p0 and p1 and the given radius."""
    return CapsuleSDF(p0, p1, radius)
