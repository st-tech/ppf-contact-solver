"""Object-oriented API for the ZOZO Contact Solver.

Usage in Blender scripts::

    from bl_ext.user_default.ppf_contact_solver.ops.api import solver

    # Scene parameters (attribute access)
    solver.param.step_size = 0.004
    solver.param.gravity = (0, 0, -9.8)
    solver.param.project_name = "my_project"

    # Group creation returns a Group proxy
    shell = solver.create_group("Cloth", type="SHELL")
    shell.add("Plane")
    shell.param.shell_density = 1.0
    shell.param.friction = 0.5

    # Pin creation returns a Pin proxy; methods are chainable
    left = shell.create_pin("Plane", "left")
    left.move_by(delta=(0, 0, 1.0), frame_start=1, frame_end=60)

    # Rod scenes: build a Bezier curve, then pin its control points
    # directly on the group.
    curve = solver.create_curve("Strands", bevel_depth=3e-3)
    for points, closed in strands:
        curve.add_spline(points, closed=closed)
    obj = curve.finalize()
    rod = solver.create_group("Rod", type="ROD")
    rod.add(obj.name)
    rod.create_pin(obj.name, "left", indices=left_indices)

    # Connection (falls through to bpy.ops)
    solver.connect()
"""

from .collider import (
    _ColliderParamProxy,
    _InvisibleSphereBuilder,
    _InvisibleWallBuilder,
)
from .curve import _CurveBuilder
from .dynamics import _DynParamBuilder, _SceneProxy
from .group import _Group, _ParamProxy
from .pin import _Pin
from .solver import _Solver, solver


# ---------------------------------------------------------------------------
# Public aliases used by the documentation generator.
#
# The underscore-prefixed class names above are the runtime identifiers;
# these aliases are what the docs generator references and what scripts
# should use for type hints (``group: Group = solver.create_group(...)``).
# Runtime behavior is unchanged; aliases are simple name bindings.
# ---------------------------------------------------------------------------

Solver = _Solver
Group = _Group
Pin = _Pin
Curve = _CurveBuilder
SceneParam = _SceneProxy
GroupParam = _ParamProxy
ColliderParam = _ColliderParamProxy
DynParam = _DynParamBuilder
Wall = _InvisibleWallBuilder
Sphere = _InvisibleSphereBuilder


__all__ = [
    "solver",
    "Solver",
    "Group",
    "Pin",
    "Curve",
    "SceneParam",
    "GroupParam",
    "ColliderParam",
    "DynParam",
    "Wall",
    "Sphere",
    # Private names re-exported so existing callers (e.g. core.mutation,
    # debug scenarios) keep working unchanged.
    "_Solver",
    "_Group",
    "_Pin",
    "_CurveBuilder",
    "_SceneProxy",
    "_ParamProxy",
    "_ColliderParamProxy",
    "_DynParamBuilder",
    "_InvisibleWallBuilder",
    "_InvisibleSphereBuilder",
]
