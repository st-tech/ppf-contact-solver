# File: _preconditioner_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""The linear solver's preconditioners, and the scenes each one is refused on.

The ``precond`` parameter names the preconditioner of the conjugate gradient
solve that every Newton step runs:

* ``"block-jacobi"`` (the default) inverts each vertex's own 3x3 block. An
  iteration is as cheap as one can be, and a fine or stiff mesh takes
  hundreds of them.
* ``"schwarz"`` is multilevel additive Schwarz over aggregates of vertices,
  with ``schwarz-levels`` levels.
* ``"multigrid"`` is a geometric multigrid V-cycle over grids laid on each
  connected piece of mesh. It takes far fewer iterations at several times the
  cost of each, so it pays on large cloth and soft bodies and loses on small
  scenes and on stacked solids. It is not offered beside a rigid body.

Cosserat rods are solved with their frames in one augmented system, which
carries a preconditioner of its own, so a scene that holds one takes
``"block-jacobi"`` only.

A name outside that list, the multigrid in a scene that holds a rigid body,
and anything but block-Jacobi in a scene that holds a Cosserat rod are
refused when the session is built. The solver would stop the last two when
the run starts, after the scene has been exported, and the first is a
spelling a run cannot be allowed to answer with another preconditioner.
"""

import numpy as np

# Every name ``precond`` takes, in the order the documentation lists them.
PRECONDITIONERS = ("block-jacobi", "schwarz", "multigrid")


def refuse_unsolvable_preconditioner(scene, param) -> None:
    """Refuse ``param``'s preconditioner when ``scene`` cannot be solved with it.

    Args:
        scene: A ``FixedScene``.
        param: The session's ``ParamManager``.

    Raises:
        ValueError: ``precond`` is not one of :data:`PRECONDITIONERS`, or it
            is ``"multigrid"`` and the scene holds a rigid (PDRD) body, or it
            is not ``"block-jacobi"`` and the scene holds a Cosserat rod. The
            message names the choices in the first case, the bodies and both
            remedies in the second, and the segment count in the third.

    Example:
        Check a scene against the preconditioner before building a session::

            fixed = scene.build()
            session.param.set("precond", "multigrid")
            refuse_unsolvable_preconditioner(fixed, session.param)
    """
    names = param.values("precond")
    for name in names:
        if name not in PRECONDITIONERS:
            choices = ", ".join(repr(choice) for choice in PRECONDITIONERS)
            raise ValueError(
                f"precond is {name!r}, which is not a preconditioner. "
                f"Choose one of {choices}."
            )
    # A scene built before Cosserat rods existed has no marker array at all.
    marker = getattr(scene, "_cosserat_edge", None)
    segments = 0 if marker is None else int(np.count_nonzero(marker))
    if segments:
        for name in names:
            if name != "block-jacobi":
                raise ValueError(
                    f"precond is {name!r}, which does not solve a scene with "
                    f"Cosserat rods, and this scene has {segments} Cosserat "
                    "segments. Their frames are solved with the vertices in "
                    "one system that carries its own preconditioner: set "
                    "precond to 'block-jacobi'."
                )
    if "multigrid" not in names:
        return
    rigid = [p.label for p in scene.contact_gap_participants if p.role == "rigid"]
    if not rigid:
        return
    shown = ", ".join(repr(label) for label in rigid[:4])
    if len(rigid) > 4:
        shown += f" and {len(rigid) - 4} more"
    raise ValueError(
        "precond is 'multigrid', which does not solve a scene with rigid "
        f"bodies, and this scene has {len(rigid)}: {shown}. Set precond to "
        "'block-jacobi' or 'schwarz', or make those objects deformable."
    )
