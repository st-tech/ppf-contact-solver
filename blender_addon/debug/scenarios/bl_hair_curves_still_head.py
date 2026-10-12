# File: scenarios/bl_hair_curves_still_head.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Hair Curves rooted on a head that does not move (issue #146). The driver
# and its six checks are bl_hair_curves_scalp's, run with a head that
# stays where it is.
#
# It is its own scenario because the collider is a different thing in the
# solver. A STATIC that moves is decoded to a shell of pinned vertices in
# the solved mesh; one that stays still is the collision mesh, a separate
# namespace with its own contact visitors, sweeps and intersection scan.
# A strand's first segment is pinned and starts ON a face of the head, so
# both the build gate and the solver have to treat that pair as one with
# nothing free to move in each namespace, and neither run says anything
# about the other. A still head is also the case where nothing moves the
# hair: Capture Deformation is not offered and the roots are plain pins.

from __future__ import annotations

from . import _runner as r
from . import bl_hair_curves_scalp as scalp


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND: the contact and the pins are the solver's.
BACKENDS = ("real",)


def build_driver(ctx: r.ScenarioContext) -> str:
    return scalp.build_driver_for(
        ctx, travel=0.0, project="hair_curves_still_head",
    )


def run(ctx: r.ScenarioContext) -> dict:
    return scalp.run(ctx)
