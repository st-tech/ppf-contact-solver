# File: _contact_gap_limit_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""The largest contact gap a mesh's own neighbors allow, and the limit under it.

The solver forms a contact pair between a vertex and every vertex and edge of
its own mesh that it is not part of, the opposite edge of its own triangle and
the far end of its own rod segment included. A pair engages once its distance
is below the mean of the two sides' contact gaps plus the sum of their contact
offsets. So a gap wider than the distance between mesh neighbors, less the
offsets, starts the run with those neighbors already pressing on each other:
the barrier pushes them apart, the elasticity pulls them back, and the mesh
swells and stiffens against bending. Gomez-Nogales et al. (2025) name this
contact locking and remove it by lowering the activation distance of the pairs
whose distance in the undeformed mesh is below it.

Here that distance is taken per element, and an object's ``contact-gap-limit``
holds each element's gap to a fraction of it:

    clearance = smallest size among the element and its vertex neighbors
    room      = clearance - 2 * (the object's largest contact offset)
    gap       = min(authored gap, fraction * room)

It is measured on the mesh as it enters the solve and only among elements that
share a vertex, so two parts of a mesh that merely start close together are
contact, not locking, and keep the authored gap.

Reference:
    Gonzalo Gomez-Nogales, Zhen Chen, Rosalie Martin, Elena Garces and
    Danny M. Kaufman. "Unlocking Thickness Modeling for Codimensional Contact
    Simulation." arXiv:2510.10256, 2025. https://arxiv.org/abs/2510.10256
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

# The fractions a limit accepts. Zero is the registry's default and means no
# limit; any other value has to leave the neighbors room on both sides.
LIMIT_RANGE = (0.01, 0.5)

_KIND = {2: "segment", 3: "triangle"}


def validated_fraction(value, label: str) -> float:
    """``value`` as a limit fraction: 0.0 for no limit, or inside the range.

    Args:
        value: The object's ``contact-gap-limit``.
        label: The name a refusal gives the object.

    Raises:
        ValueError: ``value`` is not finite, or is neither 0.0 nor inside
            :data:`LIMIT_RANGE`.
    """
    fraction = float(value)
    low, high = LIMIT_RANGE
    if fraction == 0.0 or low <= fraction <= high:
        return fraction
    raise ValueError(
        f"'{label}' has contact-gap-limit {fraction:g}; it is a fraction "
        f"between {low:g} and {high:g}, or 0 for no limit"
    )


def element_clearance(vert: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """How close each element's mesh neighbors start, one length per element.

    An element's own size is the distance of the closest pair inside it: a
    triangle's smallest altitude, where a vertex meets the opposite edge, and
    a rod segment's length, where its two vertices meet. A degenerate triangle
    has size zero.

    The clearance is the smallest size among the element and every element
    sharing a vertex with it. The solver gives a vertex the mean of the gaps
    of the elements around it and an edge the mean of its faces', so a gap
    sized from one element alone would reach a smaller neighbor through the
    vertex they share. With the minimum taken over the vertex neighborhood,
    every vertex and edge gap is at most the fraction of every pair distance
    around it.

    Args:
        vert: The vertices, shape ``(n, 3)``.
        elements: A triangle table, shape ``(m, 3)``, or a rod segment table,
            shape ``(m, 2)``.

    Returns:
        The clearance of each element, shape ``(m,)``.
    """
    vert = np.asarray(vert, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)
    if elements.ndim != 2 or elements.shape[1] not in _KIND:
        raise ValueError(
            "a clearance is measured over triangles or rod segments, not a "
            f"table of shape {elements.shape}"
        )
    if elements.shape[1] == 2:
        size = np.linalg.norm(vert[elements[:, 1]] - vert[elements[:, 0]], axis=1)
    else:
        a, b, c = (vert[elements[:, k]] for k in range(3))
        longest = np.maximum.reduce([
            np.linalg.norm(b - a, axis=1),
            np.linalg.norm(c - b, axis=1),
            np.linalg.norm(a - c, axis=1),
        ])
        doubled_area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
        # The altitude onto the longest edge, whose foot always lies inside
        # that edge, so it is a vertex-to-edge distance and not a
        # vertex-to-line one.
        size = np.divide(
            doubled_area, longest, out=np.zeros_like(longest), where=longest > 0.0
        )
    around = np.full(len(vert), np.inf)
    np.minimum.at(around, elements, size[:, None])
    return around[elements].min(axis=1)


@dataclass(frozen=True)
class GapLimit:
    """One object's contact gap limit, resolved onto its elements.

    Attributes:
        fraction: The object's ``contact-gap-limit``.
        offset: The object's largest contact offset over the run.
        kind: ``"triangle"`` or ``"segment"``, what an element is.
        room: Per element, the clearance less twice the offset: the largest
            gap at which the element's mesh neighbors are not engaged.
        bound: Per element, ``fraction * room``, and infinity where there is
            no room.
    """

    fraction: float
    offset: float
    kind: str
    room: np.ndarray
    bound: np.ndarray

    def apply(self, gap) -> np.ndarray:
        """``gap``, one value per element, held to the bound."""
        return np.minimum(np.asarray(gap, dtype=np.float64), self.bound)

    def tightest(self) -> Optional[int]:
        """The element with the smallest bound, or None when none is bounded."""
        index = int(np.argmin(self.bound))
        return index if np.isfinite(self.bound[index]) else None

    def describe(self, index: int) -> str:
        """What holds element ``index``'s gap, as a refusal words it."""
        return (
            f"{100.0 * self.fraction:g}% of the {self.room[index]:.3e} "
            f"between mesh neighbors at {self.kind} {index}"
        )


def resolve_gap_limit(
    fraction: float, vert: np.ndarray, elements: np.ndarray, offset: float
) -> GapLimit:
    """An object's limit as one bound per element.

    An element no wider than twice the offset has no room: its neighbors are
    inside their offsets whatever the gap is, so no gap keeps them apart and
    the limit has nothing to scale. Its bound is infinite, which leaves the
    authored gap, and the pairs themselves are the overlap report's to refuse
    or an intersection allowance's to exempt.

    Args:
        fraction: The object's ``contact-gap-limit``, inside
            :data:`LIMIT_RANGE`.
        vert: The object's vertices as they enter the solve.
        elements: Its triangle or rod segment table.
        offset: Its largest contact offset over the run.
    """
    elements = np.asarray(elements)
    room = element_clearance(vert, elements) - 2.0 * float(offset)
    bound = np.where(room > 0.0, float(fraction) * room, np.inf)
    return GapLimit(
        fraction=float(fraction),
        offset=float(offset),
        kind=_KIND[elements.shape[1]],
        room=room,
        bound=bound,
    )
