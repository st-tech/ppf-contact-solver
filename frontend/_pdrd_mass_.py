# File: _pdrd_mass_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""The solid a rigid body's surface encloses: its volume and second moment.

A PDRD body is given as a triangle surface, and its mass is its density times
the volume that surface encloses. The divergence theorem turns both integrals
this module returns into sums over the triangles, and that step is valid for a
surface that is closed and wound consistently, and for no other. So the two
conditions are checked here, exactly and on the edges, before any sum is
trusted.

WHICH WAY the surface faces is not a condition. Reversing every triangle
negates both sums and changes nothing about the solid, so the body's
orientation is read off its total signed volume and divided out. What a
consistent winding does carry is the difference between the parts of one body:
a closed part wound against the rest is a cavity, which is how a hollow body is
written, and it is accepted as one only where it lies inside the rest.

Nothing here is measured against an absolute length, area or volume. A body of
any size is accepted as long as its volume is resolved above the round-off of
the sum that produced it.
"""

import numpy as np

_EPS = float(np.finfo(np.float64).eps)


def _weld(corner: np.ndarray) -> np.ndarray:
    """One id per distinct POSITION among the triangle corners, shape (T, 3).

    Whether a surface is closed is a statement about where its edges are, not
    about how its vertices are indexed, so two corners at exactly the same
    position are the same vertex here. A mesh stored with a vertex per face
    corner is closed when its triangles meet, as it should be. `+ 0.0` folds a
    negative zero onto zero, which compare equal and must share an id.
    """
    point = corner.reshape(-1, 3) + 0.0
    _, weld = np.unique(point, axis=0, return_inverse=True)
    return np.asarray(weld).reshape(-1, 3)


def _parts(count: int, edge: np.ndarray, slot_tri: np.ndarray) -> np.ndarray:
    """The edge-connected part each of `count` triangles belongs to.

    Two triangles are in the same part when a chain of shared edges joins
    them. `edge[s]` is the undirected edge of slot `s` and `slot_tri[s]` the
    triangle that slot belongs to. Every triangle starts as its own root; a
    round hooks the larger of two joined roots under the smaller and then
    shortens every chain to its root, which at least halves the roots of an
    unfinished part.
    """
    order = np.argsort(edge, kind="stable")
    sorted_edge, sorted_tri = edge[order], slot_tri[order]
    shared = sorted_edge[1:] == sorted_edge[:-1]
    a, b = sorted_tri[:-1][shared], sorted_tri[1:][shared]
    root = np.arange(count)
    while True:
        root_a, root_b = root[a], root[b]
        apart = root_a != root_b
        if not apart.any():
            break
        a, b, root_a, root_b = a[apart], b[apart], root_a[apart], root_b[apart]
        np.minimum.at(root, np.maximum(root_a, root_b), np.minimum(root_a, root_b))
        while True:
            hop = root[root]
            if np.array_equal(hop, root):
                break
            root = hop
    return np.unique(root, return_inverse=True)[1]


def _turns(point: np.ndarray, corner: np.ndarray) -> float:
    """How many times the triangles `corner` wind around `point`.

    The summed solid angle over 4 pi (van Oosterom and Strackee, 1983). For a
    closed surface it is an integer away from the surface itself: +1 inside a
    part whose signed volume is positive, -1 inside one whose signed volume is
    negative, 0 outside.
    """
    a, b, c = (corner[:, k] - point for k in range(3))
    la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
    numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
    denominator = (
        la * lb * lc
        + np.einsum("ij,ij->i", a, b) * lc
        + np.einsum("ij,ij->i", b, c) * la
        + np.einsum("ij,ij->i", c, a) * lb
    )
    return float(np.arctan2(numerator, denominator).sum() / (2.0 * np.pi))


def enclosed_moments(
    label: str, vert: np.ndarray, tri: np.ndarray, origin: np.ndarray
) -> tuple[float, float]:
    """Volume and second moment of the solid a rigid body's surface encloses.

    Args:
        label: The name a refusal gives the body.
        vert: Vertex positions, shape (V, 3), which `tri` indexes.
        tri: The body's surface triangles, shape (T, 3), with T >= 1.
        origin: The point the second moment is taken about.

    Returns:
        ``(volume, second_moment)``: the enclosed volume, positive, and the
        integral of ``||r - origin||^2`` over it. Both are the same whichever
        way the surface faces.

    Raises:
        ValueError: The surface is open, its faces are not wound consistently,
            it encloses no volume, or one of its closed parts is wound against
            the rest without being a cavity inside them.
    """
    tri = np.asarray(tri, dtype=np.int64)
    corner = np.asarray(vert, dtype=np.float64)[tri]
    weld = _weld(corner)

    # Each triangle contributes its three edges, directed in its winding
    # order. A surface is closed and consistently wound exactly when every edge
    # appears as often in one direction as in the other, which is the statement
    # that its boundary is empty. An edge whose two corners weld together has
    # no length and is no edge.
    tail = weld.ravel()
    head = weld[:, [1, 2, 0]].ravel()
    proper = tail != head
    slot_tri = np.repeat(np.arange(len(tri)), 3)[proper]
    tail, head = tail[proper], head[proper]
    count = int(weld.max()) + 1
    key = np.minimum(tail, head) * count + np.maximum(tail, head)
    _, edge, incidence = np.unique(key, return_inverse=True, return_counts=True)
    net = np.bincount(edge, weights=np.where(tail < head, 1.0, -1.0))
    border = int(np.count_nonzero(incidence % 2))
    if border:
        raise ValueError(
            f"PDRD body {label!r}: its surface is open. {border} "
            f"edge{'s' if border > 1 else ''} border a hole or carry an odd "
            "number of faces. The mass of a rigid body is its density times "
            "the volume its surface encloses, so the surface must be closed."
        )
    opposed = int(np.count_nonzero(net))
    if opposed:
        raise ValueError(
            f"PDRD body {label!r}: its faces are not wound consistently. "
            f"{opposed} edge{'s' if opposed > 1 else ''} join faces wound "
            "opposite ways, so the surface has no inside. Recalculate its "
            "normals so every face is wound the same way; which way they "
            "face does not matter."
        )

    # Signed volume of the tetrahedron from `origin` to each triangle, through
    # the triangle's own edge vectors. The edges are differences of nearby
    # positions and stay accurate however far the triangle is from `origin`,
    # which the product of three absolute positions does not. `reach` bounds
    # each term by the lengths of its three factors, and a sum of n terms is
    # resolved above n * eps times that bound.
    a = corner[:, 0] - origin
    e1 = corner[:, 1] - corner[:, 0]
    e2 = corner[:, 2] - corner[:, 0]
    tet = np.einsum("ij,ij->i", a, np.cross(e1, e2)) / 6.0
    reach = (
        np.linalg.norm(a, axis=1)
        * np.linalg.norm(e1, axis=1)
        * np.linalg.norm(e2, axis=1)
        / 6.0
    )

    # A closed part has a signed volume of its own, whatever point it is
    # measured from, and the body's is their sum. A part whose own volume is
    # not resolved (a flat, two-sided fin) takes no side below.
    part = _parts(len(tri), edge, slot_tri)
    part_volume = np.bincount(part, weights=tet)
    resolved = np.abs(part_volume) > np.bincount(part) * _EPS * np.bincount(
        part, weights=reach
    )
    signed = float(tet.sum())
    if not abs(signed) > len(tri) * _EPS * float(reach.sum()):
        sides = part_volume[resolved]
        if (sides > 0.0).any() and (sides < 0.0).any():
            raise ValueError(
                f"PDRD body {label!r}: its closed parts are wound against each "
                "other and their volumes cancel. Recalculate the normals so "
                "every part is wound the same way."
            )
        raise ValueError(
            f"PDRD body {label!r}: its surface encloses no volume. It is flat "
            "or folded onto itself, and a rigid body needs a surface with an "
            "inside."
        )
    orientation = 1.0 if signed > 0.0 else -1.0

    # A part that agrees with the body is solid. A part that disagrees
    # removes volume, which describes a cavity when it lies inside the rest
    # of the body and describes nothing otherwise: a region of negative mass.
    # One point of the part settles which, since a part that does not cross
    # the rest lies wholly on one side of it, and the rest winds once around
    # that point for a cavity and not at all for a stray part.
    against = np.flatnonzero(resolved & (part_volume * orientation < 0.0))
    if len(against):
        order = np.argsort(part, kind="stable")
        start = np.searchsorted(part[order], np.arange(len(part_volume) + 1))
        lower = np.minimum.reduceat(corner.min(axis=1)[order], start[:-1])
        upper = np.maximum.reduceat(corner.max(axis=1)[order], start[:-1])
        stray = 0
        for k in against:
            point = corner[order[start[k]], 0]
            # A closed part winds around no point outside its bounding box.
            around = resolved & np.all(
                (lower <= point) & (point <= upper), axis=1
            )
            around[k] = False
            rest = np.concatenate(
                [order[start[j]:start[j + 1]] for j in np.flatnonzero(around)]
                or [np.empty(0, dtype=np.int64)]
            )
            if not orientation * _turns(point, corner[rest]) > 0.5:
                stray += 1
        if stray:
            raise ValueError(
                f"PDRD body {label!r}: {stray} of its {int(resolved.sum())} "
                f"closed parts {'are' if stray > 1 else 'is'} wound against the rest "
                "without lying inside them, so "
                f"{'they are' if stray > 1 else 'it is'} neither solid nor a "
                "cavity. Recalculate the normals so every part is wound the "
                "same way."
            )

    # Second moment about `origin` of the same tetrahedra: for one with a
    # corner at the origin and the others at p, q, r it is
    # V / 20 * (|p + q + r|^2 + |p|^2 + |q|^2 + |r|^2).
    b = corner[:, 1] - origin
    c = corner[:, 2] - origin
    total = a + b + c
    spread = (
        np.einsum("ij,ij->i", total, total)
        + np.einsum("ij,ij->i", a, a)
        + np.einsum("ij,ij->i", b, b)
        + np.einsum("ij,ij->i", c, c)
    )
    second_moment = float(np.sum(tet / 20.0 * spread))
    return orientation * signed, orientation * second_moment
