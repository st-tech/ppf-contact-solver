# File: intersection_correction.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Push apart closed objects that start out interpenetrating, by a penalty
# relaxation in which every object moves as a rigid body.
#
# The solver refuses a scene that starts with two surfaces through each
# other, and a chain converted from Geometry Nodes can arrive with hundreds of
# shallow overlaps between its links. Each object here is VOLUMETRIC: its
# surface is closed, so a point of another object is either inside it or not,
# and "how far inside, and which way out" has an answer. That answer is the
# penalty: a point of A found inside B is pushed out of it, and B is pushed
# back the other way.
#
# OFFSET AND CLEARANCE. The solver's start check refuses two surfaces closer
# than the sum of their contact offsets as well as two that cross, and two
# left exactly at that distance are no better a start. So each pair is opened
# to its offsets plus a clearance, and counts as touching when it is closer
# than its offsets plus half the clearance. Both come from one device: every
# object is swollen along its vertex normals by its offset and a quarter of
# its clearance, and the passes below resolve the crossings of the SWOLLEN
# surfaces, opening them by the remaining half of the clearance.
#
# ONE PASS.
#   1. Triangle pairs that cross, between objects whose boxes overlap
#      (`BVHTree.overlap`, which runs in C).
#   2. Where each EDGE of one object crosses a triangle of the other, and
#      whether it is going in or coming out there. Sorted along the edge, the
#      crossings bound the stretches of that edge inside the other object.
#   3. One sample per stretch: the vertex when the stretch reaches an end of
#      the edge, the midpoint otherwise. The crossings also trace the curves
#      the two surfaces meet along. The curves that bound one connected
#      PIECE of the overlap give that piece ONE way out: the direction that
#      shrinks its volume fastest. Each sample is pushed the way of its own
#      piece as far as a ray takes to leave the other object
#      (`BVHTree.ray_cast`), plus half the clearance.
#   4. The smallest rigid motions, of all objects at once, that open every
#      contact by what it asks: a least-squares solve in which a rotation is
#      costed as the displacement it would give the object's own vertices. A
#      chain whose every link overlaps the next closes up in one solve.
#   5. The motions are scaled down to what one pass may do: a bounded
#      rotation (`_MAX_TURN`), a bounded closing of any two objects
#      (`_MAX_STEP`), and no two of the objects' own surfaces taken from apart
#      to crossing (`_admitted`).
# Passes repeat until no two triangles cross, a pass moves nothing, or the
# budget is spent.
#
# ONE WAY OUT PER PIECE, NOT PER PAIR. A chain link held tight in the next
# one's hole meets it on both sides of the hole: two pieces of overlap, whose
# ways out point opposite ways. Summed into one, they cancel to a direction
# that is no way out of either, and the samples asked to travel along it are
# pushed through the other link's wire, which unlinks the chain. Apart, they
# center the link in the hole. Measured on the chain of public issue #157
# (eleven 6 mm links): summed, one or two of the ten pairs came unlinked at a
# tenth to a third of its contact gap as clearance; apart, none.
#
# THE CLEARANCE IS FITTED. A pair asks for its clearance on every side, and
# a link threaded through a hole has a room of its own the clearance may not
# fit in: that chain's 1 mm contact gap, with a 1.1 mm wire, is one. Then no
# motion opens every contact, and a run that keeps trying moves the links
# around the chain: there, nine of the ten pairs came unlinked at the whole
# gap with a way out per piece. So a run that leaves a pair with a movable
# object is repeated from the start at half the clearance, down to what the
# tree resolves, and the report says which fraction of the setting was used
# (`resolve`). Only a pair still left then is reported as left.
#
# WHAT IT DOES NOT SEE. An object wholly inside another crosses no surface
# and is not found. An object that passes clean through another has two
# crossing curves bounding one piece, whose ways out cancel; the piece then
# goes the way most of its samples' nearest surface points lie
# (`_consensus`), and when even that cancels, as for a rod through the very
# center of a block, each sample goes to its own nearest surface point,
# which is right for a shallow overlap only.
#
# Everything is in world space and in Blender units. Only `mathutils` and
# numpy are used: Blender's own Python carries no numba or scipy, and the
# queries that dominate (BVH construction, tree overlap, nearest point) are
# Blender's C code.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_EPS = float(np.finfo(np.float64).eps)
# Blender's tree holds positions in single precision, so at a distance `m`
# from the origin it places a point to within `_TREE_EPS * m`: its resolution
# there.
_TREE_EPS = float(np.finfo(np.float32).eps)
# The smallest clearance the passes can deliver, in units of that resolution.
# A contact is solved to within a quarter of its clearance (`slack`), and the
# distance it is asked to open comes from a ray cast through the tree, which
# can be off by a unit for where the ray starts, a unit for the triangle it
# ends on and a unit for the arithmetic between. A slack of four units stays
# above their sum. Below it a pass can leave a pair crossing by less than the
# tree resolves, and the next pass cannot see enough of it to finish. The
# object transform a result is written to is single precision as well, and
# storing it moves a surface by up to a unit, which half a clearance of this
# size also covers.
_CLEARANCE_FLOOR = 4 * 4

# How far past the plain distance a vertex may travel along its normal so that
# every face around it moves outward by the whole distance. A vertex whose
# faces all lie within acos(1/4), 75.5 degrees, of its normal gets the whole
# distance on each of them: every corner of a box (54.7) and of a regular
# tetrahedron (70.5) does. A sharper spike gets less on its steepest face,
# because unbounded travel would put the swollen tip far outside the object,
# where it reads as touching things the object is nowhere near.
_MITER_LIMIT = 4.0

# The largest rotation, in radians, one pass may give an object. The contacts
# are solved as LINEAR in the rotation vector `w`, which moves a point at a
# lever arm `r` by `w x r`, while the rotation applied moves it by
# `sin|w|` along that and `1 - cos|w|` toward the axis. At 0.1 the second is
# 5% of the first. Contacts that ask for more than an object can give (two
# pieces on opposite sides of a hole asking for more room than it has) are
# answered by a least-squares rotation of any size: on the chain of public
# issue #157, at half its contact gap as clearance, that turned links by up
# to 3.8 mm of travel and one through its neighbor. A larger motion takes
# more passes, each on the contacts where the last one left them.
_MAX_TURN = 0.1

# The damping of the least-squares solve of a pass (`_least_motion`), as a
# fraction of the size of its matrix. The way of a contact is the difference
# of two positions held in single precision, over their distance, so a
# direction the contacts reach only by a component of that size is
# round-off, not a constraint: contacts all along x that ask for different
# distances (the samples of a rod leaving a block sideways) reach y and z by
# 1e-6 of their size, and an undamped solve divided the disagreement by that
# and moved the rod by 50000. Damped at the square root of the single
# precision epsilon, such a direction is left alone, and one the contacts
# reach by a hundredth of their size is solved to within a thousandth.
_DAMPING = float(np.sqrt(np.finfo(np.float32).eps))

# How far two objects whose boxes meet may move toward each other in one
# pass, as a fraction of the thinner one's thickness (`Body.thickness`). A
# pass is a jump, with nothing between where it starts and where it lands,
# so a step longer than a wire carries one link through the other's wire and
# nothing sees it: there is no overlap left to find on the far side. On a
# chain built like the one of public issue #157, an unbounded step walked
# the links out of each other's holes over a few passes, no pass leaving a
# crossing behind.
_MAX_STEP = 0.5


@dataclass
class Body:
    """One closed object, moved as a rigid body.

    Attributes:
        name: The name a report gives it.
        vert: World-space vertex positions at the pose the scene starts in,
            shape (n, 3).
        tri: Surface triangles as vertex indices, shape (t, 3).
        movable: False for an object that is held where it is. It still
            pushes the others.
        offset: The distance this object's surface keeps from anything, as
            the solver's contact offset does. A pair must end at least the
            sum of its two objects' values apart.
        clearance: The distance this object asks to be left from another
            surface on top of the offsets. A pair uses the mean of its two
            objects' values, and counts as touching below half of it. Raised
            to what the tree can resolve where the object is, when it is
            below that (``margin``).
    """

    name: str
    vert: np.ndarray
    tri: np.ndarray
    movable: bool = True
    offset: float = 0.0
    clearance: float = 0.0
    # Derived by `_prepare`.
    centroid: np.ndarray = field(default=None, repr=False)
    spread: float = 0.0
    reach: float = 0.0
    margin: float = 0.0
    outward: float = 1.0
    # The thinnest the object is: the shortest way through it from a vertex,
    # against its normal, to the surface on the other side. Infinite when no
    # such way is found.
    thickness: float = float("inf")
    swell: np.ndarray = field(default=None, repr=False)
    # Every mesh edge once, as (lower vertex, higher vertex).
    edges: np.ndarray = field(default=None, repr=False)
    rotation: np.ndarray = field(default=None, repr=False)
    translation: np.ndarray = field(default=None, repr=False)
    # World positions of the SWOLLEN surface at the current pose.
    current: np.ndarray = field(default=None, repr=False)
    tree: object = field(default=None, repr=False)
    lower: np.ndarray = field(default=None, repr=False)
    upper: np.ndarray = field(default=None, repr=False)
    # The same for the object's own surface, not swollen (`_admitted`).
    true_tree: object = field(default=None, repr=False)
    true_lower: np.ndarray = field(default=None, repr=False)
    true_upper: np.ndarray = field(default=None, repr=False)
    stale: bool = True
    displaced: bool = False

    def matrix(self) -> np.ndarray:
        """The 4x4 world transform that carries the starting pose to the result."""
        out = np.eye(4)
        out[:3, :3] = self.rotation
        out[:3, 3] = (
            self.centroid + self.translation - self.rotation @ self.centroid
        )
        return out


def resolvable_clearance(clearance: float, extent: float) -> float:
    """``clearance``, raised to what single precision resolves at ``extent``.

    ``extent`` is the largest coordinate magnitude of the geometry, in the
    units of ``clearance``. Blender stores vertex positions and object
    transforms in single precision, so a clearance below ``_CLEARANCE_FLOOR``
    units of resolution there is lost to the rounding of what is written.
    """
    return max(clearance, _CLEARANCE_FLOOR * _TREE_EPS * extent)


def signed_volume(vert: np.ndarray, tri: np.ndarray) -> float:
    """Signed volume a closed triangle surface encloses, positive when its
    faces are wound outward."""
    origin = vert.mean(axis=0)
    a = vert[tri[:, 0]] - origin
    e1 = vert[tri[:, 1]] - vert[tri[:, 0]]
    e2 = vert[tri[:, 2]] - vert[tri[:, 0]]
    return float(np.einsum("ij,ij->i", a, np.cross(e1, e2)).sum() / 6.0)


def _prepare(body: Body) -> None:
    body.vert = np.ascontiguousarray(body.vert, dtype=np.float64)
    body.tri = np.ascontiguousarray(body.tri, dtype=np.int64)
    body.centroid = body.vert.mean(axis=0)
    offset = body.vert - body.centroid
    # Mean squared distance of the object's vertices from its centroid: what
    # a unit rotation costs, in the same units as a squared translation.
    body.spread = float(np.einsum("ij,ij->i", offset, offset).mean())
    # The farthest any vertex is from the centroid: what a rotation moves a
    # vertex by, per radian, at most.
    body.reach = float(np.sqrt(np.einsum("ij,ij->i", offset, offset).max()))
    # Which way a face normal points relative to the inside. An inside-out
    # object is as volumetric as any other, so its sign is divided out.
    body.outward = 1.0 if signed_volume(body.vert, body.tri) >= 0.0 else -1.0
    # Each vertex's step onto the swollen surface the passes work on: the
    # offset and a quarter of the clearance, along its outward normal. Two
    # objects swollen so cross when their true surfaces are within their
    # offsets plus half the clearance, which is what "touching" means here,
    # and opening the swollen surfaces by the other half leaves the true ones
    # their offsets and a whole clearance apart.
    corner = [body.vert[body.tri[:, k]] for k in range(3)]
    face = np.cross(corner[1] - corner[0], corner[2] - corner[0])
    # A face of no area has no normal: it adds nothing to a vertex normal and
    # constrains nothing below.
    area = np.linalg.norm(face, axis=1)
    solid = area > 0.0
    unit = face[solid] / area[solid, None]
    # A vertex normal is the mean of its faces' normals, each weighted by the
    # angle the face subtends at the vertex. Weighted so, it does not depend
    # on how a flat face was cut into triangles, which a weighting by area
    # does: at the corner of a box cut into triangles, one face would count
    # twice as much as its neighbor.
    normal = np.zeros_like(body.vert)
    for k in range(3):
        along = corner[(k + 1) % 3][solid] - corner[k][solid]
        back = corner[(k + 2) % 3][solid] - corner[k][solid]
        angle = np.arctan2(area[solid], np.einsum("ij,ij->i", along, back))
        np.add.at(normal, body.tri[solid, k], angle[:, None] * unit)
    length = np.linalg.norm(normal, axis=1)
    normal[length > 0.0] /= length[length > 0.0, None]
    # A step of `s` along a vertex normal moves a face around that vertex
    # outward by `s` times the cosine between the two normals, which at the
    # corner of a box is 0.58 `s`. Dividing by the smallest cosine among the
    # vertex's faces moves every one of them by at least `s`.
    steepest = np.full(len(body.vert), np.inf)
    for k in range(3):
        index = body.tri[solid, k]
        np.minimum.at(
            steepest, index, np.einsum("ij,ij->i", normal[index], unit)
        )
    travel = np.where(
        steepest * _MITER_LIMIT > 1.0, 1.0 / steepest, _MITER_LIMIT
    )
    # A small object far from the origin can ask for a clearance its
    # positions do not resolve: a 6 mm chain link a meter out, with a contact
    # gap of a hundred nanometers, is one.
    body.margin = resolvable_clearance(
        body.clearance, float(np.abs(body.vert).max())
    )
    distance = body.offset + 0.25 * body.margin
    body.swell = (distance * body.outward) * travel[:, None] * normal
    ends = np.concatenate(
        [body.tri[:, [0, 1]], body.tri[:, [1, 2]], body.tri[:, [2, 0]]]
    )
    body.edges = np.unique(np.sort(ends, axis=1), axis=0)
    body.thickness = _thickness(body, normal)
    body.rotation = np.eye(3)
    body.translation = np.zeros(3)
    body.stale = True
    body.displaced = False


def _thickness(body: Body, normal: np.ndarray) -> float:
    """The shortest way through ``body``, along any vertex's inward normal.

    The ray starts a little inside, so it does not stop on the triangles
    around its own vertex: as far in as the tree resolves positions there.
    """
    from mathutils.bvhtree import BVHTree  # pyright: ignore

    tree = BVHTree.FromPolygons(
        body.vert.tolist(), body.tri.tolist(), all_triangles=True
    )
    start = _CLEARANCE_FLOOR * _TREE_EPS * max(float(np.abs(body.vert).max()), 1.0)
    inward = -body.outward * normal
    thinnest = float("inf")
    for v, d in zip(body.vert, inward, strict=True):
        if not np.any(d):
            continue
        location, _n, _index, distance = tree.ray_cast(
            (v + start * d).tolist(), d.tolist()
        )
        if location is not None:
            thinnest = min(thinnest, distance + start)
    return thinnest


def _refresh(body: Body) -> None:
    """Rebuild a moved object's world positions, box and tree."""
    from mathutils.bvhtree import BVHTree  # pyright: ignore

    body.current = (
        (body.vert + body.swell - body.centroid) @ body.rotation.T
        + body.centroid
        + body.translation
    )
    body.lower = body.current.min(axis=0)
    body.upper = body.current.max(axis=0)
    body.tree = BVHTree.FromPolygons(
        body.current.tolist(), body.tri.tolist(), all_triangles=True
    )
    body.true_tree, body.true_lower, body.true_upper = _true_surface(
        body, body.rotation, body.translation
    )
    body.stale = False


def _true_surface(body: Body, rotation: np.ndarray, translation: np.ndarray):
    """The tree and box of ``body``'s own surface, not swollen, at a pose."""
    from mathutils.bvhtree import BVHTree  # pyright: ignore

    vert = (body.vert - body.centroid) @ rotation.T + body.centroid + translation
    tree = BVHTree.FromPolygons(
        vert.tolist(), body.tri.tolist(), all_triangles=True
    )
    return tree, vert.min(axis=0), vert.max(axis=0)


def _box_pairs(bodies: list[Body]) -> list[tuple[int, int]]:
    """Index pairs whose boxes overlap."""
    return _boxes_meeting(
        np.array([b.lower for b in bodies]), np.array([b.upper for b in bodies])
    )


def _boxes_meeting(lower: np.ndarray, upper: np.ndarray) -> list[tuple[int, int]]:
    """Index pairs of the boxes ``lower[k]``..``upper[k]`` that overlap.

    A sweep along x: sorted by lower bound, a box can only overlap the ones
    that start before it ends.
    """
    order = np.argsort(lower[:, 0], kind="stable")
    starts = lower[order, 0]
    pairs = []
    for rank, i in enumerate(order):
        stop = int(np.searchsorted(starts, upper[i, 0], side="right"))
        others = order[rank + 1 : stop]
        if len(others) == 0:
            continue
        hit = (
            (lower[others, 1] <= upper[i, 1])
            & (upper[others, 1] >= lower[i, 1])
            & (lower[others, 2] <= upper[i, 2])
            & (upper[others, 2] >= lower[i, 2])
        )
        pairs.extend((int(i), int(j)) for j in others[hit])
    return pairs


def _cross(a, b):
    """Row-wise cross product of two (n, 3) arrays.

    Written out because the crossing test below relies on ``a x b`` being the
    exact negation of ``b x a``, component by component, and because
    ``numpy.cross`` spends most of its time on axis bookkeeping for arrays
    this small.
    """
    out = np.empty_like(a)
    out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return out


def _triple(a, b, c):
    return np.einsum("ij,ij->i", a, _cross(b, c))


def _crossings(seg0, seg1, tri0, tri1, tri2):
    """Where segments cross triangles, one row per (segment, triangle).

    Returns ``(hit, t, side)``: whether row k crosses, the parameter of the
    crossing along its segment, and the sign of the segment's direction
    against the triangle's normal (negative when it goes in through a face
    wound outward).

    The test is WATERTIGHT, which a barycentric test is not. A segment is
    inside a triangle's prism when its three signed volumes against the
    triangle's edges agree in sign, and the volume against an edge is
    computed from that edge's two ends alone: the triangle on the other side
    of the edge forms the same number with the ends exchanged, which is its
    exact negation. So a segment passing within round-off of an edge two
    triangles share is given to one of them, or to both when the volume is
    exactly zero, and never to neither. Given to neither, two tubes crossing
    edge on edge read as not crossing at all.
    """
    direction = seg1 - seg0
    a, b, c = tri0 - seg0, tri1 - seg0, tri2 - seg0
    s_ab = _triple(direction, a, b)
    s_bc = _triple(direction, b, c)
    s_ca = _triple(direction, c, a)
    within = (
        ((s_ab >= 0.0) & (s_bc >= 0.0) & (s_ca >= 0.0))
        | ((s_ab <= 0.0) & (s_bc <= 0.0) & (s_ca <= 0.0))
    )
    normal = _cross(tri1 - tri0, tri2 - tri0)
    before = np.einsum("ij,ij->i", seg0 - tri0, normal)
    after = np.einsum("ij,ij->i", seg1 - tri0, normal)
    side = after - before
    # Strictly through the plane, or ending on it from one side. A segment
    # lying in the plane has `side == 0` and crosses nothing.
    through = ((before < 0.0) & (after >= 0.0)) | ((before >= 0.0) & (after < 0.0))
    hit = within & through & (side != 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(hit, -before / side, 0.0)
    return hit, t, side


def _inside_stretches(t: np.ndarray, entering: np.ndarray) -> list[tuple[float, float]]:
    """The stretches of one edge that lie inside the other object.

    ``t`` is sorted. A crossing through an edge or a vertex of the other mesh
    is reported once per triangle that shares it, so a run of crossings of the
    same kind is one crossing: the first of a run going in, the last of a run
    coming out.
    """
    stretches = []
    start = 0.0 if not entering[0] else None
    k = 0
    n = len(t)
    while k < n:
        run_end = k
        while run_end + 1 < n and entering[run_end + 1] == entering[k]:
            run_end += 1
        if entering[k]:
            start = float(t[k])
        elif start is not None:
            stretches.append((start, float(t[run_end])))
            start = None
        k = run_end + 1
    if start is not None:
        stretches.append((start, 1.0))
    return stretches


def _labels(count: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Component label of each of ``count`` nodes joined by the links
    ``a[k]``-``b[k]``: the smallest node of its component."""
    label = np.arange(count)
    if not len(a):
        return label
    while True:
        low = np.minimum(label[a], label[b])
        new = label.copy()
        np.minimum.at(new, a, low)
        np.minimum.at(new, b, low)
        new = new[new]
        if np.array_equal(new, label):
            return label
        label = new


def _inside_parts(body: Body, crossed: np.ndarray, seeds: list[int]) -> np.ndarray:
    """Which piece of ``body``'s surface inside the other object each vertex
    is on, -1 for a vertex outside.

    ``crossed`` are the edges, as (lower, higher) vertex, that cross the other
    surface, and ``seeds`` vertices known to be inside. Cut along the crossed
    edges, the mesh falls into parts each wholly inside or wholly outside,
    and a part is inside when it holds a seed.
    """
    n = len(body.vert)
    edges = body.edges
    keep = np.ones(len(edges), dtype=bool)
    if len(crossed):
        code = edges[:, 0] * n + edges[:, 1]
        keep = ~np.isin(code, crossed[:, 0] * n + crossed[:, 1])
    label = _labels(n, edges[keep, 0], edges[keep, 1])
    part = np.full(n, -1, dtype=np.int64)
    if seeds:
        inside = np.isin(label, label[np.asarray(seeds, dtype=np.int64)])
        part[inside] = label[inside]
    return part


def _pair_contacts(bodies, i, j, pairs):
    """What one crossing pair of objects needs to be pushed apart.

    Returns a list of ``(area, samples)``, one per connected piece of the
    overlap. ``samples`` are the points of each object inside the other, as
    ``(owner, other, point)``. ``area`` is the vector area of the part of
    object ``i``'s surface that bounds that piece, read off the curves the
    two surfaces cross along: moving ``i`` by ``d`` changes the piece's
    volume by ``d . area``, so ``-area`` is the direction that takes ``i``
    out of it fastest, however deep it is. None when it has no single way out
    (`_vector_area`).

    Two crossing curves bound one piece when the surface between them, of
    either object, lies inside the other: an object passing clean through
    another, whose two curves cancel. They bound two pieces when they are
    apart on both: a link touching both sides of the hole it is threaded
    through. Summed, those two would cancel as well, and the way out of
    neither is what is left (see the header).
    """
    pair = np.asarray(pairs, dtype=np.int64)
    count = len(pair)
    samples = []
    # Per crossing found: the triangle pair it belongs to, the point, and a
    # key naming the edge and the triangle it crosses, which the triangle
    # pair on the other side of that edge shares.
    hit_pair, hit_point, hit_key = [], [], []
    parts = []
    for owner, other, own_col, other_col in ((i, j, 0, 1), (j, i, 1, 0)):
        a, b = bodies[owner], bodies[other]
        own_tri = a.tri[pair[:, own_col]]
        face = pair[:, other_col]
        b_tri = b.tri[face]
        # Every edge of the owner's triangles against the other's triangle,
        # each edge directed from its lower vertex index to its higher one so
        # the two triangles sharing it agree on the parameter.
        v0 = np.concatenate([own_tri[:, 0], own_tri[:, 1], own_tri[:, 2]])
        v1 = np.concatenate([own_tri[:, 1], own_tri[:, 2], own_tri[:, 0]])
        lo, hi = np.minimum(v0, v1), np.maximum(v0, v1)
        face3 = np.tile(face, 3)
        b3 = np.tile(b_tri, (3, 1))
        hit, t, side = _crossings(
            a.current[lo], a.current[hi],
            b.current[b3[:, 0]], b.current[b3[:, 1]], b.current[b3[:, 2]],
        )
        if not hit.any():
            parts.append(np.full(len(a.vert), -1, dtype=np.int64))
            continue
        lo, hi, face3, t, side = lo[hit], hi[hit], face3[hit], t[hit], side[hit]
        where = a.current[lo] + t[:, None] * (a.current[hi] - a.current[lo])
        hit_pair.append(np.tile(np.arange(count), 3)[hit])
        hit_point.append(where)
        hit_key.append(
            np.stack([np.full(len(lo), own_col), lo, hi, face3], axis=1)
        )
        # The same edge meets the same triangle once, however many of the
        # owner's triangles brought it here.
        _, first = np.unique(
            np.stack([lo, hi, face3], axis=1), axis=0, return_index=True
        )
        lo, hi, t, side = lo[first], hi[first], t[first], side[first]
        entering = side * b.outward < 0.0
        order = np.lexsort((t, hi, lo))
        lo, hi, t, entering = lo[order], hi[order], t[order], entering[order]
        change = np.flatnonzero((lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])) + 1
        bounds = np.concatenate([[0], change, [len(lo)]])
        inside_vertices = set()
        for k in range(len(bounds) - 1):
            lo_k, hi_k = int(lo[bounds[k]]), int(hi[bounds[k]])
            p0, p1 = a.current[lo_k], a.current[hi_k]
            for t0, t1 in _inside_stretches(
                t[bounds[k] : bounds[k + 1]], entering[bounds[k] : bounds[k + 1]]
            ):
                if t0 == 0.0:
                    inside_vertices.add(lo_k)
                elif t1 == 1.0:
                    inside_vertices.add(hi_k)
                else:
                    samples.append(
                        (owner, other, p0 + 0.5 * (t0 + t1) * (p1 - p0), -1)
                    )
        samples.extend(
            (owner, other, a.current[v], v) for v in sorted(inside_vertices)
        )
        crossed = np.stack([lo[bounds[:-1]], hi[bounds[:-1]]], axis=1)
        parts.append(_inside_parts(a, crossed, sorted(inside_vertices)))
    return _pieces(bodies, i, j, pair, hit_pair, hit_point, hit_key, parts,
                   samples)


def _pieces(bodies, i, j, pair, hit_pair, hit_point, hit_key, parts, samples):
    """The samples of a pair of objects grouped by the piece of overlap they
    are in, each group with its way out (see `_pair_contacts`)."""
    a, b = bodies[i], bodies[j]
    plain = [(owner, other, point) for owner, other, point, _v in samples]
    if not hit_pair or not samples:
        return [(None, plain)] if samples else []
    index = np.concatenate(hit_pair)
    point = np.concatenate(hit_point)
    key = np.concatenate(hit_key)
    order = np.argsort(index, kind="stable")
    index, point, key = index[order], point[order], key[order]
    # Triangle pairs with exactly two crossings: the ends of their segment.
    first = np.flatnonzero(np.concatenate([[True], index[1:] != index[:-1]]))
    size = np.diff(np.concatenate([first, [len(index)]]))
    first = first[size == 2]
    if len(first) == 0:
        return [(None, plain)]
    # Nodes: the segment ends (by key), then the inside parts of each object.
    _, end = np.unique(key, axis=0, return_inverse=True)
    end = end.reshape(-1)
    keys = int(end.max()) + 1
    part_ids = []
    nodes = keys
    for side in parts:
        ids = {}
        for value in np.unique(side[side >= 0]).tolist():
            ids[value] = nodes
            nodes += 1
        part_ids.append(ids)
    link_a = [end[first]]
    link_b = [end[first + 1]]
    # A segment lies on a triangle of each object; it bounds the inside part
    # every inside vertex of those triangles is on.
    segment_pair = index[first]
    for column, side in enumerate(parts):
        corner = (a if column == 0 else b).tri[pair[segment_pair, column]]
        for k in range(3):
            label = side[corner[:, k]]
            on = label >= 0
            if on.any():
                ids = part_ids[column]
                link_a.append(end[first][on])
                link_b.append(np.array([ids[int(v)] for v in label[on]]))
    # Two segments on one triangle bound one inside part between them when
    # each faces the other with its inside: the side of a rod passing through
    # a block, a long triangle with no vertex inside, is crossed so at both
    # faces of the block.
    middle = 0.5 * (point[first] + point[first + 1])
    unit = []
    for column, body in enumerate((a, b)):
        corner = body.current[body.tri[pair[segment_pair, column]]]
        n = body.outward * _cross(
            corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0]
        )
        length = np.linalg.norm(n, axis=1)
        unit.append(n / np.where(length > 0.0, length, 1.0)[:, None])
    for column in (0, 1):
        own, theirs = unit[column], unit[1 - column]
        # Toward the inside of the other object, in this triangle's plane.
        inward = -(theirs - np.einsum("ij,ij->i", theirs, own)[:, None] * own)
        triangle = pair[segment_pair, column]
        order = np.argsort(triangle, kind="stable")
        sorted_triangle = triangle[order]
        bounds = np.flatnonzero(
            np.concatenate(
                [[True], sorted_triangle[1:] != sorted_triangle[:-1], [True]]
            )
        )
        for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
            rows = order[start:stop]
            for x in range(len(rows)):
                for y in range(x + 1, len(rows)):
                    s, t = rows[x], rows[y]
                    apart = middle[t] - middle[s]
                    if inward[s] @ apart > 0.0 and inward[t] @ apart < 0.0:
                        link_a.append(end[first][[s]])
                        link_b.append(end[first][[t]])
    label = _labels(nodes, np.concatenate(link_a), np.concatenate(link_b))
    segment_piece = label[end[first]]
    pieces = np.unique(segment_piece)
    # A vertex sample is in the piece of its inside part; a sample inside an
    # edge whose ends are both outside, in the piece of the nearest segment.
    points = np.concatenate([point[first], point[first + 1]])
    point_piece = np.concatenate([segment_piece, segment_piece])
    groups = {int(p): [] for p in pieces}
    for owner, other, sample, vertex in samples:
        column = 0 if owner == i else 1
        piece = None
        if vertex >= 0:
            found = int(label[part_ids[column][int(parts[column][vertex])]])
            if found in groups:
                piece = found
        if piece is None:
            offset = points - sample
            nearest = int(np.argmin(np.einsum("ij,ij->i", offset, offset)))
            piece = int(point_piece[nearest])
        groups[piece].append((owner, other, sample))
    out = []
    for piece, members in groups.items():
        if not members:
            continue
        mine = first[segment_piece == piece]
        out.append((_vector_area(a, b, pair, index, point, mine), members))
    return out


def _vector_area(a, b, pair, index, point, first):
    """Vector area of the part of ``a``'s surface inside ``b`` that the
    segments starting at rows ``first`` of ``index`` and ``point`` bound, or
    None.

    Two crossing triangles meet along a segment whose ends are two of the
    edge crossings found above. Directed along ``n_a x n_b`` (outward
    normals), the segments chain into closed curves, and half the sum of
    ``p x q`` over them is the vector area of any surface they bound, which
    the part of ``a`` inside ``b`` is. None when the sum is not resolved
    above its own round-off, as when one object passes clean through the
    other and its two curves cancel.
    """
    k = index[first]
    p, q = point[first], point[first + 1]
    ta, tb = a.tri[pair[k, 0]], b.tri[pair[k, 1]]
    normal_a = a.outward * _cross(
        a.current[ta[:, 1]] - a.current[ta[:, 0]],
        a.current[ta[:, 2]] - a.current[ta[:, 0]],
    )
    normal_b = b.outward * _cross(
        b.current[tb[:, 1]] - b.current[tb[:, 0]],
        b.current[tb[:, 2]] - b.current[tb[:, 0]],
    )
    along = _cross(normal_a, normal_b)
    flip = np.einsum("ij,ij->i", q - p, along) < 0.0
    p, q = np.where(flip[:, None], q, p), np.where(flip[:, None], p, q)
    origin = p.mean(axis=0)
    term = 0.5 * _cross(p - origin, q - origin)
    area = term.sum(axis=0)
    gross = float(np.linalg.norm(term, axis=1).sum())
    if not float(np.linalg.norm(area)) > len(term) * _EPS * gross:
        return None
    return area


def _least_motion(bodies, first, second, way, point_first, point_second, gap,
                  slack):
    """The smallest rigid motions that open every contact by its gap.

    Contact k asks that its point on object ``first[k]`` move ``gap[k]``
    farther along ``way[k]`` than its point on object ``second[k]`` does.
    Each object contributes a translation ``d`` and a rotation vector ``w``
    about its centroid, a held object contributes nothing, and the contacts
    are linear in those: ``way . (d + w x r)`` at a lever arm ``r``. All of
    them are solved at once, in the least-squares sense when they disagree,
    and among the motions that satisfy them the one that moves the objects
    least is taken, with a rotation costed as the displacement it gives the
    object's own vertices (``sqrt(spread)`` per radian). A chain whose every
    link overlaps the next therefore closes up in one solve, where pushing
    pairs apart one at a time would need a pass per link.

    Conjugate gradients on the normal equations, started from zero, which is
    what converges to the least motion. It stops once every contact is within
    ``slack`` of what it asked, or when the normal equations are solved to
    round-off, which is where contacts that disagree end up: an object
    squeezed between two held ones that ask it to go opposite ways is there
    before the first step, and gets no motion at all. Returns ``(d, w)``,
    each (count, 3).
    """
    count = len(bodies)
    center = np.array([b.centroid + b.translation for b in bodies])
    radius = np.sqrt(np.array([b.spread for b in bodies]))
    # An object with no extent has no lever arm, so any unit serves it.
    radius[radius == 0.0] = 1.0
    free = np.array([b.movable for b in bodies], dtype=np.float64)

    def coefficients(index, point, sign):
        lever = _cross(point - center[index], way) / radius[index, None]
        return sign * free[index, None] * np.concatenate([way, lever], axis=1)

    c_first = coefficients(first, point_first, 1.0)
    c_second = coefficients(second, point_second, -1.0)
    # The transpose product sums each object's rows. Sorted by object once,
    # that is a segmented sum.
    owner = np.concatenate([first, second])
    order = np.argsort(owner, kind="stable")
    owner = owner[order]
    start = np.flatnonzero(np.concatenate([[True], owner[1:] != owner[:-1]]))
    present = owner[start]
    stacked = np.concatenate([c_first, c_second])[order]
    row = np.concatenate([np.arange(len(first))] * 2)[order]

    def forward(x):
        return (
            np.einsum("ij,ij->i", c_first, x[first])
            + np.einsum("ij,ij->i", c_second, x[second])
        )

    def transpose(y):
        out = np.zeros((count, 6))
        out[present] = np.add.reduceat(stacked * y[row, None], start, axis=0)
        return out

    magnitude = np.abs(stacked)

    def round_off(y):
        """Squared norm `transpose(y)` can have from rounding alone: each of
        its entries is a sum of products, known to within a unit round-off of
        the sum of their magnitudes per term."""
        gross = np.add.reduceat(magnitude * np.abs(y)[row, None], start, axis=0)
        return float(np.vdot(gross, gross)) * (len(y) * _EPS) ** 2

    # A motion the contacts constrain only to round-off is not taken: see
    # `_DAMPING`. The damped normal equations are (A^T A + mu^2) x = A^T b.
    mu2 = (_DAMPING * float(np.sqrt(np.vdot(stacked, stacked)))) ** 2
    x = np.zeros((count, 6))
    residual = gap.astype(np.float64).copy()
    s = transpose(residual)
    direction = s.copy()
    gamma = float(np.vdot(s, s))
    floor = gamma * _EPS
    for _ in range(20 * count):
        if not gamma > max(floor, round_off(residual)):
            break
        if not np.abs(residual).max() > slack:
            break
        q = forward(direction)
        qq = float(np.dot(q, q)) + mu2 * float(np.vdot(direction, direction))
        if not qq > 0.0:
            break
        alpha = gamma / qq
        x += alpha * direction
        residual -= alpha * q
        s = transpose(residual) - mu2 * x
        gamma_next = float(np.vdot(s, s))
        direction = s + (gamma_next / gamma) * direction
        gamma = gamma_next
    return x[:, :3], x[:, 3:] / radius[:, None]


def _consensus(bodies: list[Body], i: int, samples) -> np.ndarray | None:
    """One way out, for object ``i``, of a piece of overlap whose curves
    cancel (`_vector_area`), or None.

    Each sample's way is to the nearest surface point of the other object,
    and their sum, resolved above its own round-off, is the piece's: a rod
    through a block off its center has most of its samples nearest the
    near face. Taken one sample at a time instead, the ways change as the
    rod moves, and a rod whose exit takes several passes is carried along
    the face by the samples near the block's other faces.
    """
    total = np.zeros(3)
    gross = 0.0
    for owner, other, sample in samples:
        location, _n, _index, distance = bodies[other].tree.find_nearest(
            sample.tolist()
        )
        if location is None or not distance > 0.0:
            continue
        heading = (np.array(location) - sample) / distance
        total += heading if owner == i else -heading
        gross += 1.0
    length = float(np.linalg.norm(total))
    if not length > gross * _EPS * 16.0 or not length > 0.0:
        return None
    return total / length


def _resolved(body: Body, shift: np.ndarray, turn: np.ndarray) -> bool:
    """Whether a motion is above the resolution the tree holds ``body`` at."""
    travel = float(np.linalg.norm(shift) + np.linalg.norm(turn) * body.reach)
    return travel > _TREE_EPS * float(np.abs(body.current).max())


def _admitted(bodies: list[Body], shift: np.ndarray, turn: np.ndarray):
    """The motions of a pass, halved until no two surfaces cross that did not.

    A pass opens the pairs that cross, and may make two others cross: a link
    asked by its two neighbors for more room than its holes have is pushed
    into a wire. Pushed on over later passes, it comes out on the wire's far
    side, where every check is satisfied and the chain is unlinked. So a pass
    never takes two of the objects' own surfaces (not swollen) from apart to
    crossing. Each pass closes any two objects by less than the thinner one's
    thickness (`_MAX_STEP`), so a surface cannot cross another and leave it
    within one. When no motion above the tree's resolution is admitted, the
    pass moves nothing, the run ends with its pairs, and a lower clearance is
    tried (`resolve`).
    """
    while any(
        _resolved(b, shift[k], turn[k]) for k, b in enumerate(bodies)
    ):
        poses = []
        for k, b in enumerate(bodies):
            if _resolved(b, shift[k], turn[k]):
                rotation = _rotation(turn[k]) @ b.rotation
                poses.append(_true_surface(b, rotation, b.translation + shift[k]))
            else:
                poses.append((b.true_tree, b.true_lower, b.true_upper))
        lower = np.array([p[1] for p in poses])
        upper = np.array([p[2] for p in poses])
        crossed = False
        for i, j in _boxes_meeting(lower, upper):
            if poses[i][0] is bodies[i].true_tree and poses[j][0] is bodies[j].true_tree:
                continue
            if not poses[i][0].overlap(poses[j][0]):
                continue
            before = bodies[i].true_tree.overlap(bodies[j].true_tree)
            if not before:
                crossed = True
                break
        if not crossed:
            return shift, turn
        shift, turn = 0.5 * shift, 0.5 * turn
    return np.zeros_like(shift), np.zeros_like(turn)


def _rotation(w: np.ndarray) -> np.ndarray:
    """Rotation matrix of the rotation vector ``w`` (Rodrigues)."""
    angle = float(np.linalg.norm(w))
    if angle == 0.0:
        return np.eye(3)
    k = w / angle
    cross = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def resolve(bodies: list[Body], *, max_iterations: int, progress=None) -> dict:
    """Move ``bodies`` rigidly until no two of their surfaces cross.

    The clearance is fitted (see the header): when a run leaves a pair with
    a movable object, the run is repeated from the start with every
    object's clearance halved, until one leaves none or the clearances are
    at what the tree resolves, where halving changes nothing. Each run gets
    the whole budget, so a budget too small for a scene lowers its
    clearance too. The run that cleared the scene is kept; when none did, the
    one that left the fewest pairs, the larger clearance on a tie, so a second
    click goes on from the run that got furthest.

    Args:
        bodies: The closed objects. Each is updated in place; read the result
            from :meth:`Body.matrix`.
        max_iterations: The number of passes each run may spend.
        progress: Called with ``(pass index, crossing object pairs)`` after
            each pass, when given.

    Returns:
        dict: ``iterations`` (passes run), ``initial_pairs`` and
        ``remaining_pairs`` (name pairs whose surfaces crossed before the
        first pass and after the last, two held objects included),
        ``moved`` (``{name: distance}``, the largest displacement of any
        vertex of each object that moved), and ``clearance_scale``, the
        fraction of each object's clearance the run used: 1.0 as asked.
    """
    asked = [body.clearance for body in bodies]
    movable = {body.name for body in bodies if body.movable}

    def run(scale):
        for body, clearance in zip(bodies, asked, strict=True):
            body.clearance = clearance * scale
        outcome = _relax(
            bodies, max_iterations=max_iterations, progress=progress
        )
        outcome["clearance_scale"] = scale
        left = sum(
            1 for a, b in outcome["remaining_pairs"] if a in movable or b in movable
        )
        # `_prepare` raised every margin to what the tree resolves; below
        # that a halving asks for nothing new.
        floor = all(body.margin > body.clearance for body in bodies)
        return outcome, left, floor

    def pose():
        return [
            (b.rotation.copy(), b.translation.copy(), b.displaced) for b in bodies
        ]

    outcome, left, floor = run(1.0)
    best = left, outcome, pose()
    scale = 1.0
    while left and not floor:
        scale *= 0.5
        outcome, left, floor = run(scale)
        if left < best[0]:
            best = left, outcome, pose()
    if left:
        # No clearance cleared the scene. What is kept and reported is the
        # run that left the fewest pairs, the larger clearance on a tie, so an
        # object no clearance can free keeps the run at the setting. Keeping
        # the run at the setting in every case would stall a scene whose
        # setting cannot fit the same way on every click: on Windows CI four
        # clicks on a chain a quarter shorter than its taut length each left
        # the same 19 pairs.
        _left, outcome, kept = best
        for body, (rotation, translation, displaced) in zip(bodies, kept, strict=True):
            body.rotation, body.translation = rotation, translation
            body.displaced = displaced
    for body, clearance in zip(bodies, asked, strict=True):
        body.clearance = clearance
    return outcome


def _relax(bodies: list[Body], *, max_iterations: int, progress=None) -> dict:
    """One run of `resolve` at the clearances the bodies carry."""
    for body in bodies:
        _prepare(body)
    crossing: dict[tuple[int, int], tuple] = {}
    checked: set[tuple[int, int]] = set()
    initial = None
    iterations = 0
    while True:
        moved_now = {k for k, b in enumerate(bodies) if b.stale}
        for k in moved_now:
            _refresh(bodies[k])
        candidates = _box_pairs(bodies)
        candidate_set = set(candidates)
        # A pair is tested again only when one of its objects moved since it
        # was last tested; an untouched pair keeps its verdict.
        for key in list(crossing):
            if key not in candidate_set or key[0] in moved_now or key[1] in moved_now:
                del crossing[key]
        checked &= candidate_set
        for key in candidates:
            i, j = key
            if key in checked and i not in moved_now and j not in moved_now:
                continue
            checked.add(key)
            # The tree proposes triangle pairs and the edge test decides: a
            # pair of objects crosses when some edge of one passes through a
            # triangle of the other.
            pairs = bodies[i].tree.overlap(bodies[j].tree)
            if pairs:
                pieces = _pair_contacts(bodies, i, j, pairs)
                if pieces:
                    crossing[key] = pieces
        names = sorted(
            tuple(sorted((bodies[i].name, bodies[j].name))) for i, j in crossing
        )
        if initial is None:
            initial = names
        if progress is not None:
            progress(iterations, len(crossing))
        if not crossing or iterations >= max_iterations:
            break

        first, second, way, at_first, at_second, gap = [], [], [], [], [], []
        slack = np.inf
        for (i, j), pieces in crossing.items():
            # Two held objects that cross are found and reported, since the
            # result must not call such a scene clear, and are left as they
            # are.
            if not (bodies[i].movable or bodies[j].movable):
                continue
            # Half the pair's clearance, on top of what separates the swollen
            # surfaces: see `_prepare`.
            clearance = 0.25 * (bodies[i].margin + bodies[j].margin)
            # A contact opened to within half of that is opened: the swollen
            # surfaces then clear each other.
            slack = min(slack, 0.5 * clearance)
            for area, samples in pieces:
                # The way out of `j` for `i` from this piece. Every sample of
                # the piece is asked to travel as far that way as the deepest
                # of them needs, so one translation satisfies the piece.
                out = (
                    _consensus(bodies, i, samples) if area is None
                    else -area / np.linalg.norm(area)
                )
                rows, reach = [], 0.0
                if out is not None:
                    for owner, other, sample in samples:
                        heading = out if owner == i else -out
                        location, _n, _index, distance = (
                            bodies[other].tree.ray_cast(
                                sample.tolist(), heading.tolist()
                            )
                        )
                        if location is None:
                            continue
                        hit = np.array(location)
                        rows.append(
                            (out, sample, hit) if owner == i
                            else (out, hit, sample)
                        )
                        reach = max(reach, distance)
                if rows:
                    for direction, on_i, on_j in rows:
                        first.append(i)
                        second.append(j)
                        way.append(direction)
                        at_first.append(on_i)
                        at_second.append(on_j)
                        gap.append(reach + clearance)
                    continue
                # No single way out: each sample goes to the nearest surface
                # point of the other object instead.
                for owner, other, sample in samples:
                    location, face_normal, _index, distance = (
                        bodies[other].tree.find_nearest(sample.tolist())
                    )
                    if location is None:
                        continue
                    hit = np.array(location)
                    if distance > 0.0:
                        heading = (hit - sample) / distance
                    else:
                        heading = bodies[other].outward * np.array(face_normal)
                    first.append(i)
                    second.append(j)
                    way.append(heading if owner == i else -heading)
                    at_first.append(sample if owner == i else hit)
                    at_second.append(hit if owner == i else sample)
                    gap.append(distance + clearance)
        if not first:
            break
        shift, turn = _least_motion(
            bodies, np.array(first), np.array(second), np.array(way),
            np.array(at_first), np.array(at_second), np.array(gap), slack,
        )
        # Scaled as a whole, the motions keep the proportions the contacts
        # asked for between objects.
        scale = 1.0
        most = float(np.linalg.norm(turn, axis=1).max())
        if most > _MAX_TURN:
            scale = _MAX_TURN / most
        # Two objects whose boxes meet close by at most the step, the
        # translations' difference and both rotations' largest travel. A
        # chain spreading as a whole moves its end links farthest and each
        # pair of neighbors little, so the bound is on the pair, not on how
        # far an object travels.
        for i, j in candidates:
            closing = (
                float(np.linalg.norm(shift[i] - shift[j]))
                + float(np.linalg.norm(turn[i])) * bodies[i].reach
                + float(np.linalg.norm(turn[j])) * bodies[j].reach
            )
            limit = _MAX_STEP * min(bodies[i].thickness, bodies[j].thickness)
            if closing * scale > limit:
                scale = limit / closing
        shift, turn = _admitted(bodies, shift * scale, turn * scale)
        for k, body in enumerate(bodies):
            # A motion below the resolution the tree holds this object's
            # positions at changes nothing the next pass could see. Contacts
            # that disagree settle into motions of that size, each pass
            # answering the rounding of the one before.
            if not _resolved(body, shift[k], turn[k]):
                continue
            body.rotation = _rotation(turn[k]) @ body.rotation
            body.translation = body.translation + shift[k]
            body.stale = True
            body.displaced = True
        iterations += 1
        # A pass that moved nothing leaves the next one the same contacts,
        # as when an object is squeezed between two held ones that ask it to
        # go opposite ways.
        if not any(body.stale for body in bodies):
            break

    moved = {}
    for body in bodies:
        if not body.displaced:
            continue
        carried = (
            (body.vert - body.centroid) @ body.rotation.T
            + body.centroid
            + body.translation
        )
        moved[body.name] = float(np.linalg.norm(carried - body.vert, axis=1).max())
    return {
        "iterations": iterations,
        "initial_pairs": initial or [],
        "remaining_pairs": names,
        "moved": moved,
    }
