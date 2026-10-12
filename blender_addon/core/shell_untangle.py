# File: shell_untangle.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Remove the intersections of open triangle surfaces (cloth) by deforming them.
# Built on Minarcik, Liu, Crane and Li, "Untangling Surfaces via Shape and Mesh
# Repulsion" (ACM Trans. Graph. 45(4), 163, 2026), and departing from it where
# measurements on layered, crossing, folded and poked sheets asked for it.
#
# A shell has no inside, so the rigid push-apart of `intersection_correction`
# does not apply, and the two places a local push goes wrong have two remedies.
#
#   * A part of a surface that has POKED THROUGH another is the minority: the
#     rest of it lies on the other side. Every triangle pair of the poke is
#     asked to leave that way (`_vote_sides`, `_sat`'s preferred exit), which a
#     strong contact push then does in a few iterations. Without the side the
#     cheapest exit of a pair straddling the rim is sometimes through, sometimes
#     back, and the poke is locked.
#   * Two surfaces that CROSS EACH OTHER have no right side, and the same lock
#     holds them. The Gaussian self-contact energy breaks it by pushing the two
#     apart as wholes; the lock is an unstable equilibrium of it, which grows
#     slowly, so it starts weak and is strengthened whenever the iterate stops
#     moving.
#
# The sum minimized over the movable vertices is
#
#   E = wG E_G + wM E_M + wR E_R
#
#   E_G  Gaussian self-contact energy, between area-weighted cluster centroids
#        (`_Coarse`) so that its cost does not grow with the mesh. Pairs of
#        clusters that are near each other over the surface are left out. Its
#        bandwidth is the one that maximizes it for the geometry
#        (`_update_bandwidth`), the scale at which the surface meets itself.
#   E_M  Contact penalty. For two triangles, the depth by which the origin lies
#        inside their Minkowski difference, found as the smallest overlap over
#        the 17 separating axes a pair of triangles can have, plus a margin.
#   E_R  As-rigid-as-possible energy of every triangle against the input shape.
#
# STAGES (`untangle`).
#   0. Vote the side of each overlapping pair of surfaces.
#   1. Minimize the sum with L-BFGS until no pair is inside its margin. With a
#      side chosen the Gaussian waits for the first stall; without, it is on
#      from the start at a small fraction of the contact force.
#   2. Stage 1 ends the moment the pairs clear, having moved the surface more
#      than it had to. The second stage pulls it back toward its input, staying
#      embedded (`_recover`).
#   The contact push is a ladder: a strong oriented one first where a side was
#   chosen, then the weak push with the Gaussian, then a strong one without a
#   side. The first rung that embeds the surface decides the result, and a run
#   that embeds nothing returns the input.
#
# CLEARANCE. The solver refuses surfaces closer than the sum of their contact
# offsets as well as surfaces that cross. E_M is therefore evaluated with a
# margin: a pair counts as overlapping while it is closer, along every
# separating axis, than offset_a + offset_b + clearance. The run ends when no
# pair is, so the result is embedded AND clear of every offset. Closeness along
# a separating axis is a lower bound on the distance, so this errs toward
# keeping surfaces apart.
#
# HELD GEOMETRY. A collider with many triangles is cut down to the ones near the
# movable surfaces before the run, and the result is checked against the region
# that was kept (`untangle`). The unit of length every step and force is measured
# in is the mean edge over the edges with a movable end, so a dense collider
# next to a coarse garment does not set it.
#
# THE RUN IN A CHILD PROCESS. `Run` is `untangle` in another process, for a
# caller that must stay responsive: see its docstring for why it is not a thread.
#
# DEPARTURES FROM THE PAPER.
#   * The side of each pair of surfaces, voted from the whole surface. The paper
#     has no such notion; it relies on the Gaussian for every lock.
#   * The Gaussian on cluster centroids, and with its near field cut at a
#     fraction of the bandwidth over the surface rather than at one ring: with
#     the paper's 0/1 adjacency weight the bandwidth that maximizes the energy
#     falls with the mesh resolution (0.20, 0.14, 0.10 of the diagonal at 8, 16
#     and 32 divisions of the same sheets).
#   * Triangles that share a vertex are in the penalty too (with no margin),
#     because the solver's crossing check reports two neighbors folded through
#     each other. The paper sums over non-adjacent pairs only.
#   * Six in-plane axes in the separating-axis set, which coplanar triangles
#     need and the paper's set lacks.
#   * The paper's reverse flow is Newton with an IPC barrier and continuous
#     collision detection. Here it is L-BFGS on a smooth penalty that starts a
#     guard band before the margin (narrowed in turn), a cap on each vertex's
#     step below half the thinnest margin, and a rejection of any step that ends
#     inside a margin: a pair cannot cross a margin and come out the far side
#     within one step. It is not a proof for a step that rotates a long
#     triangle.
#   * Regularization is ARAP alone (the paper adds a bending term), and a spring
#     to the input in the second stage.
#
# Everything is numpy. Blender's own Python carries numpy and no scipy or numba,
# so the cost is numpy's: the contact pass over triangle pairs dominates, at
# about 0.04 s per thousand triangles per evaluation, a few hundred evaluations.

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from dataclasses import asdict, dataclass, field, replace

import numpy as np

# Normalization exponent of the kernel: 2 for a surface (a 2D area measure).
_P = 2

# A separating axis built from two edges is dropped when it is shorter than this
# fraction of the product of the two edge lengths: the edges are parallel, and
# the axis is then a rounding artifact rather than a face of the Minkowski
# difference.
_AXIS_TOL = 1e-9

# The largest Gaussian bandwidth the surface-distance table is sized for, in
# units of the scene's diagonal. A bandwidth above it excludes pairs only as far
# as the table reaches.
_HOP_REACH = 0.3

# The most entries the dense cluster basis may have (free vertices times
# clusters).
_BASIS_LIMIT = 40_000_000

# How far from perpendicular to a pair's preferred exit direction an axis may be
# and still be used in both directions (the cosine). A pair whose cheapest way
# out is along the surface, or across it toward the preferred side, is kept;
# one that would leave toward the other side is not.
_SIDE_TOL = 0.3

# Number of separating axes of a triangle pair: two face normals, nine
# edge-edge cross products, and six in-plane edge normals. The last six are
# axes of the Minkowski difference only when the two triangles are coplanar, and
# there the nine cross products all degenerate to the plane normal, so without
# them two coplanar triangles apart in the plane would read as overlapping.
_N_AXES = 17


@dataclass
class Surface:
    """One open or closed triangle surface.

    Attributes:
        name: The name a report gives it.
        vert: World-space vertex positions, shape (n, 3).
        tri: Triangles as vertex indices, shape (t, 3).
        movable: True when every vertex may move, False when none may, or a
            boolean array of shape (n,) for a surface with held vertices (pins).
            A held vertex still repels the others.
        offset: The distance this surface keeps from any other, as the solver's
            contact offset does.
    """

    name: str
    vert: np.ndarray
    tri: np.ndarray
    movable: object = True
    offset: float = 0.0


@dataclass
class Settings:
    """Weights and limits of one run. Lengths are in the scene's own units."""

    # Extra distance asked on top of the offsets, so a result does not sit
    # exactly at the solver's threshold.
    clearance: float = 0.0
    # Strength of the contact force of one overlapping pair, as a multiple of the
    # shape stiffness times a mean triangle's area (in units of the scene's
    # diagonal). A sheet within its margin of another has a pair for every
    # triangle, so the number of pairs grows as the square of the resolution, and
    # a pair's force falls with the area it stands for to keep the push on the
    # sheet the same. A tuple is a ladder: each rung is tried from the input in
    # turn until one embeds the surface. A weak push leaves the Gaussian to do
    # the untangling, which moves the surface least and is the only one that
    # frees sheets that cross; but it is slow to close a dense field of small
    # fold-overs, which a strong push closes at the price of locking crossing
    # sheets (measured on layered, crossing, folded and poked sheets).
    mink_push: tuple = (0.05, 5.0)
    # The pushes tried, in order, when a side was chosen for some pair of
    # surfaces, as a length in units of the scene's diagonal: every pair of a poke
    # asks for the same exit, and the rim wins against the shape energy about
    # 1.5 times this far. The ladder above is the fallback.
    mink_push_oriented: tuple = (0.03,)
    # The Gaussian's force on the most pushed vertex is this fraction of one
    # contact force at first, and the fraction grows by `gauss_growth` at every
    # stall.
    gauss_start: float = 0.3
    gauss_growth: float = 2.0
    # Switch the Gaussian on from the first iteration of an attempt that has no
    # side chosen. When False it waits for the first stall, which leaves a scene
    # the local terms can clear untouched by it; an attempt with a side chosen
    # always waits.
    gauss_first: bool = True
    # Stalls after which the run gives up.
    max_stalls: int = 12
    max_iter: int = 2000
    memory: int = 10
    # The Gaussian is strengthened when the iterate stops moving (a step shorter
    # than `still_step` mesh edges, five times running: it has reached a local
    # minimum of the objective, where a lock between crossing sheets holds), or
    # when `level_iter` iterations at one strength have not embedded the
    # surface. The growth of an unstable mode out of a lock is slow at first, so
    # a level is given time to show it.
    still_step: float = 1e-3
    level_iter: int = 60
    # Give each pair of overlapping surfaces the side the contact between them
    # mostly lies on (`_vote_sides`), unless the vote is closer than this.
    orient_sides: bool = True
    side_confidence: float = 0.25
    # Number of clusters the Gaussian is evaluated between (see `_Coarse`). A
    # mesh with fewer vertices than this is its own clusters.
    clusters: int = 500
    # A mesh with this many free vertices or more takes each stage in two
    # levels: first as one translation per cluster, smoothed over the mesh, which
    # moves a whole poke or layer at once, then on the vertices. A smooth motion
    # otherwise takes as many iterations as it spans mesh rings.
    coarse_min_vertices: int = 1500
    coarse_rounds: int = 4
    coarse_iter: int = 80
    coarse_recover_iter: int = 150
    # The coarse level of the second stage is for a surface that has to come a
    # long way back: it is skipped when no vertex is farther than this many mean
    # edge lengths from its input.
    coarse_recover_edges: float = 1.0
    # Weight of the old squared bandwidth in a damped Jacobi update.
    bandwidth_damping: float = 0.25
    # Re-solve the bandwidth every this many iterations. 0 freezes it at the
    # value found when the Gaussian is switched on.
    bandwidth_every: int = 5
    # A run whose vertices travel farther than this many scene diagonals has
    # diverged, and ends there.
    max_travel: float = 2.0
    # Held geometry is cut down to the triangles near the movable surfaces once
    # there are this many of them: every table and the broad phase are built over
    # all triangles, and most of a large collider lies far from the cloth.
    crop_min_triangles: int = 2000
    # How far from the box of the movable surfaces a held triangle is kept, in
    # units of that box's diagonal. A result that leaves the region, so that a
    # triangle outside it could be within a margin, is run again with the region
    # grown.
    crop_reach: float = 0.15
    # A scene with this many triangles or more is solved in pieces (`_untangle_local`):
    # only the places where a pair is inside its margin are given vertices to
    # move, grown by `local_rings` rings of the mesh, and the places that do not
    # reach one another are solved one at a time, each against the geometry
    # within `local_reach` mesh edges of it. The result is checked against the
    # whole scene, and what is left is solved again with the rings and the reach
    # doubled, up to `local_rounds` times.
    local_min_triangles: int = 30000
    local_rings: int = 3
    local_reach: float = 1.5
    local_rounds: int = 3
    # The second stage of a place gets this many iterations, and the sweep that
    # settles the corners a third of it again: the surface of a large scene is
    # most of the way back long before the last of it is, and each iteration costs
    # the whole place.
    local_recover_iter: int = 150
    # Surfaces that cannot meet are solved as scenes of their own (`_islands`): two
    # movable surfaces are in one island when their boxes overlap, each grown by
    # this fraction of its own diagonal. How far a surface travels while it is
    # untangled is of the order of its size, so a surface farther than that from
    # every other one has no business with them, and a scene that fails in one
    # place leaves the others as they were solved.
    island_reach: float = 0.5
    # Filled in by a local solve as it goes ({"round", "group", "groups"}), for
    # whoever shows the progress.
    status: object = None
    # From a scene no attempt embeds, return the attempt that left the fewest
    # pairs inside their margin instead of the input: most of the work is done in
    # it. A scene solved a place at a time asks for it, and goes on from there.
    keep_best: bool = False
    # Stalls of an attempt with a side chosen that are answered by pushing
    # `push_boost` times harder before the Gaussian is brought in. The Gaussian
    # moves whole surfaces, which is not what the last few pairs of a large scene
    # need.
    push_boosts: int = 0
    push_boost: float = 3.0
    # Second stage: pull the untangled surface back toward the input.
    recover: bool = True
    recover_iter: int = 1000
    # A guard band is left when the objective fell by less than this fraction
    # over the last `recover_window` iterations, and the surface is no longer
    # moving at the speed the step cap allows. The bands that bring the surface
    # back toward its input use `recover_gain`; the sweep after them, which
    # settles the corners and the boundary, uses the smaller `polish_gain`
    # (the corners of a sheet are the last to come back).
    recover_gain: float = 0.02
    polish_gain: float = 0.001
    recover_window: int = 20
    # The same test for the coarse level, which only has to get the surface close.
    coarse_gain: float = 0.02
    # Stiffness of the pull, in units of the shape energy's.
    recover_pull: float = 1.0
    # The contact penalty is this many times the strongest pull, so a pair at
    # its margin is held.
    recover_hold: float = 4.0
    # Widths of the guard band the penalty starts before the margin, in units of
    # the longest step, narrowed in turn.
    recover_bands: tuple = (2.0, 1.0, 0.5, 0.25)
    # Use the smooth penalty in the recovery.
    recover_smooth: bool = True
    verbose: bool = False
    # Called as progress(stage, iteration, pairs) once per iteration, with stage
    # "untangle" or "recover" and `pairs` the number of pairs inside their margin.
    # Returning False cancels the run: from the first stage it returns the input
    # unchanged, from the second the embedded state reached so far.
    progress: object = None


class RunCanceledError(Exception):
    """Raised inside a run when `Settings.progress` returns False."""


class _NotEmbeddedError(Exception):
    """Every rung of the contact ladder ended without embedding the surface."""


@dataclass
class Report:
    iterations: int = 0
    converged: bool = False
    # Pairs still closer than their margin when the run ended.
    remaining: int = 0
    # Pairs closer than their margin before the first iteration.
    initial: int = 0
    seconds: float = 0.0
    history: list = field(default_factory=list)
    # Vertex displacement, in the scene's own units.
    max_move: float = 0.0
    mean_move: float = 0.0
    # Number of stalls the run escalated through.
    levels: int = 0
    # Mean vertex travel at the end of the first stage, in the scene's units.
    untangled_move: float = 0.0
    recover_iterations: int = 0
    # One line per contact push tried.
    attempts: list = field(default_factory=list)
    # Pairs of triangles inside their margin whose every vertex is held: nothing
    # here can move them, so a surface that has any cannot be fully embedded.
    held_pairs: int = 0
    # Held triangles the run used, of how many there were, and how many times the
    # cut was widened because a result left the region it kept (`Settings.crop_reach`).
    held_used: int = 0
    held_total: int = 0
    crop_widenings: int = 0
    # Rounds of local solves and the number of places solved in the last one (zero
    # when the scene was solved whole).
    rounds: int = 0
    groups: int = 0
    # (surface a, surface b, side, confidence) of each pair of surfaces that
    # overlapped at the start; side is 0 where the vote was too close to call.
    sides: list = field(default_factory=list)
    message: str = ""


# ---------------------------------------------------------------------------
# Scene assembly


@dataclass
class _Scene:
    vert: np.ndarray
    tri: np.ndarray
    free: np.ndarray
    tri_off: np.ndarray
    clearance: float
    # Mesh edges as directed pairs sorted by their first vertex, so a row block
    # of the Gaussian finds its neighbors with a binary search.
    adj_i: np.ndarray
    adj_j: np.ndarray
    # Mean length of the mesh edges with a movable end.
    edge: float
    # Per-triangle rest data of the ARAP term.
    rest_inv: np.ndarray
    rest_area: np.ndarray
    # Both ends of the scene's normalization.
    center: np.ndarray
    scale: float
    spans: list
    # Surface each triangle belongs to, and for every ordered pair of surfaces
    # (a < b) the side of b's triangles that a's pairs are asked to leave by: +1
    # along b's face normal, -1 against it, 0 for no preference. `Settings` and
    # `_vote_sides` decide it.
    tri_surface: np.ndarray = None
    side: np.ndarray = None
    # Extra distance the contact penalty keeps beyond the margin a result must
    # meet. Zero while untangling; the recovery sets it so that the penalty acts
    # before the margin is reached rather than at it.
    guard: float = 0.0
    # When positive the contact penalty is smooth, v^2 / (2 quad) instead of v,
    # so that a quasi-Newton step has no kink to stall at. The recovery uses it.
    quad: float = 0.0
    # The triangles the shape energy is measured over, with their rest data: the
    # ones with a movable vertex. The rest cannot change, and contribute nothing
    # to a gradient that is read at the movable vertices.
    arap_tri: np.ndarray = None
    arap_inv: np.ndarray = None
    arap_area: np.ndarray = None
    # The candidate pairs found by the last `_candidate_pairs` and what they are
    # good for (see there).
    nlist: object = None


def _vertex_area(vert: np.ndarray, tri: np.ndarray) -> np.ndarray:
    cross = np.cross(vert[tri[:, 1]] - vert[tri[:, 0]], vert[tri[:, 2]] - vert[tri[:, 0]])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    out = np.zeros(len(vert))
    for k in range(3):
        np.add.at(out, tri[:, k], area / 3.0)
    return out


def _directed_edges(tri: np.ndarray):
    pair = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    pair = np.sort(pair, axis=1)
    pair = np.unique(pair, axis=0)
    both = np.concatenate([pair, pair[:, ::-1]])
    order = np.lexsort((both[:, 1], both[:, 0]))
    both = both[order]
    return both[:, 0].copy(), both[:, 1].copy()


def _hop_table(adj_i: np.ndarray, adj_j: np.ndarray, n: int, hop_max: int):
    """Hop distance between every pair of vertices within `hop_max` mesh edges.

    Breadth-first expansion of the set of pairs one edge at a time. Returns
    directed pairs sorted by their first vertex, with their distance.
    """
    start = np.searchsorted(adj_i, np.arange(n + 1))
    deg = np.diff(start)
    seen = adj_i.astype(np.int64) * n + adj_j
    dist = np.ones(len(seen), dtype=np.int8)
    keys = [seen]
    dists = [dist]
    frontier_i, frontier_j = adj_i.astype(np.int64), adj_j.astype(np.int64)
    known = np.sort(seen)
    for hop in range(2, hop_max + 1):
        count = deg[frontier_j]
        src = np.repeat(np.arange(len(frontier_j)), count)
        offset = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
        nxt = adj_j[start[frontier_j[src]] + offset].astype(np.int64)
        new_key = np.unique(frontier_i[src] * n + nxt)
        new_key = new_key[(new_key // n) != (new_key % n)]
        at = np.searchsorted(known, new_key)
        at[at == len(known)] = len(known) - 1
        new_key = new_key[known[at] != new_key]
        if not len(new_key):
            break
        keys.append(new_key)
        dists.append(np.full(len(new_key), hop, dtype=np.int8))
        known = np.sort(np.concatenate([known, new_key]))
        frontier_i, frontier_j = new_key // n, new_key % n
    key = np.concatenate(keys)
    d = np.concatenate(dists)
    order = np.argsort(key, kind="stable")
    key, d = key[order], d[order]
    return key // n, key % n, d


def _rest_frames(vert: np.ndarray, tri: np.ndarray):
    """Inverse of each triangle's rest edge matrix in a frame of its own plane."""
    e1 = vert[tri[:, 1]] - vert[tri[:, 0]]
    e2 = vert[tri[:, 2]] - vert[tri[:, 0]]
    l1 = np.linalg.norm(e1, axis=1)
    cross = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    ok = (l1 > 0.0) & (area > 1e-14 * np.maximum(l1, 1.0) ** 2)
    u = e1 / np.where(ok, l1, 1.0)[:, None]
    along = np.einsum("ij,ij->i", e2, u)
    w = e2 - along[:, None] * u
    lw = np.linalg.norm(w, axis=1)
    mat = np.zeros((len(tri), 2, 2))
    mat[:, 0, 0] = np.where(ok, l1, 1.0)
    mat[:, 0, 1] = along
    mat[:, 1, 1] = np.where(ok, lw, 1.0)
    inv = np.linalg.inv(mat)
    return inv, np.where(ok, area, 0.0)


def _with_shape_subset(sc: _Scene) -> _Scene:
    """Fill in the triangles `_arap` is measured over."""
    touched = sc.free[sc.tri].any(axis=1)
    if touched.all():
        sc.arap_tri, sc.arap_inv, sc.arap_area = sc.tri, sc.rest_inv, sc.rest_area
    else:
        sc.arap_tri, sc.arap_inv, sc.arap_area = sc.tri[touched], sc.rest_inv[touched], sc.rest_area[touched]
    return sc


def _assemble(surfaces: list[Surface], clearance: float) -> _Scene:
    verts, tris, free, offs, spans, owner = [], [], [], [], [], []
    base = 0
    for index, s in enumerate(surfaces):
        v = np.ascontiguousarray(s.vert, dtype=np.float64)
        t = np.ascontiguousarray(s.tri, dtype=np.int64)
        if v.ndim != 2 or v.shape[1] != 3 or t.ndim != 2 or t.shape[1] != 3:
            raise ValueError(f"{s.name}: expected (n, 3) vertices and (t, 3) triangles")
        if len(t) and (t.min() < 0 or t.max() >= len(v)):
            raise ValueError(f"{s.name}: a triangle names a vertex that does not exist")
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{s.name}: a vertex position is not finite")
        m = np.broadcast_to(np.asarray(s.movable, dtype=bool), (len(v),)).copy()
        verts.append(v)
        tris.append(t + base)
        free.append(m)
        offs.append(np.full(len(t), float(s.offset)))
        owner.append(np.full(len(t), index))
        spans.append((s.name, base, base + len(v)))
        base += len(v)
    vert = np.concatenate(verts)
    tri = np.concatenate(tris)
    lo, hi = vert.min(axis=0), vert.max(axis=0)
    center = 0.5 * (lo + hi)
    diag = float(np.linalg.norm(hi - lo))
    if diag <= 0.0:
        raise ValueError("every vertex is at one point")
    scale = 1.0 / diag
    vert = (vert - center) * scale
    adj_i, adj_j = _directed_edges(tri)
    movable = np.concatenate(free)
    # The edge length is the unit of every step and force below, so it is the
    # mean over the edges the run can move. A dense collider next to a coarse
    # garment would otherwise set it.
    touched = movable[adj_i] | movable[adj_j]
    lengths = np.linalg.norm(vert[adj_i] - vert[adj_j], axis=1)
    edge = float(lengths[touched].mean() if touched.any() else lengths.mean())
    inv, area = _rest_frames(vert, tri)
    return _with_shape_subset(_Scene(
        vert=vert,
        tri=tri,
        free=movable,
        tri_off=np.concatenate(offs) * scale,
        clearance=clearance * scale,
        adj_i=adj_i,
        adj_j=adj_j,
        edge=edge,
        rest_inv=inv,
        rest_area=area,
        center=center,
        scale=scale,
        spans=spans,
        tri_surface=np.concatenate(owner),
        side=np.zeros((len(surfaces), len(surfaces))),
    ))


# ---------------------------------------------------------------------------
# Gaussian self-contact energy


# A vertex pair counts in the Gaussian only when the two are farther apart over
# the surface than this fraction of their bandwidth.
_EXCLUDE_RATIO = 0.5


@dataclass
class _Coarse:
    """The vertices grouped into clusters the Gaussian is evaluated between.

    The Gaussian couples every pair of vertices, so evaluating it on the
    vertices costs the square of their number. It is a smooth, long-range
    interaction, so the clusters stand in for their vertices: it is evaluated on
    the area-weighted centroid of each cluster, and a cluster's force is shared
    out over its vertices by the same weights. That is the exact gradient of the
    energy of the centroids as a function of the vertices, so the objective stays
    consistent for the line search while the cost depends on the number of
    clusters alone.
    """

    cluster: np.ndarray
    count: int
    # Mesh hop distance between clusters within `hop_max` edges of one another
    # over the cluster graph, as directed pairs sorted by their first cluster.
    hop_i: np.ndarray
    hop_j: np.ndarray
    hop_d: np.ndarray
    hop_max: int
    # Mean distance between neighboring cluster centroids: what a hop is worth.
    edge: float


def _unique_directed(i: np.ndarray, j: np.ndarray, n: int):
    key = np.unique(i.astype(np.int64) * n + j)
    return key // n, key % n


def _coarsen(sc: _Scene, area: np.ndarray, target: int) -> _Coarse:
    """Group vertices into about `target` clusters that are connected over the
    mesh and of similar area.

    Each round every cluster names its nearest neighbor whose merged area stays
    under a cap, and clusters that name each other merge, which is how a
    multigrid hierarchy aggregates. Clusters are connected along mesh edges, so
    two layers that lie close in space but far apart over the surface never
    share one.
    """
    n = len(sc.vert)
    ai, aj = sc.adj_i.astype(np.int64), sc.adj_j.astype(np.int64)
    label = np.arange(n)
    node_area = area.copy()
    node_pos = sc.vert.copy()
    count = n
    cap = 2.2 * float(area.sum()) / max(target, 1)
    while count > target and len(ai):
        d = np.linalg.norm(node_pos[ai] - node_pos[aj], axis=1)
        ok = node_area[ai] + node_area[aj] <= cap
        if not ok.any():
            cap *= 1.5
            continue
        ci, cj, cd = ai[ok], aj[ok], d[ok]
        order = np.lexsort((cd, ci))
        ci, cj, cd = ci[order], cj[order], cd[order]
        first = np.ones(len(ci), bool)
        first[1:] = ci[1:] != ci[:-1]
        src, dst, dist = ci[first], cj[first], cd[first]
        pick = np.full(count, -1, dtype=np.int64)
        pick[src] = dst
        mutual = (src < dst) & (pick[dst] == src)
        src, dst, dist = src[mutual], dst[mutual], dist[mutual]
        if not len(src):
            cap *= 1.5
            continue
        need = count - target
        if len(src) > need:
            keep = np.argsort(dist, kind="stable")[:need]
            src, dst = src[keep], dst[keep]
        parent = np.arange(count)
        parent[dst] = src
        alive = parent == np.arange(count)
        new_id = np.cumsum(alive) - 1
        mp = new_id[parent]
        new_count = int(alive.sum())
        weighted = node_pos * node_area[:, None]
        new_area = np.bincount(mp, weights=node_area, minlength=new_count)
        node_pos = np.stack(
            [np.bincount(mp, weights=weighted[:, k], minlength=new_count) for k in range(3)], axis=1
        ) / np.where(new_area > 0.0, new_area, 1.0)[:, None]
        node_area = new_area
        ai, aj = mp[ai], mp[aj]
        keep = ai != aj
        ai, aj = _unique_directed(ai[keep], aj[keep], new_count)
        label = mp[label]
        count = new_count
    edge = float(np.linalg.norm(node_pos[ai] - node_pos[aj], axis=1).mean()) if len(ai) else sc.edge
    hop_max = max(1, int(np.ceil(_EXCLUDE_RATIO * _HOP_REACH / edge)))
    hop_i, hop_j, hop_d = _hop_table(ai, aj, count, hop_max)
    return _Coarse(label, count, hop_i, hop_j, hop_d, hop_max, edge)


def _gauss_pass(sc, vert, eps2, area, want_grad=True, chunk_pairs=1_500_000, ratio=_EXCLUDE_RATIO):
    """Energy, gradient and the sums the bandwidth update needs.

    Sums over unordered pairs of vertices that are not near each other along the
    surface:
        f_ij = A_i A_j eps_ij^-p exp(-r_ij^2 / eps_ij^2),  eps_ij^2 = (eps_i^2 + eps_j^2) / 2
    The intrinsic weight is zero for two vertices within `ratio` bandwidths of
    one another over the mesh, measured in hops, and at least for neighbors.
    Without it a pair of vertices a few edges apart on the same sheet would
    count as much as one on two sheets, so the sum would depend on how finely
    the sheet is cut (the bandwidth that maximizes it falls as the mesh
    refines) rather than on where the surface meets itself. The areas and the
    bandwidths are constants of the differentiation.

    Returns (energy, gradient, b, br2, beps2), the last three being, per vertex
    i, the sums over j of b_ij, b_ij r_ij^2 and b_ij eps_j^2 with
    b_ij = A_i A_j eps_ij^-(p+4) exp(-r_ij^2 / eps_ij^2).
    """
    n = len(vert)
    rows = max(1, chunk_pairs // n)
    energy = 0.0
    grad = np.zeros_like(vert) if want_grad else None
    s_b = np.zeros(n)
    s_br2 = np.zeros(n)
    s_be = np.zeros(n)
    for r0 in range(0, n, rows):
        r1 = min(n, r0 + rows)
        d = vert[r0:r1, None, :] - vert[None, :, :]
        r2 = np.einsum("ijk,ijk->ij", d, d)
        e2 = 0.5 * (eps2[r0:r1, None] + eps2[None, :])
        ex = np.exp(-r2 / e2)
        w = area[r0:r1, None] * area[None, :]
        # Hop distance of the pairs within reach, and "far" for the rest.
        lo = np.searchsorted(sc.hop_i, r0, "left")
        hi = np.searchsorted(sc.hop_i, r1, "left")
        hops = np.full((r1 - r0, n), sc.hop_max + 1, dtype=np.int16)
        hops[sc.hop_i[lo:hi] - r0, sc.hop_j[lo:hi]] = sc.hop_d[lo:hi]
        hops[np.arange(r1 - r0), np.arange(r0, r1)] = 0
        near = hops <= np.maximum(1.0, ratio * np.sqrt(e2) / sc.edge)
        ex[near] = 0.0
        f = w * ex / e2  # eps^-p with p = 2
        energy += 0.5 * float(f.sum())
        if want_grad:
            coef = -2.0 * f / e2
            grad[r0:r1] += np.einsum("ij,ijk->ik", coef, d)
        b = f / (e2 * e2)
        s_b[r0:r1] += b.sum(axis=1)
        s_br2[r0:r1] += (b * r2).sum(axis=1)
        s_be[r0:r1] += (b * eps2[None, :]).sum(axis=1)
    return energy, grad, s_b, s_br2, s_be


def _update_bandwidth(eps2, s_b, s_br2, s_be, theta):
    """One damped Jacobi step of the stationarity condition of E_G in eps_i.

    eps_i^2 = (4/p) sum_j b_ij r_ij^2 / sum_j b_ij - sum_j b_ij eps_j^2 / sum_j b_ij

    The right side can go negative when a vertex sees only far neighbors weighted
    by a much larger bandwidth; a bandwidth must be positive, so the update is
    floored at a fraction of the current value, and the number floored is
    returned so a run can show it.
    """
    safe = s_b > 0.0
    target = np.where(
        safe, (4.0 / _P) * s_br2 / np.where(safe, s_b, 1.0) - s_be / np.where(safe, s_b, 1.0), eps2
    )
    new = (1.0 - theta) * eps2 + theta * target
    floor = 0.25 * eps2
    clipped = int(np.count_nonzero(new < floor))
    return np.maximum(new, floor), clipped


def _global_bandwidth(gs, pos: np.ndarray, area: np.ndarray, start: float = 0.3, steps: int = 40) -> float:
    """The eps^2 that satisfies eps^2 = (2/p) M2 / M0, iterated to convergence, for
    a single scene-wide bandwidth."""
    eps2 = np.full(len(pos), start * start)
    for _ in range(steps):
        _, _, s_b, s_br2, _ = _gauss_pass(gs, pos, eps2, area, want_grad=False)
        if not s_b.sum() > 0.0:
            # No pair of clusters is within reach of this bandwidth (the
            # scene is a few pieces far apart): widen it until some are.
            eps2 *= 4.0
            continue
        new = (2.0 / _P) * s_br2.sum() / s_b.sum()
        if abs(new - eps2[0]) <= 1e-6 * eps2[0]:
            eps2[:] = new
            break
        eps2[:] = 0.5 * eps2[0] + 0.5 * new
    return float(eps2[0])


# ---------------------------------------------------------------------------
# As-rigid-as-possible energy


def _arap(sc: _Scene, vert: np.ndarray, want_grad=True):
    tri = sc.arap_tri
    e1 = vert[tri[:, 1]] - vert[tri[:, 0]]
    e2 = vert[tri[:, 2]] - vert[tri[:, 0]]
    edge = np.stack([e1, e2], axis=2)  # (t, 3, 2)
    f = edge @ sc.arap_inv
    u, _, vh = np.linalg.svd(f, full_matrices=False)
    r = u @ vh
    diff = f - r
    energy = float((sc.arap_area * np.einsum("tij,tij->t", diff, diff)).sum())
    if not want_grad:
        return energy, None
    p = 2.0 * sc.arap_area[:, None, None] * diff
    g = p @ np.swapaxes(sc.arap_inv, 1, 2)  # gradient w.r.t. (e1, e2)
    grad = np.zeros_like(vert)
    np.add.at(grad, tri[:, 1], g[:, :, 0])
    np.add.at(grad, tri[:, 2], g[:, :, 1])
    np.add.at(grad, tri[:, 0], -(g[:, :, 0] + g[:, :, 1]))
    return energy, grad


# ---------------------------------------------------------------------------
# Minkowski penalty


def _box_pairs(lo: np.ndarray, hi: np.ndarray, max_entries: int = 40_000_000):
    """Index pairs (i < j) of boxes that overlap, from a uniform grid.

    Each box is registered in every cell it touches. A pair is emitted from the
    one cell holding the larger of its two lower corners, so no pair is found
    twice and none needs a sort to be deduplicated.
    """
    m = len(lo)
    if m < 2:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    extent = (hi - lo).max(axis=1)
    h = float(np.mean(extent))
    if not h > 0.0:
        h = float(np.max(hi - lo)) or 1.0
    cl = np.floor(lo / h).astype(np.int64)
    ch = np.floor(hi / h).astype(np.int64)
    span = ch - cl + 1
    cnt = span.prod(axis=1)
    total = int(cnt.sum())
    if total > max_entries:
        raise MemoryError(
            f"the broad phase would register {total} box-cell entries; a triangle is far larger than the rest"
        )
    owner = np.repeat(np.arange(m), cnt)
    first = np.cumsum(cnt) - cnt
    local = np.arange(total) - np.repeat(first, cnt)
    sx, sy = span[owner, 0], span[owner, 1]
    cell = cl[owner] + np.stack([local % sx, (local // sx) % sy, local // (sx * sy)], axis=1)
    dims = cell.max(axis=0) - cell.min(axis=0) + 1
    rel = cell - cell.min(axis=0)
    key = (rel[:, 0] * dims[1] + rel[:, 1]) * dims[2] + rel[:, 2]
    order = np.argsort(key, kind="stable")
    key, owner, cell = key[order], owner[order], cell[order]
    # Run end for every entry: the position where its key changes.
    change = np.flatnonzero(np.diff(key)) + 1
    ends = np.concatenate([change, [total]])
    run_end = np.repeat(ends, np.diff(np.concatenate([[0], ends])))
    count = run_end - np.arange(total) - 1
    pairs = int(count.sum())
    if pairs > max_entries:
        raise MemoryError(f"the broad phase would test {pairs} candidate pairs")
    a_pos = np.repeat(np.arange(total), count)
    b_pos = a_pos + 1 + (np.arange(pairs) - np.repeat(np.cumsum(count) - count, count))
    ia, ib = owner[a_pos], owner[b_pos]
    mx = np.floor(np.maximum(lo[ia], lo[ib]) / h).astype(np.int64)
    keep = np.all(mx == cell[a_pos], axis=1)
    keep &= np.all(lo[ia] <= hi[ib], axis=1) & np.all(lo[ib] <= hi[ia], axis=1)
    ia, ib = ia[keep], ib[keep]
    swap = ia > ib
    ia[swap], ib[swap] = ib[swap].copy(), ia[swap].copy()
    return ia, ib


def _candidate_pairs(sc: _Scene, vert: np.ndarray, held_too: bool = False):
    """Pairs of triangles that may be within their margin of one another.

    Returns the two index arrays and two flags per pair. `share` is True when
    the triangles share a vertex: such a pair touches by construction, so it is
    asked for no clearance and counts only when the triangles overlap beyond the
    shared corner, which the solver's own crossing check also reports. `near` is
    True when the boxes of the two triangles, each grown by its offset and half
    the clearance, overlap. Closeness along a separating axis is a lower bound
    on the distance, so a pair can read as inside its margin while the boxes
    are farther apart than the margin; such a pair is not a violation, and
    `near` is how a count of violations tells.
    """
    tv = vert[sc.tri]  # (t, 3, 3)
    reach = (sc.tri_off + 0.5 * sc.clearance)[:, None]
    grown = reach + 0.5 * sc.guard
    lo = tv.min(axis=1)
    hi = tv.max(axis=1)
    # The pairs are found with every box grown by a skin as well, and kept while no
    # vertex has moved by half of it: two boxes then cannot have met that were
    # farther apart than the skin when the list was made, so the list holds every
    # pair the exact test would, plus some it rejects. The steps of the second
    # stage are small, and the list is good for many of them.
    cached = sc.nlist
    if (
        cached is not None
        and cached["held_too"] == held_too
        and float(grown.max()) <= cached["grown"]
        and 2.0 * float(np.abs(vert - cached["vert"]).max()) <= cached["skin"]
    ):
        ia, ib, share = cached["ia"], cached["ib"], cached["share"]
    else:
        skin = _SKIN * sc.edge
        ia, ib = _box_pairs(lo - grown - skin, hi + grown + skin)
        ta, tb = sc.tri[ia], sc.tri[ib]
        share = (ta[:, :, None] == tb[:, None, :]).any(axis=(1, 2))
        # A pair whose every vertex is held can move nothing and is not a pair to
        # resolve (`held_too` asks for them, to count).
        if not held_too:
            keep = sc.free[ta].any(axis=1) | sc.free[tb].any(axis=1)
            ia, ib, share = ia[keep], ib[keep], share[keep]
        sc.nlist = {
            "held_too": held_too, "grown": float(grown.max()), "skin": skin,
            "vert": vert.copy(), "ia": ia, "ib": ib, "share": share,
        }
    if not len(ia):
        return ia, ib, np.empty(0, bool), np.empty(0, bool)
    # The pairs the list holds for the skin's sake are not candidates: a pair is one
    # when its boxes, grown, overlap.
    meet = np.all(lo[ia] - grown[ia] <= hi[ib] + grown[ib], axis=1) & np.all(
        lo[ib] - grown[ib] <= hi[ia] + grown[ia], axis=1
    )
    ia, ib, share = ia[meet], ib[meet], share[meet]
    near = np.all(lo[ia] - reach[ia] <= hi[ib] + reach[ib], axis=1) & np.all(
        lo[ib] - reach[ib] <= hi[ia] + reach[ia], axis=1
    )
    return ia, ib, share, near


def _cross_back(e1, e2, u):
    """Gradient of u . (e1 x e2) with respect to e1 and to e2."""
    return np.cross(e2, u), np.cross(u, e1)


def _sat(a: np.ndarray, b: np.ndarray, margin: np.ndarray, want_grad: bool, pref=None, want_sym=False):
    """Depth of the origin in the Minkowski difference of triangle pairs, plus margin.

    a, b: (p, 3, 3) corners. The penetration depth of two convex shapes is the
    smallest overlap of their projections over the face normals of A - B, and
    those normals are among the 17 axes built here. The axis and the corners
    that realize it are returned for the gradient. The value is the depth plus
    the margin, so a positive one means the pair is closer than the margin
    along every axis.

    pref: optional (p, 3) preferred exit direction of A, a unit vector or zero.
    Two triangles are separated by moving A along one of the axes either way,
    and the cheapest way out of a poke can be the wrong one; with a preference,
    A may leave only along directions within `_SIDE_TOL` of perpendicular to it
    or on its side of it, so every pair of a patch asks for the same exit. With
    want_sym the unrestricted value is returned as a fourth result.
    """
    p = len(a)
    ea = np.roll(a, -1, axis=1) - a
    eb = np.roll(b, -1, axis=1) - b
    ca = np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0])
    cb = np.cross(b[:, 1] - b[:, 0], b[:, 2] - b[:, 0])
    axes = np.empty((p, _N_AXES, 3))
    scale = np.empty((p, _N_AXES))
    la = np.linalg.norm(ea, axis=2)
    lb = np.linalg.norm(eb, axis=2)
    lca = np.linalg.norm(ca, axis=1)
    lcb = np.linalg.norm(cb, axis=1)
    axes[:, 0], axes[:, 1] = ca, cb
    scale[:, 0] = np.linalg.norm(a[:, 1] - a[:, 0], axis=1) * np.linalg.norm(a[:, 2] - a[:, 0], axis=1)
    scale[:, 1] = np.linalg.norm(b[:, 1] - b[:, 0], axis=1) * np.linalg.norm(b[:, 2] - b[:, 0], axis=1)
    axes[:, 2:11] = np.cross(ea[:, :, None, :], eb[:, None, :, :]).reshape(p, 9, 3)
    scale[:, 2:11] = (la[:, :, None] * lb[:, None, :]).reshape(p, 9)
    axes[:, 11:14] = np.cross(ca[:, None, :], ea)
    scale[:, 11:14] = lca[:, None] * la
    axes[:, 14:17] = np.cross(cb[:, None, :], eb)
    scale[:, 14:17] = lcb[:, None] * lb
    length = np.linalg.norm(axes, axis=2)
    valid = (length > _AXIS_TOL * scale) & (scale > 0.0)
    unit = axes / np.where(valid, length, 1.0)[:, :, None]
    pa = np.einsum("pkc,pic->pki", unit, a)
    pb = np.einsum("pkc,pic->pki", unit, b)
    plus = pa.max(axis=2) - pb.min(axis=2)
    minus = pb.max(axis=2) - pa.min(axis=2)
    rows = np.arange(p)
    sym = np.where(valid, np.minimum(plus, minus), np.inf)
    if pref is None:
        plus_eff, minus_eff, overlap = plus, minus, sym
        value_sym = None
    else:
        dot = np.einsum("pkc,pc->pk", unit, pref)
        plus_eff = np.where(dot <= _SIDE_TOL, plus, np.inf)
        minus_eff = np.where(dot >= -_SIDE_TOL, minus, np.inf)
        overlap = np.where(valid, np.minimum(plus_eff, minus_eff), np.inf)
        value_sym = sym.min(axis=1) + margin
    k = overlap.argmin(axis=1)
    value = overlap[rows, k] + margin
    if value_sym is None:
        value_sym = value
    if not want_grad:
        return (value, None, None, value_sym) if want_sym else (value, None, None)
    use_plus = plus_eff[rows, k] <= minus_eff[rows, k]
    sigma = np.where(use_plus, 1.0, -1.0)
    ip = np.where(use_plus, pa[rows, k].argmax(axis=1), pa[rows, k].argmin(axis=1))
    iq = np.where(use_plus, pb[rows, k].argmin(axis=1), pb[rows, k].argmax(axis=1))
    out = (value, (k, sigma, ip, iq, unit[rows, k], length[rows, k]), (ea, eb, ca, cb))
    return out + (value_sym,) if want_sym else out


def _sat_gradient(a, b, info, edges):
    """Gradient of the overlap on the chosen axis with respect to the corners."""
    k, sigma, ip, iq, n, ln = info
    ea, eb, ca, cb = edges
    p = len(a)
    rows = np.arange(p)
    d = a[rows, ip] - b[rows, iq]
    u = sigma[:, None] * (d - n * np.einsum("pc,pc->p", n, d)[:, None]) / ln[:, None]
    ga = np.zeros_like(a)
    gb = np.zeros_like(b)
    ga[rows, ip] += sigma[:, None] * n
    gb[rows, iq] -= sigma[:, None] * n

    def face(g, c_pts, mask):
        if not mask.any():
            return
        pts = c_pts[mask]
        e1, e2 = pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0]
        d1, d2 = _cross_back(e1, e2, u[mask])
        g[mask, 1] += d1
        g[mask, 2] += d2
        g[mask, 0] -= d1 + d2

    face(ga, a, k == 0)
    face(gb, b, k == 1)

    cross = (k >= 2) & (k < 11)
    if cross.any():
        idx = np.flatnonzero(cross)
        i = (k[idx] - 2) // 3
        j = (k[idx] - 2) % 3
        d_a, d_b = _cross_back(ea[idx, i], eb[idx, j], u[idx])
        ga[idx, (i + 1) % 3] += d_a
        ga[idx, i] -= d_a
        gb[idx, (j + 1) % 3] += d_b
        gb[idx, j] -= d_b

    for lo, pts, g, c, e in ((11, a, ga, ca, ea), (14, b, gb, cb, eb)):
        sel = (k >= lo) & (k < lo + 3)
        if not sel.any():
            continue
        idx = np.flatnonzero(sel)
        i = k[idx] - lo
        ei = e[idx, i]
        # axis = c x e_i, with c itself the cross product of two corner edges
        w = np.cross(ei, u[idx])
        direct = np.cross(u[idx], c[idx])
        q = pts[idx]
        e1, e2 = q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]
        d1, d2 = _cross_back(e1, e2, w)
        g[idx, 1] += d1
        g[idx, 2] += d2
        g[idx, 0] -= d1 + d2
        g[idx, (i + 1) % 3] += direct
        g[idx, i] -= direct
    return ga, gb


# A depth below this is rounding, in units of the scene's diagonal. A pair of
# triangles that share a vertex touches there, so its depth is zero up to a few
# ulps either way. And the penalty drives a pair to the edge of its margin, where
# the sum of the depth and the margin is zero up to the same rounding, so a pair
# is inside its margin only past it: the run then ends on a state a later
# evaluation, with other terms added and subtracted, reads the same way.
_DEPTH_TOL = 1e-12

# The skin of the candidate-pair list, in mean edge lengths (see `_candidate_pairs`).
_SKIN = 0.25


def _screen(a: np.ndarray, b: np.ndarray, pref=None) -> np.ndarray:
    """An upper bound on the smallest overlap of each triangle pair, from the two
    face normals alone.

    The overlap over all 17 axes is at most its value over any two of them (with
    the same restriction on exit directions, if there is one), so a pair whose
    bound plus margin is not positive is not inside its margin and needs no more
    work. Most candidates are layers lying parallel to one another or neighbors
    lying flat, and one of the face normals separates them.
    """
    bound = np.full(len(a), np.inf)
    for p, q in ((a, b), (b, a)):
        c = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
        length = np.linalg.norm(c, axis=1)
        scale = np.linalg.norm(p[:, 1] - p[:, 0], axis=1) * np.linalg.norm(p[:, 2] - p[:, 0], axis=1)
        ok = (length > _AXIS_TOL * scale) & (scale > 0.0)
        n = c / np.where(ok, length, 1.0)[:, None]
        pp = np.einsum("pc,pic->pi", n, p)
        pq = np.einsum("pc,pic->pi", n, q)
        # p is A or B; the sums below are those of A's projection against B's,
        # so for p = B the roles of the two overlaps are exchanged.
        hi_a, lo_a = (pp.max(axis=1), pp.min(axis=1)) if p is a else (pq.max(axis=1), pq.min(axis=1))
        hi_b, lo_b = (pq.max(axis=1), pq.min(axis=1)) if p is a else (pp.max(axis=1), pp.min(axis=1))
        plus = hi_a - lo_b
        minus = hi_b - lo_a
        if pref is not None:
            dot = np.einsum("pc,pc->p", n, pref)
            plus = np.where(dot <= _SIDE_TOL, plus, np.inf)
            minus = np.where(dot >= -_SIDE_TOL, minus, np.inf)
        bound = np.minimum(bound, np.where(ok, np.minimum(plus, minus), np.inf))
    return bound


def _screen_edges(a: np.ndarray, b: np.ndarray, pref=None) -> np.ndarray:
    """The same bound over the nine edge-edge axes: for the pairs the face normals
    could not separate, which are mostly triangles leaning toward each other that
    are still apart."""
    ea = np.roll(a, -1, axis=1) - a
    eb = np.roll(b, -1, axis=1) - b
    axes = np.cross(ea[:, :, None, :], eb[:, None, :, :]).reshape(len(a), 9, 3)
    la = np.linalg.norm(ea, axis=2)
    lb = np.linalg.norm(eb, axis=2)
    scale = (la[:, :, None] * lb[:, None, :]).reshape(len(a), 9)
    length = np.linalg.norm(axes, axis=2)
    valid = (length > _AXIS_TOL * scale) & (scale > 0.0)
    unit = axes / np.where(valid, length, 1.0)[:, :, None]
    pa = np.einsum("pkc,pic->pki", unit, a)
    pb = np.einsum("pkc,pic->pki", unit, b)
    plus = pa.max(axis=2) - pb.min(axis=2)
    minus = pb.max(axis=2) - pa.min(axis=2)
    if pref is not None:
        dot = np.einsum("pkc,pc->pk", unit, pref)
        plus = np.where(dot <= _SIDE_TOL, plus, np.inf)
        minus = np.where(dot >= -_SIDE_TOL, minus, np.inf)
    return np.where(valid, np.minimum(plus, minus), np.inf).min(axis=1)


def _preferred_exit(sc: _Scene, ia: np.ndarray, ib: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Preferred exit direction of the first triangle of each pair, or None when
    no pair has one: the side chosen for its surface times the face normal of the
    second triangle."""
    if not sc.side.any():
        return None
    sigma = sc.side[sc.tri_surface[ia], sc.tri_surface[ib]]
    if not sigma.any():
        return None
    c = np.cross(b[:, 1] - b[:, 0], b[:, 2] - b[:, 0])
    length = np.linalg.norm(c, axis=1)
    n = c / np.where(length > 0.0, length, 1.0)[:, None]
    return sigma[:, None] * n


def _mink_pass(sc: _Scene, vert: np.ndarray, want_grad=True, chunk=20000):
    """Penalty, gradient and the number of pairs inside their margin.

    penalty = sum over pairs of max(0, depth + margin + guard), margin = offset_a
    + offset_b + clearance, or 0 for two triangles that share a vertex, with the
    depth measured along the exits the pair's surfaces prefer (`_sat`). The
    count is of pairs inside their margin proper, without the guard and without
    the preference: it asks whether the surface is embedded, not whether it is
    on the side chosen for it.
    """
    grad = np.zeros_like(vert) if want_grad else None
    ia, ib, share, near = _candidate_pairs(sc, vert)
    total = 0.0
    violating = 0
    worst = 0.0
    for c0 in range(0, len(ia), chunk):
        sl = slice(c0, c0 + chunk)
        ta, tb = sc.tri[ia[sl]], sc.tri[ib[sl]]
        a, b = vert[ta], vert[tb]
        shared = share[sl]
        guard = np.where(shared, 0.0, sc.guard)
        margin = np.where(shared, 0.0, sc.tri_off[ia[sl]] + sc.tri_off[ib[sl]] + sc.clearance)
        floor = _DEPTH_TOL
        pref = _preferred_exit(sc, ia[sl], ib[sl], b)
        # Pairs the face normals already separate are not inside their margin,
        # and then the pairs the edge-edge axes separate; the rest get all 17.
        which = np.arange(len(a))
        for bound in (_screen, _screen_edges):
            keep = bound(a, b, pref) + margin + guard > floor
            if not keep.all():
                a, b, ta, tb = a[keep], b[keep], ta[keep], tb[keep]
                margin, guard, which = margin[keep], guard[keep], which[keep]
                pref = None if pref is None else pref[keep]
            if not len(a):
                break
        if not len(a):
            continue
        value, info, edges, value_sym = _sat(a, b, margin + guard, want_grad, pref, want_sym=True)
        hit = value > floor
        violating += int(np.count_nonzero((value_sym - guard > floor) & near[sl][which]))
        if not hit.any():
            continue
        worst = max(worst, float(value[hit].max()))
        if sc.quad > 0.0:
            total += float((0.5 * value[hit] ** 2 / sc.quad).sum())
        else:
            total += float(value[hit].sum())
        if want_grad:
            sub = np.flatnonzero(hit)
            sel_info = tuple(x[sub] for x in info)
            sel_edges = tuple(x[sub] for x in edges)
            ga, gb = _sat_gradient(a[sub], b[sub], sel_info, sel_edges)
            if sc.quad > 0.0:
                w = (value[sub] / sc.quad)[:, None, None]
                ga, gb = ga * w, gb * w
            np.add.at(grad, ta[sub].ravel(), ga.reshape(-1, 3))
            np.add.at(grad, tb[sub].ravel(), gb.reshape(-1, 3))
    return total, grad, violating, worst


def _vote_sides(sc: _Scene, coarse: _Coarse, area: np.ndarray, confidence: float):
    """Decide, for every pair of surfaces that overlap, which side of the second
    the first is to leave by.

    A part of a surface that has poked through another is the minority: the rest
    of it lies on the other side, and the evidence is the whole surface, not the
    few triangles that touch. So each cluster of A is placed against its nearest
    cluster of B, on the side its face normal faces or the opposite one, and the
    larger share of A's area decides. The overlap itself says nothing, since a
    poke is balanced around its rim. Clusters offset from B along its surface
    rather than across it have no side and do not vote. When the shares are
    within `confidence` of each other (or too little of A has a side) the sheets
    cross each other and no side is right: the preference is left at zero and the
    Gaussian is what decides.

    Returns a list of (surface a, surface b, side, confidence) for the report.
    """
    ns = len(sc.side)
    sc.side[:] = 0.0
    if ns < 2:
        return []
    saved = sc.guard
    sc.guard = 0.0
    ia, ib, _, _ = _candidate_pairs(sc, sc.vert)
    sc.guard = saved
    other = sc.tri_surface[ia] != sc.tri_surface[ib]
    ia, ib = ia[other], ib[other]
    if not len(ia):
        return []
    a, b = sc.vert[sc.tri[ia]], sc.vert[sc.tri[ib]]
    margin = sc.tri_off[ia] + sc.tri_off[ib] + sc.clearance
    hit = _sat(a, b, margin, False)[0] > _DEPTH_TOL
    pairs = {(int(x), int(y)) for x, y in zip(sc.tri_surface[ia][hit], sc.tri_surface[ib][hit], strict=True)}
    if not pairs:
        return []
    # Cluster centroids and areas of every surface, and B's triangles.
    k = coarse.count
    cl = coarse.cluster
    total = np.bincount(cl, weights=area, minlength=k)
    share = area / np.where(total[cl] > 0.0, total[cl], 1.0)
    cent = np.stack([np.bincount(cl, weights=share * sc.vert[:, c], minlength=k) for c in range(3)], axis=1)
    surface_of_vertex = np.empty(len(sc.vert), dtype=np.int64)
    for index, (_, lo, hi) in enumerate(sc.spans):
        surface_of_vertex[lo:hi] = index
    surface = np.zeros(k, dtype=np.int64)
    surface[cl] = surface_of_vertex
    corner = sc.vert[sc.tri]
    c = np.cross(corner[:, 1] - corner[:, 0], corner[:, 2] - corner[:, 0])
    length = np.linalg.norm(c, axis=1)
    normal = c / np.where(length > 0.0, length, 1.0)[:, None]
    middle = corner.mean(axis=1)
    radius = np.linalg.norm(corner - middle[:, None, :], axis=2).max(axis=1)
    votes = []
    for x, y in sorted(pairs):
        mine = np.flatnonzero(surface == x)
        theirs = np.flatnonzero((sc.tri_surface == y) & (length > 0.0))
        if not len(mine) or not len(theirs):
            # A surface of nothing but degenerate triangles has no plane to be on a side of.
            continue
        # The nearest triangle of B to each cluster of A, and which side of that
        # triangle's plane the cluster is on.
        d = cent[mine][:, None, :] - middle[theirs][None, :, :]
        dist2 = np.einsum("ijc,ijc->ij", d, d)
        near = dist2.argmin(axis=1)
        row = np.arange(len(mine))
        off = d[row, near]
        along = np.einsum("pc,pc->p", off, normal[theirs][near])
        gap = np.sqrt(dist2[row, near])
        # A side is read where the cluster lies over the triangle (within a
        # triangle's reach of it along the plane) or well off it along the
        # normal; a cluster far to the side of everything has none.
        lateral = np.sqrt(np.maximum(gap**2 - along**2, 0.0))
        decided = (lateral <= 1.5 * radius[theirs][near]) | (np.abs(along) >= 0.25 * gap)
        w = total[mine]
        front = float(w[decided & (along > 0.0)].sum())
        back = float(w[decided & (along < 0.0)].sum())
        counted = front + back
        sign, sure = 0.0, 0.0
        if counted >= 0.25 * float(w.sum()):
            sure = abs(front - back) / counted
            if sure >= confidence:
                sign = 1.0 if front > back else -1.0
        sc.side[x, y] = sign
        votes.append((x, y, sign, float(sure)))
    return votes


# ---------------------------------------------------------------------------
# Optimization


class _Evaluator:
    """The objective at a point, with weights the run changes as it escalates."""

    def __init__(self, sc: _Scene, coarse: _Coarse, area: np.ndarray):
        self.sc, self.coarse, self.area = sc, coarse, area
        # Bandwidth of every cluster.
        self.eps2 = np.ones(coarse.count)
        self.free_idx = np.flatnonzero(sc.free)
        self.wg = self.wm = self.wr = self.wp = 0.0
        # Anchor of the pull term: the positions the recovery returns toward,
        # and the per-vertex weight of that pull.
        self.anchor = sc.vert.copy()
        self.pull = np.zeros(len(sc.vert))
        self.stats = None
        self.parts = (0.0, 0.0, 0.0)
        self.active = 0
        self.worst = 0.0

    def full(self, x: np.ndarray) -> np.ndarray:
        vert = self.sc.vert.copy()
        vert[self.free_idx] = x.reshape(-1, 3)
        return vert

    def cluster_areas(self) -> np.ndarray:
        return np.bincount(self.coarse.cluster, weights=self.area, minlength=self.coarse.count)

    def centroids(self, vert: np.ndarray):
        """Area-weighted centroid of every cluster, and each vertex's share of it."""
        cl, k = self.coarse.cluster, self.coarse.count
        total = self.cluster_areas()
        share = self.area / np.where(total[cl] > 0.0, total[cl], 1.0)
        cent = np.stack([np.bincount(cl, weights=share * vert[:, c], minlength=k) for c in range(3)], axis=1)
        return cent, share, total

    def gauss(self, vert: np.ndarray, want_grad=True):
        """Gaussian energy of the cluster centroids, its gradient with respect to
        the vertices, and the sums the bandwidth update needs."""
        cent, share, total = self.centroids(vert)
        eg, gc, s_b, s_br2, s_be = _gauss_pass(self.coarse, cent, self.eps2, total, want_grad)
        grad = gc[self.coarse.cluster] * share[:, None] if want_grad else None
        return eg, grad, (s_b, s_br2, s_be)

    def __call__(self, x: np.ndarray):
        vert = self.full(x)
        eg = 0.0
        g = np.zeros_like(vert)
        if self.wg > 0.0:
            eg, gg, self.stats = self.gauss(vert)
            g += self.wg * gg
        em, gm, self.active, self.worst = _mink_pass(self.sc, vert)
        er, gr = _arap(self.sc, vert)
        self.parts = (eg, em, er)
        g += self.wm * gm + self.wr * gr
        f = self.wg * eg + self.wm * em + self.wr * er
        if self.wp > 0.0:
            delta = vert - self.anchor
            f += self.wp * float((self.pull * np.einsum("ij,ij->i", delta, delta)).sum())
            g += 2.0 * self.wp * self.pull[:, None] * delta
        return f, g[self.free_idx].ravel()


class _Lbfgs:
    """Limited-memory BFGS with an Armijo backtracking line search."""

    def __init__(self, memory: int):
        self.memory = memory
        self.pairs: list = []

    def reset(self):
        self.pairs.clear()

    def _direction(self, g):
        q = g.copy()
        alpha = []
        for s, y, rho in reversed(self.pairs):
            a = rho * float(s @ q)
            alpha.append(a)
            q -= a * y
        if self.pairs:
            s, y, _ = self.pairs[-1]
            q *= float(s @ y) / float(y @ y)
        for (s, y, rho), a in zip(self.pairs, reversed(alpha), strict=True):
            b = rho * float(y @ q)
            q += s * (a - b)
        return -q

    def step(
        self,
        ev: _Evaluator,
        x,
        f,
        g,
        first: float,
        cap: float | None = None,
        feasible: bool = False,
        shortest: float = 1e-10,
        relative: bool = False,
    ):
        """One iteration. Returns (x, f, g, taken), `taken` False when the line
        search found nothing.

        first: the longest first step, as a vertex displacement, when there is no
            curvature yet. cap: the longest step any vertex may take, always.
        feasible: accept only points with no pair inside its margin.
        """
        d = self._direction(g)
        slope = float(g @ d)
        if slope >= 0.0:
            self.pairs.clear()
            d = -g
            slope = -float(g @ g)
        longest = float(np.linalg.norm(d.reshape(-1, 3), axis=1).max())
        if longest <= 0.0:
            return x, f, g, False
        alpha = 1.0 if self.pairs else min(1.0, first / longest)
        if cap is not None:
            alpha = min(alpha, cap / longest)
        limit = shortest * alpha if relative else shortest
        while alpha >= limit:
            xn = x + alpha * d
            try:
                fn, gn = ev(xn)
            except MemoryError:
                # A step this long scatters the vertices so that the broad
                # phase finds more pairs than fit: not a point to move to.
                alpha *= 0.5
                continue
            if fn <= f + 1e-4 * alpha * slope and (not feasible or ev.active == 0):
                s, y = xn - x, gn - g
                sy = float(s @ y)
                if sy > 1e-12 * float(np.linalg.norm(s) * np.linalg.norm(y)):
                    self.pairs.append((s, y, 1.0 / sy))
                    if len(self.pairs) > self.memory:
                        self.pairs.pop(0)
                return xn, fn, gn, True
            alpha *= 0.5
        return x, f, g, False


def _smooth_basis(sc: _Scene, coarse: _Coarse, rounds: int) -> np.ndarray:
    """Blend weights of every free vertex over the clusters, shape (free, clusters).

    The indicator of each cluster, smoothed over the mesh a few rounds, so that a
    vertex follows its own cluster and the ones next to it, and a translation of
    the clusters moves the surface smoothly. Rows sum to one.
    """
    n, k = len(sc.vert), coarse.count
    weights = np.zeros((k, n))
    weights[coarse.cluster, np.arange(n)] = 1.0
    degree = np.bincount(sc.adj_i, minlength=n).astype(np.float64)
    has = degree > 0.0
    for _ in range(rounds):
        mean = np.empty_like(weights)
        for c in range(k):
            mean[c] = np.bincount(sc.adj_i, weights=weights[c][sc.adj_j], minlength=n)
        mean[:, has] /= degree[has]
        mean[:, ~has] = weights[:, ~has]
        weights = 0.5 * (weights + mean)
    return np.ascontiguousarray(weights[:, sc.free].T)


class _Reduced:
    """The objective restricted to smooth motions: x = base + basis @ c, with a
    translation c for every cluster. Same interface as `_Evaluator`."""

    def __init__(self, ev: _Evaluator, basis: np.ndarray, base: np.ndarray):
        self.ev, self.basis = ev, basis
        self.base = base.reshape(-1, 3)
        self.active = 0

    def positions(self, c: np.ndarray) -> np.ndarray:
        return (self.base + self.basis @ c.reshape(-1, 3)).ravel()

    def __call__(self, c: np.ndarray):
        f, g = self.ev(self.positions(c))
        self.active = self.ev.active
        return f, (self.basis.T @ g.reshape(-1, 3)).ravel()


def _reduced_run(st, ev, basis, x_base, stage, first, cap, feasible, limit):
    """L-BFGS over the cluster translations, from x_base, until the surface is
    embedded (when it is not required to start so), the objective stops paying,
    or the limit. Returns the vertex positions reached."""
    red = _Reduced(ev, basis, x_base)
    c = np.zeros(basis.shape[1] * 3)
    f, g = red(c)
    solver = _Lbfgs(st.memory)
    recent = [f]
    for it in range(1, limit + 1):
        _tick(st, stage, it, ev.active)
        if not feasible and ev.active == 0:
            break
        cn, fn, gn, taken = solver.step(red, c, f, g, first=first, cap=cap, feasible=feasible)
        if not taken:
            if solver.pairs:
                solver.reset()
                continue
            break
        c, f, g = cn, fn, gn
        recent.append(f)
        if len(recent) > st.recover_window:
            recent.pop(0)
            if recent[0] - f <= st.coarse_gain * abs(recent[0]):
                break
    return red.positions(c)


def _tick(st: Settings, stage: str, iteration: int, pairs: int):
    if st.progress is not None and st.progress(stage, iteration, pairs) is False:
        raise RunCanceledError


def _recover(sc, ev, x_start, x_orig, st, edge, rep, note, basis=None):
    """Pull the untangled surface back toward the input, staying embedded.

    The reverse flow of the paper, with its IPC barrier and continuous collision
    detection replaced by two cheaper guards: the contact penalty is weighted
    well above the pull so it holds a pair at its margin, and a step is taken
    only if it ends with no pair inside its margin, each vertex moving less than
    half the thinnest margin so that no pair can cross a margin and come out the
    far side within one step.
    """
    ev.wg = 0.0
    # Any exit holds a pair at its margin now that the surface is embedded.
    sc.side[:] = 0.0
    ev.wr = 1.0 / max(float(ev.area.sum()), 1e-12)
    ev.anchor = sc.vert.copy()
    ev.anchor[ev.free_idx] = x_orig.reshape(-1, 3)
    # A spring to the input about as stiff as the shape energy it works with:
    # vertex area over the squared edge length is a dimensionless count of
    # triangles.
    area_pull = ev.area / (float(ev.area.sum()) * edge * edge)
    ev.wp = st.recover_pull
    delta = x_start.reshape(-1, 3) - x_orig.reshape(-1, 3)
    # The contact penalty is sized against the strongest force of that spring.
    pull_force = 2.0 * ev.wp * float((area_pull[ev.free_idx] * np.linalg.norm(delta, axis=1)).max())
    ev.pull = area_pull
    ev.wm = max(st.recover_hold * pull_force, ev.wr * edge)
    margin = float((2.0 * sc.tri_off + sc.clearance).min())
    cap = 0.5 * margin if margin > 0.0 else 0.25 * edge
    cap = max(cap, 1e-3 * edge)
    solver = _Lbfgs(st.memory)
    x = x_start.copy()
    # The penalty starts a guard band before the margin: a pair at the margin
    # itself has no force on it and any step toward it would end inside. The pull
    # then holds a pair at the edge of the band, not at the margin, so the band
    # is narrowed in turn and the surface settles toward the margin itself.
    sc.guard = 2.0 * cap
    sc.quad = sc.guard if st.recover_smooth else 0.0
    f, g = ev(x)
    if ev.active:
        sc.guard = sc.quad = 0.0
        note("recovery skipped: the start is not embedded")
        return x
    it = 0
    canceled = False
    if basis is not None and float(np.linalg.norm(delta, axis=1).max()) > st.coarse_recover_edges * edge:
        try:
            x = _reduced_run(st, ev, basis, x, "recover", cap, cap, True, st.coarse_recover_iter)
        except RunCanceledError:
            note("recovery canceled; the embedded state reached so far is returned")
            sc.guard = sc.quad = 0.0
            return x
    def band(scale, gain, limit):
        nonlocal x, it, canceled
        sc.guard = scale * cap
        sc.quad = sc.guard if st.recover_smooth else 0.0
        solver.reset()
        f, g = ev(x)
        quiet = 0
        recent = [f]
        travel = []
        while it < limit:
            it += 1
            try:
                _tick(st, "recover", it, ev.active)
            except RunCanceledError:
                note("recovery canceled; the embedded state reached so far is returned")
                canceled = True
                break
            xn, fn, gn, taken = solver.step(ev, x, f, g, first=cap, cap=cap, feasible=True, shortest=1e-3, relative=True)
            if not taken:
                if solver.pairs:
                    solver.reset()
                    continue
                break
            moved = float(np.linalg.norm((xn - x).reshape(-1, 3), axis=1).max())
            x, f, g = xn, fn, gn
            rep.recover_iterations = it
            quiet = quiet + 1 if moved < 1e-2 * cap else 0
            if st.verbose and it % 25 == 0:
                dist = float(np.linalg.norm((x - x_orig).reshape(-1, 3), axis=1).mean())
                print(f"recover {it:4d} band {scale:.2f}  f {f:.4e}  mean distance to input {dist / sc.scale:.4f}  step {moved:.2e}")
            if quiet >= 5:
                break
            # A band is done when the objective has stopped paying for the
            # iterations: it fell by under `recover_gain` over the last window,
            # and the surface is no longer travelling at the speed the step cap
            # allows. A surface far from its input falls by little per capped
            # step next to an objective that is large, and that is slow, not
            # finished.
            recent.append(f)
            travel.append(moved)
            if len(recent) > st.recover_window:
                recent.pop(0)
                travel.pop(0)
                if recent[0] - f <= gain * abs(recent[0]) and float(np.mean(travel)) < 0.25 * cap:
                    break

    for scale in st.recover_bands:
        band(scale, st.recover_gain, st.recover_iter)
        if canceled:
            break
    if not canceled:
        # With the surface back near its input, the bands again from the second
        # on, with the spring never weaker than a mean vertex's. A vertex on the
        # boundary has a fraction of an interior vertex's area, and a corner a
        # sixth of it, so the area-weighted spring leaves the corner triangles
        # (which the shape energy does not hold in place, since a lone triangle
        # can turn freely) too little to be pulled back, and they stay curled
        # up (14 degrees off the input's normal in two wavy sheets, against 1
        # degree on average). The floor is applied only here: while the surface
        # is far from its input, a stiff boundary spring against contacts that
        # hold the middle makes a valley too sharp for the steps to follow (a
        # sheet poked by a ball stopped 0.2 of its diagonal from its input).
        ev.pull = np.maximum(area_pull, float(area_pull[ev.free_idx].mean()))
        limit = it + max(st.recover_iter // 3, 30)
        for scale in st.recover_bands[1:] or st.recover_bands[-1:]:
            band(scale, st.polish_gain, limit)
            if canceled:
                break
    sc.guard = sc.quad = 0.0
    return x


@dataclass
class _Prepared:
    """What every attempt on one input shares."""

    sc: _Scene
    coarse: _Coarse
    area: np.ndarray
    # (surface a, surface b, side, confidence) per overlapping pair of surfaces,
    # by index, and the side matrix they were written to.
    votes: list
    side: np.ndarray
    # Blend weights of the free vertices over the clusters, built on first use.
    basis: object = None


def _prepare(surfaces: list[Surface], st: Settings) -> _Prepared:
    sc = _assemble(surfaces, st.clearance)
    if not sc.free.any():
        raise ValueError("no vertex is movable")
    area = _vertex_area(sc.vert, sc.tri)
    coarse = _coarsen(sc, area, st.clusters)
    votes = _vote_sides(sc, coarse, area, st.side_confidence) if st.orient_sides else []
    return _Prepared(sc, coarse, area, votes, sc.side.copy())


def _basis_for(prep: _Prepared, st: Settings):
    """The cluster basis when the mesh is large enough to want two levels, else
    None."""
    free = int(prep.sc.free.sum())
    if free < st.coarse_min_vertices or prep.coarse.count >= free:
        return None
    # The basis is dense, free vertices by clusters. Past this many entries
    # (320 MB) the coarse level is skipped rather than risking the memory of the
    # session it runs in.
    if free * prep.coarse.count > _BASIS_LIMIT:
        return None
    if prep.basis is None:
        prep.basis = _smooth_basis(prep.sc, prep.coarse, st.coarse_rounds)
    return prep.basis


def _untangle_once(surfaces: list[Surface], st: Settings, prep: _Prepared, mink_push: float, oriented: bool):
    """One attempt, at one contact push, with or without the chosen sides. See
    `untangle`."""
    t0 = time.perf_counter()
    sc, coarse, area = prep.sc, prep.coarse, prep.area
    sc.side[:] = prep.side if oriented else 0.0
    sc.guard = sc.quad = 0.0
    rep = Report()

    def finish(final: np.ndarray):
        out = []
        for surf, (_, a, b) in zip(surfaces, sc.spans, strict=True):
            res = np.array(surf.vert, dtype=np.float64)
            moved = final[a:b] / sc.scale + sc.center
            free = sc.free[a:b]
            res[free] = moved[free]
            out.append(res)
        return out

    def note(message: str):
        rep.message = message
        if st.verbose:
            print(message)

    ev = _Evaluator(sc, coarse, area)
    total_area = float(area.sum())
    edge = sc.edge
    ev.wr = 1.0 / max(total_area, 1e-12)
    # A poke is taken back by the pairs along its rim, a number that grows with
    # the resolution while the shape energy's stiffness per vertex does not, so
    # an oriented push is a length: how far the rim can win against the shape
    # energy. The weak push is area-consistent instead (see `Settings`).
    push = ev.wr * (mink_push if oriented else edge * edge * mink_push)
    ev.wm = push

    x = sc.vert[sc.free].ravel().copy()
    x_orig = x.copy()
    f, g = ev(x)
    rep.initial = ev.active
    if rep.initial == 0:
        rep.converged = True
        note("no pair is within its margin")
        rep.seconds = time.perf_counter() - t0
        return finish(sc.vert), rep

    gauss_unit = 0.0
    beta = 0.0
    stalls = 0
    boosts = 0
    count = 0

    def switch_on_gauss(x):
        nonlocal gauss_unit, beta
        ev.area = area
        cent, _, total = ev.centroids(ev.full(x))
        ev.eps2[:] = _global_bandwidth(coarse, cent, total)
        beta = st.gauss_start
        _, gg, _ = ev.gauss(ev.full(x))
        raw = float(np.linalg.norm(gg[ev.free_idx], axis=1).max())
        gauss_unit = push / max(raw, 1e-300)
        ev.wg = beta * gauss_unit

    def descend(obj, p, to_x, budget, max_stalls):
        """L-BFGS on `obj` from `p` until the surface is embedded, the budget is
        spent, it diverges, or the Gaussian has been strengthened as far as it
        goes. `to_x` gives the vertex positions of a point. Returns the point and
        which of those ended it."""
        nonlocal stalls, beta, count, boosts
        solver = _Lbfgs(st.memory)
        level_stalls = still = in_level = 0
        f, g = obj(p)
        # The point with the fewest pairs inside their margin, which is what a
        # level that does not embed the surface hands on: the iterates after a
        # stall can be a good deal worse than the one before it.
        best_active, best_p = ev.active, p.copy()
        for _ in range(budget):
            if ev.active == 0:
                return p, "embedded"
            count += 1
            _tick(st, "untangle", count, ev.active)
            if ev.wg > 0.0 and st.bandwidth_every and (count % st.bandwidth_every == 0):
                ev.area = _vertex_area(ev.full(to_x(p)), sc.tri)
                ev.eps2[:], _ = _update_bandwidth(ev.eps2, *ev.stats, st.bandwidth_damping)
                f, g = obj(p)
                solver.reset()
            p_prev = p
            p, f, g, taken = solver.step(obj, p, f, g, first=0.5 * edge)
            if taken and ev.active < best_active:
                best_active, best_p = ev.active, p.copy()
            moved = float(np.linalg.norm((p - p_prev).reshape(-1, 3), axis=1).max())
            still = still + 1 if (not taken or moved < st.still_step * edge) else 0
            in_level += 1
            eg, em, er = ev.parts
            rep.iterations = count
            rep.history.append((count, eg, em, er, ev.active, float(np.sqrt(ev.eps2).mean()) if ev.wg > 0 else 0.0, stalls))
            if st.verbose and count % 10 == 0:
                print(
                    f"it {count:4d} L{stalls:<2d} f {f:.4e}  EG {eg:.3e}  EM {em:.3e}  ER {er:.3e}  "
                    f"pairs {ev.active:6d}  eps {np.sqrt(ev.eps2).mean() if ev.wg > 0 else 0.0:.3f}  wg {ev.wg:.2e}"
                )
            if ev.active == 0:
                return p, "embedded"
            if float(np.abs(to_x(p) - x_orig).max()) > st.max_travel:
                return best_p, "diverged"
            if still >= 5 or in_level >= st.level_iter:
                stalls += 1
                level_stalls += 1
                if level_stalls > max_stalls:
                    return best_p, "gave up"
                if oriented and boosts < st.push_boosts:
                    boosts += 1
                    ev.wm *= st.push_boost
                elif ev.wg == 0.0:
                    switch_on_gauss(to_x(p))
                else:
                    beta *= st.gauss_growth
                    ev.wg = beta * gauss_unit
                solver.reset()
                still = in_level = 0
                f, g = obj(p)
        return best_p, "budget"

    # With a side chosen, the contact penalty alone resolves a poke, and the
    # Gaussian would only move the whole surface; it waits for the first stall.
    if st.gauss_first and not oriented:
        switch_on_gauss(x)

    outcome = ""
    basis = _basis_for(prep, st)
    if basis is not None:
        # The coarse level: one translation per cluster.
        red = _Reduced(ev, basis, x)
        # With a side chosen the Gaussian is not what the coarse level is for: when
        # its steps die out it hands what is left to the vertices instead of
        # strengthening a force that moves whole clusters (on a dress's seams, the
        # last ten pairs of thirteen thousand did that and sent the count back up
        # to two thousand).
        c, outcome = descend(
            red, np.zeros(basis.shape[1] * 3), red.positions, st.coarse_iter, 0 if oriented else st.max_stalls
        )
        x = red.positions(c)
    if outcome not in ("embedded", "diverged"):
        x, outcome = descend(ev, x, lambda v: v, st.max_iter, st.max_stalls)
    f, g = ev(x)
    if outcome == "embedded":
        rep.converged = True
    elif outcome == "diverged":
        note("diverged")
    elif outcome == "gave up":
        note("not embedded after the strongest Gaussian")
    else:
        note("iteration budget spent")
    rep.untangled_move = float(np.linalg.norm((x - x_orig).reshape(-1, 3), axis=1).mean()) / sc.scale
    if rep.converged and st.recover:
        x = _recover(sc, ev, x, x_orig, st, edge, rep, note, _basis_for(prep, st))
        ev.wg = 0.0
        ev.wm = 1.0
        ev(x)
    rep.remaining = ev.active
    final = ev.full(x)
    moved = np.linalg.norm(final - sc.vert, axis=1)[sc.free] / sc.scale
    rep.max_move = float(moved.max())
    rep.mean_move = float(moved.mean())
    rep.levels = stalls
    rep.seconds = time.perf_counter() - t0
    return finish(final), rep


def _count_held_pairs(sc: _Scene) -> int:
    """Pairs inside their margin whose triangles have every vertex held."""
    if sc.free.all():
        return 0
    ia, ib, share, _ = _candidate_pairs(sc, sc.vert, held_too=True)
    held = ~(sc.free[sc.tri[ia]].any(axis=1) | sc.free[sc.tri[ib]].any(axis=1))
    ia, ib, share = ia[held], ib[held], share[held]
    if not len(ia):
        return 0
    margin = np.where(share, 0.0, sc.tri_off[ia] + sc.tri_off[ib] + sc.clearance)
    return int(np.count_nonzero(_sat(sc.vert[sc.tri[ia]], sc.vert[sc.tri[ib]], margin, False)[0] > _DEPTH_TOL))


def _untangle_surfaces(surfaces: list[Surface], st: Settings):
    """`untangle` on exactly the surfaces given."""
    t0 = time.perf_counter()
    prep = _prepare(surfaces, st)
    held_pairs = _count_held_pairs(prep.sc)
    rungs = [(float(push), False) for push in st.mink_push]
    if any(sign != 0.0 for _, _, sign, _ in prep.votes):
        rungs = [(float(push), True) for push in st.mink_push_oriented] + rungs
    if not rungs:
        raise ValueError("no contact push to try")
    notes = []
    best = None
    sides = [(surfaces[x].name, surfaces[y].name, sign, sure) for x, y, sign, sure in prep.votes]
    try:
        for push, oriented in rungs:
            label = f"push {push:g}{' oriented' if oriented else ''}"
            try:
                out, rep = _untangle_once(surfaces, st, prep, push, oriented)
            except MemoryError as error:
                # A push strong enough to collapse triangles makes the broad
                # phase report more pairs than fit: the attempt has diverged.
                rep = Report(message="diverged")
                notes.append(f"{label}: diverged ({error})")
                continue
            notes.append(f"{label}: {rep.message or 'embedded'} ({rep.iterations} iterations)")
            if rep.converged:
                break
            if st.keep_best and rep.remaining < rep.initial and (best is None or rep.remaining < best[1].remaining):
                best = (out, rep)
        else:
            if best is None:
                raise _NotEmbeddedError
            out, rep = best
            rep.message = f"{rep.remaining} of {rep.initial} pair(s) are still inside their margin"
    except RunCanceledError:
        out = [np.array(s.vert, dtype=np.float64) for s in surfaces]
        rep = Report(message="canceled; the input is returned unchanged")
    except _NotEmbeddedError:
        out = [np.array(s.vert, dtype=np.float64) for s in surfaces]
        rep.max_move = rep.mean_move = 0.0
        rep.message = "not embedded; the input is returned unchanged (" + "; ".join(notes) + ")"
    rep.attempts = notes
    rep.sides = sides
    rep.held_pairs = held_pairs
    if held_pairs:
        rep.message = (rep.message + "; " if rep.message else "") + (
            f"{held_pairs} pair(s) of triangles with every vertex held are inside their margin and cannot be moved"
        )
    rep.seconds = time.perf_counter() - t0
    return out, rep


def _has_free(surface: Surface) -> bool:
    return bool(np.any(surface.movable))


def _cut(surfaces: list[Surface], lo: np.ndarray, hi: np.ndarray):
    """The surfaces with every held triangle whose box misses [lo, hi] dropped.

    Returns ``(kept, used)``: ``kept`` is a list of ``(index, surface)`` for the
    surfaces that remain, in order, and ``used`` the number of held triangles
    left. A surface with a movable vertex is kept whole.
    """
    kept, used = [], 0
    for index, s in enumerate(surfaces):
        if _has_free(s):
            kept.append((index, s))
            continue
        vert = np.asarray(s.vert, dtype=np.float64)
        tri = np.asarray(s.tri, dtype=np.int64)
        corners = vert[tri]
        near = np.all(corners.max(axis=1) >= lo, axis=1) & np.all(corners.min(axis=1) <= hi, axis=1)
        if not near.any():
            continue
        vertices = np.unique(tri[near])
        renumber = np.full(len(vert), -1, dtype=np.int64)
        renumber[vertices] = np.arange(len(vertices))
        kept.append((index, Surface(s.name, vert[vertices], renumber[tri[near]], False, s.offset)))
        used += int(near.sum())
    return kept, used


def _untangle_cropped(surfaces: list[Surface], st: Settings):
    """`untangle` of surfaces that are whole pieces: held geometry far from the
    movable surfaces is cut away first (`Settings.crop_min_triangles`).
    """
    t0 = time.perf_counter()
    held = sum(len(s.tri) for s in surfaces if not _has_free(s))
    free = [np.asarray(s.vert, dtype=np.float64) for s in surfaces if _has_free(s)]
    if held < st.crop_min_triangles or not free:
        out, rep = _untangle_surfaces(surfaces, st)
        rep.held_used = rep.held_total = held
        rep.seconds = time.perf_counter() - t0
        return out, rep

    movable = np.concatenate(free)
    lo, hi = movable.min(axis=0), movable.max(axis=0)
    # The widest margin any pair is asked to keep: a held triangle farther from
    # the cloth than this is no concern of the cloth's.
    margin = 2.0 * max(float(s.offset) for s in surfaces) + st.clearance
    reach = st.crop_reach * float(np.linalg.norm(hi - lo)) + margin
    widenings = 0
    while True:
        kept, used = _cut(surfaces, lo - reach, hi + reach)
        out_kept, rep = _untangle_surfaces([s for _, s in kept], st)
        out = [np.array(s.vert, dtype=np.float64) for s in surfaces]
        for (index, s), pos in zip(kept, out_kept, strict=True):
            if _has_free(s):
                out[index] = pos
        final = np.concatenate([out[i] for i, s in enumerate(surfaces) if _has_free(s)])
        outside = max(
            float((lo - reach - (final.min(axis=0) - margin)).max()),
            float(((final.max(axis=0) + margin) - (hi + reach)).max()),
        )
        everything = used == held
        if outside <= 0.0 or everything or not rep.converged:
            break
        reach = max(2.0 * reach, reach + 1.5 * outside)
        widenings += 1
    rep.held_used, rep.held_total, rep.crop_widenings = used, held, widenings
    rep.seconds = time.perf_counter() - t0
    return out, rep



# ---------------------------------------------------------------------------
# Pieces, and scenes solved in pieces


@dataclass
class _Piece:
    """One connected component of a surface, as a surface of its own."""

    parent: int
    ids: np.ndarray
    surface: Surface


def _vertex_components(n: int, tri: np.ndarray) -> np.ndarray:
    """Connected-component label of every vertex of the mesh ``tri``, -1 for a
    vertex no triangle uses."""
    label = np.full(n, -1, dtype=np.int64)
    if not len(tri):
        return label
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    i = np.concatenate([e[:, 0], e[:, 1]])
    j = np.concatenate([e[:, 1], e[:, 0]])
    order = np.argsort(i, kind="stable")
    start = np.searchsorted(i[order], np.arange(n + 1)).tolist()
    neighbors = j[order].tolist()
    found = [-1] * n
    count = 0
    for v in np.unique(tri).tolist():
        if found[v] >= 0:
            continue
        found[v] = count
        stack = [v]
        while stack:
            u = stack.pop()
            for w in neighbors[start[u] : start[u + 1]]:
                if found[w] < 0:
                    found[w] = count
                    stack.append(w)
        count += 1
    return np.array(found, dtype=np.int64)


def _pieces(surfaces: list[Surface]) -> list[_Piece]:
    """The surfaces cut into their connected components.

    A side is voted between two surfaces (`_vote_sides`), and one mesh that holds
    several pieces (a dress's front and back, its straps) has none to vote
    between unless the pieces are told apart. A vertex no triangle uses takes no
    part and keeps its input position.
    """
    out = []
    for index, s in enumerate(surfaces):
        vert = np.ascontiguousarray(s.vert, dtype=np.float64)
        tri = np.ascontiguousarray(s.tri, dtype=np.int64)
        if vert.ndim != 2 or vert.shape[1] != 3 or tri.ndim != 2 or tri.shape[1] != 3:
            raise ValueError(f"{s.name}: expected (n, 3) vertices and (t, 3) triangles")
        if len(tri) and (tri.min() < 0 or tri.max() >= len(vert)):
            raise ValueError(f"{s.name}: a triangle names a vertex that does not exist")
        if not np.all(np.isfinite(vert)):
            raise ValueError(f"{s.name}: a vertex position is not finite")
        free = np.broadcast_to(np.asarray(s.movable, dtype=bool), (len(vert),))
        label = _vertex_components(len(vert), tri)
        count = int(label.max()) + 1
        if count == 0:
            continue
        order = np.argsort(label, kind="stable")
        bounds = np.searchsorted(label[order], np.arange(count + 1))
        tlabel = label[tri[:, 0]]
        torder = np.argsort(tlabel, kind="stable")
        tbounds = np.searchsorted(tlabel[torder], np.arange(count + 1))
        for k in range(count):
            ids = order[bounds[k] : bounds[k + 1]]
            remap = np.full(len(vert), -1, dtype=np.int64)
            remap[ids] = np.arange(len(ids))
            mine = free[ids]
            movable = True if mine.all() else (False if not mine.any() else mine)
            name = s.name if count == 1 else f"{s.name}#{k}"
            piece = Surface(name, vert[ids], remap[tri[torder[tbounds[k] : tbounds[k + 1]]]], movable, s.offset)
            out.append(_Piece(index, ids, piece))
    return out


def _from_pieces(surfaces: list[Surface], pieces: list[_Piece], positions: list[np.ndarray]) -> list[np.ndarray]:
    out = [np.array(s.vert, dtype=np.float64) for s in surfaces]
    for piece, pos in zip(pieces, positions, strict=True):
        out[piece.parent][piece.ids] = pos
    return out


def _link_labels(n: int, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Label of each of ``n`` nodes after linking ``a[k]`` with ``b[k]``, the
    smallest index of the group it ends in."""
    label = np.arange(n)
    if not len(a):
        return label
    key = np.unique(np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b))
    a, b = key // n, key % n
    while True:
        low = np.minimum(label[a], label[b])
        new = label.copy()
        np.minimum.at(new, a, low)
        np.minimum.at(new, b, low)
        new = new[new]
        if np.array_equal(new, label):
            return label
        label = new


def _grow(adj_i: np.ndarray, adj_j: np.ndarray, mask: np.ndarray, rings: int) -> np.ndarray:
    """``mask`` and every vertex within ``rings`` mesh edges of it."""
    mask = mask.copy()
    for _ in range(rings):
        grown = mask.copy()
        grown[adj_j[mask[adj_i]]] = True
        mask = grown
    return mask


def _cell_keys(lo: np.ndarray, hi: np.ndarray, cell: float):
    """``(owner, key)`` for every cell of a grid of size ``cell`` that a box touches."""
    low = np.floor(lo / cell).astype(np.int64)
    span = np.floor(hi / cell).astype(np.int64) - low + 1
    count = span.prod(axis=1)
    owner = np.repeat(np.arange(len(lo)), count)
    first = np.cumsum(count) - count
    k = np.arange(int(count.sum())) - np.repeat(first, count)
    sx, sy = span[owner, 0], span[owner, 1]
    c = low[owner] + np.stack([k % sx, (k // sx) % sy, k // (sx * sy)], axis=1)
    bias = 1 << 20
    return owner, ((c[:, 0] + bias) * (1 << 21) + (c[:, 1] + bias)) * (1 << 21) + (c[:, 2] + bias)


def _near(lo: np.ndarray, hi: np.ndarray, seeds: np.ndarray, cell: float) -> np.ndarray:
    """Indices of the boxes that share a cell, or touch the neighbor of one, with
    a box of ``seeds`` (cells of size ``cell``)."""
    _, mine = _cell_keys(lo[seeds], hi[seeds], cell)
    step = np.array([1 << 42, 1 << 21, 1], dtype=np.int64)
    offsets = np.array([dx * step[0] + dy * step[1] + dz for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)])
    occupied = np.unique((np.unique(mine)[:, None] + offsets[None, :]).ravel())
    owner, keys = _cell_keys(lo, hi, cell)
    return np.unique(owner[np.isin(keys, occupied)])


def _bad_pairs(sc: _Scene, vert: np.ndarray, chunk: int = 20000):
    """The pairs of triangles that are inside their margin, as two index arrays."""
    ia, ib, share, near = _candidate_pairs(sc, vert)
    found_a, found_b = [], []
    for c0 in range(0, len(ia), chunk):
        sl = slice(c0, c0 + chunk)
        a, b = vert[sc.tri[ia[sl]]], vert[sc.tri[ib[sl]]]
        margin = np.where(share[sl], 0.0, sc.tri_off[ia[sl]] + sc.tri_off[ib[sl]] + sc.clearance)
        which = np.arange(len(a))
        for bound in (_screen, _screen_edges):
            keep = bound(a, b, None) + margin > _DEPTH_TOL
            if not keep.all():
                a, b, margin, which = a[keep], b[keep], margin[keep], which[keep]
            if not len(a):
                break
        if not len(a):
            continue
        _, _, _, value = _sat(a, b, margin, False, None, want_sym=True)
        hit = (value > _DEPTH_TOL) & near[sl][which]
        found_a.append(ia[sl][which][hit])
        found_b.append(ib[sl][which][hit])
    if not found_a:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(found_a), np.concatenate(found_b)


def _untangle_local(surfaces: list[Surface], st: Settings):
    """`untangle` of a large scene, a place at a time.

    A scene of tens of thousands of triangles has pairs inside their margin in
    a few places, and solving all of it moves and measures a hundred thousand
    triangles for each evaluation of the few that matter. Here the places are
    found once (`_bad_pairs`), the vertices within `Settings.local_rings` mesh
    rings of them are the only ones that may move, the places are grouped so
    that two that could touch each other are solved together, and each group is
    solved as a scene of its own: its movable vertices, the triangles that touch
    them, and the triangles of everything else near it as held geometry. What
    one group leaves may still be inside a margin against something its scene
    did not hold, or the group may have needed more room, so the whole scene is
    measured again and what is left is solved again with the rings and the reach
    doubled. A group that cannot be embedded stays as it was.
    """
    t0 = time.perf_counter()
    counts = [len(s.vert) for s in surfaces]
    base = np.concatenate([[0], np.cumsum(counts)])
    pos = np.concatenate([np.asarray(s.vert, dtype=np.float64) for s in surfaces])
    first = pos.copy()
    tri = np.concatenate([np.asarray(s.tri, dtype=np.int64) + base[k] for k, s in enumerate(surfaces)])
    tri_piece = np.concatenate([np.full(len(s.tri), k) for k, s in enumerate(surfaces)])
    free = np.concatenate([np.broadcast_to(np.asarray(s.movable, dtype=bool), (len(s.vert),)) for s in surfaces])
    n = len(pos)
    if st.status is None:
        st.status = {}
    canceled = [False]

    def tick(stage: str, iteration: int, pairs: int):
        verdict = st.progress(stage, iteration, pairs) if st.progress is not None else None
        if verdict is False:
            canceled[0] = True
        return verdict

    rep = Report()
    notes = []
    initial = remaining = 0
    rounds = groups_last = 0

    def scene() -> _Scene:
        return _assemble(
            [Surface(s.name, pos[base[k] : base[k + 1]], s.tri, s.movable, s.offset) for k, s in enumerate(surfaces)],
            st.clearance,
        )

    for round_index in range(st.local_rounds + 1):
        sc = scene()
        ia, ib = _bad_pairs(sc, sc.vert)
        remaining = len(ia)
        if round_index == 0:
            initial = remaining
            rep.held_pairs = _count_held_pairs(sc)
        if not remaining or round_index == st.local_rounds or canceled[0]:
            break
        rounds = round_index + 1
        rings = st.local_rings * 2**round_index
        edge = float(sc.edge / sc.scale)
        margin = 2.0 * max(float(s.offset) for s in surfaces) + st.clearance
        reach = max(st.local_reach * 2**round_index * edge, 2.0 * margin)

        seed = np.zeros(n, dtype=bool)
        seed[tri[ia].ravel()] = True
        seed[tri[ib].ravel()] = True
        region = _grow(sc.adj_i, sc.adj_j, seed, rings) & free
        # A place is a set of region triangles that share vertices, that are
        # paired with one another, or that are within a reach of one another.
        rt = np.flatnonzero(region[tri].any(axis=1))
        corners = np.concatenate([tri[rt][:, k] for k in range(3)])
        owners = np.tile(np.arange(len(rt)), 3)
        order = np.argsort(corners, kind="stable")
        corners, owners = corners[order], owners[order]
        same = corners[1:] == corners[:-1]
        label = _link_labels(len(rt), owners[:-1][same], owners[1:][same])
        where = np.full(len(tri), -1, dtype=np.int64)
        where[rt] = np.arange(len(rt))
        pa, pb = where[ia], where[ib]
        both = (pa >= 0) & (pb >= 0)
        corners_lo = pos[tri[rt]].min(axis=1)
        corners_hi = pos[tri[rt]].max(axis=1)
        owner, keys = _cell_keys(corners_lo - 0.5 * reach, corners_hi + 0.5 * reach, reach)
        o = np.argsort(keys, kind="stable")
        keys, owner = keys[o], owner[o]
        shared = keys[1:] == keys[:-1]
        label = _link_labels(
            len(rt),
            np.concatenate([pa[both], owner[:-1][shared], np.arange(len(rt))]),
            np.concatenate([pb[both], owner[1:][shared], label]),
        )
        places = np.unique(label)
        sizes = np.array([(label == g).sum() for g in places])
        places = places[np.argsort(-sizes, kind="stable")]
        groups_last = len(places)

        tri_pos = pos[tri]
        lo, hi = tri_pos.min(axis=1), tri_pos.max(axis=1)
        for number, g in enumerate(places, start=1):
            st.status.update({"round": round_index + 1, "group": number, "groups": len(places)})
            rt_g = rt[label == g]
            mine = np.zeros(n, dtype=bool)
            mine[tri[rt_g].ravel()] = True
            mine &= region
            around = np.union1d(_near(lo, hi, rt_g, reach), rt_g)
            subs, backs = [], []
            for k in np.unique(tri_piece[around]):
                tk = around[tri_piece[around] == k]
                gv = np.unique(tri[tk].ravel())
                remap = np.full(n, -1, dtype=np.int64)
                remap[gv] = np.arange(len(gv))
                m = mine[gv]
                subs.append(Surface(surfaces[k].name, pos[gv], remap[tri[tk]], m if m.any() else False, surfaces[k].offset))
                backs.append(gv)
            place_settings = replace(
                st,
                progress=tick,
                keep_best=True,
                push_boosts=3,
                max_stalls=min(st.max_stalls, 6),
                recover_iter=min(st.recover_iter, st.local_recover_iter),
            )
            out, sub_rep = _untangle_surfaces(subs, place_settings)
            for gv, new in zip(backs, out, strict=True):
                m = mine[gv]
                pos[gv[m]] = new[m]
            rep.iterations += sub_rep.iterations
            rep.recover_iterations += sub_rep.recover_iterations
            rep.levels = max(rep.levels, sub_rep.levels)
            notes.append(
                f"round {round_index + 1} place {number} of {len(places)}: {int(mine.sum())} movable vertices, "
                f"{sum(len(s.tri) for s in subs)} triangles, "
                + (f"embedded ({sub_rep.iterations} iterations)" if sub_rep.converged else f"not embedded: {sub_rep.message}")
            )
            if canceled[0]:
                break
        if canceled[0]:
            break

    rep.initial = initial
    rep.remaining = remaining
    rep.converged = remaining == 0
    rep.rounds = rounds
    rep.groups = groups_last
    rep.attempts = notes
    if canceled[0]:
        rep.message = "canceled; the places solved so far are kept"
    elif remaining:
        rep.message = f"{remaining} pair(s) are still inside their margin after {rounds} round(s) of local solves"
    moved = np.linalg.norm(pos - first, axis=1)[free]
    rep.max_move = float(moved.max()) if len(moved) else 0.0
    rep.mean_move = float(moved.mean()) if len(moved) else 0.0
    rep.seconds = time.perf_counter() - t0
    return [pos[base[k] : base[k + 1]].copy() for k in range(len(surfaces))], rep


def _solve_scene(surfaces: list[Surface], st: Settings):
    """One scene: solved whole, or a place at a time when it is large."""
    if sum(len(s.tri) for s in surfaces) >= st.local_min_triangles:
        return _untangle_local(surfaces, st)
    return _untangle_cropped(surfaces, st)


def _islands(surfaces: list[Surface], st: Settings) -> list[list[int]]:
    """The movable surfaces, grouped by whether they could meet.

    Two surfaces are in one island when their boxes overlap, each grown by
    `Settings.island_reach` of its own diagonal and half the widest margin a pair
    keeps, and islands that share a surface are one. A surface nothing else can
    reach is an island of its own. A surface with no movable vertex belongs to no
    island: it is a collider, and an island is given the ones near it
    (`_untangle_islands`).
    """
    movable = [k for k, s in enumerate(surfaces) if _has_free(s)]
    if len(movable) < 2:
        return [movable] if movable else []
    margin = 2.0 * max(float(s.offset) for s in surfaces) + st.clearance
    lo = np.empty((len(movable), 3))
    hi = np.empty((len(movable), 3))
    for n, k in enumerate(movable):
        vert = np.asarray(surfaces[k].vert, dtype=np.float64)
        a, b = vert.min(axis=0), vert.max(axis=0)
        grow = st.island_reach * float(np.linalg.norm(b - a)) + 0.5 * margin
        lo[n], hi[n] = a - grow, b + grow
    ia, ib = _box_pairs(lo, hi)
    label = _link_labels(len(movable), ia, ib)
    islands: dict[int, list[int]] = {}
    for n, k in enumerate(movable):
        islands.setdefault(int(label[n]), []).append(k)
    return [islands[key] for key in sorted(islands)]


def _untangle_islands(surfaces: list[Surface], islands: list[list[int]], st: Settings):
    """`untangle` of a scene that falls into islands, each solved as a scene of its own.

    The settings of a single scene are the ones that give the surface the least
    motion, and they only work on a scene whose surfaces take part in one
    tangle: one that cannot be embedded anywhere stops the others from being
    embedded, and the Gaussian between surfaces that never meet is wasted. An
    island is solved with the held surfaces near it and nothing else, and what
    it leaves is kept even when it is not embedded. The whole scene is then
    measured, so what the islands could not clear is counted and named.
    """
    t0 = time.perf_counter()
    pos = [np.array(s.vert, dtype=np.float64) for s in surfaces]
    first = [p.copy() for p in pos]
    if st.status is None:
        st.status = {}
    canceled = [False]

    def tick(stage: str, iteration: int, pairs: int):
        verdict = st.progress(stage, iteration, pairs) if st.progress is not None else None
        if verdict is False:
            canceled[0] = True
        return verdict

    def placed(positions) -> list[Surface]:
        return [Surface(s.name, p, s.tri, s.movable, s.offset) for s, p in zip(surfaces, positions, strict=True)]

    def count(positions):
        sc = _assemble(placed(positions), st.clearance)
        return len(_bad_pairs(sc, sc.vert)[0]), sc

    initial, _ = count(pos)
    margin = 2.0 * max(float(s.offset) for s in surfaces) + st.clearance
    held = [k for k, s in enumerate(surfaces) if not _has_free(s)]
    rep = Report()
    notes, sides, failed = [], [], []
    for number, members in enumerate(islands, start=1):
        if canceled[0]:
            break
        st.status.update({"round": 1, "group": number, "groups": len(islands)})
        verts = np.concatenate([pos[k] for k in members])
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        reach = st.island_reach * float(np.linalg.norm(hi - lo)) + margin
        near = [
            k
            for k in held
            if np.all(pos[k].max(axis=0) >= lo - reach) and np.all(pos[k].min(axis=0) <= hi + reach)
        ]
        subset = sorted(members + near)
        scene = [
            Surface(surfaces[k].name, pos[k], surfaces[k].tri, surfaces[k].movable, surfaces[k].offset)
            for k in subset
        ]
        out, sub_rep = _solve_scene(scene, replace(st, progress=tick, keep_best=True))
        for k, new in zip(subset, out, strict=True):
            if k in members:
                pos[k] = new
        rep.iterations += sub_rep.iterations
        rep.recover_iterations += sub_rep.recover_iterations
        rep.levels = max(rep.levels, sub_rep.levels)
        sides += sub_rep.sides
        names = ", ".join(surfaces[k].name for k in members)
        if not sub_rep.converged:
            failed.append(names)
        notes.append(
            f"island {number} of {len(islands)} ({names}): "
            + (f"embedded ({sub_rep.iterations} iterations)" if sub_rep.converged else f"not embedded: {sub_rep.message}")
        )

    remaining, sc = count(pos)
    rep.initial = initial
    rep.remaining = remaining
    rep.converged = remaining == 0
    rep.rounds = 1
    rep.groups = len(islands)
    rep.attempts = notes
    rep.sides = sides
    rep.held_pairs = _count_held_pairs(sc)
    if canceled[0]:
        rep.message = "canceled; the islands solved so far are kept"
    elif remaining:
        rep.message = f"{remaining} pair(s) are still inside their margin" + (
            f"; not embedded: {'; '.join(failed)}" if failed else ""
        )
    if rep.held_pairs:
        rep.message = (rep.message + "; " if rep.message else "") + (
            f"{rep.held_pairs} pair(s) of triangles with every vertex held are inside their margin and cannot be moved"
        )
    free = [np.broadcast_to(np.asarray(s.movable, dtype=bool), (len(s.vert),)) for s in surfaces]
    moved = np.concatenate([np.linalg.norm(p - f, axis=1)[m] for p, f, m in zip(pos, first, free, strict=True)])
    rep.max_move = float(moved.max()) if len(moved) else 0.0
    rep.mean_move = float(moved.mean()) if len(moved) else 0.0
    rep.seconds = time.perf_counter() - t0
    return pos, rep


def untangle(surfaces: list[Surface], settings: Settings | None = None):
    """Deform the movable vertices until no two triangles are within margin.

    Two stages. The first minimizes the Gaussian energy, the contact penalty and
    the shape energy, and ends at the first embedded state; when it stalls in a
    local minimum the Gaussian is strengthened, so a scene gets as much of it as
    it needs. The second pulls the surface back toward its input while keeping
    it embedded, which undoes most of the first stage's motion.

    Where one surface lies mostly on one side of another, a part of it that has
    poked through is taken back to that side (`_vote_sides`), which a strong
    contact push does at once. Where no side is right, the Gaussian frees the
    sheets with a weak push. The attempts are a ladder over those, the first
    that embeds the surface deciding the result (`Settings.mink_push`).

    Each surface is first cut into its connected components, so that the pieces
    of one mesh (a dress's front, back and straps) have sides between them. A
    scene of `Settings.local_min_triangles` triangles or more is then solved a
    place at a time (`_untangle_local`); a smaller one is solved whole, with a
    large collider cut down to the triangles near the cloth
    (`Settings.crop_min_triangles`). The result is checked against the geometry
    given: if the cloth, grown by the largest margin, left the region that was
    kept, a held triangle outside could be within that margin, and the run is
    repeated with a wider region.

    Returns (positions, report): a list of (n, 3) arrays, one per surface, in
    the scene's own units, and a `Report`. A run that embeds nothing returns the
    input unchanged and says why in the report, except a scene solved a place at
    a time, which keeps the places it did embed; it raises only on an unusable
    input.
    """
    st = settings or Settings()
    pieces = _pieces(surfaces)
    flat = [p.surface for p in pieces]
    if not any(_has_free(s) for s in flat):
        raise ValueError("no vertex is movable")
    islands = _islands(flat, st)
    if len(islands) > 1:
        out, rep = _untangle_islands(flat, islands, st)
    else:
        out, rep = _solve_scene(flat, st)
    return _from_pieces(surfaces, pieces, out), rep


# ---------------------------------------------------------------------------
# The run in a child process


class Run:
    """`untangle` in a child process, for a caller that has to stay responsive.

    A process and not a thread because the caller is Blender: a second Python
    thread running beside Blender's interface with this add-on loaded corrupts
    the interpreter within seconds (measured on Blender 5.2: a `tuple()` over a
    generator in a worker thread raises SystemError ``Objects/tupleobject.c``
    inside a few million calls), however little the thread touches. The child
    shares nothing with the caller. It is the interpreter that runs the caller,
    started on this file, and it reads and writes numpy arrays and small text
    files in a directory of its own.

    The caller starts it, polls `finished` and shows `stage`, `iteration` and
    `pairs`, and may `cancel()`, which takes effect at the next iteration, or
    at the end of the setup before the first one. `result()` hands back what
    `untangle` returned, or raises what it raised. `kill()` ends a run that is
    no longer wanted. ``settings`` must carry no `Settings.progress` hook: a
    function cannot cross to the child.
    """

    def __init__(self, surfaces: list[Surface], settings: Settings | None = None):
        st = settings or Settings()
        if st.progress is not None:
            raise ValueError("a run in a child process cannot take a progress hook")
        interpreter = sys.executable
        if not interpreter or os.path.basename(interpreter).lower().startswith("blender"):
            raise RuntimeError(
                f"cannot start the untangler: {interpreter!r} is not a Python interpreter, "
                "so there is nothing to run its child process with"
            )
        self._interpreter = interpreter
        self._dir = tempfile.mkdtemp(prefix="ppf-untangle-")
        self._cancel_path = os.path.join(self._dir, "cancel")
        self._progress_path = os.path.join(self._dir, "progress.json")
        self._result_path = os.path.join(self._dir, "result.npz")
        self._error_path = os.path.join(self._dir, "error.json")
        self._log_path = os.path.join(self._dir, "child.log")
        self._write_job(surfaces, st)
        self._process = None
        self._canceled = False
        self._result = None
        self._shown = ("", 0, 0)
        self._status = {}

    def _write_job(self, surfaces: list[Surface], st: Settings) -> None:
        arrays = {}
        names, offsets = [], []
        for index, s in enumerate(surfaces):
            vert = np.ascontiguousarray(s.vert, dtype=np.float64)
            arrays[f"v{index}"] = vert
            arrays[f"t{index}"] = np.ascontiguousarray(s.tri, dtype=np.int64)
            arrays[f"m{index}"] = np.broadcast_to(np.asarray(s.movable, dtype=bool), (len(vert),)).copy()
            names.append(s.name)
            offsets.append(float(s.offset))
        meta = {"names": names, "offsets": offsets, "settings": asdict(st)}
        np.savez(os.path.join(self._dir, "job.npz"), meta=json.dumps(meta), **arrays)

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError("the run has already started")
        command = [
            self._interpreter,
            "-I",
            os.path.abspath(__file__),
            self._dir,
        ]
        options = {"stdin": subprocess.DEVNULL, "close_fds": True}
        log = open(self._log_path, "wb")  # noqa: SIM115 - held open by the child for its lifetime
        options["stdout"] = options["stderr"] = log
        if os.name == "nt":
            # Below the interface in priority, and no console window. On other
            # systems the child lowers its own priority (`_child`).
            options["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW
        try:
            self._process = subprocess.Popen(command, **options)  # noqa: S603
        finally:
            log.close()

    def _refresh(self) -> None:
        try:
            with open(self._progress_path, encoding="utf-8") as handle:
                data = json.load(handle)
            self._shown = (str(data["stage"]), int(data["iteration"]), int(data["pairs"]))
            self._status = {k: int(v) for k, v in data.get("status", {}).items()}
        except (OSError, ValueError, KeyError):
            pass

    @property
    def stage(self) -> str:
        self._refresh()
        return self._shown[0]

    @property
    def iteration(self) -> int:
        self._refresh()
        return self._shown[1]

    @property
    def pairs(self) -> int:
        self._refresh()
        return self._shown[2]

    @property
    def status(self) -> dict:
        """Where a scene solved a place at a time is: ``round``, ``group`` and
        ``groups``, or empty (`Settings.status`)."""
        self._refresh()
        return dict(self._status)

    @property
    def finished(self) -> bool:
        return self._process is not None and self._process.poll() is not None

    @property
    def canceled(self) -> bool:
        return self._canceled

    def cancel(self) -> None:
        self._canceled = True
        with open(self._cancel_path, "w", encoding="utf-8"):
            pass

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the run finishes or ``timeout`` seconds pass; True when it finished."""
        if self._process is None:
            raise RuntimeError("the run has not started")
        try:
            self._process.wait(timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    def kill(self) -> None:
        """End the run if it is still going, and discard whatever it wrote."""
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        shutil.rmtree(self._dir, ignore_errors=True)

    def result(self):
        """``(positions, report)`` of the finished run; raises what the run raised."""
        if self._result is not None:
            return self._result
        if not self.finished:
            raise RuntimeError("the run has not finished")
        try:
            self._refresh()
            if os.path.exists(self._error_path):
                with open(self._error_path, encoding="utf-8") as handle:
                    error = json.load(handle)
                raise (ValueError if error["type"] == "ValueError" else RuntimeError)(error["message"])
            if self._process.returncode != 0 or not os.path.exists(self._result_path):
                try:
                    with open(self._log_path, encoding="utf-8", errors="replace") as handle:
                        tail = handle.read()[-2000:]
                except OSError:
                    tail = ""
                raise RuntimeError(f"the untangler process exited with status {self._process.returncode}: {tail}")
            with np.load(self._result_path, allow_pickle=False) as data:
                count = sum(1 for key in data.files if key.startswith("p"))
                report = Report(**json.loads(str(data["report"])))
                self._result = ([np.array(data[f"p{index}"]) for index in range(count)], report)
        finally:
            shutil.rmtree(self._dir, ignore_errors=True)
        return self._result


def _child(directory: str) -> int:
    """Run the job in ``directory`` and write its result there (see `Run`)."""
    if os.name != "nt":
        os.nice(10)
    job = np.load(os.path.join(directory, "job.npz"), allow_pickle=False)
    meta = json.loads(str(job["meta"]))
    surfaces = [
        Surface(name, job[f"v{i}"], job[f"t{i}"], job[f"m{i}"], offset)
        for i, (name, offset) in enumerate(zip(meta["names"], meta["offsets"], strict=True))
    ]
    fields = meta["settings"]
    for key, default in asdict(Settings()).items():
        if isinstance(default, tuple):
            fields[key] = tuple(fields[key])
    settings = Settings(**fields)

    cancel_path = os.path.join(directory, "cancel")
    progress_path = os.path.join(directory, "progress.json")
    last = [0.0]

    def hook(stage: str, iteration: int, pairs: int) -> bool:
        now = time.monotonic()
        if now - last[0] >= 0.1:
            last[0] = now
            with open(progress_path + ".tmp", "w", encoding="utf-8") as handle:
                json.dump(
                    {"stage": stage, "iteration": iteration, "pairs": pairs, "status": dict(settings.status or {})},
                    handle,
                )
            os.replace(progress_path + ".tmp", progress_path)
        return not os.path.exists(cancel_path)

    settings.progress = hook
    try:
        out, rep = untangle(surfaces, settings)
    except Exception as error:
        with open(os.path.join(directory, "error.json"), "w", encoding="utf-8") as handle:
            json.dump({"type": type(error).__name__, "message": str(error)}, handle)
        return 2

    def plain(value):
        return value.tolist() if hasattr(value, "tolist") else float(value)

    np.savez(
        os.path.join(directory, "result.npz"),
        report=json.dumps(asdict(rep), default=plain),
        **{f"p{i}": np.asarray(p, dtype=np.float64) for i, p in enumerate(out)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_child(sys.argv[1]))
