# File: addon_host_tests/_shell_untangle_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The pieces of the shell untangler (`core/shell_untangle.py`), each against an
# answer computed another way.
#
#   * The Minkowski depth of a triangle pair against a brute-force
#     edge-through-triangle intersection test.
#   * The analytic gradient of every energy against central differences, with
#     one pair per separating-axis class so that a wrong chain rule in any of
#     the 17 is found.
#   * The broad phase against all pairs.
#   * A small end-to-end run: two sheets through each other come apart, and
#     nothing is closer than its margin afterward.

from __future__ import annotations

import numpy as np
import pytest

from conftest import load_addon_module


@pytest.fixture(scope="module")
def su():
    return load_addon_module("core.shell_untangle")


def _grid(n, size=1.0, z=0.0, origin=(0.0, 0.0)):
    """An n x n sheet in the plane z, as (vert, tri)."""
    xs = np.linspace(0.0, size, n + 1)
    x, y = np.meshgrid(xs + origin[0], xs + origin[1], indexing="ij")
    vert = np.stack([x.ravel(), y.ravel(), np.full(x.size, z)], axis=1)
    idx = np.arange((n + 1) * (n + 1)).reshape(n + 1, n + 1)
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    tri = np.concatenate(
        [np.stack([a.ravel(), b.ravel(), c.ravel()], 1), np.stack([a.ravel(), c.ravel(), d.ravel()], 1)]
    )
    return vert, tri


def _segment_hits_triangle(p, q, tri):
    """Moller-Trumbore on the closed segment pq, for a non-degenerate case."""
    e1, e2 = tri[1] - tri[0], tri[2] - tri[0]
    d = q - p
    h = np.cross(d, e2)
    det = e1 @ h
    if abs(det) < 1e-14:
        return False
    f = 1.0 / det
    s = p - tri[0]
    u = f * (s @ h)
    if u < 0.0 or u > 1.0:
        return False
    qv = np.cross(s, e1)
    v = f * (d @ qv)
    if v < 0.0 or u + v > 1.0:
        return False
    t = f * (e2 @ qv)
    return 0.0 <= t <= 1.0


def _intersect(a, b):
    for i in range(3):
        if _segment_hits_triangle(a[i], a[(i + 1) % 3], b):
            return True
        if _segment_hits_triangle(b[i], b[(i + 1) % 3], a):
            return True
    return False


def _random_pairs(rng, count):
    a = rng.normal(size=(count, 3, 3))
    b = rng.normal(size=(count, 3, 3)) * 0.8 + rng.normal(size=(count, 1, 3)) * 0.3
    return a, b


def test_depth_sign_matches_intersection(su):
    rng = np.random.default_rng(7)
    a, b = _random_pairs(rng, 400)
    value, _, _ = su._sat(a, b, np.zeros(len(a)), False)
    truth = np.array([_intersect(x, y) for x, y in zip(a, b, strict=True)])
    assert truth.any() and (~truth).any()
    assert np.array_equal(value > 0.0, truth)


def test_coplanar_triangles_apart_in_the_plane_do_not_overlap(su):
    a = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], float)
    b = a + np.array([2.0, 0.0, 0.0])
    value, _, _ = su._sat(a, b, np.zeros(1), False)
    assert value[0] < 0.0
    # Within the margin they do count, and the margin is what is added.
    value, _, _ = su._sat(a, b, np.full(1, 1.5), False)
    assert value[0] == pytest.approx(0.5)


def test_depth_of_two_perpendicular_triangles(su):
    # A flat triangle at z = 0 pierced by a vertical one: the shortest way out
    # is sideways along the thin direction of the vertical triangle.
    a = np.array([[[-1, -1, 0], [1, -1, 0], [0, 1, 0]]], float)
    b = np.array([[[0, -0.2, -1], [0, -0.2, 1], [0, 0.4, 0.0]]], float)
    value, _, _ = su._sat(a, b, np.zeros(1), False)
    assert value[0] > 0.0


def _fd(fun, x, h=1e-6):
    g = np.zeros_like(x)
    flat = x.ravel()
    gf = g.ravel()
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + h
        up = fun(x)
        flat[i] = old - h
        dn = fun(x)
        flat[i] = old
        gf[i] = (up - dn) / (2.0 * h)
    return g


def _identity(su, sc, area):
    """Clusters of one vertex each: the Gaussian on the vertices themselves."""
    return su._coarsen(sc, area, 10**9)


def test_mink_gradient_for_every_axis_class(su):
    rng = np.random.default_rng(11)
    seen = set()
    for n in range(20000):
        a = rng.normal(size=(1, 3, 3))
        b = rng.normal(size=(1, 3, 3)) * 0.7
        if n % 2:
            # Near-coplanar triangles close in their plane: the in-plane edge
            # normals are the axes that decide these.
            a[..., 2] *= 1e-3
            b[..., 2] *= 1e-3
        margin = np.full(1, 0.05)
        value, info, edges = su._sat(a, b, margin, True)
        if value[0] <= 0.0:
            continue
        k = int(info[0][0])
        cls = 0 if k == 0 else 1 if k == 1 else 2 if k < 11 else 3 if k < 14 else 4
        if cls in seen:
            continue
        ga, gb = su._sat_gradient(a, b, info, edges)

        def f(x, margin=margin):
            v, _, _ = su._sat(x[:3][None], x[3:][None], margin, False)
            return v[0]

        x = np.concatenate([a[0], b[0]])
        num = _fd(f, x)
        got = np.concatenate([ga[0], gb[0]])
        assert np.allclose(got, num, atol=1e-5), (cls, k, got - num)
        seen.add(cls)
        if len(seen) == 5:
            break
    assert seen == {0, 1, 2, 3, 4}


def test_gauss_gradient(su):
    rng = np.random.default_rng(3)
    vert, tri = _grid(4)
    vert = vert + rng.normal(size=vert.shape) * 0.03
    sc = su._assemble([su.Surface("s", vert, tri)], 0.0)
    area = su._vertex_area(sc.vert, sc.tri)
    co = _identity(su, sc, area)
    eps2 = 0.05 + 0.02 * rng.random(len(sc.vert))
    _, grad, *_ = su._gauss_pass(co, sc.vert, eps2, area)

    def f(x):
        return su._gauss_pass(co, x, eps2, area, want_grad=False)[0]

    assert np.allclose(grad, _fd(f, sc.vert.copy(), 1e-6), rtol=1e-4, atol=1e-8)


def test_cluster_gaussian_gradient_is_exact_in_the_vertices(su):
    # The clusters' energy is a function of the vertices through the centroids,
    # and its gradient shares each cluster's force by the same weights.
    rng = np.random.default_rng(8)
    v0, t0 = _grid(8)
    v1, t1 = _grid(8, z=0.1)
    sc = su._assemble([su.Surface("a", v0, t0), su.Surface("b", v1, t1)], 0.0)
    area = su._vertex_area(sc.vert, sc.tri)
    co = su._coarsen(sc, area, 30)
    assert 20 <= co.count <= 30
    ev = su._Evaluator(sc, co, area)
    ev.eps2[:] = 0.08 + 0.02 * rng.random(co.count)
    vert = sc.vert + rng.normal(size=sc.vert.shape) * 0.01
    _, grad, _ = ev.gauss(vert)
    num = _fd(lambda x: ev.gauss(x, want_grad=False)[0], vert.copy(), 1e-6)
    assert np.allclose(grad, num, rtol=1e-4, atol=1e-9)


def test_clusters_are_connected_over_the_mesh_and_never_span_two_surfaces(su):
    # Two sheets a hair apart: nearest in space is not near over the surface.
    v0, t0 = _grid(10)
    v1, t1 = _grid(10, z=0.002)
    sc = su._assemble([su.Surface("a", v0, t0), su.Surface("b", v1, t1)], 0.0)
    area = su._vertex_area(sc.vert, sc.tri)
    co = su._coarsen(sc, area, 40)
    n0 = len(v0)
    assert not set(co.cluster[:n0].tolist()) & set(co.cluster[n0:].tolist())
    # Every cluster is one piece of mesh: flood fill along mesh edges.
    nbr = {}
    for i, j in zip(sc.adj_i.tolist(), sc.adj_j.tolist(), strict=True):
        nbr.setdefault(i, []).append(j)
    for k in range(co.count):
        members = set(np.flatnonzero(co.cluster == k).tolist())
        seen = {next(iter(members))}
        stack = list(seen)
        while stack:
            for w in nbr.get(stack.pop(), []):
                if w in members and w not in seen:
                    seen.add(w)
                    stack.append(w)
        assert seen == members


def test_cluster_force_between_sheets_tracks_the_vertex_force(su):
    # The clusters are an approximation. What breaks a lock between sheets is
    # the net push of one sheet on the other (the pushes inside a sheet cancel
    # in the sum), so that is what must agree with the exact sum over vertices.
    v0, t0 = _grid(20)
    v1, t1 = _grid(20, z=0.15)
    v0[:, 2] += 0.03 * np.sin(6 * v0[:, 0])
    sc = su._assemble([su.Surface("a", v0, t0), su.Surface("b", v1, t1)], 0.0)
    area = su._vertex_area(sc.vert, sc.tri)
    n0 = len(v0)
    exact = _identity(su, sc, area)
    eps2 = np.full(exact.count, 0.15**2)
    g_exact = su._gauss_pass(exact, sc.vert, eps2, area)[1][n0:].sum(axis=0)
    ev = su._Evaluator(sc, su._coarsen(sc, area, 120), area)
    ev.eps2[:] = 0.15**2
    g_coarse = ev.gauss(sc.vert)[1][n0:].sum(axis=0)
    assert np.linalg.norm(g_coarse - g_exact) <= 0.1 * np.linalg.norm(g_exact)


def test_gauss_ignores_mesh_neighbors(su):
    # One quad cut into two triangles: every pair of its four vertices is
    # joined by an edge except the two corners the diagonal does not join, so
    # the energy is that one pair's term and nothing else.
    vert, tri = _grid(1)
    sc = su._assemble([su.Surface("s", vert, tri)], 0.0)
    area = su._vertex_area(sc.vert, sc.tri)
    eps2 = 0.2
    energy, *_ = su._gauss_pass(_identity(su, sc, area), sc.vert, np.full(4, eps2), area, want_grad=False)
    joined = {frozenset(e) for e in zip(sc.adj_i.tolist(), sc.adj_j.tolist(), strict=True)}
    free = [(i, j) for i in range(4) for j in range(i + 1, 4) if frozenset((i, j)) not in joined]
    assert len(free) == 1
    i, j = free[0]
    r2 = float(np.sum((sc.vert[i] - sc.vert[j]) ** 2))
    assert energy == pytest.approx(area[i] * area[j] / eps2 * np.exp(-r2 / eps2))


def test_hop_table_matches_breadth_first_search(su):
    vert, tri = _grid(6)
    sc = su._assemble([su.Surface("s", vert, tri)], 0.0)
    n = len(vert)
    nbr = [[] for _ in range(n)]
    for i, j in zip(sc.adj_i.tolist(), sc.adj_j.tolist(), strict=True):
        nbr[i].append(j)
    hop_i, hop_j, hop_d = su._hop_table(sc.adj_i, sc.adj_j, n, 4)
    got = {(i, j): d for i, j, d in zip(hop_i.tolist(), hop_j.tolist(), hop_d.tolist(), strict=True)}
    want = {}
    for src in range(n):
        dist = {src: 0}
        frontier = [src]
        for hop in range(1, 5):
            nxt = []
            for u in frontier:
                for v in nbr[u]:
                    if v not in dist:
                        dist[v] = hop
                        nxt.append(v)
            frontier = nxt
        for v, d in dist.items():
            if d > 0:
                want[(src, v)] = d
    assert got == want
    assert list(hop_i) == sorted(hop_i)


def test_bandwidth_does_not_depend_on_how_finely_the_sheet_is_cut(su):
    # The exclusion of near pairs is a fixed fraction of the bandwidth over the
    # surface, so refining a mesh leaves the characteristic scale where it was.
    eps = []
    for n in (16, 24):
        v0, t0 = _grid(n)
        v1, t1 = _grid(n, z=0.05)
        sc = su._assemble([su.Surface("a", v0, t0), su.Surface("b", v1, t1)], 0.0)
        area = su._vertex_area(sc.vert, sc.tri)
        eps.append(np.sqrt(su._global_bandwidth(_identity(su, sc, area), sc.vert, area)))
    assert eps[1] == pytest.approx(eps[0], rel=0.15)


def test_arap_gradient_and_rest(su):
    rng = np.random.default_rng(5)
    vert, tri = _grid(3)
    sc = su._assemble([su.Surface("s", vert, tri)], 0.0)
    energy, _ = su._arap(sc, sc.vert)
    assert energy == pytest.approx(0.0, abs=1e-20)
    moved = sc.vert + rng.normal(size=sc.vert.shape) * 0.02
    _, grad = su._arap(sc, moved)

    def f(x):
        return su._arap(sc, x, want_grad=False)[0]

    assert np.allclose(grad, _fd(f, moved.copy(), 1e-6), rtol=1e-4, atol=1e-9)


def test_arap_is_invariant_to_rigid_motion(su):
    vert, tri = _grid(3)
    sc = su._assemble([su.Surface("s", vert, tri)], 0.0)
    th = 0.7
    rot = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    moved = sc.vert @ rot.T + np.array([0.3, -0.2, 0.5])
    assert su._arap(sc, moved, want_grad=False)[0] == pytest.approx(0.0, abs=1e-20)


def test_broad_phase_matches_all_pairs(su):
    rng = np.random.default_rng(2)
    lo = rng.random((300, 3))
    hi = lo + rng.random((300, 3)) * 0.15
    ia, ib = su._box_pairs(lo, hi)
    got = set(zip(ia.tolist(), ib.tolist(), strict=True))
    want = set()
    for i in range(300):
        for j in range(i + 1, 300):
            if np.all(lo[i] <= hi[j]) and np.all(lo[j] <= hi[i]):
                want.add((i, j))
    assert got == want
    assert len(got) == len(ia)


def test_mink_pass_total_gradient(su):
    rng = np.random.default_rng(13)
    v0, t0 = _grid(4)
    v1, t1 = _grid(4, z=0.01)
    v1 = v1 + rng.normal(size=v1.shape) * 0.02
    surfaces = [su.Surface("a", v0, t0, offset=0.01), su.Surface("b", v1, t1, offset=0.01)]
    sc = su._assemble(surfaces, 0.005)
    total, grad, active, _ = su._mink_pass(sc, sc.vert)
    assert active > 0
    num = _fd(lambda x: su._mink_pass(sc, x, want_grad=False)[0], sc.vert.copy(), 1e-7)
    # The penalty is piecewise linear, so a central difference is exact
    # except where a step crosses a change of the active axis.
    close = np.isclose(grad, num, atol=1e-4)
    assert close.mean() > 0.98


def test_two_crossing_sheets_come_apart(su):
    v0, t0 = _grid(8)
    v1, t1 = _grid(8)
    # The second sheet stands on edge through the middle of the first.
    v1 = np.stack([v1[:, 0], np.full(len(v1), 0.5), v1[:, 1] - 0.5], axis=1)
    surfaces = [su.Surface("flat", v0, t0, offset=0.004), su.Surface("upright", v1, t1, offset=0.004)]
    out, rep = su.untangle(surfaces, su.Settings(clearance=0.004, max_iter=600))
    assert rep.initial > 0
    assert rep.converged, rep.message
    assert rep.remaining == 0
    sc = su._assemble(
        [su.Surface("flat", out[0], t0, offset=0.004), su.Surface("upright", out[1], t1, offset=0.004)], 0.004
    )
    _, _, active, _ = su._mink_pass(sc, sc.vert, want_grad=False)
    assert active == 0


def test_held_vertices_do_not_move(su):
    v0, t0 = _grid(6)
    v1, t1 = _grid(6)
    v1 = np.stack([v1[:, 0], np.full(len(v1), 0.5), v1[:, 1] - 0.5], axis=1)
    surfaces = [
        su.Surface("held", v0, t0, movable=False, offset=0.004),
        su.Surface("free", v1, t1, offset=0.004),
    ]
    out, rep = su.untangle(surfaces, su.Settings(clearance=0.004, max_iter=800))
    assert np.array_equal(out[0], v0)
    assert rep.converged, rep.message


def test_clean_input_is_returned_unchanged(su):
    v0, t0 = _grid(5)
    v1, t1 = _grid(5, z=0.2)
    out, rep = su.untangle([su.Surface("a", v0, t0, offset=0.01), su.Surface("b", v1, t1, offset=0.01)])
    assert rep.converged and rep.iterations == 0
    assert np.allclose(out[0], v0) and np.allclose(out[1], v1)


def test_a_run_that_embeds_nothing_returns_the_input(su):
    v0, t0 = _grid(8)
    v1, t1 = _grid(8)
    v1 = np.stack([v1[:, 0], np.full(len(v1), 0.5), v1[:, 1] - 0.5], axis=1)
    surfaces = [su.Surface("flat", v0, t0, offset=0.004), su.Surface("upright", v1, t1, offset=0.004)]
    out, rep = su.untangle(surfaces, su.Settings(clearance=0.004, max_iter=3))
    assert not rep.converged
    assert "input is returned unchanged" in rep.message
    assert len(rep.attempts) == 2
    assert np.array_equal(out[0], v0) and np.array_equal(out[1], v1)
    assert rep.max_move == 0.0


def _crossing_pair(su):
    v0, t0 = _grid(8)
    v1, t1 = _grid(8)
    v1 = np.stack([v1[:, 0], np.full(len(v1), 0.5), v1[:, 1] - 0.5], axis=1)
    return v0, v1, [su.Surface("flat", v0, t0, offset=0.004), su.Surface("upright", v1, t1, offset=0.004)]


def test_canceling_the_first_stage_returns_the_input(su):
    v0, v1, surfaces = _crossing_pair(su)
    seen = []

    def progress(stage, iteration, pairs):
        seen.append((stage, iteration))
        return iteration < 3

    out, rep = su.untangle(surfaces, su.Settings(clearance=0.004, progress=progress))
    assert not rep.converged and "canceled" in rep.message
    assert seen[-1] == ("untangle", 3)
    assert np.array_equal(out[0], v0) and np.array_equal(out[1], v1)


def test_canceling_the_recovery_keeps_an_embedded_state(su):
    v0, v1, surfaces = _crossing_pair(su)

    def progress(stage, iteration, pairs):
        return stage != "recover" or iteration < 4

    out, rep = su.untangle(surfaces, su.Settings(clearance=0.004, progress=progress))
    assert rep.converged and "recovery canceled" in rep.message
    sc = su._assemble(
        [su.Surface("flat", out[0], surfaces[0].tri, offset=0.004), su.Surface("upright", out[1], surfaces[1].tri, offset=0.004)],
        0.004,
    )
    assert su._mink_pass(sc, sc.vert, want_grad=False)[2] == 0


def _normal(tri):
    c = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    return c / np.linalg.norm(c)


def test_a_preferred_exit_is_never_taken_the_other_way(su):
    rng = np.random.default_rng(21)
    checked = 0
    for _ in range(3000):
        a = rng.normal(size=(1, 3, 3))
        b = rng.normal(size=(1, 3, 3)) * 0.8
        pref = _normal(b[0])[None] * rng.choice([-1.0, 1.0])
        value, info, _ = su._sat(a, b, np.full(1, 0.02), True, pref)
        if value[0] <= 0.0:
            continue
        k, sigma, _, _, unit, _ = info
        # A leaves along -axis when the "plus" overlap is the one used.
        exit_dir = -sigma[0] * unit[0]
        assert exit_dir @ pref[0] >= -su._SIDE_TOL - 1e-12
        checked += 1
    assert checked > 200


def test_oriented_gradient_matches_finite_differences(su):
    rng = np.random.default_rng(23)
    done = 0
    for _ in range(2000):
        a = rng.normal(size=(1, 3, 3))
        b = rng.normal(size=(1, 3, 3)) * 0.7
        pref = _normal(b[0])[None]
        margin = np.full(1, 0.05)
        value, info, edges = su._sat(a, b, margin, True, pref)
        if value[0] <= 0.0:
            continue
        ga, gb = su._sat_gradient(a, b, info, edges)

        def f(x, margin=margin, pref=pref):
            return su._sat(x[:3][None], x[3:][None], margin, False, pref)[0][0]

        num = _fd(f, np.concatenate([a[0], b[0]]))
        assert np.allclose(np.concatenate([ga[0], gb[0]]), num, atol=1e-5)
        done += 1
        if done == 40:
            break
    assert done == 40


def _poked(su, bump=0.08):
    """A sheet 0.04 above another that has a bump through it."""
    v0, t0 = _grid(14)
    v1, t1 = _grid(14, z=0.04)
    r2 = (v0[:, 0] - 0.5) ** 2 + (v0[:, 1] - 0.5) ** 2
    v0[:, 2] += bump * np.exp(-r2 / 0.01)
    return v0, t0, v1, t1, [su.Surface("lower", v0, t0, offset=0.004), su.Surface("upper", v1, t1, offset=0.004)]


def test_the_vote_takes_the_side_most_of_a_surface_lies_on(su):
    *_, surfaces = _poked(su)
    sc = su._assemble(surfaces, 0.002)
    area = su._vertex_area(sc.vert, sc.tri)
    co = su._coarsen(sc, area, 120)
    votes = su._vote_sides(sc, co, area, 0.5)
    assert len(votes) == 1
    x, y, sign, sure = votes[0]
    # The upper sheet (the second) lies on the +z side of the lower sheet's
    # face normals, so the lower sheet is the one to leave along +n of the
    # upper's ... whichever way the grid is wound, the sign is decisive.
    assert (x, y) == (0, 1) and sign != 0.0 and sure > 0.8


def test_the_vote_declines_where_sheets_cross_each_other(su):
    _, _, surfaces = _crossing_pair(su)
    sc = su._assemble(surfaces, 0.002)
    area = su._vertex_area(sc.vert, sc.tri)
    votes = su._vote_sides(sc, su._coarsen(sc, area, 120), area, 0.5)
    assert all(sign == 0.0 for _, _, sign, _ in votes)


def test_a_poke_is_taken_back_by_a_strong_oriented_push(su):
    v0, t0, v1, t1, surfaces = _poked(su)
    out, rep = su.untangle(surfaces, su.Settings(clearance=0.002))
    assert rep.converged and "oriented" in rep.attempts[0] and len(rep.attempts) == 1
    assert rep.iterations < 60
    # Either sheet may give way, and the second stage returns what did not have
    # to move: the bump (0.08 high, through a sheet 0.04 up) is pressed down by
    # most of its excess and the upper sheet is raised by a fraction of it.
    assert np.abs(out[1] - v1).max() < 0.05
    assert out[0][:, 2].max() < 0.06


def test_loose_vertices_and_degenerate_triangles_do_not_break_a_run(su):
    v0, v1, surfaces = _crossing_pair(su)
    flat = surfaces[0]
    # A vertex no triangle uses, a triangle of zero area (three collinear
    # vertices) and a triangle that names one vertex twice.
    extra = np.array([[0.2, 0.2, 0.3], [0.1, 0.1, 0.0], [0.2, 0.2, 0.0], [0.3, 0.3, 0.0]])
    n = len(flat.vert)
    vert = np.concatenate([flat.vert, extra])
    tri = np.concatenate([flat.tri, [[n + 1, n + 2, n + 3], [0, 0, 1]]])
    surfaces[0] = su.Surface("flat", vert, tri, offset=0.004)
    out, rep = su.untangle(surfaces, su.Settings(clearance=0.004))
    assert all(np.all(np.isfinite(o)) for o in out)
    assert rep.converged, rep.message
    assert np.array_equal(out[0][n], extra[0])  # nothing pushes a vertex no triangle uses


def test_a_broad_phase_overflow_is_a_failed_attempt_not_a_crash(su, monkeypatch):
    v0, v1, surfaces = _crossing_pair(su)
    real = su._box_pairs
    calls = []

    def overflow(lo, hi, max_entries=40_000_000):
        calls.append(1)
        # The vote and the first evaluation of an attempt see the input; after
        # that the surface has "collapsed".
        if len(calls) > 3:
            raise MemoryError("the broad phase would test 10**9 candidate pairs")
        return real(lo, hi, max_entries)

    monkeypatch.setattr(su, "_box_pairs", overflow)
    out, rep = su.untangle(surfaces, su.Settings(clearance=0.004))
    assert not rep.converged and "diverged" in rep.message
    assert np.array_equal(out[0], v0) and np.array_equal(out[1], v1)


def test_pairs_between_held_geometry_are_counted_and_reported(su):
    # Two sheets that cross, both held entirely: nothing can move them.
    v0, v1, surfaces = _crossing_pair(su)
    surfaces = [
        su.Surface("flat", surfaces[0].vert, surfaces[0].tri, movable=False, offset=0.004),
        su.Surface("upright", surfaces[1].vert, surfaces[1].tri, movable=np.arange(len(v1)) == 0, offset=0.004),
    ]
    out, rep = su.untangle(surfaces, su.Settings(clearance=0.004, max_iter=5))
    assert rep.held_pairs > 0
    assert "cannot be moved" in rep.message


# ---------------------------------------------------------------------------
# The run in a child process


def test_a_run_in_a_child_process_gives_what_a_call_gives(su):
    v0, v1, surfaces = _crossing_pair(su)
    settings = su.Settings(clearance=0.004)
    expected, _ = su.untangle(surfaces, settings)
    run = su.Run(surfaces, settings)
    assert not run.finished
    run.start()
    assert run.wait(120.0)
    assert run.finished
    out, rep = run.result()
    assert rep.converged, rep.message
    for got, want in zip(out, expected, strict=True):
        assert np.array_equal(got, want)
    # The run reported where it was while it worked.
    assert run.iteration > 0 and run.stage in ("untangle", "recover")
    assert not run.canceled
    # The result is kept, and the child's directory is gone.
    assert run.result()[1] is rep


def test_a_run_canceled_before_it_starts_returns_the_input(su):
    v0, v1, surfaces = _crossing_pair(su)
    run = su.Run(surfaces, su.Settings(clearance=0.004))
    run.cancel()
    run.start()
    assert run.wait(120.0)
    out, rep = run.result()
    assert run.canceled
    assert not rep.converged and "canceled" in rep.message
    assert np.array_equal(out[0], v0) and np.array_equal(out[1], v1)


def test_a_run_canceled_while_it_works_ends_early(su):
    import time

    # A larger pair than the others, so that the run is still going when the
    # first progress shows. It ends either canceled or, if it won the race,
    # converged: what must hold is that the cancel request is honored when
    # there is time, and that the caller gets an answer either way.
    v0, t0 = _grid(20)
    v1, t1 = _grid(20)
    v1 = np.stack([v1[:, 0], np.full(len(v1), 0.5), v1[:, 1] - 0.5], axis=1)
    surfaces = [su.Surface("flat", v0, t0, offset=0.004), su.Surface("upright", v1, t1, offset=0.004)]
    run = su.Run(surfaces, su.Settings(clearance=0.004))
    run.start()
    deadline = time.time() + 120.0
    while run.iteration == 0 and not run.finished and time.time() < deadline:
        time.sleep(0.02)
    run.cancel()
    assert run.wait(120.0)
    out, rep = run.result()
    assert rep.converged or "canceled" in rep.message
    assert len(out) == 2


def test_an_error_in_the_child_is_raised_by_result(su):
    v0, t0 = _grid(4)
    v0[3, 0] = np.nan
    run = su.Run([su.Surface("bad", v0, t0)])
    with pytest.raises(RuntimeError, match="not finished"):
        run.result()
    run.start()
    with pytest.raises(RuntimeError, match="already started"):
        run.start()
    assert run.wait(60.0)
    with pytest.raises(ValueError, match="not finite"):
        run.result()


def test_a_run_refuses_what_cannot_cross_to_a_child(su, monkeypatch):
    _, _, surfaces = _crossing_pair(su)
    with pytest.raises(ValueError, match="progress hook"):
        su.Run(surfaces, su.Settings(progress=lambda stage, it, pairs: True))
    # Blender's own executable is no interpreter to start the child with.
    monkeypatch.setattr("sys.executable", "/opt/blender/blender")
    with pytest.raises(RuntimeError, match="not a Python interpreter"):
        su.Run(surfaces)


def test_killing_a_run_discards_it(su):
    import os

    _, _, surfaces = _crossing_pair(su)
    run = su.Run(surfaces, su.Settings(clearance=0.004))
    directory = run._dir
    run.start()
    run.kill()
    assert not os.path.exists(directory)
    assert run.finished


# ---------------------------------------------------------------------------
# Held geometry far from the cloth


def _with_a_far_collider(su, **settings):
    """Two crossing sheets and a finely cut held floor far below them."""
    v0, v1, surfaces = _crossing_pair(su)
    fv, ft = _grid(40, size=6.0, z=-5.0, origin=(-2.5, -2.5))
    floor = su.Surface("floor", fv, ft, movable=False, offset=0.004)
    scene = [*surfaces, floor]
    return scene, su.untangle(scene, su.Settings(clearance=0.004, crop_min_triangles=500, **settings))


def _embedded_against_everything(su, scene, out):
    placed = [su.Surface(s.name, p, s.tri, s.movable, s.offset) for s, p in zip(scene, out, strict=True)]
    sc = su._assemble(placed, 0.004)
    return su._mink_pass(sc, sc.vert, want_grad=False)[2] == 0


def test_a_collider_far_from_the_cloth_is_left_out_of_the_run(su):
    scene, (out, rep) = _with_a_far_collider(su)
    assert rep.converged, rep.message
    assert rep.held_total == len(scene[2].tri) and rep.held_used == 0
    assert rep.crop_widenings == 0
    assert np.array_equal(out[2], scene[2].vert)
    assert _embedded_against_everything(su, scene, out)


def test_a_result_that_leaves_the_region_is_run_again_with_a_wider_one(su):
    # No reach beyond the margin: the sheets, once apart, are outside the box
    # they started in, so the first result cannot be trusted against the floor.
    scene, (out, rep) = _with_a_far_collider(su, crop_reach=0.0)
    assert rep.converged, rep.message
    assert rep.crop_widenings >= 1
    assert _embedded_against_everything(su, scene, out)


def test_a_cut_is_not_made_below_the_triangle_count(su):
    v0, v1, surfaces = _crossing_pair(su)
    fv, ft = _grid(4, size=6.0, z=-5.0, origin=(-2.5, -2.5))
    scene = [*surfaces, su.Surface("floor", fv, ft, movable=False, offset=0.004)]
    out, rep = su.untangle(scene, su.Settings(clearance=0.004))
    assert rep.held_used == rep.held_total == len(ft)


# ---------------------------------------------------------------------------
# The shape a run leaves


def _corner_turn(s, p):
    """Largest angle, in degrees, a triangle at a corner of ``s`` was turned by."""

    def normal(v, t):
        c = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
        return c / np.linalg.norm(c, axis=1, keepdims=True)

    turn = np.degrees(np.arccos(np.clip((normal(s.vert, s.tri) * normal(p, s.tri)).sum(1), -1.0, 1.0)))
    edges = np.sort(np.concatenate([s.tri[:, [0, 1]], s.tri[:, [1, 2]], s.tri[:, [2, 0]]]), axis=1)
    unique, count = np.unique(edges, axis=0, return_counts=True)
    boundary = np.unique(unique[count == 1].ravel())
    corners = boundary[np.bincount(s.tri.ravel(), minlength=len(s.vert))[boundary] <= 3]
    return float(turn[np.isin(s.tri, corners).any(axis=1)].max())


def test_the_corners_of_a_sheet_are_not_left_curled_up(su):
    # Two wavy sheets laid through each other. A corner triangle is held by one
    # vertex-spring and no neighbor's shape energy, so a run that pulls the
    # sheets back toward their input with springs weighted by vertex area left
    # the corners turned by 14 degrees (the middle of the sheets by 1).
    def sheet(n, phase, lift, fx, fy):
        us = np.linspace(0.0, 1.0, n + 1)
        vert = np.array([[u, w, lift + 0.05 * np.sin(fx * u + phase) * np.cos(fy * w)] for u in us for w in us])
        idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
        a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
        tri = np.concatenate(
            [np.stack([a.ravel(), b.ravel(), c.ravel()], 1), np.stack([a.ravel(), c.ravel(), d.ravel()], 1)]
        )
        return vert, tri

    (va, ta), (vb, tb) = sheet(16, 0.0, 0.0, 5.0, 4.0), sheet(16, 0.6, 0.01, 4.0, 5.0)
    surfaces = [su.Surface("a", va, ta, offset=0.002), su.Surface("b", vb, tb, offset=0.002)]
    out, rep = su.untangle(surfaces, su.Settings(clearance=0.001))
    assert rep.converged, rep.message
    assert max(_corner_turn(s, p) for s, p in zip(surfaces, out, strict=True)) < 8.0


# ---------------------------------------------------------------------------
# Pieces, and scenes solved a place at a time


def test_a_surface_is_cut_into_its_connected_components(su):
    va, ta = _grid(3)
    vb, tb = _grid(2, z=1.0)
    loose = np.array([[9.0, 9.0, 9.0]])
    vert = np.concatenate([va, vb, loose])
    tri = np.concatenate([ta, tb + len(va)])
    movable = np.ones(len(vert), bool)
    movable[0] = False
    surface = su.Surface("mesh", vert, tri, movable, 0.01)
    pieces = su._pieces([surface])
    assert [p.surface.name for p in pieces] == ["mesh#0", "mesh#1"]
    assert [len(p.ids) for p in pieces] == [len(va), len(vb)]
    assert not np.broadcast_to(pieces[0].surface.movable, (len(va),))[0]
    assert pieces[1].surface.movable is True and pieces[0].surface.offset == 0.01
    # Put back, the pieces are the surface, and a vertex no triangle uses is left alone.
    moved = [p.surface.vert + 1.0 for p in pieces]
    back = su._from_pieces([surface], pieces, moved)[0]
    assert np.array_equal(back[: len(va) + len(vb)], vert[: len(va) + len(vb)] + 1.0)
    assert np.array_equal(back[-1], loose[0])


def test_a_single_component_keeps_its_name(su):
    v, t = _grid(3)
    (piece,) = su._pieces([su.Surface("sheet", v, t)])
    assert piece.surface.name == "sheet" and np.array_equal(piece.ids, np.arange(len(v)))


def test_two_pieces_of_one_mesh_have_a_side_between_them(su):
    # A poke through a second sheet, both in ONE mesh: it can only be voted a side
    # because the pieces are told apart.
    v0, t0 = _grid(8)
    v1, t1 = _grid(8, z=0.2)
    v1[40, 2] = -0.1  # one vertex of the upper sheet poked through the lower one
    one = su.Surface("both", np.concatenate([v0, v1]), np.concatenate([t0, t1 + len(v0)]), offset=0.004)
    flat = [p.surface for p in su._pieces([one])]
    prep = su._prepare(flat, su.Settings(clearance=0.004))
    assert prep.votes and prep.votes[0][2] != 0.0
    out, rep = su.untangle([one], su.Settings(clearance=0.004))
    assert rep.converged, rep.message


def test_links_between_labels_merge_into_groups(su):
    label = su._link_labels(7, np.array([0, 2, 5]), np.array([1, 1, 6]))
    assert label.tolist() == [0, 0, 0, 3, 4, 5, 5]
    assert su._link_labels(3, np.zeros(0, int), np.zeros(0, int)).tolist() == [0, 1, 2]


def test_rings_grow_a_set_of_vertices_along_the_mesh(su):
    v, t = _grid(6)
    sc = su._assemble([su.Surface("s", v, t)], 0.0)
    seed = np.zeros(len(v), bool)
    seed[0] = True
    for rings in (0, 1, 2):
        grown = su._grow(sc.adj_i, sc.adj_j, seed, rings)
        # On a grid cut as the test cuts it, a ring is at least a step along an edge.
        assert grown.sum() >= (1, 3, 6)[rings] and grown[0]
        assert not grown[-1]


def test_near_finds_the_boxes_close_to_a_seed_and_not_the_far_ones(su):
    lo = np.array([[0.0, 0, 0], [0.9, 0, 0], [5.0, 5, 5], [0.0, 0, 3.0]])
    hi = lo + 0.5
    near = su._near(lo, hi, np.array([0]), cell=1.0)
    assert 0 in near and 1 in near and 2 not in near and 3 not in near


def test_connected_components_label_a_mesh_by_piece(su):
    tri = np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6]])
    label = su._vertex_components(8, tri)
    assert label.tolist() == [0, 0, 0, 0, 1, 1, 1, -1]


def _long_strip_with_a_poke(su):
    """A long sheet with a corner of a second sheet through it at one end."""
    v0, t0 = _grid(30)
    v0[:, 0] *= 6.0  # 6 long, 1 wide
    v1, t1 = _grid(4, size=0.5, z=0.003, origin=(0.1, 0.25))
    v1[12, 2] = -0.08  # one vertex through the long sheet
    return [su.Surface("strip", v0, t0, offset=0.004), su.Surface("cap", v1, t1, offset=0.004)], v0.copy()


def test_a_scene_solved_a_place_at_a_time_leaves_what_is_far_from_the_trouble_alone(su):
    scene, strip = _long_strip_with_a_poke(su)
    out, rep = su.untangle(scene, su.Settings(clearance=0.004, local_min_triangles=1))
    assert rep.converged, rep.message
    assert rep.rounds >= 1 and rep.groups >= 1 and rep.initial > 0 and rep.remaining == 0
    # The far end of the strip is more than the rings and the reach away: bit for bit as it was.
    far = strip[:, 0] > 3.0
    assert far.sum() > 100 and np.array_equal(out[0][far], strip[far])
    assert np.any(out[0] != strip)
    placed = su._assemble([su.Surface(s.name, o, s.tri, s.movable, s.offset) for s, o in zip(scene, out, strict=True)], 0.004)
    assert su._bad_pairs(placed, placed.vert)[0].size == 0


def test_a_scene_solved_whole_and_a_place_at_a_time_both_embed(su):
    scene, _ = _long_strip_with_a_poke(su)
    whole, rep = su.untangle(scene, su.Settings(clearance=0.004))
    assert rep.converged and rep.rounds == 0


def test_a_place_that_is_not_embedded_keeps_the_pairs_it_did_clear(su):
    v0, v1, surfaces = _crossing_pair(su)
    kept = su.untangle(surfaces, su.Settings(clearance=0.004, max_iter=4, keep_best=True))
    nothing = su.untangle(surfaces, su.Settings(clearance=0.004, max_iter=4))
    assert not nothing[1].converged and np.array_equal(nothing[0][0], v0)
    out, rep = kept
    if not rep.converged:
        assert rep.remaining < rep.initial and "still inside their margin" in rep.message
    assert all(np.all(np.isfinite(o)) for o in out)


def test_the_places_of_a_local_solve_are_reported_while_it_runs(su):
    scene, _ = _long_strip_with_a_poke(su)
    seen = set()
    settings = su.Settings(clearance=0.004, local_min_triangles=1)
    settings.progress = lambda stage, iteration, pairs: seen.add(tuple(sorted(settings.status.items()))) or True
    su.untangle(scene, settings)
    assert any(dict(entry).get("groups") for entry in seen)


def test_a_run_in_a_child_process_reports_its_place(su):
    scene, strip = _long_strip_with_a_poke(su)
    run = su.Run(scene, su.Settings(clearance=0.004, local_min_triangles=1))
    run.start()
    assert run.wait(300.0)
    out, rep = run.result()
    assert rep.converged and rep.rounds >= 1
    assert run.status.get("groups", 0) >= 1 and run.status.get("round") == 1
    assert np.array_equal(out[0][strip[:, 0] > 3.0], strip[strip[:, 0] > 3.0])


# ---------------------------------------------------------------------------
# Islands: surfaces that cannot meet are solved as scenes of their own


def _crossing_pair_at(su, dx, prefix, n=8):
    """Two sheets, one standing through the other, moved `dx` along x."""
    v0, t0 = _grid(n)
    v1, t1 = _grid(n)
    v1 = np.stack([v1[:, 0], np.full(len(v1), 0.5), v1[:, 1] - 0.5], axis=1)
    shift = np.array([dx, 0.0, 0.0])
    return [
        su.Surface(prefix + "flat", v0 + shift, t0, offset=0.004),
        su.Surface(prefix + "upright", v1 + shift, t1, offset=0.004),
    ]


def test_surfaces_that_cannot_meet_are_islands_of_their_own(su):
    st = su.Settings(clearance=0.004)
    near = _crossing_pair_at(su, 0.0, "a_") + _crossing_pair_at(su, 0.3, "b_")
    far = _crossing_pair_at(su, 0.0, "a_") + _crossing_pair_at(su, 20.0, "b_")
    assert su._islands(near, st) == [[0, 1, 2, 3]]
    assert su._islands(far, st) == [[0, 1], [2, 3]]
    assert su._islands(far[:2], st) == [[0, 1]]
    assert su._islands(far[:1], st) == [[0]]


def test_a_collider_belongs_to_no_island_and_is_given_to_the_ones_near_it(su):
    scene = _crossing_pair_at(su, 0.0, "a_") + _crossing_pair_at(su, 20.0, "b_")
    fv, ft = _grid(6, size=3.0, z=-0.5, origin=(-1.0, -1.0))
    floor = su.Surface("floor", fv, ft, movable=False, offset=0.004)
    assert su._islands(scene + [floor], su.Settings(clearance=0.004)) == [[0, 1], [2, 3]]
    assert su._islands([floor], su.Settings(clearance=0.004)) == []


def test_independent_tangles_are_each_solved_as_they_are_alone(su):
    settings = {"clearance": 0.004, "max_iter": 600}
    alone, alone_rep = su.untangle(_crossing_pair_at(su, 0.0, "a_", n=5), su.Settings(**settings))
    assert alone_rep.converged, alone_rep.message
    scene = _crossing_pair_at(su, 0.0, "a_", n=5) + _crossing_pair_at(su, 20.0, "b_", n=5)
    out, rep = su.untangle(scene, su.Settings(**settings))
    assert rep.converged, rep.message
    assert rep.groups == 2 and rep.initial > alone_rep.initial
    assert _embedded_against_everything(su, scene, out)
    # An island is the scene it would be alone, so neither pair is moved more than
    # the one that was solved by itself (with room for the order of the vertices).
    solo = max(float(np.abs(o - s.vert).max()) for o, s in zip(alone, scene[:2], strict=True))
    for o, s in zip(out, scene, strict=True):
        assert float(np.abs(o - s.vert).max()) <= 1.5 * solo + 1e-9


def test_an_island_that_cannot_be_embedded_does_not_hold_back_the_others(su, monkeypatch):
    real = su._untangle_cropped

    def refusing(surfaces, st):
        # A stand-in for a tangle no attempt embeds: it returns what it was given.
        if any(s.name.startswith("b_") for s in surfaces):
            rep = su.Report(initial=7, remaining=7, message="not embedded; stand-in")
            return [np.array(s.vert, dtype=np.float64) for s in surfaces], rep
        return real(surfaces, st)

    monkeypatch.setattr(su, "_untangle_cropped", refusing)
    scene = _crossing_pair_at(su, 0.0, "a_", n=5) + _crossing_pair_at(su, 20.0, "b_", n=5)
    out, rep = su.untangle(scene, su.Settings(clearance=0.004, max_iter=600))
    assert not rep.converged and rep.remaining > 0
    assert "b_flat, b_upright" in rep.message and "a_" not in rep.message
    # The pair that could be solved is, and the other is as it was given.
    assert float(np.abs(out[0] - scene[0].vert).max()) > 0.0
    assert _embedded_against_everything(su, scene[:2], out[:2])
    assert all(np.array_equal(o, s.vert) for o, s in zip(out[2:], scene[2:], strict=True))
