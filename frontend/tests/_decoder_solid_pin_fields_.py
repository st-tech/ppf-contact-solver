# File: frontend/tests/_decoder_solid_pin_fields_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The pin field of a PARTIALLY pinned SOLID is built strictly.
#
# ``_build_solid_pin_fields`` diffuses the pinned Blender vertices into a
# per-tet-vertex pull weight. When it returns nothing, the caller pins a
# different set of tet vertices (the surface-only path), so every way it can
# return nothing is a different simulation of the same scene. These tests
# pin down which ways exist:
#
#   * SciPy missing raises, naming SciPy and how to install it;
#   * a solve that fails, an interior the surface cannot reach and a pin
#     index outside the Blender surface raise, naming the object;
#   * None comes back only for the two documented geometric cases: a tet
#     surface piece no Blender vertex maps onto, and a field that reaches the
#     weight floor at no tet vertex. The caller then takes the surface-only
#     path, and refuses by name when that path reaches no vertex either.

from __future__ import annotations

import sys
import types

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy.sparse")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from frontend import _decoder_ as decoder
    from frontend._decoder_ import (
        _PIN_WEIGHT_EPS,
        SceneDecoder,
        _build_solid_pin_fields,
        _build_solid_weight_transfer,
        _harmonic_interior_operator_strict,
    )
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f"frontend / _ppf_cts_py not importable in this environment: {exc}",
        allow_module_level=True,
    )


def freudenthal_grid(n: int, offset=(0.0, 0.0, 0.0)):
    """A unit cube as an ``n x n x n`` grid of cubes, six tets per cube."""
    axis = np.linspace(0.0, 1.0, n + 1)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    verts = grid.reshape(-1, 3) + np.asarray(offset, dtype=np.float64)

    def vid(i, j, k):
        return (i * (n + 1) + j) * (n + 1) + k

    tets = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c = [
                    vid(i, j, k), vid(i + 1, j, k), vid(i + 1, j + 1, k),
                    vid(i, j + 1, k), vid(i, j, k + 1), vid(i + 1, j, k + 1),
                    vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1),
                ]
                for a, b, d, e in (
                    (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6),
                    (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6),
                ):
                    tets.append([c[a], c[b], c[d], c[e]])
    return verts, np.asarray(tets, dtype=np.int64)


def boundary_faces(tets):
    """The triangles belonging to exactly one tetrahedron."""
    faces = {}
    for tet in tets:
        for tri in (
            (tet[0], tet[1], tet[2]), (tet[0], tet[1], tet[3]),
            (tet[0], tet[2], tet[3]), (tet[1], tet[2], tet[3]),
        ):
            key = tuple(sorted(tri))
            faces[key] = faces.get(key, 0) + 1
    return np.asarray(
        [list(k) for k, count in faces.items() if count == 1], dtype=np.int64
    )


class TetMesh(tuple):
    """The ``(V, F, T)`` triple the decoder receives, with the record of the
    Blender surface the tetrahedra were built from."""

    def has_surface_mapping(self):
        return True


def surface_of(verts, tets):
    faces = boundary_faces(tets)
    surf_ids = np.unique(faces.reshape(-1))
    remap = np.full(verts.shape[0], -1, dtype=np.int64)
    remap[surf_ids] = np.arange(surf_ids.size)
    return faces, surf_ids, remap


def cube(n=3, copies=1):
    """A tet cube whose Blender surface is its own boundary.

    ``copies`` repeats every Blender vertex in place, and the Blender
    triangles name the first copy of each. Coincident copies all map onto the
    same tet vertex, so a Blender mesh far finer than the tets around a pin
    can be built without a large grid.
    """
    verts, tets = freudenthal_grid(n)
    faces, surf_ids, remap = surface_of(verts, tets)
    bl_verts = np.repeat(verts[surf_ids], copies, axis=0)
    bl_tris = remap[faces] * copies
    mesh = TetMesh((verts, faces, tets))
    mesh._pin_blender_surface = (bl_verts, bl_tris)
    return mesh, faces


def low_x_pins(mesh, bound=0.2):
    bl_verts = mesh._pin_blender_surface[0]
    return [int(i) for i in np.flatnonzero(bl_verts[:, 0] < bound)]


# --------------------------------------------------------------------------
# The field itself
# --------------------------------------------------------------------------


def test_a_partial_pin_builds_a_weight_field_through_the_interior():
    mesh, faces = cube()
    fields = _build_solid_pin_fields(mesh, faces, low_x_pins(mesh), "Cube")
    assert fields is not None
    w_surf = fields["w_surf"]
    interior_w = fields["interior_w"]
    assert fields["interior_ids"], "a 3x3x3 grid has interior vertices"
    assert interior_w is not None
    assert len(interior_w) == len(fields["interior_ids"])
    full = np.concatenate([w_surf, interior_w])
    assert np.all(np.isfinite(full))
    assert full.min() >= 0.0 and full.max() <= 1.0
    # A partial pin grades from held to free rather than switching.
    assert full.max() > 0.5
    assert np.any(full < 0.1)
    assert np.any((full > 0.1) & (full < 0.9))


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module", ["scipy.sparse.csgraph", "scipy.sparse.linalg"]
)
def test_missing_scipy_is_refused_naming_it(monkeypatch, module):
    mesh, faces = cube(n=2)
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(RuntimeError) as info:
        _build_solid_pin_fields(mesh, faces, low_x_pins(mesh), "Cube")
    message = str(info.value)
    assert "SciPy is required" in message
    assert "Cube" in message
    assert "warmup.py" in message and "pip install scipy" in message


def test_missing_scipy_is_refused_by_the_interior_and_material_maps(
    monkeypatch,
):
    mesh, faces = cube(n=2)
    verts, _faces, tets = mesh
    monkeypatch.setitem(sys.modules, "scipy.sparse.linalg", None)
    with pytest.raises(RuntimeError, match="SciPy is required"):
        _harmonic_interior_operator_strict(
            verts.shape[0], tets, np.unique(faces), [13],
        )
    with pytest.raises(RuntimeError, match="SciPy is required.*'Cube'"):
        _build_solid_weight_transfer(mesh, faces, "Cube")


def test_a_failed_surface_solve_is_refused_naming_the_object(monkeypatch):
    mesh, faces = cube(n=2)

    class Singular:
        def __init__(self, matrix, rhs_map):
            raise RuntimeError("Factor is exactly singular")

    monkeypatch.setattr(decoder, "_SparseLinearMap", Singular)
    with pytest.raises(RuntimeError) as info:
        _build_solid_pin_fields(mesh, faces, low_x_pins(mesh), "Cube")
    message = str(info.value)
    assert "SOLID 'Cube'" in message
    assert "could not be solved" in message
    assert "exactly singular" in message


def test_a_non_finite_surface_field_is_refused_naming_the_object(
    monkeypatch,
):
    mesh, faces = cube(n=2)
    real = decoder._SparseLinearMap

    class Poisoned(real):
        def apply(self, values):
            raise ValueError("sparse pin diffusion produced non-finite values")

    monkeypatch.setattr(decoder, "_SparseLinearMap", Poisoned)
    with pytest.raises(RuntimeError, match="SOLID 'Cube'.*non-finite"):
        _build_solid_pin_fields(mesh, faces, low_x_pins(mesh), "Cube")


def test_an_interior_the_surface_cannot_reach_is_refused_naming_it():
    # One extra vertex that no tetrahedron references is an interior vertex
    # with no edge to the surface, so no boundary value determines it.
    mesh, faces = cube()
    verts, _faces, tets = mesh
    stray = np.vstack([verts, [[0.5, 0.5, 0.5]]])
    lonely = TetMesh((stray, faces, tets))
    lonely._pin_blender_surface = mesh._pin_blender_surface
    with pytest.raises(RuntimeError) as info:
        _build_solid_pin_fields(lonely, faces, low_x_pins(mesh), "Cube")
    message = str(info.value)
    assert "SOLID 'Cube'" in message
    assert "no edge path" in message


def test_a_pin_outside_the_blender_surface_is_refused_naming_the_object():
    mesh, faces = cube(n=2)
    n_input = mesh._pin_blender_surface[0].shape[0]
    with pytest.raises(RuntimeError) as info:
        _build_solid_pin_fields(mesh, faces, [0, n_input], "Cube")
    message = str(info.value)
    assert "SOLID 'Cube'" in message
    assert f"vertex {n_input}" in message


def test_a_tet_mesh_with_no_blender_surface_record_is_refused():
    verts, tets = freudenthal_grid(2)
    faces = boundary_faces(tets)
    with pytest.raises(RuntimeError, match="SOLID 'Cube'"):
        _build_solid_pin_fields(TetMesh((verts, faces, tets)), faces, [0],
                                "Cube")


# --------------------------------------------------------------------------
# The two documented None cases
# --------------------------------------------------------------------------


def test_a_surface_piece_no_blender_vertex_reaches_returns_none(capsys):
    # Two disjoint cubes of tets, and a Blender surface that is only the
    # first cube's: the second cube's surface receives no Blender vertex.
    va, ta = freudenthal_grid(2)
    vb, tb = freudenthal_grid(2, offset=(3.0, 0.0, 0.0))
    verts = np.vstack([va, vb])
    tets = np.vstack([ta, tb + va.shape[0]])
    faces = boundary_faces(tets)
    fa, surf_a, remap_a = surface_of(va, ta)
    mesh = TetMesh((verts, faces, tets))
    mesh._pin_blender_surface = (va[surf_a], remap_a[fa])
    fields = _build_solid_pin_fields(mesh, faces, [0, 1], "Cube")
    assert fields is None
    notice = capsys.readouterr().err
    assert "SOLID 'Cube'" in notice
    assert "no Blender vertex mapped onto them" in notice


def test_a_field_below_the_weight_floor_returns_none(capsys):
    # 20000 coincident Blender vertices per tet vertex and one of them
    # pinned: the diffused weight is about 1 / 20000 everywhere, below the
    # floor a tet vertex needs to be driven.
    copies = 20000
    assert 1.0 / copies < _PIN_WEIGHT_EPS
    mesh, faces = cube(n=1, copies=copies)
    fields = _build_solid_pin_fields(mesh, faces, [0], "Cube")
    assert fields is None
    notice = capsys.readouterr().err
    assert "SOLID 'Cube'" in notice
    assert "below the" in notice


# --------------------------------------------------------------------------
# The caller: which path each outcome takes
# --------------------------------------------------------------------------


class _Holder:
    def __init__(self, index):
        self.index = list(index)
        self._data = types.SimpleNamespace()


class _Object:
    def __init__(self):
        self.holders = []

    def pin(self, index):
        holder = _Holder(index)
        self.holders.append(holder)
        return holder


def _map_pins(mesh, faces, pin_index):
    verts = mesh[0]
    target = _Object()
    SceneDecoder._apply_pin_mapping(
        None, {"pin": list(pin_index)}, target, "Cube", "SOLID",
        mesh._pin_blender_surface[0], verts, faces, mesh, False,
    )
    return target.holders


def test_the_caller_drives_a_partial_pin_by_the_weight_field():
    mesh, faces = cube()
    holders = _map_pins(mesh, faces, low_x_pins(mesh))
    assert len(holders) == 1
    weights = holders[0]._data._solid_pin_weights
    assert len(weights) == len(holders[0].index)
    assert float(np.min(weights)) < float(np.max(weights))


def test_the_caller_takes_the_surface_only_path_below_the_floor():
    mesh, faces = cube(n=1, copies=20000)
    holders = _map_pins(mesh, faces, [0])
    assert len(holders) == 1
    data = holders[0]._data
    assert not hasattr(data, "_solid_pin_weights")
    assert data._sim_blender_weights
    assert all(
        b == 0 for pairs in data._sim_blender_weights for b, _w in pairs
    )


def test_a_full_pin_whose_interior_cannot_be_reached_is_refused():
    # A full pin drives the interior by the harmonic extension of the whole
    # surface, which an unreferenced interior vertex makes singular.
    mesh, faces = cube()
    verts, _faces, tets = mesh
    lonely = TetMesh((np.vstack([verts, [[0.5, 0.5, 0.5]]]), faces, tets))
    lonely._pin_blender_surface = mesh._pin_blender_surface
    n_input = mesh._pin_blender_surface[0].shape[0]
    with pytest.raises(RuntimeError) as info:
        _map_pins(lonely, faces, range(n_input))
    message = str(info.value)
    assert "SOLID 'Cube'" in message
    assert "no edge path" in message


def test_a_full_pin_drives_the_interior_harmonically():
    mesh, faces = cube()
    n_input = mesh._pin_blender_surface[0].shape[0]
    holders = _map_pins(mesh, faces, range(n_input))
    assert len(holders) == 1
    n_surface, harmonic = holders[0]._data._harmonic
    assert len(holders[0].index) == mesh[0].shape[0]
    assert n_surface == len(np.unique(faces))
    assert harmonic is not None


def test_the_caller_refuses_a_pin_the_surface_only_path_cannot_reach():
    # Blender vertex 1 is a coincident copy no Blender triangle names, so no
    # tet vertex has a triangle with a pinned corner.
    mesh, faces = cube(n=1, copies=20000)
    with pytest.raises(ValueError) as info:
        _map_pins(mesh, faces, [1])
    message = str(info.value)
    assert "SOLID 'Cube'" in message
    assert "reaches no vertex" in message
