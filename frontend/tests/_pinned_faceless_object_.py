# File: frontend/tests/_pinned_faceless_object_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# AN OBJECT WITH NO FACES IS NEVER A STATIC COLLISION MESH. Pinning every
# vertex with no pin operation makes a SURFACE object static: it leaves the
# solved vertices for the contact-only collision mesh, which holds triangles.
# A rod or a point set has no triangle, so built static it would contribute
# nothing, and everything would fall through it (a yarn strip hung from a dowel
# pinned whole). It stays solved instead, every vertex held exactly by its fix
# pins, and the solver's pair filter makes it an obstacle.
#
# Covers:
#   * a rod pinned whole is not static and its vertices reach the solved
#     namespace, beside a free rod;
#   * a Cosserat rod pinned whole is the same;
#   * a point set pinned whole is the same;
#   * a triangle mesh pinned whole is still static (the collision mesh path
#     is unchanged for a surface).
#
# The solve half (that the solved, fully pinned rod actually catches a falling
# rod, Cosserat rod, shell and solid) is the rig scenario rig_pinned_rod_obstacle.

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from frontend import _rust  # noqa: F401
    from frontend._asset_ import AssetManager
    from frontend._mesh_ import MeshManager
    from frontend._plot_ import PlotManager
    from frontend._scene_ import Scene
except ImportError:
    pytest.skip(
        "frontend._rust extension not built; run `cargo build --release` first",
        allow_module_level=True,
    )

SCRATCH_ROOT = REPO_ROOT / "frontend" / "tests" / "_scratch_pinned_faceless_object"

N_DOWEL = 11
N_FREE = 8


def line(y, n, axis):
    t = np.linspace(-0.5, 0.5, n)
    V = np.zeros((n, 3))
    V[:, axis] = t
    V[:, 1] = y
    E = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1).astype(np.uint32)
    return V, E


@pytest.fixture
def scene():
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        asset = AssetManager()
        mesh = MeshManager(str(SCRATCH_ROOT / "cache"))
        asset.add.rod("dowel", *line(0.0, N_DOWEL, 2))
        asset.add.rod("free", *line(0.1, N_FREE, 0))
        V, F = mesh.square(res=4)
        asset.add.tri("sheet", V, F)
        P = np.stack([np.linspace(-0.2, 0.2, N_DOWEL), np.zeros(N_DOWEL),
                      np.zeros(N_DOWEL)], axis=1)
        asset.add.points("grains", P)
        yield Scene("pinned_faceless", PlotManager(), asset)
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def solved_vertex_count(fixed):
    return int(len(fixed._vert[1]))


@pytest.mark.parametrize("cosserat", [False, True])
def test_a_rod_pinned_whole_stays_solved(scene, cosserat):
    free = scene.add("free")
    dowel = scene.add("dowel")
    if cosserat:
        free.as_cosserat()
        dowel.as_cosserat()
    dowel.pin()
    fixed = scene.build(quiet=True)
    assert not dowel.static
    assert solved_vertex_count(fixed) == N_FREE + N_DOWEL


def test_a_point_set_pinned_whole_stays_solved(scene):
    scene.add("free")
    grains = scene.add("grains")
    grains.pin()
    grains.update_static()
    assert not grains.static


def test_a_surface_pinned_whole_is_still_static(scene):
    scene.add("free")
    sheet = scene.add("sheet").at(0, -1, 0)
    sheet.pin()
    fixed = scene.build(quiet=True)
    assert sheet.static
    assert solved_vertex_count(fixed) == N_FREE
