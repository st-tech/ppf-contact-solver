# File: frontend/tests/_tie_scene_build_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Tie constraints through `Scene.tie`, `Scene.build()` and
# `FixedScene.export_fixed()`: the three bins the solver reads, in solver
# vertex indices through each object's own index map, padded to eight slots;
# the refusals at the call and at the build; and `tie_weights`.

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
    from frontend._scene_ import TIE_SLOTS, Scene
    from frontend._tie_ import tie_weights
except ImportError:
    pytest.skip(
        "frontend._rust extension not built; run `cargo build --release` first",
        allow_module_level=True,
    )

SCRATCH_ROOT = REPO_ROOT / "frontend" / "tests" / "_scratch_tie_scene_build"


def strand(n, y, z, reverse=False, across=False):
    x = np.linspace(-0.5, 0.5, n)
    vert = np.stack([x, np.full(n, y), np.full(n, z)], axis=1)
    if across:
        vert = vert[:, [1, 0, 2]]
    edge = np.array([[i, i + 1] for i in range(n - 1)], dtype=np.uint32)
    if reverse:
        edge = edge[::-1, ::-1].copy()
    return vert, edge


class Workspace:
    def __init__(self, root: Path):
        self.root = root
        self.asset = AssetManager()
        self.plot = PlotManager()
        # Two crossing ground strands at different heights (together a 3D
        # affine frame) and a floss strand above them; the second ground
        # strand is written tip first so its index map is not the identity.
        self.asset.add.rod("lo", *strand(6, 0.0, -0.05))
        self.asset.add.rod("hi", *strand(6, 0.0, 0.05, reverse=True, across=True))
        self.asset.add.rod("floss", *strand(6, 0.03, 0.2))

    def scene(self) -> Scene:
        scene = Scene("tie", self.plot, self.asset)
        scene.add("lo", "lo")
        scene.add("hi", "hi")
        scene.add("floss", "floss")
        return scene


@pytest.fixture
def ws():
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        yield Workspace(SCRATCH_ROOT)
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def ground(ws):
    lo = np.asarray(ws.asset.fetch.rod("lo")[0], dtype=np.float64)
    hi = np.asarray(ws.asset.fetch.rod("hi")[0], dtype=np.float64)
    return lo, hi


def test_a_tie_exports_solver_indices_padded_to_eight_slots(ws):
    lo, hi = ground(ws)
    floss = np.asarray(ws.asset.fetch.rod("floss")[0], dtype=np.float64)
    cand = np.vstack([lo, hi])
    held = np.array([1, 2, 3, 4])
    ind, w = tie_weights(floss[held], cand, k=6)
    target_of = (ind >= len(lo)).astype(np.int64)
    local = np.where(target_of == 0, ind, ind - len(lo))
    scene = ws.scene()
    scene.tie("floss", held, ["lo", "hi"], local, w, target_of=target_of)
    fixed = scene.build(quiet=True)
    path = ws.root / "export"
    fixed.export_fixed(str(path), delete_exist=True)
    vertex = np.fromfile(path / "bin" / "tie_vertex.bin", dtype=np.uint32)
    tind = np.fromfile(path / "bin" / "tie_ind.bin", dtype=np.uint32).reshape(-1, TIE_SLOTS)
    tw = np.fromfile(path / "bin" / "tie_w.bin", dtype=np.float32).reshape(-1, TIE_SLOTS)
    assert len(vertex) == 4 and tind.shape == (4, 8) and tw.shape == (4, 8)
    fmap = np.asarray(fixed._map_by_name["floss"])
    assert np.array_equal(vertex, fmap[held])
    assert np.all(tw[:, 6:] == 0.0)
    assert np.allclose(tw.sum(axis=1), 1.0, atol=1e-6)
    # The exported combination reproduces each held vertex in the built scene.
    V = fixed.vertex(True)
    target = np.einsum("nk,nkd->nd", tw.astype(np.float64), V[tind])
    assert np.allclose(target, V[vertex], atol=1e-6)


def test_no_tie_writes_no_tie_file(ws):
    fixed = ws.scene().build(quiet=True)
    path = ws.root / "plain"
    fixed.export_fixed(str(path), delete_exist=True)
    assert not (path / "bin" / "tie_vertex.bin").exists()


def test_refusals_at_the_call(ws):
    scene = ws.scene()
    with pytest.raises(ValueError, match="sum to"):
        scene.tie("floss", [1], "lo", [[0, 1]], [[0.5, 0.4]])
    with pytest.raises(ValueError, match="target_of is required"):
        scene.tie("floss", [1], ["lo", "hi"], [[0, 1]], [[0.5, 0.5]])
    with pytest.raises(ValueError, match="span"):
        scene.tie("floss", [1], "lo", [[0, 99]], [[0.5, 0.5]])
    with pytest.raises(ValueError, match="1 <= K"):
        scene.tie("floss", [1], "lo", [list(range(9))], [[1.0 / 9] * 9])


def test_a_tie_off_its_combination_is_refused_at_the_build(ws):
    scene = ws.scene()
    # Vertex 1 of the floss is not the midpoint of lo's vertices 0 and 1.
    scene.tie("floss", [1], "lo", [[0, 1]], [[0.5, 0.5]])
    with pytest.raises(ValueError, match="misses its combination"):
        scene.build(quiet=True)


def test_tie_weights_refuses_a_flat_candidate_set():
    flat = np.array([[x, y, 0.0] for x in range(3) for y in range(3)], dtype=float)
    with pytest.raises(ValueError, match="affine span"):
        tie_weights([[1.0, 1.0, 0.5]], flat, k=6)
    ind, w = tie_weights([[1.0, 1.0, 0.0]], flat, k=6)
    assert np.allclose(w.sum(), 1.0)
    assert np.allclose(w[0] @ flat[ind[0]], [1.0, 1.0, 0.0])
