# File: frontend/tests/_cosserat_scene_build_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Cosserat rods through `Scene.build()` and `FixedScene.export_fixed()`, the
# data the solver reads for them.
#
# A Cosserat rod is a rod object whose segments carry an orientation frame. Its
# segments share the rod table with every plain rod's, so the scene build owes
# the solver three aligned things:
#   * `bin/cosserat_edge.bin`, one uint8 per rod edge in `rod.bin` order, 1
#     where the edge belongs to a Cosserat rod; written only when one does;
#   * the rod parameter files, one value per rod edge, where the Cosserat keys
#     (`rod-cosserat-*.bin`) hold a Cosserat rod's values on its edges and 0.0
#     on a plain rod's; a scene with no Cosserat rod writes none of them;
#   * `bin/cosserat_tilt.bin`, one float32 roll per dynamic vertex in
#     `vert.bin` order, scattered from each rod's OWN vertex order through its
#     index map; written only when an object authored a roll.
#
# Covers as well: the refusals of `as_cosserat` and `tilt`, a spatial map on a
# Cosserat key beside a plain rod, `MeshManager.helix`, and the decoder's
# COSSERAT group (the scene payload and the group params).

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
    from frontend import _cbor_bridge_ as _cbor
    from frontend import _rust  # noqa: F401
    from frontend._asset_ import AssetManager
    from frontend._decoder_ import ParamDecoder, SceneDecoder
    from frontend._mesh_ import MeshManager
    from frontend._plot_ import PlotManager
    from frontend._scene_ import Scene
except ImportError:
    pytest.skip(
        "frontend._rust extension not built; run `cargo build --release` first",
        allow_module_level=True,
    )

# Scratch directory lives under the repo, never under /tmp: removed by the
# fixture.
SCRATCH_ROOT = REPO_ROOT / "frontend" / "tests" / "_scratch_cosserat_scene_build"

COSSERAT_KEYS = (
    "cosserat-shear",
    "cosserat-bend",
    "cosserat-twist",
    "cosserat-rest-twist",
    "cosserat-fix-ends",
    "cosserat-end-twist-rate",
    "cosserat-straight-rest",
)


def strand(n_segments=6, length=0.6):
    y = np.linspace(0.0, -length, n_segments + 1)
    vert = np.stack([np.zeros_like(y), y, np.zeros_like(y)], axis=1)
    edge = np.array([[i, i + 1] for i in range(n_segments)], dtype=np.uint32)
    return vert, edge


def reversed_strand(n_segments=6, length=0.6):
    """The same strand with every segment written tip first, so the index map
    assigns its vertices in DESCENDING local order. A roll gathered in global
    order rather than scattered through the map would come out reversed."""
    vert, edge = strand(n_segments, length)
    return vert, edge[::-1, ::-1].copy()


class Workspace:
    def __init__(self, root: Path):
        self.root = root
        self.asset = AssetManager()
        self.mesh = MeshManager(str(root / "cache"))
        self.plot = PlotManager()
        self.asset.add.rod("strand", *strand())
        self.asset.add.rod("reversed", *reversed_strand())
        self.asset.add.rod("short", *strand(n_segments=3, length=0.3))
        V, F = self.mesh.square(res=3)
        self.asset.add.tri("sheet", V, F)
        self._exports = 0

    def scene(self, name: str = "cosserat") -> Scene:
        return Scene(name, self.plot, self.asset)

    def export(self, fixed) -> Path:
        self._exports += 1
        path = self.root / f"export_{self._exports}"
        fixed.export_fixed(str(path), delete_exist=True)
        return path


@pytest.fixture
def ws():
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        yield Workspace(SCRATCH_ROOT)
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def read(path: Path, name: str, dtype) -> np.ndarray:
    return np.fromfile(path / "bin" / name, dtype=dtype)


def rod_edges(path: Path) -> np.ndarray:
    return read(path, "rod.bin", np.uint64).reshape(-1, 2).astype(np.int64)


def rod_param(path: Path, key: str) -> np.ndarray:
    return np.fromfile(path / "bin" / "param" / f"rod-{key}.bin", dtype=np.float32)


def rod_param_files(path: Path) -> set[str]:
    return {p.name for p in (path / "bin" / "param").glob("rod-*.bin")}


def owned_edges(fixed, path: Path, key: str) -> np.ndarray:
    """A mask over the exported rod edges: True where both ends belong to the
    scene object `key`, found from the vertex map rather than from the order
    the objects were added in."""
    own = set(np.asarray(fixed._map_by_name[key]).tolist())
    edges = rod_edges(path)
    return np.array([a in own and b in own for a, b in edges], dtype=bool)


# ----- the marker and the rod table ------------------------------------------


@pytest.mark.parametrize("cosserat_first", [False, True])
def test_a_cosserat_rod_beside_plain_rods_exports_aligned_tables(ws, cosserat_first):
    scene = ws.scene()

    def add_plain():
        plain = scene.add("short", "plain").at(-1.0, 0.0, 0.0)
        plain.param.set("young-mod", 2.0e4)
        return plain

    def add_cosserat():
        rod = scene.add("strand", "cosserat").as_cosserat()
        rod.param.set("cosserat-shear", 500.0).set("cosserat-twist", 70.0)
        rod.param.set("cosserat-fix-ends", 1.0)
        return rod

    if cosserat_first:
        cosserat, plain = add_cosserat(), add_plain()
    else:
        plain, cosserat = add_plain(), add_cosserat()
    scene.add("reversed", "other").at(1.0, 0.0, 0.0)
    # A shell beside the rods, whose triangles take no part in the rod table.
    scene.add("sheet", "sheet").at(0.0, 2.0, 0.0)
    assert cosserat.cosserat and not plain.cosserat

    fixed = scene.build(quiet=True)
    path = ws.export(fixed)

    edges = rod_edges(path)
    n_rod = len(edges)
    assert n_rod == 3 + 6 + 6
    is_cosserat = owned_edges(fixed, path, "cosserat")
    assert int(is_cosserat.sum()) == 6

    marker = read(path, "cosserat_edge.bin", np.uint8)
    assert len(marker) == n_rod
    assert np.array_equal(marker, is_cosserat.astype(np.uint8))

    # Every rod table holds one value per rod edge, the Cosserat keys included.
    for name in rod_param_files(path):
        if name == "rod-model.bin":
            values = np.fromfile(path / "bin" / "param" / name, dtype=np.uint8)
        else:
            values = np.fromfile(path / "bin" / "param" / name, dtype=np.float32)
        assert len(values) == n_rod, name
    for key in COSSERAT_KEYS:
        assert f"rod-{key}.bin" in rod_param_files(path), key

    # A Cosserat rod's values on its own edges, 0.0 on every plain rod's.
    expected = {
        "cosserat-shear": 500.0,
        "cosserat-bend": 1.0e4,
        "cosserat-twist": 70.0,
        "cosserat-rest-twist": 0.0,
        "cosserat-fix-ends": 1.0,
        "cosserat-end-twist-rate": 0.0,
        "cosserat-straight-rest": 0.0,
    }
    for key, value in expected.items():
        values = rod_param(path, key)
        assert np.all(values[is_cosserat] == np.float32(value)), key
        assert np.all(values[~is_cosserat] == 0.0), key

    # The keys every rod carries stay with their own edges.
    young = rod_param(path, "young-mod")
    is_plain = owned_edges(fixed, path, "plain")
    assert np.all(young[is_plain] == np.float32(2.0e4))
    assert np.all(young[~is_plain] == np.float32(1.0e4))
    # The position hinge of a Cosserat rod is inert.
    assert np.all(rod_param(path, "bend")[is_cosserat] == 0.0)

    # No roll was authored, so there is no roll file.
    assert not (path / "bin" / "cosserat_tilt.bin").exists()


def test_a_scene_of_plain_rods_writes_nothing_cosserat(ws):
    scene = ws.scene()
    scene.add("short", "plain").at(-1.0, 0.0, 0.0)
    scene.add("strand", "other")
    fixed = scene.build(quiet=True)
    assert fixed._cosserat_edge is None
    assert fixed._cosserat_tilt is None
    path = ws.export(fixed)
    assert not (path / "bin" / "cosserat_edge.bin").exists()
    assert not (path / "bin" / "cosserat_tilt.bin").exists()
    assert not any("cosserat" in name for name in rod_param_files(path))


def test_a_scene_of_cosserat_rods_alone_marks_every_rod_edge(ws):
    scene = ws.scene()
    scene.add("strand", "a").as_cosserat()
    scene.add("short", "b").at(1.0, 0.0, 0.0).as_cosserat()
    path = ws.export(scene.build(quiet=True))
    marker = read(path, "cosserat_edge.bin", np.uint8)
    assert len(marker) == 9 and np.all(marker == 1)
    assert np.all(rod_param(path, "cosserat-bend") == np.float32(1.0e4))


def test_a_spatial_map_on_a_cosserat_key_beside_a_plain_rod(ws):
    scene = ws.scene()
    scene.add("short", "plain").at(-1.0, 0.0, 0.0)
    rod = scene.add("strand", "cosserat").as_cosserat()
    rod.param.set("cosserat-bend", 10.0)
    weights = np.linspace(0.0, 1.0, len(rod.get("V")), dtype=np.float32)
    rod.set_param_spatial("cosserat-bend", weights, 300.0)
    fixed = scene.build(quiet=True)
    path = ws.export(fixed)

    is_cosserat = owned_edges(fixed, path, "cosserat")
    values = rod_param(path, "cosserat-bend")
    assert np.all(values[~is_cosserat] == 0.0)
    # The rod's own segments, in the order its edges are listed.
    w = weights.astype(np.float64)[np.asarray(rod.get("E"))].mean(axis=1)
    assert values[is_cosserat] == pytest.approx(10.0 + (300.0 - 10.0) * w, rel=1e-6)


# ----- the authored roll ------------------------------------------------------


def test_an_authored_roll_reaches_each_vertex_through_the_index_map(ws):
    scene = ws.scene()
    scene.add("short", "plain").at(-1.0, 0.0, 0.0)
    rod = scene.add("reversed", "cosserat").as_cosserat()
    roll = np.linspace(0.0, np.pi, len(rod.get("V")), dtype=np.float32)
    rod.tilt(roll)
    fixed = scene.build(quiet=True)
    path = ws.export(fixed)

    indices = np.asarray(fixed._map_by_name["cosserat"], dtype=np.int64)
    # The test is only discriminating if the map does not keep the rod's own
    # vertex order.
    assert np.any(np.diff(indices) < 0)

    n_vert = len(read(path, "vert.bin", np.float64)) // 3
    tilt = read(path, "cosserat_tilt.bin", np.float32)
    assert len(tilt) == n_vert
    assert np.array_equal(tilt[indices], roll)
    others = np.ones(n_vert, dtype=bool)
    others[indices] = False
    assert np.all(tilt[others] == 0.0)


def test_tilt_and_as_cosserat_refuse_what_they_cannot_honor(ws):
    scene = ws.scene()
    sheet = scene.add("sheet", "sheet")
    with pytest.raises(ValueError, match="requires a Rod asset"):
        sheet.as_cosserat()
    assert not sheet.cosserat

    plain = scene.add("strand", "plain")
    n = len(plain.get("V"))
    with pytest.raises(ValueError, match="requires a Cosserat rod"):
        plain.tilt(np.zeros(n))

    rod = scene.add("short", "cosserat").as_cosserat()
    n = len(rod.get("V"))
    with pytest.raises(ValueError, match="one roll angle per vertex"):
        rod.tilt(np.zeros(n + 1))
    bad = np.zeros(n)
    bad[1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        rod.tilt(bad)

    rail = scene.add("strand", "rail")
    rail.pin()
    rail.update_static()
    assert rail.static
    with pytest.raises(ValueError, match="static"):
        rail.as_cosserat()


def test_a_cosserat_rod_carries_the_seven_keys_and_a_plain_rod_none(ws):
    scene = ws.scene()
    plain = scene.add("strand", "plain")
    rod = scene.add("short", "cosserat").as_cosserat()
    for key in COSSERAT_KEYS:
        assert key in rod.param.key_list()
        assert key not in plain.param.key_list()
    assert set(rod.param.key_list()) == set(plain.param.key_list()) | set(COSSERAT_KEYS)
    assert float(rod.param.get("bend")) == 0.0
    with pytest.raises(KeyError):
        plain.param.set("cosserat-twist", 1.0)


# ----- MeshManager.helix ------------------------------------------------------


def test_helix_winds_about_its_axis_and_builds_as_a_cosserat_rod(ws):
    radius, pitch, turns, n = 0.08, 0.35, 2.5, 101
    V, E = ws.mesh.helix(radius=radius, pitch=pitch, turns=turns, n=n)
    assert V.shape == (n, 3) and V.dtype == np.float64
    assert E.shape == (n - 1, 2) and E.dtype == np.uint32
    assert np.array_equal(E[:, 1] - E[:, 0], np.ones(n - 1, dtype=np.uint32))
    # Default axis is +y: constant distance from it, and the rise spans
    # pitch * turns centered on the origin.
    assert np.hypot(V[:, 0], V[:, 2]) == pytest.approx(radius, rel=1e-12)
    assert V[:, 1].max() - V[:, 1].min() == pytest.approx(pitch * turns)
    assert V[:, 1].mean() == pytest.approx(0.0, abs=1e-12)

    V, _ = ws.mesh.helix(radius=0.2, pitch=0.1, turns=1.0, n=50,
                         center=[1.0, 2.0, 3.0], axis=[0.0, 0.0, 2.0])
    assert np.hypot(V[:, 0] - 1.0, V[:, 1] - 2.0) == pytest.approx(0.2, rel=1e-12)
    assert V[:, 2].mean() == pytest.approx(3.0)

    for bad in ({"n": 1}, {"radius": 0.0}, {"turns": 0.0},
                {"axis": [0.0, 0.0, 0.0]}, {"pitch": np.inf}):
        with pytest.raises(ValueError):
            ws.mesh.helix(**bad)

    ws.asset.add.rod("spring", *ws.mesh.helix(n=40))
    scene = ws.scene()
    scene.add("spring", "spring").as_cosserat()
    path = ws.export(scene.build(quiet=True))
    marker = read(path, "cosserat_edge.bin", np.uint8)
    assert len(marker) == 39 and np.all(marker == 1)


# ----- the decoder ------------------------------------------------------------


def write_scene_payload(root: Path, groups: list) -> str:
    path = root / "data.pickle"
    path.write_bytes(_cbor.dumps_envelope(_cbor.KIND_SCENE, groups))
    return str(path)


def rod_object(name: str, uuid: str, vert, edge, **extra) -> dict:
    return {
        "name": name,
        "uuid": uuid,
        "vert": np.asarray(vert, dtype=np.float32).tolist(),
        "edge": np.asarray(edge, dtype=np.uint32).tolist(),
        "transform": np.eye(4).tolist(),
        **extra,
    }


def test_the_decoder_builds_a_cosserat_group_with_its_roll(ws):
    vert, edge = reversed_strand()
    roll = np.linspace(-1.0, 1.0, len(vert)).tolist()
    plain_vert, plain_edge = strand(n_segments=3, length=0.3)
    plain_vert = plain_vert + np.array([-1.0, 0.0, 0.0])
    payload = write_scene_payload(ws.root, [
        {"type": "ROD", "object": [
            rod_object("Rope", "uuid-rope", plain_vert, plain_edge),
        ]},
        {"type": "COSSERAT", "object": [
            rod_object("Cable", "uuid-cable", vert, edge, tilt=roll),
            rod_object("Wire", "uuid-wire", vert + np.array([1.0, 0.0, 0.0]), edge),
        ]},
    ])
    asset = AssetManager()
    scene = Scene("decoded", ws.plot, asset)
    decoder = SceneDecoder(payload, asset, ws.mesh)
    decoder.populate_objects(scene)

    rope = scene.select("uuid-rope")
    cable = scene.select("uuid-cable")
    wire = scene.select("uuid-wire")
    assert not rope.cosserat and cable.cosserat and wire.cosserat
    assert np.array_equal(cable._cosserat_tilt, np.asarray(roll, dtype=np.float32))
    assert wire._cosserat_tilt is None
    assert decoder._object_info["uuid-cable"].type == "COSSERAT"

    # The group params reach a Cosserat object's keys; a plain rod has none.
    params = ParamDecoder()
    params._data = {"group": [
        ({"cosserat-twist": 42.0, "cosserat-fix-ends": 1.0},
         ["Cable", "Wire"], ["uuid-cable", "uuid-wire"]),
    ]}
    params.apply_to_objects(scene)
    assert float(cable.param.get("cosserat-twist")) == 42.0
    assert float(wire.param.get("cosserat-fix-ends")) == 1.0
    refused = ParamDecoder()
    refused._data = {"group": [
        ({"cosserat-twist": 42.0}, ["Rope"], ["uuid-rope"]),
    ]}
    with pytest.raises(KeyError):
        refused.apply_to_objects(scene)

    fixed = scene.build(quiet=True)
    path = ws.export(fixed)
    is_cosserat = (
        owned_edges(fixed, path, "uuid-cable") | owned_edges(fixed, path, "uuid-wire")
    )
    marker = read(path, "cosserat_edge.bin", np.uint8)
    assert np.array_equal(marker, is_cosserat.astype(np.uint8))
    assert np.all(rod_param(path, "cosserat-twist")[is_cosserat] == np.float32(42.0))
    assert np.all(rod_param(path, "cosserat-twist")[~is_cosserat] == 0.0)
    tilt = read(path, "cosserat_tilt.bin", np.float32)
    indices = np.asarray(fixed._map_by_name["uuid-cable"], dtype=np.int64)
    assert np.array_equal(tilt[indices], np.asarray(roll, dtype=np.float32))
    assert np.all(tilt[np.asarray(fixed._map_by_name["uuid-wire"])] == 0.0)


def test_the_decoder_refuses_a_roll_of_the_wrong_length(ws):
    vert, edge = strand()
    payload = write_scene_payload(ws.root, [
        {"type": "COSSERAT", "object": [
            rod_object("Cable", "uuid-cable", vert, edge, tilt=[0.0, 1.0]),
        ]},
    ])
    asset = AssetManager()
    scene = Scene("decoded", ws.plot, asset)
    with pytest.raises(ValueError, match="one roll angle per vertex"):
        SceneDecoder(payload, asset, ws.mesh).populate_objects(scene)
