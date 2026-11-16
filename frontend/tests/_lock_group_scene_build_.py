# File: frontend/tests/_lock_group_scene_build_.py
# License: Apache v2.0
#
# Integration test for `Object.lock_group` through `Scene.build()` /
# `FixedScene.export_fixed()`.
#
# Verifies the contract the solver's decoder (scene.rs
# `decode_optional_lock_group`) and builder depend on:
#   * a scene in which no object names a lock group writes no
#     ``bin/lock_group.bin``, which the solver reads as the identity;
#   * otherwise the map holds one uint32 per dmap entry, each entry the
#     smallest dmap index of its group (the identity for an ungrouped
#     entry), so the map is canonical;
#   * members of one group stating different lock settings are refused.

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

# Scratch directory lives under the repo, never under /tmp: cleaned up
# by each test.
SCRATCH_ROOT = REPO_ROOT / "frontend" / "tests" / "_scratch_lock_group"


class Workspace:
    """A scratch asset/mesh/plot bundle backing a from-scratch Scene."""

    def __enter__(self) -> "Workspace":
        self.root = SCRATCH_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        self.asset = AssetManager()
        self.mesh = MeshManager(str(self.root / "cache"))
        self.plot = PlotManager()
        V, F = self.mesh.square(res=3)
        self.asset.add.tri("sheet", V, F)
        V, F = self.mesh.box(0.5, 0.5, 0.5)
        self.asset.add.tri("box", V, F)
        return self

    def __exit__(self, *_exc) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def scene(self, name: str) -> Scene:
        return Scene(name, self.plot, self.asset)


def _index(scene: Scene, obj) -> int:
    by_identity = {id(o): ref for ref, o in scene.object_dict.items()}
    return list(scene.object_dict.keys()).index(by_identity[id(obj)])


def test_no_lock_group_writes_no_map():
    with Workspace() as ws:
        scene = ws.scene("plain")
        scene.add("sheet").at(0, 5, 0).lock_rotation(0, 1, 0)
        scene.add("box").at(0, 0, 0).pin()
        fixed = scene.build(quiet=True)
        assert fixed._lock_group is None
        export_dir = ws.root / "export_plain"
        fixed.export_fixed(str(export_dir), delete_exist=True)
        assert not (export_dir / "bin" / "lock_group.bin").exists()


def test_lock_group_maps_members_to_their_smallest_index():
    with Workspace() as ws:
        scene = ws.scene("grouped")
        sheets = []
        for k in range(4):
            s = scene.add("sheet").at(0, 5 + 5 * k, 0)
            s.lock_rotation(0, 1, 0).lock_rotation_prohibit_axis(True)
            sheets.append(s)
        sheets[1].lock_group("a")
        sheets[3].lock_group("a")
        sheets[2].lock_group("b")
        collider = scene.add("box").at(0, 0, 0)
        collider.pin()

        fixed = scene.build(quiet=True)
        table = fixed._lock_group
        assert table is not None
        assert table.dtype == np.uint32
        assert table.shape == (5,)
        i = [_index(scene, s) for s in sheets]
        expect = np.arange(5, dtype=np.uint32)
        expect[i[3]] = min(i[1], i[3])
        expect[i[1]] = min(i[1], i[3])
        np.testing.assert_array_equal(table, expect)
        # canonical: rep[d] <= d and rep[rep[d]] == rep[d]
        assert np.all(table <= np.arange(5))
        np.testing.assert_array_equal(table[table], table)

        export_dir = ws.root / "export_grouped"
        fixed.export_fixed(str(export_dir), delete_exist=True)
        on_disk = np.fromfile(export_dir / "bin" / "lock_group.bin", dtype=np.uint32)
        np.testing.assert_array_equal(on_disk, table)


def test_lock_group_refuses_members_with_other_settings():
    with Workspace() as ws:
        scene = ws.scene("disagree")
        a = scene.add("sheet").at(0, 5, 0)
        a.lock_rotation(0, 1, 0).lock_rotation_prohibit_axis(True).lock_group("g")
        b = scene.add("sheet").at(0, 10, 0)
        b.lock_rotation(0, 1, 0).lock_group("g")
        with pytest.raises(ValueError, match="lock group 'g'"):
            scene.build(quiet=True)


def test_lock_group_refuses_an_empty_name():
    with Workspace() as ws:
        scene = ws.scene("empty")
        with pytest.raises(ValueError):
            scene.add("sheet").lock_group("")
