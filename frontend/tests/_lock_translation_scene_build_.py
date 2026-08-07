# File: frontend/tests/_lock_translation_scene_build_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Integration test for Lock Translation through `Scene.build()` /
# `FixedScene.export_fixed()`.
#
# Verifies the data-pipeline contract the solver-facing side of this
# feature depends on:
#   * `translation_lock` is shaped ``(n_dmap, 3)``, one row per object
#     (dynamic AND static) in the same dmap order as `displacement.bin`
#     (`concat_displacement`);
#   * a locked object's row is its normalized axis; every other row
#     (unlocked dynamic objects, and every static object, since Lock
#     Translation never applies to statics) is the zero vector;
#   * a scene with nothing locked never even allocates the array
#     (``FixedScene._translation_lock is None``) and does not write
#     ``bin/translation_lock.bin`` at all;
#   * a scene with at least one locked object writes
#     ``bin/translation_lock.bin`` as a flat little-endian float32
#     array whose byte size is exactly ``n_dmap * 3 * 4``.

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
        "frontend._rust extension not built; run `cargo build` or "
        "`cargo build-emul` first",
        allow_module_level=True,
    )

# Scratch directory lives under the repo, never under /tmp: cleaned up
# by each test via a try/finally.
SCRATCH_ROOT = REPO_ROOT / "frontend" / "tests" / "_scratch_lock_translation"


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


def test_no_lock_translation_leaves_scene_untouched():
    with Workspace() as ws:
        scene = ws.scene("plain")
        scene.add("sheet").at(0, 5, 0)
        scene.add("box").at(0, 0, 0).pin()  # static collider

        fixed = scene.build(quiet=True)
        assert fixed._translation_lock is None

        export_dir = ws.root / "export_plain"
        fixed.export_fixed(str(export_dir), delete_exist=True)
        assert not (export_dir / "bin" / "translation_lock.bin").exists()


def test_locked_object_row_matches_dmap_order():
    with Workspace() as ws:
        scene = ws.scene("locked")
        lower = scene.add("sheet").at(0, 5, 0)
        lower.lock_translation(0, 1, 0)
        upper = scene.add("sheet").at(0, 10, 0)  # left unlocked
        collider = scene.add("box").at(0, 0, 0)
        collider.pin()  # static; never eligible for Lock Translation

        fixed = scene.build(quiet=True)
        assert fixed._translation_lock is not None
        table = fixed._translation_lock
        assert table.shape == (3, 3)
        assert table.dtype == np.float32

        # dmap order mirrors `Scene._object` insertion order: lower (0),
        # upper (1, auto-renamed "sheet_1" since "sheet" is taken),
        # collider (2). ``Object.name`` is the ASSET reference name
        # ("sheet" for both), not the per-scene ref_name used as the
        # dmap key, so resolve indices by identity through
        # ``scene.object_dict`` instead.
        by_identity = {id(obj): ref for ref, obj in scene.object_dict.items()}
        names = list(scene.object_dict.keys())
        lower_row = table[names.index(by_identity[id(lower)])]
        upper_row = table[names.index(by_identity[id(upper)])]
        box_row = table[names.index(by_identity[id(collider)])]

        np.testing.assert_allclose(lower_row, [0.0, 1.0, 0.0])
        np.testing.assert_allclose(upper_row, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(box_row, [0.0, 0.0, 0.0])

        export_dir = ws.root / "export_locked"
        fixed.export_fixed(str(export_dir), delete_exist=True)
        bin_path = export_dir / "bin" / "translation_lock.bin"
        assert bin_path.exists()
        assert bin_path.stat().st_size == table.shape[0] * 3 * 4
        on_disk = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 3)
        np.testing.assert_allclose(on_disk, table)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
