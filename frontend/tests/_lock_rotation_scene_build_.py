# File: frontend/tests/_lock_rotation_scene_build_.py
# License: Apache v2.0
#
# Integration test for Lock Rotation through `Scene.build()` /
# `FixedScene.export_fixed()`.
#
# Verifies the data-pipeline contract the solver-facing side of this
# feature depends on:
#   * `rotation_lock` is shaped ``(n_dmap, 3)``, one row per object
#     (dynamic AND static) in the same dmap order as `displacement.bin`
#     (`concat_displacement`);
#   * a locked object's row is its normalized axis; every other row
#     (unlocked dynamic objects, and every static object, since Lock
#     Rotation never applies to statics) is the zero vector;
#   * a scene with nothing locked never even allocates the array
#     (``FixedScene._rotation_lock is None``) and does not write
#     ``bin/rotation_lock.bin`` at all;
#   * a scene with at least one locked object writes
#     ``bin/rotation_lock.bin`` as a flat little-endian float32 array
#     whose byte size is exactly ``n_dmap * 3 * 4``;
#   * Lock Rotation and Lock Translation are independent: an object can
#     set one, both, or neither, and each is exported to its own bin
#     file with its own table;
#   * ``rotation_lock_mode`` is shaped ``(n_dmap,)`` uint32, aligned
#     with ``rotation_lock``: 0 for a disabled row or an enabled
#     allow-only row (the default), 1 for an enabled row whose object
#     called ``lock_rotation_prohibit_axis(True)``. Written to
#     ``bin/rotation_lock_mode.bin`` only when at least one object has
#     Lock Rotation enabled (same emptiness condition as
#     ``rotation_lock.bin``).

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
SCRATCH_ROOT = REPO_ROOT / "frontend" / "tests" / "_scratch_lock_rotation"


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


def test_no_lock_rotation_leaves_scene_untouched():
    with Workspace() as ws:
        scene = ws.scene("plain")
        scene.add("sheet").at(0, 5, 0)
        scene.add("box").at(0, 0, 0).pin()  # static collider

        fixed = scene.build(quiet=True)
        assert fixed._rotation_lock is None
        assert fixed._rotation_lock_mode is None

        export_dir = ws.root / "export_plain"
        fixed.export_fixed(str(export_dir), delete_exist=True)
        assert not (export_dir / "bin" / "rotation_lock.bin").exists()
        assert not (export_dir / "bin" / "rotation_lock_mode.bin").exists()


def test_locked_object_row_matches_dmap_order():
    with Workspace() as ws:
        scene = ws.scene("locked")
        lower = scene.add("sheet").at(0, 5, 0)
        lower.lock_rotation(0, 1, 0)
        upper = scene.add("sheet").at(0, 10, 0)  # left unlocked
        collider = scene.add("box").at(0, 0, 0)
        collider.pin()  # static; never eligible for Lock Rotation

        fixed = scene.build(quiet=True)
        assert fixed._rotation_lock is not None
        table = fixed._rotation_lock
        assert table.shape == (3, 3)
        assert table.dtype == np.float32
        assert fixed._rotation_lock_mode is not None
        mode_table = fixed._rotation_lock_mode
        assert mode_table.shape == (3,)
        assert mode_table.dtype == np.uint32
        # `lower` never called lock_rotation_prohibit_axis(), so its mode
        # stays the default allow-only (0), same as every disabled row.
        assert np.all(mode_table == 0)

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
        bin_path = export_dir / "bin" / "rotation_lock.bin"
        assert bin_path.exists()
        assert bin_path.stat().st_size == table.shape[0] * 3 * 4
        on_disk = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 3)
        np.testing.assert_allclose(on_disk, table)


def test_lock_translation_and_lock_rotation_are_independent():
    with Workspace() as ws:
        scene = ws.scene("both")
        both = scene.add("sheet").at(0, 5, 0)
        both.lock_translation(1, 0, 0)
        both.lock_rotation(0, 1, 0)
        rot_only = scene.add("sheet").at(0, 10, 0)
        rot_only.lock_rotation(0, 0, 1)

        fixed = scene.build(quiet=True)
        assert fixed._translation_lock is not None
        assert fixed._rotation_lock is not None

        by_identity = {id(obj): ref for ref, obj in scene.object_dict.items()}
        names = list(scene.object_dict.keys())
        both_i = names.index(by_identity[id(both)])
        rot_only_i = names.index(by_identity[id(rot_only)])

        np.testing.assert_allclose(fixed._translation_lock[both_i], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(fixed._rotation_lock[both_i], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(
            fixed._translation_lock[rot_only_i], [0.0, 0.0, 0.0]
        )
        np.testing.assert_allclose(fixed._rotation_lock[rot_only_i], [0.0, 0.0, 1.0])

        export_dir = ws.root / "export_both"
        fixed.export_fixed(str(export_dir), delete_exist=True)
        assert (export_dir / "bin" / "translation_lock.bin").exists()
        assert (export_dir / "bin" / "rotation_lock.bin").exists()


def test_rotation_lock_mode_distinguishes_allow_only_and_prohibit_axis():
    with Workspace() as ws:
        scene = ws.scene("modes")
        allow_only = scene.add("sheet").at(0, 5, 0)
        allow_only.lock_rotation(1, 0, 0)
        prohibited = scene.add("sheet").at(0, 10, 0)
        prohibited.lock_rotation(0, 1, 0)
        prohibited.lock_rotation_prohibit_axis(True)
        unlocked = scene.add("sheet").at(0, 15, 0)

        fixed = scene.build(quiet=True)
        mode_table = fixed._rotation_lock_mode
        assert mode_table is not None
        assert mode_table.dtype == np.uint32

        by_identity = {id(obj): ref for ref, obj in scene.object_dict.items()}
        names = list(scene.object_dict.keys())
        allow_only_i = names.index(by_identity[id(allow_only)])
        prohibited_i = names.index(by_identity[id(prohibited)])
        unlocked_i = names.index(by_identity[id(unlocked)])

        assert mode_table[allow_only_i] == 0
        assert mode_table[prohibited_i] == 1
        assert mode_table[unlocked_i] == 0

        export_dir = ws.root / "export_modes"
        fixed.export_fixed(str(export_dir), delete_exist=True)
        bin_path = export_dir / "bin" / "rotation_lock_mode.bin"
        assert bin_path.exists()
        assert bin_path.stat().st_size == mode_table.shape[0] * 4
        on_disk = np.fromfile(bin_path, dtype=np.uint32)
        np.testing.assert_array_equal(on_disk, mode_table)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
