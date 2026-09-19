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
#     called ``lock_rotation_prohibit_axis(True)``, 2 for one whose
#     object called ``lock_all_rotations()``. Written to
#     ``bin/rotation_lock_mode.bin`` only when at least one object has
#     Lock Rotation enabled (same emptiness condition as
#     ``rotation_lock.bin``);
#   * ``translation_lock_mode`` is the same table for Lock Translation
#     (0 = axis, 1 = all axes), written to
#     ``bin/translation_lock_mode.bin`` under the same emptiness
#     condition as ``translation_lock.bin``;
#   * an all-axes lock carries an EXACTLY ZERO axis row, since no
#     direction exists in that mode and the solver asserts that
#     canonical spelling at load;
#   * a scene whose only locked objects lock ALL axes still exports both
#     tables. The emptiness test asks whether any object is locked in
#     any mode, not whether any object has an axis: an all-axes row is
#     all zeros, so an axis-only test would drop the whole file and the
#     solver would read an unlocked scene with no wrong number anywhere
#     to notice.
#
# The decoder half of the contract is covered here too, since it is what
# turns an addon payload into the object state these tables are built
# from: an all-axes UUID is absent from the paired axis dict (the
# encoder omits it, because there is no axis to encode), and a UUID
# carrying both an axis and a true all-axes flag is refused in whichever
# order the two keys arrive.

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
    from frontend._decoder_ import ParamDecoder
    from frontend._mesh_ import MeshManager
    from frontend._plot_ import PlotManager
    from frontend._scene_ import (
        ROTATION_LOCK_ALL,
        ROTATION_LOCK_ALLOW_ONLY,
        ROTATION_LOCK_PROHIBIT_AXIS,
        TRANSLATION_LOCK_ALL,
        TRANSLATION_LOCK_AXIS,
        Scene,
    )
except ImportError:
    pytest.skip(
        "frontend._rust extension not built; run `cargo build --release` first",
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
        assert fixed._translation_lock is None
        assert fixed._translation_lock_mode is None

        export_dir = ws.root / "export_plain"
        fixed.export_fixed(str(export_dir), delete_exist=True)
        assert not (export_dir / "bin" / "rotation_lock.bin").exists()
        assert not (export_dir / "bin" / "rotation_lock_mode.bin").exists()
        assert not (export_dir / "bin" / "translation_lock.bin").exists()
        assert not (export_dir / "bin" / "translation_lock_mode.bin").exists()


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

        # Both objects lock on an axis, so every mode row is the axis
        # value in both tables.
        assert fixed._translation_lock_mode is not None
        assert np.all(fixed._translation_lock_mode == TRANSLATION_LOCK_AXIS)
        assert np.all(fixed._rotation_lock_mode == ROTATION_LOCK_ALLOW_ONLY)

        export_dir = ws.root / "export_both"
        fixed.export_fixed(str(export_dir), delete_exist=True)
        assert (export_dir / "bin" / "translation_lock.bin").exists()
        assert (export_dir / "bin" / "translation_lock_mode.bin").exists()
        assert (export_dir / "bin" / "rotation_lock.bin").exists()
        assert (export_dir / "bin" / "rotation_lock_mode.bin").exists()


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


def test_all_axes_rotation_lock_exports_zero_axis_and_all_mode():
    # The ONLY locked object in this scene locks every axis, which is
    # what makes this the emptiness test: an all-axes row is all zeros,
    # so a "does any object have an axis?" gate would export no file at
    # all and the solver would run the scene unlocked.
    with Workspace() as ws:
        scene = ws.scene("rot_all")
        locked = scene.add("sheet").at(0, 5, 0)
        locked.lock_all_rotations()
        free = scene.add("sheet").at(0, 10, 0)

        fixed = scene.build(quiet=True)
        assert fixed._rotation_lock is not None
        assert fixed._rotation_lock_mode is not None
        table = fixed._rotation_lock
        mode_table = fixed._rotation_lock_mode
        assert table.shape == (2, 3)
        assert mode_table.shape == (2,)
        assert mode_table.dtype == np.uint32

        by_identity = {id(obj): ref for ref, obj in scene.object_dict.items()}
        names = list(scene.object_dict.keys())
        locked_i = names.index(by_identity[id(locked)])
        free_i = names.index(by_identity[id(free)])

        # An all-axes row carries an EXACTLY zero axis (not merely a
        # small one): that biconditional is asserted solver-side, so the
        # comparison here is exact rather than a tolerance.
        np.testing.assert_array_equal(table[locked_i], np.zeros(3, dtype=np.float32))
        assert mode_table[locked_i] == ROTATION_LOCK_ALL
        np.testing.assert_array_equal(table[free_i], np.zeros(3, dtype=np.float32))
        assert mode_table[free_i] == ROTATION_LOCK_ALLOW_ONLY

        export_dir = ws.root / "export_rot_all"
        fixed.export_fixed(str(export_dir), delete_exist=True)
        axis_path = export_dir / "bin" / "rotation_lock.bin"
        mode_path = export_dir / "bin" / "rotation_lock_mode.bin"
        assert axis_path.exists()
        assert mode_path.exists()
        assert axis_path.stat().st_size == table.shape[0] * 3 * 4
        assert mode_path.stat().st_size == mode_table.shape[0] * 4
        np.testing.assert_array_equal(
            np.fromfile(mode_path, dtype=np.uint32), mode_table
        )
        # Lock Translation stays untouched by a rotation-only scene.
        assert not (export_dir / "bin" / "translation_lock.bin").exists()
        assert not (export_dir / "bin" / "translation_lock_mode.bin").exists()


def test_all_axes_translation_lock_exports_zero_axis_and_all_mode():
    # The translation half of the emptiness test above, on a scene whose
    # only locked object again carries no axis.
    with Workspace() as ws:
        scene = ws.scene("trans_all")
        locked = scene.add("sheet").at(0, 5, 0)
        locked.lock_all_translations()
        free = scene.add("sheet").at(0, 10, 0)

        fixed = scene.build(quiet=True)
        assert fixed._translation_lock is not None
        assert fixed._translation_lock_mode is not None
        table = fixed._translation_lock
        mode_table = fixed._translation_lock_mode
        assert table.shape == (2, 3)
        assert table.dtype == np.float32
        assert mode_table.shape == (2,)
        assert mode_table.dtype == np.uint32

        by_identity = {id(obj): ref for ref, obj in scene.object_dict.items()}
        names = list(scene.object_dict.keys())
        locked_i = names.index(by_identity[id(locked)])
        free_i = names.index(by_identity[id(free)])

        np.testing.assert_array_equal(table[locked_i], np.zeros(3, dtype=np.float32))
        assert mode_table[locked_i] == TRANSLATION_LOCK_ALL
        np.testing.assert_array_equal(table[free_i], np.zeros(3, dtype=np.float32))
        assert mode_table[free_i] == TRANSLATION_LOCK_AXIS

        export_dir = ws.root / "export_trans_all"
        fixed.export_fixed(str(export_dir), delete_exist=True)
        axis_path = export_dir / "bin" / "translation_lock.bin"
        mode_path = export_dir / "bin" / "translation_lock_mode.bin"
        assert axis_path.exists()
        assert mode_path.exists()
        assert axis_path.stat().st_size == table.shape[0] * 3 * 4
        assert mode_path.stat().st_size == mode_table.shape[0] * 4
        np.testing.assert_array_equal(
            np.fromfile(axis_path, dtype=np.float32).reshape(-1, 3), table
        )
        np.testing.assert_array_equal(
            np.fromfile(mode_path, dtype=np.uint32), mode_table
        )
        assert not (export_dir / "bin" / "rotation_lock.bin").exists()
        assert not (export_dir / "bin" / "rotation_lock_mode.bin").exists()


def test_every_rotation_mode_coexists_in_one_scene():
    with Workspace() as ws:
        scene = ws.scene("all_modes")
        allow_only = scene.add("sheet").at(0, 5, 0)
        allow_only.lock_rotation(1, 0, 0)
        prohibited = scene.add("sheet").at(0, 10, 0)
        prohibited.lock_rotation(0, 1, 0).lock_rotation_prohibit_axis(True)
        every_axis = scene.add("sheet").at(0, 15, 0)
        every_axis.lock_all_rotations()
        unlocked = scene.add("sheet").at(0, 20, 0)

        fixed = scene.build(quiet=True)
        table = fixed._rotation_lock
        mode_table = fixed._rotation_lock_mode
        assert table is not None
        assert mode_table is not None

        by_identity = {id(obj): ref for ref, obj in scene.object_dict.items()}
        names = list(scene.object_dict.keys())
        rows = {
            key: names.index(by_identity[id(obj)])
            for key, obj in (
                ("allow_only", allow_only),
                ("prohibited", prohibited),
                ("every_axis", every_axis),
                ("unlocked", unlocked),
            )
        }

        np.testing.assert_allclose(table[rows["allow_only"]], [1.0, 0.0, 0.0])
        assert mode_table[rows["allow_only"]] == ROTATION_LOCK_ALLOW_ONLY
        np.testing.assert_allclose(table[rows["prohibited"]], [0.0, 1.0, 0.0])
        assert mode_table[rows["prohibited"]] == ROTATION_LOCK_PROHIBIT_AXIS
        np.testing.assert_array_equal(
            table[rows["every_axis"]], np.zeros(3, dtype=np.float32)
        )
        assert mode_table[rows["every_axis"]] == ROTATION_LOCK_ALL
        np.testing.assert_array_equal(
            table[rows["unlocked"]], np.zeros(3, dtype=np.float32)
        )
        assert mode_table[rows["unlocked"]] == ROTATION_LOCK_ALLOW_ONLY

        # Exactly one row per mode, and exactly one object left free:
        # a mode leaking onto a bystander row is what this counts.
        assert int(np.count_nonzero(mode_table == ROTATION_LOCK_ALL)) == 1
        assert int(np.count_nonzero(mode_table == ROTATION_LOCK_PROHIBIT_AXIS)) == 1
        assert int(np.count_nonzero(mode_table == ROTATION_LOCK_ALLOW_ONLY)) == 2
        assert int(np.count_nonzero(np.any(table != 0.0, axis=1))) == 2


def _decode_lock_params(scene: Scene, params: dict, ref_name: str) -> None:
    """Drive ``ParamDecoder.apply_to_objects`` over one object's params.

    The decoder addresses objects by UUID through ``Scene.select``, so a
    scene whose reference name IS the UUID exercises the real per-object
    loop without a saved payload on disk.
    """
    decoder = ParamDecoder()
    decoder._data = {"group": [(params, [ref_name], [ref_name])]}
    decoder.apply_to_objects(scene)


def test_decoder_accepts_an_all_axes_lock_with_no_paired_axis():
    # The encoder omits an all-locked object from the axis dicts, since
    # the axis encoder has no axis to encode and the wire record must
    # stay canonical. A UUID present only in the all-dict is therefore
    # the expected shape, not a missing-axis error.
    with Workspace() as ws:
        scene = ws.scene("decode_all")
        scene.add("sheet").at(0, 5, 0)
        _decode_lock_params(
            scene,
            {
                "lock-translation": {},
                "lock-all-translations": {"sheet": True},
                "lock-rotation": {},
                "lock-all-rotations": {"sheet": True},
            },
            "sheet",
        )

        obj = scene.select("sheet")
        assert obj._translation_lock_all is True
        assert obj._translation_lock is None
        assert obj._rotation_lock_all is True
        assert obj._rotation_lock is None
        assert obj._rotation_lock_prohibit_axis is False


def test_decoder_leaves_an_axis_lock_alone_when_the_all_flag_is_false():
    with Workspace() as ws:
        scene = ws.scene("decode_axis")
        scene.add("sheet").at(0, 5, 0)
        _decode_lock_params(
            scene,
            {
                "lock-translation": {"sheet": [0.0, 1.0, 0.0]},
                "lock-all-translations": {"sheet": False},
                "lock-rotation": {"sheet": [0.0, 0.0, 1.0]},
                "lock-all-rotations": {"sheet": False},
            },
            "sheet",
        )

        obj = scene.select("sheet")
        assert obj._translation_lock_all is False
        np.testing.assert_allclose(obj._translation_lock, [0.0, 1.0, 0.0])
        assert obj._rotation_lock_all is False
        np.testing.assert_allclose(obj._rotation_lock, [0.0, 0.0, 1.0])


@pytest.mark.parametrize("all_key_first", [True, False])
def test_decoder_refuses_an_axis_beside_an_all_axes_translation_lock(all_key_first):
    # A user typing two calls is expressing intent and last-call-wins
    # settles it, but an ENCODER emitting both spellings of one lock has
    # a bug, and resolving it silently would pick a winner by map order.
    # Both key orders must refuse.
    with Workspace() as ws:
        scene = ws.scene("decode_conflict_t")
        scene.add("sheet").at(0, 5, 0)
        entries = [
            ("lock-all-translations", {"sheet": True}),
            ("lock-translation", {"sheet": [1.0, 0.0, 0.0]}),
        ]
        if not all_key_first:
            entries.reverse()
        with pytest.raises(ValueError, match="lock-all-translations"):
            _decode_lock_params(scene, dict(entries), "sheet")


@pytest.mark.parametrize("all_key_first", [True, False])
def test_decoder_refuses_an_axis_beside_an_all_axes_rotation_lock(all_key_first):
    with Workspace() as ws:
        scene = ws.scene("decode_conflict_r")
        scene.add("sheet").at(0, 5, 0)
        entries = [
            ("lock-all-rotations", {"sheet": True}),
            ("lock-rotation", {"sheet": [1.0, 0.0, 0.0]}),
        ]
        if not all_key_first:
            entries.reverse()
        with pytest.raises(ValueError, match="lock-all-rotations"):
            _decode_lock_params(scene, dict(entries), "sheet")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
