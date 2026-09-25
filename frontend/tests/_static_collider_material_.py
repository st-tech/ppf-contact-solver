# File: frontend/tests/_static_collider_material_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# An object built as a static collision mesh keeps one value per face, written
# once at build. The decoder hands every object of an add-on group that group's
# animated values and spatial maps, and a rest-pose STATIC collider or a fully
# pinned shell with no pin operation is exactly such an object, so the build
# refuses the combination by name rather than solving on the plain values.
#
# Covers:
#   * an animated value on a static object is refused at build, naming the
#     object and the key;
#   * a spatial map on a static object is refused the same way;
#   * the same animation on an object whose pin pulls (so it is simulated)
#     builds, and its per-frame table carries the animated values;
#   * an object keyed by a UUID, as the Blender add-on's decoder keys every
#     object, is named in the refusal by the Blender name it records.

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

SCRATCH_ROOT = REPO_ROOT / "frontend" / "tests" / "_scratch_static_collider_material"


@pytest.fixture
def scene():
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        asset = AssetManager()
        mesh = MeshManager(str(SCRATCH_ROOT / "cache"))
        V, F = mesh.square(res=4)
        asset.add.tri("sheet", V, F)
        V, F = mesh.box(0.5, 0.5, 0.5)
        asset.add.tri("box", V, F)
        scene = Scene("static_material", PlotManager(), asset)
        scene.set_param_anim_times([0.0, 1.0])
        yield scene
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def test_an_animated_value_on_a_static_object_is_refused(scene):
    scene.add("sheet").at(0, 2, 0)
    collider = scene.add("box", "collider")
    collider.pin()
    collider.set_param_anim("contact-gap", [1e-3, 5e-2])
    with pytest.raises(ValueError, match="'collider' is built as a static") as err:
        scene.build(quiet=True)
    assert "'contact-gap'" in str(err.value)
    assert collider.static


def test_a_decoded_object_is_refused_by_its_blender_name(scene):
    scene.add("sheet").at(0, 2, 0)
    key = "6f0c2a3e-1b9d-4c55-9a0e-3d8f7b2c1e44"
    collider = scene.add("box", key)
    collider._statistics_name = "Table"
    collider.pin()
    collider.set_param_anim("contact-gap", [1e-3, 5e-2])
    with pytest.raises(ValueError) as err:
        scene.build(quiet=True)
    assert "'Table' is built as a static" in str(err.value)
    assert key not in str(err.value)


def test_a_spatial_map_on_a_static_object_is_refused(scene):
    scene.add("sheet").at(0, 2, 0)
    collider = scene.add("box", "collider")
    collider.pin()
    weights = np.linspace(0.0, 1.0, len(collider.get("V")))
    collider.set_param_spatial("friction", weights, 0.8)
    with pytest.raises(ValueError, match="animates or maps 'friction'"):
        scene.build(quiet=True)


def test_the_same_animation_on_a_simulated_object_reaches_its_table(scene):
    scene.add("sheet").at(0, 2, 0)
    collider = scene.add("box", "collider")
    collider.pin().pull(1.0)
    collider.set_param_anim("contact-gap", [1e-3, 5e-2])
    fixed = scene.build(quiet=True)
    assert not collider.static
    frames = fixed._tri_param_anim["contact-gap"]
    n_face = len(collider.get("F"))
    assert frames[0][-n_face:] == pytest.approx([1e-3] * n_face)
    assert frames[1][-n_face:] == pytest.approx([5e-2] * n_face)
