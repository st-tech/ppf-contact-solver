# File: frontend/tests/_blender_parity_api_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The public calls that give a notebook what the Blender add-on reaches
# through the decoder: cross-stitches between objects, the three pin
# settings the decoder sets on a holder, and rest-shape
# tracking. The decoder calls the same methods, so a refusal here is a
# refusal on both paths.
#
# Covers:
#   * `Scene.cross_stitch` keeps a valid set and refuses every malformed one
#     by name (unknown or repeated object, shape, index range, weight sums
#     and signs, stiffness);
#   * a fully pinned object a stitch names is SIMULATED, so the stitch index
#     can reach it, while an unnamed one stays a static collision mesh, and a
#     stitched PDRD body is refused at build;
#   * `PinHolder.set_pin_group_id` moves the Python record and the Rust
#     mirror together and refuses an empty id, which the solver would read
#     as one shared group;
#   * `PinHolder.track_rest_shape` builds a rest-shape schedule when the
#     tracking pins hold the whole object, and refuses a partial or
#     operation-less one rather than tearing the rest shape.

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

SCRATCH_ROOT = REPO_ROOT / "frontend" / "tests" / "_scratch_blender_parity_api"


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
        yield Scene("parity", PlotManager(), asset)
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def _vertex_stitch(k: int, source_vertex: int, target_tri) -> tuple:
    """`k` rows joining one source vertex to the centroid of a target face."""
    ind = [[source_vertex] * 3 + list(target_tri)] * k
    w = [[1.0, 0.0, 0.0, 1 / 3, 1 / 3, 1 / 3]] * k
    return ind, w


def test_a_valid_stitch_set_is_kept(scene):
    scene.add("sheet")
    box = scene.add("box")
    tri = [int(i) for i in box.get("F")[0]]
    ind, w = _vertex_stitch(3, 0, tri)
    assert scene.cross_stitch("sheet", "box", ind, w, stiffness=2.5) is scene
    [kept] = scene._cross_stitch
    assert kept["source_name"] == "sheet" and kept["target_name"] == "box"
    assert kept["ind"].shape == (3, 6) and kept["stitch_stiffness"] == 2.5


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda a: a.update(target="sheet"), "joins two objects"),
        (lambda a: a.update(ind=np.zeros((0, 6), dtype=np.int64),
                            w=np.zeros((0, 6))), "(K, 6) with K > 0"),
        (lambda a: a.update(ind=[[0, 0, 0, 0]]), "(K, 6) with K > 0"),
        (lambda a: a.update(w=[[1.0, 0, 0, 1.0, 0, 0]] * 2), "shape"),
        (lambda a: a.update(ind=[[0.0, 0, 0, 0, 1, 2]]), "integers"),
        (lambda a: a.update(ind=[[99, 99, 99, 0, 1, 2]]), "source indices span"),
        (lambda a: a.update(ind=[[0, 0, 0, 0, 1, 999]]), "target indices span"),
        (lambda a: a.update(w=[[0.5, 0.0, 0.0, 1.0, 0.0, 0.0]]), "source weights of row 0"),
        (lambda a: a.update(w=[[1.0, 0.0, 0.0, 0.7, 0.7, -0.4]]), "non-negative"),
        (lambda a: a.update(w=[[1.0, 0.0, 0.0, np.nan, 0.5, 0.5]]), "finite"),
        (lambda a: a.update(stiffness=-1.0), "stiffness"),
        (lambda a: a.update(stiffness=float("inf")), "stiffness"),
    ],
)
def test_a_malformed_stitch_set_is_refused(scene, mutate, message):
    scene.add("sheet")
    scene.add("box")
    args = dict(
        source="sheet",
        target="box",
        ind=[[0, 0, 0, 0, 1, 2]],
        w=[[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]],
        stiffness=1.0,
    )
    mutate(args)
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        scene.cross_stitch(**args)
    assert scene._cross_stitch == []


def test_an_unknown_object_is_refused(scene):
    scene.add("sheet")
    with pytest.raises(Exception, match="nowhere"):
        scene.cross_stitch("sheet", "nowhere", [[0, 0, 0, 0, 0, 0]],
                           [[1.0, 0, 0, 1.0, 0, 0]])


def test_a_stitched_anchor_is_simulated_and_an_unstitched_one_is_not(scene):
    sheet = scene.add("sheet").at(0, 3, 0)
    sheet.pin([0])
    anchor = scene.add("box", "anchor")
    anchor.pin()
    loose = scene.add("box", "loose").at(3, 0, 0)
    loose.pin()
    tri = [int(i) for i in anchor.get("F")[0]]
    ind, w = _vertex_stitch(2, 5, tri)
    scene.cross_stitch("sheet", "anchor", ind, w)
    fixed = scene.build(quiet=True)
    assert not anchor.static, "a stitch endpoint must stay in the solved namespace"
    assert loose.static, "an object no stitch names stays a collision mesh"
    n_sheet, n_box = len(sheet.get("V")), len(anchor.get("V"))
    assert len(fixed.vertex()) == n_sheet + n_box
    assert fixed._stitch_ind.shape[0] == 2


def test_a_stitched_pdrd_body_is_refused_at_build(scene):
    scene.add("sheet")
    body = scene.add("box").as_pdrd()
    tri = [int(i) for i in body.get("F")[0]]
    ind, w = _vertex_stitch(1, 0, tri)
    scene.cross_stitch("sheet", "box", ind, w)
    with pytest.raises(ValueError, match="PDRD body"):
        scene.build(quiet=True)


def test_pin_settings_reach_the_record_and_the_mirror(scene):
    holder = scene.add("sheet").pin([0, 1])
    assert holder.set_allow_intersection() is holder
    assert holder.allow_intersection is True
    holder.set_allow_intersection(False)
    assert holder.allow_intersection is False
    holder.set_pin_group_id("rim")
    assert holder.pin_group_id == "rim"
    assert holder._rust.pin_group_id == "rim"
    with pytest.raises(ValueError, match="must not be empty"):
        holder.set_pin_group_id("")
    with pytest.raises(TypeError):
        holder.set_pin_group_id(3)
    assert holder.pin_group_id == "rim" and holder._rust.pin_group_id == "rim"


def test_a_full_tracking_pin_builds_a_rest_shape_schedule(scene):
    sheet = scene.add("sheet")
    pin = sheet.pin()
    pin.move_by([0.0, 0.1, 0.0], t_start=0.0, t_end=1.0)
    pin.pull(1.0).track_rest_shape()
    fixed = scene.build(quiet=True)
    n_vert = len(sheet.get("V"))
    assert fixed._rest_vert_times.tolist() == [0.0, 1.0]
    frames = fixed._rest_vert_anim.reshape(2, n_vert, 3)
    np.testing.assert_allclose(frames[1] - frames[0], [[0.0, 0.1, 0.0]] * n_vert)


def test_tracking_pins_may_split_the_object_between_them(scene):
    sheet = scene.add("sheet")
    n_vert = len(sheet.get("V"))
    half = n_vert // 2
    for part in (range(half), range(half, n_vert)):
        pin = sheet.pin(list(part))
        pin.move_by([0.0, 0.1, 0.0], t_start=0.0, t_end=1.0)
        pin.track_rest_shape()
    fixed = scene.build(quiet=True)
    assert fixed._rest_vert_anim is not None


def test_a_partial_tracking_pin_is_refused(scene):
    sheet = scene.add("sheet")
    pin = sheet.pin([0, 1, 2])
    pin.move_by([0.0, 0.1, 0.0], t_start=0.0, t_end=1.0)
    pin.track_rest_shape()
    with pytest.raises(ValueError, match="hold 3 of its"):
        scene.build(quiet=True)


def test_a_tracking_pin_with_nothing_to_track_is_refused(scene):
    scene.add("sheet").pin().pull(1.0).track_rest_shape()
    with pytest.raises(ValueError, match="carries no operation"):
        scene.build(quiet=True)


def test_turning_tracking_off_again_builds_no_schedule(scene):
    pin = scene.add("sheet").pin([0, 1, 2])
    pin.move_by([0.0, 0.1, 0.0], t_start=0.0, t_end=1.0)
    pin.track_rest_shape().track_rest_shape(False)
    fixed = scene.build(quiet=True)
    assert fixed._rest_vert_anim is None


@pytest.mark.parametrize("hint", [None, -1, 999])
def test_a_torque_without_a_vertex_to_orient_it_is_refused(scene, hint):
    holder = scene.add("sheet").pin([0, 1, 2])
    with pytest.raises(ValueError, match="needs a hint_vertex"):
        holder.torque(magnitude=1.0, axis_component=0, hint_vertex=hint)
    assert holder.operations == []


def test_a_torque_keeps_its_object_local_hint(scene):
    holder = scene.add("sheet").pin([0, 1, 2])
    holder.torque(magnitude=1.0, axis_component=0, hint_vertex=4)
    [op] = holder.operations
    assert op.hint_vertex == 4


def test_a_tet_object_animates_only_what_its_surface_carries(scene):
    mesh = MeshManager(str(SCRATCH_ROOT / "cache"))
    V, F, T = mesh.tet_box(1.0, 1.0, 1.0)
    scene.asset_manager.add.tet("block", V, F, T)
    block = scene.add("block")
    for key in ("young-mod", "poiss-rat", "plasticity", "deformation-damping"):
        with pytest.raises(ValueError, match="tetrahedral object"):
            block.set_param_anim(key, [1.0, 2.0])
    block.set_param_anim("friction", [0.1, 0.4])
    block.set_param_anim("contact-gap", [1e-3, 2e-3])
    assert set(block.param_anim) == {"friction", "contact-gap"}


def test_a_tracked_rest_shape_refuses_plasticity(scene):
    sheet = scene.add("sheet")
    sheet.param.set("plasticity", 0.5)
    pin = sheet.pin()
    pin.move_by([0.0, 0.1, 0.0], t_start=0.0, t_end=1.0)
    pin.track_rest_shape()
    with pytest.raises(ValueError, match="both rewrite the rest shape"):
        scene.build(quiet=True)
