# File: frontend/tests/_contact_gap_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A contact gap too small for the step size is refused when the session is
# built (`frontend/_contact_gap_.py`), instead of ending the run in its first
# frames with `cg failed`. The bound is `step / 2^12`, where `step` is the
# longest step the solver takes: `dt`, capped at one frame, times `playback`.
#
# Covers, on the function:
#   * the bound itself, and that it moves with the step;
#   * two rigid bodies engage at the MEAN of their gaps, and the refusal names
#     both, both numbers and both remedies;
#   * a small gap beside a large one is accepted when their mean is solvable;
#   * a deformable object is judged against itself, a rigid body and a
#     collider are not, and two colliders form no pair;
#   * an invisible wall or sphere engages at its own gap;
#   * an object an allowance takes out of contact is not judged.
#
# Covers, through `Scene.build()` and the session's parameters:
#   * a scene of two rigid bodies with a gap below the bound is refused by the
#     names the scene gave them, and the same scene builds at a smaller `dt`;
#   * `dt` longer than a frame is capped at the frame, and `playback` scales
#     the step;
#   * an animated gap is judged at its smallest value, and a mapped one at its
#     smallest element;
#   * a fully pinned object is a collider, so its own small gap forms no pair
#     with itself;
#   * `disable-contact` turns the check off.

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
    from frontend._contact_gap_ import (
        GapParticipant,
        longest_step,
        refuse_unsolvable_gaps,
        refuse_unsolvable_scene,
        smallest_solvable_gap,
    )
    from frontend._mesh_ import MeshManager
    from frontend._plot_ import PlotManager
    from frontend._scene_ import Scene
    from frontend._session_param_ import ParamManager
except ImportError:
    pytest.skip(
        "frontend._rust extension not built; run `cargo build --release` first",
        allow_module_level=True,
    )

SCRATCH_ROOT = REPO_ROOT / "frontend" / "tests" / "_scratch_contact_gap"

STEP = 0.01
BOUND = STEP / 4096.0
BELOW = 0.5 * BOUND
ABOVE = 2.0 * BOUND


def rigid(label, gap):
    return GapParticipant(label, gap, "rigid")


def solved(label, gap, own_pairs=True, cross_pairs=True):
    return GapParticipant(label, gap, "solved", own_pairs, cross_pairs)


# ----- the function ---------------------------------------------------------


def test_the_bound_is_the_step_over_two_to_the_twelfth():
    assert smallest_solvable_gap(0.01) == 0.01 / 4096.0
    assert smallest_solvable_gap(0.001) == pytest.approx(
        smallest_solvable_gap(0.01) / 10.0
    )


def test_two_rigid_bodies_engage_at_the_mean_of_their_gaps():
    pair = [rigid("A", BELOW), rigid("B", BELOW)]
    with pytest.raises(ValueError) as err:
        refuse_unsolvable_gaps(pair, STEP)
    message = str(err.value)
    assert "between 'A' and 'B'" in message
    assert f"is {BELOW:.3e}" in message
    assert f"Raise the contact gap to at least {BOUND:.3e}" in message
    assert f"lower dt to at most {BELOW * 4096.0:.3e}" in message
    # The same pair at a step short enough for it.
    refuse_unsolvable_gaps(pair, BELOW * 4096.0)


def test_a_small_gap_beside_a_large_one_is_judged_by_their_mean():
    refuse_unsolvable_gaps([rigid("A", BELOW), rigid("B", 3.0 * BOUND)], STEP)
    with pytest.raises(ValueError, match="between 'A' and 'B'"):
        refuse_unsolvable_gaps([rigid("A", BELOW), rigid("B", 1.2 * BOUND)], STEP)


def test_the_pair_is_the_two_smallest_gaps():
    bodies = [rigid("Far", ABOVE), rigid("B", 0.25 * BOUND), rigid("A", 0.125 * BOUND)]
    with pytest.raises(ValueError, match="between 'A' and 'B'"):
        refuse_unsolvable_gaps(bodies, STEP)


def test_a_deformable_object_is_judged_against_itself():
    with pytest.raises(ValueError, match="of 'Sheet' is"):
        refuse_unsolvable_gaps([solved("Sheet", BELOW)], STEP)


def test_a_rigid_body_alone_forms_no_pair():
    refuse_unsolvable_gaps([rigid("Body", BELOW)], STEP)


def test_two_colliders_form_no_pair():
    colliders = [
        GapParticipant("Floor", BELOW, "collider"),
        GapParticipant("Wall", BELOW, "analytic"),
    ]
    refuse_unsolvable_gaps(colliders, STEP)


def test_a_collision_mesh_engages_at_the_mean_with_what_touches_it():
    floor = GapParticipant("Floor", BELOW, "collider")
    with pytest.raises(ValueError, match="between 'Body' and 'Floor'"):
        refuse_unsolvable_gaps([rigid("Body", BELOW), floor], STEP)
    refuse_unsolvable_gaps([rigid("Body", 3.0 * BOUND), floor], STEP)


def test_an_analytic_collider_engages_at_its_own_gap():
    wall = GapParticipant("wall 1", BELOW, "analytic")
    with pytest.raises(ValueError, match="of 'wall 1' is"):
        refuse_unsolvable_gaps([rigid("Body", 100.0 * BOUND), wall], STEP)


def test_an_object_an_allowance_takes_out_of_contact_is_not_judged():
    ghost = solved("Ghost", BELOW, own_pairs=False, cross_pairs=False)
    refuse_unsolvable_gaps([ghost, rigid("Body", BELOW)], STEP)
    # Its own pairs are still judged when only the others are excused.
    with pytest.raises(ValueError, match="of 'Sheet' is"):
        refuse_unsolvable_gaps([solved("Sheet", BELOW, cross_pairs=False)], STEP)


# ----- through Scene.build() and the session's parameters -------------------


def cube(side=0.4):
    h = side / 2
    vert = np.array(
        [
            [-h, -h, -h], [+h, -h, -h], [+h, +h, -h], [-h, +h, -h],
            [-h, -h, +h], [+h, -h, +h], [+h, +h, +h], [-h, +h, +h],
        ],
        dtype=np.float32,
    )
    tri = np.array(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [3, 6, 2], [3, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ],
        dtype=np.uint32,
    )
    return vert, tri


@pytest.fixture
def scene():
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        asset = AssetManager()
        mesh = MeshManager(str(SCRATCH_ROOT / "cache"))
        asset.add.tri("cube", *cube())
        vert, tri = mesh.square(res=4)
        asset.add.tri("sheet", vert, tri)
        yield Scene("contact_gap", PlotManager(), asset)
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def session_param(**values):
    param = ParamManager()
    for key, value in values.items():
        param.set(key.replace("_", "-"), value)
    return param


def two_bodies(scene, gap):
    left = scene.add("cube", "left").at(-1.0, 0.0, 0.0).as_pdrd()
    right = scene.add("cube", "right").at(+1.0, 0.0, 0.0).as_pdrd()
    left.param.set("contact-gap", gap)
    right.param.set("contact-gap", gap)
    return left, right


def test_a_scene_below_the_bound_is_refused_by_name_and_builds_at_a_smaller_dt(scene):
    left, right = two_bodies(scene, BELOW)
    left._statistics_name = "Left Link"
    fixed = scene.build(quiet=True)
    with pytest.raises(ValueError, match="between 'Left Link' and 'right'"):
        refuse_unsolvable_scene(fixed, session_param(dt=STEP, fps=60.0))
    refuse_unsolvable_scene(fixed, session_param(dt=BELOW * 4096.0, fps=60.0))


def test_dt_is_capped_at_one_frame_and_scaled_by_playback(scene):
    assert longest_step(session_param(dt=0.05, fps=60.0)) == pytest.approx(1 / 60)
    assert longest_step(session_param(dt=0.01, fps=60.0)) == 0.01
    assert longest_step(session_param(dt=0.01, fps=60.0, playback=0.5)) == 0.005
    # A gap one frame's step can solve is accepted however long `dt` asks for.
    two_bodies(scene, 1.5 * (1 / 60) / 4096.0)
    fixed = scene.build(quiet=True)
    refuse_unsolvable_scene(fixed, session_param(dt=0.05, fps=60.0))
    with pytest.raises(ValueError, match="between 'left' and 'right'"):
        refuse_unsolvable_scene(fixed, session_param(dt=0.05, fps=30.0))


def test_a_scheduled_step_is_judged_at_its_longest():
    param = session_param(dt=0.001, fps=60.0)
    param.dyn("dt").time(0.5).change(0.01)
    assert longest_step(param) == 0.01


def test_an_animated_gap_is_judged_at_its_smallest_value(scene):
    scene.set_param_anim_times([0.0, 1.0])
    sheet = scene.add("sheet").at(0.0, 1.0, 0.0)
    sheet.param.set("contact-gap", ABOVE)
    sheet.set_param_anim("contact-gap", [ABOVE, BELOW])
    fixed = scene.build(quiet=True)
    with pytest.raises(ValueError, match="of 'sheet' is"):
        refuse_unsolvable_scene(fixed, session_param(dt=STEP, fps=60.0))


def test_a_mapped_gap_is_judged_at_its_smallest_element(scene):
    # The map blends each element from the object's value toward the target
    # by the mean weight of its vertices.
    sheet = scene.add("sheet").at(0.0, 1.0, 0.0)
    sheet.param.set("contact-gap", ABOVE)
    n = len(sheet.get("V"))
    fixed = scene.build(quiet=True)
    refuse_unsolvable_scene(fixed, session_param(dt=STEP, fps=60.0))
    sheet.set_param_spatial("contact-gap", np.ones(n, dtype=np.float32), BELOW)
    fixed = scene.build(quiet=True)
    with pytest.raises(ValueError, match="of 'sheet' is"):
        refuse_unsolvable_scene(fixed, session_param(dt=STEP, fps=60.0))


def test_a_map_nothing_follows_changes_nothing(scene):
    sheet = scene.add("sheet").at(0.0, 1.0, 0.0)
    sheet.param.set("contact-gap", ABOVE)
    n = len(sheet.get("V"))
    sheet.set_param_spatial("contact-gap", np.zeros(n, dtype=np.float32), BELOW)
    fixed = scene.build(quiet=True)
    refuse_unsolvable_scene(fixed, session_param(dt=STEP, fps=60.0))


def test_a_fully_pinned_object_is_a_collider_and_forms_no_pair_alone(scene):
    floor = scene.add("sheet", "floor")
    floor.param.set("contact-gap", BELOW)
    floor.pin()
    fixed = scene.build(quiet=True)
    assert floor.static
    refuse_unsolvable_scene(fixed, session_param(dt=STEP, fps=60.0))


def test_disable_contact_turns_the_check_off(scene):
    two_bodies(scene, BELOW)
    fixed = scene.build(quiet=True)
    param = session_param(dt=STEP, fps=60.0)
    param.set("disable-contact", True)
    refuse_unsolvable_scene(fixed, param)
