# File: frontend/tests/_preconditioner_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The `precond` parameter is checked when the session is built
# (`frontend/_preconditioner_.py`): a name that is not a preconditioner is
# refused instead of being run as another one, and the multigrid is refused
# in a scene that holds a rigid body instead of being stopped by the solver
# after the scene has been exported.
#
# Covers:
#   * the three names the parameter takes, and that the registry's default is
#     one of them and its description names every one;
#   * each name is accepted on a scene with no rigid body;
#   * a misspelled name is refused, and the refusal lists the choices;
#   * the multigrid beside rigid bodies is refused by the names the scene gave
#     them with both remedies, and the other two preconditioners build the
#     same scene;
#   * a static collider is not a rigid body, so it does not refuse the
#     multigrid;
#   * a scene with a Cosserat rod takes block-Jacobi only, and the refusal of
#     the other two names the segment count;
#   * a preconditioner scheduled to change mid-run is judged at every value.

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
    from frontend._preconditioner_ import (
        PRECONDITIONERS,
        refuse_unsolvable_preconditioner,
    )
    from frontend._scene_ import Scene
    from frontend._session_param_ import ParamManager
except ImportError:
    pytest.skip(
        "frontend._rust extension not built; run `cargo build --release` first",
        allow_module_level=True,
    )

SCRATCH_ROOT = REPO_ROOT / "frontend" / "tests" / "_scratch_preconditioner"


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
        asset.add.tri("sheet", vert[:, :3], tri)
        # A straight strand of eight segments along x.
        points = np.zeros((9, 3), dtype=np.float32)
        points[:, 0] = np.linspace(0.0, 0.8, 9)
        edges = np.array([[i, i + 1] for i in range(8)], dtype=np.uint32)
        asset.add.rod("strand", points, edges)
        yield Scene("preconditioner", PlotManager(), asset)
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def session_param(precond):
    return ParamManager().set("precond", precond)


def test_the_parameter_takes_three_names_and_the_registry_agrees():
    assert PRECONDITIONERS == ("block-jacobi", "schwarz", "multigrid")
    param = ParamManager()
    assert param.get("precond") == "block-jacobi"
    _, description = param._param.get_desc("precond")
    for name in PRECONDITIONERS:
        assert name in description, f"the registry does not describe {name!r}"


def test_every_name_is_accepted_without_a_rigid_body(scene):
    scene.add("sheet", "drape").at(0.0, 1.0, 0.0)
    fixed = scene.build(quiet=True)
    for name in PRECONDITIONERS:
        refuse_unsolvable_preconditioner(fixed, session_param(name))


def test_a_misspelled_name_is_refused_with_the_choices(scene):
    scene.add("sheet", "drape")
    fixed = scene.build(quiet=True)
    with pytest.raises(ValueError) as err:
        refuse_unsolvable_preconditioner(fixed, session_param("multi-grid"))
    message = str(err.value)
    assert "'multi-grid'" in message
    for name in PRECONDITIONERS:
        assert repr(name) in message


def test_the_multigrid_beside_rigid_bodies_is_refused_by_name(scene):
    # Clear of both cubes along the axis the sheet has no extent in.
    scene.add("sheet", "drape").at(0.0, 0.0, 3.0)
    left = scene.add("cube", "left").at(-1.0, 0.0, 0.0).as_pdrd()
    scene.add("cube", "right").at(+1.0, 0.0, 0.0).as_pdrd()
    left._statistics_name = "Left Link"
    fixed = scene.build(quiet=True)
    with pytest.raises(ValueError) as err:
        refuse_unsolvable_preconditioner(fixed, session_param("multigrid"))
    message = str(err.value)
    assert "this scene has 2: 'Left Link', 'right'" in message
    assert "'drape'" not in message
    assert "'block-jacobi' or 'schwarz'" in message
    assert "make those objects deformable" in message
    # The scene itself is sound: the other two preconditioners take it.
    refuse_unsolvable_preconditioner(fixed, session_param("block-jacobi"))
    refuse_unsolvable_preconditioner(fixed, session_param("schwarz"))


def test_a_static_collider_does_not_refuse_the_multigrid(scene):
    scene.add("sheet", "drape").at(0.0, 1.0, 0.0)
    scene.add("cube", "floor").at(0.0, -1.0, 0.0).pin()
    fixed = scene.build(quiet=True)
    roles = {p.label: p.role for p in fixed.contact_gap_participants}
    assert roles["floor"] == "collider", roles
    refuse_unsolvable_preconditioner(fixed, session_param("multigrid"))


def test_a_cosserat_rod_takes_block_jacobi_only(scene):
    scene.add("strand", "rope").at(0.0, 2.0, 0.0).as_cosserat()
    fixed = scene.build(quiet=True)
    refuse_unsolvable_preconditioner(fixed, session_param("block-jacobi"))
    for name in ("schwarz", "multigrid"):
        with pytest.raises(ValueError) as err:
            refuse_unsolvable_preconditioner(fixed, session_param(name))
        message = str(err.value)
        assert repr(name) in message
        assert "this scene has 8 Cosserat segments" in message
        assert "set precond to 'block-jacobi'" in message


def test_a_plain_rod_does_not_refuse_the_multigrid(scene):
    scene.add("strand", "rope").at(0.0, 2.0, 0.0)
    fixed = scene.build(quiet=True)
    refuse_unsolvable_preconditioner(fixed, session_param("multigrid"))


def test_a_scheduled_preconditioner_is_judged_at_every_value(scene):
    scene.add("cube", "body").as_pdrd()
    fixed = scene.build(quiet=True)
    param = session_param("block-jacobi")
    refuse_unsolvable_preconditioner(fixed, param)
    param.dyn("precond").time(0.5).change("multigrid")
    with pytest.raises(ValueError, match="this scene has 1: 'body'"):
        refuse_unsolvable_preconditioner(fixed, param)
