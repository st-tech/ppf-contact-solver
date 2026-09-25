# File: frontend/tests/_decoder_cross_stitch_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# How the decoder turns the add-on's stitches into the solver's, and that a
# stitch it cannot place is refused by name rather than dropped.
#
# Covers:
#   * cross_stitch_apply_batch keeps a SHELL or STATIC side's snap-time rows
#     verbatim and places a SOLID side from its recorded points on the
#     tetrahedral surface, one row per payload row;
#   * every way an entry cannot be applied (a SOLID side with no points, a
#     points count that is not the row count, a surface that places no
#     point, no rows, ind and w of different lengths, a stiffness that is not
#     a number) raises a ValueError naming the objects;
#   * a SOLID's own loose-edge stitch (_solid_stitch_rows) becomes 6-column
#     point-to-point rows on the tetrahedral surface, both ends at the world
#     points the Blender rows name, and a malformed or unplaceable one is
#     refused.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from frontend import _rust
    from frontend._decoder_ import SceneDecoder
except ImportError:
    pytest.skip(
        "frontend._rust extension not built; run `cargo build --release` first",
        allow_module_level=True,
    )


# A unit cube's surface, as the tetrahedral surface a SOLID side carries.
CUBE_V = np.array(
    [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ],
    dtype=np.float64,
)
CUBE_F = np.array(
    [
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3],
    ],
    dtype=np.int64,
)
SHEET_V = np.array(
    [[0, 0, 1.2], [1, 0, 1.2], [1, 1, 1.2], [0, 1, 1.2]], dtype=np.float64
)


def _obj_info():
    return {
        "u-cube": {
            "type": "SOLID", "vert": CUBE_V, "V": CUBE_V, "F": CUBE_F,
            "name": "Cube",
        },
        "u-sheet": {
            "type": "SHELL", "vert": SHEET_V, "V": SHEET_V,
            "F": np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
            "name": "Sheet",
        },
    }


def _entry(n=2, **overrides):
    # Source slots [0..2] name a sheet vertex; target slots [3..5] are the
    # snap-time surface point, which a SOLID target replaces from its points.
    ind = np.array([[k, k, k, 0, 1, 2] for k in range(n)], dtype=np.int64)
    w = np.array([[1, 0, 0, 0.2, 0.3, 0.5]] * n, dtype=np.float32)
    entry = {
        "source_uuid": "u-sheet",
        "target_uuid": "u-cube",
        "ind": ind,
        "w": w,
        "target_points": SHEET_V[:n] - np.array([0.0, 0.0, 0.2]),
        "stitch_stiffness": 2.5,
    }
    entry.update(overrides)
    return entry


def _apply(entries):
    out: list = []
    count = _rust.cross_stitch_apply_batch(entries, _obj_info(), out, False)
    return count, out


def test_a_shell_side_is_kept_and_a_solid_side_is_placed():
    count, out = _apply([_entry()])
    assert count == 1 and len(out) == 1
    row = out[0]
    assert (row["source_name"], row["target_name"]) == ("u-sheet", "u-cube")
    assert row["stitch_stiffness"] == 2.5
    ind = np.asarray(row["ind"])
    w = np.asarray(row["w"])
    assert ind.shape == w.shape == (2, 6)
    # The SHELL source is its snap-time row, verbatim.
    assert ind[:, :3].tolist() == [[0, 0, 0], [1, 1, 1]]
    assert np.allclose(w[:, :3], [[1, 0, 0]] * 2)
    # The SOLID target lands on the cube's top at the recorded point.
    for r in range(2):
        placed = w[r, 3:] @ CUBE_V[ind[r, 3:]]
        assert np.allclose(placed, SHEET_V[r] - [0.0, 0.0, 0.2], atol=1e-6)


def test_a_solid_source_is_placed_and_a_static_target_is_kept():
    info = _obj_info()
    info["u-sheet"]["type"] = "STATIC"
    points = CUBE_V[[6, 7]] - np.array([0.0, 0.0, 0.0])
    entry = _entry(
        source_uuid="u-cube",
        target_uuid="u-sheet",
        source_points=points,
    )
    del entry["target_points"]
    out: list = []
    assert _rust.cross_stitch_apply_batch([entry], info, out, False) == 1
    ind = np.asarray(out[0]["ind"])
    w = np.asarray(out[0]["w"])
    # The STATIC target keeps its snap-time slots.
    assert ind[:, 3:].tolist() == [[0, 1, 2], [0, 1, 2]]
    assert np.allclose(w[:, 3:], [[0.2, 0.3, 0.5]] * 2)
    # The SOLID source lands at its recorded points on the cube.
    for r in range(2):
        assert np.allclose(w[r, :3] @ CUBE_V[ind[r, :3]], points[r], atol=1e-6)


@pytest.mark.parametrize(
    "overrides, needle",
    [
        ({"target_points": None}, "carries no target_points"),
        ({"target_points": SHEET_V[:3]}, "carries 3 target_points for 2 rows"),
        ({"ind": np.zeros((0, 6), np.int64), "w": np.zeros((0, 6), np.float32)},
         "has 0 index rows"),
        ({"w": np.array([[1, 0, 0, 0.2, 0.3, 0.5]], np.float32)},
         "has 2 index rows and 1 weight rows"),
        ({"stitch_stiffness": "stiff"}, "stitch_stiffness that is not a number"),
    ],
)
def test_an_entry_that_cannot_be_applied_is_refused_by_name(overrides, needle):
    entry = _entry(**overrides)
    if entry.get("target_points") is None:
        del entry["target_points"]
    with pytest.raises(ValueError) as err:
        _apply([_entry(), entry])
    message = str(err.value)
    assert needle in message
    assert "Cube" in message


def test_a_surface_that_places_no_point_is_refused():
    info = _obj_info()
    # Every face degenerate: the projection places nothing.
    info["u-cube"]["F"] = np.array([[0, 1, 1], [2, 3, 3]], dtype=np.int64)
    out: list = []
    with pytest.raises(ValueError, match="found no triangle.*'Cube'"):
        _rust.cross_stitch_apply_batch([_entry()], info, out, False)
    assert out == []


def test_an_unknown_endpoint_is_refused():
    with pytest.raises(Exception, match="u-gone"):
        _apply([_entry(target_uuid="u-gone")])


def test_a_loose_edge_stitch_becomes_point_to_point_rows():
    # A Blender mesh whose vertices sit on the cube's surface but are
    # numbered unlike the tetrahedral vertices: rows naming Blender indices
    # must be placed by position, not reused as tet indices.
    blender = np.array(
        [[0.5, 0.5, 1.0], [1.0, 0.25, 0.5], [1.0, 0.75, 0.5], [1.0, 0.5, 1.0]],
        dtype=np.float64,
    )
    ind = np.array([[0, 1, 2, 3]], dtype=np.int64)
    w = np.array([[1.0, 0.5, 0.25, 0.25]], dtype=np.float64)
    rows_ind, rows_w = SceneDecoder._solid_stitch_rows(
        "Cube", (ind, w), blender, CUBE_V, CUBE_F
    )
    assert rows_ind.shape == rows_w.shape == (1, 6)
    assert rows_ind.dtype == np.int64 and rows_w.dtype == np.float32
    source = rows_w[0, :3] @ CUBE_V[rows_ind[0, :3]]
    target = rows_w[0, 3:] @ CUBE_V[rows_ind[0, 3:]]
    assert np.allclose(source, blender[0], atol=1e-6)
    expected = 0.5 * blender[1] + 0.25 * blender[2] + 0.25 * blender[3]
    assert np.allclose(target, expected, atol=1e-6)


@pytest.mark.parametrize(
    "ind, w, needle",
    [
        (np.zeros((1, 3), np.int64), np.zeros((1, 3)), "4-column rows"),
        (np.zeros((0, 4), np.int64), np.zeros((0, 4)), "no rows"),
        (np.array([[0, 1, 2, 9]]), np.ones((1, 4)), "names vertex 9"),
    ],
)
def test_a_malformed_loose_edge_stitch_is_refused(ind, w, needle):
    with pytest.raises(ValueError, match=needle):
        SceneDecoder._solid_stitch_rows("Cube", (ind, w), CUBE_V, CUBE_V, CUBE_F)


def test_an_unplaceable_loose_edge_stitch_is_refused():
    degenerate = np.array([[0, 1, 1], [2, 3, 3]], dtype=np.int64)
    with pytest.raises(ValueError, match="found no triangle"):
        SceneDecoder._solid_stitch_rows(
            "Cube",
            (np.array([[0, 1, 2, 3]]), np.array([[1.0, 0.4, 0.3, 0.3]])),
            CUBE_V, CUBE_V, degenerate,
        )
