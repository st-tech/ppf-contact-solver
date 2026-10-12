# File: frontend/tests/_pdrd_mass_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The mass of a PDRD body is its density times the volume its surface
# encloses, read off the surface triangles (`frontend/_pdrd_mass_.py`). That
# reading is valid for a closed, consistently wound surface and for no other,
# and it does not depend on which way the surface faces or on how large the
# body is.
#
# Covers, on the function:
#   * a cube's volume and second moment against their closed forms;
#   * the same surface wound inside-out gives the same two numbers;
#   * no size is refused: the two numbers scale as a^3 and a^5 from a
#     nanometer to a kilometer;
#   * an open surface, a face wound against its neighbors, and a flat
#     two-sided surface are each refused, by name;
#   * a closed part wound against the rest is a cavity when it lies inside
#     them, whichever way the body as a whole faces, and is refused when it
#     does not;
#   * a surface stored with a vertex per face corner is closed when its
#     triangles meet;
#   * small parts far from the origin keep their volume, and a flat fin beside
#     a solid part changes nothing.
#
# Covers, through `Scene.build()`:
#   * an inside-out body builds and carries the per-vertex mass and volume of
#     the same body wound outward;
#   * a body far below a cubic millimeter builds, with the mass its density
#     and size give;
#   * an open body is refused, and the refusal names the Blender object when
#     the scene was keyed by UUID.

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
    from frontend._pdrd_mass_ import enclosed_moments
    from frontend._plot_ import PlotManager
    from frontend._scene_ import Scene
except ImportError:
    pytest.skip(
        "frontend._rust extension not built; run `cargo build --release` first",
        allow_module_level=True,
    )

SCRATCH_ROOT = REPO_ROOT / "frontend" / "tests" / "_scratch_pdrd_mass"

# Body row layout of `Scene.build()`'s PDRD table (`PDRD_BODY_ROW_LEN`).
ROW_LEN = 23
ROW_VOLUME = 2
ROW_MASS_PER_VERTEX = 15


def cube(side: float = 1.0, at=(0.0, 0.0, 0.0)):
    """A cube of the given side centered at `at`, wound outward."""
    h = side / 2
    vert = np.array(
        [
            [-h, -h, -h], [+h, -h, -h], [+h, +h, -h], [-h, +h, -h],
            [-h, -h, +h], [+h, -h, +h], [+h, +h, +h], [-h, +h, +h],
        ],
        dtype=np.float64,
    ) + np.asarray(at, dtype=np.float64)
    tri = np.array(
        [
            [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4], [3, 6, 2], [3, 7, 6],
            [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
        ],
        dtype=np.int64,
    )
    return vert, tri


def inside_out(mesh):
    vert, tri = mesh
    return vert, tri[:, ::-1]


def join(*meshes):
    verts, tris, offset = [], [], 0
    for vert, tri in meshes:
        verts.append(vert)
        tris.append(tri + offset)
        offset += len(vert)
    return np.vstack(verts), np.vstack(tris)


def moments(mesh, origin=None):
    vert, tri = mesh
    origin = vert.mean(axis=0) if origin is None else np.asarray(origin, float)
    return enclosed_moments("body", vert, tri, origin)


# ----- the function ---------------------------------------------------------


def test_a_cube_has_its_volume_and_second_moment():
    # For a cube of side a about its center: V = a^3, and the integral of
    # |r|^2 is 3 * a^5 / 12 = a^5 / 4.
    volume, second = moments(cube(2.0))
    assert volume == pytest.approx(8.0, rel=1e-14)
    assert second == pytest.approx(8.0, rel=1e-14)


def test_an_inside_out_surface_gives_the_same_moments():
    assert moments(inside_out(cube(2.0))) == moments(cube(2.0))


@pytest.mark.parametrize("side", [1e-9, 1e-6, 4e-4, 1e-3, 1.0, 1e3])
def test_no_size_is_refused(side):
    volume, second = moments(cube(side))
    assert volume == pytest.approx(side**3, rel=1e-12)
    assert second == pytest.approx(side**5 / 4.0, rel=1e-12)


def test_an_open_surface_is_refused():
    vert, tri = cube()
    with pytest.raises(ValueError, match="'body': its surface is open"):
        moments((vert, tri[:-1]))


def test_a_face_wound_against_its_neighbors_is_refused():
    vert, tri = cube()
    tri = tri.copy()
    tri[0] = tri[0][::-1]
    with pytest.raises(ValueError, match="not wound consistently"):
        moments((vert, tri))


def test_a_flat_two_sided_surface_is_refused():
    vert = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    tri = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]], dtype=np.int64)
    with pytest.raises(ValueError, match="encloses no volume"):
        moments((vert, tri))


def test_a_cavity_removes_its_volume_whichever_way_the_body_faces():
    outer, inner = cube(1.0), cube(0.5)
    hollow = join(outer, inside_out(inner))
    volume, second = moments(hollow, origin=(0.0, 0.0, 0.0))
    assert volume == pytest.approx(1.0 - 0.125, rel=1e-14)
    assert second == pytest.approx((1.0 - 0.5**5) / 4.0, rel=1e-14)
    reversed_hollow = join(inside_out(outer), inner)
    assert moments(reversed_hollow, origin=(0.0, 0.0, 0.0)) == (volume, second)


def test_a_part_wound_against_the_rest_outside_them_is_refused():
    big, small = cube(1.0), cube(0.5, at=(3.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="1 of its 2 closed parts is wound against"):
        moments(join(big, inside_out(small)))
    # Wound the other way round the majority changes, and the answer does not.
    with pytest.raises(ValueError, match="1 of its 2 closed parts is wound against"):
        moments(join(inside_out(big), small))


def test_parts_whose_volumes_cancel_are_refused_as_opposed_parts():
    left, right = cube(1.0), cube(1.0, at=(3.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="wound against each other"):
        moments(join(left, inside_out(right)))


def test_two_parts_wound_alike_add_whichever_way_they_face():
    left, right = cube(1.0), cube(1.0, at=(3.0, 0.0, 0.0))
    volume, second = moments(join(left, right))
    assert volume == pytest.approx(2.0, rel=1e-14)
    # Each cube: its own a^5 / 4 plus V * d^2 for its 1.5 offset.
    assert second == pytest.approx(2.0 * (0.25 + 1.5**2), rel=1e-14)
    both_reversed = join(inside_out(left), inside_out(right))
    assert moments(both_reversed) == (volume, second)


def test_a_surface_with_a_vertex_per_corner_is_closed():
    vert, tri = cube()
    soup_vert = vert[tri].reshape(-1, 3)
    soup_tri = np.arange(len(soup_vert)).reshape(-1, 3)
    assert moments((soup_vert, soup_tri)) == pytest.approx(moments((vert, tri)))


def test_small_parts_far_from_the_origin_keep_their_volume():
    rng = np.random.default_rng(0)
    side, count = 1e-3, 200
    scattered = join(
        *[cube(side, at=tuple(rng.uniform(-5.0, 5.0, 3))) for _ in range(count)]
    )
    volume, _ = moments(scattered)
    assert volume == pytest.approx(count * side**3, rel=1e-9)


def test_a_flat_fin_beside_a_solid_part_changes_nothing():
    fin_vert = np.array(
        [[2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0]], dtype=np.float64
    )
    fin_tri = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]], dtype=np.int64)
    origin = (0.0, 0.0, 0.0)
    with_fin = moments(join(cube(), (fin_vert, fin_tri)), origin=origin)
    assert with_fin == pytest.approx(moments(cube(), origin=origin), rel=1e-14)


# ----- through Scene.build() ------------------------------------------------


@pytest.fixture
def asset():
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        yield AssetManager()
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def add_tri(asset, name, mesh):
    vert, tri = mesh
    asset.add.tri(name, vert.astype(np.float32), tri.astype(np.uint32))


def body_rows(fixed):
    return np.asarray(fixed._pdrd_body_rows, dtype=np.float64).reshape(-1, ROW_LEN)


def test_an_inside_out_body_builds_with_the_mass_of_its_twin(asset):
    add_tri(asset, "outward", cube(0.4))
    add_tri(asset, "inward", inside_out(cube(0.4)))
    scene = Scene("pdrd_mass_twin", PlotManager(), asset)
    scene.add("outward").at(-1.0, 0.0, 0.0).as_pdrd()
    scene.add("inward").at(+1.0, 0.0, 0.0).as_pdrd()
    rows = body_rows(scene.build(quiet=True))
    assert rows.shape[0] == 2
    assert rows[0, ROW_VOLUME] == pytest.approx(0.4**3, rel=1e-5)
    assert rows[1, ROW_VOLUME] == pytest.approx(rows[0, ROW_VOLUME], rel=1e-5)
    assert rows[0, ROW_MASS_PER_VERTEX] > 0.0
    assert rows[1, ROW_MASS_PER_VERTEX] == pytest.approx(
        rows[0, ROW_MASS_PER_VERTEX], rel=1e-5
    )


def test_a_body_far_below_a_cubic_millimeter_builds(asset):
    # 0.4 mm on a side is 6.4e-11 m^3. A cube's solid inertia is a third of
    # its eight corners carrying the same mass, so each corner gets
    # density * a^3 / 24, at any size.
    side = 4e-4
    add_tri(asset, "speck", cube(side))
    scene = Scene("pdrd_mass_speck", PlotManager(), asset)
    speck = scene.add("speck").as_pdrd()
    density = float(speck.param.get("density"))
    rows = body_rows(scene.build(quiet=True))
    assert rows[0, ROW_VOLUME] == pytest.approx(side**3, rel=1e-5)
    assert rows[0, ROW_MASS_PER_VERTEX] == pytest.approx(
        density * side**3 / 24.0, rel=1e-5
    )


def test_an_open_body_is_refused_by_its_blender_name(asset):
    vert, tri = cube(0.4)
    add_tri(asset, "bowl", (vert, tri[:-2]))
    scene = Scene("pdrd_mass_open", PlotManager(), asset)
    bowl = scene.add("bowl", "5f0c1d0e-uuid").as_pdrd()
    bowl._statistics_name = "Bowl"
    with pytest.raises(ValueError, match="PDRD body 'Bowl': its surface is open"):
        scene.build(quiet=True)
