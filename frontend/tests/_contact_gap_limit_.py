# File: frontend/tests/_contact_gap_limit_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# An object's `contact-gap-limit` holds each element's contact gap to a
# fraction of the room its own mesh leaves between neighbors
# (`frontend/_contact_gap_limit_.py`), so a mesh finer than its gap does not
# start the solve with its own neighbors pressing on each other.
#
# Covers, on the functions:
#   * a triangle's size is its smallest altitude and a rod segment's is its
#     length, and a degenerate triangle has none;
#   * the clearance is the smallest size around each of the element's
#     vertices, and stops there;
#   * the bound is the fraction of the clearance less twice the offset, and an
#     element with no room is left unbounded;
#   * the accepted fractions;
#   * ON A GRADED MESH, the gaps the solver derives from the bounded faces (a
#     vertex and an edge take the area-weighted mean of the faces around
#     them, and a pair the mean of its two sides) leave every vertex clear of
#     its mesh neighbors at the largest fraction a limit accepts, and a face
#     sized from its own altitude alone does not.
#
# Covers, through `Scene.build()`:
#   * a fine sheet is lowered triangle by triangle and a coarse one is left
#     exactly as authored, and no limit is the default;
#   * a rod is lowered segment by segment, and a solid on its surface;
#   * the room is measured against the largest offset the object takes;
#   * a static collider, a PDRD body and an object that allows its own
#     intersections are left as authored;
#   * every frame of an animated gap is held, the frames of an object that
#     animates nothing included, and so is a mapped gap;
#   * a limit that is animated, mapped or out of range is refused;
#   * the build names each object it lowered, and the solvability refusal
#     names the limit where a larger gap would not help.

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
    from frontend._contact_gap_ import refuse_unsolvable_gaps
    from frontend._contact_gap_limit_ import (
        LIMIT_RANGE,
        element_clearance,
        resolve_gap_limit,
        validated_fraction,
    )
    from frontend._mesh_ import MeshManager
    from frontend._plot_ import PlotManager
    from frontend._scene_ import Scene, ValidationError
except ImportError:
    pytest.skip(
        "frontend._rust extension not built; run `cargo build --release` first",
        allow_module_level=True,
    )

SCRATCH_ROOT = REPO_ROOT / "frontend" / "tests" / "_scratch_contact_gap_limit"

# A gap no mesh in this file has room for, so the limit decides every element
# it applies to.
WIDE = 1.0
FRACTION = 0.25


# ----- an independent reading of the measure --------------------------------


def point_segment_distance(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def triangle_size(vert, tri):
    """The closest a vertex of `tri` comes to the opposite edge."""
    return min(
        point_segment_distance(vert[tri[k]], vert[tri[(k + 1) % 3]], vert[tri[(k + 2) % 3]])
        for k in range(3)
    )


def expected_clearance(vert, elements):
    """Per element, the smallest size among the elements at its vertices."""
    vert = np.asarray(vert, dtype=np.float64)
    elements = np.asarray(elements, dtype=np.int64)
    if elements.shape[1] == 2:
        size = [float(np.linalg.norm(vert[b] - vert[a])) for a, b in elements]
    else:
        size = [triangle_size(vert, tri) for tri in elements]
    around = {}
    for element, s in zip(elements, size):
        for v in element:
            around[int(v)] = min(around.get(int(v), np.inf), s)
    return np.array([min(around[int(v)] for v in element) for element in elements])


# ----- the functions --------------------------------------------------------


def test_a_triangle_is_as_wide_as_its_smallest_altitude():
    vert = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    # A 3-4-5 triangle: the altitude onto the hypotenuse is 12 / 5.
    assert element_clearance(vert, [[0, 1, 2]]) == pytest.approx([2.4])


def test_an_obtuse_triangle_is_measured_to_the_edge_and_not_to_its_line():
    # The vertex between the two far ones sits 0.1 off their edge, and each
    # far vertex is closest to the near END of the edge opposite it.
    vert = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 0.1, 0.0]])
    assert element_clearance(vert, [[0, 1, 2]]) == pytest.approx([0.1])
    assert triangle_size(vert, [0, 1, 2]) == pytest.approx(0.1)


def test_a_rod_segment_is_as_wide_as_it_is_long():
    vert = np.array([[0.0, 0.0, 0.0], [0.0, 2.5, 0.0]])
    assert element_clearance(vert, [[0, 1]]) == pytest.approx([2.5])


def test_a_degenerate_triangle_has_no_width():
    collinear = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert element_clearance(collinear, [[0, 1, 2]]) == pytest.approx([0.0])
    point = np.zeros((3, 3))
    assert element_clearance(point, [[0, 1, 2]]) == pytest.approx([0.0])


def test_the_clearance_is_the_smallest_size_around_the_elements_vertices():
    # Segments of length 1, 0.1, 1, 1, 1 along a line. The short one lowers
    # the two it touches and no further.
    x = np.cumsum([0.0, 1.0, 0.1, 1.0, 1.0, 1.0])
    vert = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    edges = [[i, i + 1] for i in range(5)]
    assert element_clearance(vert, edges) == pytest.approx([0.1, 0.1, 0.1, 1.0, 1.0])


def test_a_triangle_sharing_only_an_edge_neighbor_keeps_its_own_size():
    # A strip of four triangles, the first one thin. The second and third
    # share a vertex with it and the fourth does not.
    vert = np.array([
        [0.0, 0.0, 0.0], [0.0, 0.05, 0.0],
        [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0], [2.0, 1.0, 0.0],
    ])
    tri = [[0, 2, 1], [1, 2, 3], [2, 4, 3], [3, 4, 5]]
    clearance = element_clearance(vert, tri)
    thin = triangle_size(vert, tri[0])
    assert clearance[:3] == pytest.approx([thin, thin, expected_clearance(vert, tri)[2]])
    assert clearance[3] == pytest.approx(triangle_size(vert, tri[3]))
    assert clearance[3] > 10.0 * thin
    assert clearance == pytest.approx(expected_clearance(vert, tri))


def test_only_triangles_and_segments_are_measured():
    with pytest.raises(ValueError, match="triangles or rod segments"):
        element_clearance(np.zeros((4, 3)), [[0, 1, 2, 3]])


def test_the_bound_is_the_fraction_of_the_room():
    vert = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.5, 0.0]])
    limit = resolve_gap_limit(0.25, vert, [[0, 1], [1, 2]], offset=0.1)
    # Both segments are bounded by the shorter one, 0.5 long.
    assert limit.kind == "segment"
    assert limit.room == pytest.approx([0.3, 0.3])
    assert limit.bound == pytest.approx([0.075, 0.075])
    assert limit.apply([1.0, 0.01]) == pytest.approx([0.075, 0.01])


def test_an_element_no_wider_than_twice_the_offset_is_left_unbounded():
    vert = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 5.0, 0.0], [0.0, 9.0, 0.0]])
    limit = resolve_gap_limit(0.5, vert, [[0, 1], [1, 2], [2, 3]], offset=0.5)
    # The first two reach the short segment, which is exactly two offsets long.
    assert np.isinf(limit.bound[:2]).all()
    assert limit.bound[2] == pytest.approx(0.5 * (4.0 - 1.0))
    assert limit.apply([7.0, 7.0, 7.0]) == pytest.approx([7.0, 7.0, 1.5])
    assert limit.tightest() == 2
    unbounded = resolve_gap_limit(0.5, vert, [[0, 1]], offset=0.5)
    assert unbounded.tightest() is None


@pytest.mark.parametrize("value", [0.0, LIMIT_RANGE[0], 0.25, LIMIT_RANGE[1]])
def test_no_limit_and_the_range_are_accepted(value):
    assert validated_fraction(value, "Sheet") == value


@pytest.mark.parametrize("value", [-0.25, 0.005, 0.51, 1.0, float("nan"), float("inf")])
def test_a_fraction_outside_the_range_is_refused_by_name(value):
    with pytest.raises(ValueError, match="'Sheet' has contact-gap-limit"):
        validated_fraction(value, "Sheet")


def graded_mesh(seed=7, n=14, ratio=1.35, jitter=0.3):
    """A planar triangulation whose triangles span two decades of size."""
    rng = np.random.default_rng(seed)
    axis = np.concatenate([[0.0], np.cumsum(ratio ** np.arange(n - 1))])
    step = np.gradient(axis)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    dx = jitter * step[:, None] * rng.uniform(-0.5, 0.5, x.shape)
    dy = jitter * step[None, :] * rng.uniform(-0.5, 0.5, y.shape)
    vert = np.stack([x + dx, y + dy, np.zeros_like(x)], axis=-1).reshape(-1, 3)
    tri = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            # Alternate the diagonal so vertex valences and shapes vary.
            tri += [[a, b, c], [a, c, d]] if (i + j) % 2 else [[a, b, d], [b, c, d]]
    return vert, np.array(tri)


def neighbor_slack(vert, tri, face_gap, offset):
    """How far the closest pair of mesh neighbors is from engaging.

    Builds the solver's tables from one gap per face: a vertex and an edge
    take the area-weighted mean of the faces around them (`builder.rs`,
    `averaged_vertex_param` and `averaged_edge_param`), and a pair engages
    inside the mean of its two sides' gaps plus the sum of their offsets.
    Negative when some vertex starts engaged with a vertex or an edge of a
    triangle it belongs to.
    """
    a, b, c = (vert[tri[:, k]] for k in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    weight = np.zeros(len(vert))
    vertex_gap = np.zeros(len(vert))
    np.add.at(weight, tri, area[:, None])
    np.add.at(vertex_gap, tri, (area * face_gap)[:, None])
    vertex_gap /= weight
    edge_faces = {}
    for f, t in enumerate(tri):
        for k in range(3):
            edge_faces.setdefault(frozenset((int(t[k]), int(t[(k + 1) % 3]))), []).append(f)
    edge_gap = {
        edge: float(np.dot(area[faces], face_gap[faces]) / area[faces].sum())
        for edge, faces in edge_faces.items()
    }
    slack = []
    for t in tri:
        for k in range(3):
            v, p, q = (int(t[(k + i) % 3]) for i in range(3))
            # The vertex against the opposite edge of its own triangle.
            reach = 0.5 * (vertex_gap[v] + edge_gap[frozenset((p, q))]) + 2.0 * offset
            slack.append(point_segment_distance(vert[v], vert[p], vert[q]) - reach)
            # The vertex against the next one along the triangle.
            reach = 0.5 * (vertex_gap[v] + vertex_gap[p]) + 2.0 * offset
            slack.append(float(np.linalg.norm(vert[v] - vert[p])) - reach)
    return min(slack)


@pytest.mark.parametrize("offset_share", [0.0, 0.2])
def test_no_vertex_starts_engaged_with_a_mesh_neighbor_on_a_graded_mesh(offset_share):
    vert, tri = graded_mesh()
    sizes = np.array([triangle_size(vert, t) for t in tri])
    assert sizes.max() > 30.0 * sizes.min()
    offset = offset_share * sizes.min()
    limit = resolve_gap_limit(LIMIT_RANGE[1], vert, tri, offset)
    face_gap = limit.apply(np.full(len(tri), 1e9))
    assert np.isfinite(face_gap).all()
    assert neighbor_slack(vert, tri, face_gap, offset) > 0.0


def test_a_small_triangle_beside_a_large_one_is_why_the_neighborhood_is_taken():
    # Two triangles meeting at one vertex, a hundred times apart in size. The
    # shared vertex takes the area-weighted mean of both gaps, which is the
    # large triangle's.
    vert = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
        [-0.01, 0.0, 0.0], [0.0, -0.01, 0.0],
    ])
    tri = np.array([[0, 1, 2], [0, 3, 4]])
    sizes = np.array([triangle_size(vert, t) for t in tri])
    # Each face sized from its own altitude: the shared vertex reaches across
    # the small triangle.
    assert neighbor_slack(vert, tri, LIMIT_RANGE[1] * sizes, 0.0) < 0.0
    limit = resolve_gap_limit(LIMIT_RANGE[1], vert, tri, 0.0)
    assert limit.bound == pytest.approx(LIMIT_RANGE[1] * sizes.min() * np.ones(2))
    assert neighbor_slack(vert, tri, limit.bound, 0.0) > 0.0


# ----- through Scene.build() ------------------------------------------------


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


def graded_rod():
    y = np.cumsum([0.0, 0.5, 0.05, 0.5, 0.5, 0.2, 0.5])
    vert = np.stack([np.zeros_like(y), y, np.zeros_like(y)], axis=1)
    edge = np.array([[i, i + 1] for i in range(len(y) - 1)], dtype=np.uint32)
    return vert, edge


@pytest.fixture
def scene():
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        asset = AssetManager()
        mesh = MeshManager(str(SCRATCH_ROOT / "cache"))
        asset.add.tri("cube", *cube())
        asset.add.tri("sheet", *mesh.square(res=6))
        asset.add.tri("coarse", *mesh.square(res=2))
        asset.add.rod("strand", *graded_rod())
        asset.add.tet("block", *mesh.tet_box(1.0, 0.5, 0.25))
        yield Scene("contact_gap_limit", PlotManager(), asset)
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def limited(obj, gap=WIDE, fraction=FRACTION):
    obj.param.set("contact-gap", gap).set("contact-gap-limit", fraction)
    return obj


def bounds_of(obj, elements="F", fraction=FRACTION, offset=None):
    """The bound of each element of `obj`, at the offset it carries."""
    if offset is None:
        offset = float(obj.param.get("contact-offset"))
    room = expected_clearance(obj.vertex(False), obj.get(elements)) - 2.0 * offset
    return fraction * room


def tri_gaps(fixed):
    return np.array(fixed.tri_param["contact-gap"], dtype=np.float64)


def test_no_limit_is_the_default_and_leaves_the_gap_as_authored(scene):
    sheet = scene.add("sheet").at(0.0, 1.0, 0.0)
    sheet.param.set("contact-gap", WIDE)
    assert float(sheet.param.get("contact-gap-limit")) == 0.0
    fixed = scene.build(quiet=True)
    assert (tri_gaps(fixed) == WIDE).all()
    assert "contact-gap-limit" not in {k for k, v in fixed.tri_param.items() if v}


def test_a_fine_sheet_is_lowered_triangle_by_triangle(scene):
    sheet = limited(scene.add("sheet").at(0.0, 1.0, 0.0))
    fixed = scene.build(quiet=True)
    expected = bounds_of(sheet)
    assert (expected < WIDE).all()
    assert tri_gaps(fixed) == pytest.approx(expected, rel=1e-6)


def test_a_scaled_object_is_measured_as_it_enters_the_solve(scene):
    sheet = limited(scene.add("sheet").at(0.0, 1.0, 0.0).scale(0.1))
    fixed = scene.build(quiet=True)
    unscaled = FRACTION * expected_clearance(sheet.get("V"), sheet.get("F"))
    assert tri_gaps(fixed) == pytest.approx(0.1 * unscaled, rel=1e-5)


def test_a_coarse_sheet_keeps_a_gap_its_mesh_has_room_for(scene):
    coarse = scene.add("coarse").at(0.0, 1.0, 0.0)
    gap = 0.5 * float(bounds_of(coarse).min())
    limited(coarse, gap=gap)
    fixed = scene.build(quiet=True)
    assert (tri_gaps(fixed) == np.float64(gap)).all()


def test_a_rod_is_lowered_segment_by_segment(scene):
    strand = limited(scene.add("strand"))
    strand.param.set("contact-offset", 0.0)
    fixed = scene.build(quiet=True)
    expected = bounds_of(strand, "E")
    # The 0.05 segment holds its two neighbors, and the 0.2 one the last two.
    assert expected == pytest.approx(
        FRACTION * np.array([0.05, 0.05, 0.05, 0.2, 0.2, 0.2])
    )
    assert np.array(fixed._rod_param["contact-gap"]) == pytest.approx(expected, rel=1e-6)


def test_a_solid_is_lowered_on_its_surface(scene):
    block = limited(scene.add("block"))
    fixed = scene.build(quiet=True)
    assert tri_gaps(fixed) == pytest.approx(bounds_of(block), rel=1e-6)
    assert not fixed._tet_param.get("contact-gap")


def test_the_room_is_measured_against_the_largest_offset(scene):
    scene.set_param_anim_times([0.0, 1.0])
    strand = limited(scene.add("strand"), fraction=0.5)
    strand.param.set("contact-offset", 0.01)
    strand.set_param_anim("contact-offset", [0.01, 0.02])
    fixed = scene.build(quiet=True)
    expected = bounds_of(strand, "E", fraction=0.5, offset=0.02)
    assert np.array(fixed._rod_param["contact-gap"]) == pytest.approx(expected, rel=1e-6)


def test_segments_with_no_room_keep_the_authored_gap(scene):
    strand = limited(scene.add("strand"))
    strand.param.set("contact-offset", 0.03)
    # Segments inside each other's offsets are a contact-offset violation,
    # which refuses the scene unless the object exempts the pairs it starts
    # with. The limit is not what decides either outcome.
    with pytest.raises(ValidationError, match="too close"):
        scene.build(quiet=True)
    strand.param.set("allow-existing-intersection", 1.0)
    fixed = scene.build(quiet=True)
    gaps = np.array(fixed._rod_param["contact-gap"])
    # The 0.05 segment is shorter than two offsets, and it is the clearance
    # of the two segments it touches.
    assert (gaps[:3] == WIDE).all()
    assert gaps[3:] == pytest.approx(FRACTION * (0.2 - 0.06), rel=1e-6)


def test_a_static_collider_is_left_as_authored(scene):
    floor = limited(scene.add("sheet", "floor"))
    floor.pin()
    fixed = scene.build(quiet=True)
    assert floor.static
    assert (np.array(fixed._static_param["contact-gap"]) == WIDE).all()


def test_a_rigid_body_is_left_as_authored(scene):
    limited(scene.add("cube").as_pdrd())
    fixed = scene.build(quiet=True)
    assert (tri_gaps(fixed) == WIDE).all()


def test_an_object_that_allows_its_own_intersections_is_left_as_authored(scene):
    sheet = limited(scene.add("sheet").at(0.0, 1.0, 0.0))
    sheet.param.set("allow-self-intersection", 1.0)
    fixed = scene.build(quiet=True)
    assert (tri_gaps(fixed) == WIDE).all()


def test_every_frame_of_an_animated_gap_is_held(scene):
    scene.set_param_anim_times([0.0, 1.0, 2.0])
    moving = limited(scene.add("sheet", "moving").at(0.0, 1.0, 0.0))
    bounds = bounds_of(moving)
    low = 0.5 * float(bounds.min())
    moving.set_param_anim("contact-gap", [WIDE, low, 2.0 * float(bounds.max())])
    # Animates nothing, and is still written into every frame of the key.
    still = limited(scene.add("sheet", "still").at(0.0, 1.0, 3.0), fraction=0.1)
    fixed = scene.build(quiet=True)
    frames = [np.array(f) for f in fixed._tri_param_anim["contact-gap"]]
    n = len(bounds)
    assert frames[0][:n] == pytest.approx(bounds, rel=1e-6)
    assert frames[1][:n] == pytest.approx(np.full(n, low), rel=1e-6)
    assert frames[2][:n] == pytest.approx(bounds, rel=1e-6)
    for frame in frames:
        assert frame[n:] == pytest.approx(bounds_of(still, fraction=0.1), rel=1e-6)
    assert tri_gaps(fixed)[:n] == pytest.approx(bounds, rel=1e-6)


def test_a_mapped_gap_is_held_element_by_element(scene):
    sheet = scene.add("sheet").at(0.0, 1.0, 0.0)
    bounds = bounds_of(sheet)
    base = 0.5 * float(bounds.min())
    limited(sheet, gap=base)
    weights = (sheet.get("V")[:, 0] > 0.0).astype(np.float32)
    sheet.set_param_spatial("contact-gap", weights, WIDE)
    fixed = scene.build(quiet=True)
    mapped = sheet.element_param_values("contact-gap", base, 0.0)
    assert mapped.max() == WIDE and mapped.min() == base
    assert tri_gaps(fixed) == pytest.approx(np.minimum(mapped, bounds), rel=1e-6)


@pytest.mark.parametrize("how", ["animated", "mapped"])
def test_a_limit_that_varies_is_refused(scene, how):
    scene.set_param_anim_times([0.0, 1.0])
    sheet = limited(scene.add("sheet").at(0.0, 1.0, 0.0))
    if how == "animated":
        sheet.set_param_anim("contact-gap-limit", [0.1, 0.2])
    else:
        n = len(sheet.get("V"))
        sheet.set_param_spatial("contact-gap-limit", np.ones(n, dtype=np.float32), 0.1)
    with pytest.raises(ValueError, match="'sheet' animates or maps 'contact-gap-limit'"):
        scene.build(quiet=True)


def test_a_limit_out_of_range_is_refused_by_the_name_the_scene_gave(scene):
    sheet = limited(scene.add("sheet").at(0.0, 1.0, 0.0), fraction=0.75)
    sheet._statistics_name = "Curtain"
    with pytest.raises(ValueError, match="'Curtain' has contact-gap-limit 0.75"):
        scene.build(quiet=True)


def test_a_limit_out_of_range_is_refused_on_a_collider_too(scene):
    floor = limited(scene.add("sheet", "floor"), fraction=0.75)
    floor.pin()
    with pytest.raises(ValueError, match="'floor' has contact-gap-limit 0.75"):
        scene.build(quiet=True)


def test_the_build_names_each_object_it_lowered(scene, capsys):
    limited(scene.add("sheet").at(0.0, 1.0, 0.0))
    coarse = scene.add("coarse").at(0.0, 5.0, 0.0)
    limited(coarse, gap=0.5 * float(bounds_of(coarse).min()))
    scene.build()
    lines = [
        line for line in capsys.readouterr().out.splitlines()
        if line.startswith("contact gap limit:")
    ]
    n = len(scene.select("sheet").get("F"))
    assert len(lines) == 1
    assert f"'sheet' lowered on {n} of {n} triangles to 25%" in lines[0]


def test_an_unsolvable_gap_the_limit_set_names_the_limit(scene):
    sheet = limited(scene.add("sheet").at(0.0, 1.0, 0.0), fraction=0.01)
    fixed = scene.build(quiet=True)
    [participant] = fixed.contact_gap_participants
    bound = float(bounds_of(sheet, fraction=0.01).min())
    assert participant.gap == pytest.approx(bound)
    assert participant.limit.startswith("1% of the ")
    assert "between mesh neighbors at triangle " in participant.limit
    step = 2.0 * bound * 4096.0
    with pytest.raises(ValueError) as err:
        refuse_unsolvable_gaps(fixed.contact_gap_participants, step)
    message = str(err.value)
    assert "The contact gap of 'sheet' is" in message
    assert f"'sheet' is held to {participant.limit} by its contact gap limit" in message
    assert "Raise the contact gap" not in message
    assert f"lower dt to at most {bound * 4096.0:.3e}" in message
    refuse_unsolvable_gaps(fixed.contact_gap_participants, 0.5 * bound * 4096.0)


def test_an_authored_gap_below_its_bound_is_not_blamed_on_the_limit(scene):
    coarse = scene.add("coarse").at(0.0, 1.0, 0.0)
    gap = 0.5 * float(bounds_of(coarse).min())
    limited(coarse, gap=gap)
    fixed = scene.build(quiet=True)
    [participant] = fixed.contact_gap_participants
    assert participant.gap == gap
    assert participant.limit == ""
    with pytest.raises(ValueError, match="Raise the contact gap to at least"):
        refuse_unsolvable_gaps(fixed.contact_gap_participants, 2.0 * gap * 4096.0)
