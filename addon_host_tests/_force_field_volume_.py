# File: addon_host_tests/_force_field_volume_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Host-side gates for the arithmetic of a Fluid Flow's volume in the add-on
# (`blender_addon/core/force_field.py`), the parts that touch no Blender data:
#
#   * the row-by-row brick cutter must produce `field_eval.fluid_bricks`'s
#     table and bricks value for value, since the solver and the reference
#     evaluator both read that layout;
#   * de-staggering must put a face-sampled (MAC) grid's values at the cell
#     centers exactly for a linear field;
#   * a large byte string must travel as `bytes` chunks of at most the
#     frontend's chunk size, and join back to the same bytes;
#   * a volume's index-to-world map splits into a rotation and per-axis
#     cells, and a shear or a mirror is refused;
#   * three constants the add-on mirrors from elsewhere stay equal to their
#     sources: the chunk size (frontend/_cbor_bridge_.py), the transport cap
#     (the server's MAX_PAYLOAD_BYTES) and the per-axis cell limit (the
#     solver's FLUID_MAX_RESOLUTION).
#
# What needs Blender (reading a baked domain or a Volume, the refusals) is
# measured headless in Blender and held by the rig's force-field scenarios.

from __future__ import annotations

import re
import zlib

from pathlib import Path

import numpy as np
import pytest

from conftest import load_addon_module

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ff():
    return load_addon_module("core.force_field")


@pytest.fixture(scope="module")
def fe():
    return load_addon_module("core.field_eval")


def _rng(*key):
    # A generator seeded by the case itself, the same on every run. Not by
    # hash(): a str's hash is salted per process (PYTHONHASHSEED), so a key
    # holding one would draw different data each run.
    return np.random.default_rng(zlib.crc32(repr(key).encode()))


def _sparse_frame(rng, n, density: bool, fill: float):
    u = rng.normal(size=tuple(n) + (3,))
    u[rng.random(tuple(n)) > fill] = 0.0
    rho = None
    if density:
        rho = rng.random(tuple(n))
        rho[rng.random(tuple(n)) > fill] = 0.0
    return u, rho


def _cut(ff, u, rho):
    cutter = ff._BrickCutter(u.shape[:3], "test")
    for y0, dy, z0, dz in cutter.rows():
        r = None if rho is None else rho[:, y0:y0 + dy, z0:z0 + dz]
        cutter.add(u[:, y0:y0 + dy, z0:z0 + dz], r)
    return cutter.finish()


@pytest.mark.parametrize("n", [(1, 1, 1), (8, 8, 8), (9, 7, 17), (16, 5, 8), (20, 3, 1),
                               (3, 24, 9)])
@pytest.mark.parametrize("density", [False, True])
@pytest.mark.parametrize("fill", [0.0, 0.02, 0.5, 1.0])
def test_the_row_cutter_is_fluid_bricks_value_for_value(ff, fe, n, density, fill):
    rng = _rng(n, density, fill)
    u, rho = _sparse_frame(rng, n, density, fill)
    want_table, want_bricks = fe.fluid_bricks(u, rho)
    table, bricks = _cut(ff, u, rho)
    assert table.dtype == np.uint32 and bricks.dtype == np.float32
    assert np.array_equal(table, want_table)
    assert bricks.shape == want_bricks.shape
    assert np.array_equal(bricks, want_bricks)


def _cut_sparse(ff, u, rho, indices):
    cutter = ff._BrickCutter(u.shape[:3], "test", sparse=True)
    for index in indices:
        (x0, y0, z0), (dx, dy, dz) = cutter.brick(index)
        r = None if rho is None else rho[x0:x0 + dx, y0:y0 + dy, z0:z0 + dz]
        cutter.add_brick(index, u[x0:x0 + dx, y0:y0 + dy, z0:z0 + dz], r)
    return cutter.finish()


@pytest.mark.parametrize("n", [(1, 1, 1), (9, 7, 17), (16, 5, 8), (3, 24, 9)])
@pytest.mark.parametrize("density", [False, True])
@pytest.mark.parametrize("fill", [0.0, 0.02, 0.5])
def test_the_sparse_cutter_is_fluid_bricks_given_any_superset(ff, fe, n, density, fill):
    # Given every brick that holds anything, and some that hold nothing, the
    # brick-by-brick cut is the row cut.
    rng = _rng(n, density, fill, "sparse")
    u, rho = _sparse_frame(rng, n, density, fill)
    want_table, want_bricks = fe.fluid_bricks(u, rho)
    total = len(want_table)
    held = np.nonzero(want_table != fe.FLUID_EMPTY)[0]
    extra = rng.choice(total, size=min(total, 3), replace=False)
    table, bricks = _cut_sparse(ff, u, rho, sorted(set(held.tolist()) | set(extra.tolist())))
    assert np.array_equal(table, want_table)
    assert np.array_equal(bricks, want_bricks)


def test_a_sparse_cut_takes_bricks_in_order_and_no_rows(ff):
    cutter = ff._BrickCutter((16, 16, 16), "test", sparse=True)
    cutter.add_brick(3, np.zeros((8, 8, 8, 3)))
    with pytest.raises(ValueError, match="after brick 3"):
        cutter.add_brick(2, np.zeros((8, 8, 8, 3)))
    with pytest.raises(ValueError, match="sparse cut"):
        cutter.add(np.zeros((16, 8, 8, 3)))
    rows = ff._BrickCutter((16, 16, 16), "test")
    with pytest.raises(ValueError, match="cut by rows"):
        rows.add_brick(0, np.zeros((8, 8, 8, 3)))


class _Value:
    def __init__(self, lo, hi):
        self.min, self.max = tuple(lo), tuple(hi)


class _Grid:
    """The two things `_touched_bricks` asks of an openvdb grid."""

    def __init__(self, values, background=(0.0, 0.0, 0.0)):
        self.values = values
        self.background = background

    def citerOnValues(self):  # noqa: N802 - openvdb's own spelling
        return iter(self.values)


def test_touched_bricks_follow_voxels_tiles_and_stagger_offsets(ff):
    lo, n = (10, 20, 30), (40, 24, 16)  # 5 x 3 x 2 bricks
    nbx, nby = 5, 3

    def at(bx, by, bz):
        return bx + nbx * (by + nby * bz)

    voxel = _Grid([_Value((10 + 17, 20 + 8, 30 + 0), (10 + 17, 20 + 8, 30 + 0))])
    assert ff._touched_bricks([voxel], [((0, 0, 0),)], lo, n) == [at(2, 1, 0)]
    # A face at the low side of its brick is also read by the voxel below.
    face = _Grid([_Value((10 + 16, 20, 30), (10 + 16, 20, 30))])
    got = ff._touched_bricks([face], [((0, 0, 0), (-1, 0, 0))], lo, n)
    assert got == [at(1, 0, 0), at(2, 0, 0)]
    # A tile spans its bricks; one reaching past the box is clipped to it.
    tile = _Grid([_Value((10 + 8, 20 + 0, 30 + 8), (10 + 100, 20 + 7, 30 + 15))])
    assert ff._touched_bricks([tile], [((0, 0, 0),)], lo, n) == [at(1, 0, 1), at(2, 0, 1),
                                                                 at(3, 0, 1), at(4, 0, 1)]
    # A value outside the box (a density voxel past the velocity's) reaches
    # nothing.
    outside = _Grid([_Value((0, 0, 0), (0, 0, 0))])
    assert ff._touched_bricks([outside], [((0, 0, 0),)], lo, n) == []
    # A background that is not zero reads everywhere, so no brick list is
    # given.
    assert ff._touched_bricks([_Grid([], background=(0.0, 1.0, 0.0))], [((0, 0, 0),)],
                              lo, n) is None


def test_an_empty_frame_cuts_to_nothing(ff):
    cutter = ff._BrickCutter((0, 0, 0), "test")
    assert list(cutter.rows()) == []
    table, bricks = cutter.finish()
    assert table.shape == (0,) and bricks.shape == (0, 512, 4)


def test_a_value_that_is_not_finite_is_refused(ff):
    u = np.zeros((4, 4, 4, 3))
    u[1, 2, 3, 0] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        _cut(ff, u, None)


def test_a_row_of_the_wrong_shape_is_refused(ff):
    cutter = ff._BrickCutter((9, 9, 9), "test")
    with pytest.raises(ValueError, match="brick row"):
        cutter.add(np.zeros((9, 9, 9, 3)))


def test_an_unfinished_cut_is_refused(ff):
    cutter = ff._BrickCutter((9, 9, 9), "test")
    cutter.add(np.zeros((9, 8, 8, 3)))
    with pytest.raises(ValueError, match="rows were cut"):
        cutter.finish()


def test_destaggering_puts_a_linear_field_at_the_cell_centers(ff):
    # Component c of voxel i is sampled on the face at i - 1/2 along c, so a
    # field linear in position reads, after de-staggering, its value at the
    # voxel's center exactly.
    a = np.array([0.7, -1.3, 2.1])
    b = np.array([[0.2, 0.0, -0.4], [1.1, 0.3, 0.0], [0.0, -0.6, 0.9]])
    shape = (5, 4, 6)
    idx = np.stack(np.meshgrid(*[np.arange(k + 1) for k in shape], indexing="ij"), axis=-1)
    faces = np.zeros(idx.shape[:3] + (3,))
    for c in range(3):
        pos = idx.astype(np.float64)
        pos[..., c] -= 0.5
        faces[..., c] = a[c] + pos @ b[c]
    centers = np.stack(np.meshgrid(*[np.arange(k) for k in shape], indexing="ij"),
                       axis=-1).astype(np.float64)
    want = a + centers @ b.T
    got = ff._destaggered(faces, (True, True, True))
    assert got.shape == shape + (3,)
    assert np.allclose(got, want, atol=1e-12)


def test_destaggering_leaves_a_cell_centered_component_alone(ff):
    rng = np.random.default_rng(3)
    block = rng.normal(size=(4, 5, 6, 3))
    got = ff._destaggered(block, (False, True, False))
    assert np.array_equal(got[..., 0], block[:-1, :-1, :-1, 0])
    assert np.array_equal(got[..., 2], block[:-1, :-1, :-1, 2])
    assert np.allclose(got[..., 1], 0.5 * (block[:-1, :-1, :-1, 1] + block[:-1, 1:, :-1, 1]))


@pytest.mark.parametrize("nbytes", [0, 1, 4096, (1 << 20) - 4, 1 << 20, (1 << 20) + 4,
                                    3 * (1 << 20) + 12])
def test_chunks_are_bytes_under_the_cap_and_join_back(ff, nbytes):
    data = np.arange(nbytes // 4, dtype="<f4")
    chunks = ff._chunks(data)
    assert all(type(c) is bytes for c in chunks)
    assert all(len(c) <= ff.PAYLOAD_CHUNK_BYTES for c in chunks)
    assert b"".join(chunks) == data.tobytes()
    assert (len(chunks) == 0) == (data.nbytes == 0)


def test_chunks_of_a_brick_array_keep_its_layout(ff):
    bricks = np.random.default_rng(5).normal(size=(700, 512, 4)).astype("<f4")
    joined = b"".join(ff._chunks(bricks))
    assert np.array_equal(np.frombuffer(joined, dtype="<f4").reshape(-1, 512, 4), bricks)


def _rotation(angles):
    out = np.eye(3)
    for axis, t in enumerate(angles):
        c, s = np.cos(t), np.sin(t)
        r = np.eye(3)
        i, j = [k for k in range(3) if k != axis]
        r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
        out = out @ r
    return out


def test_a_rotation_with_per_axis_scales_splits(ff):
    rot = _rotation((0.3, -0.7, 1.1))
    scales = np.array([0.25, 0.26, 0.4])
    got = ff._rotation_and_scales(rot @ np.diag(scales))
    assert got is not None
    assert np.allclose(got[0], rot) and np.allclose(got[1], scales)


@pytest.mark.parametrize("linear", [
    np.array([[1.0, 0.3, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),  # a shear
    np.diag([1.0, -1.0, 1.0]),  # a mirror
    np.diag([1.0, 0.0, 1.0]),  # a zero scale
])
def test_a_shear_a_mirror_or_a_zero_scale_does_not_split(ff, linear):
    assert ff._rotation_and_scales(linear) is None


class _Transform:
    """An affine index-to-world map with openvdb's `indexToWorld` spelling."""

    def __init__(self, linear, origin):
        self.linear = np.asarray(linear, dtype=np.float64)
        self.origin = np.asarray(origin, dtype=np.float64)

    def indexToWorld(self, ijk):  # noqa: N802 - openvdb's own spelling
        return tuple(self.origin + self.linear @ np.asarray(ijk, dtype=np.float64))


def test_grids_on_one_transform_are_recognized_and_others_are_not(ff):
    base = _Transform(np.diag([0.2, 0.2, 0.2]), (1.0, -2.0, 0.5))
    assert ff._same_transform(base, _Transform(np.diag([0.2, 0.2, 0.2]), (1.0, -2.0, 0.5)))
    assert not ff._same_transform(base, _Transform(np.diag([0.1, 0.1, 0.1]), (1.0, -2.0, 0.5)))
    assert not ff._same_transform(base, _Transform(np.diag([0.2, 0.2, 0.2]), (1.1, -2.0, 0.5)))


def _rust_const(path: Path, name: str) -> int:
    text = path.read_text()
    m = re.search(rf"const {name}: u\d+ = ([0-9 *<]+);", text)
    assert m, f"{name} is not declared in {path}"
    expr = m.group(1)
    assert re.fullmatch(r"[0-9 *<]+", expr)
    return int(eval(expr))  # digits, spaces, `*` and `<<` only, checked above


def test_the_chunk_size_is_the_frontends(ff):
    text = (ROOT / "frontend" / "_cbor_bridge_.py").read_text()
    m = re.search(r"^PICKLE_CHUNK_BYTES = ([0-9 <]+)$", text, re.M)
    assert m, "PICKLE_CHUNK_BYTES is not declared in frontend/_cbor_bridge_.py"
    assert re.fullmatch(r"[0-9 <]+", m.group(1))
    assert int(eval(m.group(1))) == ff.PAYLOAD_CHUNK_BYTES  # digits and `<<` only


def test_the_transport_cap_is_the_servers(ff):
    path = ROOT / "crates" / "ppf-cts-server" / "src" / "wire" / "mod.rs"
    assert _rust_const(path, "MAX_PAYLOAD_BYTES") == ff.TRANSPORT_CAP_BYTES


def test_the_cell_limit_is_the_solvers(ff):
    path = ROOT / "crates" / "ppf-cts-solver" / "src" / "force_field.rs"
    assert _rust_const(path, "FLUID_MAX_RESOLUTION") == ff.FLUID_MAX_RESOLUTION


def test_a_report_line_names_the_field_the_source_and_the_size(ff):
    line = ff.volume_report_line({"field": "Smoke", "source": "Domain", "frames": 12,
                                  "resolution": [64, 32, 48], "bricks": 30,
                                  "brick_slots": 120, "occupancy": 0.25,
                                  "bytes": 3 * 1024 * 1024, "reused": False})
    assert "'Smoke'" in line and "'Domain'" in line and "12 frames" in line
    assert "64x32x48" in line and "3.0 MiB" in line and "25%" in line


def test_a_size_reads_in_the_largest_unit_it_fills(ff):
    # A refusal lists a kilobyte image beside a gigabyte volume; the small
    # part must not read as 0.0 MiB.
    assert ff._size(3 * 1024 ** 3) == "3.00 GiB"
    assert ff._size(3 * 1024 ** 2) == "3.0 MiB"
    assert ff._size(12 * 1024 + 512) == "12.5 KiB"
    assert ff._size(645) == "645 bytes"
    # Two sizes that would read alike against the cap are given in bytes.
    assert ff._past_cap(ff.TRANSPORT_CAP_BYTES + 1).startswith(
        f"{ff.TRANSPORT_CAP_BYTES + 1:,} bytes, past the {ff.TRANSPORT_CAP_BYTES:,} bytes")


def test_a_settings_key_is_independent_of_set_order(ff):
    # An enum flag reads as a set, whose iteration order is not its content.
    assert ff._hashable({"B", "A", "C"}) == ff._hashable({"C", "A", "B"}) == ("A", "B", "C")
    assert ff._hashable([[1.0, 2.0], [3.0, 4.0]]) == ((1.0, 2.0), (3.0, 4.0))
