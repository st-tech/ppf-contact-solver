# File: frontend/tests/_tetra_cache_key_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The tetrahedralization cache key: one composer, one probe, one dedup key.
#
# A SOLID object's tet mesh is cached under a filename that stands for the
# mesh and the tetrahedralizer settings together. Three properties hold and
# are exercised here:
#
#   * the build planner and the mesh writer compose that filename with the
#     same helper, so the plan reports on the file the build reads;
#   * the file carries the settings it was written for, so a name that
#     addresses it is confirmed rather than trusted;
#   * two objects share a tetrahedralization only when they address the same
#     file, so equal geometry with different settings gets its own run.
#
# A cache path that can be neither read nor shown to be absent is reported
# instead of being counted as a miss.

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The cdylib is the one import an unbuilt tree cannot satisfy, so it is the
# only one a skip stands for. The names below it are plain Python in this
# repository: one of them going missing is a regression, and it reaches the
# report as a collection error rather than as a module that quietly did not
# run.
try:
    from frontend import _rust  # type: ignore[attr-defined]
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f"frontend / _ppf_cts_py not importable in this environment: {exc}",
        allow_module_level=True,
    )

from frontend._decoder_ import SceneDecoder  # noqa: E402
from frontend._mesh_ import (  # noqa: E402
    CachePathUnusableError,
    TriMesh,
    _cache_probe,
)


# A float32 Blender property reaches Python's str() with its full binary
# expansion, which is what made an unbounded cache name reachable in the
# first place. Both override sets below use that spelling.
OVERRIDE_KWARGS = {"edge_length_fac": float(np.float32(0.05))}
TETGEN_KWARGS = {"backend": "tetgen"}


# ---------------------------------------------------------------------------
# A cube surface plus a tetrahedralization of it, so the cache read / write
# path runs without a real tetrahedralizer.
# ---------------------------------------------------------------------------

CUBE_VERT = np.array(
    [
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)
CUBE_TRI = np.array(
    [
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7],
    ],
    dtype=np.uint32,
)
CUBE_TET = np.array(
    [
        [0, 1, 2, 5], [0, 2, 3, 7], [0, 2, 5, 7], [0, 4, 5, 7], [2, 5, 6, 7],
    ],
    dtype=np.uint32,
)


@pytest.fixture
def cache_dir(tmp_path):
    d = tmp_path / "cash"
    d.mkdir()
    return str(d)


@pytest.fixture
def cube(cache_dir):
    return TriMesh.create(CUBE_VERT.copy(), CUBE_TRI.copy(), cache_dir)


@pytest.fixture
def canned_backend(monkeypatch):
    """Stand in for the tetrahedralizer so the cache path is what is tested.

    Records every call, so a test can tell a cache hit from a fresh run.
    """
    calls = []

    def _fake(self, kwargs, status_callback, status_interval, timeout=None, retries=None):
        calls.append(dict(kwargs))
        return CUBE_VERT.copy(), CUBE_TRI.copy(), CUBE_TET.copy()

    monkeypatch.setattr(TriMesh, "_tetrahedralize_ftetwild", _fake)
    return calls


def written_cache_files(cache_dir):
    return sorted(
        f for f in os.listdir(cache_dir) if f.endswith(".npz")
    )


# ---------------------------------------------------------------------------
# The key stored in the file
# ---------------------------------------------------------------------------


def test_cache_key_round_trips(cube, cache_dir, canned_backend):
    """The written file names the settings it was written for, and a second
    call with those settings loads it instead of running again."""
    cube.tetrahedralize(**OVERRIDE_KWARGS)
    assert len(canned_backend) == 1

    files = written_cache_files(cache_dir)
    assert len(files) == 1, files
    data = np.load(os.path.join(cache_dir, files[0]))
    _name, expected_key = _rust.mesh_tetra_cache_key(
        cube.hash, [], [(k, str(v)) for k, v in OVERRIDE_KWARGS.items()]
    )
    assert str(data["cache_key"]) == expected_key
    assert expected_key != ""

    cube.tetrahedralize(**OVERRIDE_KWARGS)
    assert len(canned_backend) == 1, "the second call re-ran the tetrahedralizer"


def test_cache_key_mismatch_raises(cube, cache_dir, canned_backend):
    """A file whose stored key disagrees with the requested settings is
    rejected, not loaded."""
    cube.tetrahedralize(**OVERRIDE_KWARGS)
    path = os.path.join(cache_dir, written_cache_files(cache_dir)[0])
    data = dict(np.load(path))
    data["cache_key"] = np.array("edge_length_fac=999.0")
    np.savez(path, **data)

    with pytest.raises(ValueError) as excinfo:
        cube.tetrahedralize(**OVERRIDE_KWARGS)
    assert "edge_length_fac=999.0" in str(excinfo.value)


def test_absent_cache_key_is_accepted_only_for_the_default_name(
    cube, cache_dir, canned_backend
):
    """A file with no stored key is accepted for the default, argument-free
    name and refused for a name that stands for a digest."""
    # Default settings: no key is required, since no digest names the file.
    cube.tetrahedralize()
    default_path = os.path.join(cache_dir, written_cache_files(cache_dir)[0])
    default = dict(np.load(default_path))
    del default["cache_key"]
    np.savez(default_path, **default)
    cube.tetrahedralize()
    assert len(canned_backend) == 1, "a keyless default cache was refused"

    # Overridden settings: the name is a digest, so the key has to be there.
    cube.tetrahedralize(**OVERRIDE_KWARGS)
    override_name, _key = _rust.mesh_tetra_cache_key(
        cube.hash, [], [(k, str(v)) for k, v in OVERRIDE_KWARGS.items()]
    )
    override_path = cube.cache_path(override_name)
    stripped = dict(np.load(override_path))
    del stripped["cache_key"]
    np.savez(override_path, **stripped)
    with pytest.raises(ValueError) as excinfo:
        cube.tetrahedralize(**OVERRIDE_KWARGS)
    assert "cache_key" in str(excinfo.value)


# ---------------------------------------------------------------------------
# One composer for the planner and the writer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs", [OVERRIDE_KWARGS, TETGEN_KWARGS, {}], ids=["override", "tetgen", "default"]
)
def test_planner_probe_matches_writer_path(cube, cache_dir, canned_backend, monkeypatch, kwargs):
    """The path the build plan probes is the path the build writes."""

    def _fake_tetgen(self, kw, status_callback):
        return CUBE_VERT.copy(), CUBE_TRI.copy(), CUBE_TET.copy()

    monkeypatch.setattr(TriMesh, "_tetrahedralize_tetgen", _fake_tetgen)

    planner_path = cube.cache_path(SceneDecoder._tetra_cache_name(cube, kwargs))
    assert not os.path.exists(planner_path)

    cube.tetrahedralize(**kwargs)

    written = written_cache_files(cache_dir)
    assert written == [os.path.basename(planner_path)], (
        f"planner probed {os.path.basename(planner_path)!r}, "
        f"build wrote {written!r}"
    )


def test_planner_name_is_bounded():
    """No argument set can push the cache filename past what a Windows path
    component holds."""
    widest = {
        "edge_length_fac": float(np.float32(0.05)),
        "epsilon": float(np.float32(0.001)),
        "stop_energy": float(np.float32(10.0)),
        "num_opt_iter": 80,
        "optimize": True,
        "simplify": True,
        "coarsen": True,
    }

    class _Stub:
        hash = "f" * 64

    name = SceneDecoder._tetra_cache_name(_Stub(), widest)
    component = os.path.basename(_rust.mesh_cache_path("/c", _Stub.hash, name))
    assert len(component) < 255, f"{len(component)}: {component}"


# ---------------------------------------------------------------------------
# Dedup keys on the whole cache key, not on the geometry alone
# ---------------------------------------------------------------------------


class _StubTriMesh:
    def __init__(self, hash_value: str):
        self.hash = hash_value


def _entry(hash_value: str, cache_name: str) -> dict:
    return {
        "group_type": "SOLID",
        "tri_mesh": _StubTriMesh(hash_value),
        "tetra_cache_name": cache_name,
        "tetra_weight": 8.0,
        "tetra_index": None,
        "name": cache_name,
    }


def test_dedup_respects_tet_kwargs():
    """Equal geometry with different tetrahedralizer settings gets its own
    run rather than silently inheriting the first object's mesh."""
    entries = [
        _entry("aa", "aa_tetrahedralize_.npz"),
        _entry("aa", "aa_tetrahedralize_0123456789abcdef.npz"),
    ]
    jobs, _work_delta = _rust.dedup_and_rebuild_tetra_jobs(entries)
    assert entries[0]["tetra_reuse_from"] is None
    assert entries[1]["tetra_reuse_from"] is None
    assert len(jobs) == 2


def test_dedup_still_collapses_identical_kwargs():
    """Equal geometry with equal settings runs once."""
    entries = [
        _entry("aa", "aa_tetrahedralize_.npz"),
        _entry("aa", "aa_tetrahedralize_.npz"),
    ]
    jobs, work_delta = _rust.dedup_and_rebuild_tetra_jobs(entries)
    assert entries[0]["tetra_reuse_from"] is None
    assert entries[1]["tetra_reuse_from"] == 0
    assert len(jobs) == 1
    assert work_delta == -8.0


def test_dedup_rejects_an_entry_with_no_cache_name():
    """A SOLID plan entry that lost its cache name is a planner defect and
    stops the build instead of deduping on the geometry alone."""
    broken = _entry("aa", "aa_tetrahedralize_.npz")
    del broken["tetra_cache_name"]
    with pytest.raises(ValueError):
        _rust.dedup_and_rebuild_tetra_jobs([broken])


# ---------------------------------------------------------------------------
# Absent versus unusable
# ---------------------------------------------------------------------------


def test_missing_cache_path_is_a_silent_miss(tmp_path):
    """An absent file, and an absent directory above it, are ordinary
    misses."""
    assert _cache_probe(str(tmp_path / "nothing.npz")) is False
    assert _cache_probe(str(tmp_path / "no" / "such" / "dir" / "x.npz")) is False
    existing = tmp_path / "there.npz"
    existing.write_bytes(b"")
    assert _cache_probe(str(existing)) is True


def test_unusable_cache_path_is_not_a_miss(tmp_path):
    """A path the process cannot classify is reported, not counted as a
    cache miss that the following write would fail on anyway."""
    # A name longer than the filesystem accepts.
    with pytest.raises(CachePathUnusableError):
        _cache_probe(str(tmp_path / ("x" * 300 + ".npz")))

    if os.name == "nt":
        pytest.skip("POSIX directory permissions do not apply on Windows")
    if os.geteuid() == 0:
        pytest.skip("root searches a mode-000 directory regardless")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    target = blocked / "cache.npz"
    target.write_bytes(b"")
    os.chmod(blocked, 0)
    try:
        with pytest.raises(CachePathUnusableError):
            _cache_probe(str(target))
    finally:
        os.chmod(blocked, stat.S_IRWXU)


@pytest.mark.skipif(
    os.name == "nt",
    reason="measured on POSIX, where a non-directory path component is ENOTDIR",
)
def test_non_directory_in_the_path_is_not_a_miss(tmp_path):
    """A regular file partway down the cache path is reported.

    The write that a miss authorizes goes to the same path and gets the
    same errno, so answering "absent" here would send the build to a
    location it cannot create.
    """
    intruder = tmp_path / "not-a-dir"
    intruder.write_bytes(b"")
    blocked = str(intruder / "cache.npz")
    with pytest.raises(OSError):
        open(blocked, "wb").close()
    with pytest.raises(CachePathUnusableError):
        _cache_probe(blocked)


def test_unusable_cache_path_stops_the_build(cube, tmp_path, canned_backend):
    """``tetrahedralize`` raises on an unusable cache location instead of
    running the tetrahedralizer and failing on the write afterwards."""
    if os.name == "nt":
        pytest.skip("POSIX directory permissions do not apply on Windows")
    if os.geteuid() == 0:
        pytest.skip("root searches a mode-000 directory regardless")
    blocked = tmp_path / "blocked-cache"
    blocked.mkdir()
    os.chmod(blocked, 0)
    try:
        cube.set_cache_dir(str(blocked))
        with pytest.raises(CachePathUnusableError):
            cube.tetrahedralize()
    finally:
        os.chmod(blocked, stat.S_IRWXU)
    assert canned_backend == [], "the tetrahedralizer ran before the write failed"
