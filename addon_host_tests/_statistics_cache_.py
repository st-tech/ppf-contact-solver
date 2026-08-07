# File: addon_host_tests/_statistics_cache_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Host-side gates for ``blender_addon/core/statistics_cache.py``.
#
# The cache is a flat array of fixed-size records indexed by solver frame:
# ``write_frame_blob`` seeks to ``header + solver_frame * record`` and
# zero-fills everything up to it. The frame index arrives inside a
# CBOR payload written by the solver, so it is untrusted input to a
# byte-offset computation, and it must be bounded on BOTH sides before it
# reaches the file.

from __future__ import annotations

import os

import pytest

pytest.importorskip("cbor2")


STATISTICS_VERSION = 1
FRAME_KIND = "StatisticsFrame"
MANIFEST_KIND = "StatisticsManifest"
UUID = "obj-0000"

# One channel bit (LOCATION_X) is enough: the mask only has to be a subset
# of what the manifest declares as supported.
SUPPORTED = 0b1
VALID = 0b1


def _envelope(kind, payload):
    import cbor2

    return cbor2.dumps({"version": STATISTICS_VERSION, "kind": kind, "payload": payload})


def _manifest_blob():
    return _envelope(
        MANIFEST_KIND,
        {
            "objects": [
                {"object_index": 0, "object_uuid": UUID, "supported_channels": SUPPORTED}
            ]
        },
    )


def _frame_blob(solver_frame):
    record = {
        "object_index": 0,
        "valid_channels": VALID,
        "location": [0.0, 0.0, 0.0],
        "velocity": [0.0, 0.0, 0.0],
        "acceleration": [0.0, 0.0, 0.0],
        "angular_velocity": [0.0, 0.0, 0.0],
        "angular_axis": [0.0, 0.0, 0.0],
    }
    return _envelope(
        FRAME_KIND,
        {"solver_frame": solver_frame, "time_seconds": 0.0, "objects": [record]},
    )


@pytest.fixture
def cache(statistics_cache, monkeypatch, tmp_path):
    """The module with its on-disk root redirected under ``tmp_path``.

    ``get_pc2_dir`` is bound into the module namespace at import, so the
    redirect is applied there rather than on ``core.pc2``.
    """
    monkeypatch.setattr(statistics_cache, "get_pc2_dir", lambda: str(tmp_path))
    return statistics_cache


def test_writes_a_frame_at_its_own_offset(cache, tmp_path):
    """Baseline: a plausible frame index lands one record into the file."""
    manifest = cache.install_manifest(_manifest_blob())
    assert cache.write_frame_blob(_frame_blob(3), manifest, max_solver_frame=3) == 3
    record = cache.read_record(UUID, 3)
    assert record is not None and record["solver_frame"] == 3
    assert cache.read_record(UUID, 2) is None


def test_negative_frame_index_is_rejected(cache):
    """A negative index would seek before the header. It is refused during
    decode, so the offset is never computed. Pinned so the guard cannot be
    dropped while the seek stays unguarded.
    """
    manifest = cache.install_manifest(_manifest_blob())
    with pytest.raises(cache.StatisticsCacheError):
        cache.write_frame_blob(_frame_blob(-1), manifest, max_solver_frame=0)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "PENDING dev-preexisting-fixes | G4: solver_frame is bounded below "
        "but not above, so an absurd decoded index zero-fills the cache "
        "file without limit"
    ),
)
def test_absurd_frame_index_is_rejected(cache):
    """An index far past any plausible run must be reported, not honored.

    ``write_frame_blob`` zero-fills from the end of the file up to
    ``header + solver_frame * record``, so an unbounded index turns a
    corrupted CBOR field into unbounded disk consumption. The bound
    therefore belongs in ``decode_frame``, beside the lower bound that is
    already there, so it applies before any byte offset is computed and
    protects every caller rather than one.

    The index below is 1e12, which at 128 bytes per record asks the write
    path for about 128 TB. That is also why this drives ``decode_frame``
    directly instead of ``write_frame_blob``: against the current code the
    write would not raise, it would start filling.
    """
    manifest = cache.decode_manifest(_manifest_blob())
    with pytest.raises(cache.StatisticsCacheError):
        cache.decode_frame(_frame_blob(10**12), manifest)


def test_frame_offset_is_where_the_index_says(cache, tmp_path):
    """Pins the layout the bound above is protecting: record ``n`` sits at
    ``header + n * record``, so the index scales the file linearly and an
    unchecked index scales it without limit.
    """
    manifest = cache.install_manifest(_manifest_blob())
    cache.write_frame_blob(_frame_blob(4), manifest, max_solver_frame=4)
    written = [p for p in tmp_path.iterdir() if p.suffix == ".stats"]
    assert len(written) == 1, [p.name for p in tmp_path.iterdir()]
    expected = cache._HEADER.size + 5 * cache._RECORD.size
    assert os.path.getsize(written[0]) == expected
