# File: frontend/tests/_session_statistics_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# `SessionGet.statistics` and its two companions read the per-object
# statistics the solver writes beside its frames, which the Blender add-on's
# Object Statistics panel shows.
#
# The files are written here in the solver's own envelope rather than by a
# solve, because this tier builds the extension module and no solver. The
# envelope is `{version, kind, payload}` with `STATISTICS_VERSION` and the
# `StatisticsManifest` / `StatisticsFrame` kinds of
# `crates/ppf-cts-formats/src/statistics.rs`, so a change to that format fails
# here as well as in the solver.
#
# Covers:
#   * the latest frame is read by default and a named one on request, keyed
#     by object name, with its time;
#   * a channel the object SUPPORTS but the solver marked invalid reads None,
#     one it does not support is absent, and the contact count is an int;
#   * a series leaves out the invalid frames rather than filling them, and
#     refuses an unknown object or an unsupported channel by name;
#   * a frame that breaks the manifest's contract is refused, not read.

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from frontend import _rust  # noqa: F401
    from frontend._session_inspect_ import SessionGet
except ImportError:
    pytest.skip(
        "frontend._rust extension not built; run `cargo build --release` first",
        allow_module_level=True,
    )

cbor2 = pytest.importorskip("cbor2")

STATISTICS_VERSION = 1
# Bit positions of `StatisticChannel`.
LOCATION = (1 << 0) | (1 << 1) | (1 << 2)
SURFACE_AREA = 1 << 4
ROD_LENGTH = 1 << 6
SPEED = 1 << 11
ACCELERATION_MAGNITUDE = 1 << 15
CONTACT_COUNT = 1 << 24


def _write(path: Path, kind: str, payload: dict) -> None:
    path.write_bytes(
        cbor2.dumps({"version": STATISTICS_VERSION, "kind": kind, "payload": payload})
    )


def _object(index: int, **overrides) -> dict:
    record = {
        "object_index": index,
        "location": [0.0, 0.0, 0.0],
        "volume": 0.0,
        "volume_stretch": 0.0,
        "surface_area": 0.0,
        "area_stretch": 0.0,
        "rod_length": 0.0,
        "length_stretch": 0.0,
        "velocity": [0.0, 0.0, 0.0],
        "speed": 0.0,
        "acceleration": [0.0, 0.0, 0.0],
        "acceleration_magnitude": 0.0,
        "angular_velocity": [0.0, 0.0, 0.0],
        "angular_speed": 0.0,
        "angular_axis": [0.0, 0.0, 0.0],
        "contact_count": 0,
        "valid_channels": 0,
    }
    record.update(overrides)
    return record


SHEET_SUPPORTED = LOCATION | SURFACE_AREA | SPEED | ACCELERATION_MAGNITUDE | CONTACT_COUNT
ROD_SUPPORTED = LOCATION | ROD_LENGTH | SPEED


@pytest.fixture
def get(tmp_path):
    _write(
        tmp_path / "statistics_manifest.cbor",
        "StatisticsManifest",
        {
            "objects": [
                {
                    "object_index": 0,
                    "object_uuid": "uuid-sheet",
                    "object_name": "sheet",
                    "dynamics_type": "SHELL",
                    "supported_channels": SHEET_SUPPORTED,
                },
                {
                    "object_index": 1,
                    "object_uuid": "uuid-rod",
                    "object_name": "rod",
                    "dynamics_type": "ROD",
                    "supported_channels": ROD_SUPPORTED,
                },
            ]
        },
    )
    for frame, speed in ((1, 0.5), (2, 1.5), (3, 2.5)):
        # Frame 1 has no acceleration yet, which the solver marks invalid.
        sheet_valid = LOCATION | SURFACE_AREA | SPEED | CONTACT_COUNT
        if frame > 1:
            sheet_valid |= ACCELERATION_MAGNITUDE
        _write(
            tmp_path / f"statistics_{frame}.cbor",
            "StatisticsFrame",
            {
                "solver_frame": frame,
                "time_seconds": frame / 10,
                "objects": [
                    _object(
                        0,
                        location=[0.0, float(frame), 0.0],
                        surface_area=4.0,
                        speed=speed,
                        acceleration_magnitude=9.8,
                        contact_count=7 * frame,
                        valid_channels=sheet_valid,
                    ),
                    _object(
                        1,
                        rod_length=1.25,
                        speed=0.0,
                        valid_channels=LOCATION | ROD_LENGTH | SPEED,
                    ),
                ],
            },
        )
    fixed = SimpleNamespace(
        output=SimpleNamespace(path=str(tmp_path)),
        session=SimpleNamespace(proj_root=str(REPO_ROOT)),
    )
    return SessionGet(fixed)


def test_the_latest_frame_is_read_by_default(get):
    assert get.statistics_frames() == [1, 2, 3]
    stats = get.statistics()
    assert stats["frame"] == 3 and stats["time"] == pytest.approx(0.3)
    sheet = stats["objects"]["sheet"]
    assert sheet["uuid"] == "uuid-sheet" and sheet["type"] == "SHELL"
    assert sheet["values"]["speed"] == pytest.approx(2.5)
    assert sheet["values"]["location_y"] == pytest.approx(3.0)
    assert sheet["values"]["contact_count"] == 21
    assert isinstance(sheet["values"]["contact_count"], int)


def test_invalid_reads_none_and_unsupported_is_absent(get):
    first = get.statistics(1)["objects"]
    assert first["sheet"]["values"]["acceleration_magnitude"] is None
    assert "volume" not in first["sheet"]["values"]
    assert "surface_area" not in first["rod"]["values"]
    assert first["rod"]["values"]["rod_length"] == pytest.approx(1.25)


def test_a_series_skips_the_frames_with_no_value(get):
    t, speed = get.statistics_series("sheet", "speed")
    np.testing.assert_allclose(t, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(speed, [0.5, 1.5, 2.5])
    t, accel = get.statistics_series("sheet", "acceleration_magnitude")
    np.testing.assert_allclose(t, [0.2, 0.3])
    with pytest.raises(KeyError, match="no statistics record names"):
        get.statistics_series("nowhere", "speed")
    with pytest.raises(KeyError, match="ROD object"):
        get.statistics_series("rod", "volume")


def test_no_frame_yet_reads_none(tmp_path):
    fixed = SimpleNamespace(
        output=SimpleNamespace(path=str(tmp_path / "not-started")),
        session=SimpleNamespace(proj_root=str(REPO_ROOT)),
    )
    get = SessionGet(fixed)
    assert get.statistics_frames() == []
    assert get.statistics() is None


def test_a_missing_frame_is_refused(get):
    with pytest.raises(OSError, match="statistics_9.cbor"):
        get.statistics(9)


def test_a_frame_breaking_the_manifest_is_refused(get, tmp_path):
    # Marks a channel valid that the rod does not support.
    _write(
        tmp_path / "statistics_4.cbor",
        "StatisticsFrame",
        {
            "solver_frame": 4,
            "time_seconds": 0.4,
            "objects": [
                _object(0),
                _object(1, valid_channels=SURFACE_AREA),
            ],
        },
    )
    with pytest.raises(ValueError, match="unsupported channels"):
        get.statistics(4)
