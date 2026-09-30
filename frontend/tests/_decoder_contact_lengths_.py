# File: frontend/tests/_decoder_contact_lengths_.py
# License: Apache v2.0
#
# The decoder half of the per-object contact lengths.
#
# A group's `contact-gap` and `contact-offset` reach the decoder in one of two
# shapes. A sender that authors one distance for the whole group sends a plain
# number. A sender that sizes the distance from each object's own bounding box
# sends `{uuid: length}`, because the same fraction is a different length on
# each object, and a keyframed fraction sends `{uuid: [length per time]}` under
# `param-anim`. Either way the decoder sets the value on each object through
# the calls a notebook makes, `obj.param.set` and `obj.set_param_anim`.
#
# What this pins:
#   * a map gives each object the value under its own uuid;
#   * a plain number gives every object of the group that number;
#   * a map that names no value for an object of the group is refused by
#     name, static or animated, rather than leaving the object at the
#     parameter's default. A contact distance has no default that is right for
#     an object of unknown size;
#   * the two shapes can arrive side by side, one key a map and the other a
#     number, which is what a group does whose offset ratio is zero only by
#     value and not by shape.

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from frontend import _rust  # noqa: F401
    from frontend._asset_ import AssetManager
    from frontend._decoder_ import ParamDecoder
    from frontend._mesh_ import MeshManager
    from frontend._plot_ import PlotManager
    from frontend._scene_ import Scene
except ImportError:
    pytest.skip(
        "frontend._rust extension not built; run `cargo build --release` first",
        allow_module_level=True,
    )

# Scratch directory lives under the repo, never under /tmp: cleaned up
# by each test via a try/finally.
SCRATCH_ROOT = REPO_ROOT / "frontend" / "tests" / "_scratch_contact_lengths"

BIG = "uuid-big"
SMALL = "uuid-small"


class Workspace:
    """A scene holding two sheets whose reference names are their uuids.

    The decoder addresses objects by uuid through ``Scene.select``, so naming
    each object by its uuid drives the real per-object loop with no payload on
    disk.
    """

    def __enter__(self) -> "Workspace":
        self.root = SCRATCH_ROOT
        self.root.mkdir(parents=True, exist_ok=True)
        self.asset = AssetManager()
        self.mesh = MeshManager(str(self.root / "cache"))
        self.plot = PlotManager()
        V, F = self.mesh.square(res=3)
        self.asset.add.tri("sheet", V, F)
        self.scene = Scene("contact-lengths", self.plot, self.asset)
        self.scene.add("sheet", BIG).at(0, 5, 0)
        self.scene.add("sheet", SMALL).at(0, 9, 0)
        return self

    def __exit__(self, *_exc) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def decode(self, params: dict, times=None) -> None:
        decoder = ParamDecoder()
        decoder._data = {"group": [(params, [BIG, SMALL], [BIG, SMALL])]}
        if times is not None:
            decoder._data["param_anim_times"] = times
        decoder.apply_to_objects(self.scene)

    def value(self, uuid: str, key: str) -> float:
        return float(self.scene.select(uuid).param.get(key))


def test_a_map_gives_each_object_its_own_length():
    with Workspace() as ws:
        ws.decode({
            "contact-gap": {BIG: 0.05, SMALL: 0.0125},
            "contact-offset": {BIG: 0.01, SMALL: 0.0025},
        })
        assert ws.value(BIG, "contact-gap") == pytest.approx(0.05)
        assert ws.value(SMALL, "contact-gap") == pytest.approx(0.0125)
        assert ws.value(BIG, "contact-offset") == pytest.approx(0.01)
        assert ws.value(SMALL, "contact-offset") == pytest.approx(0.0025)


def test_a_number_gives_every_object_that_number():
    with Workspace() as ws:
        ws.decode({"contact-gap": 0.004, "contact-offset": 0.002})
        for uuid in (BIG, SMALL):
            assert ws.value(uuid, "contact-gap") == pytest.approx(0.004)
            assert ws.value(uuid, "contact-offset") == pytest.approx(0.002)


def test_the_two_shapes_arrive_side_by_side():
    with Workspace() as ws:
        ws.decode({
            "contact-gap": {BIG: 0.05, SMALL: 0.0125},
            "contact-offset": 0.0,
        })
        assert ws.value(BIG, "contact-gap") == pytest.approx(0.05)
        assert ws.value(SMALL, "contact-gap") == pytest.approx(0.0125)
        for uuid in (BIG, SMALL):
            assert ws.value(uuid, "contact-offset") == pytest.approx(0.0)


@pytest.mark.parametrize("key", ["contact-gap", "contact-offset"])
def test_a_map_that_names_no_value_for_an_object_is_refused(key):
    with Workspace() as ws:
        with pytest.raises(ValueError) as err:
            ws.decode({key: {BIG: 0.05}})
        message = str(err.value)
        assert key in message
        assert SMALL in message


def test_an_animated_map_gives_each_object_its_own_series():
    with Workspace() as ws:
        ws.decode(
            {
                "contact-gap": {BIG: 0.05, SMALL: 0.0125},
                "contact-offset": 0.0,
                "param-anim": {
                    "contact-gap": {
                        BIG: [0.05, 0.1],
                        SMALL: [0.0125, 0.025],
                    },
                },
            },
            times=[0.0, 1.0],
        )
        big = ws.scene.select(BIG).param_anim["contact-gap"]
        small = ws.scene.select(SMALL).param_anim["contact-gap"]
        assert big == pytest.approx([0.05, 0.1])
        assert small == pytest.approx([0.0125, 0.025])


def test_an_animated_list_is_shared_by_the_group():
    with Workspace() as ws:
        ws.decode(
            {
                "contact-gap": 0.004,
                "contact-offset": 0.0,
                "param-anim": {"contact-gap": [0.004, 0.008]},
            },
            times=[0.0, 1.0],
        )
        for uuid in (BIG, SMALL):
            series = ws.scene.select(uuid).param_anim["contact-gap"]
            assert series == pytest.approx([0.004, 0.008])


def test_an_animated_map_that_names_no_series_for_an_object_is_refused():
    with Workspace() as ws:
        with pytest.raises(ValueError) as err:
            ws.decode(
                {
                    "contact-gap": {BIG: 0.05, SMALL: 0.0125},
                    "contact-offset": 0.0,
                    "param-anim": {"contact-gap": {BIG: [0.05, 0.1]}},
                },
                times=[0.0, 1.0],
            )
        message = str(err.value)
        assert "contact-gap" in message
        assert SMALL in message
