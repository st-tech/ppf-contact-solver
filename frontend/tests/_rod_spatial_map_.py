# File: frontend/tests/_rod_spatial_map_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A spatial material map on a rod varies one parameter along the strand: the
# value at a segment is lerp(base, target, w) with `w` the mean of the weights
# at its two ends, and the scene build writes one value per segment where an
# unmapped rod repeats a single scalar.
#
# Covers, through `Scene.build()`:
#   * a mapped key reaches the rod table segment by segment, in the order of
#     the rod's own segments, and a weight of zero everywhere reproduces the
#     unmapped table exactly;
#   * every key a rod segment reads can be mapped, and the keys beside the
#     mapped one stay as authored;
#   * a second rod in the scene keeps its own values, on either side of the
#     mapped one;
#   * a mapped contact gap is what the solvability check and the gap limit
#     read, as they do for a sheet.
#
# Covers, on the object:
#   * a weight array of the wrong length, outside [0, 1] or not finite is
#     refused, and so is a second map on one key;
#   * a map keyed over time is refused by name, since the per-frame material
#     tables are written for triangles only;
#   * an object with no elements is still refused.

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
    from frontend._plot_ import PlotManager
    from frontend._scene_ import Scene
except ImportError:
    pytest.skip(
        "frontend._rust extension not built; run `cargo build --release` first",
        allow_module_level=True,
    )

SCRATCH_ROOT = REPO_ROOT / "frontend" / "tests" / "_scratch_rod_spatial_map"

# The keys a rod segment reads from its own material record and a map may
# therefore vary, with a base and a target apart enough to tell them from each
# other and from the default.
ROD_MAP_KEYS = {
    "young-mod": (2.0e4, 8.0e5),
    "bend": (0.5, 40.0),
    "friction": (0.1, 0.7),
    "deformation-damping": (0.0, 0.02),
    "bending-damping": (0.001, 0.05),
    "strain-limit": (0.02, 0.2),
    "bend-plasticity": (0.5, 6.0),
}


def strand(n_segments=6, length=0.6):
    y = np.linspace(0.0, -length, n_segments + 1)
    vert = np.stack([np.zeros_like(y), y, np.zeros_like(y)], axis=1)
    edge = np.array([[i, i + 1] for i in range(n_segments)], dtype=np.uint32)
    return vert, edge


def reversed_strand(n_segments=6, length=0.6):
    """The same strand with every segment written tip first, so a reduction
    that assumed (i, i + 1) would still pass on `strand` and fail here."""
    vert, edge = strand(n_segments, length)
    return vert, edge[::-1, ::-1].copy()


@pytest.fixture
def scene():
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        asset = AssetManager()
        asset.add.rod("strand", *strand())
        asset.add.rod("reversed", *reversed_strand())
        asset.add.rod("short", *strand(n_segments=3, length=0.3))
        yield Scene("rod_spatial_map", PlotManager(), asset)
    finally:
        shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)


def rod_values(fixed, key):
    return np.array(fixed._rod_param[key], dtype=np.float64)


def ramp(obj):
    """One weight per vertex: 0 at the first vertex, 1 at the last."""
    n = len(obj.get("V"))
    return np.linspace(0.0, 1.0, n, dtype=np.float32)


def expected_segments(obj, weights, base, target):
    """lerp(base, target, mean of the two end weights), per segment of `obj`."""
    w = np.asarray(weights, dtype=np.float64)[np.asarray(obj.get("E"))].mean(axis=1)
    return base + (target - base) * w


# ----- the build -------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(ROD_MAP_KEYS))
def test_a_mapped_key_reaches_the_rod_segment_by_segment(scene, key):
    base, target = ROD_MAP_KEYS[key]
    rod = scene.add("strand")
    rod.param.set(key, base)
    weights = ramp(rod)
    rod.set_param_spatial(key, weights, target)
    fixed = scene.build(quiet=True)
    expected = expected_segments(rod, weights, base, target)
    # The ramp gives every segment a different mean, so an order mistake or a
    # replicated scalar cannot pass.
    assert len(np.unique(np.round(expected, 9))) == len(expected)
    assert rod_values(fixed, key) == pytest.approx(expected, rel=1e-6)


def test_a_segment_takes_the_mean_of_its_own_ends_whatever_their_order(scene):
    base, target = ROD_MAP_KEYS["bend"]
    rod = scene.add("reversed")
    rod.param.set("bend", base)
    weights = ramp(rod) ** 2
    rod.set_param_spatial("bend", weights, target)
    fixed = scene.build(quiet=True)
    expected = expected_segments(rod, weights, base, target)
    assert rod_values(fixed, "bend") == pytest.approx(expected, rel=1e-6)
    # Written tip first, the stiff end comes first in the table.
    assert expected[0] > expected[-1]


def test_a_weight_of_zero_reproduces_the_unmapped_table(scene):
    base, target = ROD_MAP_KEYS["young-mod"]
    plain = scene.add("strand")
    plain.param.set("young-mod", base)
    unmapped = scene.build(quiet=True)

    scene.clear()
    mapped = scene.add("strand")
    mapped.param.set("young-mod", base)
    mapped.set_param_spatial(
        "young-mod", np.zeros(len(mapped.get("V")), dtype=np.float32), target
    )
    fixed = scene.build(quiet=True)
    for key, values in unmapped._rod_param.items():
        assert fixed._rod_param[key] == values, key


def test_the_keys_beside_the_mapped_one_stay_as_authored(scene):
    rod = scene.add("strand")
    rod.param.set("bend", 3.0).set("friction", 0.25)
    rod.set_param_spatial("bend", ramp(rod), 30.0)
    fixed = scene.build(quiet=True)
    assert (rod_values(fixed, "friction") == np.float64(np.float32(0.25))).all()
    assert len(set(fixed._rod_param["young-mod"])) == 1


def test_each_rod_keeps_its_own_values_beside_a_mapped_one(scene):
    first = scene.add("short").at(-1.0, 0.0, 0.0)
    first.param.set("bend", 1.0)
    mapped = scene.add("strand")
    mapped.param.set("bend", 2.0)
    weights = ramp(mapped)
    mapped.set_param_spatial("bend", weights, 20.0)
    last = scene.add("reversed").at(1.0, 0.0, 0.0)
    last.param.set("bend", 5.0)
    fixed = scene.build(quiet=True)

    counts = {
        name: len(obj.get("E"))
        for name, obj in (("short", first), ("strand", mapped), ("reversed", last))
    }
    values = rod_values(fixed, "bend")
    assert len(values) == sum(counts.values())
    # The rods are written in the order the scene holds them, which is not
    # the order they were added in, so each one is found by its own values.
    expected = {
        "short": np.full(counts["short"], 1.0),
        "strand": expected_segments(mapped, weights, 2.0, 20.0),
        "reversed": np.full(counts["reversed"], 5.0),
    }
    cursor = 0
    seen = []
    while cursor < len(values):
        for name, block in expected.items():
            chunk = values[cursor:cursor + len(block)]
            if name not in seen and len(chunk) == len(block) and np.allclose(
                chunk, block, rtol=1e-6
            ):
                seen.append(name)
                cursor += len(block)
                break
        else:
            raise AssertionError(
                f"no rod's values start at segment {cursor}: {values.tolist()}"
            )
    assert sorted(seen) == sorted(expected)


def test_a_mapped_contact_gap_is_read_by_the_gap_checks(scene):
    rod = scene.add("strand")
    rod.param.set("contact-offset", 0.0).set("contact-gap", 1e-3)
    rod.param.set("contact-gap-limit", 0.25)
    weights = ramp(rod)
    rod.set_param_spatial("contact-gap", weights, 1.0)
    fixed = scene.build(quiet=True)
    # A 0.1 m segment has room for a quarter of its length, and the map asks
    # for up to a meter, so the limit decides wherever the map passes it.
    mapped = expected_segments(rod, weights, 1e-3, 1.0)
    assert rod_values(fixed, "contact-gap") == pytest.approx(
        np.minimum(mapped, 0.25 * 0.1), rel=1e-5
    )


# ----- the object ------------------------------------------------------------


def test_a_weight_array_of_the_wrong_length_is_refused(scene):
    rod = scene.add("strand")
    with pytest.raises(ValueError, match="has 3 weights but the object has 7"):
        rod.set_param_spatial("bend", [0.0, 0.5, 1.0], 10.0)


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
def test_a_weight_outside_the_interval_is_refused(scene, bad):
    rod = scene.add("strand")
    weights = ramp(rod)
    weights[2] = bad
    with pytest.raises(ValueError, match="spatial map"):
        rod.set_param_spatial("bend", weights, 10.0)


def test_a_second_map_on_one_key_is_refused(scene):
    rod = scene.add("strand")
    rod.set_param_spatial("bend", ramp(rod), 10.0)
    with pytest.raises(ValueError, match="already carries a spatial map"):
        rod.set_param_spatial("bend", ramp(rod), 20.0)


def test_a_key_a_rod_does_not_carry_is_refused(scene):
    rod = scene.add("strand")
    with pytest.raises(KeyError, match="not found on a 'rod' object"):
        rod.set_param_spatial("plasticity", ramp(rod), 0.3)


def test_a_map_keyed_over_time_is_refused_by_name(scene):
    rod = scene.add("strand")
    frames = [ramp(rod), ramp(rod)[::-1].copy()]
    with pytest.raises(ValueError, match="on a rod cannot change over time"):
        rod.set_param_spatial_anim("bend", frames, [0.0, 1.0], 10.0)
    # Refused before it is recorded, so the build cannot ship it either.
    assert "bend" not in rod.param_spatial


def test_the_element_table_a_map_is_reduced_over(scene):
    assert scene.add("strand").map_element_key == "E"
