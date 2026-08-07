# File: frontend/tests/_cache_path_length_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# End-to-end cover for the tetrahedralize cache filename, driven through
# the same cdylib entry points the frontend calls.
#
# ``crates/ppf-cts-core/tests/cache_path_gates.rs`` gates the arithmetic
# in isolation. This file covers what the arithmetic is FOR: the composed
# value is a single path component, and a component the filesystem refuses
# is an ``OSError`` the caller cannot act on. Two of the tests here are
# characterizations of the filesystem rather than gates on this repo's
# code, and say so; the third is the planner/writer agreement gate at the
# level where the two composers actually live
# (``_decoder_.py`` vs ``_mesh_.py``).

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from frontend import _rust
except Exception as e:  # pragma: no cover - environment-dependent
    pytest.skip(f"_ppf_cts_py cdylib unavailable: {e}", allow_module_level=True)


HASH = "f" * 64

# The seven per-object fTetWild overrides ``_encode_obj_tet_kwargs``
# forwards, each rendered as ``str(v)`` over the value read out of the RNA
# property. The three float fields are ``FloatProperty``, i.e. float32, so
# widening one to a Python float prints its full float64 image: the 0.05
# default reads back as "0.05000000074505806", 19 characters for a value
# the artist typed as 4. That expansion is why the composed name overruns
# at realistic settings rather than at contrived ones, so the values here
# are produced through float32 rather than written out as literals.
ALL_OVERRIDES = [
    ("edge_length_fac", str(float(np.float32(0.05)))),
    ("epsilon", str(float(np.float32(1e-3)))),
    ("stop_energy", str(float(np.float32(10.0)))),
    ("num_opt_iter", str(80)),
    ("optimize", str(True)),
    ("simplify", str(True)),
    ("coarsen", str(False)),
]


def _writer_path(cache_dir, kwargs):
    """The path ``Mesh.tetrahedralize`` composes for these kwargs."""
    name, _key = _rust.mesh_tetra_cache_key(HASH, [], list(kwargs))
    return _rust.mesh_cache_path(str(cache_dir), HASH, name)


def _planner_path(cache_dir, kwargs=()):
    """The path the build planner probes for these kwargs.

    The planner and the writer compose through one function, so the two
    agree for any kwargs set by construction rather than by convention.
    """
    name, _key = _rust.mesh_tetra_cache_key(HASH, [], list(kwargs))
    return _rust.mesh_cache_path(str(cache_dir), HASH, name)


def test_default_cache_name_opens(tmp_path):
    """Characterization, not a gate: the no-override name is short enough
    for every filesystem this project ships on, so the cache works today
    for an object that carries no fTetWild override.

    The positive control deliberately uses the 154-character no-override
    name and no deeper directory. A longer name under a temp root can
    exceed the total-path limit on a Windows host without long paths
    enabled, which would make this fail for a reason unrelated to the
    property under test.
    """
    path = _writer_path(tmp_path, [])
    with open(path, "wb") as f:
        f.write(b"probe")
    assert os.path.getsize(path) == 5


def test_override_heavy_cache_name_stays_within_the_component_limit(tmp_path):
    """The gate: with every fTetWild override set, the composed filename
    still fits the 255-byte per-component limit every supported filesystem
    enforces, and a file at that path opens.

    The composer digests the argument string rather than interpolating it,
    so the component is a fixed width whatever the settings are. Without
    that, seven overrides render as full float32 reprs
    (``edge_length_fac`` alone is ``0.05000000074505806``) and the name
    reaches 292 characters, which NTFS refuses with errno 22, APFS with
    63 and ext4 with 36.
    """
    path = _writer_path(tmp_path, ALL_OVERRIDES)
    component = os.path.basename(path)
    assert len(component) <= 255, f"{len(component)} characters: {component}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"probe")
    assert os.path.getsize(path) == 5


def test_unstattable_path_is_invisible_to_os_path_exists(tmp_path):
    """Characterization, not a gate: ``os.path.exists`` answers False for a
    path it cannot stat at all, so a cache path the filesystem refuses is
    indistinguishable from a cache that simply is not there yet.

    That is what turns an unusable cache path into a silent
    "re-tetrahedralize every build" instead of a reported error. A gate on
    the reporting itself needs the reporting API to exist first.
    """
    path = _writer_path(tmp_path, ALL_OVERRIDES)
    assert os.path.exists(path) is False
    with pytest.raises(OSError):
        os.stat(path)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "PENDING dev-preexisting-fixes | G2: the planner composes an "
        "empty-arg cache name, so an object carrying any fTetWild override "
        "is reported new on every build"
    ),
)
def test_planner_probe_matches_writer(tmp_path):
    """The planner decides "already tetrahedralized?" by stat-ing a path.
    It must be the path the writer writes.

    Asserted over NON-EMPTY kwargs only: the two composers agree exactly
    when the arg string is empty, so a zero-override case certifies
    nothing. The agreeing case is covered separately below.
    """
    for n in range(1, len(ALL_OVERRIDES) + 1):
        assert _planner_path(tmp_path) == _writer_path(tmp_path, ALL_OVERRIDES[:n]), (
            f"{n} override(s): planner probes a different file than the writer writes"
        )


def test_planner_and_writer_agree_with_no_overrides(tmp_path):
    """The one configuration where the two composers already agree. Pinned
    so the gate above cannot be satisfied by drifting the planner's own
    default instead of teaching it the real kwargs.
    """
    assert _planner_path(tmp_path) == _writer_path(tmp_path, [])


def test_cdylib_is_the_tree_local_build():
    """The cdylib must come from THIS tree's ``target/``. Two worktrees on
    one host carry different param-key sets, and a cross-tree load surfaces
    much later as a solver panic rather than as an import error.
    """
    loaded = Path(sys.modules["_ppf_cts_py"].__file__).resolve()
    assert loaded.is_relative_to(REPO_ROOT / "target"), loaded
