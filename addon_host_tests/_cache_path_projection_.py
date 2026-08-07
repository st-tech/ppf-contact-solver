# File: addon_host_tests/_cache_path_projection_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Host-side gates for the add-on's projection of the deepest cache path
# the build pipeline writes (``blender_addon/core/utils.py``).
#
# The projection exists so a Windows solver path that cannot hold the
# pipeline's files is refused in the panel, at the moment it is typed,
# rather than surfacing as an unrecognizable error deep inside a later
# Transfer. That only works if the projection is an UPPER bound on what
# the pipeline actually writes; a lower bound reports "fine" for exactly
# the objects most at risk.
#
# Two limits apply to the same path and they are different numbers:
#
#   * the total path length (``WINDOWS_MAX_PATH``, 260), waived when
#     Windows long-path support is on;
#   * the per-component length (255 bytes), which no filesystem this
#     project ships on waives, on any platform.
#
# The cache filename is one component, so the second limit is the one the
# override-heavy case reaches first.

from __future__ import annotations

import pytest


HASH = "f" * 64

# The seven per-object fTetWild overrides ``_encode_obj_tet_kwargs``
# forwards, rendered as ``str(v)`` over the value read out of the RNA
# property. The float fields are ``FloatProperty``, i.e. float32, so
# widening one to a Python float prints its full float64 image.
ALL_OVERRIDES = [
    ("edge_length_fac", "0.05000000074505806"),
    ("epsilon", "0.0010000000474974513"),
    ("stop_energy", "10.0"),
    ("num_opt_iter", "80"),
    ("optimize", "True"),
    ("simplify", "True"),
    ("coarsen", "False"),
]

MAX_COMPONENT_BYTES = 255


def _writer_component(cdylib, kwargs):
    """The cache filename ``Mesh.tetrahedralize`` composes for these kwargs."""
    name, _key = cdylib.mesh_tetra_cache_key(HASH, [], list(kwargs))
    path = cdylib.mesh_cache_path("cache", HASH, name)
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def test_projection_covers_the_default_cache_name(addon_utils, cdylib):
    """Baseline: with no override the projection already accounts for the
    exact filename the pipeline writes.
    """
    projected = addon_utils.projected_windows_cache_path_len("C:\\dev", "proj")
    assert projected > len(_writer_component(cdylib, []))


def test_projection_is_an_upper_bound_on_the_written_path(addon_utils, cdylib):
    """The projected length must be at least the length of the deepest path
    the pipeline writes, for every object the add-on can encode.

    An override-carrying object is the case that matters: its filename is
    what overruns, and the panel's warning is driven entirely by this
    number, so an under-report is a silent warning.
    """
    root = "C:\\dev"
    project = "proj"
    projected = addon_utils.projected_windows_cache_path_len(root, project)
    for n in range(1, len(ALL_OVERRIDES) + 1):
        component = _writer_component(cdylib, ALL_OVERRIDES[:n])
        assert projected >= len(root) + len(component), (
            f"{n} override(s): projection {projected} is under the "
            f"{len(component)}-character filename the pipeline writes"
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "PENDING dev-preexisting-fixes | G5: windows_path_too_long tests "
        "the 260-character total-path limit only, so a filename over the "
        "255-byte per-component limit is reported as acceptable"
    ),
)
def test_component_overrun_is_reported_even_under_a_short_root(addon_utils, cdylib):
    """A short solver path keeps the TOTAL under 260 while the filename
    alone is over 255. The filesystem refuses on the component, so the
    guard has to as well.

    This is the configuration the user's report came from: a short root,
    an override-heavy object, and no warning drawn.
    """
    root = "C:\\d"
    project = "p"
    component = _writer_component(cdylib, ALL_OVERRIDES)
    assert len(component) > MAX_COMPONENT_BYTES, "premise: the filename overruns"
    assert addon_utils.windows_path_too_long(root, project) is not None


def test_blank_path_is_not_reported(addon_utils):
    """An unset optional field is not an error. Pinned so a fix for the
    two gates above keeps returning None for a blank path, which callers
    chain against.
    """
    assert addon_utils.windows_path_too_long("", "proj") is None
    assert addon_utils.windows_path_too_long("   ", "proj") is None
