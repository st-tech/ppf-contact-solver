# File: addon_host_tests/_command_output_lines_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A COMMAND'S OUTPUT KEEPS THE LEADING TAB ON ITS FIRST LINE
# (``blender_addon/core/backends.py``, ``command_lines``).
#
# WHY THIS IS WORTH A GATE. The build probe
# (``core.remote_builds.probe_command``) prints ``<marker><TAB><directory>``
# for every directory that holds a server, and a directory with NO
# ``.ppf-backend`` marker prints its separator with nothing before it. That is
# the shape of a downloaded distribution, which the probe's own docstring names
# as the expected case rather than an odd one.
#
# ``parse_listing`` rstrips line endings ONLY, deliberately, and says in a
# comment why a general strip would lose that separator and make the directory
# vanish from the listing. Every backend then undid it one layer up with
# ``.strip()`` on the whole output before ``splitlines()``, which takes the
# leading TAB off the FIRST line. A host holding exactly one unmarked build was
# reported as holding none, and the panel drew "No solver build found under
# ...", which reads as a wrong path rather than as a missing marker.
#
# MEASURED, NOT IMAGINED. Against a container serving an unmarked build the
# probe printed ``"\t/root/ppf-contact-solver/target/release"``, the add-on
# cached ``{}``, and both device rows went dead.
#
# THE PAIR IS WHAT MATTERS, so the listing cases below drive the real
# ``parse_listing`` rather than asserting on the lines alone: the property is
# that what a backend returns is what that parser can read.

from __future__ import annotations

import importlib.util
import os

import pytest


def _load(relative, name):
    """Load one add-on module by path.

    Importing it as part of the package would execute the add-on's
    ``__init__``, which imports ``bpy``; these modules import only the standard
    library, so loading each file on its own is the whole of what they need.
    """
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "blender_addon", relative,
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


remote_builds = _load(os.path.join("core", "remote_builds.py"), "ppf_remote_builds")


def command_lines(text):
    """``backends.command_lines``, loaded without the package.

    ``backends.py`` imports the add-on package, so the function is read out of
    the source and executed on its own. Reading it from the file is what keeps
    this gate pointed at the shipping definition rather than a copy.
    """
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "blender_addon", "core", "backends.py",
    )
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    start = source.index("def command_lines(")
    end = source.index("\n\n\n", start)
    namespace: dict = {}
    exec(compile(source[start:end], path, "exec"), namespace)  # noqa: S102
    return namespace["command_lines"](text)


# ---------------------------------------------------------------------------
# The line splitting itself
# ---------------------------------------------------------------------------

def test_an_unmarked_first_line_keeps_its_separator():
    """The case that shipped broken: one build, no marker."""
    assert command_lines("\t/opt/ppf/target/release\n") == [
        "\t/opt/ppf/target/release"
    ]


def test_a_marked_line_is_unchanged():
    assert command_lines("cuda\t/opt/ppf/target/release\n") == [
        "cuda\t/opt/ppf/target/release"
    ]


def test_trailing_newline_adds_no_empty_line():
    """``splitlines`` does not invent a final empty entry, so nothing strips
    the tail to compensate."""
    assert command_lines("cpu\n") == ["cpu"]
    assert command_lines("a\nb\n") == ["a", "b"]


def test_windows_line_endings_are_removed():
    """A marker written by ``echo`` on Windows carries ``\\r\\n``."""
    assert command_lines("cpu\r\n") == ["cpu"]


def test_empty_output_is_no_lines():
    assert command_lines("") == []
    assert command_lines("\n") == []


def test_interior_whitespace_survives():
    """Only line endings come off. A path with a space in it is a path."""
    assert command_lines("\t/opt/my builds/target/release\n") == [
        "\t/opt/my builds/target/release"
    ]


# ---------------------------------------------------------------------------
# What the pair has to do together
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        # One unmarked build: the exact failure, end to end.
        (
            "\t/opt/ppf/target/release\n",
            {"/opt/ppf/target/release": ""},
        ),
        # Unmarked FIRST, marked second. Only the first line is at risk from a
        # whole-output strip, so a marked line ahead of it used to hide this.
        (
            "\t/opt/a/target/release\ncuda\t/opt/a/target/cuda/release\n",
            {
                "/opt/a/target/release": "",
                "/opt/a/target/cuda/release": "cuda",
            },
        ),
        # Both marked, which never broke and must not start.
        (
            "cuda\t/opt/a/target/release\ncpu\t/opt/a/target/cpu/release\n",
            {
                "/opt/a/target/release": "cuda",
                "/opt/a/target/cpu/release": "cpu",
            },
        ),
        # A marker with Windows line endings, stripped by parse_listing.
        (
            "cpu\r\n\t/opt/a/target/release\n",
            {"/opt/a/target/release": ""},
        ),
    ],
)
def test_a_backend_s_lines_are_what_parse_listing_can_read(raw, expected):
    assert remote_builds.parse_listing(command_lines(raw)) == expected


def test_the_whole_output_strip_that_shipped_would_fail_these():
    """The defect, stated as a test so the fix is not mistaken for a no-op.

    This is what every backend did before: ``.strip()`` on the whole output and
    then ``splitlines()``. It is asserted to LOSE the directory, which is what
    makes the cases above meaningful rather than tautological.
    """
    raw = "\t/opt/ppf/target/release\n"
    old = raw.strip().splitlines()
    assert remote_builds.parse_listing(old) == {}
    assert remote_builds.parse_listing(command_lines(raw)) != {}
