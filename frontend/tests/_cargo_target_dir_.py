# File: frontend/tests/_cargo_target_dir_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A named CARGO_TARGET_DIR is the ONLY directory the frontend loads its build
# from.
#
# WHAT WENT WRONG WITHOUT IT. The frontend searched CARGO_TARGET_DIR and then
# fell back to ``<tree root>/target``, so a named directory that held no build
# silently loaded the tree's default one instead, and ``artifact_dir`` wrote
# THAT build's solver into the session launcher. The Blender add-on picks the
# CPU or the GPU build by naming its target directory in this variable, so a
# CPU run could execute the GPU solver, or the reverse, with nothing reporting
# the swap: each binary answers ``--backend`` honestly about itself.
#
# Each case imports ``frontend`` in a fresh interpreter, because the variable
# is read once, at import, and this process has already imported it.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_frontend(cargo_target_dir: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = cargo_target_dir
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", "import frontend; print(frontend.artifact_dir())"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_an_empty_named_directory_is_refused_rather_than_substituted(tmp_path):
    """THE BUG: an empty named directory must not fall back to the tree's own.

    Asserted whether or not the tree has a build of its own, because the
    fallback is only visible when it does: on a built tree the old code
    imported cleanly out of ``<tree root>/target``.
    """
    result = _import_frontend(str(tmp_path))
    assert result.returncode != 0, (
        f"frontend imported with an empty CARGO_TARGET_DIR, from "
        f"{result.stdout.strip()!r}: that is a build the caller did not name"
    )
    assert "CARGO_TARGET_DIR is set" in result.stderr, result.stderr
    assert str(tmp_path) in result.stderr, result.stderr
    assert str(REPO_ROOT / "target" / "release") not in result.stderr, (
        "the tree's own target directory was searched, so a named directory "
        "is not the only place the build is taken from"
    )


def test_a_relative_named_directory_is_read_from_the_tree_root(tmp_path):
    """Relative to the tree root, as Cargo reads it, not to the caller's cwd."""
    relative = os.path.join("target", f"no-such-build-{os.getpid()}")
    result = _import_frontend(relative)
    assert result.returncode != 0, result.stdout
    assert str(REPO_ROOT / relative) in result.stderr, result.stderr


def test_a_named_directory_holding_the_build_is_the_one_loaded():
    """The positive half: naming the tree's own target directory loads it.

    Needs a built tree, which is the only thing that can be loaded; skipped
    when the tree has none rather than passing over nothing.
    """
    target = REPO_ROOT / "target"
    if not (target / "release").is_dir():
        pytest.skip("this tree has no release build to load")
    result = _import_frontend(str(target))
    if result.returncode != 0 and "_ppf_cts_py extension not found" in result.stderr:
        pytest.skip("this tree's release directory holds no cdylib")
    assert result.returncode == 0, result.stderr
    loaded = result.stdout.strip().splitlines()[-1]
    assert os.path.samefile(loaded, target / "release"), loaded
