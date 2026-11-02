# File: _force_field_noise_copies_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# The force field noise exists twice in Python, identical but for the license
# line:
# `frontend/_noise_.py`, which the script compiler cross-checks against, and
# `blender_addon/core/noise.py`, which the add-on draws and samples with
# (the add-on ships without the frontend). The solver's kernel runs the same
# algorithm, and `rig_force_field_noise` holds the kernel to it. A drift
# between the two Python copies would make the add-on draw one field and the
# solver run another, so the copies are compared whole.

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _without_license(path):
    # Each copy carries the license of the directory it sits in, so the one
    # line allowed to differ is the `# License:` header line.
    return b"".join(
        line
        for line in path.read_bytes().splitlines(keepends=True)
        if not line.startswith(b"# License: ")
    )


def test_the_two_noise_copies_are_identical():
    frontend = _without_license(ROOT / "frontend" / "_noise_.py")
    addon = _without_license(ROOT / "blender_addon" / "core" / "noise.py")
    assert frontend == addon, (
        "frontend/_noise_.py and blender_addon/core/noise.py differ; the "
        "noise is one algorithm, so edit one and copy it over the other, "
        "keeping each file's own `# License:` line"
    )


def test_the_two_script_api_copies_are_identical():
    # The list of what a script may call: the compiler's errors and
    # `builtins()` read one copy, the add-on's popup, MCP and Python API the
    # other, so a drift would promise a call the solver refuses.
    frontend = _without_license(ROOT / "frontend" / "_script_api_.py")
    addon = _without_license(ROOT / "blender_addon" / "core" / "script_api.py")
    assert frontend == addon, (
        "frontend/_script_api_.py and blender_addon/core/script_api.py differ; "
        "edit one and copy it over the other, keeping each file's own "
        "`# License:` line"
    )
