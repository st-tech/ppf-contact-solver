# File: _force_field_noise_copies_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The force field noise exists twice in Python, byte for byte:
# `frontend/_noise_.py`, which the script compiler cross-checks against, and
# `blender_addon/core/noise.py`, which the add-on draws and samples with
# (the add-on ships without the frontend). The solver's kernel runs the same
# algorithm, and `rig_force_field_noise` holds the kernel to it. A drift
# between the two Python copies would make the add-on draw one field and the
# solver run another, so the copies are compared whole.

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_two_noise_copies_are_identical():
    frontend = (ROOT / "frontend" / "_noise_.py").read_bytes()
    addon = (ROOT / "blender_addon" / "core" / "noise.py").read_bytes()
    assert frontend == addon, (
        "frontend/_noise_.py and blender_addon/core/noise.py differ; the "
        "noise is one algorithm, so edit one and copy it over the other"
    )


def test_the_two_script_api_copies_are_identical():
    # The list of what a script may call: the compiler's errors and
    # `builtins()` read one copy, the add-on's popup, MCP and Python API the
    # other, so a drift would promise a call the solver refuses.
    frontend = (ROOT / "frontend" / "_script_api_.py").read_bytes()
    addon = (ROOT / "blender_addon" / "core" / "script_api.py").read_bytes()
    assert frontend == addon, (
        "frontend/_script_api_.py and blender_addon/core/script_api.py differ; "
        "edit one and copy it over the other"
    )
