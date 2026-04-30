# File: __init__.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Host-side debug/control tools for the Blender addon.

Layout:
    client.py   — Transport + control primitives (pure stdlib, no Blender dep).
    output.py   — Shared response-printing helpers used by both CLIs.
    main.py     — General CLI (status, tools, call, exec, reload, scene, ...).
    perf.py     — Draw-time profiler CLI, drives ``ui/perf.py`` in Blender.

Invocation (as scripts, from any cwd):
    python blender_addon/debug/main.py status
    python blender_addon/debug/perf.py enable
"""
