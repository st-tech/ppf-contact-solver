# File: scenarios/bl_connection_default_native.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# A new scene's connection Type is the native type of the platform Blender is
# running on: Windows Native on Windows, macOS Native on macOS, Linux Native on
# Linux. Of the three native types only this platform's can work here, so it
# is the one a new scene starts on.
#
# The expected type is derived here from ``sys.platform`` with its own table,
# not read from the add-on, so a default computed from the wrong platform test
# fails instead of agreeing with itself. The scenario runs on every platform
# and the answer differs on each, which is why CI runs it natively on all
# three: one platform's green run says nothing about the other two.
#
# Checks:
#
#   * the ``server_type`` property's declared default is this platform's
#     native type, and its item is labelled "<OS> Native";
#   * a scene created now, whose field nobody has written, reads that type
#     and stores no value for it (the default is what it reads);
#   * a scene whose Type was set explicitly keeps it: the default changes
#     only what an unwritten field reads.
#
# Nothing here connects or starts a server.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True
# Property defaults only; no solver is involved.
BACKENDS = ("real",)


_DRIVER_BODY = r'''
import sys
import traceback

result.setdefault("errors", [])
result.setdefault("checks", {})


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


if sys.platform == "win32":
    expected, expected_label = "WIN_NATIVE", "Windows Native"
elif sys.platform == "darwin":
    expected, expected_label = "MAC_NATIVE", "macOS Native"
else:
    expected, expected_label = "LINUX_NATIVE", "Linux Native"

fresh = None
try:
    state_mod = __import__(pkg + ".ui.state", fromlist=["SSHState"])
    groups = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])

    prop = state_mod.SSHState.bl_rna.properties["server_type"]
    record("declared_default_is_this_platforms_native",
           prop.default == expected,
           {"platform": sys.platform, "default": prop.default,
            "expected": expected})
    labels = {item.identifier: item.name for item in prop.enum_items}
    record("default_item_is_labelled_os_native",
           labels.get(prop.default) == expected_label,
           {"label": labels.get(prop.default), "expected": expected_label})

    fresh = bpy.data.scenes.new("ppf_default_native")
    props = groups.get_addon_data(fresh).ssh_state
    record("new_scene_reads_this_platforms_native",
           props.server_type == expected,
           {"server_type": props.server_type, "expected": expected})
    record("new_scene_stores_no_type",
           "server_type" not in props.keys(),
           {"keys": sorted(props.keys())})

    props.server_type = "CUSTOM"
    record("explicit_type_is_kept",
           props.server_type == "CUSTOM" and "server_type" in props.keys(),
           {"server_type": props.server_type})
except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
finally:
    if fresh is not None:
        bpy.data.scenes.remove(fresh)
'''


def build_driver(ctx: r.ScenarioContext) -> str:
    """Return the Python source the bootstrap will exec inside Blender.

    No substitutions are needed: the scenario reads the property's declared
    default and a fresh scene's field, and reaches no server.
    """
    return _DRIVER_BODY


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
