# File: scenarios/bl_enum_props_guard.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Structural-enforcement guard for the dynamic-EnumProperty freed-memory bug.
#
# models/enum_props.py makes the "keep a reference to the items list" rule
# impossible to forget: a dynamic items callback MUST be wrapped with
# @dynamic_enum_items, and the project EnumProperty() raises TypeError at
# import time if handed an undecorated callable. This scenario verifies that
# machinery is in force and cannot be bypassed:
#
#   A. guard_raises_on_undecorated: EnumProperty(items=<plain callable>) raises
#      TypeError (the loud failure a forgotten decorator produces at load).
#   B. guard_accepts_decorated_and_static: a decorated callback and a static
#      list both pass through without error.
#   C. no_raw_bpy_props_enumproperty: an AST scan of the installed addon finds
#      no `bpy.props.EnumProperty` and no `from bpy.props import EnumProperty`
#      outside models/enum_props.py, so nothing bypasses the guarded wrapper.
#   D. all_dynamic_callbacks_retained: every dynamic items callback wired into
#      the addon carries the retention mark.
#
# Pure UI scenario: no server, no solver, no transfer.

from __future__ import annotations


from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_TEMPLATE = r"""
import ast, os, sys, time, traceback
import bpy
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def log(msg):
    result["phases"].append((round(time.time(), 3), msg))


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


try:
    enum_mod = __import__(pkg + ".models.enum_props",
                          fromlist=["EnumProperty", "dynamic_enum_items"])
    groups_mod = __import__(pkg + ".models.groups",
                            fromlist=["get_vertex_group_items"])
    og_mod = __import__(pkg + ".ui.object_group",
                        fromlist=["ObjectGroup"])
    state_mod = __import__(pkg + ".ui.state",
                           fromlist=["get_snap_objects"])

    EnumProperty = enum_mod.EnumProperty
    dynamic_enum_items = enum_mod.dynamic_enum_items

    # ----- A: an undecorated callable is rejected loudly --------------
    def _undecorated(self, context):
        return [("X", "x", "")]

    raised = False
    try:
        EnumProperty(items=_undecorated)
    except TypeError:
        raised = True
    record("A_guard_raises_on_undecorated", raised)

    # ----- B: a decorated callback actually REGISTERS ------------------
    # register_class is where Blender validates the items callback's
    # (self, context) signature; merely constructing the deferred does not.
    # This subtest therefore catches a decorator that produces a
    # Blender-invalid signature (e.g. a *args wrapper), plus a static list.
    @dynamic_enum_items
    def _probe_items(self, context):
        return [("A", "a", ""), ("B", "b", "")]

    class _PPFEnumProbe(bpy.types.PropertyGroup):
        dyn_sel: EnumProperty(items=_probe_items)
        static_sel: EnumProperty(items=[("A", "a", ""), ("B", "b", "")])

    # register_class is where Blender rejects a bad items signature (the same
    # step that failed ObjectGroup when the wrapper took *args), so a clean
    # register/unregister of the probe is the signature validation.
    reg_ok = True
    detail_b = {}
    try:
        bpy.utils.register_class(_PPFEnumProbe)
    except Exception as e:  # noqa: BLE001
        reg_ok = False
        detail_b = {"error": repr(e)}
    finally:
        try:
            bpy.utils.unregister_class(_PPFEnumProbe)
        except Exception:  # noqa: BLE001
            pass
    record("B_decorated_callback_registers", reg_ok, detail_b)

    # ----- C: no raw bpy.props.EnumProperty anywhere in the addon -----
    root = os.path.dirname(sys.modules[pkg].__file__)
    offenders = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            # enum_props.py is the ONE sanctioned caller of bpy.props.EnumProperty.
            if os.path.normpath(path).endswith(os.path.join("models", "enum_props.py")):
                continue
            try:
                tree = ast.parse(open(path, encoding="utf-8").read())
            except (SyntaxError, OSError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "bpy.props":
                    for alias in node.names:
                        if alias.name == "EnumProperty":
                            offenders.append(
                                os.path.relpath(path, root) + ":" + str(node.lineno)
                                + " imports EnumProperty from bpy.props")
                if isinstance(node, ast.Attribute) and node.attr == "EnumProperty":
                    v = node.value
                    if (isinstance(v, ast.Attribute) and v.attr == "props"
                            and isinstance(v.value, ast.Name) and v.value.id == "bpy"):
                        offenders.append(
                            os.path.relpath(path, root) + ":" + str(node.lineno)
                            + " uses bpy.props.EnumProperty")
    record("C_no_raw_bpy_props_enumproperty", not offenders,
           {"offenders": offenders})

    # ----- D: every dynamic items callback carries the retention mark -
    MARK = "_ppf_retained_enum_items"
    OG = og_mod.ObjectGroup
    dynamic_callbacks = {
        "groups.get_vertex_group_items": groups_mod.get_vertex_group_items,
        "og._get_material_profile_items": og_mod._get_material_profile_items,
        "og._get_material_preset_items": og_mod._get_material_preset_items,
        "og._get_pin_profile_items": og_mod._get_pin_profile_items,
        "og.ObjectGroup._get_velocity_object_items": OG._get_velocity_object_items,
        "og.ObjectGroup._get_collision_window_object_items": OG._get_collision_window_object_items,
        "og.ObjectGroup._get_tet_object_items": OG._get_tet_object_items,
        "og.ObjectGroup._get_pdrd_hinge_object_items": OG._get_pdrd_hinge_object_items,
        "og.ObjectGroup._get_bend_ref_object_items": OG._get_bend_ref_object_items,
        "state._get_profile_items": state_mod._get_profile_items,
        "state._get_scene_profile_items": state_mod._get_scene_profile_items,
        "state.get_snap_objects": state_mod.get_snap_objects,
    }
    unretained = [name for name, fn in dynamic_callbacks.items()
                  if not getattr(fn, MARK, False)]
    record("D_all_dynamic_callbacks_retained",
           not unretained and len(dynamic_callbacks) == 12,
           {"checked": len(dynamic_callbacks), "unretained": unretained})

    log("checks=" + str(len(result["checks"])) + " done")
except Exception as exc:
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
