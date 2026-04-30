# File: scenarios/bl_copy_paste_cross_type_material.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Material Params copy/paste with type-bound and per-group filters.
#
# The Material Params clipboard skips two classes of fields:
#
#   1. Identity / per-object-bound entries (uuid, name, index, the
#      assigned-object selectors, profile bindings, UI toggles). These
#      are listed in ``MATERIAL_CLIPBOARD_EXCLUDE`` and never go on or
#      come off the clipboard, so a paste leaves them at whatever the
#      destination already had.
#
#   2. Model-specific scalars whose name prefix doesn't match the
#      source's ``object_type``. ``material_param_applies`` rejects
#      ``solid_*`` from a SHELL source, ``shell_*`` from a SOLID
#      source, etc., so a cross-type paste does not pollute the
#      destination's relevant model fields with the source's unused
#      defaults.
#
# This scenario exercises a SHELL -> SOLID paste and asserts:
#
#   - Shared scalar fields (no model prefix) on B now match A.
#   - shell_* fields on B were overwritten with A's values (the filter
#     allows them because the source is SHELL).
#   - solid_* fields on B retained their pre-paste values (the filter
#     rejected them because the source is SHELL, not SOLID).
#   - rod_* fields on B retained their pre-paste values.
#   - All MATERIAL_CLIPBOARD_EXCLUDE fields on B retained their pre-
#     paste values (identity, profile bindings, UI toggles).

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def _perturb(prop, current, salt):
    t = prop.type
    if t == "BOOLEAN":
        return not bool(current)
    if t == "INT":
        lo = int(getattr(prop, "hard_min", -2**31))
        hi = int(getattr(prop, "hard_max", 2**31 - 1))
        for delta in (salt + 3, salt + 7, -salt - 5, 1, -1):
            cand = max(lo, min(hi, int(current) + delta))
            if cand != int(current):
                return cand
        return int(current)
    if t == "FLOAT":
        lo = float(getattr(prop, "hard_min", -1e30))
        hi = float(getattr(prop, "hard_max", 1e30))
        n = getattr(prop, "array_length", 0)
        if n > 0:
            out = []
            for i, c in enumerate(current):
                step = 0.137 + 0.071 * i + 0.013 * salt
                cand = max(lo, min(hi, float(c) + step))
                if cand == float(c):
                    cand = max(lo, min(hi, float(c) - step))
                out.append(cand)
            return tuple(out)
        step = 0.317 + 0.041 * salt
        cand = max(lo, min(hi, float(current) + step))
        if cand == float(current):
            cand = max(lo, min(hi, float(current) - step))
        return cand
    if t == "ENUM":
        items = [it.identifier for it in prop.enum_items]
        others = [c for c in items if c != current]
        if not others:
            return current
        return others[salt % len(others)]
    if t == "STRING":
        return f"{current or ''}#perturb{salt}"
    return current


def _values_close(a, b, tol=1e-4):
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(
            _values_close(x, y, tol) for x, y in zip(a, b)
        )
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= tol
        except (TypeError, ValueError):
            return a == b
    return a == b


def _read(group, name, prop):
    v = getattr(group, name)
    if prop.type == "FLOAT" and getattr(prop, "array_length", 0) > 0:
        return tuple(v)
    return v


def _make_plane(name, location):
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=location)
    obj = bpy.context.active_object
    obj.name = name
    vg = obj.vertex_groups.new(name="AllPin")
    vg.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")
    return obj


try:
    introspect = __import__(pkg + ".core.param_introspect",
                            fromlist=["MATERIAL_CLIPBOARD_EXCLUDE",
                                      "list_copyable_params"])

    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    plane_a = _make_plane("PlaneA", (-1.5, 0.0, 0.0))
    plane_b = _make_plane("PlaneB", (1.5, 0.0, 0.0))
    dh.save_blend(PROBE_DIR, "crosstype.blend")
    root = dh.configure_state(project_name="copy_paste_cross_type", frame_count=2)

    grp_a = dh.api.solver.create_group("Cloth", "SHELL")
    grp_a.add(plane_a.name)
    grp_b = dh.api.solver.create_group("Block", "SOLID")
    grp_b.add(plane_b.name)
    a = root.object_group_0
    b = root.object_group_1
    dh.log(f"groups ready: A.type={a.object_type} B.type={b.object_type}")

    rna = a.bl_rna
    # All scalar fields on the ObjectGroup, INCLUDING identity / UI
    # toggles. We need to read the excluded ones to assert they were
    # NOT touched by the paste. ``object_type`` is held stable on both
    # groups so the filter behavior is deterministic.
    all_scalar = []
    for prop in rna.properties:
        if prop.type not in ("BOOLEAN", "INT", "FLOAT", "STRING", "ENUM"):
            continue
        if getattr(prop, "is_readonly", False):
            continue
        all_scalar.append(prop.identifier)
    all_scalar = sorted(set(all_scalar) - {"object_type"})

    excluded = set(introspect.MATERIAL_CLIPBOARD_EXCLUDE)
    copyable = sorted(set(introspect.list_copyable_params(
        a, exclude=introspect.MATERIAL_CLIPBOARD_EXCLUDE,
    )) - {"object_type"})

    def _classify(name):
        if name.startswith("solid_"):
            return "solid"
        if name.startswith("shell_"):
            return "shell"
        if name.startswith("rod_"):
            return "rod"
        return "shared"

    # Perturb A and B independently. ``salt=1`` for A, ``salt=2`` for B
    # so each copyable field ends up with distinct values, including
    # solid_* / rod_* on B (which the paste must NOT clobber).
    for name in copyable:
        prop = rna.properties[name]
        try:
            setattr(a, name, _perturb(prop, getattr(a, name), 1))
        except Exception:
            pass
        try:
            setattr(b, name, _perturb(prop, getattr(b, name), 2))
        except Exception:
            pass

    # Capture B's pre-paste state for ALL scalar fields (including
    # excluded identity / UI fields that must be preserved).
    b_before = {n: _read(b, n, rna.properties[n]) for n in all_scalar}
    a_after_perturb = {n: _read(a, n, rna.properties[n]) for n in copyable}

    # Copy A, paste into B.
    bpy.ops.object.copy_material_params(group_index=0)
    wm = bpy.context.window_manager
    dh.record(
        "clipboard_src_type_is_SHELL",
        getattr(wm, "material_clipboard_src_type", "") == "SHELL"
        and bool(getattr(wm, "material_clipboard_valid", False)),
        {"src_type": getattr(wm, "material_clipboard_src_type", ""),
         "valid": bool(getattr(wm, "material_clipboard_valid", False))},
    )
    bpy.ops.object.paste_material_params(group_index=1)

    b_after = {n: _read(b, n, rna.properties[n]) for n in all_scalar}

    shared_transferred = []
    shared_unchanged = []
    shell_transferred = []
    shell_unchanged = []
    solid_clobbered = []
    solid_preserved = []
    rod_clobbered = []
    rod_preserved = []
    excluded_clobbered = []
    excluded_preserved = []

    for name in all_scalar:
        before = b_before[name]
        after = b_after[name]
        if name in excluded:
            if _values_close(before, after):
                excluded_preserved.append(name)
            else:
                excluded_clobbered.append({"name": name,
                                           "before": before,
                                           "after": after})
            continue
        klass = _classify(name)
        a_val = a_after_perturb.get(name)
        if klass in ("shared", "shell"):
            # Filter passes; expect B to now match A.
            if _values_close(after, a_val):
                (shared_transferred if klass == "shared"
                 else shell_transferred).append(name)
            else:
                (shared_unchanged if klass == "shared"
                 else shell_unchanged).append({"name": name,
                                               "expected_from_a": a_val,
                                               "actual": after})
        else:  # solid / rod
            # Filter rejects; expect B to retain its pre-paste value.
            if _values_close(after, before):
                (solid_preserved if klass == "solid"
                 else rod_preserved).append(name)
            else:
                (solid_clobbered if klass == "solid"
                 else rod_clobbered).append({"name": name,
                                             "before_b": before,
                                             "after": after,
                                             "from_a": a_val})

    dh.record(
        "shared_fields_transferred_from_A",
        not shared_unchanged and len(shared_transferred) > 0,
        {"transferred_count": len(shared_transferred),
         "untransferred": shared_unchanged[:8]},
    )
    dh.record(
        "shell_fields_transferred_from_A",
        not shell_unchanged and len(shell_transferred) > 0,
        {"transferred_count": len(shell_transferred),
         "untransferred": shell_unchanged[:8]},
    )
    dh.record(
        "solid_fields_preserved_on_B",
        not solid_clobbered and len(solid_preserved) > 0,
        {"preserved_count": len(solid_preserved),
         "clobbered": solid_clobbered[:8]},
    )
    dh.record(
        "rod_fields_preserved_on_B",
        not rod_clobbered and len(rod_preserved) > 0,
        {"preserved_count": len(rod_preserved),
         "clobbered": rod_clobbered[:8]},
    )
    dh.record(
        "identity_and_ui_fields_preserved_on_B",
        not excluded_clobbered,
        {"preserved_count": len(excluded_preserved),
         "clobbered": excluded_clobbered[:8]},
    )

    # B's per-assigned-object collection survived: paste must not have
    # touched assigned_objects (a CollectionProperty, intrinsically
    # outside the scalar clipboard).
    dh.record(
        "assigned_objects_preserved_on_B",
        len(b.assigned_objects) == 1
        and b.assigned_objects[0].name == plane_b.name,
        {"count": len(b.assigned_objects),
         "names": [a.name for a in b.assigned_objects]},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
