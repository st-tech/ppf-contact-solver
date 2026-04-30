# File: scenarios/bl_copy_paste_material_params.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Material Params copy/paste roundtrip.
#
# The Material Params box exposes COPYDOWN / PASTEDOWN buttons that
# bind to ``object.copy_material_params`` / ``object.paste_material_params``.
# Copy snapshots every scalar RNA field on the ObjectGroup not listed in
# ``MATERIAL_CLIPBOARD_EXCLUDE`` (identity, profile bindings, UI toggles,
# dynamic enums, per-assigned-object collections); paste applies the
# snapshot back, filtered by the source's ``object_type`` so cross-type
# pastes don't pollute the destination's unused model-specific fields.
#
# This scenario exercises the same-group roundtrip: build a SHELL
# group, perturb every copyable field to value-set A, copy, perturb
# again to a distinct value-set B, paste, and assert that every field
# the filter accepts (shared scalars and ``shell_*``) lands on the A
# value, while every field the filter rejects (``solid_*``, ``rod_*``)
# stays at B because the paste skipped it. ``object_type`` is held at
# SHELL throughout so the filter behavior is deterministic; cross-type
# behavior has its own scenario (bl_copy_paste_cross_type_material).

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
    # Deterministic per-prop value distinct from current, respecting
    # hard_min/hard_max so Blender doesn't silently clamp us back to
    # current and break the roundtrip assertion.
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


try:
    introspect = __import__(pkg + ".core.param_introspect",
                            fromlist=["MATERIAL_CLIPBOARD_EXCLUDE",
                                      "list_copyable_params",
                                      "material_param_applies"])

    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="MatPlane")
    dh.save_blend(PROBE_DIR, "matparams.blend")
    root = dh.configure_state(project_name="copy_paste_material", frame_count=2)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    group = root.object_group_0
    dh.log(f"group ready: type={group.object_type} name={group.name}")

    # Enumerate every copyable scalar field. We deliberately keep
    # object_type stable (SHELL throughout) so the paste filter accepts
    # every field; cross-type filter behavior has its own scenario.
    names = sorted(set(introspect.list_copyable_params(
        group, exclude=introspect.MATERIAL_CLIPBOARD_EXCLUDE,
    )) - {"object_type"})
    dh.log(f"copyable_param_count={len(names)}")
    rna = group.bl_rna

    # Pass 1: perturb to value-set A and capture snapshot_a.
    snapshot_a = {}
    for name in names:
        prop = rna.properties[name]
        try:
            setattr(group, name, _perturb(prop, getattr(group, name), 1))
        except Exception:
            pass
        snapshot_a[name] = _read(group, name, prop)

    # Copy A into the WindowManager clipboard.
    bpy.ops.object.copy_material_params(group_index=0)
    wm = bpy.context.window_manager
    dh.record(
        "clipboard_valid_after_copy",
        bool(getattr(wm, "material_clipboard_valid", False))
        and getattr(wm, "material_clipboard_src_type", "") == "SHELL",
        {"valid": bool(getattr(wm, "material_clipboard_valid", False)),
         "src_type": getattr(wm, "material_clipboard_src_type", "")},
    )

    # Pass 2: perturb to a distinct value-set B and verify each field
    # actually moved away from A. Confirms the pre-paste state is the
    # correct "different" baseline before we re-paste A.
    snapshot_b = {}
    diffs_pre_paste = []
    for name in names:
        prop = rna.properties[name]
        try:
            setattr(group, name, _perturb(prop, getattr(group, name), 2))
        except Exception:
            pass
        snapshot_b[name] = _read(group, name, prop)
        if _values_close(snapshot_a[name], snapshot_b[name]):
            diffs_pre_paste.append(name)
    dh.record(
        "perturbation_b_distinguishes_from_a",
        len(diffs_pre_paste) <= max(2, len(names) // 20),
        {"unmoved_count": len(diffs_pre_paste),
         "examples": diffs_pre_paste[:6],
         "total": len(names)},
    )

    # Paste A back. Fields the filter accepts (shared + shell_* given a
    # SHELL source) must land on snapshot_a; fields the filter rejects
    # (solid_*, rod_*) are skipped by paste and stay at snapshot_b. Both
    # behaviors are documented in core/param_introspect.material_param_applies.
    bpy.ops.object.paste_material_params(group_index=0)
    mismatched_round_trip = []
    incorrectly_clobbered = []
    accepted_count = 0
    rejected_count = 0
    for name in names:
        prop = rna.properties[name]
        post = _read(group, name, prop)
        if introspect.material_param_applies(name, "SHELL"):
            accepted_count += 1
            if not _values_close(snapshot_a[name], post):
                mismatched_round_trip.append({
                    "name": name,
                    "expected_a": snapshot_a[name],
                    "actual": post,
                    "type": prop.type,
                })
        else:
            rejected_count += 1
            if not _values_close(snapshot_b[name], post):
                incorrectly_clobbered.append({
                    "name": name,
                    "expected_b": snapshot_b[name],
                    "actual": post,
                    "type": prop.type,
                })
    dh.record(
        "filter_accepted_fields_round_trip",
        not mismatched_round_trip,
        {"mismatched_count": len(mismatched_round_trip),
         "mismatches": mismatched_round_trip[:8],
         "accepted_count": accepted_count},
    )
    dh.record(
        "filter_rejected_fields_skipped_by_paste",
        not incorrectly_clobbered,
        {"clobbered_count": len(incorrectly_clobbered),
         "clobbered": incorrectly_clobbered[:8],
         "rejected_count": rejected_count},
    )

    # Identity / per-object fields excluded from the clipboard must stay
    # whatever they were on the destination, even after a paste.
    sentinel_uuid = group.uuid
    sentinel_name = group.name
    sentinel_index = group.index
    bpy.ops.object.paste_material_params(group_index=0)
    dh.record(
        "identity_fields_preserved_across_paste",
        group.uuid == sentinel_uuid
        and group.name == sentinel_name
        and group.index == sentinel_index,
        {"uuid_kept": group.uuid == sentinel_uuid,
         "name_kept": group.name == sentinel_name,
         "index_kept": group.index == sentinel_index},
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
