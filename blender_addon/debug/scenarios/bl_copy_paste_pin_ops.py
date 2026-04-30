# File: scenarios/bl_copy_paste_pin_ops.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Pin Operations copy/paste roundtrip.
#
# Pin Operations sit on a per-pin item ``PinVertexGroupItem.operations``
# CollectionProperty. The ops list grows with one of MOVE_BY / SPIN /
# SCALE / TORQUE / EMBEDDED_MOVE entries, each with its own scalar
# payload (delta, axis, angular velocity, center mode, flip, frame
# range, transition, ...). The Operations row exposes COPYDOWN /
# PASTEDOWN buttons bound to ``object.copy_pin_ops`` /
# ``object.paste_pin_ops``: copy snapshots the current pin's whole ops
# collection field-by-field via ``copy_scalar_props`` (skipping the
# ``show_*`` viewport-only flags), and paste deep-copies the snapshot
# back, replacing the destination pin's ops list.
#
# The scenario builds one pin item, fills its operations with one op
# per supported op_type (excluding EMBEDDED_MOVE which is auto-detected
# from keyframes), perturbs every scalar field on every op to value-
# set A, snapshots, copies, replaces the operations with a much smaller
# list of unrelated ops, then pastes. Every op (and every field on
# every op except those in PIN_OP_CLIPBOARD_EXCLUDE) must round-trip.
#
# The pin item's own scalar fields (name, object_uuid, vg_hash, the
# include / pull / duration toggles) are NOT part of the clipboard
# payload and must not be touched by the paste; the test pins them to
# sentinel values pre-paste and asserts they survive.

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


def _read(op, name, prop):
    v = getattr(op, name)
    if prop.type == "FLOAT" and getattr(prop, "array_length", 0) > 0:
        return tuple(v)
    return v


def _snapshot_op(op, names):
    rna = op.bl_rna
    return {n: _read(op, n, rna.properties[n]) for n in names}


try:
    introspect = __import__(pkg + ".core.param_introspect",
                            fromlist=["PIN_OP_CLIPBOARD_EXCLUDE",
                                      "list_copyable_params"])

    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="PinPlane")
    dh.save_blend(PROBE_DIR, "pinops.blend")
    root = dh.configure_state(project_name="copy_paste_pin_ops", frame_count=2)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")

    group = root.object_group_0
    pin_item = group.pin_vertex_groups[0]
    group.pin_vertex_groups_index = 0
    pin_item.operations.clear()

    # Author one op per non-auto op_type. EMBEDDED_MOVE is excluded:
    # it's reserved for the auto-detected keyframe path and pasting
    # one onto a non-keyframe pin would be meaningless.
    op_types = ["MOVE_BY", "SPIN", "SCALE", "TORQUE"]
    for ot in op_types:
        op = pin_item.operations.add()
        op.op_type = ot

    # Build the field name list once; same for every op.
    sample_op = pin_item.operations[0]
    op_names = sorted(set(introspect.list_copyable_params(
        sample_op, exclude=introspect.PIN_OP_CLIPBOARD_EXCLUDE,
    )))
    dh.log(f"op_field_count={len(op_names)} types={op_types}")

    # Pin item scalars NOT in the clipboard payload. Pin the ones we
    # care about to sentinel values so we can detect any leak from the
    # paste path.
    pin_item.included = False
    pin_item.use_pin_duration = True
    pin_item.pin_duration = 73
    pin_item.use_pull = True
    pin_item.pull_strength = 4.25
    sentinel_pin = {
        "name": pin_item.name,
        "object_uuid": pin_item.object_uuid,
        "vg_hash": pin_item.vg_hash,
        "included": pin_item.included,
        "use_pin_duration": pin_item.use_pin_duration,
        "pin_duration": pin_item.pin_duration,
        "use_pull": pin_item.use_pull,
        "pull_strength": pin_item.pull_strength,
    }

    # Pass A: perturb every scalar on every op (skipping clipboard-
    # excluded show_* viewport flags) and snapshot.
    for idx, op in enumerate(pin_item.operations):
        rna = op.bl_rna
        for name in op_names:
            prop = rna.properties[name]
            try:
                setattr(op, name, _perturb(prop, getattr(op, name), 1 + idx))
            except Exception:
                pass
    snapshot_a = [_snapshot_op(op, op_names) for op in pin_item.operations]

    # Copy.
    bpy.ops.object.copy_pin_ops(group_index=0)
    wm = bpy.context.window_manager
    dh.record(
        "clipboard_valid_after_copy",
        bool(getattr(wm, "pin_ops_clipboard_valid", False))
        and len(wm.pin_ops_clipboard.operations) == len(op_types),
        {"valid": bool(getattr(wm, "pin_ops_clipboard_valid", False)),
         "clipboard_op_count": len(wm.pin_ops_clipboard.operations)},
    )

    # Mutate the pin's ops list to a different shape: drop down to one
    # op of an unrelated type with its own arbitrary values. After the
    # paste, this should be wholly replaced.
    pin_item.operations.clear()
    rogue = pin_item.operations.add()
    rogue.op_type = "TORQUE"
    rogue.torque_magnitude = -99.0
    rogue.frame_start = 12345
    rogue.frame_end = 12399

    dh.record(
        "pre_paste_state_distinct_from_a",
        len(pin_item.operations) == 1
        and pin_item.operations[0].torque_magnitude == -99.0,
        {"len": len(pin_item.operations),
         "torque_magnitude": pin_item.operations[0].torque_magnitude},
    )

    # Paste.
    bpy.ops.object.paste_pin_ops(group_index=0)

    # Length must match.
    dh.record(
        "operations_length_matches_after_paste",
        len(pin_item.operations) == len(snapshot_a),
        {"actual": len(pin_item.operations),
         "expected": len(snapshot_a)},
    )

    # Field-by-field roundtrip across all ops.
    mismatched = []
    n = min(len(pin_item.operations), len(snapshot_a))
    for i in range(n):
        op = pin_item.operations[i]
        rna = op.bl_rna
        for name in op_names:
            expected = snapshot_a[i][name]
            actual = _read(op, name, rna.properties[name])
            if not _values_close(expected, actual):
                mismatched.append({
                    "op_index": i, "name": name,
                    "expected": expected, "actual": actual,
                    "type": rna.properties[name].type,
                })
    dh.record(
        "all_op_fields_round_trip_after_paste",
        not mismatched,
        {"mismatched_count": len(mismatched),
         "mismatches": mismatched[:8],
         "ops_compared": n},
    )

    # Excluded show_* viewport flags are not part of the clipboard, so
    # whatever they happen to be on the destination after paste is
    # whatever the new (default) PinOperation entries shipped with --
    # i.e. they should not match the source's perturbed True values.
    # We don't assert specific values here, only that the paste did
    # not promote excluded fields (any ``show_*`` ending up at the
    # source's True is allowed if the destination's PinOperation
    # default also happened to be True; what matters is that the
    # exclusion list above is honored for the *roundtrip* assertion,
    # which it is by construction).

    # Pin-item scalars must be untouched by the paste.
    pin_after = {
        "name": pin_item.name,
        "object_uuid": pin_item.object_uuid,
        "vg_hash": pin_item.vg_hash,
        "included": pin_item.included,
        "use_pin_duration": pin_item.use_pin_duration,
        "pin_duration": pin_item.pin_duration,
        "use_pull": pin_item.use_pull,
        "pull_strength": pin_item.pull_strength,
    }
    leaks = [k for k, v in sentinel_pin.items() if pin_after[k] != v]
    dh.record(
        "pin_item_identity_preserved_across_paste",
        not leaks,
        {"leaked_fields": leaks,
         "before": sentinel_pin, "after": pin_after},
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
