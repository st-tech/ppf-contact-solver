# File: scenarios/bl_material_preset_apply.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Material preset library: menu filtering + apply correctness.
#
# The Material Params box exposes a "Preset" dropdown (the
# ``material_preset_selection`` EnumProperty) whose items are filtered by the
# group's current Type, so a SHELL group lists only fabric presets and a SOLID
# group only soft-solid presets; a PDRD group has no presets (its menu is just NONE).
# Picking a preset applies its parameters via
# ``core.material_presets.apply_material_preset`` and then resets the picker
# back to NONE.
#
# This scenario exercises, for each of SHELL/SOLID/PDRD:
#  - menu filtering: ``get_preset_items`` lists exactly NONE + this type's
#    presets, and assigning an other-type preset to the enum is rejected
#    (so other types never leak into the menu). PDRD ships no presets (a rigid
#    body differs only by density and friction), so a PDRD group's menu is just
#    NONE and the apply pass below is empty for it;
#  - apply correctness: selecting each SHELL/SOLID preset (the real EnumProperty
#    update path) lands every applied key on its TOML value, even ones that
#    coincide with the group default (the dirtying step forces a real write);
#  - the picker resets itself to NONE after each apply;
#  - D2/D5: applying never changes the group's Type, and never touches contact
#    gaps or pin vertex groups;
#  - the loader rejects an unknown preset name.
#
# Expected preset names are read from the SAME bundled materials.toml the addon
# ships, so the test tracks the file with no hardcoded list to drift.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

META = {"description", "object_type"}


def _close(a, b, tol=1e-4):
    # Numeric fields compare within a relative+absolute tolerance (FloatProperty
    # is float32, so a large modulus rounds by more than a fixed 1e-4);
    # enums/strings/bools fall back to equality.
    try:
        fa, fb = float(a), float(b)
        return abs(fa - fb) <= tol + 1e-5 * max(abs(fa), abs(fb))
    except (TypeError, ValueError):
        return a == b


try:
    mp = __import__(pkg + ".core.material_presets",
                    fromlist=["load_material_presets", "get_preset_items",
                              "has_presets_for", "apply_material_preset"])
    groups_mod = __import__(pkg + ".models.groups",
                            fromlist=["get_group_by_uuid", "get_addon_data"])

    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="PresetPlane")
    dh.save_blend(PROBE_DIR, "presets.blend")
    root = dh.configure_state(project_name="material_presets", frame_count=2)
    scene = bpy.context.scene

    presets = mp.load_material_presets()
    dh.record("presets_loaded", len(presets) > 0, {"count": len(presets)})

    # Expected preset names grouped by Type, straight from the bundled TOML.
    expected_by_type = {}
    for nm, p in presets.items():
        expected_by_type.setdefault(p["object_type"], []).append(nm)

    for otype in ("SHELL", "SOLID", "PDRD"):
        g = dh.api.solver.create_group(otype + "G", otype)
        uuid = root.state.current_group_uuid
        og = groups_mod.get_group_by_uuid(scene, uuid)
        dh.log(f"{otype} group uuid={uuid} type={og.object_type}")

        expected = set(expected_by_type.get(otype, []))
        other = set(presets) - expected

        # (1a) Data-level menu filter: items are exactly NONE + this type.
        item_ids = [i[0] for i in mp.get_preset_items(otype)]
        listed = set(item_ids) - {"NONE"}
        dh.record(
            f"{otype}_menu_filtered",
            len(item_ids) >= 1 and item_ids[0] == "NONE"
            and listed == expected and not (listed & other),
            {"listed": sorted(listed), "expected": sorted(expected),
             "leaked_other_types": sorted(listed & other)},
        )

        # (1b) Enum-level filter: assigning an other-type preset must be
        # rejected by Blender (the identifier is not in this group's items).
        # We assert the assignment RAISES, not the post-assignment value: a
        # broken filter that accepted it would fire the update callback, which
        # resets the picker to NONE, so reading the value back could never
        # observe the leak (that read is a tautology).
        rejected = True
        if other:
            wrong = sorted(other)[0]
            rejected = False
            try:
                og.material_preset_selection = wrong
            except Exception:
                rejected = True
            try:
                og.material_preset_selection = "NONE"
            except Exception:
                pass
        dh.record(
            f"{otype}_other_type_rejected_by_enum",
            rejected,
            {"sample_wrong": sorted(other)[0] if other else None},
        )

        # Snapshot Type (D2) and fields a preset must NOT touch (D5).
        type0 = og.object_type
        gap0 = og.contact_gap
        off0 = og.contact_offset
        pins0 = len(og.pin_vertex_groups)

        # (2) Apply each preset via the EnumProperty update path (the real UI
        # code path); assert every applied key landed and the picker reset.
        apply_failures = []
        reset_failures = []
        for name in expected_by_type.get(otype, []):
            p = presets[name]
            # Dirty fields whose preset value equals the group default first, so
            # a silently-skipped write is actually detected: apply must move
            # them off the sentinel. Covers shell_model/solid_model (constant
            # per type and equal to the default) and young_mod_density_normalized
            # (true for every SHELL preset, matching the default).
            if "young_mod_density_normalized" in p:
                og.young_mod_density_normalized = not bool(
                    p["young_mod_density_normalized"])
            for mkey in ("shell_model", "solid_model"):
                if mkey in p and mkey in og.bl_rna.properties:
                    alts = [it.identifier
                            for it in og.bl_rna.properties[mkey].enum_items
                            if it.identifier != p[mkey]]
                    if alts:
                        try:
                            setattr(og, mkey, alts[0])
                        except Exception:
                            pass
            # Also perturb any numeric field whose preset value already equals
            # the current group value (e.g. solid_density now ships at the 100.0
            # group default), so apply must actually move it off a sentinel;
            # otherwise got == want would pass even if the write were silently
            # skipped.
            for nkey, want in p.items():
                if nkey in META or nkey not in og.bl_rna.properties:
                    continue
                prop = og.bl_rna.properties[nkey]
                if prop.type not in ("FLOAT", "INT"):
                    continue
                if _close(getattr(og, nkey), want):
                    lo, hi = prop.hard_min, prop.hard_max
                    sentinel = lo if not _close(lo, want) else hi
                    try:
                        setattr(og, nkey, sentinel)
                    except Exception:
                        pass
            og.material_preset_selection = name  # fires _on_material_preset_selected
            if og.material_preset_selection != "NONE":
                reset_failures.append(name)
            for key, want in p.items():
                if key in META:
                    continue
                if key not in og.bl_rna.properties:
                    # Production apply skips keys absent from the group RNA;
                    # mirror that so a future stray TOML key yields a clean
                    # check rather than an AttributeError that aborts the driver.
                    continue
                got = getattr(og, key)
                if not _close(got, want):
                    apply_failures.append(
                        {"preset": name, "key": key, "want": want, "got": got})
        dh.record(
            f"{otype}_apply_sets_fields",
            not apply_failures,
            {"failure_count": len(apply_failures),
             "failures": apply_failures[:8]},
        )
        dh.record(
            f"{otype}_picker_resets_to_none",
            not reset_failures,
            {"unreset": reset_failures},
        )

        # (3) D2/D5: Type unchanged; contact gaps and pins untouched.
        dh.record(
            f"{otype}_type_and_untouched_preserved",
            og.object_type == type0
            and _close(og.contact_gap, gap0)
            and _close(og.contact_offset, off0)
            and len(og.pin_vertex_groups) == pins0,
            {"type": og.object_type, "type0": type0,
             "gap": (og.contact_gap, gap0),
             "offset": (og.contact_offset, off0),
             "pins": (len(og.pin_vertex_groups), pins0)},
        )

        # (4) Loader rejects an unknown preset (returns False, no raise).
        dh.record(
            f"{otype}_apply_unknown_rejected",
            mp.apply_material_preset("DoesNotExist", og) is False,
            {},
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
