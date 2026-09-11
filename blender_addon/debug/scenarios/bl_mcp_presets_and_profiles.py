# File: scenarios/bl_mcp_presets_and_profiles.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Presets, profiles, and the copy/paste pairs over MCP.
#
# The three are different things and the scenario keeps them apart. A PRESET
# is a bundle of material values the add-on ships, chosen by name and bound to
# one object type. A PROFILE is a named snapshot the artist saved to a file,
# one kind per settings group. Copy and paste move settings between objects
# without naming or storing them.
#
# Assertions:
#   A. ``presets_are_listed_with_their_type_and_parameters`` -- every preset
#      names the object type it targets and the parameters it writes, and the
#      object_type filter narrows the list to that type.
#   B. ``preset_applies_and_reports_what_it_wrote`` -- applying a SHELL preset
#      writes its parameters and the group reads them back.
#   C. ``preset_refusals_name_the_way_out`` -- a preset aimed at another type
#      names the type mismatch and set_group_type as the fix, and an unknown
#      name lists the presets available for that group's type.
#   D. ``material_copy_paste_round_trips`` -- copy reports the parameters it
#      took, paste reports what it wrote onto the target, and the target then
#      holds the source's value.
#   E. ``pin_operation_copy_paste_answers`` -- copy and paste on a pin report
#      the operation counts they moved and replaced.
#   F. ``profile_save_list_load_cycle`` -- a SCENE profile saves to a named
#      file, appears in the listing, and loads back reporting the keys it
#      applied.
#   G. ``profile_refusals`` -- an unknown kind lists the four valid kinds, an
#      unknown profile name says what the file does hold, and reading a kind
#      with no bound file says nothing is bound rather than reporting empty.
#   H. ``clear_profile_path_unbinds`` -- clearing the binding names the file
#      it released, and the next read reports nothing bound.

from __future__ import annotations

from . import _mcp_lib as ml
from . import _runner as r

NEEDS_BLENDER = True

# macOS runners block loopback HTTP to Blender's in-process MCP server, so the
# rig does not select this scenario there.
PLATFORMS = ("linux", "win32")


_DRIVER_BODY = r"""
import os
import shutil
import tempfile

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

_rid = [900]
workdir = tempfile.mkdtemp(prefix="bl_mcp_profiles_")


def call(name, args=None):
    _rid[0] += 1
    payload, _raw = mcp_tool(pkg, url, name, args or {}, request_id=_rid[0])
    return payload


try:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for name, loc in (("Cloth", (0, 0, 0)), ("Other", (3, 0, 0))):
        bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4,
                                        size=2, location=loc)
        bpy.context.object.name = name

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    call("clear_solver")

    shell = call("create_group")["group_uuid"]
    call("set_group_type", {"group_uuid": shell, "type": "SHELL"})
    call("add_objects_to_group", {"group_uuid": shell, "object_names": ["Cloth"]})
    target = call("create_group")["group_uuid"]
    call("set_group_type", {"group_uuid": target, "type": "SHELL"})
    call("add_objects_to_group", {"group_uuid": target, "object_names": ["Other"]})
    static = call("create_group")["group_uuid"]
    call("set_group_type", {"group_uuid": static, "type": "STATIC"})
    result["phases"].append((time.time(), "groups_built"))

    # ----- A --------------------------------------------------------
    every = call("list_material_presets")
    presets = every.get("presets") or []
    shell_only = call("list_material_presets", {"object_type": "SHELL"})
    shell_names = [p.get("name") for p in (shell_only.get("presets") or [])]
    shaped = all(
        p.get("name") and p.get("object_type") and isinstance(p.get("parameters"), dict)
        and p["parameters"]
        for p in presets
    )
    mcp_check(result, "A_presets_are_listed_with_their_type_and_parameters",
              every.get("status") == "success" and len(presets) > 0 and shaped
              and shell_only.get("status") == "success" and len(shell_names) > 0
              and all(p.get("object_type") == "SHELL"
                      for p in (shell_only.get("presets") or []))
              and len(shell_names) <= len(presets),
              {"total": len(presets), "shell": shell_names, "shaped": shaped})

    # ----- B --------------------------------------------------------
    chosen = shell_names[0]
    applied = call("apply_material_preset",
                   {"group_uuid": shell, "preset_name": chosen})
    written = applied.get("written") or {}
    report = call("get_group_material_properties", {"group_uuid": shell})
    entries = report.get("properties") or {}
    matches = {
        key: entries.get(key, {}).get("value")
        for key in written
        if key in entries
    }
    agreed = all(
        (abs(matches[k] - written[k]) < 1e-4)
        if isinstance(written[k], (int, float)) and not isinstance(written[k], bool)
        else matches[k] == written[k]
        for k in matches
    )
    mcp_check(result, "B_preset_applies_and_reports_what_it_wrote",
              applied.get("status") == "success"
              and applied.get("preset_name") == chosen
              and applied.get("object_type") == "SHELL"
              and len(written) > 0 and len(matches) > 0 and agreed,
              {"preset": chosen, "written_count": len(written),
               "compared": len(matches), "agreed": agreed})

    # ----- C --------------------------------------------------------
    wrong_type = call("apply_material_preset",
                      {"group_uuid": static, "preset_name": chosen})
    unknown = call("apply_material_preset",
                   {"group_uuid": shell, "preset_name": "NoSuchPreset"})
    mcp_check(result, "C_preset_refusals_name_the_way_out",
              wrong_type.get("status") == "error"
              and "STATIC" in wrong_type.get("message", "")
              and "set_group_type" in wrong_type.get("message", "")
              and unknown.get("status") == "error"
              and "NoSuchPreset" in unknown.get("message", "")
              and chosen in unknown.get("message", ""),
              {"wrong_type": wrong_type.get("message"),
               "unknown": unknown.get("message")})

    # ----- D --------------------------------------------------------
    copied = call("copy_material_parameters", {"group_uuid": shell})
    pasted = call("paste_material_parameters", {"group_uuid": target})
    source_report = call("get_group_material_properties",
                         {"group_uuid": shell}).get("properties") or {}
    target_report = call("get_group_material_properties",
                         {"group_uuid": target}).get("properties") or {}
    probe_key = None
    for key in (pasted.get("pasted") or []):
        if key in source_report and key in target_report:
            probe_key = key
            break
    mcp_check(result, "D_material_copy_paste_round_trips",
              copied.get("status") == "success"
              and len(copied.get("parameters") or []) > 0
              and pasted.get("status") == "success"
              and len(pasted.get("pasted") or []) > 0
              and pasted.get("source_object_type") == "SHELL"
              and pasted.get("target_object_type") == "SHELL"
              and probe_key is not None
              and source_report[probe_key].get("value") == target_report[probe_key].get("value"),
              {"copied": len(copied.get("parameters") or []),
               "pasted": len(pasted.get("pasted") or []),
               "probe_key": probe_key,
               "source": source_report.get(probe_key, {}).get("value"),
               "target": target_report.get(probe_key, {}).get("value")})

    # ----- E --------------------------------------------------------
    call("create_vertex_group",
         {"object_name": "Cloth", "name": "Edge", "indices": [0, 1, 2]})
    call("add_pin_vertex_group",
         {"group_uuid": shell, "vertex_group_identifier": "Cloth::Edge"})
    copy_ops = call("copy_pin_operations",
                    {"group_uuid": shell, "vertex_group_identifier": "Cloth::Edge"})
    paste_ops = call("paste_pin_operations",
                     {"group_uuid": shell, "vertex_group_identifier": "Cloth::Edge"})
    mcp_check(result, "E_pin_operation_copy_paste_answers",
              copy_ops.get("status") == "success"
              and copy_ops.get("operation_count") == 0
              and paste_ops.get("status") == "success"
              and paste_ops.get("operation_count") == 0
              and paste_ops.get("replaced_operation_count") == 0,
              {"copy": copy_ops, "paste": paste_ops})

    # ----- F --------------------------------------------------------
    scene_file = os.path.join(workdir, "scene_profiles.toml")
    saved = call("save_profile",
                 {"kind": "SCENE", "name": "baseline", "path": scene_file})
    listed = call("list_profiles", {"kind": "SCENE", "path": scene_file})
    names = [p.get("name") for p in (listed.get("profiles") or [])]
    loaded = call("load_profile",
                  {"kind": "SCENE", "name": "baseline", "path": scene_file})
    mcp_check(result, "F_profile_save_list_load_cycle",
              saved.get("status") == "success" and saved.get("name") == "baseline"
              and len(saved.get("entry_keys") or []) > 0
              and os.path.exists(scene_file)
              and listed.get("status") == "success" and names == ["baseline"]
              and loaded.get("status") == "success"
              and len(loaded.get("applied_keys") or []) > 0,
              {"saved_keys": len(saved.get("entry_keys") or []),
               "names": names,
               "applied": len(loaded.get("applied_keys") or [])})

    # ----- G --------------------------------------------------------
    bad_kind = call("list_profiles", {"kind": "bogus"})
    missing_name = call("load_profile",
                        {"kind": "SCENE", "name": "absent", "path": scene_file})
    unbound = call("list_profiles", {"kind": "PIN", "group_uuid": shell})
    mcp_check(result, "G_profile_refusals",
              bad_kind.get("status") == "error"
              and all(k in bad_kind.get("message", "")
                      for k in ("SCENE", "MATERIAL", "PIN", "CONNECTION"))
              and missing_name.get("status") == "error"
              and "absent" in missing_name.get("message", "")
              and "baseline" in missing_name.get("message", "")
              and unbound.get("status") == "error"
              and "bound" in unbound.get("message", ""),
              {"bad_kind": bad_kind.get("message"),
               "missing_name": missing_name.get("message"),
               "unbound": unbound.get("message")})

    # ----- H --------------------------------------------------------
    cleared = call("clear_profile_path", {"kind": "SCENE"})
    after = call("list_profiles", {"kind": "SCENE"})
    mcp_check(result, "H_clear_profile_path_unbinds",
              cleared.get("status") == "success"
              and cleared.get("cleared_path") == scene_file
              and after.get("status") == "error"
              and "bound" in after.get("message", ""),
              {"cleared": cleared, "after": after.get("message")})

    call("clear_solver")
    mcp_mod.stop_mcp_server()

except Exception as exc:
    result["errors"].append("%s: %s" % (type(exc).__name__, exc))
    result["errors"].append(traceback.format_exc())
finally:
    shutil.rmtree(workdir, ignore_errors=True)
"""


_DRIVER_TEMPLATE = ml.MCP_LIB + "\nimport traceback\n" + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 180.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
