# File: scenarios/bl_mcp_mesh_cleaning.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The MCP mesh-cleaning surface.
#
# Mesh Cleaning is the Utility Tools panel that finds and repairs the
# geometry the solver rejects. Its operators read ``context.selected_objects``,
# which an MCP caller cannot see or reason about, so the tools take explicit
# object names and set the selection themselves. That shifts two properties
# out of the UI and onto this layer, and both are what this scenario covers:
# the caller's own selection has to come back exactly as it was, and a name
# that does not resolve has to stop the whole call before any mesh is edited,
# not part way through the list.
#
# The third property is the acknowledgement. Three of the repairs change the
# vertex count, which invalidates a captured deformation and the display
# cache. The panel raises a dialog listing what would be lost; the MCP tools
# raise an error naming the same dependents unless the caller passes
# ``acknowledge``. A tool that quietly welded a mesh would take those caches
# with it and the caller would learn about it at the next Transfer.
#
# The fourth is that the reported result has to describe the edit that was
# made. A repair is reported by measuring the quantities it can move, which
# for most of them is the element counts but for a rewind is the number of
# inconsistently wound edges, since face loops are reordered in place.
#
# The fifth is that a named object has to be reached, not merely named.
# Selecting an object the view layer counts as unselectable is a no-op, so
# without a read-back the operators run over a shorter list than the caller
# passed and the result reports on a mesh nothing touched. The two readings
# that make up that result also have to come from one mode: a mesh datablock
# does not carry pending Edit Mode work, so a reading taken before the tools
# leave Edit Mode differs from the one taken after by the artist's own edits.
#
# Subtests:
#   A. scan_finds_seeded_defects
#         Every detector reports exactly what the seeded mesh carries. Note
#         this mesh is NOT an ordinary open cloth panel: its duplicate face
#         makes two edges non-manifold (notes) and two edges inconsistently
#         wound (errors). bl_mesh_cleaning subtest K owns the property that a
#         genuinely clean open quad sheet reads as defect-free.
#   B. scan_payload_is_compact
#         The report keeps the eight-index previews and drops the full
#         per-vertex index lists, which on a dense mesh dominate the payload.
#   C. count_change_needs_acknowledgement
#         merge_by_distance without acknowledge fails and the mesh is
#         untouched, and the refusal names the caches the change would
#         invalidate on an object that carries one.
#   D. merge_welds_with_acknowledgement
#         The same call with acknowledge welds the pair, and the reported
#         before/after counts match the mesh.
#   E. loose_vertex_removed
#         remove_loose_vertices drops the faceless vertex.
#   F. selection_is_restored
#         Selection and active object survive a scan and a repair unchanged,
#         including an object that was never named in the call.
#   G. unknown_name_is_atomic
#         A call naming one real and one missing object edits neither. Probed
#         on the winding, which is the only thing the named tool would move.
#   H. non_mesh_is_rejected
#         Naming a non-mesh object fails before any edit.
#   I. rewind_is_reported
#         recalculate_normals_outside takes the inconsistently wound edges to
#         zero AND says so, rather than reporting nothing to repair.
#   J. dissolve_degenerate_is_gated_and_collapses
#         The second of the three acknowledgement-gated tools refuses, then
#         collapses a zero-length edge and the face built on it.
#   K. symmetric_triangulate_pokes_shared_data_once
#         One vertex per face is added, a linked duplicate is poked once
#         rather than once per user, and the caches the added vertices
#         invalidate are left in place for Transfer to re-take.
#   L. clear_stale_caches_defaults_on
#         A weld deletes the invalidated display cache and reports it, and
#         passing the flag off keeps the cache.
#   M. multi_object_call_reports_each
#         A scan and a repair over two names aggregate and report per object.
#   N. tools_reach_the_handler_registry
#         All eight names are registered with a schema requiring
#         object_names, and are re-exported from mcp.handlers.
#   O. unselectable_object_is_rejected
#         An object that is in the view layer but cannot be selected is
#         refused by name, and a call naming it beside a repairable mesh
#         edits neither.
#   P. edit_mode_measurements_agree
#         A repair called from Edit Mode on a mesh with nothing to repair
#         reports nothing to repair, and leaves the caller in Edit Mode.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True
# No solver, no build, no connection: bmesh and bpy.ops only, so the same
# assertions hold on either backend and running it on the GPU jobs is free.
BACKENDS = ("emulated", "real")


_DRIVER_BODY = r"""
import traceback

import bmesh

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    mc = __import__(pkg + ".mcp.handlers.mesh_cleaning",
                    fromlist=["scan_meshes"])
    clean = __import__(pkg + ".mesh_ops.cleaning_ops",
                       fromlist=["find_surface_defects"])

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    def new_mesh(name, coords, faces):
        me = bpy.data.meshes.new(name + "Mesh")
        obj = bpy.data.objects.new(name, me)
        bpy.context.collection.objects.link(obj)
        me.from_pydata([tuple(c) for c in coords], [], [tuple(f) for f in faces])
        me.update()
        return obj

    def build_defective(name):
        # A 2x2 quad grid, then three seeded defects:
        #   - a vertex placed 1e-6 from an existing one (near-coincident)
        #   - a faceless vertex far from the sheet (loose)
        #   - a second face on the first face's vertex set (duplicate)
        # Built through from_pydata rather than bmesh: bmesh.faces.new
        # rejects a face whose vertex set already carries one, in either
        # winding, so the duplicate cannot be seeded there at all. This is
        # the same construction bl_duplicate_face_rejection uses, and it
        # reproduces the topology Merge by Distance leaves behind.
        coords = [(float(x), float(y), 0.0)
                  for y in (0, 1, 2) for x in (0, 1, 2)]
        coords.append((0.0, 0.0, 1e-6))   # near-coincident with vertex 0
        coords.append((5.0, 5.0, 5.0))    # loose, belongs to no face
        faces = [
            (0, 1, 4, 3),
            (1, 2, 5, 4),
            (3, 4, 7, 6),
            (4, 5, 8, 7),
            (0, 1, 4, 3),                 # duplicate of the first face
        ]
        return new_mesh(name, coords, faces)

    target = build_defective("DefectiveSheet")
    bystander = build_defective("Bystander")
    bpy.ops.object.empty_add(location=(0.0, 0.0, 3.0))
    empty = bpy.context.active_object
    empty.name = "NotAMesh"

    def counts(obj):
        return (len(obj.data.vertices), len(obj.data.polygons))

    def bad_winding(obj):
        return clean.find_surface_defects(obj.data)["bad_winding"]

    def has_cache_modifier(obj):
        return any(m.type == "MESH_CACHE" for m in obj.modifiers)

    # ----- A: the scan finds what was seeded -----------------------
    scan = mc.scan_meshes({"object_names": ["DefectiveSheet"]})
    reports = scan.get("reports", [])
    rep = reports[0] if reports else {}
    defects = rep.get("defects", {})
    near = defects.get("near_duplicates", {}).get("count", -1)
    loose = defects.get("isolated_verts", {}).get("count", -1)
    hanging = defects.get("hanging_verts", {}).get("count", -1)
    degenerate = defects.get("degenerate_faces", {}).get("count", -1)
    dup = defects.get("duplicate_faces", {}).get("count", -1)
    surface = defects.get("surface", {})
    resplit = defects.get("resplittable", {}).get("count", -1)
    # The expected counts are what the detectors actually measure, which is
    # not one per seeded defect:
    #   near == 1: the one pair within the merge threshold.
    #   loose == 2: BOTH faceless vertices. The near-coincident vertex is not
    #     attached to any face either, so it is isolated as well as near.
    #   hanging == 0: from_pydata derives edges from faces only, so the mesh
    #     carries no loose edge for a vertex to hang on.
    #   degenerate == 0: every quad has area 1.
    #   dup == 2: count_duplicate_faces counts TRIANGLES after the encoder's
    #     tessellation, and the duplicated quad is two triangles.
    #   boundary == 6: the grid's outer edges, minus the four the duplicate
    #     face turns into interior ones.
    #   non_manifold == 2: edges (1,4) and (3,4) carry three faces each.
    #   bad_winding == 2: edges (0,1) and (0,3) carry two faces that traverse
    #     them in the SAME direction, because the duplicate copies the
    #     original's winding.
    #   resplit == 5: every face is a quad.
    # n_errors sums near + loose + hanging + dup + degenerate + linked
    # duplicates + bad_winding = 1 + 2 + 0 + 2 + 0 + 0 + 2 = 7.
    dh.record(
        "A_scan_finds_seeded_defects",
        scan.get("status") == "success"
        and near == 1 and loose == 2 and hanging == 0 and degenerate == 0
        and dup == 2
        and surface.get("boundary") == 6
        and surface.get("non_manifold") == 2
        and surface.get("bad_winding") == 2
        and resplit == 5
        and rep.get("n_errors", 0) == 7
        and rep.get("n_notes", 0) == 13,
        {"status": scan.get("status"), "near": near, "loose": loose,
         "hanging": hanging, "degenerate": degenerate,
         "duplicate_faces": dup, "surface": surface, "resplittable": resplit,
         "n_errors": rep.get("n_errors"), "n_notes": rep.get("n_notes"),
         "needs_attention": scan.get("objects_needing_attention")},
    )

    # ----- B: the payload keeps previews, drops index lists --------
    has_verts_list = any(
        "verts" in d for d in defects.values() if isinstance(d, dict)
    )
    # Assert the previews by value, on the two defects that carry one only
    # when the count is non-zero, so an empty payload cannot pass.
    near_preview = defects.get("near_duplicates", {}).get("preview")
    loose_preview = defects.get("isolated_verts", {}).get("preview")
    dh.record(
        "B_scan_payload_is_compact",
        (not has_verts_list)
        and list(near_preview or []) == [(0, 9)]
        and list(loose_preview or []) == [9, 10]
        and isinstance(rep.get("dependents"), list),
        {"carries_full_vert_lists": has_verts_list,
         "near_preview": near_preview, "loose_preview": loose_preview,
         "dependents": rep.get("dependents")},
    )

    # ----- C: a count change refuses without acknowledgement -------
    # Two refusals, because _require_acknowledgement has two branches. The
    # target has no cache, so it takes the "nothing is affected" wording;
    # CachedSheet carries a display cache and must be named in the message.
    before = counts(target)
    refused = mc.merge_by_distance(
        {"object_names": ["DefectiveSheet"], "merge_threshold": 1e-3}
    )
    cached = build_defective("CachedSheet")
    cached.modifiers.new(name="ContactSolverCache", type="MESH_CACHE")
    cached_before = counts(cached)
    refused_named = mc.merge_by_distance(
        {"object_names": ["CachedSheet"], "merge_threshold": 1e-3}
    )
    dh.record(
        "C_count_change_needs_acknowledgement",
        refused.get("status") == "error"
        and "acknowledge" in refused.get("message", "")
        and "No capture cache" in refused.get("message", "")
        and counts(target) == before
        and refused_named.get("status") == "error"
        and "acknowledge" in refused_named.get("message", "")
        and "Display cache (PC2)" in refused_named.get("message", "")
        and counts(cached) == cached_before,
        {"status": refused.get("status"), "message": refused.get("message"),
         "counts_before": before, "counts_after": counts(target),
         "cached_status": refused_named.get("status"),
         "cached_message": refused_named.get("message"),
         "cached_counts": [cached_before, counts(cached)]},
    )

    # ----- D: with acknowledgement it welds ------------------------
    merged = mc.merge_by_distance(
        {"object_names": ["DefectiveSheet"], "merge_threshold": 1e-3,
         "acknowledge": True}
    )
    after = counts(target)
    changed = merged.get("changed", [])
    entry = changed[0] if changed else {}
    dh.record(
        "D_merge_welds_with_acknowledgement",
        merged.get("status") == "success"
        and after[0] == before[0] - 1
        and entry.get("vertices_before") == before[0]
        and entry.get("vertices_after") == after[0],
        {"status": merged.get("status"), "counts_before": before,
         "counts_after": after, "reported": entry},
    )

    # ----- E: the loose vertex goes --------------------------------
    before_loose = counts(target)
    loose_res = mc.remove_loose_vertices(
        {"object_names": ["DefectiveSheet"], "acknowledge": True}
    )
    after_loose = counts(target)
    rescan = mc.scan_meshes({"object_names": ["DefectiveSheet"]})
    rescan_loose = (
        rescan.get("reports", [{}])[0]
        .get("defects", {}).get("isolated_verts", {}).get("count", -1)
    )
    dh.record(
        "E_loose_vertex_removed",
        loose_res.get("status") == "success"
        and after_loose[0] == before_loose[0] - 1
        and rescan_loose == 0,
        {"status": loose_res.get("status"), "before": before_loose,
         "after": after_loose, "isolated_after_rescan": rescan_loose},
    )

    # ----- F: the caller's selection is untouched ------------------
    # Select the BYSTANDER only, then act on the target: an MCP tool that
    # leaks its selection would leave the artist's viewport rearranged.
    bpy.ops.object.select_all(action="DESELECT")
    bystander.select_set(True)
    bpy.context.view_layer.objects.active = bystander
    sel_before = sorted(o.name for o in bpy.context.selected_objects)
    active_before = bpy.context.view_layer.objects.active.name

    mc.scan_meshes({"object_names": ["DefectiveSheet"]})
    mc.delete_duplicate_faces({"object_names": ["DefectiveSheet"]})

    sel_after = sorted(o.name for o in bpy.context.selected_objects)
    active_obj = bpy.context.view_layer.objects.active
    active_after = active_obj.name if active_obj else None
    dh.record(
        "F_selection_is_restored",
        sel_before == sel_after and active_before == active_after,
        {"selection_before": sel_before, "selection_after": sel_after,
         "active_before": active_before, "active_after": active_after},
    )

    # ----- G: an unknown name stops the whole call -----------------
    # Probed on the winding rather than on the element counts: a rewind
    # reorders face loops and moves neither count, so a counts-only probe
    # could not see the edit this call would make if it were not atomic.
    target_before = counts(target)
    bystander_before = counts(bystander)
    winding_before = bad_winding(bystander)
    atomic = mc.recalculate_normals_outside(
        {"object_names": ["Bystander", "NoSuchObject"]}
    )
    dh.record(
        "G_unknown_name_is_atomic",
        atomic.get("status") == "error"
        and "NoSuchObject" in atomic.get("message", "")
        and counts(target) == target_before
        and counts(bystander) == bystander_before
        and winding_before == 2
        and bad_winding(bystander) == winding_before,
        {"status": atomic.get("status"), "message": atomic.get("message"),
         "target": [target_before, counts(target)],
         "bystander": [bystander_before, counts(bystander)],
         "bad_winding": [winding_before, bad_winding(bystander)]},
    )

    # ----- H: a non-mesh object is rejected ------------------------
    non_mesh = mc.triangulate_for_solver(
        {"object_names": ["Bystander", "NotAMesh"]}
    )
    dh.record(
        "H_non_mesh_is_rejected",
        non_mesh.get("status") == "error"
        and "NotAMesh" in non_mesh.get("message", "")
        and counts(bystander) == bystander_before,
        {"status": non_mesh.get("status"), "message": non_mesh.get("message"),
         "bystander": [bystander_before, counts(bystander)]},
    )

    # ----- I: a rewind is measured and reported --------------------
    # Bystander still carries the 2 inconsistently wound edges its duplicate
    # face seeds (G and H both failed before touching it). Recalculating
    # normals makes the two coincident faces wind oppositely, taking the
    # count to 0 while leaving both element counts alone. The result must
    # carry that, not "Nothing to repair".
    rewind_before = bad_winding(bystander)
    rewind_counts_before = counts(bystander)
    rewind = mc.recalculate_normals_outside({"object_names": ["Bystander"]})
    rewind_after = bad_winding(bystander)
    rewind_changed = rewind.get("changed", [])
    rewind_entry = rewind_changed[0] if rewind_changed else {}
    dh.record(
        "I_rewind_is_reported",
        rewind.get("status") == "success"
        and rewind_before == 2 and rewind_after == 0
        and rewind.get("changed_count") == 1
        and rewind_entry.get("object_name") == "Bystander"
        and rewind_entry.get("bad_winding_before") == 2
        and rewind_entry.get("bad_winding_after") == 0
        and "Nothing to repair" not in rewind.get("message", "")
        and counts(bystander) == rewind_counts_before,
        {"status": rewind.get("status"), "message": rewind.get("message"),
         "bad_winding": [rewind_before, rewind_after],
         "changed_count": rewind.get("changed_count"),
         "reported": rewind_entry,
         "counts": [rewind_counts_before, counts(bystander)],
         "operator_status": rewind.get("operator_status")},
    )

    # ----- J: the second gated repair refuses, then collapses ------
    # Vertex 6 sits exactly on vertex 4, so edge (4,6) has zero length and
    # the triangle built on it has zero area. Both are what Degenerate
    # Dissolve removes.
    sliver = new_mesh(
        "DegenerateSliver",
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (3, 0, 0), (4, 0, 0), (3, 0, 0)],
        [(0, 1, 2, 3), (4, 5, 6)],
    )
    sliver_before = counts(sliver)
    sliver_refused = mc.dissolve_degenerate_faces(
        {"object_names": ["DegenerateSliver"]}
    )
    sliver_gated = counts(sliver)
    sliver_res = mc.dissolve_degenerate_faces(
        {"object_names": ["DegenerateSliver"], "acknowledge": True}
    )
    sliver_after = counts(sliver)
    sliver_entry = (sliver_res.get("changed") or [{}])[0]
    dh.record(
        "J_dissolve_degenerate_is_gated_and_collapses",
        sliver_refused.get("status") == "error"
        and "acknowledge" in sliver_refused.get("message", "")
        and sliver_gated == sliver_before
        and sliver_res.get("status") == "success"
        and sliver_res.get("changed_count") == 1
        and sliver_after[0] < sliver_before[0]
        and sliver_after[1] == sliver_before[1] - 1
        and sliver_entry.get("polygons_after") == sliver_after[1],
        {"refused": sliver_refused.get("message"),
         "before": sliver_before, "gated": sliver_gated,
         "after": sliver_after, "status": sliver_res.get("status"),
         "reported": sliver_entry},
    )

    # ----- K: poking adds a vertex per face, once per datablock ----
    poke = new_mesh(
        "PokeTarget",
        [(float(x), float(y), 0.0) for y in (0, 1, 2) for x in (0, 1, 2)],
        [(0, 1, 4, 3), (1, 2, 5, 4), (3, 4, 7, 6), (4, 5, 8, 7)],
    )
    poke.modifiers.new(name="ContactSolverCache", type="MESH_CACHE")
    poke_twin = bpy.data.objects.new("PokeTwin", poke.data)
    bpy.context.collection.objects.link(poke_twin)
    poke_before = counts(poke)
    poke_res = mc.symmetric_triangulate(
        {"object_names": ["PokeTarget", "PokeTwin"]}
    )
    poke_after = counts(poke)
    poke_entries = {e.get("object_name"): e
                    for e in poke_res.get("changed", [])}
    dh.record(
        "K_symmetric_triangulate_pokes_shared_data_once",
        poke_res.get("status") == "success"
        # 9 grid vertices + one center per quad; each quad fans into 4
        # triangles. A datablock poked once per USER would read 29 and 64.
        and poke_before == (9, 4) and poke_after == (13, 16)
        and poke_res.get("changed_count") == 2
        and poke_entries.get("PokeTarget", {}).get("vertices_after") == 13
        and poke_entries.get("PokeTwin", {}).get("vertices_after") == 13
        # It is a Utility Tools operation, not a repair: it takes no
        # acknowledgement and deletes none of the caches it invalidates.
        and has_cache_modifier(poke),
        {"status": poke_res.get("status"), "before": poke_before,
         "after": poke_after, "changed_count": poke_res.get("changed_count"),
         "reported": poke_res.get("changed"),
         "cache_modifier_kept": has_cache_modifier(poke)},
    )

    # ----- L: clear_stale_caches defaults on, as the panel does ----
    def cached_weld_target(name):
        obj = new_mesh(
            name,
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.0, 0.0, 1e-6)],
            [(0, 1, 2, 3)],
        )
        obj.modifiers.new(name="ContactSolverCache", type="MESH_CACHE")
        return obj

    cache_default = cached_weld_target("CacheDefault")
    cache_kept = cached_weld_target("CacheKept")
    default_res = mc.merge_by_distance(
        {"object_names": ["CacheDefault"], "merge_threshold": 1e-3,
         "acknowledge": True}
    )
    kept_res = mc.merge_by_distance(
        {"object_names": ["CacheKept"], "merge_threshold": 1e-3,
         "acknowledge": True, "clear_stale_caches": False}
    )
    dh.record(
        "L_clear_stale_caches_defaults_on",
        default_res.get("status") == "success"
        and default_res.get("changed_count") == 1
        and not has_cache_modifier(cache_default)
        and len(default_res.get("cleared_caches", [])) == 1
        and kept_res.get("status") == "success"
        and kept_res.get("changed_count") == 1
        and has_cache_modifier(cache_kept)
        and kept_res.get("cleared_caches") == [],
        {"default_cleared": default_res.get("cleared_caches"),
         "default_modifier_kept": has_cache_modifier(cache_default),
         "kept_cleared": kept_res.get("cleared_caches"),
         "kept_modifier_kept": has_cache_modifier(cache_kept),
         "counts": [counts(cache_default), counts(cache_kept)]},
    )

    # ----- M: two names in one call, aggregated and reported -------
    multi_a = build_defective("MultiA")
    multi_b = build_defective("MultiB")
    multi_scan = mc.scan_meshes({"object_names": ["MultiA", "MultiB"]})
    multi_reports = multi_scan.get("reports", [])
    multi_before = [counts(multi_a), counts(multi_b)]
    multi_merge = mc.merge_by_distance(
        {"object_names": ["MultiA", "MultiB"], "merge_threshold": 1e-3,
         "acknowledge": True}
    )
    multi_entries = {e.get("object_name"): e
                     for e in multi_merge.get("changed", [])}
    dh.record(
        "M_multi_object_call_reports_each",
        multi_scan.get("status") == "success"
        and len(multi_reports) == 2
        and multi_scan.get("total_errors") == 14
        and sorted(multi_scan.get("objects_needing_attention", [])) == [
            "MultiA", "MultiB"]
        and multi_merge.get("status") == "success"
        and multi_merge.get("changed_count") == 2
        and multi_entries.get("MultiA", {}).get("vertices_before") == 11
        and multi_entries.get("MultiA", {}).get("vertices_after") == 10
        and multi_entries.get("MultiB", {}).get("vertices_before") == 11
        and multi_entries.get("MultiB", {}).get("vertices_after") == 10
        and counts(multi_a)[0] == 10 and counts(multi_b)[0] == 10,
        {"n_reports": len(multi_reports),
         "total_errors": multi_scan.get("total_errors"),
         "needs_attention": multi_scan.get("objects_needing_attention"),
         "before": multi_before,
         "after": [counts(multi_a), counts(multi_b)],
         "reported": multi_merge.get("changed")},
    )

    # ----- N: the tools are discoverable over MCP ------------------
    # Every subtest above calls the handlers as plain functions, which
    # bypasses discovery: without this check the whole module could be
    # missing from the registry and the scenario would still pass.
    dec = __import__(pkg + ".mcp.decorators",
                     fromlist=["get_handler_registry"])
    handlers_pkg = __import__(pkg + ".mcp.handlers", fromlist=["scan_meshes"])
    registry = dec.get_handler_registry()
    tool_names = [
        "scan_meshes", "merge_by_distance", "remove_loose_vertices",
        "dissolve_degenerate_faces", "delete_duplicate_faces",
        "triangulate_for_solver", "recalculate_normals_outside",
        "symmetric_triangulate",
    ]
    unregistered = [n for n in tool_names if n not in registry]
    missing_schema = []
    for name in tool_names:
        if name in unregistered:
            continue
        schema = registry[name].get("schema", {}).get("inputSchema", {})
        if "object_names" not in schema.get("required", []):
            missing_schema.append(name)
    unexported = [n for n in tool_names if not hasattr(handlers_pkg, n)]
    dh.record(
        "N_tools_reach_the_handler_registry",
        not unregistered and not missing_schema and not unexported,
        {"unregistered": unregistered, "missing_object_names_required":
         missing_schema, "not_reexported": unexported,
         "registry_size": len(registry)},
    )

    # ----- O: an object that cannot be selected is refused ---------
    # select_set(True) is a no-op on an object the view layer counts as
    # unselectable, and the operators read the selection, so without a
    # read-back the object is skipped and the caller is told the tool ran.
    # Disable Selection is the mechanism seeded here because it leaves the
    # object visible and in view_layer.objects, which is what separates this
    # case from the "not in the active view layer" refusal the same helper
    # already raises. Hiding by eye icon or by collection reaches the same
    # read-back.
    locked = build_defective("NoSelectSheet")
    locked.hide_select = True
    mixed = build_defective("MixedTarget")
    bpy.context.view_layer.update()

    # The whole subtest rests on this object really being unselectable, so
    # that is measured rather than assumed.
    bpy.ops.object.select_all(action="DESELECT")
    locked.select_set(True)
    locked_is_selectable = locked.select_get()
    bpy.ops.object.select_all(action="DESELECT")

    locked_before = counts(locked)
    locked_res = mc.merge_by_distance(
        {"object_names": ["NoSelectSheet"], "merge_threshold": 1e-3,
         "acknowledge": True}
    )
    locked_scan = mc.scan_meshes({"object_names": ["NoSelectSheet"]})
    mixed_winding_before = bad_winding(mixed)
    mixed_res = mc.recalculate_normals_outside(
        {"object_names": ["MixedTarget", "NoSelectSheet"]}
    )
    dh.record(
        "O_unselectable_object_is_rejected",
        (not locked_is_selectable)
        and locked_res.get("status") == "error"
        and "NoSelectSheet" in locked_res.get("message", "")
        and "not selectable" in locked_res.get("message", "")
        and counts(locked) == locked_before
        and locked_scan.get("status") == "error"
        and "not selectable" in locked_scan.get("message", "")
        and mixed_res.get("status") == "error"
        and "NoSelectSheet" in mixed_res.get("message", "")
        and "MixedTarget" not in mixed_res.get("message", "")
        and mixed_winding_before == 2
        and bad_winding(mixed) == mixed_winding_before,
        {"selectable_after_flag": locked_is_selectable,
         "merge_status": locked_res.get("status"),
         "merge_message": locked_res.get("message"),
         "locked_counts": [locked_before, counts(locked)],
         "scan_status": locked_scan.get("status"),
         "scan_message": locked_scan.get("message"),
         "mixed_status": mixed_res.get("status"),
         "mixed_message": mixed_res.get("message"),
         "mixed_bad_winding": [mixed_winding_before, bad_winding(mixed)]},
    )

    # ----- P: both readings come from one mode ---------------------
    # A mesh datablock does not carry pending Edit Mode work: the BMesh is
    # written back when the mode is left, which is what these tools do on
    # entry. EditStale carries no duplicate face, so the only correct answer
    # is that there is nothing to repair. A reading taken before the
    # write-back differs from the one taken after by exactly the three
    # vertices added here, and would be reported as this repair's doing.
    stale = new_mesh(
        "EditStale",
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        [(0, 1, 2, 3)],
    )
    bpy.ops.object.select_all(action="DESELECT")
    stale.select_set(True)
    bpy.context.view_layer.objects.active = stale
    bpy.ops.object.mode_set(mode="EDIT")
    stale_bm = bmesh.from_edit_mesh(stale.data)
    for i in range(3):
        stale_bm.verts.new((3.0 + float(i), 0.0, 0.0))
    bmesh.update_edit_mesh(stale.data)
    # Recorded, not asserted. Predicted 4: the datablock still holds the mesh
    # as it stood when Edit Mode was entered. A build that wrote the BMesh
    # back eagerly would read 7 and there would be nothing here to get wrong,
    # while every assertion below still states the property the tool holds.
    stale_datablock_verts = len(stale.data.vertices)

    stale_res = mc.delete_duplicate_faces({"object_names": ["EditStale"]})
    stale_mode_after = bpy.context.mode
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    stale_counts = counts(stale)
    dh.record(
        "P_edit_mode_measurements_agree",
        stale_res.get("status") == "success"
        and stale_res.get("changed_count") == 0
        and stale_res.get("changed") == []
        and "Nothing to repair" in stale_res.get("message", "")
        and stale_counts == (7, 1)
        and stale_mode_after == "EDIT_MESH",
        {"datablock_verts_in_edit_mode": stale_datablock_verts,
         "status": stale_res.get("status"),
         "message": stale_res.get("message"),
         "changed": stale_res.get("changed"),
         "operator_status": stale_res.get("operator_status"),
         "counts_after": stale_counts,
         "mode_after_call": stale_mode_after},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 120.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
