# File: scenarios/bl_mesh_cleaning.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Mesh Cleaning coverage (Utility Tools panel, mesh_ops/cleaning_ops.py).
#
# The tool scans a mesh for the geometry defects that make a Transfer fail or
# make the solver abort, then repairs the repairable ones. The check that
# matters most has no other guard anywhere in the pipeline: a pair of
# near-coincident vertices survives into the solver, where the contact
# barrier's mass/gap^2 dynamic stiffness contributes Hessian entries orders of
# magnitude above the rest of the row, the fp32 Newton matrix loses rank, and
# the run stops on the SPD guard (p^T A p <= 0) naming no geometry at all.
# fTetWild masks such a pair by welding it away, so the failure only appears
# once the artist switches to TetGen's surface-preserving mode.
#
# Subtests:
#   A. detects_near_duplicate_pair
#         Two vertices 2e-8 apart are reported, with the smallest gap and the
#         world-space scale of it; a clean mesh reports nothing.
#   B. detects_isolated_and_degenerate
#         A face-less stray vertex and a zero-area face are both reported.
#   C. particle_mesh_is_exempt
#         A SAND particle mesh (all vertices face-less by construction) must
#         NOT be reported as thousands of isolated vertices, and its repair
#         must be a no-op. This is the false positive that would make the
#         tool untrustworthy on every granular scene.
#   D. merge_repairs_and_report_invalidates
#         Merge by Distance welds the pair (vertex count drops), and the
#         report cache is dropped so a stale row cannot outlive the fix.
#   E. count_change_requires_acknowledgement
#         Calling a count-changing repair without the acknowledgement must
#         CANCEL and leave the mesh untouched.
#   F. resplittable_faces_and_triangulate
#         A quad is reported as re-splittable, Triangulate converts it, and
#         the vertex count is preserved (so pins and caches survive).
#   G. surface_and_winding
#         An open quad reports boundary edges; a flipped neighbor reports an
#         inconsistently wound edge that Recalculate Outside repairs.
#   H. scan_never_mutates
#         Scanning leaves vertex / polygon counts and every coordinate
#         byte-identical.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def new_mesh(name, coords, faces, edges=()):
    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata([tuple(c) for c in coords], [tuple(e) for e in edges],
                     [tuple(f) for f in faces])
    mesh.update()
    return obj


def only_select(obj):
    for o in bpy.context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


try:
    dh = DriverHelpers(pkg, result)
    clean = __import__(pkg + ".mesh_ops.cleaning_ops",
                       fromlist=["scan_object", "get_scan_report",
                                 "clear_scan_report", "is_particle_mesh",
                                 "find_surface_defects",
                                 "find_resplittable_faces",
                                 "vertex_count_dependents",
                                 "DEFAULT_MERGE_THRESHOLD"])
    uuid_mod = __import__(pkg + ".core.uuid_registry",
                          fromlist=["get_or_create_object_uuid"])

    dh.log("setup_start")
    dh.reset_scene_to_pinned_plane(name="CleanBaseMesh")
    dh.save_blend(PROBE_DIR, "mesh_cleaning.blend")
    dh.configure_state(project_name="mesh_cleaning", frame_count=6)

    TH = clean.DEFAULT_MERGE_THRESHOLD
    # A gap that float32 vertex storage can actually hold at coordinate 1.0
    # (eps there is ~1.2e-7, so this is ~84 ULP) and that sits under the
    # default merge distance of 1e-4.
    GAP = 1e-5
    # The production gap was 1.9e-6 local at coordinate ~12 on an object
    # scaled 0.01, i.e. ~19 nm in world units. At coordinate 1.0 a gap that
    # small is BELOW one float32 ULP, so Blender stores the pair as exactly
    # coincident. Subtest A2 pins that down: the tool must still catch it.
    SUB_ULP_GAP = 2e-8

    # ----- A: near-duplicate detection ---------------------------------
    # Two quads side by side, where the shared corners are duplicated GAP away.
    dup = new_mesh(
        "NearDup",
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (1 + GAP, 0, 0), (1 + GAP, 1, 0), (2, 0, 0), (2, 1, 0)],
        [(0, 1, 2, 3), (4, 6, 7, 5)],
    )
    dup.scale = (0.01, 0.01, 0.01)  # exercise the world-space reporting
    bpy.context.view_layer.update()
    rep_dup = clean.scan_object(dup, merge_threshold=TH, area_eps=0.0)
    near = rep_dup["defects"]["near_duplicates"]

    clean_obj = new_mesh("CleanQuad", [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                         [(0, 1, 2, 3)])
    rep_clean = clean.scan_object(clean_obj, merge_threshold=TH, area_eps=0.0)

    dh.record(
        "A_detects_near_duplicate_pair",
        near["count"] == 2
        and abs(near["min_dist"] - GAP) < 1e-7
        and "min_dist_world" in near
        and abs(near["min_dist_world"] - GAP * 0.01) < 1e-9
        and rep_clean["defects"]["near_duplicates"]["count"] == 0,
        {"count": near["count"], "min_dist": near.get("min_dist"),
         "min_dist_world": near.get("min_dist_world"),
         "clean_count": rep_clean["defects"]["near_duplicates"]["count"]},
    )

    # ----- A2: a sub-ULP gap collapses to exactly coincident ------------
    # Blender vertex coordinates are float32, so a pair closer than one ULP
    # is stored at the SAME position. That is still the defect (it is the
    # worst case of it), so the scan must report it rather than divide by a
    # zero gap or miss it.
    sub_ulp = new_mesh(
        "SubUlpDup",
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (1 + SUB_ULP_GAP, 0, 0), (1 + SUB_ULP_GAP, 1, 0), (2, 0, 0), (2, 1, 0)],
        [(0, 1, 2, 3), (4, 6, 7, 5)],
    )
    rep_sub = clean.scan_object(sub_ulp, merge_threshold=TH, area_eps=0.0)
    near_sub = rep_sub["defects"]["near_duplicates"]
    stored_gap = abs(sub_ulp.data.vertices[4].co[0] - sub_ulp.data.vertices[1].co[0])
    dh.record(
        "A2_sub_ulp_gap_is_coincident_and_caught",
        near_sub["count"] == 2
        and stored_gap == 0.0
        and near_sub["min_dist"] == 0.0,
        {"count": near_sub["count"], "stored_gap": stored_gap,
         "min_dist": near_sub.get("min_dist")},
    )

    # ----- B: isolated vertex + degenerate face ------------------------
    # A quad, a stray face-less vertex (4), and a zero-area triangle.
    bad = new_mesh(
        "StrayAndDegenerate",
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (5, 5, 5),
         (2, 0, 0), (3, 0, 0), (4, 0, 0)],
        [(0, 1, 2, 3), (5, 6, 7)],
    )
    rep_bad = clean.scan_object(bad, merge_threshold=TH, area_eps=0.0)
    iso = rep_bad["defects"]["isolated_verts"]
    deg = rep_bad["defects"]["degenerate_faces"]
    dh.record(
        "B_detects_isolated_and_degenerate",
        iso["count"] == 1 and iso["verts"] == [4] and deg["count"] == 1,
        {"iso": iso["count"], "iso_verts": iso["verts"], "deg": deg["count"]},
    )

    # ----- C: SAND particle mesh is exempt -----------------------------
    grains = new_mesh("Grains", [(i * 0.5, 0, 0) for i in range(40)], [])
    grains["ppf_particle_mesh"] = 1
    rep_grains = clean.scan_object(grains, merge_threshold=TH, area_eps=0.0)
    g_iso = rep_grains["defects"]["isolated_verts"]["count"]
    g_surf = rep_grains["defects"]["surface"]["count"]
    only_select(grains)
    grains_before = len(grains.data.vertices)
    rm_res = bpy.ops.object.ppf_remove_loose_vertices(acknowledge=True)
    dh.record(
        "C_particle_mesh_is_exempt",
        clean.is_particle_mesh(grains)
        and g_iso == 0 and g_surf == 0
        and len(grains.data.vertices) == grains_before,
        {"is_particle": clean.is_particle_mesh(grains), "iso": g_iso,
         "surface": g_surf, "op": str(rm_res),
         "verts": len(grains.data.vertices), "before": grains_before},
    )

    # ----- D: merge repairs, and the cached report is dropped ----------
    only_select(dup)
    bpy.ops.object.ppf_scan_mesh_defects(merge_threshold=TH, area_eps=0.0)
    cached_before = clean.get_scan_report(dup.name) is not None
    verts_before = len(dup.data.vertices)
    merge_res = bpy.ops.object.ppf_merge_by_distance(
        merge_threshold=TH, acknowledge=True
    )
    verts_after = len(dup.data.vertices)
    cached_after = clean.get_scan_report(dup.name)
    rep_after = clean.scan_object(dup, merge_threshold=TH, area_eps=0.0)
    dh.record(
        "D_merge_repairs_and_invalidates_report",
        cached_before
        and "FINISHED" in merge_res
        and verts_after == verts_before - 2
        and cached_after is None
        and rep_after["defects"]["near_duplicates"]["count"] == 0,
        {"cached_before": cached_before, "op": str(merge_res),
         "before": verts_before, "after": verts_after,
         "cached_after": cached_after is None,
         "near_after": rep_after["defects"]["near_duplicates"]["count"]},
    )

    # ----- E: count change without acknowledgement is refused ----------
    dup2 = new_mesh(
        "NearDup2",
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (1 + GAP, 0, 0), (1 + GAP, 1, 0), (2, 0, 0), (2, 1, 0)],
        [(0, 1, 2, 3), (4, 6, 7, 5)],
    )
    only_select(dup2)
    before_guard = len(dup2.data.vertices)
    # An {"ERROR"} report makes bpy.ops raise instead of returning the
    # CANCELLED set, which is the loud refusal we want; either way the mesh
    # must be untouched.
    guard_res = "raised"
    guard_msg = ""
    try:
        guard_res = str(bpy.ops.object.ppf_merge_by_distance(
            merge_threshold=TH, acknowledge=False
        ))
    except RuntimeError as e:
        guard_msg = str(e)
    refused = guard_msg != "" or "CANCELLED" in guard_res
    dh.record(
        "E_count_change_requires_acknowledgement",
        refused
        and "vertex count" in (guard_msg or guard_res).lower()
        and len(dup2.data.vertices) == before_guard,
        {"op": guard_res, "err": guard_msg[:160], "before": before_guard,
         "after": len(dup2.data.vertices)},
    )

    # ----- F: re-splittable faces + triangulate preserves vertex count --
    quad = new_mesh("FoldedQuad",
                    [(0, 0, 0), (1, 0, 0.4), (1, 1, 0), (0, 1, 0.4)],
                    [(0, 1, 2, 3)])
    resp = clean.find_resplittable_faces(quad.data)
    only_select(quad)
    q_verts_before = len(quad.data.vertices)
    tri_res = bpy.ops.object.ppf_triangulate_for_solver()
    resp_after = clean.find_resplittable_faces(quad.data)
    dh.record(
        "F_resplittable_faces_and_triangulate",
        resp["count"] == 1 and resp.get("max_fold_deg", 0.0) > 0.0
        and "FINISHED" in tri_res
        and resp_after["count"] == 0
        and len(quad.data.polygons) == 2
        and len(quad.data.vertices) == q_verts_before,
        {"resp": resp, "op": str(tri_res), "after": resp_after,
         "polys": len(quad.data.polygons),
         "verts": len(quad.data.vertices), "verts_before": q_verts_before},
    )

    # ----- G: open boundary + inconsistent winding ----------------------
    open_quad = new_mesh("OpenQuad", [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                         [(0, 1, 2, 3)])
    surf_open = clean.find_surface_defects(open_quad.data)

    # Two triangles sharing edge 1-2. Face (0,1,2) traverses it 1->2; face
    # (1,2,3) traverses it 1->2 as well, i.e. the SAME direction, which is
    # exactly the disagreement a consistent surface never has. (The correctly
    # wound neighbor would be (1,3,2), which traverses 2->1.)
    flipped = new_mesh("FlippedPair",
                       [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 0, 0)],
                       [(0, 1, 2), (1, 2, 3)])
    surf_flip_before = clean.find_surface_defects(flipped.data)
    only_select(flipped)
    recalc_res = bpy.ops.object.ppf_recalc_normals_outside()
    surf_flip_after = clean.find_surface_defects(flipped.data)
    dh.record(
        "G_surface_and_winding",
        surf_open["boundary"] == 4
        and surf_open["non_manifold"] == 0
        and surf_flip_before["bad_winding"] >= 1
        and surf_flip_after["bad_winding"] == 0,
        {"open": surf_open, "flip_before": surf_flip_before,
         "flip_after": surf_flip_after, "op": str(recalc_res)},
    )

    # ----- H: scanning never mutates the mesh ---------------------------
    probe = new_mesh(
        "ScanProbe",
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (1 + GAP, 0, 0), (7, 7, 7)],
        [(0, 1, 2, 3)],
    )
    co_before = [tuple(v.co) for v in probe.data.vertices]
    np_before = (len(probe.data.vertices), len(probe.data.polygons))
    only_select(probe)
    bpy.ops.object.ppf_scan_mesh_defects(merge_threshold=TH, area_eps=0.0)
    co_after = [tuple(v.co) for v in probe.data.vertices]
    np_after = (len(probe.data.vertices), len(probe.data.polygons))
    dh.record(
        "H_scan_never_mutates",
        co_before == co_after and np_before == np_after,
        {"counts_before": np_before, "counts_after": np_after,
         "coords_equal": co_before == co_after},
    )

    # ----- I: the report carries the invalidation list ------------------
    # The panel must never recompute this at draw time: resolving it walks
    # every group's pins and lazy-loads the capture cache from disk, which
    # would be a file open on every mouse move.
    cached_obj = new_mesh(
        "CachedDup",
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (1 + GAP, 0, 0), (1 + GAP, 1, 0), (2, 0, 0), (2, 1, 0)],
        [(0, 1, 2, 3), (4, 6, 7, 5)],
    )
    cached_obj.modifiers.new(name="ContactSolverCache", type="MESH_CACHE")
    rep_cached = clean.scan_object(cached_obj, merge_threshold=TH, area_eps=0.0)
    plain = clean.scan_object(clean_obj, merge_threshold=TH, area_eps=0.0)
    dh.record(
        "I_report_carries_dependents",
        "dependents" in rep_cached
        and any("PC2" in d for d in rep_cached["dependents"])
        and "dependents" in plain
        and not any("PC2" in d for d in plain["dependents"]),
        {"cached": rep_cached.get("dependents"),
         "plain": plain.get("dependents")},
    )

    # ----- J: the panel draw path runs for every defect row -------------
    # A real sidebar draw does not reliably fire in the rig's hidden window,
    # so drive the draw function with a recording stub layout. This is where
    # a bad iface_ format placeholder would raise KeyError / IndexError, and
    # it covers every renderer including the ones a clean mesh never reaches.
    class StubLayout:
        def __init__(self, sink):
            self.sink = sink
            self.enabled = True
            self.alert = False

        def box(self):
            return StubLayout(self.sink)

        def column(self, **kwargs):
            return StubLayout(self.sink)

        def row(self, **kwargs):
            return StubLayout(self.sink)

        def label(self, **kwargs):
            self.sink.append(("label", kwargs.get("text", "")))

        def prop(self, *args, **kwargs):
            self.sink.append(("prop", args[1] if len(args) > 1 else ""))

        def operator(self, idname, **kwargs):
            self.sink.append(("operator", idname))

            class _Op:
                pass
            return _Op()

    draw_sink = []
    draw_err = ""
    try:
        # A mesh carrying every defect at once, so no renderer is skipped.
        allbad = new_mesh(
            "AllDefects",
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
             (1 + GAP, 0, 0), (1 + GAP, 1, 0), (2, 0, 0), (2, 1, 0),
             (9, 9, 9),
             (3, 0, 0), (4, 0, 0), (5, 0, 0)],
            [(0, 1, 2, 3), (4, 6, 7, 5), (0, 1, 2, 3), (9, 10, 11)],
        )
        only_select(allbad)
        bpy.ops.object.ppf_scan_mesh_defects(merge_threshold=TH, area_eps=0.0)
        rep_all = clean.get_scan_report(allbad.name)
        clean.draw_mesh_cleaning(StubLayout(draw_sink), bpy.context)
        # And the genuinely-clean branch: a closed, triangulated,
        # consistently wound tetrahedron has neither errors nor notes.
        tet = new_mesh("CleanTet",
                       [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
                       [(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)])
        only_select(tet)
        bpy.ops.object.ppf_scan_mesh_defects(merge_threshold=TH, area_eps=0.0)
        rep_tet = clean.get_scan_report(tet.name)
        clean.draw_mesh_cleaning(StubLayout(draw_sink), bpy.context)
    except Exception as e:
        draw_err = f"{type(e).__name__}: {e}"

    drawn_ops = {name for kind, name in draw_sink if kind == "operator"}
    labels = [text for kind, text in draw_sink if kind == "label"]
    dh.record(
        "J_panel_draw_renders_every_row",
        draw_err == ""
        and "object.ppf_scan_mesh_defects" in drawn_ops
        and "object.ppf_merge_by_distance" in drawn_ops
        and "object.ppf_delete_duplicate_faces" in drawn_ops
        and "object.ppf_remove_loose_vertices" in drawn_ops
        and "object.ppf_dissolve_degenerate" in drawn_ops
        and "object.ppf_triangulate_for_solver" in drawn_ops
        and any("No defects found" in t for t in labels)
        and (rep_tet or {}).get("n_errors") == 0
        and (rep_tet or {}).get("n_notes") == 0,
        {"err": draw_err, "ops": sorted(drawn_ops), "n_labels": len(labels),
         "all_errors": (rep_all or {}).get("n_errors"),
         "tet": {k: (rep_tet or {}).get(k) for k in ("n_errors", "n_notes")}},
    )

    # ----- K: an ordinary cloth panel is not flagged as broken ----------
    # An open surface built from quads is exactly what cloth looks like. It
    # must report zero ERRORS (so the panel does not cry wolf) while still
    # surfacing the boundary / quad observations as notes.
    sheet_coords = []
    sheet_faces = []
    for iy in range(4):
        for ix in range(4):
            sheet_coords.append((ix * 0.25, iy * 0.25, 0.0))
    for iy in range(3):
        for ix in range(3):
            a = iy * 4 + ix
            sheet_faces.append((a, a + 1, a + 5, a + 4))
    sheet = new_mesh("ClothSheet", sheet_coords, sheet_faces)
    rep_sheet = clean.scan_object(sheet, merge_threshold=TH, area_eps=0.0)
    dh.record(
        "K_plain_cloth_sheet_has_no_errors",
        rep_sheet["n_errors"] == 0
        and rep_sheet["n_notes"] > 0
        and rep_sheet["defects"]["surface"]["boundary"] == 12
        and rep_sheet["defects"]["resplittable"]["count"] == 9,
        {"n_errors": rep_sheet["n_errors"], "n_notes": rep_sheet["n_notes"],
         "surface": rep_sheet["defects"]["surface"],
         "resplittable": rep_sheet["defects"]["resplittable"]},
    )

    # ----- L: a PINNED face-less vertex is exempt -----------------------
    # A pin is a Dirichlet condition, so a face-less pinned vertex (a sewn
    # curtain hook) is valid, not stray, and the encoder exempts it. The scan
    # resolves the pin set LAZILY for speed (pinned_vertices_for walks every
    # vertex group through an O(verts x groups) RNA loop, measured at 336 ms
    # on a rigged 6.4k-vertex body), so this subtest exists to prove the lazy
    # path still applies the exemption rather than skipping it.
    pin_err = ""
    try:
        hook = new_mesh(
            "HookedSeam",
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 2.0, 0)],
            [(0, 1, 2, 3)],
        )
        # Vertex 4 is in no face -> would normally be reported as isolated.
        vg = hook.vertex_groups.new(name="Hooks")
        vg.add([4], 1.0, "REPLACE")

        unpinned = clean.scan_object(hook, merge_threshold=TH, area_eps=0.0)

        # Register the pin the way the addon does, then rescan.
        hook_group = dh.api.solver.create_group("Hooked", "SHELL")
        root2 = dh.groups.get_addon_data(bpy.context.scene)
        target_group = None
        for gi in range(32):
            g = getattr(root2, f"object_group_{gi}", None)
            if g is not None and g.name == "Hooked":
                target_group = g
                break
        item = target_group.assigned_objects.add()
        item.name = hook.name
        item.uuid = uuid_mod.get_or_create_object_uuid(hook)
        pin_item = target_group.pin_vertex_groups.add()
        pin_item.name = dh.groups.encode_vertex_group_identifier(
            hook.name, "Hooks")
        pin_item.object_uuid = item.uuid

        resolved = clean.pinned_vertices_for(hook)
        pinned_rep = clean.scan_object(hook, merge_threshold=TH, area_eps=0.0)
    except Exception as e:
        pin_err = f"{type(e).__name__}: {e}"
        unpinned = pinned_rep = {"defects": {}}
        resolved = set()

    dh.record(
        "L_pinned_faceless_vertex_is_exempt",
        pin_err == ""
        and unpinned["defects"]["isolated_verts"]["count"] == 1
        and unpinned["defects"]["isolated_verts"]["verts"] == [4]
        and 4 in resolved
        and pinned_rep["defects"]["isolated_verts"]["count"] == 0,
        {"err": pin_err,
         "unpinned": unpinned["defects"].get("isolated_verts"),
         "resolved_pins": sorted(resolved),
         "pinned": pinned_rep["defects"].get("isolated_verts")},
    )

    # ----- M: a failing overlay builder must not rebuild every redraw ----
    # The overlay's three scene-topology builders each own a cache key, and a
    # builder that raises promotes its key so it is not retried until the
    # scene state changes. Both halves matter for frame time: _build_pin_data
    # resolves pin vertex groups through an O(verts x groups) walk (a few
    # hundred ms on a rigged 6.4k-vertex body), so a builder retried per draw
    # is a permanent frame cost, and a shared key would spread one builder's
    # failure onto the other two.
    ov = __import__(pkg + ".ui.dynamics.overlay",
                    fromlist=["_rebuild_cached", "_overlay_cache"])
    calls = {"n": 0}

    def _always_fails():
        calls["n"] += 1
        raise RuntimeError("synthetic builder failure")

    ov._overlay_cache["pin_key"] = None
    ov._overlay_cache["failures"].pop("pin", None)
    key = (7, 3)
    for _ in range(5):
        ov._rebuild_cached("pin", key, _always_fails)
    calls_same_key = calls["n"]
    # A genuine scene change must still retry it.
    ov._rebuild_cached("pin", (8, 3), _always_fails)
    calls_after_change = calls["n"]

    # And one builder failing must not force a healthy sibling to rebuild.
    ok_calls = {"n": 0}

    def _succeeds():
        ok_calls["n"] += 1

    ov._overlay_cache["rod_key"] = None
    for _ in range(4):
        ov._rebuild_cached("rod", key, _succeeds)

    dh.record(
        "M_failing_overlay_builder_does_not_spin",
        calls_same_key == 1
        and calls_after_change == 2
        and ok_calls["n"] == 1
        and ov._overlay_cache["pin_key"] == (8, 3),
        {"calls_same_key": calls_same_key,
         "calls_after_change": calls_after_change,
         "healthy_sibling_calls": ok_calls["n"],
         "pin_key": ov._overlay_cache["pin_key"]},
    )

    # ----- N: a count-changing repair clears the invalidated caches ------
    cache_err = ""
    try:
        victim = new_mesh(
            "CacheVictim",
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
             (1 + GAP, 0, 0), (1 + GAP, 1, 0), (2, 0, 0), (2, 1, 0)],
            [(0, 1, 2, 3), (4, 6, 7, 5)],
        )
        victim.modifiers.new(name="ContactSolverCache", type="MESH_CACHE")
        had_mod = any(m.type == "MESH_CACHE" for m in victim.modifiers)
        only_select(victim)
        v_before = len(victim.data.vertices)
        res_clear = bpy.ops.object.ppf_merge_by_distance(
            merge_threshold=TH, acknowledge=True, clear_stale_caches=True)
        mod_after = any(m.type == "MESH_CACHE" for m in victim.modifiers)
        merged_ok = len(victim.data.vertices) == v_before - 2

        # Opting out must leave the cache in place.
        keeper = new_mesh(
            "CacheKeeper",
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
             (1 + GAP, 0, 0), (1 + GAP, 1, 0), (2, 0, 0), (2, 1, 0)],
            [(0, 1, 2, 3), (4, 6, 7, 5)],
        )
        keeper.modifiers.new(name="ContactSolverCache", type="MESH_CACHE")
        only_select(keeper)
        bpy.ops.object.ppf_merge_by_distance(
            merge_threshold=TH, acknowledge=True, clear_stale_caches=False)
        keeper_mod = any(m.type == "MESH_CACHE" for m in keeper.modifiers)
    except Exception as e:
        cache_err = f"{type(e).__name__}: {e}"
        had_mod = mod_after = keeper_mod = merged_ok = False

    dh.record(
        "N_count_change_clears_invalidated_caches",
        cache_err == ""
        and had_mod and merged_ok
        and not mod_after   # cleared when opted in
        and keeper_mod,     # kept when opted out
        {"err": cache_err, "had_mod": had_mod, "merged": merged_ok,
         "mod_after_clear": mod_after, "mod_after_keep": keeper_mod},
    )

    # ----- O: a vertex-group CONTENT change must not trigger a rename scan -
    # A pin's stored content hash exists to follow a RENAME, and resolve_vg_name
    # must only pay its scan when the stored name is gone. Welding vertices
    # rewrites a group's index list, so the hash stops matching while the name
    # stays valid; searching for a rename there would rehash EVERY vertex group
    # on the object (~450k vertex iterations on a 70-group character) on every
    # draw of the Groups panel, and find nothing. Two guarantees are asserted:
    # the scan does not run while the name exists, and a genuine rename is
    # still followed.
    reg = __import__(pkg + ".core.uuid_registry",
                     fromlist=["resolve_vg_name", "compute_vg_hash",
                               "_iter_vg_candidates"])
    scan_calls = {"n": 0}
    real_iter = reg._iter_vg_candidates

    def _counting_iter(obj):
        scan_calls["n"] += 1
        return real_iter(obj)

    scan_err = ""
    try:
        subject = new_mesh("HashSubject",
                           [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                           [(0, 1, 2, 3)])
        vgroup = subject.vertex_groups.new(name="Pinned")
        vgroup.add([0, 1], 1.0, "REPLACE")
        # Pad with extra groups so a scan would be visibly expensive.
        for k in range(8):
            subject.vertex_groups.new(name=f"Filler{k}")
        stored = reg.compute_vg_hash(subject, "Pinned")

        reg._iter_vg_candidates = _counting_iter

        # 1. Unchanged: resolves to itself, no scan.
        same = reg.resolve_vg_name(subject, "Pinned", stored)
        after_same = scan_calls["n"]

        # 2. CONTENT changed (the merge case): hash no longer matches, but the
        #    name is still valid, so it must resolve without scanning.
        vgroup.add([2], 1.0, "REPLACE")
        changed = reg.resolve_vg_name(subject, "Pinned", stored)
        after_changed = scan_calls["n"]

        # 3. Genuine RENAME: old name gone, so the scan runs and finds it.
        fresh_hash = reg.compute_vg_hash(subject, "Pinned")
        vgroup.name = "PinnedRenamed"
        renamed = reg.resolve_vg_name(subject, "Pinned", fresh_hash)
        after_rename = scan_calls["n"]
    except Exception as e:
        scan_err = f"{type(e).__name__}: {e}"
        same = changed = renamed = None
        after_same = after_changed = after_rename = -1
    finally:
        reg._iter_vg_candidates = real_iter

    dh.record(
        "O_content_change_does_not_trigger_rename_scan",
        scan_err == ""
        and same == "Pinned" and after_same == 0
        and changed == "Pinned" and after_changed == 0
        and renamed == "PinnedRenamed" and after_rename == 1,
        {"err": scan_err, "same": same, "scans_after_same": after_same,
         "changed": changed, "scans_after_change": after_changed,
         "renamed": renamed, "scans_after_rename": after_rename},
    )

    # ----- P: a count-changing repair re-stamps the pin hash --------------
    stamp_err = ""
    try:
        stamped = new_mesh(
            "HashRestamp",
            [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
             (1 + GAP, 0, 0), (1 + GAP, 1, 0), (2, 0, 0), (2, 1, 0)],
            [(0, 1, 2, 3), (4, 6, 7, 5)],
        )
        svg = stamped.vertex_groups.new(name="Held")
        svg.add(list(range(len(stamped.data.vertices))), 1.0, "REPLACE")

        sgroup = dh.api.solver.create_group("Restamp", "SHELL")
        root3 = dh.groups.get_addon_data(bpy.context.scene)
        tgt = None
        for gi in range(32):
            g = getattr(root3, f"object_group_{gi}", None)
            if g is not None and g.name == "Restamp":
                tgt = g
                break
        a_item = tgt.assigned_objects.add()
        a_item.name = stamped.name
        a_item.uuid = uuid_mod.get_or_create_object_uuid(stamped)
        p_item = tgt.pin_vertex_groups.add()
        p_item.name = dh.groups.encode_vertex_group_identifier(
            stamped.name, "Held")
        p_item.object_uuid = a_item.uuid
        p_item.vg_hash = str(reg.compute_vg_hash(stamped, "Held"))
        hash_before = p_item.vg_hash

        only_select(stamped)
        bpy.ops.object.ppf_merge_by_distance(
            merge_threshold=TH, acknowledge=True, clear_stale_caches=True)

        hash_after = p_item.vg_hash
        expected = str(reg.compute_vg_hash(stamped, "Held"))
    except Exception as e:
        stamp_err = f"{type(e).__name__}: {e}"
        hash_before = hash_after = expected = ""

    dh.record(
        "P_repair_restamps_pin_vg_hash",
        stamp_err == ""
        and hash_before != hash_after   # the weld changed the contents
        and hash_after == expected,     # and the stored hash now matches
        {"err": stamp_err, "before": hash_before, "after": hash_after,
         "expected": expected},
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
