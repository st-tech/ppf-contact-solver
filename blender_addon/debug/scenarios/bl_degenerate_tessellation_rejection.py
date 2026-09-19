# File: scenarios/bl_degenerate_tessellation_rejection.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Tessellation triangles with no usable rest shape: rejection at Transfer.
#
# A quad carrying a vertex on, or very near, the straight edge between two
# of its neighbors has a perfectly good area of its own, but Blender
# tessellates it along the diagonal that puts all three of those vertices
# in one triangle. The encoder ships ``mesh.loop_triangles`` verbatim (see
# ``bl_ngon_triangulation``), so that triangle reaches the solver, and it
# costs the artist the run in one of two ways that look nothing alike:
#
#   * EXACTLY collinear aborts at scene build with ``degenerate face N:
#     area is zero (collinear or duplicate vertex indices)``
#     (``triutils::face_areas``, measured against the CUDA backend).
#   * NEARLY collinear clears that assertion. It is finite, it inverts, and
#     the inverse it returns is finite too. What it carries is magnitude:
#     ``inv_rest`` goes as ``1 / smin``, the elastic Hessian squares that,
#     and the linear solve reports ``p^T A p is not-a-number at iter 0``.
#     That is issue #144, whose reporter could say only that a build
#     failed, and whose mesh ran once they triangulated it by hand.
#
# Both name an index into the solver's own concatenated mesh, so neither
# tells the artist which object to fix, and both cost an upload and a build
# to arrive. ``utils.find_degenerate_tessellation`` finds the triangles
# first and ``encoder.mesh._build_obj_data`` refuses the Transfer, naming
# the object, the offending faces, and which of the two repairs applies.
#
# The test is the rest matrix's singular-value ratio against
# ``utils.min_rest_condition()``, ``sqrt(float32 eps)``, and it has to be:
# an exact-area test sees the first case and is blind to the second, which
# is what let issue #144's mesh through. ``rig_degenerate_rest_shape``
# covers the solver's own gate on the same quantity; both grant the same
# set.
#
# Subtests:
#   A. tessellation_makes_a_zero_area_triangle
#         The mechanism, measured: the inline-vertex quad has a positive
#         polygon area, and its loop-triangle tessellation still contains
#         exactly one zero-area triangle.
#   B. encoder_rejects_degenerate_tessellation
#         Assigned to a SHELL group, encode must raise ValueError naming
#         the zero-area triangles and pointing at Triangulate Faces.
#   C. triangulated_quad_passes
#         The reporter's own workaround, verified: after a BEAUTY
#         triangulation of that quad no zero-area triangle is left and the
#         same scene encodes without raising.
#   D. coincident_vertex_face_names_the_merge_remedy
#         A quad with two coincident CONSECUTIVE vertices has a zero-length
#         boundary edge, and every triangulation contains every boundary
#         edge, so no triangulation of it can avoid a zero-area triangle
#         (measured: Ctrl+T leaves one). The error must say to merge rather
#         than to triangulate.
#   E. plain_quad_passes
#         Control: a well-formed quad is not flagged, so the check costs
#         valid meshes nothing.
#   F. near_collinear_quad_is_flagged
#         The issue-#144 shape: vertex 1 sits 3.75e-07 above the line, so
#         the quad has area, no coincident corners, and a tessellation the
#         old exact-area test could not see. It must be flagged, and as
#         repairable by triangulating.
#   G. triangulated_near_collinear_quad_passes
#         Ctrl+T on that quad picks the other diagonal and clears it, which
#         is the reporter's own workaround at the addon gate.
#   H. thin_but_sound_quad_passes
#         Control on the other side of the threshold: vertex 1 at 2.5e-03
#         is a ratio of 1e-03, an aspect ratio of about 800:1. Thin
#         geometry is legitimate and must still transfer.
#   I. threshold_matches_the_solver
#         ``min_rest_condition()`` is computed, not written down, and has
#         to equal the solver's ``REST_SHAPE_MIN_CONDITION``. Two gates
#         disagreeing on the number is two gates granting different sets.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. A full rig sweep against a CPU build passed it, and that run is the
# evidence this line rests on.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import bmesh
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def make_quad(name, coords):
    mesh = bpy.data.meshes.new(name + "Mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.from_pydata([tuple(c) for c in coords], [], [tuple(range(len(coords)))])
    mesh.update()
    return obj


def triangulate(obj):
    # What Face > Triangulate Faces (Ctrl+T) runs.
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.triangulate(bm, faces=bm.faces[:],
                          quad_method="BEAUTY", ngon_method="BEAUTY")
    bm.to_mesh(obj.data)
    bm.free()
    obj.data.update()


# Vertex 1 sits exactly on the straight edge from vertex 0 to vertex 2.
INLINE_QUAD = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)]
PLAIN_QUAD = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
# Vertices 0 and 1 are coincident, so the face carries a zero-length boundary
# edge even though the face as a whole still has area.
COINCIDENT_QUAD = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)]

# Vertex 1 sits a height h above the straight line from vertex 0 to vertex 2,
# so face (0, 1, 2) of the 0-2 split has singular values sqrt(5) and
# 2h/sqrt(5): a conditioning of exactly 0.4 * h. h is therefore the only thing
# that moves a case across the threshold, and the two below sit on either side
# of it with everything else held equal.
def _kite(h):
    return [(0.0, 0.0, 0.0), (1.0, h, 0.0), (2.0, 0.0, 0.0), (1.0, -1.0, 0.0)]


# The worst face of the reporter's own mesh, as a standalone TRIANGLE: real
# area (3.3e-10), corners 2.5 cm apart, and near-collinear all the same. No
# split can help it, and nothing about it is coincident or zero-area, so it is
# the case that separates the two unrepairable remedies from each other.
NEAR_COLLINEAR_TRIANGLE = [
    (0.0, 0.0, 0.0),
    (0.056_781_4, 0.0, 0.0),
    (0.032_217_3, 1.172_5e-8, 0.0),
]
# 1.5e-07, the conditioning of the worst face of the reporter's own mesh.
NEAR_COLLINEAR_QUAD = _kite(3.75e-07)
# 1.0e-03, an aspect ratio of about 800:1 and still sound.
THIN_SOUND_QUAD = _kite(2.5e-03)


try:
    dh = DriverHelpers(pkg, result)
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["encode_obj", "compute_data_hash"])
    utils = __import__(pkg + ".core.utils",
                       fromlist=["find_degenerate_tessellation"])
    uuid_mod = __import__(pkg + ".core.uuid_registry",
                          fromlist=["get_or_create_object_uuid"])

    dh.log("setup_start")
    dh.reset_scene_to_pinned_plane(name="DegenBaseMesh")
    dh.save_blend(PROBE_DIR, "degenerate_tessellation.blend")
    root = dh.configure_state(project_name="degenerate_tessellation",
                              frame_count=6)
    dh.api.solver.create_group("Cloth", "SHELL")

    def assign(obj):
        item = root.object_group_0.assigned_objects.add()
        item.name = obj.name
        item.uuid = uuid_mod.get_or_create_object_uuid(obj)
        return item

    def clear_assigned():
        while len(root.object_group_0.assigned_objects):
            root.object_group_0.assigned_objects.remove(
                len(root.object_group_0.assigned_objects) - 1
            )

    def encode_error():
        try:
            encoder_mesh.compute_data_hash(bpy.context)
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"
        return ""

    # ----- A: the mechanism -------------------------------------------
    inline = make_quad("InlineVertexQuad", INLINE_QUAD)
    found = utils.find_degenerate_tessellation(inline)
    poly_area = inline.data.polygons[0].area
    dh.record(
        "A_tessellation_makes_a_zero_area_triangle",
        poly_area > 0.0
        and found["count"] == 1
        and found["polygons"] == [0]
        and found["all_repairable_by_triangulation"] is True,
        {"polygon_area": poly_area, "found": found,
         "n_tris": len(inline.data.loop_triangles)},
    )

    # ----- B: rejected at Transfer, pointing at Ctrl+T -----------------
    assign(inline)
    err_b = encode_error()
    dh.record(
        "B_encoder_rejects_degenerate_tessellation",
        "no usable rest shape" in err_b
        and "InlineVertexQuad" in err_b
        and "Triangulate Faces" in err_b
        and "Merge > By Distance" not in err_b,
        {"err": err_b[:400]},
    )

    # ----- C: the reported workaround actually clears it ---------------
    triangulate(inline)
    after = utils.find_degenerate_tessellation(inline)
    err_c = encode_error()
    dh.record(
        "C_triangulated_quad_passes",
        after["count"] == 0 and err_c == "",
        {"after": after, "err": err_c[:400],
         "n_tris": len(inline.data.loop_triangles)},
    )

    # ----- D: a face that no triangulation can rescue -------------------
    clear_assigned()
    coincident = make_quad("CoincidentQuad", COINCIDENT_QUAD)
    found_d = utils.find_degenerate_tessellation(coincident)
    assign(coincident)
    err_d = encode_error()
    dh.record(
        "D_coincident_vertex_face_names_the_merge_remedy",
        found_d["count"] >= 1
        and found_d["all_repairable_by_triangulation"] is False
        and "no usable rest shape" in err_d
        and "Merge > By Distance" in err_d
        and "Triangulate Faces" not in err_d,
        {"found": found_d, "err": err_d[:400]},
    )

    # ----- E: control, a valid quad is untouched ------------------------
    clear_assigned()
    plain = make_quad("PlainQuad", PLAIN_QUAD)
    found_e = utils.find_degenerate_tessellation(plain)
    assign(plain)
    err_e = encode_error()
    dh.record(
        "E_plain_quad_passes",
        found_e["count"] == 0 and err_e == "",
        {"found": found_e, "err": err_e[:400]},
    )

    # ----- F: the issue-#144 shape, which has area and still fails -------
    clear_assigned()
    near = make_quad("NearCollinearQuad", NEAR_COLLINEAR_QUAD)
    found_f = utils.find_degenerate_tessellation(near)
    poly_area_f = near.data.polygons[0].area
    # An exact-area test measured this and saw nothing, which is the
    # whole point of the case: every triangle here has area.
    tri_areas = [t.area for t in near.data.loop_triangles]
    assign(near)
    err_f = encode_error()
    dh.record(
        "F_near_collinear_quad_is_flagged",
        poly_area_f > 0.0
        and min(tri_areas) > 0.0
        and found_f["count"] == 1
        and found_f["polygons"] == [0]
        and found_f["all_repairable_by_triangulation"] is True
        and "no usable rest shape" in err_f
        and "NearCollinearQuad" in err_f
        and "Triangulate Faces" in err_f,
        {"polygon_area": poly_area_f, "tri_areas": tri_areas,
         "found": found_f, "err": err_f[:400]},
    )

    # ----- G: Ctrl+T clears it, the reported workaround ------------------
    triangulate(near)
    found_g = utils.find_degenerate_tessellation(near)
    err_g = encode_error()
    dh.record(
        "G_triangulated_near_collinear_quad_passes",
        found_g["count"] == 0 and err_g == "",
        {"found": found_g, "err": err_g[:400],
         "n_tris": len(near.data.loop_triangles)},
    )

    # ----- H: control on the other side of the threshold -----------------
    clear_assigned()
    thin = make_quad("ThinSoundQuad", THIN_SOUND_QUAD)
    found_h = utils.find_degenerate_tessellation(thin)
    assign(thin)
    err_h = encode_error()
    dh.record(
        "H_thin_but_sound_quad_passes",
        found_h["count"] == 0 and err_h == "",
        {"found": found_h, "err": err_h[:400]},
    )

    # ----- J: a flagged TRIANGLE gets its own remedy ---------------------
    # It has real area and no coincident corners, so the merge remedy would
    # describe geometry it does not have and Merge By Distance would need a
    # 2 cm threshold to weld anything. It is also its own only triangulation,
    # so the triangulate remedy would name a repair that cannot exist.
    clear_assigned()
    tri = make_quad("NearCollinearTriangle", NEAR_COLLINEAR_TRIANGLE)
    found_j = utils.find_degenerate_tessellation(tri)
    assign(tri)
    err_j = encode_error()
    dh.record(
        "J_flagged_triangle_names_its_own_remedy",
        found_j["count"] >= 1
        and found_j["all_repairable_by_triangulation"] is False
        and found_j["triangle_polygons"] == found_j["polygons"]
        and "no usable rest shape" in err_j
        and "already a triangle" in err_j
        and "Triangulate Faces" not in err_j
        and "Merge > By Distance" not in err_j,
        {"found": found_j, "err": err_j[:500]},
    )

    # ----- K: the verdict matches what triangulating actually does -------
    # The predicate scores the fill the repair applies (BEAUTY), so its promise
    # and the repair's result have to agree on every flagged fixture. Scoring
    # any other fill lets it promise a repair the button does not deliver.
    clear_assigned()
    agreement = {}
    for label, coords in (
        ("near_collinear_quad", NEAR_COLLINEAR_QUAD),
        ("inline_quad", INLINE_QUAD),
        ("coincident_quad", COINCIDENT_QUAD),
        ("near_collinear_triangle", NEAR_COLLINEAR_TRIANGLE),
    ):
        probe = make_quad("Agree_" + label, coords)
        before = utils.find_degenerate_tessellation(probe)
        triangulate(probe)
        after = utils.find_degenerate_tessellation(probe)
        agreement[label] = {
            "flagged": before["count"],
            "promised": before["all_repairable_by_triangulation"],
            "delivered": after["count"] == 0,
        }
    dh.record(
        "K_verdict_matches_what_triangulating_does",
        all(a["flagged"] >= 1 for a in agreement.values())
        and all(a["promised"] == a["delivered"] for a in agreement.values()),
        agreement,
    )

    # ----- L: the gate judges the pose it is HANDED, not the base cage ----
    # The encoder ships the starting frame's deform-evaluated positions, and
    # hands them to this gate, so a deform that moves a corner onto the line
    # between its neighbors is refused here rather than at scene build with an
    # index into the solver's own concatenated mesh. The deform mechanism is
    # the encoder's business; what this pins is that the gate measures the
    # positions it is given.
    # The two shapes whose verdicts checks A and E already pin: INLINE_QUAD is
    # flagged, PLAIN_QUAD is not. Building on the first and handing it the
    # second's positions has to flip the verdict, which it can only do if the
    # handed pose is what gets measured.
    clear_assigned()
    posed = make_quad("PoseJudgedQuad", INLINE_QUAD)
    as_authored = utils.find_degenerate_tessellation(posed)
    as_shipped = utils.find_degenerate_tessellation(
        posed, local_verts=PLAIN_QUAD
    )
    dh.record(
        "L_gate_judges_the_pose_it_is_handed",
        as_authored["count"] >= 1 and as_shipped["count"] == 0,
        {"as_authored": as_authored, "as_shipped": as_shipped,
         "note": "the base cage is flagged; the handed pose is not"},
    )

    # ----- O: Mesh Cleaning and the Transfer gate agree -------------------
    # Certifying a mesh clean and then refusing the Transfer on it is the one
    # outcome the cleaning tool must not produce. An exact-area test cannot see
    # a near-collinear tessellation: the face has real area.
    clear_assigned()
    cleaning = __import__(pkg + ".mesh_ops.cleaning_ops",
                          fromlist=["scan_object"])
    near = make_quad("CleanScanNearCollinear", NEAR_COLLINEAR_QUAD)
    report = cleaning.scan_object(
        near, merge_threshold=1e-4, area_eps=1e-12
    )["defects"]
    assign(near)
    err_o = encode_error()
    dh.record(
        "O_mesh_cleaning_sees_what_the_transfer_gate_refuses",
        report["degenerate_tessellation"]["count"] >= 1
        and report["degenerate_faces"]["count"] == 0
        and bool(err_o),
        {"degenerate_tessellation": report["degenerate_tessellation"]["count"],
         "degenerate_faces_area_test": report["degenerate_faces"]["count"],
         "transfer_refused": bool(err_o)},
    )

    # ----- N: a MIXED offender set names every subset ---------------------
    # A repairable quad and a flagged triangle in one mesh. A mutually
    # exclusive remedy chain names one of them and leaves the artist repairing
    # half the scene, then meeting the same refusal again.
    clear_assigned()
    mixed_co = list(NEAR_COLLINEAR_QUAD) + [
        (10.0, 0.0, 0.0), (10.056_781_4, 0.0, 0.0), (10.032_217_3, 1.172_5e-8, 0.0),
    ]
    mesh = bpy.data.meshes.new("MixedMesh")
    mixed = bpy.data.objects.new("MixedOffenders", mesh)
    bpy.context.collection.objects.link(mixed)
    mesh.from_pydata(mixed_co, [], [(0, 1, 2, 3), (4, 5, 6)])
    mesh.update()
    found_n = utils.find_degenerate_tessellation(mixed)
    assign(mixed)
    err_n = encode_error()
    dh.record(
        "N_a_mixed_offender_set_names_every_subset",
        len(found_n["repairable_polygons"]) >= 1
        and len(found_n["triangle_polygons"]) >= 1
        and "Triangulate Faces" in err_n
        and "already a" in err_n,
        {"found": found_n, "err": err_n[:500]},
    )

    # ----- M: the conditioning test applies only where it decides -------
    # A SHELL's triangles become elastic elements and go through
    # `invert_rest_or_panic2`. A stationary STATIC collider and an fTetWild
    # SOLID's surface go to `make_collision_mesh`, whose only per-triangle
    # check is that the area is positive, so refusing those on conditioning
    # rejects geometry the solver accepts. A ZERO-area triangle is refused
    # everywhere, because that is what the solver itself asserts.
    verdicts = {}
    for label, object_type, coords in (
        ("shell_near_collinear", "SHELL", NEAR_COLLINEAR_QUAD),
        ("static_near_collinear", "STATIC", NEAR_COLLINEAR_QUAD),
        ("shell_zero_area", "SHELL", INLINE_QUAD),
        ("static_zero_area", "STATIC", INLINE_QUAD),
    ):
        clear_assigned()
        root.object_group_0.object_type = object_type
        probe = make_quad("Scoped_" + label, coords)
        assign(probe)
        verdicts[label] = bool(encode_error())
    clear_assigned()
    root.object_group_0.object_type = "SHELL"
    dh.record(
        "M_conditioning_applies_only_to_elastic_triangles",
        verdicts == {
            "shell_near_collinear": True,
            "static_near_collinear": False,
            "shell_zero_area": True,
            "static_zero_area": True,
        },
        {"verdicts": verdicts,
         "note": "True means the encoder refused it"},
    )

    # ----- I: the two gates share the number -----------------------------
    import numpy as _np
    threshold = utils.min_rest_condition()
    dh.record(
        "I_threshold_matches_the_solver",
        threshold == float(_np.sqrt(_np.finfo(_np.float32).eps))
        # builder::REST_SHAPE_MIN_CONDITION, written out there because
        # sqrt is not const-evaluable in Rust.
        and abs(threshold - 3.4526698e-4) < 1e-11,
        {"threshold": threshold},
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
