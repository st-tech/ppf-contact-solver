# File: scenarios/bl_pin_resolution_refusals.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Five ways the encoder read a pin, a curve or a contact length from somewhere
# other than what the artist sees, each shipping a plausible payload that
# simulated something else. Encode-only: no build and no run.
#
# Subtests:
#   A. contact_lengths_are_blender_units_and_stateless: the snap operator's
#      per-object keep-out distance is contact gap + offset in Blender units,
#      the same before and after an encode, at any World Scaling; the encoder
#      alone scales it. The group stores no computed copy.
#   B. pin_on_a_missing_group_refused: a pin whose vertex group was renamed
#      and then edited (so nothing identifies it) is refused naming the pin
#      and the object, rather than the object being transferred unpinned.
#   C. twin_group_does_not_take_the_pin: a pin on a Copy Vertex Group twin
#      that is then renamed cannot tell which of two identical groups is its
#      own, so it stays unresolved and is refused rather than bound to the
#      other one.
#   D. keyframes_come_from_the_meshs_own_slot: a pin's keyframed track reads
#      only the action slot animating its mesh; another mesh's curves in the
#      same action do not drive it.
#   E. curve_pins_map_to_sampled_vertices: a pin on a curve lands on the rod
#      vertex its control point became: after a one-point spline (which is not
#      sampled), at a NURBS arc end; a pin the rod has no vertex for is
#      refused by name.
#   F. unsampled_nurbs_points_refused: a NURBS spline whose points do not
#      fill whole arcs (an open order-4 spline with 5 points) is refused at
#      encode, since its last point would not be simulated; Blender's NURBS
#      circle (order 3, 8 cyclic points, four whole arcs) is accepted.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True

# Encode-only: it builds nothing and asks the solver for nothing.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def encode_error(which="both"):
    try:
        if which in ("both", "data"):
            dh.encoder_mesh.encode_obj(bpy.context)
        if which in ("both", "param"):
            params_mod.encode_param(bpy.context)
        return ""
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def plane(name, x, groups_spec=()):
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(x, 0.0, 0.0))
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.subdivide(number_cuts=2)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj = bpy.context.active_object
    obj.name = name
    obj.data.name = name + "Mesh"
    for vg_name, members in groups_spec:
        vg = obj.vertex_groups.new(name=vg_name)
        vg.add(list(members), 1.0, "REPLACE")
    return obj


def curve(name, splines):
    data = bpy.data.curves.new(name + "Data", type="CURVE")
    data.dimensions = "3D"
    for kind, points, order in splines:
        s = data.splines.new(kind)
        s.points.add(len(points) - 1)
        for p, co in zip(s.points, points):
            p.co = (co[0], co[1], co[2], 1.0)
        if kind == "NURBS":
            s.order_u = order
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    return obj


try:
    dh = DriverHelpers(pkg, result)
    params_mod = __import__(pkg + ".core.encoder.params",
                            fromlist=["encode_param", "group_contact_lengths"])
    pin_mod = __import__(pkg + ".core.encoder.pin",
                         fromlist=["_collect_pin_vertex_fcurve_frames"])
    utils = __import__(pkg + ".core.utils", fromlist=["get_id_fcurves"])
    snap_ops = __import__(pkg + ".mesh_ops.snap_ops",
                          fromlist=["_get_pair_contact_gaps"])
    curve_rod = __import__(pkg + ".core.curve_rod",
                           fromlist=["map_cp_pins_to_sampled"])
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    dh.save_blend(PROBE_DIR, "pin_resolution_refusals.blend")
    root = dh.configure_state(project_name="pin_resolution_refusals",
                              frame_count=30)
    state = root.state
    scene = bpy.context.scene

    # ---- A: contact lengths ---------------------------------------------
    left = plane("Left", 0.0)
    right = plane("Right", 2.0)
    g_left = dh.api.solver.create_group("LeftCloth", "SHELL")
    g_left.add(left.name)
    g_right = dh.api.solver.create_group("RightCloth", "SHELL")
    g_right.add(right.name)
    for api_group in (g_left, g_right):
        pg = dh.groups.get_active_group_by_uuid(scene, api_group.uuid)
        pg.use_group_bounding_box_diagonal = False
        pg.contact_gap = 0.002
        pg.contact_offset = 0.01
    state.world_scaling = 0.5
    before = snap_ops._get_pair_contact_gaps(scene, left, right)
    err_a = encode_error("param")
    after = snap_ops._get_pair_contact_gaps(scene, left, right)
    offsets = []
    if not err_a:
        built = params_mod._build_param_dict(bpy.context)
        offsets = sorted(float(p.get("contact-offset", -1)) for p, _n, _u in built["group"])
    pg_left = dh.groups.get_active_group_by_uuid(scene, g_left.uuid)
    dh.record(
        "A_contact_lengths_are_blender_units_and_stateless",
        err_a == ""
        and all(abs(v - 0.012) < 1e-9 for v in before + after)
        and offsets and all(abs(o - 0.005) < 1e-7 for o in offsets)
        and not hasattr(pg_left, "computed_contact_gap"),
        {"before": before, "after": after, "encoded_offsets": offsets,
         "error": err_a[:300]},
    )
    state.world_scaling = 1.0
    g_left.remove(left.name)
    g_right.remove(right.name)

    # ---- B: a pin on a group that is gone ---------------------------------
    sheet = plane("Sheet", 0.0, [("Row", range(0, 4))])
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(sheet.name)
    cloth.create_pin(sheet.name, "Row").move_by(
        delta=(0.1, 0.0, 0.0), frame_start=1, frame_end=10)
    ok_b = encode_error()
    vg = sheet.vertex_groups["Row"]
    vg.name = "Hold"
    vg.add([5], 1.0, "REPLACE")
    err_b = encode_error("data")
    dh.record(
        "B_pin_on_a_missing_group_refused",
        ok_b == ""
        and "Pin 'Row' on 'Sheet'" in err_b
        and "no longer exists" in err_b,
        {"before": ok_b[:300], "error": err_b[:300]},
    )
    cloth.remove(sheet.name)

    # ---- C: a Copy Vertex Group twin --------------------------------------
    twin = plane("Twin", 4.0, [("Top", range(0, 4)), ("TopCopy", range(0, 4))])
    cloth.add(twin.name)
    cloth.create_pin(twin.name, "TopCopy").move_by(
        delta=(0.1, 0.0, 0.0), frame_start=1, frame_end=10)
    twin.vertex_groups["TopCopy"].name = "Hold"
    err_c = encode_error("data")
    pg = dh.groups.get_active_group_by_uuid(scene, cloth.uuid)
    stored = [p.name for p in pg.pin_vertex_groups]
    dh.record(
        "C_twin_group_does_not_take_the_pin",
        "Pin 'TopCopy' on 'Twin'" in err_c
        and not any(n.endswith("[Top]") for n in stored),
        {"error": err_c[:300], "stored": stored},
    )
    cloth.remove(twin.name)

    # ---- D: keyframes from the mesh's own slot ----------------------------
    alpha = plane("Alpha", 0.0, [("AllPin", range(16))])
    beta = plane("Beta", 3.0)
    for v in alpha.data.vertices:
        v.keyframe_insert("co", frame=1)
    for v in alpha.data.vertices:
        v.co.x += 0.6
        v.keyframe_insert("co", frame=20)
    action = alpha.data.animation_data.action
    beta_ad = beta.data.animation_data_create()
    beta_ad.action = action
    beta_ad.action_slot = action.slots.new(id_type="MESH", name="BetaMesh")
    beta.data.vertices[0].keyframe_insert("co", frame=5)
    beta.data.vertices[0].co.x += 3.0
    beta.data.vertices[0].keyframe_insert("co", frame=40)
    frames, lookup = pin_mod._collect_pin_vertex_fcurve_frames(alpha, "AllPin")
    own = utils.get_id_fcurves(alpha.data)
    foreign = [key for key, fc in lookup.items()
               if not any(fc == mine for mine in own)]
    dh.record(
        "D_keyframes_come_from_the_meshs_own_slot",
        frames == [1, 20] and not foreign and len(lookup) == 48,
        {"frames": frames, "foreign": foreign[:6], "n_lookup": len(lookup)},
    )

    # ---- E: curve pins ----------------------------------------------------
    two = curve("TwoSplines", [
        ("POLY", [(0.0, 5.0, 0.0)], 2),
        ("POLY", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)], 2),
    ])
    mapped = curve_rod.map_cp_pins_to_sampled(two, [1, 3])
    try:
        curve_rod.map_cp_pins_to_sampled(two, [0])
        lone = ""
    except ValueError as exc:
        lone = str(exc)
    nurbs = curve("Arcs", [
        ("NURBS", [(float(i), 0.0, 0.0) for i in range(5)], 3),
    ])
    arc_end = curve_rod.map_cp_pins_to_sampled(nurbs, [4])
    try:
        curve_rod.map_cp_pins_to_sampled(nurbs, [1])
        interior = ""
    except ValueError as exc:
        interior = str(exc)
    dh.record(
        "E_curve_pins_map_to_sampled_vertices",
        mapped == [0, 2] and arc_end == [6]
        and "spline 0 has 1 point" in lone
        and "control point 1 is not a vertex of the simulated rod" in interior,
        {"mapped": mapped, "arc_end": arc_end, "lone": lone[:200],
         "interior": interior[:200]},
    )

    # ---- F: NURBS points the rod never samples ----------------------------
    whole = curve("WholeArcs", [
        ("NURBS", [(float(i), 2.0, 0.0) for i in range(4)], 4),
    ])
    bpy.ops.curve.primitive_nurbs_circle_add(location=(0.0, 8.0, 0.0))
    circle = bpy.context.active_object
    circle.name = "Ring"
    ragged = curve("Ragged", [
        ("NURBS", [(float(i), 4.0, 0.0) for i in range(5)], 4),
    ])
    rods = dh.api.solver.create_group("Rods", "ROD")
    rods.add(whole.name)
    rods.add(circle.name)
    ok_f = encode_error("data")
    rods.add(ragged.name)
    err_f = encode_error("data")
    dh.record(
        "F_unsampled_nurbs_points_refused",
        curve_rod.nurbs_points_off_the_rod(whole) == []
        and curve_rod.nurbs_points_off_the_rod(circle) == []
        and ok_f == ""
        and "Curve 'Ragged'" in err_f and "last 1 point(s)" in err_f,
        {"whole_and_ring_error": ok_f[:300], "error": err_f[:300]},
    )
except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 180.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
