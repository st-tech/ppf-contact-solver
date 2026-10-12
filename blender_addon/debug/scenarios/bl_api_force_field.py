# File: scenarios/bl_api_force_field.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The add-on Python API's read-only force field listing,
# `solver.get_force_fields()`: one row per field object Transfer considers,
# with Blender's name for its type, why Transfer would refuse it, the groups
# it is narrowed to, the objects it pushes, whether it is sampled at every
# solve frame, and a Fluid Flow's source.
#
# The MCP tool get_force_field_settings returns the same rows as `fields`;
# subtest E holds the two equal after the JSON round trip the MCP reply
# travels in. It calls the registered MCP handler in process rather than over
# HTTP, so this scenario needs no loopback server and runs on every platform
# (`bl_mcp_force_field` covers the wire).
#
# Subtests:
#   A. rows_follow_the_scene: one row per field object, in the order the
#      encoder sends them, each with Blender's `field.type` identifier and the
#      name its Physics tab shows (measured on Blender 5.2).
#   B. refusals_name_the_reason: each kind of reason Transfer refuses for
#      reaches `refused`: a setting (a Charge, a Turbulence of Size 0,
#      Harmonic's Multiple Springs, Flow on a Force, Apply Density off a
#      Fluid Flow), the scene around the field (a Wind with Air Density 0, a
#      target group turned Static), and a Fluid Flow's source (no Domain
#      Object, a Volume whose file holds no velocity grid). A sendable field
#      reads None, and no reason names its own object.
#   C. pushes_and_groups: a field that is not narrowed has groups None and
#      pushes every simulated object; one narrowed to a group has that
#      group's uuid and pushes only its object; one narrowed to a group with
#      no objects pushes nothing.
#   D. tracked_is_whether_transfer_samples_every_frame: a keyed field, a
#      parented one and every Fluid Flow are tracked; a field nothing moves
#      is not.
#   E. rows_equal_the_mcp_listing: the API rows, through a JSON round trip,
#      equal get_force_field_settings' `fields`.
#   F. listing_changes_nothing: the listing writes no target entry, no object
#      uuid, moves no frame, and builds no field record (a Fluid Flow's source
#      is read from its metadata, never its voxels).
#   G. a_volume_source_is_described_from_its_header: the Fluid Flow on a
#      Volume reports the file, the grid its Volume names, no density grid,
#      and the resolution of the voxels written here (9 x 2 x 18).
#   H. the_collection_bounds_the_listing: with a Collection chosen, only its
#      field objects are listed.
#   I. the_solver_bridge_passes_only_what_it_is_given: `solver.<tool>()`, the
#      fallthrough to the MCP handlers, passes the arguments a call names and
#      no others, as an MCP call does: a call naming only the collection
#      leaves Visualize on, an explicit '' clears the collection, an explicit
#      False turns Visualize off, and a call leaving out a required argument
#      is refused naming it.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True
# RUN AGAINST A CPU BUILD on dev-head; it starts no solve, so nothing in it is
# backend-specific.
BACKENDS = ("real",)

_DRIVER_BODY = r"""
import json
import os
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

try:
    dh = DriverHelpers(pkg, result)
    ff = __import__(pkg + ".core.force_field", fromlist=["encode"])
    targets = __import__(pkg + ".models.force_field_targets", fromlist=["entry_for"])
    uuids = __import__(pkg + ".core.uuid_registry", fromlist=["get_object_uuid"])
    decorators = __import__(pkg + ".mcp.decorators", fromlist=["get_handler_registry"])
    __import__(pkg + ".mcp.handlers.force_field", fromlist=["get_force_field_settings"])
    solver = dh.api.solver
    scene = bpy.context.scene

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(size=1.0)
    sheet = bpy.context.object
    sheet.name = "FFSheet"
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(2.0, 0.0, 0.0))
    box = bpy.context.object
    box.name = "FFBox"
    dh.save_blend(PROBE_DIR, "api_force_field.blend")
    root = dh.configure_state(project_name="api_force_field", frame_count=11)
    state = root.state
    solver.create_group("Cloth", "SHELL").add(sheet.name)
    second = solver.create_group("Second", "SHELL")
    second.add(box.name)
    third = solver.create_group("Third", "SHELL")
    nobody = solver.create_group("Nobody", "SHELL")
    # Third needs an object to be a group a field can be narrowed to.
    bpy.ops.mesh.primitive_plane_add(size=0.2, location=(0.0, 3.0, 0.0))
    bpy.context.object.name = "FFThirdPlane"
    third.add("FFThirdPlane")

    def field(kind, name, loc=(0.0, 0.0, 0.0)):
        bpy.ops.object.effector_add(type=kind, location=loc)
        o = bpy.context.object
        o.name = name
        return o

    force = field("FORCE", "FFForce")
    keyed = field("FORCE", "FFForceKeyed")
    keyed.field.keyframe_insert("strength", frame=1)
    bpy.ops.object.empty_add(location=(0.0, 0.0, 1.0))
    parent = bpy.context.object
    parent.name = "FFParent"
    parented = field("FORCE", "FFForceParented")
    parented.parent = parent
    narrow = field("TURBULENCE", "FFTurbNarrow")
    narrow.field.size = 1.0
    solver.set_force_field_targets(narrow.name, [second])
    lonely = field("FORCE", "FFForceLonely")
    solver.set_force_field_targets(lonely.name, [nobody])
    charge = field("CHARGE", "FFCharge")
    sizeless = field("TURBULENCE", "FFTurbSizeless")
    springs = field("HARMONIC", "FFHarmonicSprings")
    springs.field.use_multiple_springs = True
    flowing = field("FORCE", "FFForceFlow")
    flowing.field.flow = 1.0
    dense = field("FORCE", "FFForceDensity")
    dense.field.use_smoke_density = True
    wind = field("WIND", "FFWind")
    static_target = field("FORCE", "FFForceStaticTarget")
    solver.set_force_field_targets(static_target.name, [third])
    bpy.ops.zozo_contact_solver.set_group_type(group_uuid=third.uuid, type="STATIC")
    no_source = field("FLUID", "FFFluidNoSource")

    import openvdb

    def volume(name, grids):
        path = os.path.join(PROBE_DIR, name + ".vdb")
        openvdb.write(path, grids=grids)
        data = bpy.data.volumes.new(name)
        data.filepath = path
        data.grids.load()
        data.velocity_unit = "SECOND"
        obj = bpy.data.objects.new(name, data)
        scene.collection.objects.link(obj)
        return obj, path

    d = openvdb.FloatGrid()
    d.name = "density"
    d.getAccessor().setValueOn((0, 0, 0), 1.0)
    no_grid_vol, _ = volume("FFApiDensityOnly", [d])
    # Blender names no velocity grid for a file that holds none (measured:
    # a name set by hand that the file does not hold reads back empty).
    no_grid = field("FLUID", "FFFluidNoGrid")
    no_grid.field.source_object = no_grid_vol
    g = openvdb.Vec3SGrid()
    g.name = "vel"
    g.transform = openvdb.createLinearTransform(voxelSize=0.2)
    acc = g.getAccessor()
    # Active voxels spanning index (1, 2, 3) .. (9, 3, 20): 9 x 2 x 18 cells.
    for ijk, value in {(1, 2, 3): (1.0, 2.0, 3.0), (9, 2, 3): (4.0, 0.0, 0.0),
                       (1, 3, 20): (0.0, 0.0, 5.0)}.items():
        acc.setValueOn(ijk, value)
    good_vol, good_path = volume("FFApiVel", [g])
    on_volume = field("FLUID", "FFFluidVol")
    on_volume.field.source_object = good_vol
    bpy.context.view_layer.update()

    # ----- F (first, on a scene the listing has never seen) --------------
    def snapshot():
        return {
            "frame": scene.frame_current,
            "entries": sorted(e.source_name for e in state.force_field_targets),
            "uuids": sorted(o.name for o in bpy.data.objects if uuids.get_object_uuid(o)),
            "records": sorted(ff._RECORDS),
            "encoded": sorted(ff._ENCODED),
            "texts": len(bpy.data.texts),
        }

    before = snapshot()
    rows = solver.get_force_fields()
    after = snapshot()
    dh.record("F_listing_changes_nothing", before == after,
              {"before": before, "after": after})

    by_name = {row["object"]: row for row in rows}

    # ----- A --------------------------------------------------------------
    BLENDER_NAMES = {"FORCE": "Force", "WIND": "Wind", "TURBULENCE": "Turbulence",
                     "HARMONIC": "Harmonic", "CHARGE": "Charge", "FLUID_FLOW": "Fluid Flow"}
    names = [o.name for o in ff.field_objects(scene, state)]
    wrong = {row["object"]: [row["type"], row["type_name"]] for row in rows
             if row["type"] != bpy.data.objects[row["object"]].field.type
             or row["type_name"] != BLENDER_NAMES.get(row["type"])}
    dh.record("A_rows_follow_the_scene",
              [row["object"] for row in rows] == names and len(rows) == 15 and not wrong,
              {"rows": [row["object"] for row in rows], "fields": names, "wrong": wrong})

    # ----- B --------------------------------------------------------------
    expected = {
        "FFCharge": "Charge fields have no effect on cloth",
        "FFTurbSizeless": "Size is 0",
        "FFHarmonicSprings": "Multiple Springs is not supported",
        "FFForceFlow": "applies only to a Wind or a Fluid Flow",
        "FFForceDensity": "Apply Density is read only on a Fluid Flow",
        "FFWind": "Air Density is 0",
        "FFForceStaticTarget": "group 'Third' is Static",
        "FFFluidNoSource": "no Domain Object",
        "FFFluidNoGrid": "its Volume 'FFApiDensityOnly' names no velocity grid",
    }
    sendable = ["FFForce", "FFForceKeyed", "FFForceParented", "FFTurbNarrow",
                "FFForceLonely", "FFFluidVol"]
    missed = {k: by_name.get(k, {}).get("refused") for k, frag in expected.items()
              if frag not in (by_name.get(k, {}).get("refused") or "")}
    spurious = {k: by_name.get(k, {}).get("refused") for k in sendable
                if by_name.get(k, {}).get("refused") is not None}
    self_named = [k for k, row in by_name.items() if k in (row.get("refused") or "")]
    dh.record("B_refusals_name_the_reason", not missed and not spurious and not self_named,
              {"missed": missed, "spurious": spurious, "self_named": self_named})

    # ----- C --------------------------------------------------------------
    everyone = sorted(["FFSheet", "FFBox"])
    got = {k: (by_name.get(k, {}).get("groups"), sorted(by_name.get(k, {}).get("pushes") or []))
           for k in ("FFForce", "FFTurbNarrow", "FFForceLonely")}
    dh.record("C_pushes_and_groups",
              got["FFForce"] == (None, everyone)
              and got["FFTurbNarrow"] == ([second.uuid], ["FFBox"])
              and got["FFForceLonely"] == ([nobody.uuid], []),
              {"got": got, "second": second.uuid, "nobody": nobody.uuid})

    # ----- D --------------------------------------------------------------
    tracked = {k: by_name.get(k, {}).get("tracked")
               for k in ("FFForce", "FFForceKeyed", "FFForceParented", "FFFluidVol",
                         "FFFluidNoSource")}
    dh.record("D_tracked_is_whether_transfer_samples_every_frame",
              tracked == {"FFForce": False, "FFForceKeyed": True, "FFForceParented": True,
                          "FFFluidVol": True, "FFFluidNoSource": True},
              tracked)

    # ----- E --------------------------------------------------------------
    handler = decorators.get_handler_registry()["get_force_field_settings"]["func"]
    reply = handler({})
    mcp_rows = reply.get("fields")
    api_rows = json.loads(json.dumps(rows))
    diff = [a.get("object") for a, b in zip(api_rows, mcp_rows or []) if a != b]
    dh.record("E_rows_equal_the_mcp_listing",
              reply.get("status") == "success" and mcp_rows == api_rows,
              {"status": reply.get("status"), "differing": diff,
               "counts": [len(api_rows), len(mcp_rows or [])]})

    # ----- G --------------------------------------------------------------
    fluid = by_name.get("FFFluidVol", {}).get("fluid") or {}
    same_file = bool(fluid.get("file")) and (
        os.path.normcase(os.path.realpath(fluid["file"]))
        == os.path.normcase(os.path.realpath(good_path)))
    ok = (fluid.get("source") == "FFApiVel" and fluid.get("kind") == "volume"
          and same_file and fluid.get("sequence") is False
          and fluid.get("velocity_grids") == ["vel"] and fluid.get("density_grid") is None
          and fluid.get("resolution") == [9, 2, 18] and fluid.get("problem") is None
          and fluid.get("solve_frames") == 11 and fluid.get("last_read") is None)
    dh.record("G_a_volume_source_is_described_from_its_header", ok, fluid)

    # ----- H --------------------------------------------------------------
    coll = bpy.data.collections.new("FFApiPicked")
    scene.collection.children.link(coll)
    coll.objects.link(force)
    coll.objects.link(wind)
    solver.param.force_field_collection = coll
    picked = [row["object"] for row in solver.get_force_fields()]
    solver.param.force_field_collection = None
    dh.record("H_the_collection_bounds_the_listing",
              picked == ["FFForce", "FFWind"]
              and len(solver.get_force_fields()) == len(rows),
              {"picked": picked})

    # ----- I --------------------------------------------------------------
    state.force_field_visualize = True
    solver.set_force_field_settings(collection=coll.name)
    named = {"visualize": bool(state.force_field_visualize),
             "collection": getattr(state.force_field_collection, "name", None)}
    solver.set_force_field_settings(collection="")
    cleared = getattr(state.force_field_collection, "name", None)
    solver.set_force_field_settings(visualize=False)
    off = bool(state.force_field_visualize)
    try:
        solver.set_force_field_targets()
        missing = "no error"
    except Exception as e:
        missing = f"{type(e).__name__}: {e}"
    dh.record("I_the_solver_bridge_passes_only_what_it_is_given",
              named == {"visualize": True, "collection": coll.name} and cleared is None
              and off is False and "source" in missing and missing != "no error",
              {"only_the_collection": named, "after_empty_string": cleared,
               "visualize_after_false": off, "required_left_out": missing})

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
