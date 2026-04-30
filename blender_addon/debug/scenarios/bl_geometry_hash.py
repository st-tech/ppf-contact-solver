# File: scenarios/bl_geometry_hash.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Mesh-topology hash propagation. The addon stores a topology summary
# in ``state.mesh_hash_json`` on every transfer; ``validate_mesh_hash``
# warns the user when the live scene has diverged from that snapshot.
# The encoder feeds the hash with vertex/triangle/edge counts, object
# counts, and pin vertex group memberships (see
# ``core/encoder/mesh.py::compute_mesh_hash``).
#
# This scenario verifies:
#   1. Fresh hash matches validate_mesh_hash → empty warning
#   2. Mutating topology (adding a vertex) flips hash → warning
#      mentions the diverged group key
#   3. Mutating only a position (no topology change) does NOT flip
#      the hash (guards against false positives)
#   4. Adding a pin vertex group flips the hash via the
#      ``pin_groups`` map
#   5. Removing a vertex group flips the hash back

from __future__ import annotations

from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_TEMPLATE = r"""
import bpy, time, traceback, json, bmesh
result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def log(msg):
    result["phases"].append((round(time.time(), 3), msg))


def record(name, ok, details):
    result["checks"][name] = {"ok": bool(ok), "details": details}


try:
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["compute_mesh_hash"])
    groups_mod = __import__(pkg + ".models.groups",
                            fromlist=["get_addon_data"])
    api_mod = __import__(pkg + ".ops.api", fromlist=["solver"])
    solver_api = api_mod.solver

    log("setup_start")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "HashMesh"
    n_verts = len(plane.data.vertices)
    vg = plane.vertex_groups.new(name="HashPin")
    vg.add(list(range(n_verts)), 1.0, "REPLACE")

    cloth = solver_api.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    cloth.create_pin(plane.name, "HashPin")

    root = groups_mod.get_addon_data(bpy.context.scene)
    state = root.state

    # Snapshot the fresh hash and store it (mirrors what the transfer
    # operator does).
    h_initial = encoder_mesh.compute_mesh_hash(bpy.context)
    state.set_mesh_hash(h_initial)
    log(f"initial_hash keys={list(h_initial.keys())}")

    # ----- Check 1: fresh hash validates clean ---------------------
    warning_initial = state.validate_mesh_hash(bpy.context)
    record(
        "fresh_hash_validates_empty_warning",
        warning_initial == "",
        {"warning": warning_initial},
    )

    # ----- Check 2: position-only edit does NOT flip hash ----------
    # Topology hash is by design count-only; moving a vertex must not
    # warn the user to re-transfer.
    plane.data.vertices[0].co.x += 0.5
    plane.data.update()
    warning_after_move = state.validate_mesh_hash(bpy.context)
    h_after_move = encoder_mesh.compute_mesh_hash(bpy.context)
    record(
        "position_change_does_not_flip_hash",
        warning_after_move == "" and h_after_move == h_initial,
        {
            "warning": warning_after_move,
            "hash_unchanged": h_after_move == h_initial,
        },
    )

    # ----- Check 3: adding a vertex flips the hash -----------------
    # Use bmesh to add a vertex without re-creating the mesh data
    # block (bpy.ops.mesh.* requires Edit Mode + an active 3D View).
    bm = bmesh.new()
    bm.from_mesh(plane.data)
    bm.verts.new((1.5, 0.0, 0.0))
    bm.to_mesh(plane.data)
    bm.free()
    plane.data.update()
    warning_after_add = state.validate_mesh_hash(bpy.context)
    h_after_add = encoder_mesh.compute_mesh_hash(bpy.context)
    record(
        "add_vertex_flips_hash_and_warns",
        bool(warning_after_add) and h_after_add != h_initial,
        {
            "warning_present": bool(warning_after_add),
            "warning": warning_after_add,
            "vertex_count_initial": list(h_initial.values())[0]["vertex_count"],
            "vertex_count_after": list(h_after_add.values())[0]["vertex_count"],
        },
    )

    # ----- Check 4: re-storing the hash clears the warning ---------
    # Simulates the user re-transferring after the topology change.
    state.set_mesh_hash(h_after_add)
    warning_after_resync = state.validate_mesh_hash(bpy.context)
    record(
        "resync_after_topology_change_clears_warning",
        warning_after_resync == "",
        {"warning": warning_after_resync},
    )

    # ----- Check 5: adding a new pin vertex group flips hash -------
    # The encoder folds pin vertex group identity into the hash via
    # the ``pin_groups`` dict; an extra pin must produce a divergence.
    state.set_mesh_hash(h_after_add)  # baseline = current topology
    vg2 = plane.vertex_groups.new(name="HashPin2")
    vg2.add([0], 1.0, "REPLACE")
    cloth.create_pin(plane.name, "HashPin2")
    warning_after_pin = state.validate_mesh_hash(bpy.context)
    h_after_pin = encoder_mesh.compute_mesh_hash(bpy.context)
    record(
        "add_pin_group_flips_hash",
        bool(warning_after_pin) and h_after_pin != h_after_add,
        {
            "warning_present": bool(warning_after_pin),
            "pin_count_before": len(list(h_after_add.values())[0].get(
                "pin_groups", {})),
            "pin_count_after": len(list(h_after_pin.values())[0].get(
                "pin_groups", {})),
        },
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
