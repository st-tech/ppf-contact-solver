# File: scenarios/bl_allow_rest_proximity.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# The ADD-ON half of Allow Rest Proximity: the group's distance field, the
# encoder that ships it scaled by World Scale, and a real build and run.
#
# `rig_rest_proximity` covers what the option MEANS through the frontend and
# never loads Blender, so it cannot see whether the field reaches anything.
# This scenario drives real Blender through real builds and a real run.
#
# THE SCENE is two SHELL groups: a grid pinned in place, which makes it a
# STATIC collision mesh, and a smaller grid 4 mm above it, parallel. Nothing overlaps and both contact offsets are zero, so the
# scene builds whatever the field says; the field alone decides whether the
# pairs between the two sheets keep contact.
#
# THE WITNESSES are the encoded param payload (the group's
# `allow-rest-proximity` value), `bin/start_link.bin`, which the frontend
# writes only when the build linked something, and the fetched animation: with
# gravity and contact on, a sheet whose pairs with the held sheet are linked
# falls THROUGH it, and one whose pairs are not comes to rest on it.
#
# Subtests:
#   A_rna_registered_with_default  the checkbox (off) and its distance
#                                  (0.001) exist. New RNA needs a full Blender
#                                  restart.
#   B_encoded_scaled_by_world      checked, the payload carries the distance
#                                  times World Scale on the garment's group and
#                                  0 on the held group's; unchecked, 0 with the
#                                  distance still set.
#   C_profile_round_trips          a material profile carries both.
#   D_unset_build_links_nothing    unchecked, the build succeeds and
#                                  writes no start_link.bin (the control).
#   E_unset_run_rests_on_held      and the sheet comes to rest on the held one.
#   F_set_build_writes_links       checked at 0.01 on the garment (mean 0.005 over
#                                  the 4 mm spacing) the build writes a
#                                  non-empty, whole-pair start_link.bin.
#   G_exemptions_reach_the_addon   the server's `exemptions` record arrives
#                                  in the add-on's state, counted.
#   H_set_run_falls_through        the run completes and the sheet falls
#                                  through the held one.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
BACKENDS = ("real",)
NOT_PARALLELIZABLE = True


_DRIVER_BODY = r"""
import glob
import math
import os
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT_NAME = "<<PROJECT_NAME>>"
PROJECT_ROOT = "<<PROJECT_ROOT>>"
FRAMES = 20
PIN_GROUP = "All"
GAP = 0.004
DISTANCE = 0.01


def session_dir():
    hits = [os.path.dirname(p) for p in glob.glob(
        os.path.join(PROJECT_ROOT, "**", "session", "info.toml"),
        recursive=True)]
    if not hits:
        raise RuntimeError("no session/info.toml under %s" % PROJECT_ROOT)
    hits.sort(key=os.path.getmtime, reverse=True)
    return hits[0]


def grid(name, size, subdiv, z):
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=subdiv,
                                    y_subdivisions=subdiv, size=size,
                                    location=(0.0, 0.0, z))
    obj = bpy.context.object
    obj.name = name
    bpy.context.view_layer.update()
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def start_links():
    path = os.path.join(session_dir(), "bin", "start_link.bin")
    if not os.path.isfile(path):
        return None
    return np.fromfile(path, dtype=np.uint32)


def group_params(param_bytes, object_name):
    # The payload's groups are (params, object names, object uuids).
    for params, objects, _uuids in dh.decode_addon_blob(param_bytes)["group"]:
        if object_name in objects:
            return params
    return None


def build_and_run(label):
    # (built, error, links, final mean z of the garment, frame reached)
    data_bytes, param_bytes = dh.encode_payload()
    try:
        dh.build_and_wait(data_bytes, param_bytes,
                          message="rest-prox:" + label, timeout=240.0)
    except RuntimeError as error:
        return False, str(error), None, None, None, []
    links = start_links()
    exemptions = list(dh.facade.engine.state.exemptions)
    dh.run_and_wait(timeout=600.0)
    dh.force_frame_query(expected_frames=FRAMES, timeout=60.0)
    state = dh.facade.engine.state
    frame = int(state.frame) if state.solver.name != "FAILED" else -1
    dh.fetch_and_drain()
    pc2 = dh.read_pc2(dh.find_pc2_for(cloth))
    z = float(np.asarray(pc2[-1])[:, 2].mean())
    dh.settle_idle(timeout=30.0)
    return True, "", links, z, frame, exemptions


try:
    dh = DriverHelpers(pkg, result)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = FRAMES

    held = grid("HeldSheet", 1.0, 10, 0.0)
    held.vertex_groups.new(name=PIN_GROUP).add(
        list(range(len(held.data.vertices))), 1.0, "REPLACE")
    cloth = grid("NearSheet", 0.5, 6, GAP)

    dh.save_blend(PROBE_DIR, "allow_rest_proximity.blend")
    root = dh.configure_state(project_name=PROJECT_NAME, frame_count=FRAMES,
                              gravity=(0.0, 0.0, -9.8))
    root.state.disable_contact = False
    groups_mod = __import__(pkg + ".models.groups",
                            fromlist=["get_group_slot_index"])
    addon_root = dh.groups.get_addon_data(scene)

    held_group = dh.api.solver.create_group("Held", "SHELL")
    held_group.add("HeldSheet")
    held_group.create_pin("HeldSheet", PIN_GROUP)
    garment = dh.api.solver.create_group("Felt", "SHELL")
    garment.add("NearSheet")

    def rna_of(group):
        return getattr(addon_root, "object_group_%d"
                       % groups_mod.get_group_slot_index(scene, group.uuid))

    rna = rna_of(garment)
    held_rna = rna_of(held_group)
    # FRICTIONLESS, so the runs exercise the contact the field decides about
    # and nothing else: what is measured is where the sheet comes to rest, and
    # friction would only add the friction memory's own per-step bookkeeping.
    for g in (rna, held_rna):
        g.friction = 0.0
        g.contact_gap = 0.001
        g.contact_offset = 0.0
        g.use_group_bounding_box_diagonal = False

    # A: a checkbox, off, and its distance, 0.001.
    box = rna.bl_rna.properties.get("allow_rest_proximity")
    dist = rna.bl_rna.properties.get("rest_proximity_distance")
    dh.record(
        "A_rna_registered_with_default",
        box is not None and box.type == "BOOLEAN"
        and dist is not None and dist.type == "FLOAT"
        and rna.allow_rest_proximity is False
        and abs(rna.rest_proximity_distance - 0.001) < 1e-9,
        {"box": None if box is None else box.type,
         "distance": None if dist is None else dist.type,
         "values": None if box is None or dist is None else
         [rna.allow_rest_proximity, rna.rest_proximity_distance],
         "note": "new RNA needs a full Blender restart, not a reload"},
    )

    # B: checked, the distance times World Scale on the garment's group only;
    # unchecked, 0 with the same distance set.
    def encoded(scale):
        saved_scale = root.state.world_scaling
        root.state.world_scaling = scale
        _data, param_bytes = dh.encode_payload()
        root.state.world_scaling = saved_scale
        got = group_params(param_bytes, "NearSheet") or {}
        got_held = group_params(param_bytes, "HeldSheet") or {}
        return got.get("allow-rest-proximity"), got_held.get("allow-rest-proximity")

    rna.rest_proximity_distance = DISTANCE
    rna.allow_rest_proximity = True
    value, value_held = encoded(2.0)
    rna.allow_rest_proximity = False
    value_off, _ = encoded(2.0)
    dh.record(
        "B_encoded_scaled_by_world",
        value is not None and abs(float(value) - 2.0 * DISTANCE) < 1e-6
        and value_held is not None and float(value_held) == 0.0
        and value_off is not None and float(value_off) == 0.0,
        {"garment": None if value is None else float(value),
         "held": None if value_held is None else float(value_held),
         "garment_unchecked": None if value_off is None else float(value_off)},
    )

    # C: a material profile carries both ways.
    profile_mod = __import__(pkg + ".core.profile",
                             fromlist=["read_material_profile"])
    rna.allow_rest_proximity = True
    entry = profile_mod.read_material_profile(rna)
    rna.allow_rest_proximity = False
    profile_mod.apply_material_profile(
        {"allow_rest_proximity": True, "rest_proximity_distance": 0.003},
        held_rna)
    applied = (held_rna.allow_rest_proximity, held_rna.rest_proximity_distance)
    held_rna.allow_rest_proximity = False
    dh.record(
        "C_profile_round_trips",
        entry.get("allow_rest_proximity") is True
        and abs(float(entry.get("rest_proximity_distance", -1.0)) - DISTANCE) < 1e-6
        and applied[0] is True and abs(applied[1] - 0.003) < 1e-6,
        {"read": [entry.get("allow_rest_proximity"),
                  entry.get("rest_proximity_distance")],
         "applied": list(applied)},
    )

    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root.state.project_name)

    # D, E: the control, the box unchecked with the distance still set. Run
    # first, so no earlier build's start_link.bin can be standing in the
    # session directory.
    rna.allow_rest_proximity = False
    built, text, links, z, frame, _ex = build_and_run("unset")
    dh.record("D_unset_build_links_nothing", built and links is None,
              {"error": text[-600:],
               "start_link_bin": None if links is None else int(links.size)})
    dh.record("E_unset_run_rests_on_held",
              built and frame >= FRAMES - 1 and z is not None and z > -0.01,
              {"frame": frame, "final_mean_z": z})

    # F, G, H: checked, the distance covering the 4 mm spacing.
    rna.allow_rest_proximity = True
    built, text, links, z, frame, exemptions = build_and_run("set")
    dh.record(
        "F_set_build_writes_links",
        built and links is not None and links.size > 0 and links.size % 2 == 0,
        {"error": text[-600:],
         "start_link_bin": None if links is None else int(links.size)},
    )
    first = exemptions[0] if exemptions else {}
    dh.record(
        "G_exemptions_reach_the_addon",
        len(exemptions) == 1 and int(first.get("count", 0)) > 0,
        {"records": len(exemptions), "type": first.get("type"),
         "count": first.get("count")},
    )
    dh.record("H_set_run_falls_through",
              built and frame >= FRAMES - 1 and z is not None and z < -0.05,
              {"frame": frame, "final_mean_z": z})

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
        .replace("<<PROJECT_NAME>>", ctx.project_name)
        .replace("<<PROJECT_ROOT>>", ctx.project_root.replace("\\", "/"))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 900.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
