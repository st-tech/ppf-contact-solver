# File: scenarios/bl_bend_aniso_reference.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Gates the two halves of shell bending working TOGETHER: the rest angle a
# bending reference supplies, and the directional stiffness `bend-warp` /
# `bend-weft` apply to it.
#
# They are separate mechanisms and it is easy for one to quietly ignore the
# other. The rest angle says WHERE the bending energy is at its minimum;
# builder.rs computes it per hinge from the reference shape and ships only that
# angle. The directional stiffness says HOW STIFF the well is, from the hinge
# edge's direction in the UV material frame. The CUDA kernel composes them by
# construction (`face_compute_force_hessian(..., prop.rest_angle, ...)` scaled
# by stiff_k), but the emulator reconstructs its own rest configuration, so
# nothing forces the two backends to agree unless it is checked.
#
# Four strips, identical in geometry, pins and material except where noted:
#
#   RefIso     curved reference, directional stiffness OFF (both 0)
#   FlatIso    NO reference (flat rest), directional stiffness OFF
#   RefAniso   curved reference, directional stiffness ON
#   FlatAniso  NO reference (flat rest), same directional stiffness
#
# Each is a cantilever pinned along its +y edge. The reference is a
# positions-only copy curled in +z by x^2, so its hinges carry a non-zero rest
# angle and the sheet is driven toward that curl rather than toward flat.
#
# Checks:
#   A. all_pc2_present        - every strip produced every output frame.
#   B. reference_honored_isotropic - RefIso settles differently from FlatIso.
#                               Same material, same pins, both with the
#                               directional stiffness OFF; the ONLY difference
#                               is the curved rest angle.
#   C. reference_honored_directional - RefAniso settles differently from
#                               FlatAniso, the same comparison with the
#                               directional stiffness ON. Stiffness is held
#                               equal WITHIN each pair, so neither check can be
#                               satisfied by one path simply being stiffer.
#   D. simulation_stable      - all positions finite and bounded.
#
# B and C are the point together: the rest angle and the directional stiffness
# are separate mechanisms, and a backend can honor one while ignoring the other.
# The emulator did exactly that until it reconstructed its rest configuration
# from HingeProp::rest_angle, since the reference VERTICES never reach a
# backend, only the per-hinge angle builder.rs derives from them.
#
# How strongly the direction itself bites is bl_bend_anisotropy_uv's job, not
# this one: with a curved reference the drape measure mixes falling with curling
# toward the reference, so it does not isolate direction cleanly.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

KNOBS = {"PPF_EMULATED_ELASTIC": "1", "PPF_EMULATED_STEP_MS": "0"}


_DRIVER_BODY = r"""
import os
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = 24
SUBDIV = 10

# The isotropic base is what the soft fold direction gets, so it sits where
# bending competes with gravity; the warp value then puts the stiff direction
# well clear of it. Same reasoning as bl_bend_anisotropy_uv.
BEND_STIFFNESS = 8.0
WARP_STIFFNESS = 150.0
WEFT_STIFFNESS = 0.0
# How hard the reference curls, as z = CURL * x^2 over a [-1, 1] sheet.
CURL = 0.35


# A grid pinned along its +y edge with an explicit UV map. `rotate_uv` turns the
# layout 90 degrees, leaving it congruent so only the material direction of each
# hinge edge changes.
def _make_strip(name, x_offset, rotate_uv):
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=SUBDIV, y_subdivisions=SUBDIV, size=2,
        location=(x_offset, 0, 0),
    )
    obj = bpy.context.object
    obj.name = name
    mesh = obj.data
    uv = mesh.uv_layers.active
    if uv is None:
        uv = mesh.uv_layers.new(name="UVMap")
    mesh.uv_layers.active = uv
    for loop in mesh.loops:
        co = mesh.vertices[loop.vertex_index].co
        u, v = (co.y, -co.x) if rotate_uv else (co.x, co.y)
        uv.data[loop.index].uv = (0.5 * (u + 1.0), 0.5 * (v + 1.0))
    mesh.update()

    pinned = [i for i, vv in enumerate(mesh.vertices) if vv.co.y > 0.99]
    free = [i for i, vv in enumerate(mesh.vertices) if vv.co.y <= 0.99]
    vg = obj.vertex_groups.new(name="TopEdge")
    vg.add(pinned, 1.0, "REPLACE")
    return obj, pinned, free


# A positions-only copy of `src`, curled in +z by x^2. Same topology, which is
# what validate_bend_reference requires.
def _make_reference(src, name, x_offset):
    ref = src.copy()
    ref.data = src.data.copy()
    ref.name = name
    bpy.context.collection.objects.link(ref)
    ref.location = (x_offset, 0.0, 0.0)
    for vv in ref.data.vertices:
        vv.co.z = CURL * vv.co.x * vv.co.x
    ref.data.update()
    return ref


def _raw_group(dh, name):
    for g in dh.groups.iterate_active_object_groups(bpy.context.scene):
        if g.name == name:
            return g
    return None


def _mean_droop(arr, idx):
    sel = np.array(idx, dtype=int)
    return float(-np.mean((arr[-1][sel] - arr[0][sel])[:, 2]))


try:
    dh = DriverHelpers(pkg, result)
    uuidreg = __import__(pkg + ".core.uuid_registry",
                         fromlist=["get_or_create_object_uuid"])
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # name, x, rotate_uv, group, wants_reference. The UV layout is the SAME
    # for all four: this scenario varies only the rest angle and the group's
    # stiffness, so orientation is deliberately held fixed.
    layout = [
        ("RefIso", -9.0, False, "Iso", True),
        ("FlatIso", -3.0, False, "Iso", False),
        ("RefAniso", 3.0, False, "Aniso", True),
        ("FlatAniso", 9.0, False, "Aniso", False),
    ]
    strips = {}
    for name, x, rot, _grp, wants_ref in layout:
        obj, pinned, free = _make_strip(name, x, rot)
        info = {"obj": obj, "pinned": pinned, "free": free, "ref": None}
        if wants_ref:
            info["ref"] = _make_reference(obj, name + "_Reference", x)
        strips[name] = info
        dh.log(f"strip {name} rot_uv={rot} reference={wants_ref}")

    # The encoder reads the depsgraph-evaluated mesh, which still holds the
    # grid's shipped UV until the graph is flushed.
    bpy.context.view_layer.update()

    dh.save_blend(PROBE_DIR, "bend_aniso_reference.blend")
    root = dh.configure_state(
        project_name="bend_aniso_reference",
        frame_count=FRAME_COUNT,
        gravity=(0.0, 0.0, -9.8),
    )

    for group_name in ("Aniso", "Iso"):
        api = dh.api.solver.create_group(group_name, "SHELL")
        members = [e for e in layout if e[3] == group_name]
        for name, _x, _rot, _g, _wr in members:
            api.add(name)
            api.create_pin(name, "TopEdge")
        raw = _raw_group(dh, group_name)
        if raw is None:
            raise RuntimeError(f"group {group_name} not found")
        raw.bend = BEND_STIFFNESS
        if group_name == "Aniso":
            raw.bend_warp = WARP_STIFFNESS
            raw.bend_weft = WEFT_STIFFNESS
        else:
            raw.bend_warp = 0.0
            raw.bend_weft = 0.0
        # Any member with a reference switches the group's reference mode on;
        # the per-object bend_ref_enable is what selects which ones use it.
        if any(wr for _n, _x, _r, _g, wr in members):
            raw.bend_rest_from_reference = True
        for name, _x, _rot, _g, wants_ref in members:
            if not wants_ref:
                continue
            uid = uuidreg.get_or_create_object_uuid(strips[name]["obj"])
            assigned = next(
                (a for a in raw.assigned_objects if a.uuid == uid), None)
            if assigned is None:
                raise RuntimeError(f"assigned entry for {name} not found")
            assigned.bend_ref_enable = True
            assigned.bend_ref_uuid = uuidreg.get_or_create_object_uuid(
                strips[name]["ref"])
        dh.log(
            f"group {group_name}: bend={raw.bend} warp={raw.bend_warp} "
            f"weft={raw.bend_weft} from_reference={raw.bend_rest_from_reference}"
        )

    encoder_pkg = __import__(pkg + ".core.encoder", fromlist=["prepare_upload"])
    data_bytes, param_bytes, _dh_, _ph_ = encoder_pkg.prepare_upload(bpy.context)

    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.build_and_wait(data_bytes, param_bytes, "aniso-ref:build", timeout=180.0)
    saw_running = dh.run_and_wait(timeout=180.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT, timeout=90.0)
    applied, total = dh.fetch_and_drain()
    dh.log(f"ran saw_running={saw_running} fetch={applied}/{total}")

    frames_ok = True
    for name, info in strips.items():
        p = dh.find_pc2_for(info["obj"])
        if not p or not os.path.isfile(p):
            raise RuntimeError(f"no PC2 for {name} (path={p!r})")
        info["arr"] = dh.read_pc2(p)
        if info["arr"].shape[0] != FRAME_COUNT:
            frames_ok = False

    dh.record("A_all_pc2_present", frames_ok,
              {n: int(i["arr"].shape[0]) for n, i in strips.items()})

    droop = {n: _mean_droop(i["arr"], i["free"]) for n, i in strips.items()}
    dh.log(f"droop={ {k: round(v, 5) for k, v in droop.items()} }")

    def relgap(a, b):
        return abs(a - b) / max(abs(a), abs(b), 1e-9)

    # B: the reference changes the outcome with the directional stiffness OFF.
    b_gap = relgap(droop["RefIso"], droop["FlatIso"])
    dh.record(
        "B_reference_honored_isotropic", b_gap > 0.02,
        {"with_reference": round(droop["RefIso"], 5),
         "flat_rest": round(droop["FlatIso"], 5),
         "relative_gap": round(b_gap, 5), "threshold": 0.02},
    )

    # C: and with it ON. Stiffness is identical within this pair, so a build
    # that dropped the rest angle once the directional term was in play would
    # collapse the two together.
    c_gap = relgap(droop["RefAniso"], droop["FlatAniso"])
    dh.record(
        "C_reference_honored_directional", c_gap > 0.02,
        {"with_reference": round(droop["RefAniso"], 5),
         "flat_rest": round(droop["FlatAniso"], 5),
         "relative_gap": round(c_gap, 5), "threshold": 0.02},
    )

    allv = np.concatenate([i["arr"].reshape(-1, 3) for i in strips.values()])
    finite = bool(np.all(np.isfinite(allv)))
    max_abs = float(np.max(np.abs(allv)))
    dh.record("D_simulation_stable", finite and max_abs < 100.0,
              {"finite": finite, "max_abs": round(max_abs, 4)})

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
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 420.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
