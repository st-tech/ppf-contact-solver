# File: scenarios/bl_existing_intersection.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The ADD-ON half of Allow Existing Intersections: the group checkbox, its
# "Apply to All Objects" narrowing, and the encoder that ships it.
#
# `rig_existing_intersection_build` and `rig_existing_intersection_run` cover
# what the option MEANS through the frontend and never load Blender, so they
# cannot see whether the checkbox reaches anything. This scenario drives real
# Blender through a real build and a real run, and reads what the build wrote.
#
# THE SCENE is two SHELL groups: a held grid, and a smaller grid tilted 30
# degrees and lifted a little so it crosses the held one along a line. That is
# the tangled start the option exists for, and without the option the build
# refuses it.
#
# THE WITNESS is `bin/start_link.bin`, which the frontend writes only when the
# build linked something: u32 pairs of vertex indices. Its presence after a
# flagged build, and the build succeeding at all, are what separate a wired
# checkbox from a declared one. The refusals are the controls, and they come
# LAST because a refused build leaves the solver in FAILED.
#
# Subtests:
#   A_rna_registered_with_defaults    the checkbox and its switch exist,
#                                     default off and on. New RNA needs a full
#                                     Blender restart.
#   B_flagged_build_succeeds          the tangled scene builds with the
#                                     checkbox on the tilted sheet's group.
#   C_flagged_build_writes_links      and writes a non-empty, whole-pair
#                                     start_link.bin.
#   D_flagged_run_reaches_last_frame  the run completes, which the solver's
#                                     per-step intersection gate requires.
#   G_exemptions_reach_the_addon      the server's `exemptions` arrive in the
#                                     add-on's state after the flagged build,
#                                     typed and counted. Turning them into a
#                                     GPU batch needs a GPU context, which the
#                                     headless Windows rig lacks, so that is
#                                     `bl_existing_intersection_draws`,
#                                     which the Windows leg does not select.
#   I_drawn_on_the_start_frame_only   the overlay's frame rule: every record on
#                                     the frame the simulation starts from,
#                                     none on another frame.
#   H_refused_build_clears_exemptions after the refused build the state holds
#                                     none, so the overlay does not draw a
#                                     previous build's tangle.
#   E_narrowed_to_nothing_is_refused  "Apply to All Objects" off with an empty
#                                     list reaches no object, so the tangle is
#                                     refused again: the narrowing is wired.
#   F_unflagged_build_is_refused      the checkbox off: the build refuses the
#                                     tangle by name.

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
FRAMES = 6
PIN_GROUP = "All"


def session_dir():
    hits = [os.path.dirname(p) for p in glob.glob(
        os.path.join(PROJECT_ROOT, "**", "session", "info.toml"),
        recursive=True)]
    if not hits:
        raise RuntimeError("no session/info.toml under %s" % PROJECT_ROOT)
    hits.sort(key=os.path.getmtime, reverse=True)
    return hits[0]


def grid(name, size, subdiv, location, tilt):
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=subdiv,
                                    y_subdivisions=subdiv, size=size,
                                    location=location)
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = (math.radians(tilt), 0.0, 0.0)
    bpy.context.view_layer.update()
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    return obj


def try_build(label):
    # (succeeded, message). A refused build raises from build_and_wait with
    # the solver's own failure text, which is what the refusal checks read.
    data_bytes, param_bytes = dh.encode_payload()
    try:
        dh.build_and_wait(data_bytes, param_bytes,
                          message="existing-isect:" + label, timeout=240.0)
    except RuntimeError as error:
        return False, str(error)
    return True, ""


try:
    dh = DriverHelpers(pkg, result)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = FRAMES

    held = grid("HeldSheet", 1.0, 10, (0.0, 0.0, 0.0), 0.0)
    held.vertex_groups.new(name=PIN_GROUP).add(
        list(range(len(held.data.vertices))), 1.0, "REPLACE")
    grid("TiltedSheet", 0.5, 6, (0.0, 0.0, 0.013), 30.0)

    dh.save_blend(PROBE_DIR, "existing_intersection.blend")
    root = dh.configure_state(project_name=PROJECT_NAME, frame_count=FRAMES)
    groups_mod = __import__(pkg + ".models.groups",
                            fromlist=["get_group_slot_index"])
    addon_root = dh.groups.get_addon_data(scene)

    held_group = dh.api.solver.create_group("Held", "SHELL")
    held_group.add("HeldSheet")
    held_group.create_pin("HeldSheet", PIN_GROUP)
    garment = dh.api.solver.create_group("Garment", "SHELL")
    garment.add("TiltedSheet")
    rna = getattr(addon_root, "object_group_%d"
                  % groups_mod.get_group_slot_index(scene, garment.uuid))

    props = set(rna.bl_rna.properties.keys())
    missing = [p for p in ("allow_existing_intersection",
                           "allow_existing_intersection_all_objects",
                           "allow_existing_intersection_objects")
               if p not in props]
    dh.record(
        "A_rna_registered_with_defaults",
        not missing and rna.allow_existing_intersection is False
        and rna.allow_existing_intersection_all_objects is True,
        {"missing": missing,
         "note": "new RNA needs a full Blender restart, not a reload"},
    )

    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root.state.project_name)

    rna.allow_existing_intersection = True
    ok, text = try_build("flagged")
    dh.record("B_flagged_build_succeeds", ok, {"error": text[-600:]})

    links = None
    if ok:
        path = os.path.join(session_dir(), "bin", "start_link.bin")
        if os.path.isfile(path):
            links = np.fromfile(path, dtype=np.uint32)
    dh.record(
        "C_flagged_build_writes_links",
        links is not None and links.size > 0 and links.size % 2 == 0,
        {"start_link_bin": None if links is None else int(links.size)},
    )

    # G: the overlay pipeline's input. The state mirrors the server's
    # response, in the shape `bl_existing_intersection_draws` turns into a
    # batch where a GPU context exists.
    exemptions = list(dh.facade.engine.state.exemptions) if ok else []
    first = exemptions[0] if exemptions else {}
    pairs = first.get("pairs", [])
    dh.record(
        "G_exemptions_reach_the_addon",
        len(exemptions) == 1 and first.get("type") == "existing_intersection"
        and int(first.get("count", 0)) > 0 and len(pairs) > 0
        and all(len(pair.get("a", [])) in (2, 3) and len(pair.get("b", [])) in (2, 3)
                for pair in pairs),
        {"records": len(exemptions), "type": first.get("type"),
         "count": first.get("count"), "pairs": len(pairs)},
    )

    # I: the overlay's frame rule, the function its draw handler calls.
    overlay_mod = __import__(pkg + ".ui.dynamics.overlay",
                             fromlist=["exemptions_to_draw"])
    scene.frame_set(1)
    on_start = overlay_mod.exemptions_to_draw(scene, exemptions)
    scene.frame_set(3)
    off_start = overlay_mod.exemptions_to_draw(scene, exemptions)
    scene.frame_set(1)
    dh.record("I_drawn_on_the_start_frame_only",
              len(exemptions) > 0 and on_start == exemptions and off_start == [],
              {"records": len(exemptions), "on_start": len(on_start),
               "off_start": len(off_start)})

    reached = False
    if ok:
        dh.run_and_wait(timeout=600.0)
        dh.force_frame_query(expected_frames=FRAMES, timeout=60.0)
        state = dh.facade.engine.state
        reached = (state.solver.name != "FAILED"
                   and state.frame >= FRAMES)
        dh.record("D_flagged_run_reaches_last_frame", reached,
                  {"frame": state.frame, "solver": state.solver.name,
                   "error": state.error})
    else:
        dh.record("D_flagged_run_reaches_last_frame", False,
                  {"skipped": "the flagged build failed"})
    dh.settle_idle(timeout=30.0)

    # E: narrowed to an empty list, the allowance reaches no object.
    rna.allow_existing_intersection_all_objects = False
    rna.allow_existing_intersection_objects.clear()
    ok, text = try_build("narrowed")
    dh.record("E_narrowed_to_nothing_is_refused",
              not ok and "self-intersection" in text,
              {"built": ok, "error": text[-600:]})
    dh.facade.engine.dispatch(dh.events.PollTick())
    dh.facade.tick()
    left = list(dh.facade.engine.state.exemptions)
    dh.record("H_refused_build_clears_exemptions", not ok and not left,
              {"built": ok, "exemptions_left": len(left)})

    # F: the checkbox off.
    rna.allow_existing_intersection_all_objects = True
    rna.allow_existing_intersection = False
    ok, text = try_build("unflagged")
    dh.record("F_unflagged_build_is_refused",
              not ok and "self-intersection" in text,
              {"built": ok, "error": text[-600:]})

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
