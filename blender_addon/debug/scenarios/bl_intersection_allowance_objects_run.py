# File: scenarios/bl_intersection_allowance_objects_run.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The object subset of an intersection allowance, end to end: authored in
# Blender, built, and SIMULATED.
#
# `bl_intersection_allowance_objects` proves the subset reaches the right
# per-vertex policy byte by reading the session the build wrote. That is a
# statement about a file. This one is a statement about the SCENE: the same
# group, the same checkbox, differing only in WHICH object the allowance
# lists, either refuses to build or builds and runs to completion.
#
# Two objects in one SHELL group:
#
#   Tangled  two triangles sharing a vertex, one piercing the other, so the
#            object is self-intersecting at rest. Its VERTICAL triangle is
#            pinned and its horizontal one is left free, which is the shape
#            issue #138 is about: a prescribed side against a free one. The
#            pin carries a MOVE_BY op so the prescribed side moves rather
#            than sitting still. Pinning EVERY vertex would be worse than
#            useless: a triangle whose three vertices are all pinned is a
#            collider for the scene-build check, which skips a pair of
#            colliders, so the object would report no self-intersection at
#            all and both phases below would build.
#   Drape    an ordinary grid pinned along one edge, well away from Tangled.
#            It is what makes the run a simulation rather than a formality:
#            it falls under gravity for the whole clip.
#
# The group has "Allow Self-Intersections" ON and "Apply to All Objects" OFF
# in BOTH phases. Only the list changes:
#
#   phase 1  the list names Drape, the object that is not tangled  -> the
#            build must be REFUSED, naming the self-intersection.
#   phase 2  the list names Tangled                                -> the
#            build succeeds and the run reaches the last frame.
#
# The first phase is what makes the second mean something. A checkbox that
# reached the whole group regardless of the list would build in BOTH phases,
# so a scenario that ran only the second would pass against an add-on that
# ignored the subset entirely.
#
# Subtests:
#   A. tangled_geometry_authored           - the piercing edge really crosses
#                                            the other triangle's plane inside
#                                            it, checked before any transfer.
#   B. wrong_object_listed_refuses_build   - solver FAILED, and the surfaced
#                                            error names the self-intersection
#                                            so a user could act on it.
#   C. right_object_listed_builds          - the same scene, with the list
#                                            changed and nothing else, reaches
#                                            solver READY.
#   D. simulation_runs_to_the_last_frame   - Run reaches the requested frame
#                                            count, so the tolerated scene is
#                                            one the solver actually steps
#                                            rather than one it merely accepts.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. It drives a real build and a real solve; nothing in it is
# backend-specific.
BACKENDS = ("real",)

# Solver frames the run must reach. The addon counts Blender frames and the
# solver counts from 0, so a 5-frame clip is 4 solver frames.
FRAME_COUNT = 5
SOLVER_FRAMES = FRAME_COUNT - 1


_DRIVER_BODY = r"""
import time
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT_NAME = "<<PROJECT_NAME>>"
FRAME_COUNT = <<FRAME_COUNT>>
SOLVER_FRAMES = <<SOLVER_FRAMES>>


def make_tangled(name):
    # Two triangles sharing one vertex, so the mesh is connected: the encoder
    # drops disconnected components on some shell paths, and a dropped half
    # would leave a mesh that is not self-intersecting at all, which reads as
    # the allowance working when nothing was ever tangled.
    #
    # Triangle A lies in z=0. Triangle B is vertical and shares v2; its edge
    # (v3 -> v4) runs along x=0.5, y=0.5 from z=-1 to z=1, so it crosses z=0
    # at (0.5, 0.5, 0), which is inside triangle A.
    import bmesh
    mesh = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bm = bmesh.new()
    coords = [
        (0.0, 0.0, 0.0),   # v0
        (2.0, 0.0, 0.0),   # v1
        (0.0, 2.0, 0.0),   # v2  shared
        (0.5, 0.5, -1.0),  # v3
        (0.5, 0.5, 1.0),   # v4  edge (v3, v4) pierces triangle A
    ]
    verts = [bm.verts.new(c) for c in coords]
    bm.verts.ensure_lookup_table()
    bm.faces.new((verts[0], verts[1], verts[2]))
    bm.faces.new((verts[3], verts[4], verts[2]))
    bm.to_mesh(mesh)
    bm.free()
    # Pin the VERTICAL triangle only (v2, v3, v4), which is three of the five
    # vertices. Not all five, and this is the trap: the scene-build check
    # marks a triangle a COLLIDER when all three of its vertices are pinned
    # (`scene_build/assembly.rs`), and it skips a pair whose BOTH sides are
    # colliders, so an object with every vertex pinned reports no
    # self-intersection at all and both phases below would build. Leaving
    # triangle A's v0 and v1 free keeps one side of the pair dynamic, which is
    # what makes the pair reportable and the allowance the thing that decides
    # it. The move op added on the group's pin is what keeps the object a
    # pinned DYNAMIC object rather than a rest-pose STATIC collider.
    obj.vertex_groups.new(name="Pin").add([2, 3, 4], 1.0, "REPLACE")
    return obj


def surfaced_error(dh, pkg):
    # Everything the add-on showed the user about the last failure: the
    # engine's own two fields plus the console the panel prints into.
    state = dh.facade.engine.state
    console_mod = __import__(pkg + ".models.console", fromlist=["console"])
    msgs = [getattr(m, "text", str(m))
            for m in getattr(console_mod.console, "messages", [])]
    return "\n".join(
        [state.error or "", getattr(state, "server_error", "") or ""] + msgs)


def build_once(dh, pkg, message, timeout=90.0):
    # One build, driven to a terminal solver state, and the state it reached.
    #
    # Distinct from `build_and_wait`, which raises on a failed build: a
    # refusal is the expected outcome of one of the two phases, so this
    # returns it.
    #
    # THE HARD PART IS KNOWING WHICH BUILD A TERMINAL STATE BELONGS TO. A
    # second build dispatched while the engine still carries the FIRST one's
    # verdict passes through activity=IDLE with that verdict still on it,
    # both the instant the dispatch returns and again between the upload and
    # the server-side build. A loop that waits for "IDLE and terminal"
    # therefore exits on the PREVIOUS build's answer. Measured here: phase 2
    # was reported FAILED with phase 1's error while its own build had
    # produced no violations at all, which is exactly the verdict a broken
    # allowance would also produce. Waiting for activity to leave IDLE does
    # not fix it (the upload's SENDING satisfies that), and waiting for
    # activity=BUILDING never returns, because BUILDING is a SOLVER state
    # reported by the server, not a client activity.
    #
    # Two things therefore have to hold before a terminal state is this
    # build's: the server must be holding THIS payload
    # (`server_param_hash`), and the verdict must have MOVED off the one the
    # previous build left. Both are persistent state, so neither depends on
    # catching a transient between two polls.
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["compute_data_hash"])
    encoder_params = __import__(pkg + ".core.encoder.params",
                                fromlist=["compute_param_hash"])
    data_bytes, param_bytes = dh.encode_payload()
    param_hash = encoder_params.compute_param_hash(bpy.context)
    before = dh.facade.engine.state.solver.name
    dh.facade.engine.dispatch(dh.events.BuildPipelineRequested(
        data=data_bytes, param=param_bytes,
        data_hash=encoder_mesh.compute_data_hash(bpy.context),
        param_hash=param_hash,
        message=message,
    ))
    deadline = time.time() + timeout
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if (s.activity.name == "IDLE"
                and s.solver.name in ("READY", "RESUMABLE", "FAILED")
                and s.solver.name != before
                and s.server_param_hash == param_hash):
            return s.solver.name, s.activity.name
        time.sleep(0.05)
    s = dh.facade.engine.state
    trail = "\n".join(
        "  %.3f %s %s" % (t, name, rep)
        for t, name, rep in dh.facade.engine.recent_events[-30:])
    raise RuntimeError(
        "build %r did not produce its own verdict within %ss: "
        "solver=%s (was %s) activity=%s server_param_hash=%r wanted %r "
        "error=%r\nrecent events:\n%s"
        % (message, timeout, s.solver.name, before, s.activity.name,
           s.server_param_hash, param_hash, s.error, trail))


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = FRAME_COUNT

    tangled = make_tangled("Tangled")

    # An ordinary sheet, far enough away that it meets nothing, pinned along
    # one edge so it drapes for the whole clip. It is what gives the run
    # something to integrate: Tangled is two triangles, most of it
    # prescribed, and it would make a "simulation" that is nearly a
    # formality on its own.
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=6, y_subdivisions=6,
                                    size=1.0, location=(8.0, 0.0, 0.0))
    drape = bpy.context.object
    drape.name = "Drape"
    ymax = max(v.co.y for v in drape.data.vertices)
    drape.vertex_groups.new(name="Edge").add(
        [i for i, v in enumerate(drape.data.vertices) if v.co.y > ymax - 1e-4],
        1.0, "REPLACE")

    dh.save_blend(PROBE_DIR, "intersection_allowance_run.blend")
    root = dh.configure_state(project_name=PROJECT_NAME,
                              frame_count=FRAME_COUNT)

    groups_mod = __import__(pkg + ".models.groups",
                            fromlist=["get_group_slot_index"])
    addon_root = dh.groups.get_addon_data(scene)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add("Tangled")
    cloth.add("Drape")
    cloth.create_pin("Tangled", "Pin")
    cloth.create_pin("Drape", "Edge")
    slot = groups_mod.get_group_slot_index(scene, cloth.uuid)
    group = getattr(addon_root, "object_group_%d" % slot)

    # The move op on Tangled's pin, which makes the prescribed side MOVE and
    # so keeps the fixture on the shape issue #138 is about rather than on a
    # static tangle. It is also what an object with EVERY vertex pinned would
    # need to stay in the solved namespace at all, since `update_static`
    # promotes a fully pinned object with no operations to a rest-pose STATIC
    # collider; this one is pinned on three of five vertices and stays
    # dynamic either way.
    pin_item = None
    for item in group.pin_vertex_groups:
        if "Pin]" in item.name and "Tangled" in item.name:
            pin_item = item
            break
    if pin_item is None:
        raise RuntimeError("no Tangled pin on the group: %r"
                           % [i.name for i in group.pin_vertex_groups])
    op = pin_item.operations.add()
    op.op_type = "MOVE_BY"
    op.delta = (0.0, 0.0, 0.05)
    op.frame_start = 1
    op.frame_end = FRAME_COUNT
    dh.log("groups_created slot=%d pins=%r"
           % (slot, [i.name for i in group.pin_vertex_groups]))

    # A: the tangle is real, checked on the authored coordinates rather than
    # on the build's verdict, so a refusal below cannot be credited to a mesh
    # that was never crossing.
    pos = [list(v.co) for v in tangled.data.vertices]
    crossing = (
        len(pos) == 5
        and len(tangled.data.polygons) == 2
        # The piercing edge spans z=0 ...
        and pos[3][2] < 0.0 < pos[4][2]
        # ... at a point strictly inside triangle A, whose legs run from the
        # origin along +x and +y to 2.0.
        and 0.0 < pos[3][0] < 2.0 and 0.0 < pos[3][1] < 2.0
        and pos[3][0] + pos[3][1] < 2.0
        and abs(pos[3][0] - pos[4][0]) < 1e-6
        and abs(pos[3][1] - pos[4][1]) < 1e-6
    )
    dh.record(
        "A_tangled_geometry_authored", crossing,
        {"n_verts": len(pos), "n_faces": len(tangled.data.polygons),
         "piercing_edge": [pos[3], pos[4]] if len(pos) == 5 else pos,
         "note": "edge (v3, v4) must cross z=0 inside triangle (v0, v1, v2)"},
    )

    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root.state.project_name)
    dh.log("connected")

    # ---- phase 1: the allowance lists the object that is NOT tangled ----
    group.allow_self_intersection = True
    group.allow_self_intersection_all_objects = False
    bpy.ops.object.select_all(action="DESELECT")
    drape.select_set(True)
    bpy.context.view_layer.objects.active = drape
    bpy.ops.object.add_intersection_allowance_objects(
        group_index=slot, allowance="self")
    listed_wrong = [i.name for i in group.allow_self_intersection_objects]
    dh.log("phase1 listed=%r" % (listed_wrong,))

    solver_name, activity = build_once(dh, pkg, "isect-run:wrong-object")
    wrong_error = surfaced_error(dh, pkg)
    dh.record(
        "B_wrong_object_listed_refuses_build",
        listed_wrong == ["Drape"]
        and solver_name == "FAILED" and activity == "IDLE"
        and "self-intersect" in wrong_error.lower(),
        {"listed": listed_wrong, "solver": solver_name, "activity": activity,
         "error_mentions_self_intersection":
             "self-intersect" in wrong_error.lower(),
         "error_tail": wrong_error[-400:],
         "note": "the allowance is ON and reaches only Drape, so Tangled is "
                 "still reported and the build is refused"},
    )

    # PHASE 2 PROVES NOTHING IF PHASE 1 DID NOT REFUSE. A build that already
    # succeeded with the WRONG object listed succeeded for a reason that has
    # nothing to do with the list, and the second build would then be asked
    # to change a verdict that is already what it wants, which it cannot do:
    # `build_once` would spend its whole budget waiting for a change and fail
    # as a timeout instead of naming the cause. Say the cause here.
    if solver_name != "FAILED":
        for name in ("C_right_object_listed_builds",
                     "D_simulation_runs_to_the_last_frame"):
            dh.record(name, False, {
                "skipped": True,
                "phase1_solver": solver_name,
                "note": "phase 1 built with the allowance narrowed to the "
                        "object that is NOT tangled, so the subset reached "
                        "Tangled anyway and nothing phase 2 could do would "
                        "mean anything",
            })
        raise RuntimeError(
            "phase 1 built with the allowance narrowed to %r, which is not "
            "the tangled object: the subset did not narrow the allowance"
            % (listed_wrong,))

    # ---- phase 2: the same scene, the list changed and nothing else ----
    bpy.ops.object.clear_intersection_allowance_objects(
        group_index=slot, allowance="self")
    bpy.ops.object.select_all(action="DESELECT")
    tangled.select_set(True)
    bpy.context.view_layer.objects.active = tangled
    bpy.ops.object.add_intersection_allowance_objects(
        group_index=slot, allowance="self")
    listed_right = [i.name for i in group.allow_self_intersection_objects]
    dh.log("phase2 listed=%r" % (listed_right,))

    solver_name, activity = build_once(dh, pkg, "isect-run:right-object")
    right_error = surfaced_error(dh, pkg)
    dh.record(
        "C_right_object_listed_builds",
        listed_right == ["Tangled"]
        and solver_name in ("READY", "RESUMABLE") and activity == "IDLE",
        {"listed": listed_right, "solver": solver_name, "activity": activity,
         "error_tail": right_error[-400:] if solver_name == "FAILED" else "",
         "note": "only the list differs from phase 1"},
    )

    # ---- D: it actually runs ----
    saw_running = False
    frame = 0
    run_error = ""
    if solver_name in ("READY", "RESUMABLE"):
        saw_running = dh.run_and_wait(timeout=240.0)
        dh.force_frame_query(expected_frames=SOLVER_FRAMES, timeout=60.0)
        state = dh.facade.engine.state
        frame = int(state.frame)
        if state.solver.name == "FAILED":
            run_error = surfaced_error(dh, pkg)[-400:]
    dh.record(
        "D_simulation_runs_to_the_last_frame",
        saw_running and frame >= SOLVER_FRAMES
        and dh.facade.engine.state.solver.name in ("READY", "RESUMABLE"),
        {"saw_running": saw_running, "frame": frame,
         "expected_frames": SOLVER_FRAMES,
         "solver": dh.facade.engine.state.solver.name,
         "run_error_tail": run_error,
         "note": "a tolerated scene has to be one the solver STEPS, not one "
                 "it merely accepts at build"},
    )

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
        .replace("<<FRAME_COUNT>>", str(FRAME_COUNT))
        .replace("<<SOLVER_FRAMES>>", str(SOLVER_FRAMES))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 600.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
