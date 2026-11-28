# File: scenarios/bl_fetch_parent_offset.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# Public issue #167: Fetch All Animation does not respect a parent's offset.
#
# A simulated cloth is parented to an Empty. After a run, the artist moves the
# Empty, clears the local animation and fetches all frames again. The cloth
# should follow its parent, as it does when the Empty is moved while the
# fetched animation is still on the object. The report says it lands back where
# it was simulated instead.
#
# Sequence:
#   1. An Empty at (1, 0, 0); a fully pinned plane at the origin parented to it
#      the way Ctrl+P > Object does (keep transform). The pin moves the plane
#      +0.2 in X over frames 1..4.
#   2. Transfer through SOLVER_OT_Transfer's own stages, whose Upload stage
#      records the world matrix each object was sent with; run; Fetch All.
#      Read the plane's evaluated WORLD-space centroid at the last fetched
#      frame: A.
#   3. Move the Empty by D. Read the centroid again: B. The fetched cache is in
#      object-local space, so B = A + D.
#   4. Clear Local Animation, Fetch All again with the Empty still moved. Read
#      the centroid: C. C = A + D = B: fetching after the move gives what
#      moving after the fetch gives. The issue reported C = A.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

BACKENDS = ("real",)


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
TOL = 1.0e-4
PARENT_AT = (1.0, 0.0, 0.0)
MOVE_BY = (0.0, 0.5, 1.5)


def world_centroid(obj, frame):
    scene = bpy.context.scene
    scene.frame_set(frame)
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = ev.to_mesh()
    co = np.empty(len(me.vertices) * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    ev.to_mesh_clear()
    m = np.array(ev.matrix_world, dtype=np.float64)
    co = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
    return co.mean(axis=0)


def close(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) <= TOL


def vec(v):
    return [round(float(x), 6) for x in v]


try:
    dh = DriverHelpers(pkg, result)
    animation_mod = __import__(pkg + ".core.animation",
                               fromlist=["clear_animation_data"])
    encoder_mod = __import__(pkg + ".core.encoder",
                             fromlist=["resolve_start_frame"])

    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="ParentedCloth")
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=PARENT_AT)
    empty = bpy.context.active_object
    empty.name = "ClothParent"
    bpy.context.view_layer.update()
    # Ctrl+P > Object: the child keeps its world transform.
    plane.parent = empty
    plane.matrix_parent_inverse = empty.matrix_world.inverted()
    bpy.context.view_layer.update()
    dh.save_blend(PROBE_DIR, "fetchparent.blend")
    root = dh.configure_state(project_name="fetch_parent_offset",
                              frame_count=6)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.2, 0.0, 0.0), frame_start=1, frame_end=4,
                transition="LINEAR")

    start = encoder_mod.resolve_start_frame(root.state)
    rest_centroid = world_centroid(plane, start)

    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root.state.project_name)
    # Transfer through the operator's own stages, not encode + build_pipeline:
    # the Upload stage is where the transferred world matrices are recorded.
    deadline = time.time() + 10.0
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        if (dh.com.info.response.get("data") == "NO_DATA"
                and dh.facade.engine.state.remote_root):
            break
        time.sleep(0.1)
    Transfer = __import__(pkg + ".ui.solver",
                          fromlist=["SOLVER_OT_Transfer"]).SOLVER_OT_Transfer
    stub = dh.staged_stub(Transfer)
    verdict = Transfer.execute(stub, bpy.context)
    drained = stub.drain_stages(bpy.context)
    deadline = time.time() + 90.0
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if (s.activity.name == "IDLE"
                and s.solver.name in ("READY", "RESUMABLE", "FAILED")):
            break
        time.sleep(0.2)
    uuid_mod = __import__(pkg + ".core.uuid_registry",
                          fromlist=["get_object_uuid"])
    sent = root.state.get_transfer_world()
    plane_uuid = uuid_mod.get_object_uuid(plane)
    dh.record(
        "transfer_recorded_world_matrix",
        verdict == {"RUNNING_MODAL"} and drained is None
        and dh.facade.engine.state.solver.name in ("READY", "RESUMABLE")
        and plane_uuid in sent,
        {"verdict": list(verdict), "drained": drained,
         "solver": dh.facade.engine.state.solver.name,
         "recorded": sorted(sent), "plane_uuid": plane_uuid,
         "reports": stub.captured[:3]},
    )
    dh.run_and_wait(timeout=60.0)
    dh.force_frame_query(expected_frames=1, timeout=10.0)
    dh.settle_idle(timeout=15.0)

    applied, total = dh.fetch_and_drain()
    dh.log(f"fetch.drained applied={applied}/{total}")
    last_n = max(int(f.value) for f in root.state.fetched_frame)
    frame = last_n + start
    a = world_centroid(plane, frame)
    dh.record(
        "first_fetch_follows_pin",
        dh.has_mesh_cache(plane)
        and close(a, rest_centroid + np.array((0.2, 0.0, 0.0))),
        {"rest_centroid": vec(rest_centroid), "A": vec(a),
         "frame": frame, "applied": applied, "total": total},
    )

    empty.location = tuple(p + d for p, d in zip(PARENT_AT, MOVE_BY))
    bpy.context.view_layer.update()
    b = world_centroid(plane, frame)
    dh.record(
        "moving_parent_moves_fetched_cloth",
        close(b, a + np.array(MOVE_BY)),
        {"A": vec(a), "B": vec(b), "expected": vec(a + np.array(MOVE_BY))},
    )

    animation_mod.clear_animation_data(bpy.context)
    dh.log("clear.done")
    applied2, total2 = dh.fetch_and_drain()
    dh.log(f"refetch.drained applied={applied2}/{total2}")
    c = world_centroid(plane, frame)
    dh.record(
        "refetch_respects_parent_offset",
        dh.has_mesh_cache(plane) and close(c, a + np.array(MOVE_BY)),
        {"A": vec(a), "C": vec(c), "expected": vec(a + np.array(MOVE_BY)),
         "C_minus_A": vec(c - a), "applied": applied2, "total": total2},
    )
    dh.record(
        "fetch_order_does_not_matter",
        close(c, b),
        {"B_moved_after_fetch": vec(b), "C_fetched_after_move": vec(c)},
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
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
