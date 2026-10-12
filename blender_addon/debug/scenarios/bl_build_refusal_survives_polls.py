# File: scenarios/bl_build_refusal_survives_polls.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A refused Transfer stays refused, however often the server is asked.
#
# The server reports a refused build as `NO_BUILD` carrying the error, on
# every status response until an upload replaces the payload, and the add-on
# reads that as a failed build. It keeps being asked after the build has
# ended: the ping polls a connected, idle server every `Ping Interval`
# seconds, and a status poll already in flight lands behind the one that
# ended the build. Each of those answers says the same thing and has to be
# read the same way. Read as "no build yet", the panel goes back to waiting
# for a build that was refused, and a caller waiting for the verdict of a
# refused build waits until its timeout while the server's own log shows the
# refusal.
#
# Two rigid cubes 2 cm on a side with a contact gap the step cannot solve,
# which the session build refuses by name (`bl_contact_gap_unsolvable` is
# about that refusal; here it is only a refusal that needs no geometry).
#
#   `A_the_refused_build_reads_failed`: FAILED, and the refusal is in the
#       Console. The server's report carries the sentence twice, as its
#       headline and as the last line of its traceback, so the checks below
#       compare against the number of lines this one write left.
#   `B_three_more_status_queries_still_read_failed`: three status queries,
#       each answered. FAILED after each, the activity idle, and no line
#       added to the Console.
#   `C_the_ping_leaves_it_failed`: Ping Interval 1 s and the main thread
#       asleep for 3.5 s, so that nothing but the ping thread reaches the
#       server. At least two pings answered, FAILED, the Console unchanged.
#   `D_the_corrected_scene_builds`: the gap raised to its default, READY, and
#       READY after one more ping.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it against a CPU build.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import time
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT = "build_refusal_survives_polls"
SIDE = 0.02
REFUSAL = "The contact gap between 'Left' and 'Right' is"


try:
    dh = DriverHelpers(pkg, result)
    console = __import__(pkg + ".models.console", fromlist=["console"]).console
    dh.log("setup_start")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for name, x in (("Left", -0.05), ("Right", 0.05)):
        bpy.ops.mesh.primitive_cube_add(size=SIDE, location=(x, 0.0, 0.0))
        bpy.context.active_object.name = name

    root = dh.configure_state(project_name=PROJECT, frame_count=10,
                              frame_rate=60, step_size=0.01)
    root.state.disable_contact = False
    cubes = dh.api.solver.create_group("Cubes", "PDRD")
    cubes.add("Left", "Right")
    cubes.param.use_group_bounding_box_diagonal = True

    def standing():
        s = dh.facade.engine.state
        return {"solver": s.solver.name, "activity": s.activity.name,
                "server_error": (s.server_error or "")[:300]}

    def refusal_lines():
        console.process_messages()
        return [line.body[:160] for line in console.get_or_create().lines
                if REFUSAL in line.body]

    def refusals_in_console():
        return len(refusal_lines())

    def answered_since(start):
        return sum(1 for at, name, _ in dh.facade.engine.recent_events
                   if name == "ServerPolled" and at >= start)

    def settle(seconds):
        end = time.time() + seconds
        while time.time() < end:
            dh.facade.tick()
            time.sleep(0.05)

    def reads_failed(at):
        return (at["solver"] == "FAILED" and at["activity"] == "IDLE"
                and REFUSAL in at["server_error"])

    # ----- A: the refusal ----------------------------------------------
    cubes.param.contact_gap_rat = 1e-5
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=PROJECT)
    verdict, error = dh.build_verdict("a gap the step cannot solve",
                                      timeout=120.0)
    settle(0.3)
    at = standing()
    written_at_verdict = refusals_in_console()
    dh.record("A_the_refused_build_reads_failed",
              verdict == "FAILED" and REFUSAL in error and reads_failed(at)
              and written_at_verdict >= 1,
              {"verdict": verdict, "error": error[:300], "standing": at,
               "refusal_lines_in_console": written_at_verdict,
               "lines": refusal_lines()})

    # ----- B: asked again, three times ----------------------------------
    after_each = []
    for _ in range(3):
        start = time.monotonic()
        dh.facade.engine.dispatch(dh.events.QueryRequested())
        deadline = time.time() + 15.0
        while time.time() < deadline and answered_since(start) < 1:
            dh.facade.tick()
            time.sleep(0.05)
        settle(0.3)
        after_each.append({"answered": answered_since(start), **standing()})
    written = refusals_in_console()
    dh.record("B_three_more_status_queries_still_read_failed",
              all(a["answered"] >= 1 and reads_failed(a) for a in after_each)
              and written == written_at_verdict,
              {"after_each": after_each, "refusal_lines_in_console": written,
               "at_verdict": written_at_verdict})

    # ----- C: the ping --------------------------------------------------
    root.ssh_state.ping_interval = 1
    start = time.monotonic()
    time.sleep(3.5)
    settle(0.5)
    pings = answered_since(start)
    at = standing()
    written = refusals_in_console()
    dh.record("C_the_ping_leaves_it_failed",
              pings >= 2 and reads_failed(at)
              and written == written_at_verdict,
              {"pings_answered": pings, "standing": at,
               "refusal_lines_in_console": written,
               "at_verdict": written_at_verdict})

    # ----- D: the corrected scene ---------------------------------------
    cubes.param.contact_gap_rat = 1e-3
    verdict, error = dh.build_verdict("the default gap", timeout=120.0)
    start = time.monotonic()
    time.sleep(1.5)
    settle(0.5)
    at = standing()
    dh.record("D_the_corrected_scene_builds",
              verdict == "READY" and answered_since(start) >= 1
              and at["solver"] == "READY" and at["server_error"] == "",
              {"verdict": verdict, "error": error[:300],
               "pings_answered": answered_since(start), "standing": at})

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
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 600.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
