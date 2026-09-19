# File: scenarios/bl_stale_statistics_manifest.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Changing a scene's dynamics groups between two runs must not leave the
# runner decoding the new run's frames against the previous run's statistics
# manifest.
#
# The three session artifacts (vertex map, surface map, statistics manifest)
# are downloaded only while unset, so they outlive the dataset they describe
# unless something drops them. Fetch and disconnect do; Transfer and Run emit
# DoClearAnimation, which clears the queue and the counters and leaves the
# artifacts in place. A second run of a scene whose group set changed then
# decodes frames carrying one object count against a manifest naming another,
# and ``decode_frame`` rejects every frame of that run with "statistics frame
# object count does not match manifest": the sim appears frozen while the frame
# counter climbs, and the panel reports nothing because the apply error is
# raised outside FETCHING / APPLYING.
#
# ``_drop_stale_session_artifacts`` binds the artifacts to the ``upload_id``
# they were fetched under, so the scenario asserts both halves: the frames of
# the second run land, and the id the runner holds actually changed (a pass
# that came from never caching anything would not be the fix).
#
# Both directions are covered, since the reported failure appears when a group
# is deleted and when one is added back.

from __future__ import annotations

from . import REPO_ROOT_POSIX
from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True
# RUNS ON THE REAL BACKEND, established by RUNNING it: it passes a
# real-backend run unchanged.
BACKENDS = ("real",)
# This scenario carries no pacing or elasticity knobs: a real backend has
# no artificial per-step sleep and always computes real elasticity, so the
# intent is preserved by asking for neither.


_DRIVER_BODY = r"""
import os
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = 6
MISMATCH = "statistics frame object count does not match manifest"


def manifest_count(blob):
    # Number of objects a manifest blob describes, or None when there is
    # no blob at all.
    if not blob:
        return None
    import cbor2  # type: ignore
    return len(cbor2.loads(blob)["payload"]["objects"])


try:
    dh = DriverHelpers(pkg, result)
    console_mod = __import__(pkg + ".models.console", fromlist=["console"])
    runner = dh.facade.runner

    # Console capture. ``client.apply_animation`` reports a failed apply
    # through the console, so a regression shows up here rather than as a
    # raised exception.
    captured = []
    _original_write = console_mod.console.write

    def _spy(message, timestamp=True):
        captured.append(str(message))
        return _original_write(message, timestamp)

    console_mod.console.write = _spy

    def mismatches():
        return [line for line in captured if MISMATCH in line]

    def pump(seconds, applied_so_far=0):
        # Poll and apply the way the running addon does: the status poll
        # queues the newest frame and the FramePump modal applies it. The
        # driver owns the main thread, so it stands in for the modal.
        deadline = time.time() + seconds
        applied = applied_so_far
        while time.time() < deadline:
            dh.facade.engine.dispatch(dh.events.PollTick())
            dh.facade.tick()
            before = runner._anim_applied
            dh.client.apply_animation()
            applied += max(0, runner._anim_applied - before)
            time.sleep(0.05)
        return applied

    def run_and_apply(timeout=180.0, tail=3.0):
        # The Run button is ``com.run(context)``: the context argument is
        # what clears the fetched-frame set, so the new run's frames are
        # fetched again instead of being taken for ones already applied.
        dh.com.run(bpy.context)
        applied = 0
        saw_running = False
        deadline = time.time() + timeout
        while time.time() < deadline:
            dh.facade.engine.dispatch(dh.events.PollTick())
            dh.facade.tick()
            before = runner._anim_applied
            dh.client.apply_animation()
            applied += max(0, runner._anim_applied - before)
            state = dh.facade.engine.state
            if state.solver.name == "RUNNING" or state.frame > 0:
                saw_running = True
            if saw_running and state.solver.name in ("READY", "RESUMABLE", "FAILED"):
                break
            time.sleep(0.05)
        # The last frame is fetched on the poll after the run ends, so keep
        # pumping past completion.
        return pump(tail, applied)

    def wait_for_cached_manifest(seconds=20.0):
        deadline = time.time() + seconds
        while time.time() < deadline:
            dh.facade.engine.dispatch(dh.events.PollTick())
            dh.facade.tick()
            if runner._anim_statistics_manifest:
                return True
            time.sleep(0.1)
        return False

    # -- scene: two groups, one object each -------------------------------
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0.0, 0.0, 0.0))
    cube = bpy.context.object
    cube.name = "StaleCube"
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=4, y_subdivisions=4, size=1.0, location=(0.0, 0.0, 1.0)
    )
    sheet = bpy.context.object
    sheet.name = "StaleSheet"
    dh.save_blend(PROBE_DIR, "stale_statistics_manifest.blend")
    root = dh.configure_state(
        project_name="stale_statistics_manifest",
        frame_count=FRAME_COUNT,
        frame_rate=24,
        step_size=1.0 / 24.0,
    )

    first_group = dh.api.solver.create_group("Solids", "SHELL")
    first_group.add(cube.name)
    second_group = dh.api.solver.create_group("Shells", "SHELL")
    second_group.add(sheet.name)

    dh.connect(
        local_path=LOCAL_PATH,
        server_port=SERVER_PORT,
        project_name=root.state.project_name,
    )

    # -- run 1: both groups present ---------------------------------------
    data_bytes, param_bytes = dh.encode_payload()
    dh.build_and_wait(data_bytes, param_bytes, "stale-manifest:build-two", timeout=180.0)
    taken_run1 = run_and_apply()
    landed_run1 = len(root.state.fetched_frame)
    cached_after_run1 = wait_for_cached_manifest()
    dh.settle_idle(timeout=20.0)

    remote_root = dh.facade.engine.state.remote_root
    manifest_on_disk = os.path.join(
        remote_root, "session", "output", "statistics_manifest.cbor"
    )

    def disk_manifest_count():
        with open(manifest_on_disk, "rb") as handle:
            return manifest_count(handle.read())

    run1_cached_count = manifest_count(runner._anim_statistics_manifest)
    run1_disk_count = disk_manifest_count()
    run1_upload_id = runner._anim_upload_id

    # A: the first run caches the manifest naming both objects and applies
    # its frames.
    dh.record(
        "A_first_run_caches_two_object_manifest",
        cached_after_run1
        and run1_cached_count == 2
        and run1_disk_count == 2
        and landed_run1 > 0
        and bool(run1_upload_id)
        and not mismatches(),
        {
            "cached": cached_after_run1,
            "cached_objects": run1_cached_count,
            "disk_objects": run1_disk_count,
            "frames_taken": taken_run1,
            "frames_landed": landed_run1,
            "upload_id": run1_upload_id,
        },
    )

    # -- delete one group, transfer, run again ----------------------------
    captured.clear()
    second_group.delete()
    group_count = len(dh.api.solver.get_groups())
    data_bytes, param_bytes = dh.encode_payload()
    dh.build_and_wait(data_bytes, param_bytes, "stale-manifest:build-one", timeout=180.0)
    taken_run2 = run_and_apply()
    landed_run2 = len(root.state.fetched_frame)
    dh.settle_idle(timeout=20.0)

    run2_cached_count = manifest_count(runner._anim_statistics_manifest)
    run2_disk_count = disk_manifest_count()
    run2_upload_id = runner._anim_upload_id

    # B: the transfer landed a new upload, and the runner refreshed the
    # manifest to the one that upload produced rather than keeping the one
    # it already had.
    dh.record(
        "B_manifest_refreshed_on_new_upload",
        group_count == 1
        and run2_disk_count == 1
        and run2_cached_count == 1
        and bool(run2_upload_id)
        and run2_upload_id != run1_upload_id,
        {
            "groups": group_count,
            "disk_objects": run2_disk_count,
            "cached_objects": run2_cached_count,
            "upload_id": run2_upload_id,
            "previous_upload_id": run1_upload_id,
        },
    )

    # C: the reported failure. Every live frame of the second run reaches
    # the mesh cache and the console carries no decode error.
    dh.record(
        "C_second_run_frames_land",
        taken_run2 > 0 and landed_run2 == taken_run2 and not mismatches(),
        {
            "frames_taken": taken_run2,
            "frames_landed": landed_run2,
            "mismatch_lines": len(mismatches()),
            "console_tail": captured[-4:],
        },
    )

    # D: a full fetch of the same run still applies every frame against the
    # refreshed manifest.
    captured.clear()
    fetch_applied, fetch_total = dh.fetch_and_drain(
        fetch_timeout=120.0, drain_timeout=60.0
    )
    fetch_cached_count = manifest_count(runner._anim_statistics_manifest)
    fetch_landed = len(root.state.fetched_frame)
    dh.record(
        "D_full_fetch_applies_every_frame",
        fetch_applied > 0
        and fetch_applied >= fetch_total
        and fetch_landed == fetch_applied
        and fetch_cached_count == 1
        and not mismatches(),
        {
            "frames_taken": fetch_applied,
            "frames_total": fetch_total,
            "frames_landed": fetch_landed,
            "cached_objects": fetch_cached_count,
            "console_tail": captured[-4:],
        },
    )

    # E: the other direction. Adding a group back grows the object count,
    # which failed the same way before the artifacts carried an identity.
    captured.clear()
    third_group = dh.api.solver.create_group("Shells", "SHELL")
    third_group.add(sheet.name)
    data_bytes, param_bytes = dh.encode_payload()
    dh.build_and_wait(data_bytes, param_bytes, "stale-manifest:build-two-again",
                      timeout=180.0)
    taken_run3 = run_and_apply()
    landed_run3 = len(root.state.fetched_frame)
    dh.settle_idle(timeout=20.0)
    run3_cached_count = manifest_count(runner._anim_statistics_manifest)
    run3_disk_count = disk_manifest_count()
    dh.record(
        "E_readding_a_group_lands_too",
        run3_disk_count == 2
        and run3_cached_count == 2
        and taken_run3 > 0
        and landed_run3 == taken_run3
        and not mismatches(),
        {
            "disk_objects": run3_disk_count,
            "cached_objects": run3_cached_count,
            "frames_taken": taken_run3,
            "frames_landed": landed_run3,
            "mismatch_lines": len(mismatches()),
        },
    )

    console_mod.console.write = _original_write

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
