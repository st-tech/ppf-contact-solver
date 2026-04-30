# File: scenarios/bl_intersection_records_roundtrip.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Rust-solver intersection-records JSON round-trip.
#
# When the Rust solver fails to advance because contact resolution
# leaves ``intersection_free=false``, ``write_intersection_records``
# in ``src/backend.rs`` dumps the offending face-edge / edge-edge /
# collision-mesh entries to ``<output>/intersection_records.json``
# before the binary panics. The server then surfaces the crash to
# the addon, which logs the failure (including the JSON path string)
# into ``state.server_error`` / the console.
#
# This scenario forces that path with the emulated knob
# ``PPF_EMULATED_FAIL_AT_FRAME=N``: after frame ``N``, the next
# ``advance`` call flips ``intersection_free`` to false and the
# emulator seeds three synthetic records (one face-edge with
# ``itype=0``, one edge-edge with ``itype=1``, one collision-mesh
# with ``itype=2``).
#
# Invocation:
#   python3 -m blender_addon.debug.main \\
#       --scenarios bl_intersection_records_roundtrip \\
#       --knob PPF_EMULATED_FAIL_AT_FRAME=2
#
# If the knob is not provided the scenario uses a default of ``2``
# so the orchestrator's default invocation still exercises the path.
#
# Asserts:
#   A. ``json_file_present``: ``intersection_records.json`` exists
#      under ``<remote_root>/session/output/``.
#   B. ``json_has_three_records``: parsed JSON has exactly 3 entries
#      (one per record type emitted by the emulator).
#   C. ``record_schema_correct``: every record has the keys the real
#      ``write_intersection_records`` writes (``type``, ``elem0``,
#      ``elem1``, ``positions0``, ``positions1``).
#   D. ``addon_log_mentions_path``: the addon-side surface (console
#      messages plus ``state.server_error``) mentions
#      ``intersection_records.json`` so the user can find the dump.

from __future__ import annotations

import os

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True

# The Rust binary needs PPF_EMULATED_FAIL_AT_FRAME set to trip the
# synthetic intersection-failure path; declare a per-scenario default
# so plain ``runtests`` (no --knob flag) still exercises the path.
# A CLI ``--knob`` overrides this.
KNOBS = {"PPF_EMULATED_FAIL_AT_FRAME": "2"}


_DRIVER_BODY = r"""
import json
import os
import time
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FAIL_AT_FRAME = <<FAIL_AT_FRAME>>
EXPECTED_KEYS = {"type", "elem0", "elem1", "positions0", "positions1"}


try:
    # Frame budget needs to outlive FAIL_AT_FRAME so the solver actually
    # reaches the failure trigger before completing naturally.
    FRAME_COUNT = max(FAIL_AT_FRAME + 4, 6)

    dh = DriverHelpers(pkg, result)
    dh.log(f"setup_start fail_at_frame={FAIL_AT_FRAME}")
    plane = dh.reset_scene_to_pinned_plane(name="IntersectMesh")
    dh.save_blend(PROBE_DIR, "intersect.blend")
    root = dh.configure_state(project_name="intersection_records",
                              frame_count=FRAME_COUNT)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.1, 0.0, 0.0), frame_start=1,
                frame_end=FRAME_COUNT - 1, transition="LINEAR")

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes,
                      message="intersection_records:build")
    dh.log(f"built solver={dh.facade.engine.state.solver.name}")

    remote_root = dh.facade.engine.state.remote_root
    output_dir = os.path.join(remote_root, "session", "output")
    json_path = os.path.join(output_dir, "intersection_records.json")
    dh.log(f"expecting json at {json_path}")

    # Run, then poll until the solver settles into FAILED (or RESUMABLE
    # if the emulator chose to checkpoint instead). The crash should
    # land on the advance after frame FAIL_AT_FRAME.
    dh.com.run()
    deadline = time.time() + 90.0
    saw_running = False
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if s.solver.name == "RUNNING":
            saw_running = True
        if saw_running and s.solver.name in ("FAILED", "READY", "RESUMABLE"):
            break
        time.sleep(0.2)
    final_solver = dh.facade.engine.state.solver.name
    dh.log(f"after_run solver={final_solver} "
           f"frame={dh.facade.engine.state.frame}")

    # Give the monitor a beat to flush server_error / console messages.
    dh.settle_idle(timeout=10.0)

    # ----- A: JSON file exists at the canonical path ---------------
    file_exists = os.path.isfile(json_path)
    file_size = os.path.getsize(json_path) if file_exists else 0
    dh.record(
        "A_json_file_present",
        file_exists and file_size > 0,
        {
            "json_path": json_path,
            "file_exists": file_exists,
            "file_size": file_size,
            "remote_root": remote_root,
            "solver_state": final_solver,
        },
    )

    # ----- B: parsed JSON contains exactly three records ----------
    payload = None
    parse_error = ""
    if file_exists:
        try:
            with open(json_path) as f:
                payload = json.load(f)
        except Exception as parse_exc:
            parse_error = f"{type(parse_exc).__name__}: {parse_exc}"
    records = (payload or {}).get("records", []) if isinstance(payload, dict) else []
    dh.record(
        "B_json_has_three_records",
        isinstance(payload, dict) and len(records) == 3,
        {
            "parse_error": parse_error,
            "record_count": len(records),
            "top_level_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
            "count_field": (payload or {}).get("count") if isinstance(payload, dict) else None,
        },
    )

    # ----- C: every record carries the write_intersection_records keys
    missing = []
    types_seen = []
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            missing.append((idx, "not_a_dict", sorted(EXPECTED_KEYS)))
            continue
        gap = EXPECTED_KEYS - set(rec.keys())
        if gap:
            missing.append((idx, sorted(gap), sorted(rec.keys())))
        types_seen.append(rec.get("type", "<missing>"))
    dh.record(
        "C_record_schema_correct",
        len(records) > 0 and not missing,
        {
            "missing": missing,
            "types_seen": types_seen,
            "expected_keys": sorted(EXPECTED_KEYS),
        },
    )

    # ----- D: addon's solver-side log mentions the JSON path ------
    # The Rust solver logs ``wrote N intersection records to <path>``
    # to its stdout before panicking. ``server/monitor.py`` packs that
    # tail into the SolverCrashed.error string, which the addon stores
    # on ``state.server_error`` and pipes through DoLog into the
    # console singleton.
    state_now = dh.facade.engine.state
    server_error = getattr(state_now, "server_error", "") or ""
    state_error = getattr(state_now, "error", "") or ""
    console_mod = __import__(pkg + ".models.console", fromlist=["console"])
    console_msgs = list(getattr(console_mod.console, "messages", []) or [])
    console_text = "\n".join(console_msgs)
    surfaces = {
        "server_error": "intersection_records.json" in server_error,
        "state_error": "intersection_records.json" in state_error,
        "console": "intersection_records.json" in console_text,
    }
    dh.record(
        "D_addon_log_mentions_path",
        any(surfaces.values()),
        {
            "surfaces": surfaces,
            "server_error_tail": server_error[-400:],
            "state_error_tail": state_error[-200:],
            "console_msg_count": len(console_msgs),
        },
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def _resolve_fail_at_frame(ctx: r.ScenarioContext) -> int:
    raw = (ctx.knobs or {}).get("PPF_EMULATED_FAIL_AT_FRAME", "")
    raw = raw.strip()
    if not raw:
        return 2
    try:
        value = int(raw)
    except ValueError:
        return 2
    return value if value >= 0 else 2


def build_driver(ctx: r.ScenarioContext) -> str:
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", repo_root)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
        .replace("<<FAIL_AT_FRAME>>", str(_resolve_fail_at_frame(ctx)))
    )


def run(ctx: r.ScenarioContext) -> dict:
    knob = (ctx.knobs or {}).get("PPF_EMULATED_FAIL_AT_FRAME", "")
    if not knob:
        ctx.log(
            "bl_intersection_records_roundtrip: PPF_EMULATED_FAIL_AT_FRAME "
            "knob not set; defaulting to 2. Pass "
            "--knob PPF_EMULATED_FAIL_AT_FRAME=N to override."
        )
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
