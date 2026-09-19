# File: scenarios/bl_material_keyframe_animates.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A keyframe on a material slider has to reach the solver.
#
# The reported symptom was that keyframing the inflation pressure did nothing:
# Blender accepted the curve, the encoder read the slider once, and the solve
# ran at a constant value. This covers the whole chain that fixes it, on the
# backend, so it runs on any host:
#
#   A. payload_carries_schedule: the encoded param blob has the sampled curve
#      under the group's "param-anim" and a scene-wide "param_anim_times".
#   B. session_carries_schedule: the built session has bin/param_anim/times.bin
#      and bin/param_anim/tri-pressure.bin, the second holding exactly
#      len(times) * n_tri floats, which is the layout the solver reads.
#   C. schedule_matches_curve: the first and last samples equal the keyframed
#      endpoints, so the values are the artist's and not a default.
#
# Whether the pressure then INFLATES is physics the solver does not
# compute; that belongs to a real-GPU run.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. A full rig sweep against a CPU build passed it, and that run is the
# evidence this line rests on.
BACKENDS = ("real",)

_FRAME_COUNT = 10
_P0 = 0.0
_P1 = 5000.0


_DRIVER_BODY = r"""
import os
import struct
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT_ROOT = "<<PROJECT_ROOT>>"
FRAME_COUNT = <<FRAME_COUNT>>
P0 = <<P0>>
P1 = <<P1>>

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="AnimMesh")
    root = dh.configure_state(project_name="mat_keyframe", frame_count=FRAME_COUNT)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    # A fully pinned object with no pin operation is promoted to a static
    # collider and leaves the solved namespace, taking its material with it.
    pin.move_by(delta=(0.05, 0.0, 0.0), frame_start=1, frame_end=4,
                transition="LINEAR")

    group = root.object_group_0
    group.enable_inflate = True
    group.inflate_pressure = P0
    group.keyframe_insert(data_path="inflate_pressure", frame=1)
    group.inflate_pressure = P1
    group.keyframe_insert(data_path="inflate_pressure", frame=FRAME_COUNT)
    dh.log("keyframed inflate_pressure")

    data_bytes, param_bytes = dh.encode_payload()
    blob = dh.decode_addon_blob(param_bytes)
    times = blob.get("param_anim_times") or []
    series = {}
    for entry in blob.get("group", []):
        pa = entry[0].get("param-anim")
        if pa:
            series = pa
    pressure = series.get("pressure") or []
    dh.record(
        "A_payload_carries_schedule",
        bool(times) and len(pressure) == len(times) and len(times) >= 2,
        {"n_times": len(times), "n_pressure": len(pressure),
         "first": pressure[0] if pressure else None,
         "last": pressure[-1] if pressure else None},
    )
    dh.record(
        "C_schedule_matches_curve",
        bool(pressure) and abs(pressure[0] - P0) < 1e-3
        and abs(pressure[-1] - P1) < 1e-3,
        {"expected": [P0, P1],
         "seen": [pressure[0], pressure[-1]] if pressure else []},
    )

    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes, message="material keyframe")
    dh.log("built")

    # The session the build produced. Globbed under the rig's own project root
    # rather than guessed from a data-root convention, so a change to that
    # convention fails here loudly instead of leaving the checks below to pass
    # on an empty directory.
    import glob
    # The build writes to <project root>/../<project name>/session, a SIBLING
    # of the slot directory PROJECT_ROOT names, so search from the parent.
    search_root = os.path.dirname(PROJECT_ROOT.rstrip("/")) or PROJECT_ROOT
    hits = glob.glob(os.path.join(search_root, "**", "session"), recursive=True)
    if not hits:
        raise RuntimeError("no session directory under %s" % search_root)
    session = sorted(hits, key=os.path.getmtime)[-1]
    anim = os.path.join(session, "bin", "param_anim")
    files = sorted(os.listdir(anim)) if os.path.isdir(anim) else []
    n_tri = 0
    tri_static = os.path.join(session, "bin", "param", "tri-pressure.bin")
    if os.path.isfile(tri_static):
        n_tri = os.path.getsize(tri_static) // 4
    n_times_on_disk = 0
    n_vals_on_disk = 0
    if "times.bin" in files:
        n_times_on_disk = os.path.getsize(os.path.join(anim, "times.bin")) // 8
    if "tri-pressure.bin" in files:
        n_vals_on_disk = os.path.getsize(os.path.join(anim, "tri-pressure.bin")) // 4
    dh.record(
        "B_session_carries_schedule",
        n_times_on_disk >= 2 and n_tri > 0
        and n_vals_on_disk == n_times_on_disk * n_tri,
        {"files": files, "n_times": n_times_on_disk, "n_tri": n_tri,
         "n_values": n_vals_on_disk,
         "expected_values": n_times_on_disk * n_tri},
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
        .replace("<<PROJECT_ROOT>>", ctx.project_root.replace("\\", "/"))
        .replace("<<FRAME_COUNT>>", str(_FRAME_COUNT))
        .replace("<<P0>>", repr(_P0))
        .replace("<<P1>>", repr(_P1))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 360.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
