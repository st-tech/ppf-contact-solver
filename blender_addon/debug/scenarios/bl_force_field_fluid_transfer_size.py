# File: scenarios/bl_force_field_fluid_transfer_size.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# What a LONG Fluid Flow sequence costs on the way to the solver: the
# parameter payload, its decode in the build worker, the session on disk, and
# the drift hash the Run button computes. A Fluid Flow's volume travels in the
# PARAMETER payload, one set of bricks per solve frame, so a long sequence is
# what makes Transfer large; each check below is a ratio against the bricks
# themselves, never an absolute size or time, so it holds on any machine.
#
# ON DEMAND (run it by name): it writes a 150-frame VDB sequence and sends
# several hundred MB through a real server, too heavy for every run.
#
# The sequence is synthetic, written here with Blender's openvdb: a 96^3 box
# (its eight corner voxels hold a small velocity, so the active bounding box,
# which is what the add-on reads, is the whole box) holding a sphere of fluid
# 28 cells in radius that drifts along x from frame to frame. About 18% of the
# box's 1728 bricks are kept, the occupancy a plume in a closed domain has.
#
# Subtests. Each bound is set from what dev-head measured (CPU build, 150
# frames, 383 MB of bricks, 18% of 259200 brick slots occupied):
#   A. payload_is_the_bricks: the encoded PARAM payload is at most 1.1x the
#      bytes of its tables and bricks. Measured 1.00007x: everything but the
#      bricks is a few hundred bytes a frame. A report that counted a quarter
#      of the bytes reads 4.0x.
#   B. decode_is_linear_in_the_frame: a frame of 1, 8 and 64 MiB of bricks,
#      chunked by the add-on's own `_chunks` and enveloped by its own encoder,
#      is decoded by the build worker's code (`frontend._cbor_bridge_.
#      loads_param`, then `_decoder_._joined_bytes`) in a child of the rig's
#      venv interpreter, whose cbor2 is the build worker's, at a per-MiB cost
#      that grows at most 3x from 8 to 64 MiB (1.11x on dev-3). A decode
#      quadratic in one byte string grows it 8x there; the same frames sent as
#      ONE byte string each grew 184x from 1 to 64 MiB on cbor2 6.0.1. The
#      1 MiB frame is reported but not compared: it fits in the CPU's cache
#      and decoded up to 18x faster per MiB on dev-3, which a growth from it
#      would read as a quadratic decode.
#   C. session_disk_is_bounded: the built session directory is at most 2.2x
#      the brick bytes. Measured 1.0004x.
#   D. run_hash_reuses_the_encode: right after the encode, the Run button's
#      drift hash (`compute_param_hash`) sends the remembered volume again
#      (its report says reused), equals the hash of the bytes the encode
#      produced, and costs at most 25% of the encode. A hash that reuses the
#      volume measured 0.36 to 0.39 s against a 3.2 to 3.3 s encode (11% to
#      12%): it still encodes and hashes the 383 MB, so the margin is 2x; one
#      that reads the sequence again measured 96% to 97%.
#   E. a_payload_over_the_cap_is_refused_before_sending: with the transport
#      cap lowered below this sequence's size, the encode refuses by name,
#      naming the cap and the projected size, before any byte is uploaded.
#   F. the_size_is_reported_before_sending: the report Transfer prints after
#      the encode names the field, the frame count, and the bytes sent.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
# RUN AGAINST A CPU BUILD on dev-head; it builds a session and starts no solve.
BACKENDS = ("real",)
ON_DEMAND = True

_FRAMES = 150
_RESOLUTION = 96

# The per-MiB decode cost may grow this much from an 8 MiB to a 64 MiB frame
# (subtest B). A decode quadratic in one byte string grows it 64 / 8 = 8x;
# a linear, chunked one keeps it near 1x (measured 1.11x on dev-3). The 1 MiB
# frame is measured and reported but not compared: it fits in the CPU's cache
# and decodes up to 18x faster per MiB than either larger frame (dev-3), so a
# growth taken from it measures the cache, not the decode.
_DECODE_GROWTH_BOUND = 3.0

_DRIVER_BODY = r"""
import glob
import os
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
result.setdefault("measured", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT_ROOT = "<<PROJECT_ROOT>>"
FRAMES = <<FRAMES>>
N = <<RESOLUTION>>

try:
    import openvdb

    dh = DriverHelpers(pkg, result)
    ff = __import__(pkg + ".core.force_field", fromlist=["encode"])
    params_mod = __import__(pkg + ".core.encoder.params", fromlist=["compute_param_hash"])
    digest = __import__(pkg + ".core.encoder.cbor_encode", fromlist=["payload_digest"]).payload_digest
    cbor_encode = __import__(pkg + ".core.encoder.cbor_encode", fromlist=["dumps_envelope"])
    scene = bpy.context.scene

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene.frame_start = 1
    scene.frame_end = FRAMES

    # ----- the sequence -----------------------------------------------------
    cell = 2.0 / N
    radius = 28
    vdb_dir = os.path.join(PROBE_DIR, "transfer_size_vdb")
    os.makedirs(vdb_dir, exist_ok=True)
    ii, jj, kk = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
    t0 = time.time()
    for frame in range(1, FRAMES + 1):
        cx = 34 + 28.0 * (frame - 1) / max(FRAMES - 1, 1)
        inside = (ii - cx) ** 2 + (jj - 48) ** 2 + (kk - 48) ** 2 <= radius * radius
        u = np.zeros((N, N, N, 3), dtype=np.float32)
        u[..., 0] = np.where(inside, 1.0 + 0.01 * (jj - 48), 0.0)
        u[..., 1] = np.where(inside, 0.2 * np.sin(0.1 * frame + 0.05 * ii), 0.0)
        u[..., 2] = np.where(inside, 0.1, 0.0)
        for c in [(0, 0, 0), (N - 1, 0, 0), (0, N - 1, 0), (0, 0, N - 1), (N - 1, N - 1, 0),
                  (N - 1, 0, N - 1), (0, N - 1, N - 1), (N - 1, N - 1, N - 1)]:
            u[c] = (1e-3, 0.0, 0.0)
        g = openvdb.Vec3SGrid()
        g.name = "vel"
        g.transform = openvdb.createLinearTransform(voxelSize=cell)
        g.copyFromArray(u)
        openvdb.write(os.path.join(vdb_dir, "vel_%04d.vdb" % frame), grids=[g])
    result["measured"]["write_sequence_s"] = round(time.time() - t0, 2)

    vol = bpy.data.volumes.new("FFLongVel")
    vol.filepath = os.path.join(vdb_dir, "vel_0001.vdb")
    vol.is_sequence = True
    vol.frame_start = 1
    vol.frame_duration = FRAMES
    vol.grids.load()
    vol.velocity_unit = "SECOND"
    vol_obj = bpy.data.objects.new("FFLongVel", vol)
    scene.collection.objects.link(vol_obj)
    vol_obj.location = (-1.0, -1.0, -1.0)

    bpy.ops.mesh.primitive_grid_add(x_subdivisions=4, y_subdivisions=4, size=0.3,
                                    location=(0.0, 0.0, 0.0))
    sheet = bpy.context.object
    sheet.name = "FFLongSheet"
    bpy.ops.object.effector_add(type="FLUID", location=(0.0, 0.0, 0.0))
    flow = bpy.context.object
    flow.name = "FFLongFlow"
    flow.field.source_object = vol_obj
    flow.field.strength = 5.0
    flow.field.flow = 1.0
    scene.frame_set(1)
    dh.save_blend(PROBE_DIR, "force_field_fluid_transfer_size.blend")
    root = dh.configure_state(project_name="force_field_fluid_transfer_size",
                              frame_count=FRAMES, frame_rate=24)
    state = root.state
    dh.api.solver.create_group("Cloth", "SHELL").add(sheet.name)

    # ----- the encode (Transfer's parameter stage) ---------------------------
    ff.release_encode_cache()
    t0 = time.time()
    param_bytes = params_mod.encode_param(bpy.context)
    encode_s = time.time() - t0
    (rep,) = ff.volume_report()
    bricks_bytes = rep["bytes"]
    result["measured"].update(
        encode_s=round(encode_s, 3), payload_bytes=len(param_bytes), brick_bytes=bricks_bytes,
        occupancy=round(rep["occupancy"], 4), frames=rep["frames"],
        resolution=rep["resolution"])

    # ----- A ----------------------------------------------------------------
    ratio_a = len(param_bytes) / float(bricks_bytes) if bricks_bytes else float("inf")
    result["measured"]["A_payload_over_bricks"] = round(ratio_a, 5)
    dh.record("A_payload_is_the_bricks",
              rep["frames"] == FRAMES and rep["resolution"] == [N, N, N]
              and 0.10 < rep["occupancy"] < 0.25 and ratio_a <= 1.1,
              {"ratio": ratio_a, "report": rep})

    # ----- D ----------------------------------------------------------------
    t0 = time.time()
    run_hash = params_mod.compute_param_hash(bpy.context)
    hash_s = time.time() - t0
    (again,) = ff.volume_report()
    ratio_d = hash_s / encode_s if encode_s > 0 else float("inf")
    # A second drift hash, for the record: what a hash costs once the cache
    # is used, whichever encode first manages to use it.
    t0 = time.time()
    params_mod.compute_param_hash(bpy.context)
    result["measured"].update(
        hash_s=round(hash_s, 3), D_hash_over_encode=round(ratio_d, 4),
        second_hash_s=round(time.time() - t0, 3),
        second_hash_reused=ff.volume_report()[0].get("reused"))
    dh.record("D_run_hash_reuses_the_encode",
              again.get("reused") is True
              and run_hash == digest(param_bytes)
              and ratio_d <= 0.25,
              {"reused": again.get("reused"), "hash_s": hash_s, "encode_s": encode_s,
               "same_hash": run_hash == digest(param_bytes)})

    # ----- F ----------------------------------------------------------------
    line = ff.volume_report_line(rep)
    dh.record("F_the_size_is_reported_before_sending",
              line.startswith("Fluid Flow 'FFLongFlow': %d frames of 'FFLongVel'" % FRAMES)
              and ff._size(bricks_bytes) in line,
              {"line": line})

    # ----- E ----------------------------------------------------------------
    cap = ff.TRANSPORT_CAP_BYTES
    ff.TRANSPORT_CAP_BYTES = bricks_bytes // 2
    ff.release_encode_cache()
    refused = None
    try:
        params_mod.encode_param(bpy.context)
    except ValueError as e:
        refused = str(e)
    finally:
        ff.TRANSPORT_CAP_BYTES = cap
        ff.release_encode_cache()
    dh.record("E_a_payload_over_the_cap_is_refused_before_sending",
              refused is not None and "FFLongFlow" in refused
              and ("past the %s a transfer carries" % ff._size(bricks_bytes // 2)) in refused
              and "projects to" in refused,
              {"error": refused})

    # ----- C: build through a real server -----------------------------------
    # The encode cache was dropped above, so this encode reads the sequence
    # again, as a first Transfer does.
    data_bytes, param_bytes = dh.encode_payload()
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=state.project_name)
    t0 = time.time()
    dh.build_and_wait(data_bytes, param_bytes, message="fluid_transfer_size:build",
                      timeout=900.0)
    result["measured"]["upload_and_build_s"] = round(time.time() - t0, 2)
    search_root = os.path.dirname(PROJECT_ROOT.rstrip("/")) or PROJECT_ROOT
    hits = glob.glob(os.path.join(search_root, "**", "session"), recursive=True)
    if not hits:
        raise RuntimeError("no session directory under %s" % search_root)
    session = sorted(hits, key=os.path.getmtime)[-1]
    disk = 0
    for dirpath, _dirs, names in os.walk(session):
        for nm in names:
            disk += os.path.getsize(os.path.join(dirpath, nm))
    ratio_c = disk / float(bricks_bytes)
    result["measured"].update(session_bytes=disk, C_session_over_bricks=round(ratio_c, 4))
    dh.record("C_session_disk_is_bounded", ratio_c <= 2.2,
              {"session": session, "bytes": disk, "brick_bytes": bricks_bytes,
               "ratio": ratio_c})

    # ----- B: frames for the host half to decode ----------------------------
    blobs = {}
    rng = np.random.default_rng(7)
    for mib in (1, 8, 64):
        bricks = rng.random((mib << 20) // 4, dtype=np.float32)
        payload = {"force_field": {"fields": [{"type": "fluid", "fluid": {
            "tables": [ff._chunks(np.zeros(1, dtype="<u4"))],
            "bricks": [ff._chunks(bricks)]}}]}}
        path = os.path.join(PROBE_DIR, "decode_%dmib.cbor" % mib)
        with open(path, "wb") as fh:
            fh.write(cbor_encode.dumps_envelope("Param", payload))
        blobs[str(mib)] = path
    result["decode_blobs"] = blobs

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
        .replace("<<FRAMES>>", str(_FRAMES))
        .replace("<<RESOLUTION>>", str(_RESOLUTION))
    )


# Run in a child of the rig's own interpreter (the venv the build worker runs
# in), so the frontend is imported there and not into the orchestrator.
_DECODE_TIMER = r"""
import json, sys, time
sys.path.insert(0, sys.argv[1])
from frontend import _cbor_bridge_ as cb
from frontend import _decoder_ as dec

out = {}
for mib, path in json.loads(sys.argv[2]).items():
    mib = int(mib)
    with open(path, "rb") as fh:
        blob = fh.read()

    def once():
        start = time.perf_counter()
        payload = cb.loads_param(blob)
        volume = payload["force_field"]["fields"][0]["fluid"]
        joined = dec._joined_bytes(volume["bricks"][0], "bricks")
        took = time.perf_counter() - start
        if len(joined) != mib << 20:
            raise RuntimeError("%s: joined %d bytes, wrote %d" % (path, len(joined), mib << 20))
        return took

    once()  # warm-up
    out[mib] = min(once() for _ in range(max(3, 128 // mib))) / mib
print(json.dumps(out))
"""


def _decode_per_mib(blobs: dict) -> dict:
    """The least time per MiB the build worker's code takes to decode each
    PARAM envelope in ``blobs`` (MiB -> path) and join its fluid frame's
    chunks, over repeated decodes after one warm-up."""
    import json
    import subprocess
    import sys

    done = subprocess.run([sys.executable, "-c", _DECODE_TIMER, REPO_ROOT_POSIX,
                           json.dumps(blobs)], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=600)
    if done.returncode != 0:
        raise RuntimeError(f"decode timer exited {done.returncode}: {done.stderr[-2000:]}")
    return {int(k): float(v) for k, v in json.loads(done.stdout.strip().splitlines()[-1]).items()}


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 1500.0))
    if err is not None:
        return err
    checks = dict(result.get("checks", {}))
    measured = dict(result.get("measured", {}))
    blobs = result.get("decode_blobs") or {}
    try:
        per_mib = _decode_per_mib(blobs)
        growth = per_mib[64] / per_mib[8]
        measured["B_decode_ms_per_mib"] = {k: round(v * 1e3, 4) for k, v in per_mib.items()}
        measured["B_growth_8_to_64"] = round(growth, 3)
        checks["B_decode_is_linear_in_the_frame"] = {
            "ok": sorted(per_mib) == [1, 8, 64] and growth <= _DECODE_GROWTH_BOUND,
            "details": {"ms_per_mib": measured["B_decode_ms_per_mib"], "growth": growth,
                        "bound": _DECODE_GROWTH_BOUND}}
    except Exception as e:  # reported as the check's failure, with its cause
        checks["B_decode_is_linear_in_the_frame"] = {
            "ok": False, "details": {"error": f"{type(e).__name__}: {e}", "blobs": blobs}}
    verdict = r.report_named_checks(checks)
    verdict.setdefault("notes", []).append(f"measured: {measured}")
    return verdict
