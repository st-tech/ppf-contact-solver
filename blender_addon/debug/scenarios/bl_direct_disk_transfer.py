# File: scenarios/bl_direct_disk_transfer.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Co-located direct-disk transfer coverage.
#
# When the addon and server share a machine (local / win_native), the
# addon writes data.pickle / param.pickle straight to the project root
# on disk and sends only a tiny upload_notify control message instead
# of streaming the payloads through the localhost socket. This is the
# production default; the rig forces every OTHER scenario back onto the
# streamed TCP transport (PPF_FORCE_TCP_TRANSFER=1, seeded in
# orchestrator.run_one) so the wire handlers stay under test. This
# scenario opts back out via its own KNOBS and verifies the disk path:
#
#   * the production gate (backends._force_tcp) resolves to disk;
#   * data.pickle / param.pickle / upload_id.txt land on disk under the
#     addon's canonical remote_root, and the on-disk pickle bytes match
#     the encoded payload (so it is this run's write, not a leftover);
#   * the server stamped the addon-minted upload_id and the data hash
#     via upload_notify, and the addon's status mirrors both (proving
#     the notify round-trip ran, not the streamed path which mints a
#     server-side id);
#   * a full run + fetch round-trips frames back off disk through the
#     direct receive_data read and applies them to the mesh.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
# Opt this scenario out of the rig-wide PPF_FORCE_TCP_TRANSFER=1 default
# so it exercises the co-located direct-disk transport. Every other
# scenario keeps the streamed wire handlers under test.
KNOBS = {"PPF_FORCE_TCP_TRANSFER": "0"}


_DRIVER_BODY = r'''
import os
import time
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT_NAME = "direct_disk_transfer"

try:
    dh = DriverHelpers(pkg, result)
    backends = __import__(pkg + ".core.backends", fromlist=["_force_tcp"])
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["compute_data_hash"])
    encoder_params = __import__(pkg + ".core.encoder.params",
                                fromlist=["compute_param_hash"])

    # The production gate must resolve to disk here. If the knob did not
    # reach the Blender process (or default-on regressed), this fails
    # loudly instead of silently testing the TCP path.
    force_tcp = backends._force_tcp()
    dh.record(
        "direct_disk_selected",
        force_tcp is False,
        {"PPF_FORCE_TCP_TRANSFER": os.environ.get("PPF_FORCE_TCP_TRANSFER"),
         "force_tcp": force_tcp},
    )

    dh.log("setup")
    plane = dh.reset_scene_to_pinned_plane(name="DirectDiskMesh")
    blend_path = os.path.join(os.path.dirname(PROBE_DIR), "direct_disk.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    dh.configure_state(project_name=PROJECT_NAME, frame_count=4)

    # Register the plane as a solvable shell with a moving pin so the
    # run produces real output frames to round-trip back off disk.
    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.1, 0.0, 0.0), frame_start=1, frame_end=4,
                transition="LINEAR")

    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=PROJECT_NAME)

    data_bytes, param_bytes = dh.encode_payload()
    data_hash = encoder_mesh.compute_data_hash(bpy.context)
    param_hash = encoder_params.compute_param_hash(bpy.context)

    # build_pipeline uploads (direct disk + upload_notify) then builds.
    dh.com.build_pipeline(data=data_bytes, param=param_bytes,
                          data_hash=data_hash, param_hash=param_hash,
                          message="direct_disk:build")
    deadline = time.time() + 90.0
    while time.time() < deadline:
        dh.facade.engine.dispatch(dh.events.PollTick())
        dh.facade.tick()
        s = dh.facade.engine.state
        if (s.activity.name == "IDLE"
                and s.solver.name in ("READY", "RESUMABLE", "FAILED")):
            break
        time.sleep(0.2)
    s = dh.facade.engine.state
    if s.solver.name == "FAILED":
        raise RuntimeError("build failed: " + repr(s.error))

    # The addon wrote the pickles straight to its canonical root.
    root = s.remote_root
    data_pickle = os.path.join(root, "data.pickle")
    param_pickle = os.path.join(root, "param.pickle")
    upload_id_txt = os.path.join(root, "upload_id.txt")
    data_hash_txt = os.path.join(root, "data_hash.txt")

    files_on_disk = (os.path.isfile(data_pickle)
                     and os.path.isfile(param_pickle)
                     and os.path.isfile(upload_id_txt))
    dh.record(
        "pickles_written_to_disk",
        files_on_disk,
        {"root": root,
         "data.pickle": os.path.isfile(data_pickle),
         "param.pickle": os.path.isfile(param_pickle),
         "upload_id.txt": os.path.isfile(upload_id_txt)},
    )

    # The on-disk pickle must equal the bytes the addon encoded this
    # run, proving the direct write happened (not a stale leftover).
    disk_match = False
    try:
        with open(data_pickle, "rb") as f:
            disk_match = (f.read() == data_bytes)
    except OSError:
        pass
    dh.record("disk_pickle_matches_payload", disk_match,
              {"payload_bytes": len(data_bytes)})

    # upload_notify carried the addon-minted id; the server stamped it
    # to disk and echoed it back, so on-disk == addon state.
    on_disk_id = ""
    try:
        with open(upload_id_txt) as f:
            on_disk_id = f.read().strip()
    except OSError:
        pass
    on_disk_hash = ""
    try:
        with open(data_hash_txt) as f:
            on_disk_hash = f.read().strip()
    except OSError:
        pass
    dh.record(
        "upload_id_round_trip",
        bool(on_disk_id) and on_disk_id == s.server_upload_id,
        {"on_disk": on_disk_id, "state": s.server_upload_id},
    )
    dh.record(
        "data_hash_round_trip",
        bool(data_hash)
        and on_disk_hash == data_hash
        and s.server_data_hash == data_hash,
        {"on_disk": on_disk_hash, "state": s.server_data_hash,
         "computed": data_hash},
    )

    # End-to-end: run + fetch reads the output frames back off disk via
    # the direct receive_data path and applies them to the mesh.
    dh.run_and_wait(timeout=90.0)
    dh.force_frame_query(expected_frames=1, timeout=30.0)
    dh.settle_idle(timeout=15.0)
    applied, total = dh.fetch_and_drain()
    pc2_path = dh.find_pc2_for(plane)
    dh.record(
        "round_trip_frames_applied",
        applied > 0 and total > 0 and applied >= total,
        {"applied": applied, "total": total, "pc2": pc2_path},
    )

except Exception as exc:
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["errors"].append(traceback.format_exc())
'''


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 180.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
