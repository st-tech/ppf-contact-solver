# File: scenarios/bl_mcp_statistics.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Per-object solver statistics over MCP, against a real Blender.
#
# The three statistics tools read a cache that the frame-fetch path writes, so
# a worker that runs no solve has nothing for them to read. Both halves of
# that are covered here: what the tools answer with no cache on disk, which
# must be a refusal naming the missing data rather than an empty or
# zero-filled record, and what they answer with a cache present, which the
# driver synthesizes through the cache module's own writers in the format the
# host gate ``addon_host_tests/_statistics_cache_.py`` pins.
#
# ``core.statistics_cache`` binds ``get_pc2_dir`` at import, so the driver
# rebinds that module attribute to a private directory for the length of the
# run and restores it afterwards, which is the redirect the host gate uses.
# The unredirected root is the shared temporary ``data`` directory the PC2
# caches live in, and another worker on the same host writes there, so the
# redirect is what makes both halves deterministic.
#
# Assertions:
#   A. ``missing_cache_refuses_listing`` -- with no manifest on disk,
#      list_statistics_objects answers isError and names the missing
#      statistics, carrying neither an object list nor a channel catalog.
#   B. ``missing_cache_refuses_record`` -- get_object_statistics refuses the
#      same way, with no channel or time field standing in for the record.
#   C. ``missing_cache_refuses_series`` -- get_object_statistics_series
#      refuses the same way, with no samples list.
#   D. ``listing_reports_channels`` -- with a manifest installed, the listing
#      names the object, its recorded name and dynamics type, the channel ids
#      measured for it, the start frame the addon's own resolver reports, and
#      a channel catalog whose entries each carry an id, a label and a unit.
#   E. ``record_round_trips`` -- get_object_statistics returns the values the
#      cache holds for a recorded frame, and a null value for a supported
#      channel that frame did not record.
#   F. ``unrecorded_frame_is_refused`` -- a frame the run never wrote is
#      refused by name, both for a frame whose record bytes exist as zero
#      fill and for one past the end of the file.
#   G. ``series_round_trips_and_windows`` -- the series carries one sample per
#      recorded frame in frame order, and the inclusive window bounds drop the
#      recorded frames outside them.
#   H. ``unknown_target_is_refused`` -- an unknown object name, an unknown
#      channel id, and a channel the object does not measure are each refused
#      with a message naming what was asked for.

from __future__ import annotations

from . import _mcp_lib as ml
from . import _runner as r

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. It drives the addon's in-process MCP server and never asks the solver to
# step, so nothing in it is backend-specific. A rig run against a CPU build
# passed it, and that run is the evidence this line rests on.
BACKENDS = ("real",)

NEEDS_BLENDER = True

# macOS GitHub-hosted runners block loopback HTTP from urllib to Blender's
# in-process MCP server, so the rig does not select this scenario there.
# Declaring it here rather than returning a pass from run() keeps a
# scenario that never executed from being counted as one that passed.
PLATFORMS = ("linux", "win32")


_DRIVER_BODY = r"""
import shutil
import tempfile

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

# The channels the synthesized run measures, and the mask bits they occupy in
# the cache's channel catalog. A solid measures a volume and a rod a length,
# so a real manifest carries a subset like this one rather than every channel.
SUPPORTED_IDS = ["LOCATION_X", "LOCATION_Y", "LOCATION_Z", "SPEED", "CONTACT_COUNT"]
SUPPORTED_MASK = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 11) | (1 << 24)
SPEED_BIT = 1 << 11
RECORDED_NAME = "StatsProbeAtRunTime"
DYNAMICS_TYPE = "SHELL"
MISSING_NAME = "NoSuchStatisticsObject"
LAST_SOLVER_FRAME = 3

# Solver frame, valid mask, location, speed, contact count, seconds. Solver
# frame 2 is deliberately absent, so the write of frame 3 leaves its record
# bytes present and zero, which is the state a zero-filled answer would hide.
# Every float here is exact in float32, which is the width the record packs
# them at, so the values compare exactly after the round trip.
FRAMES = (
    (0, SUPPORTED_MASK, (0.5, -1.25, 2.0), 3.5, 7, 0.0),
    (1, SUPPORTED_MASK, (0.75, -1.0, 2.5), 4.25, 9, 0.125),
    (3, SUPPORTED_MASK & ~SPEED_BIT, (1.5, 0.25, 3.0), 9.0, 12, 0.375),
)

stats = None
original_get_pc2_dir = None
cache_dir = ""

try:
    stats = __import__(pkg + ".core.statistics_cache", fromlist=["load_manifest"])
    cbor2 = __import__(pkg + ".core.module", fromlist=["get_cbor2"]).get_cbor2()
    encoder = __import__(pkg + ".core.encoder", fromlist=["resolve_start_frame"])
    models_groups = __import__(pkg + ".models.groups", fromlist=["get_addon_data"])
    registry = __import__(
        pkg + ".core.uuid_registry", fromlist=["get_or_create_object_uuid"]
    )

    cache_dir = tempfile.mkdtemp(prefix="ppf_stats_scenario_")
    original_get_pc2_dir = stats.get_pc2_dir
    stats.get_pc2_dir = lambda: cache_dir
    result["phases"].append((time.time(), "cache_dir=%s" % cache_dir))

    # A plain mesh is enough: the tools key on the object's UUID, and nothing
    # here reaches the solver pipeline.
    scene = bpy.context.scene
    mesh = bpy.data.meshes.new("StatsProbeMesh")
    mesh.from_pydata([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [], [(0, 1, 2)])
    mesh.update()
    probe = bpy.data.objects.new("StatsProbe", mesh)
    scene.collection.objects.link(probe)
    object_uuid = registry.get_or_create_object_uuid(probe)
    if not object_uuid:
        raise RuntimeError("the probe object was not assigned a UUID")
    if bpy.data.objects.get(MISSING_NAME) is not None:
        raise RuntimeError("%s is in the scene, so it cannot test a miss" % MISSING_NAME)
    expected_start = encoder.resolve_start_frame(models_groups.get_addon_data(scene).state)

    port = mcp_alloc_free_port()
    actual_port, mcp_mod = mcp_start_server(pkg, port)
    url = "http://127.0.0.1:%d/mcp" % actual_port
    result["phases"].append((time.time(), "mcp_started port=%d" % actual_port))

    def call(name, arguments, request_id):
        # A handler that runs and fails answers with a well-formed result
        # carrying isError, so a refusal is read off the payload rather than
        # off a JSON-RPC error frame.
        payload, raw = mcp_tool(pkg, url, name, arguments, request_id=request_id)
        return payload, bool(raw.get("isError")), payload.get("message") or ""

    # ----- A. the listing with nothing on disk --------------------
    listing, listing_error, listing_message = call("list_statistics_objects", {}, 11)
    mcp_check(
        result, "A_missing_cache_refuses_listing",
        listing_error
        and listing.get("status") == "error"
        and "statistics" in listing_message.lower()
        and "rerun" in listing_message.lower()
        and "objects" not in listing
        and "object_count" not in listing
        and "channel_catalog" not in listing,
        {
            "isError": listing_error,
            "message": listing_message,
            "payload_keys": sorted(listing),
        },
    )

    # ----- B. one record with nothing on disk ---------------------
    record, record_error, record_message = call(
        "get_object_statistics",
        {"object_name": probe.name, "frame": expected_start},
        12,
    )
    mcp_check(
        result, "B_missing_cache_refuses_record",
        record_error
        and record.get("status") == "error"
        and "statistics" in record_message.lower()
        and "rerun" in record_message.lower()
        and "channels" not in record
        and "time_s" not in record
        and "unsupported_channels" not in record,
        {
            "isError": record_error,
            "message": record_message,
            "payload_keys": sorted(record),
        },
    )

    # ----- C. one series with nothing on disk ---------------------
    series, series_error, series_message = call(
        "get_object_statistics_series",
        {"object_name": probe.name, "channel": "SPEED"},
        13,
    )
    mcp_check(
        result, "C_missing_cache_refuses_series",
        series_error
        and series.get("status") == "error"
        and "statistics" in series_message.lower()
        and "rerun" in series_message.lower()
        and "samples" not in series
        and "sample_count" not in series,
        {
            "isError": series_error,
            "message": series_message,
            "payload_keys": sorted(series),
        },
    )

    # ----- synthesize the cache the frame-fetch path would write --
    def manifest_blob():
        return cbor2.dumps({
            "version": stats.STATISTICS_VERSION,
            "kind": stats.MANIFEST_KIND,
            "payload": {
                "objects": [
                    {
                        "object_index": 0,
                        "object_uuid": object_uuid,
                        "object_name": RECORDED_NAME,
                        "dynamics_type": DYNAMICS_TYPE,
                        "supported_channels": SUPPORTED_MASK,
                    }
                ]
            },
        })

    def frame_blob(solver_frame, valid, location, speed, contact_count, seconds):
        entry = {
            "object_index": 0,
            "valid_channels": valid,
            "location": list(location),
            "velocity": [0.0, 0.0, 0.0],
            "acceleration": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
            "angular_axis": [0.0, 0.0, 0.0],
            "speed": speed,
            "contact_count": contact_count,
        }
        return cbor2.dumps({
            "version": stats.STATISTICS_VERSION,
            "kind": stats.FRAME_KIND,
            "payload": {
                "solver_frame": solver_frame,
                "time_seconds": seconds,
                "objects": [entry],
            },
        })

    manifest = stats.install_manifest(manifest_blob())
    for frame_spec in FRAMES:
        stats.write_frame_blob(
            frame_blob(*frame_spec), manifest, max_solver_frame=LAST_SOLVER_FRAME
        )
    result["phases"].append((time.time(), "cache_written frames=%d" % len(FRAMES)))

    # ----- D. the listing with a manifest installed ---------------
    listing, _listing_error, _message = call("list_statistics_objects", {}, 14)
    objects = listing.get("objects") or []
    entry = objects[0] if objects else {}
    catalog = listing.get("channel_catalog") or []
    catalog_ids = [item.get("id") for item in catalog]
    catalog_is_complete = bool(catalog) and all(
        item.get("id") and item.get("label") and item.get("unit") for item in catalog
    )
    mcp_check(
        result, "D_listing_reports_channels",
        listing.get("status") == "success"
        and listing.get("object_count") == 1
        and len(objects) == 1
        and entry.get("object_uuid") == object_uuid
        and entry.get("object_name") == probe.name
        and entry.get("recorded_name") == RECORDED_NAME
        and entry.get("dynamics_type") == DYNAMICS_TYPE
        and entry.get("channels") == SUPPORTED_IDS
        and listing.get("start_frame") == expected_start
        and catalog_is_complete
        and len(set(catalog_ids)) == len(catalog_ids)
        and set(SUPPORTED_IDS) <= set(catalog_ids),
        {
            "object_count": listing.get("object_count"),
            "entry": entry,
            "start_frame": listing.get("start_frame"),
            "expected_start_frame": expected_start,
            "catalog_size": len(catalog),
            "catalog_ids": catalog_ids,
            "catalog_is_complete": catalog_is_complete,
        },
    )

    # ----- E. one recorded frame, and one partial frame -----------
    first, _first_error, _message = call(
        "get_object_statistics",
        {"object_name": probe.name, "frame": expected_start},
        15,
    )
    first_values = {
        item.get("id"): item.get("value") for item in first.get("channels") or []
    }
    partial, _partial_error, _message = call(
        "get_object_statistics",
        {"object_name": probe.name, "frame": expected_start + 3},
        16,
    )
    partial_values = {
        item.get("id"): item.get("value") for item in partial.get("channels") or []
    }
    unsupported = set(first.get("unsupported_channels") or [])
    mcp_check(
        result, "E_record_round_trips",
        first.get("status") == "success"
        and first.get("object_uuid") == object_uuid
        and first.get("object_name") == probe.name
        and first.get("dynamics_type") == DYNAMICS_TYPE
        and first.get("frame") == expected_start
        and first.get("solver_frame") == 0
        and first.get("start_frame") == expected_start
        and first.get("time_s") == 0.0
        and first_values == {
            "LOCATION_X": 0.5,
            "LOCATION_Y": -1.25,
            "LOCATION_Z": 2.0,
            "SPEED": 3.5,
            "CONTACT_COUNT": 7,
        }
        and not unsupported & set(SUPPORTED_IDS)
        and "VOLUME" in unsupported
        and partial.get("status") == "success"
        and partial.get("solver_frame") == 3
        and partial.get("time_s") == 0.375
        and partial_values == {
            "LOCATION_X": 1.5,
            "LOCATION_Y": 0.25,
            "LOCATION_Z": 3.0,
            "SPEED": None,
            "CONTACT_COUNT": 12,
        },
        {
            "first": {
                "frame": first.get("frame"),
                "solver_frame": first.get("solver_frame"),
                "start_frame": first.get("start_frame"),
                "time_s": first.get("time_s"),
                "values": first_values,
                "unsupported_count": len(unsupported),
            },
            "partial": {
                "solver_frame": partial.get("solver_frame"),
                "time_s": partial.get("time_s"),
                "values": partial_values,
            },
        },
    )

    # ----- F. frames the run never wrote --------------------------
    zero_filled_frame = expected_start + 2
    past_end_frame = expected_start + 9
    zero_filled, zero_filled_error, zero_filled_message = call(
        "get_object_statistics",
        {"object_name": probe.name, "frame": zero_filled_frame},
        17,
    )
    past_end, past_end_error, past_end_message = call(
        "get_object_statistics",
        {"object_name": probe.name, "frame": past_end_frame},
        18,
    )
    mcp_check(
        result, "F_unrecorded_frame_is_refused",
        zero_filled_error
        and zero_filled.get("status") == "error"
        and str(zero_filled_frame) in zero_filled_message
        and probe.name in zero_filled_message
        and "get_object_statistics_series" in zero_filled_message
        and "channels" not in zero_filled
        and "time_s" not in zero_filled
        and past_end_error
        and past_end.get("status") == "error"
        and str(past_end_frame) in past_end_message
        and "channels" not in past_end,
        {
            "zero_filled": {
                "frame": zero_filled_frame,
                "isError": zero_filled_error,
                "message": zero_filled_message,
                "payload_keys": sorted(zero_filled),
            },
            "past_end": {
                "frame": past_end_frame,
                "isError": past_end_error,
                "message": past_end_message,
                "payload_keys": sorted(past_end),
            },
        },
    )

    # ----- G. the series, whole and windowed ----------------------
    whole, _whole_error, _message = call(
        "get_object_statistics_series",
        {"object_name": probe.name, "channel": "SPEED"},
        19,
    )
    whole_samples = whole.get("samples") or []
    window, _window_error, _message = call(
        "get_object_statistics_series",
        {
            "object_name": probe.name,
            "channel": "CONTACT_COUNT",
            "frame_start": expected_start + 1,
            "frame_end": expected_start + 3,
        },
        20,
    )
    window_samples = window.get("samples") or []
    mcp_check(
        result, "G_series_round_trips_and_windows",
        whole.get("status") == "success"
        and (whole.get("channel") or {}).get("id") == "SPEED"
        and whole.get("start_frame") == expected_start
        and whole.get("sample_count") == 3
        and [item.get("frame") for item in whole_samples]
        == [expected_start, expected_start + 1, expected_start + 3]
        and [item.get("time_s") for item in whole_samples] == [0.0, 0.125, 0.375]
        and [item.get("value") for item in whole_samples] == [3.5, 4.25, None]
        and window.get("status") == "success"
        and window.get("sample_count") == 2
        and [item.get("frame") for item in window_samples]
        == [expected_start + 1, expected_start + 3]
        and [item.get("value") for item in window_samples] == [9, 12],
        {
            "whole": {
                "sample_count": whole.get("sample_count"),
                "samples": whole_samples,
                "channel": whole.get("channel"),
            },
            "window": {
                "sample_count": window.get("sample_count"),
                "samples": window_samples,
                "bounds": [expected_start + 1, expected_start + 3],
            },
        },
    )

    # ----- H. targets that name nothing ---------------------------
    unknown_object, unknown_object_error, unknown_object_message = call(
        "get_object_statistics",
        {"object_name": MISSING_NAME, "frame": expected_start},
        21,
    )
    unknown_channel, unknown_channel_error, unknown_channel_message = call(
        "get_object_statistics_series",
        {"object_name": probe.name, "channel": "NOT_A_CHANNEL"},
        22,
    )
    unmeasured, unmeasured_error, unmeasured_message = call(
        "get_object_statistics_series",
        {"object_name": probe.name, "channel": "VOLUME"},
        23,
    )
    mcp_check(
        result, "H_unknown_target_is_refused",
        unknown_object_error
        and unknown_object.get("status") == "error"
        and MISSING_NAME in unknown_object_message
        and "list_statistics_objects" in unknown_object_message
        and unknown_channel_error
        and unknown_channel.get("status") == "error"
        and "NOT_A_CHANNEL" in unknown_channel_message
        and "SPEED" in unknown_channel_message
        and unmeasured_error
        and unmeasured.get("status") == "error"
        and "VOLUME" in unmeasured_message
        and probe.name in unmeasured_message
        and "LOCATION_X" in unmeasured_message
        and "samples" not in unmeasured,
        {
            "unknown_object": [unknown_object_error, unknown_object_message],
            "unknown_channel": [unknown_channel_error, unknown_channel_message],
            "unmeasured_channel": [unmeasured_error, unmeasured_message],
        },
    )

    mcp_mod.stop_mcp_server()

except Exception as exc:
    result["errors"].append("%s: %s" % (type(exc).__name__, exc))
    result["errors"].append(traceback.format_exc())

finally:
    # The redirect is a module attribute on a live add-on, so it is put back
    # whatever the assertions did, and the synthesized cache goes with it.
    if stats is not None and original_get_pc2_dir is not None:
        stats.get_pc2_dir = original_get_pc2_dir
    if cache_dir:
        shutil.rmtree(cache_dir, ignore_errors=True)
"""


_DRIVER_TEMPLATE = ml.MCP_LIB + "\nimport traceback\n" + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 180.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
