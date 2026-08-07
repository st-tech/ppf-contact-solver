# File: scenarios/bl_timeline_statistics.py
# Code: GitHub Copilot
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Solver-produced per-object timeline statistics, on both backends.
#
# The scenario drives one translating and spinning shell through build, run,
# fetch, local statistics caching, current-frame lookup, timeline-series lookup,
# and one-channel CSV export.
#
# Contact count is the one channel that separates the two backends, so it is
# the one the run has to gate on ``ctx.backend``. The emulated stub runs no
# contact assembly, so the channel is absent there and is asserted absent
# rather than approximated; the real solver advertises it and reports a
# positive count once the sheet lands on the floor.

from __future__ import annotations

from . import REPO_ROOT_POSIX
from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True
BACKENDS = ("emulated", "real")
KNOBS = {"PPF_EMULATED_ELASTIC": "1", "PPF_EMULATED_STEP_MS": "0"}


_DRIVER_BODY = r"""
import csv
import io
import math
import os
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = 16
BACKEND = "<<BACKEND>>"


try:
    dh = DriverHelpers(pkg, result)
    stats_cache = __import__(
        pkg + ".core.statistics_cache",
        fromlist=[
            "CHANNEL_BY_ID", "iter_scalar_records", "load_manifest",
            "manifest_object", "read_record", "scalar_value",
        ],
    )
    stats_ui = __import__(
        pkg + ".ui.dynamics.statistics",
        fromlist=[
            "STATISTICS_PT_Statistics", "_format_measure", "_format_vector",
            "_statistics_object_items", "statistics_frame_change_handler",
        ],
    )
    panels = __import__(
        pkg + ".ui.dynamics.panels",
        fromlist=["VISUALIZATION_PT_Visualization"],
    )

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=8,
        y_subdivisions=8,
        size=2.0,
        location=(0.0, 0.0, 0.0),
    )
    sheet = bpy.context.object
    sheet.name = "StatisticsSheet"
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, -3.0))
    collider = bpy.context.object
    collider.name = "StatisticsCollider"
    dh.save_blend(PROBE_DIR, "timeline_statistics.blend")
    root = dh.configure_state(
        project_name="timeline_statistics",
        frame_count=FRAME_COUNT,
        frame_rate=24,
        step_size=1.0 / 24.0,
    )

    group_api = dh.api.solver.create_group("Spinner", "SHELL")
    group_api.add(sheet.name)
    group_api.set_velocity(
        sheet.name,
        direction=(1.0, 0.0, 0.0),
        speed=0.2,
        frame=1,
        angular_axis="X",
        angular_speed=180.0,
        enable_translational=True,
        enable_angular=True,
    )
    assigned = root.object_group_0.assigned_objects[0]
    object_uuid = assigned.uuid
    static_group = dh.api.solver.create_group("Collider", "STATIC")
    static_group.add(collider.name)
    static_uuid = root.object_group_1.assigned_objects[0].uuid
    dh.api.solver.add_wall(
        position=(0.0, 0.0, -0.35),
        normal=(0.0, 0.0, 1.0),
    )

    data_bytes, param_bytes = dh.encode_payload()
    dh.connect_local(
        local_path=LOCAL_PATH,
        server_port=SERVER_PORT,
        project_name=root.state.project_name,
    )
    dh.build_and_wait(data_bytes, param_bytes, "timeline-statistics:build", timeout=120.0)
    dh.run_and_wait(timeout=120.0)
    dh.force_frame_query(expected_frames=FRAME_COUNT, timeout=60.0)
    dh.fetch_and_drain()

    manifest = stats_cache.load_manifest()
    manifest_entry = stats_cache.manifest_object(object_uuid, manifest)
    static_manifest_entry = stats_cache.manifest_object(static_uuid, manifest)
    supported = manifest_entry["supported_channels"] if manifest_entry else 0

    # A: panel placement and object identity survive the complete solver path.
    panel_order_ok = (
        panels.VISUALIZATION_PT_Visualization.bl_order == 100
        and stats_ui.STATISTICS_PT_Statistics.bl_order == 110
        and stats_ui.STATISTICS_PT_Statistics.bl_label == "Object Statistics"
    )
    dh.record(
        "A_panel_and_manifest",
        panel_order_ok
        and manifest_entry is not None
        and manifest_entry["object_name"] == sheet.name
        and manifest_entry["dynamics_type"] == "SHELL"
        and static_manifest_entry is not None
        and static_manifest_entry["object_name"] == collider.name
        and static_manifest_entry["dynamics_type"] == "STATIC",
        {
            "panel_order_ok": panel_order_ok,
            "manifest_entry": manifest_entry,
            "static_manifest_entry": static_manifest_entry,
        },
    )
    popup_items = stats_ui._statistics_object_items(None, bpy.context)
    dh.record(
        "A2_object_popup",
        len(popup_items) == 2
        and {item[0] for item in popup_items} == {object_uuid, static_uuid},
        {"popup_items": popup_items},
    )
    export_probe = __import__("types").SimpleNamespace(object_uuid=object_uuid)
    export_items = stats_ui._statistics_channel_items(export_probe, bpy.context)
    export_ids = {item[0] for item in export_items}
    dh.record(
        "A3_export_popup_matches_visible_statistics",
        "NONE" not in export_ids
        and {"LOCATION", "SURFACE_AREA", "VELOCITY", "ANGULAR_SPEED"}
        .issubset(export_ids)
        and "VOLUME" not in export_ids
        and "ROD_LENGTH" not in export_ids,
        {"export_items": export_items},
    )

    # B: geometry analysis is present and starts from a unit area-stretch ratio.
    area_rows = list(stats_cache.iter_scalar_records(object_uuid, "SURFACE_AREA"))
    stretch_rows = list(stats_cache.iter_scalar_records(object_uuid, "AREA_STRETCH"))
    valid_areas = [float(value) for _f, _t, value in area_rows if value is not None]
    valid_stretch = [float(value) for _f, _t, value in stretch_rows if value is not None]
    dh.record(
        "B_geometry_statistics",
        len(valid_areas) >= 2
        and min(valid_areas) > 0.0
        and valid_stretch
        and abs(valid_stretch[0] - 1.0) < 1e-5,
        {
            "area_count": len(valid_areas),
            "area_min": min(valid_areas) if valid_areas else None,
            "first_stretch": valid_stretch[0] if valid_stretch else None,
        },
    )
    area_record = stats_cache.read_record(object_uuid, 0)
    formatted_area = stats_ui._format_measure(area_record, "SURFACE_AREA")
    dh.record(
        "B1_measure_percentage_format",
        formatted_area.endswith("(100%)"),
        {"formatted_area": formatted_area},
    )

    static_area = [
        float(value)
        for _f, _t, value in stats_cache.iter_scalar_records(
            static_uuid, "SURFACE_AREA"
        )
        if value is not None
    ]
    static_volume = [
        float(value)
        for _f, _t, value in stats_cache.iter_scalar_records(static_uuid, "VOLUME")
        if value is not None
    ]
    dh.record(
        "B2_static_geometry_statistics",
        static_area
        and static_volume
        and abs(static_area[0] - 6.0) < 1e-4
        and abs(static_volume[0] - 1.0) < 1e-4,
        {
            "static_area": static_area[0] if static_area else None,
            "static_volume": static_volume[0] if static_volume else None,
        },
    )

    # C: linear motion analysis reports the translated center and nonzero speed.
    location_rows = list(stats_cache.iter_scalar_records(object_uuid, "LOCATION_X"))
    speed_rows = list(stats_cache.iter_scalar_records(object_uuid, "SPEED"))
    valid_locations = [float(v) for _f, _t, v in location_rows if v is not None]
    valid_speeds = [float(v) for _f, _t, v in speed_rows if v is not None]
    location_motion = (
        abs(valid_locations[-1] - valid_locations[0])
        if len(valid_locations) >= 2
        else 0.0
    )
    dh.record(
        "C_linear_motion_statistics",
        location_motion > 1e-4 and valid_speeds and max(valid_speeds) > 1e-4,
        {
            "location_motion": location_motion,
            "max_speed": max(valid_speeds) if valid_speeds else None,
        },
    )

    # D: angular velocity and normalized axis come from the solver rigid fit.
    angular_rows = list(stats_cache.iter_scalar_records(object_uuid, "ANGULAR_SPEED"))
    axis_x_rows = list(stats_cache.iter_scalar_records(object_uuid, "ANGULAR_AXIS_X"))
    axis_y_rows = list(stats_cache.iter_scalar_records(object_uuid, "ANGULAR_AXIS_Y"))
    axis_z_rows = list(stats_cache.iter_scalar_records(object_uuid, "ANGULAR_AXIS_Z"))
    angular = [(f, float(v)) for f, _t, v in angular_rows if v is not None]
    axes = {}
    for rows, component in (
        (axis_x_rows, 0),
        (axis_y_rows, 1),
        (axis_z_rows, 2),
    ):
        for frame, _time, value in rows:
            if value is not None:
                axes.setdefault(frame, [None, None, None])[component] = float(value)
    complete_axes = [
        axis for axis in axes.values() if all(value is not None for value in axis)
    ]
    axis_norms = [
        math.sqrt(sum(value * value for value in axis)) for axis in complete_axes
    ]
    axis_x_dominance = [
        abs(axis[0]) - max(abs(axis[1]), abs(axis[2])) for axis in complete_axes
    ]
    dh.record(
        "D_angular_motion_statistics",
        angular
        and max(value for _frame, value in angular) > 1e-4
        and axis_norms
        and max(abs(value - 1.0) for value in axis_norms) < 1e-3
        and max(axis_x_dominance) > 0.5,
        {
            "max_angular_speed": max((v for _f, v in angular), default=None),
            "axis_norms": axis_norms[:5],
            "axis_x_dominance": axis_x_dominance[:5],
        },
    )

    # E: current-frame lookup reads the matching solver-produced cache row.
    root.state.statistics_object_uuid = object_uuid
    scene = bpy.context.scene
    scene.frame_set(scene.frame_end)
    solver_frame = scene.frame_current - 1
    record = stats_cache.read_record(object_uuid, solver_frame)
    current_speed = stats_cache.scalar_value(record, "ANGULAR_SPEED")
    handler_registered = any(
        getattr(handler, "__name__", "") == "statistics_frame_change_handler"
        for handler in bpy.app.handlers.frame_change_post
    )
    stats_ui.statistics_frame_change_handler(scene)
    dh.record(
        "E_current_frame",
        current_speed is not None and handler_registered,
        {
            "solver_frame": solver_frame,
            "current_speed": current_speed,
            "cache_count": len(angular_rows),
            "frame_handler": handler_registered,
        },
    )
    velocity_text = stats_ui._format_vector(
        record, ("VELOCITY_X", "VELOCITY_Y", "VELOCITY_Z")
    )
    dh.record(
        "E2_vector_display_format",
        velocity_text.startswith("[")
        and "]" in velocity_text
        and velocity_text.count(",") == 2,
        {"velocity_text": velocity_text},
    )

    # F: one selected channel exports exactly frame,time_s,value.
    csv_path = os.path.join(os.path.dirname(PROBE_DIR), "statistics_angular_speed.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    export_result = bpy.ops.solver.export_statistics_csv(
        "EXEC_DEFAULT",
        filepath=csv_path,
        object_uuid=object_uuid,
        channel="ANGULAR_SPEED",
    )
    with open(csv_path, newline="", encoding="utf-8") as file:
        csv_rows = list(csv.reader(file))
    csv_shape_ok = (
        csv_rows
        and csv_rows[0] == ["frame", "time_s", "value"]
        and all(len(row) == 3 for row in csv_rows)
    )
    csv_values = [float(row[2]) for row in csv_rows[1:] if row[2]]
    expected_angular = [value for _frame, value in angular]
    scalar_matches = (
        len(csv_values) == len(expected_angular)
        and all(
            abs(actual - expected) < 1e-7
            for actual, expected in zip(
                csv_values, expected_angular, strict=True
            )
        )
    )
    dh.record(
        "F_scalar_csv_export",
        "FINISHED" in export_result
        and csv_shape_ok
        and scalar_matches
        and max(csv_values) > 1e-4,
        {
            "result": list(export_result),
            "header": csv_rows[0] if csv_rows else None,
            "row_count": len(csv_rows),
            "value_count": len(csv_values),
            "scalar_matches": scalar_matches,
        },
    )
    vector_csv_path = os.path.join(
        os.path.dirname(PROBE_DIR), "statistics_velocity.csv"
    )
    vector_result = bpy.ops.solver.export_statistics_csv(
        "EXEC_DEFAULT",
        filepath=vector_csv_path,
        object_uuid=object_uuid,
        channel="VELOCITY",
    )
    with open(vector_csv_path, newline="", encoding="utf-8") as file:
        vector_rows = list(csv.reader(file))
    vector_values = [row[2] for row in vector_rows[1:] if row[2]]
    velocity_channels = [
        list(stats_cache.iter_scalar_records(object_uuid, channel))
        for channel in ("VELOCITY_X", "VELOCITY_Y", "VELOCITY_Z")
    ]
    expected_vectors = []
    for entries in zip(*velocity_channels, strict=True):
        values = [entry[2] for entry in entries]
        if all(value is not None for value in values):
            expected_vectors.append([float(value) for value in values])
    parsed_vectors = [
        [float(component) for component in value[1:-1].split(",")]
        for value in vector_values
    ]
    vectors_match = (
        len(parsed_vectors) == len(expected_vectors)
        and all(
            max(abs(a - b) for a, b in zip(actual, expected, strict=True))
            < 1e-7
            for actual, expected in zip(
                parsed_vectors, expected_vectors, strict=True
            )
        )
    )
    dh.record(
        "F2_vector_csv_export",
        "FINISHED" in vector_result
        and vector_values
        and all(
            value.startswith("[")
            and value.endswith("]")
            and value.count(",") == 2
            for value in vector_values
        )
        and vectors_match,
        {
            "result": list(vector_result),
            "values": vector_values[:3],
            "vectors_match": vectors_match,
        },
    )

    # G: contact count is solver-produced. The real sheet falls onto the
    # analytic floor and must report positive contacts. The emulator does not
    # execute contact assembly and must report the channel unavailable.
    contact_bit = stats_cache.CHANNEL_BY_ID["CONTACT_COUNT"][3]
    contact_rows = list(
        stats_cache.iter_scalar_records(object_uuid, "CONTACT_COUNT")
    )
    contact_values = [
        int(value) for _frame, _time, value in contact_rows if value is not None
    ]
    dh.record(
        "G_contact_count_available",
        (
            supported & (1 << contact_bit) != 0
            and len(contact_values) == len(contact_rows)
            and any(value > 0 for value in contact_values)
            if BACKEND == "real"
            else supported & (1 << contact_bit) == 0 and not contact_values
        ),
        {
            "supported_channels": supported,
            "contact_bit": contact_bit,
            "contact_values": contact_values,
        },
    )

    # H: the cache is a separate persistent file beside PC2 and reloads from
    # disk after the .blend is saved and reopened.
    stats_path = stats_cache._object_path(object_uuid)
    pc2_path = dh.find_pc2_for(sheet)
    saved_speed = current_speed
    bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
    bpy.ops.wm.open_mainfile(filepath=bpy.data.filepath)
    reloaded_manifest = stats_cache.load_manifest()
    reloaded_entry = stats_cache.manifest_object(object_uuid, reloaded_manifest)
    reloaded_record = stats_cache.read_record(object_uuid, solver_frame)
    reloaded_speed = stats_cache.scalar_value(reloaded_record, "ANGULAR_SPEED")
    dh.record(
        "H_cache_survives_reopen",
        os.path.isfile(stats_path)
        and os.path.dirname(stats_path) == os.path.dirname(pc2_path)
        and not stats_path.endswith(".pc2")
        and reloaded_entry is not None
        and reloaded_speed is not None
        and abs(float(reloaded_speed) - float(saved_speed)) < 1e-7,
        {
            "stats_path": stats_path,
            "pc2_path": pc2_path,
            "reloaded_entry": reloaded_entry,
            "saved_speed": saved_speed,
            "reloaded_speed": reloaded_speed,
        },
    )

    # I: Clear Local Animation removes the separate statistics artifacts with
    # the PC2 cache, so reopening cannot show stale values for a cleared run.
    animation = __import__(
        pkg + ".core.animation", fromlist=["clear_animation_data"]
    )
    manifest_path = stats_cache._manifest_path()
    animation.clear_animation_data(bpy.context)
    dh.record(
        "I_clear_animation_removes_statistics",
        not os.path.exists(stats_path)
        and not os.path.exists(manifest_path)
        and not os.path.exists(pc2_path),
        {
            "stats_exists": os.path.exists(stats_path),
            "manifest_exists": os.path.exists(manifest_path),
            "pc2_exists": os.path.exists(pc2_path),
        },
    )

    # J-L cover the cache against a malformed statistics payload. Every
    # value below reaches the cache the same way a solver-produced one does,
    # through the CBOR decoders, so these exercise the guards on the real
    # read path. I above emptied the cache directory, so these install their
    # own manifest without disturbing the frames the earlier subtests read.
    cbor2 = stats_cache._cbor2()
    GUARD_UUID = "22222222-3333-4444-5555-666666666666"
    GUARD_MASK = (1 << len(stats_cache.CHANNELS)) - 1

    def guard_manifest(object_uuid):
        return cbor2.dumps({
            "version": stats_cache.STATISTICS_VERSION,
            "kind": stats_cache.MANIFEST_KIND,
            "payload": {"objects": [{
                "object_index": 0,
                "object_uuid": object_uuid,
                "supported_channels": GUARD_MASK,
            }]},
        })

    def guard_frame(solver_frame, **overrides):
        record = {
            "object_index": 0, "valid_channels": GUARD_MASK,
            "location": [0.0, 0.0, 0.0], "volume": 1.0, "surface_area": 1.0,
            "area_stretch": 1.0, "rod_length": 0.0, "length_stretch": 1.0,
            "velocity": [0.0, 0.0, 0.0], "speed": 0.0,
            "acceleration": [0.0, 0.0, 0.0], "acceleration_magnitude": 0.0,
            "angular_velocity": [0.0, 0.0, 0.0], "angular_speed": 0.0,
            "angular_axis": [0.0, 0.0, 0.0], "volume_stretch": 1.0,
            "contact_count": 0,
        }
        record.update(overrides)
        return cbor2.dumps({
            "version": stats_cache.STATISTICS_VERSION,
            "kind": stats_cache.FRAME_KIND,
            "payload": {
                "solver_frame": solver_frame,
                "time_seconds": 0.0,
                "objects": [record],
            },
        })

    def guard_error(call):
        # Report the exception type so a subtest that merely raises the
        # wrong class cannot read as a pass.
        try:
            call()
        except stats_cache.StatisticsCacheError as exc:
            return f"StatisticsCacheError: {exc}"
        except Exception as exc:
            return f"ESCAPED {type(exc).__name__}: {exc}"
        return "NO ERROR"

    # J: the frame index decoded from a blob picks the byte offset the cache
    # file is zero-filled up to, so it is bounded by the frame the caller
    # fetched. An in-bounds frame still writes.
    guard_manifest_blob = guard_manifest(GUARD_UUID)
    guard_installed = stats_cache.install_manifest(guard_manifest_blob)
    guard_path = stats_cache._object_path(GUARD_UUID)
    past_frame = guard_error(lambda: stats_cache.write_frame_blob(
        guard_frame(200000), guard_installed, max_solver_frame=0
    ))
    past_size = os.path.getsize(guard_path) if os.path.exists(guard_path) else 0
    in_bounds = stats_cache.write_frame_blob(
        guard_frame(2), guard_installed, max_solver_frame=4
    )
    bool_index = guard_error(lambda: stats_cache.write_frame_blob(
        guard_frame(True), guard_installed, max_solver_frame=4
    ))
    dh.record(
        "J_frame_index_is_bounded_by_the_fetched_frame",
        past_frame.startswith("StatisticsCacheError")
        and past_size <= stats_cache._HEADER.size + stats_cache._RECORD.size
        and in_bounds == 2
        and bool_index.startswith("StatisticsCacheError"),
        {
            "past_frame": past_frame,
            "cache_size_after_reject": past_size,
            "in_bounds_frame": in_bounds,
            "bool_index": bool_index,
        },
    )

    # K: a decoded field that is not a number, or one too wide for the
    # record, is named rather than escaping as ValueError/TypeError/
    # struct.error; and a cache file that shrinks while it is being walked
    # is reported instead of failing inside struct.unpack.
    malformed = {
        "location_component": guard_error(lambda: stats_cache.write_frame_blob(
            guard_frame(3, location=["x", 0.0, 0.0]), guard_installed,
            max_solver_frame=4,
        )),
        "volume_is_a_map": guard_error(lambda: stats_cache.write_frame_blob(
            guard_frame(3, volume={}), guard_installed, max_solver_frame=4,
        )),
        "speed_is_out_of_range": guard_error(lambda: stats_cache.write_frame_blob(
            guard_frame(3, speed=10**400), guard_installed, max_solver_frame=4,
        )),
        "contact_count_too_wide": guard_error(lambda: stats_cache.write_frame_blob(
            guard_frame(3, contact_count=2**80), guard_installed,
            max_solver_frame=4,
        )),
    }
    # Fill the cache well past one buffered read so a later truncation is
    # visible to the reader rather than served from bytes it already holds.
    #
    # The buffer this has to outrun is the READER's, and iter_scalar_records
    # opens with default buffering, so CPython sizes it from st_blksize when
    # the platform reports one and from io.DEFAULT_BUFFER_SIZE when it does
    # not. st_blksize is POSIX-only and raises AttributeError on Windows
    # rather than returning a default, so the fallback here is that same
    # constant: anything smaller would leave the whole file inside one buffered
    # read, and the truncation below would be served from bytes the reader
    # already held instead of being seen.
    blksize = max(
        getattr(
            os.stat(os.path.dirname(guard_path)),
            "st_blksize",
            io.DEFAULT_BUFFER_SIZE,
        ),
        8192,
    )
    last_frame = (3 * blksize) // stats_cache._RECORD.size
    stats_cache.write_frame_blob(
        guard_frame(last_frame), guard_installed, max_solver_frame=last_frame
    )
    keep_bytes = (
        stats_cache._HEADER.size
        + (2 * blksize // stats_cache._RECORD.size) * stats_cache._RECORD.size
        + stats_cache._RECORD.size // 2
    )

    def walk_while_truncating():
        rows = 0
        for _row in stats_cache.iter_scalar_records(GUARD_UUID, "VOLUME"):
            if rows == 0:
                os.truncate(guard_path, keep_bytes)
            rows += 1

    truncated = guard_error(walk_while_truncating)
    dh.record(
        "K_malformed_statistics_are_named_not_escaped",
        all(
            outcome.startswith("StatisticsCacheError")
            for outcome in malformed.values()
        )
        and truncated.startswith("StatisticsCacheError"),
        {"malformed": malformed, "truncated_read": truncated},
    )

    # L: the UUID is decoded data used to name a file, so every path that
    # opens one rejects a UUID that is not a plain filename component, and
    # nothing is created for it. A real UUID is a uuid4 string, which the
    # rule admits unchanged.
    escape_uuid = "..\\statistics-escape"
    cache_dir = os.path.dirname(guard_path)
    escape_manifest = stats_cache.install_manifest(guard_manifest(escape_uuid))
    before_names = set(os.listdir(cache_dir))
    escape_write = guard_error(lambda: stats_cache.write_frame_blob(
        guard_frame(0), escape_manifest, max_solver_frame=0
    ))
    escape_read = guard_error(lambda: stats_cache.read_record(escape_uuid, 0))
    plain_uuid = stats_cache._object_path(GUARD_UUID)
    new_names = set(os.listdir(cache_dir)) - before_names
    dh.record(
        "L_uuid_must_be_a_filename_component",
        escape_write.startswith("StatisticsCacheError")
        and escape_read.startswith("StatisticsCacheError")
        and not new_names
        and os.path.basename(plain_uuid) == GUARD_UUID + ".stats",
        {
            "escape_write": escape_write,
            "escape_read": escape_read,
            "new_names": sorted(new_names),
            "plain_uuid_path": os.path.basename(plain_uuid),
        },
    )
    stats_cache.clear_statistics_cache()

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE.replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX).replace(
            "<<SERVER_PORT>>", str(ctx.server_port)
        ).replace("<<BACKEND>>", ctx.backend)
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
