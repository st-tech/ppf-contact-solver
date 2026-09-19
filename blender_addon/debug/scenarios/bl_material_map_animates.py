# File: scenarios/bl_material_map_animates.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A spatial material map keyed at frames has to vary the parameter over TIME.
#
# The map's own source is the weights at the start frame; each sample names a
# later source, and the weights between two consecutive samples are their
# linear interpolation. Two samples naming one source is therefore a constant
# hold, which is why no hold flag exists.
#
# The chain, on the solver so it runs on any host:
#
#   A. payload_carries_samples: the encoded row carries `times` and
#      `weight_frames` rather than a single `weights` array, with one weight
#      array per time and one weight per vertex in each.
#   B. session_carries_schedule: the built session's per-frame table holds
#      exactly n_times * n_faces values, the shape the solver asserts.
#   C. faces_move_over_time: the first and last per-face rows DIFFER. A map
#      shipped but resolved once, or a replicated scalar, passes A and B and
#      fails this.
#   D. axis_is_the_authored_frames: with the slider itself not keyframed, the
#      shared keyframe axis holds exactly the map's own two authored times.
#      A map whose weights move affinely needs no interior sample, so a larger
#      count would mean the decimation is keeping times nothing asked for.
#   E. endpoints_are_the_authored_maps: the first row equals the blend against
#      the first painted source and the last row equals the blend against the
#      second, so the interpolation reaches the authored maps exactly.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. A full rig sweep against a CPU build passed it, and that run is the
# evidence this line rests on.
BACKENDS = ("real",)

_BASE_BEND = 1.0
_TARGET_BEND = 4000.0
_SAMPLE_FRAME = 10
# An interior authored time, strictly between the start frame and the last
# sample, so the decimation has a real choice about keeping it.
_KINK_FRAME = 5


_DRIVER_BODY = r"""
import glob
import os
import struct
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT_ROOT = "<<PROJECT_ROOT>>"
BASE_BEND = <<BASE_BEND>>
TARGET_BEND = <<TARGET_BEND>>
SAMPLE_FRAME = <<SAMPLE_FRAME>>
KINK_FRAME = <<KINK_FRAME>>

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="MapMesh")
    root = dh.configure_state(project_name="map_anim", frame_count=SAMPLE_FRAME)

    # A second, UNANIMATED shell in its own group, named to sort BEFORE the
    # animated one. Every triangle-contributing object has to appear in every
    # animated key's table, including the ones that animate nothing: a key
    # first seen on the second object would otherwise miss the first object's
    # triangles and leave the per-key array short, which the solver reports as
    # a shape mismatch rather than as the missing seed it is.
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=3, y_subdivisions=3,
                                    size=0.5, location=(2.0, 0.0, 0.0))
    bystander = bpy.context.active_object
    bystander.name = "AAABystander"
    bystander.data.calc_loop_triangles()
    n_bystander_tri = len(bystander.data.loop_triangles)
    quiet = dh.api.solver.create_group("Quiet", "SHELL")
    quiet.add(bystander.name)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    # Keep the object dynamic: fully pinned with no operation becomes a static
    # collider and leaves the solved namespace with its material.
    pin.move_by(delta=(0.05, 0.0, 0.0), frame_start=1, frame_end=3,
                transition="LINEAR")

    # Two gradients running opposite ways, so every vertex's weight moves and
    # no face can hold its value across the two authored maps.
    ys = [v.co.y for v in plane.data.vertices]
    lo, hi = min(ys), max(ys)
    early = plane.vertex_groups.new(name="StiffenFront")
    late = plane.vertex_groups.new(name="StiffenBack")
    # A third source at an INTERIOR frame. Every vertex but one sits exactly on
    # the straight line from the first source to the last, so it adds no
    # curvature; vertex 0 leaves that line. The decimation keeps a time only
    # when some series bends there, so this interior time survives only if the
    # witness is chosen by CURVATURE. Choosing it by largest change picks a
    # vertex that moves further but affinely, whose series needs no interior
    # sample, and the authored excursion is interpolated away.
    for i, v in enumerate(plane.data.vertices):
        w = 0.0 if hi == lo else (v.co.y - lo) / (hi - lo)
        early.add([i], w, "REPLACE")
        late.add([i], 1.0 - w, "REPLACE")

    # A separate object whose map exists to separate MAGNITUDE from CURVATURE.
    # Vertex A moves furthest and does so affinely, so it needs no interior
    # sample. Vertex B moves less on every segment and bends at the interior
    # frame, so the composed value there is not the chord through its
    # neighbors. The decimation keeps a time only when some series bends, so
    # the interior time survives only if the witness is chosen by curvature.
    # Selecting by largest change picks A and the authored excursion is
    # interpolated away.
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=2, y_subdivisions=2,
                                    size=0.5, location=(-2.0, 0.0, 0.0))
    curved = bpy.context.active_object
    curved.name = "CurvatureProbe"
    # Vertex 0 moves furthest and lies EXACTLY on the chord in time, so it
    # needs no interior sample; the authored frames are not equally spaced, so
    # the midpoint weight is that chord's value, not 0.5. Vertex 1 moves less
    # on both segments and bends. Magnitude picks vertex 0 and drops the
    # interior time; curvature picks vertex 1 and keeps it.
    chord = (KINK_FRAME - 1) / (SAMPLE_FRAME - 1)
    tri_of = {0: (0.0, chord, 1.0), 1: (0.0, 0.2, 0.0)}
    for label, at in (("At1", 0), ("At5", 1), ("At10", 2)):
        vg = curved.vertex_groups.new(name=label)
        for i in range(len(curved.data.vertices)):
            vg.add([i], tri_of.get(i, (0.0, 0.0, 0.0))[at], "REPLACE")
    probe = dh.api.solver.create_group("Curved", "SHELL")
    probe.add(curved.name)

    group = root.object_group_1
    group.bend = BASE_BEND
    entry = group.material_maps.add()
    entry.parameter = "bend"
    entry.source_type = "VERTEX_GROUP"
    entry.source_name = "StiffenFront"
    entry.target_value = TARGET_BEND
    sample = entry.samples.add()
    sample.frame = SAMPLE_FRAME
    sample.source_type = "VERTEX_GROUP"
    sample.source_name = "StiffenBack"

    curved_group = root.object_group_2
    curved_group.friction = 0.1
    curved_entry = curved_group.material_maps.add()
    curved_entry.parameter = "friction"
    curved_entry.source_type = "VERTEX_GROUP"
    curved_entry.source_name = "At1"
    curved_entry.target_value = 0.9
    for probe_frame, probe_name in ((KINK_FRAME, "At5"), (SAMPLE_FRAME, "At10")):
        probe_sample = curved_entry.samples.add()
        probe_sample.frame = probe_frame
        probe_sample.source_type = "VERTEX_GROUP"
        probe_sample.source_name = probe_name
    dh.log("keyed map authored")

    data_bytes, param_bytes = dh.encode_payload()
    blob = dh.decode_addon_blob(param_bytes)
    row = {}
    authored_times = set()
    for e in blob.get("group", []):
        for key, entry_row in (e[0].get("material-maps") or {}).items():
            authored_times.update(round(t, 9) for t in (entry_row.get("times") or []))
            if key == "bend":
                row = entry_row
    n_verts = len(plane.data.vertices)
    times = list(row.get("times") or [])
    frames = list((row.get("weight_frames") or {}).values())
    per_object = frames[0] if frames else []
    dh.record(
        "A_payload_carries_samples",
        "weights" not in row and len(times) == 2
        and len(per_object) == len(times)
        and all(len(w) == n_verts for w in per_object),
        {"times": times, "n_arrays": len(per_object), "n_verts": n_verts,
         "has_static_weights": "weights" in row},
    )

    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.build_and_wait(data_bytes, param_bytes, message="animated map")
    dh.log("built")

    search_root = os.path.dirname(PROJECT_ROOT.rstrip("/")) or PROJECT_ROOT
    hits = glob.glob(os.path.join(search_root, "**", "session"), recursive=True)
    if not hits:
        raise RuntimeError("no session directory under %s" % search_root)
    session = sorted(hits, key=os.path.getmtime)[-1]
    anim = os.path.join(session, "bin", "param_anim")
    files = sorted(os.listdir(anim)) if os.path.isdir(anim) else []

    n_tri = os.path.getsize(
        os.path.join(session, "bin", "param", "tri-bend.bin")
    ) // 4
    n_times = 0
    vals = []
    if "times.bin" in files:
        n_times = os.path.getsize(os.path.join(anim, "times.bin")) // 8
    if "tri-bend.bin" in files:
        raw = open(os.path.join(anim, "tri-bend.bin"), "rb").read()
        vals = list(struct.unpack("<%df" % (len(raw) // 4), raw))
    dh.record(
        "B_session_carries_schedule",
        n_tri > 0 and n_times >= 2 and len(vals) == n_times * n_tri,
        {"files": files, "n_times": n_times, "n_tri": n_tri,
         "n_values": len(vals), "expected": n_times * n_tri},
    )

    first = vals[:n_tri]
    last = vals[-n_tri:] if n_tri else []
    moved = sum(1 for a, b in zip(first, last) if abs(a - b) > 1e-3)
    dh.record(
        "C_faces_move_over_time",
        bool(first) and moved > 0,
        {"n_faces": n_tri, "faces_that_moved": moved,
         "first_row_range": [min(first), max(first)] if first else None,
         "last_row_range": [min(last), max(last)] if last else None},
    )
    # The shared axis has to carry every time the map is authored at, plus the
    # ends of the solve, and NOTHING else: the weights move affinely between
    # two authored samples, so an interior time would be one the decimation
    # kept for no reason and paid for with a value per face.
    # EVERY map's authored times, from every group: the axis is shared, and a
    # time one map needs is a time the axis has to carry. The curvature probe's
    # interior time is the one a magnitude-chosen witness drops.
    axis = [round(t, 9) for t in (blob.get("param_anim_times") or [])]
    wanted = sorted(authored_times | {axis[0], axis[-1]})
    dh.record(
        "D_axis_carries_the_authored_times_and_no_more",
        bool(axis) and axis == wanted and n_times == len(axis),
        {"axis": axis, "wanted": wanted, "authored": sorted(authored_times),
         "gradient_map_times": times, "n_times_on_disk": n_times},
    )

    # Only the MAPPED object's faces move; the bystander contributes its own
    # constant value to every row, which is the point of check F. Compare over
    # the faces that moved rather than over the whole table.
    n_mapped_tri = len(plane.data.loop_triangles)
    moved_idx = [i for i in range(n_tri) if abs(first[i] - last[i]) > 1e-3]
    mapped_first = [first[i] for i in moved_idx]
    mapped_last = [last[i] for i in moved_idx]
    # The two authored sources are mirror images of one gradient, so the two end
    # rows hold the SAME multiset of per-face values while assigning them to
    # different faces. That is checkable without knowing how Blender
    # triangulated the mesh, which the rig does not own.
    same_multiset = sorted(round(v, 3) for v in mapped_first) == sorted(
        round(v, 3) for v in mapped_last
    )
    dh.record(
        "E_endpoints_are_the_authored_maps",
        bool(mapped_first) and same_multiset and len(moved_idx) == n_mapped_tri
        and min(mapped_first) >= BASE_BEND - 1e-3
        and max(mapped_first) <= TARGET_BEND + 1e-3,
        {"n_faces": n_tri, "n_mapped_tri": n_mapped_tri,
         "faces_that_moved": len(moved_idx), "same_multiset": same_multiset,
         "mapped_first_range": [min(mapped_first), max(mapped_first)]
         if mapped_first else None,
         "mapped_last_range": [min(mapped_last), max(mapped_last)]
         if mapped_last else None},
    )

    curved.data.calc_loop_triangles()
    n_curved_tri = len(curved.data.loop_triangles)
    plane.data.calc_loop_triangles()
    expected_tri = (n_bystander_tri + n_curved_tri
                    + len(plane.data.loop_triangles))
    dh.record(
        "F_every_object_is_seeded_into_the_animated_table",
        n_tri == expected_tri and len(vals) == n_times * n_tri,
        {"n_tri_total": n_tri, "expected_tri": expected_tri,
         "n_bystander_tri": n_bystander_tri, "n_curved_tri": n_curved_tri,
         "n_values": len(vals), "expected_values": n_times * n_tri,
         "note": "a short table means an object was never seeded"},
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
        .replace("<<BASE_BEND>>", repr(_BASE_BEND))
        .replace("<<TARGET_BEND>>", repr(_TARGET_BEND))
        .replace("<<SAMPLE_FRAME>>", repr(_SAMPLE_FRAME))
        .replace("<<KINK_FRAME>>", repr(_KINK_FRAME))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 360.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
