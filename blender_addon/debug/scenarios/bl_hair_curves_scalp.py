# File: scenarios/bl_hair_curves_scalp.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Hair Curves rooted on a moving head (issue #146): the roots follow the
# scalp, the strands collide with it, and the result is placed through the
# head's motion. bl_hair_curves_run covers the object on its own; this one
# covers everything that comes from the hair being ATTACHED to something.
#
# The head is a STATIC icosphere that travels along +X. The hair is
# parented to it and names it as its surface, the way Blender's own Empty
# Hair sets an object up, with every root on a face of the head and the
# strand combed along the scalp a few millimeters above it, so gravity
# presses it onto the head within the first frames.
#
# bl_hair_curves_still_head runs this same driver with a head that does
# not travel. That is a different collider in the solver, not a smaller
# case of this one: a STATIC that moves is a shell of pinned vertices in
# the solved mesh, and one that stays still is the collision mesh.
#
# Coverage targets:
#   A. joining_pins_the_roots:   hair with a surface joins a ROD group
#                                and gets its Roots pin, the rest angle
#                                from its groom and a zero contact offset,
#                                with nothing asked.
#   B. capture_records_the_roots: Capture Deformation is offered for hair
#                                that only moves with its parent, and its
#                                cache holds the first segment of every
#                                strand at every frame, displaced by
#                                exactly the head's travel. With a still
#                                head nothing moves the hair, and the
#                                capture is not offered.
#   C. attached_hair_builds_as_it_is: with every root on a face of the
#                                head, the scene builds with no
#                                intersection allowance and exempts no
#                                pair. The root segment and the head are
#                                both prescribed, so they are no contact
#                                pair, and every other pair keeps contact.
#   D. roots_follow_the_head:    the first segment of every strand moves
#                                exactly as the solver's own head does at
#                                every frame, which takes the capture, the
#                                exact pin and the per-frame placement all
#                                being right. On a still head it stays
#                                exactly where the groom put it.
#   E. strands_stay_outside_the_head: no point of any strand is inside
#                                the head at any frame, and the strands
#                                have come down onto it: the closest free
#                                point gets within the distance at which
#                                the two contact gaps engage, and no
#                                closer than the head's surface. Without
#                                contact they would be centimeters inside.
#   F. playback_is_placed_by_the_head: at the last frame the evaluated
#                                points, through the hair's own world
#                                matrix, are the solver's world positions.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND: the contact and the pins are the solver's.
BACKENDS = ("real",)


_RADIUS = 0.1
_LIFT = 0.004
_N_POINTS = 6
_ARC_STEP = 0.2
_TRAVEL = 0.15
# Least angle between the meridians of two strands, in radians.
_MERIDIAN_STEP = 0.3
_FRAMES = 24


_DRIVER_BODY = r"""
import math
import os
import time
import traceback

import bmesh
from mathutils import Matrix, Vector

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
RADIUS = <<RADIUS>>
LIFT = <<LIFT>>
N_POINTS = <<N_POINTS>>
ARC_STEP = <<ARC_STEP>>
TRAVEL = <<TRAVEL>>
MERIDIAN_STEP = <<MERIDIAN_STEP>>
FRAMES = <<FRAMES>>


def evaluated_points(obj):
    deps = bpy.context.evaluated_depsgraph_get()
    deps.update()
    data = obj.evaluated_get(deps).data
    flat = np.empty(len(data.points) * 3, dtype=np.float32)
    data.attributes["position"].data.foreach_get("vector", flat)
    return flat.reshape(-1, 3)


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    hair_mod = __import__(pkg + ".core.hair_curves", fromlist=["x"])
    pc2_mod = __import__(pkg + ".core.pc2", fromlist=["x"])
    client_mod = __import__(pkg + ".core.client", fromlist=["x"])
    pcap = __import__(pkg + ".ui.dynamics.pin_capture_ops", fromlist=["x"])

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # ----- the head -------------------------------------------------
    head_mesh = bpy.data.meshes.new("Head")
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=3, radius=RADIUS)
    bm.to_mesh(head_mesh)
    bm.free()
    head = bpy.data.objects.new("HeadObj", head_mesh)
    bpy.context.collection.objects.link(head)
    MOVING = TRAVEL > 0.0
    if MOVING:
        head.location = (0.0, 0.0, 0.0)
        head.keyframe_insert("location", frame=1)
        head.location = (TRAVEL, 0.0, 0.0)
        head.keyframe_insert("location", frame=FRAMES)
        bpy.context.scene.frame_set(1)

    # The head is convex, so a point's signed distance to it is bounded
    # below by its largest signed distance to any face plane: positive
    # outside, negative only when the point is inside every plane.
    face_normals = np.array([list(p.normal) for p in head_mesh.polygons],
                            dtype=np.float64)
    face_offsets = np.array([p.normal.dot(p.center)
                             for p in head_mesh.polygons], dtype=np.float64)

    def clearance(points):
        # (..., 3) positions in the head's frame -> (...) signed distances.
        planes = np.asarray(points, dtype=np.float64) @ face_normals.T
        return (planes - face_offsets).max(axis=-1)

    # ----- the hair -------------------------------------------------
    # Strands around the crown, each rooted at a face's centroid and combed
    # outward and down along the meridian through its root. Two roots on
    # one meridian would comb two strands along the same line, one lying
    # exactly on the other, which no solver can start from; so each
    # meridian gets one strand, and neighboring meridians are kept apart.
    by_longitude = {}
    for face in head_mesh.polygons:
        center = face.center
        if not 0.75 < face.normal.z < 0.97:
            continue
        longitude = round(math.atan2(center.y, center.x), 3)
        held = by_longitude.get(longitude)
        if held is None or center.z < held.center.z:
            by_longitude[longitude] = face
    crown = []
    last_longitude = None
    for longitude in sorted(by_longitude):
        if last_longitude is None or longitude - last_longitude >= MERIDIAN_STEP:
            crown.append(by_longitude[longitude])
            last_longitude = longitude
    if crown and 2.0 * math.pi + sorted(by_longitude)[0] - last_longitude < MERIDIAN_STEP:
        crown.pop()
    groom = []
    for face in crown:
        root = Vector(face.center)
        outward = Vector((root.x, root.y, 0.0))
        axis = Vector((0.0, 0.0, 1.0)).cross(outward).normalized()
        groom.append(tuple(root))
        direction = root.normalized()
        for i in range(1, N_POINTS):
            turned = Matrix.Rotation(ARC_STEP * i, 3, axis) @ direction
            groom.append(tuple(turned * (RADIUS + LIFT)))
    groom = np.array(groom, dtype=np.float32)
    n_strands = len(crown)
    n_points = n_strands * N_POINTS
    # The Roots pin: the first segment of every strand.
    roots = [s * N_POINTS + i for s in range(n_strands) for i in (0, 1)]

    curves = bpy.data.hair_curves.new("Hair")
    curves.add_curves([N_POINTS] * n_strands)
    curves.attributes["position"].data.foreach_set("vector", groom.ravel())
    curves.surface = head
    hair = bpy.data.objects.new("HairObj", curves)
    bpy.context.collection.objects.link(hair)
    hair.parent = head

    dh.save_blend(PROBE_DIR, "hair_curves_scalp.blend")
    root_state = dh.configure_state(
        project_name="<<PROJECT>>", frame_count=FRAMES,
        gravity=(0.0, 0.0, -9.8),
    )
    root_state.state.disable_contact = False

    hair_group = dh.api.solver.create_group("Hair", "ROD")
    head_group = dh.api.solver.create_group("Head", "STATIC")
    group = root_state.object_group_0

    bpy.ops.object.select_all(action="DESELECT")
    hair.select_set(True)
    bpy.context.view_layer.objects.active = hair
    bpy.ops.object.add_objects_to_group(group_index=0)
    bpy.ops.object.select_all(action="DESELECT")
    head.select_set(True)
    bpy.context.view_layer.objects.active = head
    bpy.ops.object.add_objects_to_group(group_index=1)

    # ----- A: what joining the group set ---------------------------
    pin_names = [p.name for p in group.pin_vertex_groups]
    dh.record(
        "A_joining_pins_the_roots",
        pin_names == [f"[{hair.name}][{hair_mod.ROOTS_PIN_NAME}]"]
        and group.bend_rest_angle_source == "FROM_GEOMETRY"
        and float(group.contact_offset) == 0.0
        and hair_mod.pin_indices(hair, hair_mod.ROOTS_PIN_NAME) == roots,
        {"pins": pin_names, "rest": group.bend_rest_angle_source,
         "contact_offset": float(group.contact_offset),
         "strands": n_strands, "points": n_points},
    )

    # ----- B: the capture -------------------------------------------
    supports = bool(pcap.pin_object_supports_capture(hair))
    pin_item = group.pin_vertex_groups[0]
    if MOVING:
        ok, err = pcap.start_capture_for_pins(bpy.context, [(0, 0)])
        if not ok:
            raise RuntimeError(f"capture did not start: {err}")
        while True:
            more, aborted, error = pcap.advance_pin_capture(bpy.context)
            if aborted:
                raise RuntimeError(f"capture aborted: {error}")
            if not more:
                break
        pins_written, frames_written = pcap.finalize_pin_capture(bpy.context)
        pcap.cleanup_pin_capture(bpy.context)
        cache = pc2_mod.get_pin_anim_cache(hair, hair_mod.ROOTS_PIN_NAME)
        travel = (
            np.linalg.norm(cache[-1] - cache[0], axis=1)
            if cache is not None else np.zeros(0)
        )
        dh.record(
            "B_capture_records_the_roots",
            supports and pins_written == 1
            and cache is not None
            and list(cache.shape) == [FRAMES, len(roots), 3]
            and bool(np.all(np.abs(travel - TRAVEL) <= 1.0e-5))
            and bool(pin_item.has_captured_anim),
            {
                "supports_capture": supports,
                "pins_written": pins_written,
                "frames_written": frames_written,
                "cache_shape": list(cache.shape) if cache is not None else None,
                "travel_min": float(travel.min()) if travel.size else None,
                "travel_max": float(travel.max()) if travel.size else None,
                "has_captured_anim": bool(pin_item.has_captured_anim),
            },
        )
    else:
        # Nothing moves the head, so nothing moves the hair parented to
        # it: there is no motion to record and the capture is not offered.
        dh.record(
            "B_capture_records_the_roots",
            not supports and not bool(pin_item.has_captured_anim),
            {"supports_capture": supports,
             "has_captured_anim": bool(pin_item.has_captured_anim)},
        )

    # ----- C: the roots sit on the scalp ---------------------------
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root_state.state.project_name)
    built_solver, built_error = dh.build_verdict("hair_scalp:build")
    exemptions = list(dh.facade.engine.state.exemptions)
    dh.record(
        "C_attached_hair_builds_as_it_is",
        built_solver in ("READY", "RESUMABLE")
        and not group.allow_existing_intersection
        and not exemptions,
        {
            "solver": built_solver,
            "error": built_error[:600],
            "allow_existing_intersection": bool(
                group.allow_existing_intersection),
            "exemptions": len(exemptions),
        },
    )
    if built_solver not in ("READY", "RESUMABLE"):
        raise RuntimeError(f"build failed: {built_error}")

    dh.run_and_wait(timeout=600.0)
    run_state = dh.facade.engine.state
    if run_state.solver.name == "FAILED":
        raise RuntimeError(
            "the solve failed: "
            f"{run_state.server_error or run_state.error}"
        )
    dh.force_frame_query(expected_frames=1, timeout=15.0)
    dh.settle_idle(timeout=15.0)
    dh.fetch_and_drain(fetch_timeout=120.0, drain_timeout=120.0)
    dh.log("fetched")

    pc2_path = pc2_mod.get_pc2_path(client_mod.object_pc2_key(hair))
    pc2_arr = dh.read_pc2(pc2_path) if os.path.isfile(pc2_path) else None
    if pc2_arr is None or pc2_arr.shape[1] != n_points:
        raise RuntimeError(
            f"no usable PC2 at {pc2_path}: "
            f"{None if pc2_arr is None else pc2_arr.shape}"
        )

    # ----- D: the roots follow the head ----------------------------
    # Both caches are in the head's own frame (the hair is parented to it
    # with an identity offset), so the head's vertices and the hair's first
    # segments carry the SAME rigid offset at every frame: how far the
    # solver's head is from Blender's at the time that frame was written.
    # A solve whose steps are shortened by contact writes a frame slightly
    # after its nominal time, so that offset is not zero, and comparing the
    # roots with the solver's head rather than with Blender's is what makes
    # the test exact.
    head_pc2_path = pc2_mod.get_pc2_path(client_mod.object_pc2_key(head))
    head_arr = (
        dh.read_pc2(head_pc2_path) if os.path.isfile(head_pc2_path) else None
    )
    head_rest = np.array([list(v.co) for v in head_mesh.vertices],
                         dtype=np.float32)
    root_err = float("inf")
    rigid_err = float("inf")
    if not MOVING:
        # A still head is the collision mesh, which the solver does not
        # move and the fetch does not write, so the roots are compared
        # with the groom itself.
        rigid_err = 0.0 if head_arr is None else float("inf")
        root_err = float(np.max(np.abs(pc2_arr[:, roots, :] - groom[roots])))
    elif head_arr is not None and head_arr.shape[0] == pc2_arr.shape[0]:
        head_offset = head_arr - head_rest[None, :, :]
        mean_offset = head_offset.mean(axis=1)
        rigid_err = float(np.max(np.abs(
            head_offset - mean_offset[:, None, :])))
        root_offset = pc2_arr[:, roots, :] - groom[roots][None, :, :]
        root_err = float(np.max(np.abs(
            root_offset - mean_offset[:, None, :])))
    dh.record(
        "D_roots_follow_the_head",
        pc2_arr.shape[0] >= FRAMES - 1
        and rigid_err <= 1.0e-5 and root_err <= 2.0e-5,
        {"frames": int(pc2_arr.shape[0]), "moving": MOVING,
         "head_frames": None if head_arr is None else int(head_arr.shape[0]),
         "head_rigid_error": rigid_err, "max_root_error": root_err},
    )

    # ----- E: the strands stay outside the head --------------------
    # Both caches are in the head's frame, so in it the head stands still
    # up to the offset D measured, which is added back before measuring.
    held = set(roots)
    free = [i for i in range(n_points) if i not in held]
    in_head_frame = pc2_arr[:, free, :].astype(np.float64)
    if MOVING and head_arr is not None:
        in_head_frame = in_head_frame - (
            head_arr - head_rest[None, :, :]).mean(axis=1)[:, None, :]
    distances = clearance(in_head_frame)
    start_clearance = float(clearance(groom[free]).min())
    # Two surfaces engage inside half the sum of their contact gaps.
    engage = 0.5 * (float(group.contact_gap) + 1.0e-3 * 2.0 * RADIUS * 3 ** 0.5)
    # The closest approach is taken over the whole run, not at its end: on
    # the travelling head a strand that has landed lifts a little again as
    # the head slows down.
    dh.record(
        "E_strands_stay_outside_the_head",
        float(distances.min()) >= -1.0e-5
        and start_clearance > 2.0 * engage
        and float(distances.min()) <= engage,
        {
            "closest_any_frame": float(distances.min()),
            "closest_last_frame": float(distances[-1].min()),
            "start_clearance": start_clearance,
            "engagement_distance": engage,
        },
    )

    # ----- F: playback is placed by the head -----------------------
    start_frame = int(bpy.context.scene.frame_start)
    last = pc2_arr.shape[0] - 1
    bpy.context.scene.frame_set(start_frame + last)
    shown = evaluated_points(hair)
    world = np.array(hair.matrix_world, dtype=np.float64)
    shown_world = shown.astype(np.float64) @ world[:3, :3].T + world[:3, 3]
    head_now = np.array(head.matrix_world.translation, dtype=np.float64)
    expected_world = pc2_arr[last].astype(np.float64) + head_now
    place_err = float(np.max(np.abs(shown_world - expected_world)))
    dh.record(
        "F_playback_is_placed_by_the_head",
        len(shown) == n_points and place_err <= 1.0e-5
        and abs(float(head_now[0]) - TRAVEL * last / (FRAMES - 1)) < 0.05,
        {"placement_error": place_err,
         "head_x": float(head_now[0]), "last_index": int(last)},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver_for(ctx: r.ScenarioContext, *, travel: float,
                     project: str) -> str:
    """The driver with the head travelling *travel* along +X over the run,
    or staying where it is when *travel* is zero."""
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
        .replace("<<PROJECT>>", project)
        .replace("<<RADIUS>>", str(_RADIUS))
        .replace("<<LIFT>>", str(_LIFT))
        .replace("<<N_POINTS>>", str(_N_POINTS))
        .replace("<<ARC_STEP>>", str(_ARC_STEP))
        .replace("<<TRAVEL>>", str(float(travel)))
        .replace("<<MERIDIAN_STEP>>", str(_MERIDIAN_STEP))
        .replace("<<FRAMES>>", str(_FRAMES))
    )


def build_driver(ctx: r.ScenarioContext) -> str:
    return build_driver_for(ctx, travel=_TRAVEL, project="hair_curves_scalp")


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 1200.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
