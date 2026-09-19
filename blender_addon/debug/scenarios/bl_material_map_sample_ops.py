# File: scenarios/bl_material_map_sample_ops.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The authoring surface for a map's keyed weight sources.
#
# A map's own source is the weights at the start frame and its samples name
# later sources, so the samples are a frame-keyed collection and behave like
# every other one in the addon: seeded from the current frame, kept sorted,
# refusing a duplicate frame, and leaving a valid selection behind on removal.
#
#   A. add_seeds_the_current_frame: Add Sample takes the playhead's frame.
#   B. duplicate_frame_refused: a second Add at the same frame adds nothing and
#      reports rather than raising a traceback into the UI.
#   C. samples_stay_sorted: adding out of order leaves the collection ordered
#      by frame, which is what the encoder reads without re-sorting.
#   D. remove_keeps_the_index_valid: emptying the list leaves the index at 0
#      rather than -1, which would address the last row of an empty collection.
#   E. target_value_is_not_animatable: the field offers no keyframe button. The
#      encoder reads it once, from outside the start-frame evaluation context,
#      so a curve on it would be sampled at whatever frame the timeline sits on
#      and then ignored.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. A full rig sweep against a CPU build passed it, and that run is the
# evidence this line rests on.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="MapMesh")
    root = dh.configure_state(project_name="map_sample_ops", frame_count=10)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    group = root.object_group_0
    entry = group.material_maps.add()
    entry.parameter = "bend"
    entry.source_type = "VERTEX_GROUP"
    entry.source_name = "Stiffen"
    group.material_maps_index = 0

    # A sample at or before the start frame is refused at encode, so the
    # operator must not author one: the map's own source already IS the
    # weights at the start frame, and the very first click lands on frame 1.
    start_frame = int(root.state.frame_start) if hasattr(root.state, "frame_start") else 1
    bpy.context.scene.frame_set(start_frame)
    bpy.ops.object.ppf_add_material_map_sample(slot=0)
    dh.record(
        "A0_first_click_does_not_author_a_refused_frame",
        len(entry.samples) == 1 and entry.samples[0].frame > start_frame,
        {"start_frame": start_frame,
         "seeded": [s.frame for s in entry.samples]},
    )
    entry.samples.clear()

    bpy.context.scene.frame_set(6)
    bpy.ops.object.ppf_add_material_map_sample(slot=0)
    dh.record(
        "A_add_seeds_the_current_frame",
        len(entry.samples) == 1 and entry.samples[0].frame == 6,
        {"n_samples": len(entry.samples),
         "frames": [s.frame for s in entry.samples]},
    )

    before = len(entry.samples)
    status = bpy.ops.object.ppf_add_material_map_sample(slot=0)
    dh.record(
        "B_duplicate_frame_refused",
        len(entry.samples) == before and "CANCELLED" in status,
        {"status": list(status), "n_samples": len(entry.samples)},
    )

    bpy.context.scene.frame_set(3)
    bpy.ops.object.ppf_add_material_map_sample(slot=0)
    bpy.context.scene.frame_set(9)
    bpy.ops.object.ppf_add_material_map_sample(slot=0)
    frames = [s.frame for s in entry.samples]
    dh.record(
        "C_samples_stay_sorted",
        frames == sorted(frames) and frames == [3, 6, 9],
        {"frames": frames},
    )

    # Drain from the LAST row without re-seeding the index, so each removal
    # has to clamp it down. Writing the index before each call would make the
    # final assertion read back its own value and leave the clamp untested.
    entry.samples_index = len(entry.samples) - 1
    seen_indices = []
    while len(entry.samples):
        bpy.ops.object.ppf_remove_material_map_sample(slot=0)
        seen_indices.append(entry.samples_index)
    dh.record(
        "D_remove_keeps_the_index_valid",
        len(entry.samples) == 0 and entry.samples_index == 0
        and seen_indices == [1, 0, 0]
        and all(i >= 0 for i in seen_indices),
        {"n_samples": len(entry.samples), "index": entry.samples_index,
         "index_after_each_removal": seen_indices},
    )

    animatable = {
        name: entry.bl_rna.properties[name].is_animatable
        for name in ("parameter", "source_type", "source_name", "target_value",
                     "enabled")
    }
    dh.record(
        "E_target_value_is_not_animatable",
        not any(animatable.values()),
        {"is_animatable": animatable},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE.replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
