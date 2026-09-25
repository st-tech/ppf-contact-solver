# File: scenarios/bl_force_field_visualize.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The force field's Visualize overlay (owner's request): arrows at a sparse
# PREVIEW grid, separate from the transfer sampling, for the field at the
# timeline's current frame.
#
# The checks are on `preview_fields`, WHAT is drawn, and on the arrow geometry,
# neither of which needs a GPU, so every platform runs them. Building the GPU
# batches does, and Windows runs the rig headless, so that check is
# `bl_force_field_visualize_draws`, a scenario of its own that the Windows
# leg does not select.
#
# Subtests:
#   A. off_draws_nothing: with Visualize off there is nothing to draw.
#   B. preview_grid_is_its_own: Preview Resolution 4 x 3 x 2 gives 24 arrow
#      sites whatever the transfer Spacing says.
#   C. arrows_are_the_evaluator: the acceleration arrows equal the encoder's
#      evaluator at the same points, and the script's arrows equal the script.
#   D. arrows_follow_the_timeline: a Force whose strength is keyed 0 -> 5 has
#      no arrows at the timeline's first frame and arrows at its last, and the
#      overlay's cache key changes with the frame, so the drawing rebuilds.
#   E. key_follows_a_moved_field: moving the field object changes the overlay's
#      cache key, so the drawn arrows rebuild.
#   F. arrow_heads_are_sized_by_their_arrow: a short arrow's head is a quarter
#      of its length, and a long one's is capped by the shaft thickness, so a
#      weak spot draws a small arrow rather than a full head on a stub.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True
BACKENDS = ("real",)

_DRIVER_BODY = r"""
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

try:
    dh = DriverHelpers(pkg, result)
    ff = __import__(pkg + ".core.force_field", fromlist=["evaluate"])
    viz = __import__(pkg + ".ui.dynamics.overlay_geometry.force_field",
                     fromlist=["preview_fields"])
    scene = bpy.context.scene

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(size=1.0)
    sheet = bpy.context.object
    dh.save_blend(PROBE_DIR, "force_field_visualize.blend")
    root = dh.configure_state(project_name="force_field_visualize", frame_count=11)
    state = root.state
    dh.api.solver.create_group("Cloth", "SHELL").add(sheet.name)
    bpy.ops.object.effector_add(type="FORCE", location=(0.0, 0.0, -0.4))
    push = bpy.context.object
    push.field.strength = 0.0
    push.field.keyframe_insert("strength", frame=1)
    push.field.strength = 5.0
    push.field.keyframe_insert("strength", frame=11)
    scene.frame_set(11)

    dh.record("A_off_draws_nothing",
              viz.preview_fields(scene, state) is None
              and viz.build_force_field_batches(scene, state, 10.0) == [], {})

    state.force_field_visualize = True
    state.force_field_spacing = 0.02
    state.force_field_preview_resolution = (4, 3, 2)
    pts, cell, fields = viz.preview_fields(scene, state)
    dh.record("B_preview_grid_is_its_own",
              pts.shape == (24, 3) and len(fields) == 2,
              {"points": list(pts.shape), "fields": len(fields)})

    text = bpy.data.texts.new("ff_viz.py")
    # `import math` and `curl_noise` on purpose: the template uses both, and
    # the drawing path once ran scripts with no import machinery at all. The
    # noise is multiplied by zero so the expected arrows stay a closed form.
    text.from_string("import math\n\ndef eval(x, y, z, t):\n"
                     "    cx, cy, cz = curl_noise(x, y, z, octaves=2, seed=5)\n"
                     "    return (y + 0.0 * cx, -x, 0.5 * math.cos(0.0) + 0.0 * cz)\n")
    state.force_field_script = text
    pts, cell, fields = viz.preview_fields(scene, state)
    acc, _air = ff.evaluate(ff.snapshot(ff.field_objects(scene, state)), pts)
    script = fields[2][0] if len(fields) == 3 else None
    want = np.stack([pts[:, 1], -pts[:, 0], np.full(len(pts), 0.5)], axis=1)
    dh.record("C_arrows_are_the_evaluator",
              len(fields) == 3 and np.allclose(fields[0][0], acc)
              and script is not None and np.allclose(script, want),
              {"fields": len(fields)})
    state.force_field_script = None

    scene.frame_set(1)
    first = viz.preview_fields(scene, state)[2][0][0]
    key_first = viz.force_field_key(scene, state)
    scene.frame_set(11)
    last = viz.preview_fields(scene, state)[2][0][0]
    key_last = viz.force_field_key(scene, state)
    dh.record("D_arrows_follow_the_timeline",
              float(np.abs(first).max()) < 1e-9
              and float(np.abs(last).max()) > 1.0
              and key_first != key_last,
              {"max_first": float(np.abs(first).max()),
               "max_last": float(np.abs(last).max())})

    key0 = viz.force_field_key(scene, state)
    push.location = (0.3, 0.0, -0.4)
    bpy.context.view_layer.update()
    key1 = viz.force_field_key(scene, state)
    dh.record("E_key_follows_a_moved_field", key0 != key1, {})

    from mathutils import Vector

    def head_length(length, thickness):
        tris = viz._field_arrow(Vector((0.0, 0.0, 0.0)), Vector((length, 0.0, 0.0)),
                                thickness)
        # The head's ring sits where the shaft ends: the smallest x among the
        # cone's vertices, which are the last 18 (6 triangles of 3).
        return length - min(v.x for v in tris[-18:])

    short = head_length(0.02, 0.002)
    long_ = head_length(1.0, 0.002)
    dh.record("F_arrow_heads_are_sized_by_their_arrow",
              abs(short - 0.25 * 0.02) < 1e-7 and abs(long_ - 12.0 * 0.002) < 1e-6,
              {"short_head": short, "long_head": long_})

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
