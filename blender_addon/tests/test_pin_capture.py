# File: test_pin_capture.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Regression tests for Capture Pin Deformation. Runs inside Blender
# via the debug port (matches tests/test_redraw.py's contract).
#
# Usage from a host shell:
#
#     echo 'import ppf_contact_solver.tests.test_pin_capture as t; print(t.run_all())' | \
#         python blender_addon/debug/main.py exec -
#
# Each ``test_*`` function is autodiscovered by ``run_all()``. Tests
# clean up after themselves (delete all objects + reset groups) so the
# suite can run repeatedly in one Blender session.

import os
import tempfile
import traceback
import types

import bpy  # pyright: ignore
import numpy as np

import ppf_contact_solver.core.pc2 as pc2_mod
import ppf_contact_solver.ui.dynamics.pin_capture_ops as pcap_mod
import ppf_contact_solver.ui.dynamics.pin_ops as pin_ops_mod
import ppf_contact_solver.ui.dynamics.ui_lists as ui_lists_mod
from ppf_contact_solver.models.groups import get_addon_data
from ppf_contact_solver.ops.api import solver as solver_api


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_scene():
    """Delete every object and zero the addon groups."""
    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    root = get_addon_data(bpy.context.scene)
    for i in range(32):
        grp = getattr(root, f"object_group_{i}")
        grp.is_active = False
        grp.assigned_objects.clear()
        grp.pin_vertex_groups.clear()


def _build_pinned_plane(name="CapPlane", n_grid=3):
    """Create a small plane mesh, register a SHELL group, attach a pin
    that covers every vertex, and return (obj, pin_item, group_index)."""
    _reset_scene()
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    obj = bpy.context.active_object
    obj.name = name
    n_verts = len(obj.data.vertices)
    vg = obj.vertex_groups.new(name="AllPin")
    vg.add(list(range(n_verts)), 1.0, "REPLACE")

    cloth = solver_api.create_group("Cloth", "SHELL")
    cloth.add(obj.name)
    cloth.create_pin(obj.name, "AllPin")

    root = get_addon_data(bpy.context.scene)
    group = root.object_group_0
    group.pin_vertex_groups_index = 0
    return obj, group.pin_vertex_groups[0], 0


def _make_fake_capture_cache(obj, vg_name, n_frames=5):
    """Hand-write a small ``_pindeform.pc2`` cache for *obj*/*vg_name*.

    Bypasses the modal operator so unit tests don't need a depsgraph
    walk; the round-trip + encoder paths under test are the same.
    """
    pin_indices = list(range(len(obj.data.vertices)))
    n_pin = len(pin_indices)
    frames = np.zeros((n_frames, n_pin, 3), dtype=np.float32)
    # Frame k displaces every pin vertex by (k * 0.1) along +X. Lets
    # the round-trip check verify exact values.
    for k in range(n_frames):
        frames[k, :, 0] = k * 0.1
    pc2_mod.write_pin_anim_pc2(obj, vg_name, frames)
    return frames


# ---------------------------------------------------------------------------
# A. PC2 round-trip
# ---------------------------------------------------------------------------


def test_pin_anim_pc2_roundtrip():
    obj, _, _ = _build_pinned_plane()
    frames = _make_fake_capture_cache(obj, "AllPin", n_frames=4)
    key = pc2_mod.pin_anim_pc2_key(obj, "AllPin")
    # Drop the in-memory entry; loader should rebuild it from disk.
    pc2_mod.unload_pin_anim_cache(key)
    assert key not in pc2_mod._pin_anim_cache, "in-memory entry not cleared"
    reloaded = pc2_mod.get_pin_anim_cache(obj, "AllPin")
    assert reloaded is not None, "reload from disk returned None"
    assert reloaded.shape == frames.shape, (
        f"shape mismatch: {reloaded.shape} vs {frames.shape}"
    )
    assert np.allclose(reloaded, frames, atol=1e-6), "round-trip values diverged"


def test_pin_anim_pc2_key_distinct_per_vg():
    obj, _, _ = _build_pinned_plane()
    # Second pin VG on the same mesh.
    vg2 = obj.vertex_groups.new(name="OtherPin")
    vg2.add([0], 1.0, "REPLACE")
    k1 = pc2_mod.pin_anim_pc2_key(obj, "AllPin")
    k2 = pc2_mod.pin_anim_pc2_key(obj, "OtherPin")
    assert k1 != k2, "two pins on the same object yielded the same PC2 key"
    assert k1.endswith("__pindeform"), f"unexpected key shape: {k1}"
    assert "__AllPin__" in k1, f"vg name not embedded in key: {k1}"


def test_has_pin_anim_pc2_reflects_disk():
    obj, _, _ = _build_pinned_plane()
    assert not pc2_mod.has_pin_anim_pc2(obj, "AllPin"), "cache present before write"
    _make_fake_capture_cache(obj, "AllPin", n_frames=3)
    assert pc2_mod.has_pin_anim_pc2(obj, "AllPin"), "cache absent after write"
    pc2_mod.remove_pin_anim_pc2(obj, "AllPin")
    assert not pc2_mod.has_pin_anim_pc2(obj, "AllPin"), "cache survives remove"


# ---------------------------------------------------------------------------
# B. Clear-drops-sentinel logic
# ---------------------------------------------------------------------------


def test_clear_drops_sentinel_when_no_fcurves():
    obj, pin_item, group_index = _build_pinned_plane()
    _make_fake_capture_cache(obj, "AllPin", n_frames=4)
    pin_item.has_captured_anim = True
    pin_ops_mod._ensure_embedded_move_op(pin_item)
    assert any(op.op_type == "EMBEDDED_MOVE" for op in pin_item.operations), (
        "sentinel missing after _ensure_embedded_move_op"
    )
    bpy.ops.object.clear_pin_deformation(
        "EXEC_DEFAULT", group_index=group_index, pin_index=0,
    )
    assert pin_item.has_captured_anim is False, "flag not cleared"
    assert not any(op.op_type == "EMBEDDED_MOVE" for op in pin_item.operations), (
        "sentinel survived clear even though no fcurves exist"
    )


def test_clear_keeps_sentinel_when_fcurves_exist():
    obj, pin_item, group_index = _build_pinned_plane()
    _make_fake_capture_cache(obj, "AllPin", n_frames=4)
    pin_item.has_captured_anim = True
    pin_ops_mod._ensure_embedded_move_op(pin_item)
    # Author vertex-co fcurves: keyframe vertex 0 at frame 1.
    bpy.context.scene.frame_set(1)
    obj.data.vertices[0].keyframe_insert(data_path="co")
    bpy.ops.object.clear_pin_deformation(
        "EXEC_DEFAULT", group_index=group_index, pin_index=0,
    )
    # has_captured_anim cleared but EMBEDDED_MOVE survives because
    # the manual Make-Keyframe path still owns it.
    assert pin_item.has_captured_anim is False, "flag not cleared"
    assert any(op.op_type == "EMBEDDED_MOVE" for op in pin_item.operations), (
        "sentinel dropped despite live vertex-co fcurves"
    )


# ---------------------------------------------------------------------------
# C. Reject paths between Make Keyframe and Capture
# ---------------------------------------------------------------------------


def test_make_keyframe_rejected_while_captured():
    obj, pin_item, group_index = _build_pinned_plane()
    _make_fake_capture_cache(obj, "AllPin", n_frames=4)
    pin_item.has_captured_anim = True
    pin_ops_mod._ensure_embedded_move_op(pin_item)
    bpy.context.scene.frame_set(1)
    # Should report ERROR + return CANCELLED because the pin is captured.
    result = bpy.ops.object.make_pin_keyframe(
        "EXEC_DEFAULT", group_index=group_index,
    )
    assert result == {"CANCELLED"}, (
        f"make_pin_keyframe should refuse on captured pin, got {result}"
    )


def test_capture_rejected_while_fcurves_exist():
    obj, pin_item, group_index = _build_pinned_plane()
    # Author manual fcurves on every pinned vertex at frame 1.
    bpy.context.scene.frame_set(1)
    for v in obj.data.vertices:
        v.keyframe_insert(data_path="co")
    pin_ops_mod._ensure_embedded_move_op(pin_item)
    # Capture op should refuse — no cache should appear.
    bpy.ops.object.capture_pin_deformation(
        "EXEC_DEFAULT", group_index=group_index, pin_index=0,
    )
    assert not pc2_mod.has_pin_anim_pc2(obj, "AllPin"), (
        "capture should not produce a cache while fcurves exist"
    )
    assert pin_item.has_captured_anim is False, (
        "has_captured_anim should not have been set"
    )


# ---------------------------------------------------------------------------
# D. UIList label switches on has_captured_anim
# ---------------------------------------------------------------------------


class _LabelRecorder:
    """Stand-in layout that captures label() calls."""
    def __init__(self):
        self.labels = []
    def row(self, align=False):
        return self
    def label(self, text="", icon="NONE"):
        self.labels.append((text, icon))
    def prop(self, *_a, **_kw):
        pass


def _exercise_pin_uilist_label(pin_item, op):
    layout = _LabelRecorder()
    # bpy.types.UIList subclasses can't be instantiated via __new__
    # from Python (bpy_struct.__new__ rejects zero-arg construction);
    # call draw_item unbound with None for self because the
    # EMBEDDED_MOVE branch never reads self.
    ui_lists_mod.OBJECT_UL_PinOperationsList.draw_item(
        None,
        bpy.context, layout, pin_item, op, 0, None, None, 0,
    )
    assert layout.labels, "no label was drawn"
    return layout.labels[0]


def test_uilist_label_plain_when_not_captured():
    _, pin_item, _ = _build_pinned_plane()
    pin_ops_mod._ensure_embedded_move_op(pin_item)
    pin_item.has_captured_anim = False
    op = next(o for o in pin_item.operations if o.op_type == "EMBEDDED_MOVE")
    text, _ = _exercise_pin_uilist_label(pin_item, op)
    assert text == "[Embedded] Move", f"expected plain label, got {text!r}"


def test_uilist_label_captured_when_flag_set():
    _, pin_item, _ = _build_pinned_plane()
    pin_ops_mod._ensure_embedded_move_op(pin_item)
    pin_item.has_captured_anim = True
    op = next(o for o in pin_item.operations if o.op_type == "EMBEDDED_MOVE")
    text, _ = _exercise_pin_uilist_label(pin_item, op)
    assert text == "[Embedded] Move (Captured)", (
        f"expected captured-suffix label, got {text!r}"
    )


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------


def run_all():
    tests = sorted(
        (name, fn) for name, fn in globals().items()
        if name.startswith("test_") and callable(fn)
    )
    passed = []
    failed = []
    for name, fn in tests:
        try:
            fn()
            passed.append(name)
        except Exception as exc:
            failed.append({
                "test": name,
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(),
            })
    # Final cleanup so the suite leaves no residue.
    try:
        _reset_scene()
    except Exception:
        pass
    return {
        "total": len(tests),
        "passed": len(passed),
        "failed_count": len(failed),
        "failed": failed,
    }
