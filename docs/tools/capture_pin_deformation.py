"""Seed a pin that already carries a captured deformation.

Passed to ``blender_addon/capture.sh --pre-python``, this produces the
state the Pins documentation shows for **Capture Deformation**: a pin
whose object is driven by a deforming modifier, with the per-frame cache
already recorded::

    bash blender_addon/capture.sh -o /tmp/shots --panel-only \\
        --pre-python docs/tools/capture_pin_deformation.py \\
        --all DYNAMICS_PT_Groups

Two things here are not obvious.

First, the button cannot be pressed from a pre-python script.
``object.capture_pin_deformation`` returns ``RUNNING_MODAL`` and walks
the frame range from a timer, so invoking the operator would return
immediately and the capture timer would photograph a job still in
flight. The module's own job functions are driven synchronously instead
— ``_start_job`` / ``_tick_job`` until it reports no more work /
``_finalize_job`` / ``_cleanup_after_job`` — which is the same sequence
the modal handler runs, minus the waiting.

Second, the pin's object needs a real deforming MODIFIER:
``has_deforming_modifier_stack`` (core/utils.py) inspects the modifier
stack, and shape keys alone do not satisfy it despite being named in the
operator's own warning. A Wave modifier is used because it deforms as a
function of the frame with no armature, lattice or cage to place.

The number of frames recorded is the add-on's own **Frame Count**
(default 180), not the scene's frame range.
"""

import os
import runpy
import sys

import bpy

_HERE = os.path.dirname(os.path.abspath(__file__))

os.environ.setdefault("PPF_CAPTURE_CLOTH", "ClothBanner")
os.environ.setdefault("PPF_CAPTURE_PIN", "TopEdge")
os.environ.setdefault("PPF_CAPTURE_COLLIDER", "0")
runpy.run_path(os.path.join(_HERE, "capture_scene.py"), run_name="__main__")

# Resolve the add-on package from a module it has certainly imported, so
# this works under the extension name without hard-coding it.
_PKG = next(n.removesuffix(".ui.solver") for n in sys.modules
            if n.endswith(".ui.solver"))
cap_ops = __import__(_PKG + ".ui.dynamics.pin_capture_ops", fromlist=["x"])
groups_mod = __import__(_PKG + ".models.groups", fromlist=["x"])

CLOTH = os.environ["PPF_CAPTURE_CLOTH"]


def _add_deformer() -> None:
    wave = bpy.data.objects[CLOTH].modifiers.new(name="Wave", type="WAVE")
    wave.height = 0.15
    wave.width = 0.6


def _capture_first_pin() -> None:
    scene = bpy.context.scene
    scene.frame_set(scene.frame_start)

    errors: list = []
    entry = cap_ops._build_entry(scene, 0, 0, errors)
    if entry is None:
        print(f"capture_pin_deformation: nothing to capture: {errors}")
        return

    ok, err = cap_ops._start_job(bpy.context, [entry])
    if not ok:
        print(f"capture_pin_deformation: job refused: {err}")
        return

    # Bounded rather than `while True`: a job that never reports itself
    # finished would otherwise hang the capture with no output at all.
    guard = 0
    while cap_ops._tick_job(bpy.context) and guard < 5000:
        guard += 1
    n_pins, n_frames = cap_ops._finalize_job(bpy.context)
    cap_ops._cleanup_after_job(bpy.context)
    groups_mod.invalidate_overlays()
    print(f"capture_pin_deformation: {n_frames} frame(s) across {n_pins} pin(s)")


def main() -> None:
    _add_deformer()
    _capture_first_pin()
    # Show the Pins section and collapse Stats, so the crop is the part
    # the figure is about.
    for group in groups_mod.iterate_object_groups(bpy.context.scene):
        group.show_pin = True
        group.show_stats = False


main()
