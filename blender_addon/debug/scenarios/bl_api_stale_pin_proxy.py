# File: scenarios/bl_api_stale_pin_proxy.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# A pin proxy from the Python API that outlives its pin REFUSES every write,
# naming what is gone, rather than returning itself as though the write had
# landed.
#
# A script holds `pin = group.create_pin(...)` across calls that can remove
# what it names: `pin.delete()`, removing the pin from the panel, or
# `solver.clear()`. Every writing method then answered `self` and changed
# nothing, so a chain like `pin.move_by(...).spin(...)` read as success.
# Encode-only: no build and no run.
#
# Subtests:
#   A. live_proxy_writes: on a live pin every writing method lands (an
#      operation is added, pull is set, unpin sets the duration).
#   B. deleted_pin_refuses: after `pin.delete()` each writing method raises
#      a ValueError naming the pin and its object.
#   C. cleared_solver_refuses: after `solver.clear()` each writing method
#      raises a ValueError naming the missing group.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. It never asks the solver to step.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def writers(pin):
    return {
        "pull": lambda: pin.pull(strength=2.0),
        "spin": lambda: pin.spin(axis=(0.0, 0.0, 1.0), angular_velocity=90.0,
                                 frame_start=1, frame_end=3),
        "scale": lambda: pin.scale(factor=0.5, frame_start=1, frame_end=3),
        "move_by": lambda: pin.move_by(delta=(0.1, 0.0, 0.0),
                                       frame_start=1, frame_end=3),
        "unpin": lambda: pin.unpin(frame=3),
        "set_animation": lambda: pin.set_animation([[[0.0, 0.0, 0.0]] * 4]),
    }


def refusals(pin):
    out = {}
    for name, call in writers(pin).items():
        try:
            call()
            out[name] = ""
        except ValueError as exc:
            out[name] = str(exc)
        except Exception as exc:
            out[name] = f"UNEXPECTED {type(exc).__name__}: {exc}"
    return out


try:
    dh = DriverHelpers(pkg, result)
    plane = dh.reset_scene_to_pinned_plane(name="Sheet")
    dh.save_blend(PROBE_DIR, "stale_pin_proxy.blend")
    dh.configure_state(project_name="stale_pin_proxy", frame_count=6)
    solver = dh.api.solver

    # ---- A: a live proxy's writes land ------------------------------------
    cloth = solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.1, 0.0, 0.0), frame_start=1, frame_end=3)
    pin.pull(strength=2.0)
    pin.unpin(frame=3)
    group = dh.groups.get_active_group_by_uuid(bpy.context.scene, cloth.uuid)
    item = group.pin_vertex_groups[0]
    dh.record(
        "A_live_proxy_writes",
        len(item.operations) == 1 and item.use_pull
        and abs(item.pull_strength - 2.0) < 1e-6
        and item.use_pin_duration and item.pin_duration == 3,
        {"operations": len(item.operations), "use_pull": item.use_pull,
         "pin_duration": item.pin_duration},
    )

    # ---- B: a deleted pin -------------------------------------------------
    pin.delete()
    deleted = refusals(pin)
    dh.record(
        "B_deleted_pin_refuses",
        all("Pin 'AllPin' on 'Sheet' not found" in m for m in deleted.values())
        and len(group.pin_vertex_groups) == 0,
        {"messages": deleted},
    )

    # ---- C: a cleared solver ----------------------------------------------
    pin = cloth.create_pin(plane.name, "AllPin")
    solver.clear()
    cleared = refusals(pin)
    dh.record(
        "C_cleared_solver_refuses",
        all(m.startswith("Group '") and m.endswith("' not found")
            for m in cleared.values()),
        {"messages": cleared},
    )
except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
