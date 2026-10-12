# File: scenarios/bl_hair_pin_hints_draw.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The two lines the pin panel draws under the Pull row of a Hair Curves pin
# (issue #146), drawn by the REAL panel in a real Properties editor.
#
# A hair pin stored as a float attribute carries a weight per point, and the
# panel says so, since nothing else on the row does: that each point is pulled
# with the strength times its weight while the pin pulls, and that the weights
# wait for Pull while it does not. And because the later pin wins a point two
# pins share, a pulling hair pin listed after one that holds exactly says
# that the pins above lose their shared points to it, which is what turns
# exact roots into softly held ones.
#
# Neither line can be read back off a layout, so the probe counts the two
# predicates the panel asks, by wrapping them where the panel looks them up,
# and counts the draw itself only once the real draw has returned: a draw that
# raised on either branch would leave that count where it was.
#
# Coverage targets:
#   A. weighted_pulling_pin_asks_both: with Roots listed first and the
#      weighted pin pulling, the panel finds the pin weighted and finds an
#      exact pin of the same object above it.
#   B. pull_off_asks_only_the_weight: with Pull off the weight line is still
#      decided, and the order is not asked about, since a pin that holds
#      exactly takes nothing softer.
#   C. listed_first_there_is_nothing_above: moved above Roots, the pulling
#      pin finds no exact pin above it.
#   D. a_plain_pin_is_not_weighted: with the derived Roots pin selected the
#      panel finds it unweighted and draws.
#   E. a_moving_exact_pin_counts_too: Roots above again and carrying an
#      operation, as it does once it is captured to follow a head, is still
#      an exact pin above. It is not a plain one, and it loses its shared
#      points to the pull below it all the same.
#
# Assertion-only: no server / build / run.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r


NEEDS_BLENDER = True

# Declared for the real backend like every scenario; it never solves.
BACKENDS = ("real",)

# NOT ON WINDOWS, WHICH RUNS THE RIG HEADLESS: `redraw_timer` has no window
# to draw there, so nothing this scenario counts would move.
PLATFORMS = ("linux", "darwin")


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})

DREW = {"panel": 0}
ASKED = {"weighted": [], "above": []}

try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    hair_mod = __import__(pkg + ".core.hair_curves", fromlist=["x"])
    panels = __import__(pkg + ".ui.dynamics.panels", fromlist=["x"])

    if bpy.context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    curves = bpy.data.hair_curves.new("Hair")
    curves.add_curves([5, 5, 5])
    curves.attributes["position"].data.foreach_set(
        "vector",
        np.array(
            [(0.05 * i, 0.0, 0.1 * c) for c in range(3) for i in range(5)],
            dtype=np.float32,
        ).ravel(),
    )
    hair = bpy.data.objects.new("HairObj", curves)
    bpy.context.collection.objects.link(hair)
    bpy.context.view_layer.objects.active = hair
    hair.select_set(True)

    root = dh.configure_state(project_name="hair_pin_hints", frame_count=4)
    rod = dh.api.solver.create_group("Hair", "ROD")
    rod.add(hair.name)
    group = root.object_group_0
    rod.create_pin(hair.name, hair_mod.ROOTS_PIN_NAME)          # index 0
    rod.create_pin(
        hair.name, "Goal", indices=[0, 1, 2, 3], weights=[1.0, 1.0, 0.5, 0.25],
    ).pull(strength=5.0)                                        # index 1
    group.show_group = True
    group.show_pin = True
    group.pin_vertex_groups_index = 1

    _real_weighted = panels._hair_pin_weighted
    _real_above = panels._exact_pin_above

    def _counting_weighted(obj, name):
        answer = _real_weighted(obj, name)
        ASKED["weighted"].append((name, bool(answer)))
        return answer

    def _counting_above(asked_group, pin_idx, pin_item):
        answer = _real_above(asked_group, pin_idx, pin_item)
        ASKED["above"].append((pin_item.name, bool(answer)))
        return answer

    panels._hair_pin_weighted = _counting_weighted
    panels._exact_pin_above = _counting_above

    class PPF_PT_HairPinHintProbe(bpy.types.Panel):
        bl_label = "Hair Pin Hint Probe"
        bl_idname = "PPF_PT_hair_pin_hint_probe"
        bl_space_type = "PROPERTIES"
        bl_region_type = "WINDOW"
        bl_context = "object"

        def draw(self, context):
            # The REAL panel, so the branch under the Pull row is the one
            # that runs. Counted only after it returns.
            panels.DYNAMICS_PT_Groups.draw(self, context)
            DREW["panel"] += 1

    bpy.utils.register_class(PPF_PT_HairPinHintProbe)

    areas = 0
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != "PROPERTIES":
                continue
            for space in area.spaces:
                if space.type == "PROPERTIES":
                    space.context = "OBJECT"
                    areas += 1
            area.tag_redraw()
    if not areas:
        raise RuntimeError(
            "no Properties editor in this screen; the probe has no region to "
            "draw in and a pass would mean nothing"
        )

    def redraw():
        # What one round of draws asked, and whether the panel got through.
        ASKED["weighted"].clear()
        ASKED["above"].clear()
        before = DREW["panel"]
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == "PROPERTIES":
                    area.tag_redraw()
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=2)
        return (
            DREW["panel"] > before,
            sorted(set(ASKED["weighted"])),
            sorted(set(ASKED["above"])),
        )

    roots_item = f"[{hair.name}][{hair_mod.ROOTS_PIN_NAME}]"
    goal_item = f"[{hair.name}][Goal]"

    # ----- A: weighted, pulling, with an exact pin above -----------
    drew_a, weighted_a, above_a = redraw()
    dh.record(
        "A_weighted_pulling_pin_asks_both",
        drew_a and weighted_a == [("Goal", True)]
        and above_a == [(goal_item, True)],
        {"drew": drew_a, "weighted": weighted_a, "above": above_a,
         "order": [p.name for p in group.pin_vertex_groups]},
    )

    # ----- B: Pull off -----------------------------------------------
    group.pin_vertex_groups[1].use_pull = False
    drew_b, weighted_b, above_b = redraw()
    dh.record(
        "B_pull_off_asks_only_the_weight",
        drew_b and weighted_b == [("Goal", True)] and above_b == [],
        {"drew": drew_b, "weighted": weighted_b, "above": above_b},
    )
    group.pin_vertex_groups[1].use_pull = True

    # ----- C: listed first -------------------------------------------
    group.pin_vertex_groups.move(1, 0)
    group.pin_vertex_groups_index = 0
    order_c = [p.name for p in group.pin_vertex_groups]
    drew_c, weighted_c, above_c = redraw()
    dh.record(
        "C_listed_first_there_is_nothing_above",
        order_c == [goal_item, roots_item]
        and drew_c and weighted_c == [("Goal", True)]
        and above_c == [(goal_item, False)],
        {"drew": drew_c, "weighted": weighted_c, "above": above_c,
         "order": order_c},
    )

    # ----- D: the derived Roots pin ----------------------------------
    group.pin_vertex_groups_index = 1
    drew_d, weighted_d, above_d = redraw()
    dh.record(
        "D_a_plain_pin_is_not_weighted",
        drew_d and weighted_d == [(hair_mod.ROOTS_PIN_NAME, False)]
        and above_d == [],
        {"drew": drew_d, "weighted": weighted_d, "above": above_d},
    )

    # ----- E: an exact pin that moves is an exact pin above --------
    group.pin_vertex_groups.move(1, 0)
    group.pin_vertex_groups_index = 1
    roots_pin = group.pin_vertex_groups[0]
    operation = roots_pin.operations.add()
    operation.op_type = "MOVE_BY"
    groups_mod = __import__(pkg + ".models.groups", fromlist=["x"])
    plain_now = groups_mod.pin_is_plain(group, roots_pin)
    drew_e, weighted_e, above_e = redraw()
    dh.record(
        "E_a_moving_exact_pin_counts_too",
        [p.name for p in group.pin_vertex_groups] == [roots_item, goal_item]
        and not plain_now
        and drew_e and weighted_e == [("Goal", True)]
        and above_e == [(goal_item, True)],
        {"drew": drew_e, "weighted": weighted_e, "above": above_e,
         "roots_is_plain": plain_now},
    )

    panels._hair_pin_weighted = _real_weighted
    panels._exact_pin_above = _real_above
    bpy.utils.unregister_class(PPF_PT_HairPinHintProbe)

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
