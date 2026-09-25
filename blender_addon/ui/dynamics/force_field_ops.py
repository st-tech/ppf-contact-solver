# File: force_field_ops.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The force field's two operators: a new exact-script Text, and Compile and
# Check, which asks the connected server to compile the script with the same
# compiler a Transfer runs, without a Transfer.

import bpy  # pyright: ignore
from bpy.props import StringProperty  # pyright: ignore
from bpy.types import Operator, UIList  # pyright: ignore
from bpy.app.translations import pgettext_iface as iface_  # pyright: ignore

from ...core.async_op import AsyncOperator
from ...core.utils import redraw_all_areas
from ...models import force_field_targets as targets
from ...models.enum_props import EnumProperty, dynamic_enum_items
from ...models.groups import get_addon_data, iterate_active_object_groups

SCRIPT_TEMPLATE = '''import math


def eval(x, y, z, t):
    """Acceleration in m/s^2 at world point (x, y, z), Blender axes and scene
    units, at t seconds into the simulation.

    Evaluated exactly at every simulated vertex on the solver, once per step,
    in addition to any force field objects. Every path must return
    (ax, ay, az). The functions it may call, noise and curl_noise among them,
    are listed by the Built-in Functions button beside the Script.
    """
    r = math.hypot(x, y) + 1e-6
    swirl = 2.0 * math.sin(2.0 * t)
    gx, gy, gz = curl_noise(x, y, z, octaves=2, seed=1, time=t, frequency=0.5)
    return (-y / r * swirl + gx, x / r * swirl + gy, gz)
'''


def check_unavailable_reason(context) -> str:
    """Why Compile and Check cannot run now, or "" when it can.

    One predicate for the operator's poll and the panel's status line, so the
    button is never enabled for a state the operator would refuse.
    """
    from ...core.facade import communicator as com

    state = get_addon_data(context.scene).state
    if state.force_field_script is None:
        return iface_("Choose or create a script first")
    if not com.is_connected():
        return iface_("Connect to a server to check the script")
    if not com.is_server_running():
        return iface_("Start the server to check the script")
    return ""


class SCENE_OT_ForceFieldNewScript(Operator):
    """Create a Text holding a force field script template and use it"""

    bl_idname = "scene.force_field_new_script"
    bl_label = "New Script"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        text = bpy.data.texts.new("force_field.py")
        text.from_string(SCRIPT_TEMPLATE)
        get_addon_data(context.scene).state.force_field_script = text
        redraw_all_areas(context)
        return {"FINISHED"}


class SCENE_OT_ForceFieldBuiltins(Operator):
    """List every function, constant and construct a force field script may use"""

    bl_idname = "scene.force_field_builtins"
    bl_label = "Built-in Functions"

    def execute(self, context):
        return {"FINISHED"}

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=720)

    def draw(self, context):
        # The one list (`core/script_api.py`) the compiler's errors and the
        # frontend's `builtins()` read too, so the popup cannot promise a
        # call the solver refuses.
        from ...core import script_api as api

        col = self.layout.column(align=True)
        for line in api.LANGUAGE:
            col.label(text=line)
        for title, entries in api.SECTIONS:
            box = self.layout.box()
            box.label(text=title)
            grid = box.grid_flow(row_major=True, columns=2, even_columns=False, align=True)
            for _, sig, what in entries:
                grid.label(text=sig)
                grid.label(text=what)
            if entries is api.NOISE:
                for arg, what in api.NOISE_ARGUMENTS:
                    grid.label(text=f"    {arg}")
                    grid.label(text=what)


class SOLVER_OT_ForceFieldCheck(AsyncOperator):
    """Compile the force field script on the connected server and report any
    error with its line, without transferring the scene"""

    bl_idname = "solver.force_field_check"
    bl_label = "Compile and Check"
    timeout = 60.0

    @classmethod
    def poll(cls, context):
        return check_unavailable_reason(context) == ""

    def execute(self, context):
        from ...core.facade import communicator as com
        from ...core import force_field as ff

        state = get_addon_data(context.scene).state
        text = state.force_field_script
        opener = com.channel_opener()
        if opener is None:
            self.report({"ERROR"}, iface_("Not connected"))
            return {"CANCELLED"}
        self._source = text.as_string()
        ff.request_check(opener, self._source, text.name)
        self.setup_modal(context)
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        from ...core import force_field as ff

        return not ff.check_state()["running"]

    def on_complete(self, context):
        from ...core import force_field as ff

        result = ff.check_state()["result"] or {}
        if result.get("ok"):
            self.report({"INFO"}, iface_("Force field script OK: {summary}").format(
                summary=result.get("summary", "")))
        else:
            line = result.get("line")
            where = iface_("line {line}: ").format(line=line) if line else ""
            self.report({"ERROR"}, iface_("Force field script, {where}{error}").format(
                where=where, error=result.get("error", "")))
        redraw_all_areas(context)


# --- group targets ---------------------------------------------------------


class SCENE_UL_ForceFieldGroups(UIList):
    """The groups a force field source is narrowed to."""

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_property, index=0, flt_flag=0):
        problem = targets.ref_problem(context.scene, item)
        row = layout.row(align=True)
        row.label(text=targets.display_name(context.scene, item),
                  icon="ERROR" if problem else "GROUP")
        if problem:
            row.label(text=problem)


def _source(context, name):
    state = get_addon_data(context.scene).state
    return state, targets.resolve_source(state, name)


class SCENE_OT_ForceFieldChooseGroups(Operator):
    """Narrow this force field source to the groups you choose"""

    bl_idname = "scene.force_field_choose_groups"
    bl_label = "Choose Groups"
    bl_options = {"REGISTER", "UNDO"}

    source: StringProperty(name="Source", options={"HIDDEN"})  # pyright: ignore

    def execute(self, context):
        try:
            state, source = _source(context, self.source)
        except ValueError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        if source == targets.SCRIPT:
            state.force_field_script_all = False
        else:
            targets.ensure_entry(state, source).apply_all = False
        redraw_all_areas(context)
        return {"FINISHED"}


@dynamic_enum_items
def _target_group_items(_self, context):
    items = []
    for number, group in enumerate(iterate_active_object_groups(context.scene)):
        if str(group.object_type) == "STATIC":
            continue
        items.append((group.uuid, group.name or iface_("Dynamics Group"), "",
                      "GROUP", number))
    if not items:
        return [("NONE", iface_("(No Simulated Groups)"), "", "ERROR", 1000)]
    return items


class SCENE_OT_ForceFieldAddGroup(Operator):
    """Add a group this force field source pushes"""

    bl_idname = "scene.force_field_add_group"
    bl_label = "Add Group"
    bl_options = {"REGISTER", "UNDO"}

    source: StringProperty(name="Source", options={"HIDDEN"})  # pyright: ignore
    group: EnumProperty(name="Group", items=_target_group_items)  # pyright: ignore

    def execute(self, context):
        try:
            state, source = _source(context, self.source)
        except ValueError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        if self.group == "NONE":
            self.report({"ERROR"}, iface_("There is no simulated group to add"))
            return {"CANCELLED"}
        current = targets.target_group_uuids(state, source) or []
        if self.group not in current:
            try:
                targets.set_targets(context.scene, state, source, current + [self.group])
            except ValueError as e:
                self.report({"ERROR"}, str(e))
                return {"CANCELLED"}
        redraw_all_areas(context)
        return {"FINISHED"}


class SCENE_OT_ForceFieldRemoveGroup(Operator):
    """Remove the highlighted group from this force field source"""

    bl_idname = "scene.force_field_remove_group"
    bl_label = "Remove Group"
    bl_options = {"REGISTER", "UNDO"}

    source: StringProperty(name="Source", options={"HIDDEN"})  # pyright: ignore

    def execute(self, context):
        try:
            state, source = _source(context, self.source)
        except ValueError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        _, groups, owner, index_attr = targets.source_view(state, source)
        if groups is None:
            return {"CANCELLED"}
        index = getattr(owner, index_attr)
        if 0 <= index < len(groups):
            groups.remove(index)
            setattr(owner, index_attr, min(index, len(groups) - 1))
        redraw_all_areas(context)
        return {"FINISHED"}


classes = (
    SCENE_UL_ForceFieldGroups,
    SCENE_OT_ForceFieldChooseGroups,
    SCENE_OT_ForceFieldAddGroup,
    SCENE_OT_ForceFieldRemoveGroup,
    SCENE_OT_ForceFieldNewScript,
    SOLVER_OT_ForceFieldCheck,
    SCENE_OT_ForceFieldBuiltins,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
