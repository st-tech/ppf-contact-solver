# File: examples.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# The Examples panel: pick one of the Blender example scripts, edit the
# parameters of its `build()`, and populate a scene with it.
#
# The scripts are read by `core/example_catalog.py` (structure only, nothing
# executed). The panel's state lives on the window manager, not on a scene: it
# is a choice the artist is making, not part of any scene they would save, and
# Populate can make a new scene.
#
# POPULATE DOES WHAT RUNNING THE SCRIPT DOES. Each script's `__main__` calls
# `solver.clear()` and then `build()`; Populate runs the same two calls, in a
# new Blender scene by default (so nothing the artist has is touched) or in the
# current one. The script is loaded fresh from its file on every press, so an
# edited example is what gets built.

from __future__ import annotations

import importlib.util
import textwrap

import bpy  # pyright: ignore
from bpy.app.translations import pgettext_iface as iface_  # pyright: ignore
from bpy.props import (  # pyright: ignore
    BoolProperty,
    CollectionProperty,
    FloatProperty,
    FloatVectorProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import Operator, Panel, PropertyGroup  # pyright: ignore

from ..core import example_catalog as catalog
from ..core.utils import get_category_name
from ..models.enum_props import EnumProperty, dynamic_enum_items


# The catalog, read once and again on Refresh. The enum's item tuples live
# here too, so one list serves every redraw; `dynamic_enum_items` keeps the
# returned list referenced as every dynamic enum in the add-on must.
_CATALOG: list[catalog.Example] = []
_ITEMS: list[tuple[str, str, str]] = []
_LOADED = False


def _load_catalog(force: bool = False) -> list[catalog.Example]:
    global _LOADED
    if force or not _LOADED:
        _CATALOG[:] = catalog.read_catalog()
        _ITEMS[:] = [(e.key, e.title, e.description) for e in _CATALOG]
        _LOADED = True
    return _CATALOG


def find_example(key: str) -> catalog.Example | None:
    return next((e for e in _load_catalog() if e.key == key), None)


@dynamic_enum_items
def _example_items(_self, _context):
    _load_catalog()
    return _ITEMS


def _on_example_changed(self, _context):
    load_params(self)


class EXAMPLES_PG_Param(PropertyGroup):
    """One editable `build()` parameter. Only the value field of its kind is
    shown; the others stay at their defaults."""

    kind: StringProperty(options={"HIDDEN"})
    label: StringProperty(options={"HIDDEN"})
    optional: BoolProperty(options={"HIDDEN"})
    use_value: BoolProperty(
        name="Set",
        description="Pass this value. Off leaves the parameter to the script, which picks its own",
    )
    int_value: IntProperty()
    float_value: FloatProperty(precision=4)
    bool_value: BoolProperty()
    str_value: StringProperty()
    vec_value: FloatVectorProperty(size=3)


class EXAMPLES_PG_State(PropertyGroup):
    example: EnumProperty(
        name="Example",
        description="Example scene to populate",
        items=_example_items,
        update=_on_example_changed,
    )
    target: EnumProperty(
        name="Target",
        description="Which Blender scene receives the example",
        items=(
            ("NEW", "New Scene",
             "Populate a new Blender scene and switch to it, leaving the current scene untouched"),
            ("CURRENT", "Current Scene",
             "Clear this scene's solver setup and populate the example into it, "
             "as running the script from the Scripting workspace does"),
        ),
        default="NEW",
    )
    params: CollectionProperty(type=EXAMPLES_PG_Param)
    # Which example `params` was filled for.
    loaded: StringProperty(options={"HIDDEN"})


def get_state(context=None) -> EXAMPLES_PG_State:
    wm = (context or bpy.context).window_manager
    return wm.ppf_examples


def load_params(state: EXAMPLES_PG_State) -> None:
    """Fill ``state.params`` with the selected example's defaults."""
    state.params.clear()
    example = find_example(state.example)
    state.loaded = state.example
    if example is None:
        return
    for p in example.params:
        item = state.params.add()
        item.name = p.name
        item.kind = p.kind
        item.label = p.label
        item.optional = p.optional
        item.use_value = p.default is not None
        if p.default is None:
            continue
        if p.kind == catalog.INT:
            item.int_value = p.default
        elif p.kind == catalog.FLOAT:
            item.float_value = p.default
        elif p.kind == catalog.BOOL:
            item.bool_value = p.default
        elif p.kind == catalog.STR:
            item.str_value = p.default
        elif p.kind == catalog.VEC3:
            item.vec_value = p.default


def build_kwargs(state: EXAMPLES_PG_State) -> dict:
    """The keyword arguments the panel's fields stand for."""
    kwargs = {}
    for item in state.params:
        if item.optional and not item.use_value:
            kwargs[item.name] = None
        elif item.kind == catalog.INT:
            kwargs[item.name] = item.int_value
        elif item.kind == catalog.FLOAT:
            kwargs[item.name] = item.float_value
        elif item.kind == catalog.BOOL:
            kwargs[item.name] = item.bool_value
        elif item.kind == catalog.STR:
            kwargs[item.name] = item.str_value
        elif item.kind == catalog.VEC3:
            kwargs[item.name] = tuple(item.vec_value)
    return kwargs


def _load_module(example: catalog.Example):
    spec = importlib.util.spec_from_file_location(
        "ppf_example_" + example.key.replace("-", "_"), example.path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unique_scene_name(base: str) -> str:
    if base not in bpy.data.scenes:
        return base
    n = 1
    while f"{base}.{n:03d}" in bpy.data.scenes:
        n += 1
    return f"{base}.{n:03d}"


def populate(context, state: EXAMPLES_PG_State) -> bpy.types.Scene:
    """Build the selected example. Returns the populated scene; raises on any
    failure, after removing a scene it created for the attempt."""
    from ..ops.api import solver

    example = find_example(state.example)
    if example is None:
        raise LookupError(f"no example named '{state.example}'")
    if example.error:
        raise RuntimeError(f"{example.title}: {example.error}")
    if state.loaded != state.example:
        load_params(state)
    kwargs = build_kwargs(state)

    window = context.window or (
        context.window_manager.windows[0] if context.window_manager.windows else None)
    previous = window.scene if window is not None else context.scene
    if state.target == "NEW":
        scene = bpy.data.scenes.new(_unique_scene_name(example.title))
    else:
        scene = context.scene
    try:
        if window is not None:
            window.scene = scene
        with context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
            module = _load_module(example)
            solver.clear()
            module.build(**kwargs)
    except Exception:
        if state.target == "NEW":
            if window is not None:
                window.scene = previous
            bpy.data.scenes.remove(scene)
        raise
    return scene


class EXAMPLES_OT_Populate(Operator):
    """Populate a scene with the selected example and the parameters shown"""

    bl_idname = "examples.populate"
    bl_label = "Populate"
    bl_options = {"REGISTER", "UNDO"}

    # A press asks first. The script writes into this .blend (it replaces any
    # data that carries its own names, in any scene), and Current Scene also
    # clears the scene's solver setup, so the artist confirms with what will
    # change in front of them and a way to save a copy first. A call from a
    # script runs `execute` directly and is not asked.
    def invoke(self, context, _event):
        state = get_state(context)
        example = find_example(state.example)
        if example is None:
            self.report({"ERROR"}, iface_("No example selected"))
            return {"CANCELLED"}
        return context.window_manager.invoke_props_dialog(
            self, width=500, title=iface_("Populate '{example}'?").format(example=example.title),
            confirm_text=iface_("Populate"))

    def draw(self, context):
        state = get_state(context)
        example = find_example(state.example)
        layout = self.layout
        col = layout.column(align=True)
        if state.target == "NEW":
            name = _unique_scene_name(example.title) if example is not None else ""
            col.label(text=iface_("A new scene '{scene}' is created and shown.").format(scene=name),
                      icon="SCENE_DATA")
        else:
            col.label(text=iface_("Scene '{scene}' loses its solver setup: groups, pins and "
                                  "parameters reset.").format(scene=context.scene.name),
                      icon="ERROR")
            col.label(text="The example is then built into this scene.")
        col.label(text="Objects and meshes with the example's names are replaced, in every scene.",
                  icon="INFO")

        layout.separator()
        col = layout.column(align=True)
        if not bpy.data.filepath:
            col.label(text="This file has never been saved.", icon="ERROR")
        elif bpy.data.is_dirty:
            col.label(text="This file has unsaved changes.", icon="ERROR")
        col.label(text="Save the file first to keep a copy of it as it is now.")
        layout.row().operator("wm.save_as_mainfile", text="Save As...", icon="FILE_TICK")

    def execute(self, context):
        state = get_state(context)
        try:
            scene = populate(context, state)
        except Exception as exc:
            self.report({"ERROR"}, f"{type(exc).__name__}: {exc}")
            return {"CANCELLED"}
        self.report(
            {"INFO"},
            iface_("Populated '{example}' into scene '{scene}'").format(
                example=find_example(state.example).title, scene=scene.name),
        )
        return {"FINISHED"}


class EXAMPLES_OT_Refresh(Operator):
    """Read the example scripts again, picking up added or edited ones"""

    bl_idname = "examples.refresh"
    bl_label = "Refresh Examples"

    def execute(self, context):
        _load_catalog(force=True)
        state = get_state(context)
        if find_example(state.example) is None and _CATALOG:
            state.example = _CATALOG[0].key
        load_params(state)
        return {"FINISHED"}


class EXAMPLES_OT_ResetParams(Operator):
    """Set every parameter back to the script's own default"""

    bl_idname = "examples.reset_params"
    bl_label = "Reset to Defaults"

    def execute(self, context):
        load_params(get_state(context))
        return {"FINISHED"}


_SYNC_PENDING = False


def _sync_params():
    # The selection can change without the enum's update callback (a fresh
    # window manager, a reload), and a panel's draw may not write properties,
    # so the fill is deferred to a timer.
    global _SYNC_PENDING
    _SYNC_PENDING = False
    try:
        state = get_state()
        if state.loaded != state.example:
            load_params(state)
    except Exception:
        pass
    return None


def _draw_param(layout, item) -> None:
    row = layout.row(align=True)
    if item.optional:
        row.prop(item, "use_value", text="")
        sub = row.row(align=True)
        sub.active = item.use_value
        row = sub
    field = {
        catalog.INT: "int_value",
        catalog.FLOAT: "float_value",
        catalog.BOOL: "bool_value",
        catalog.STR: "str_value",
        catalog.VEC3: "vec_value",
    }.get(item.kind)
    if field is None:
        return
    if item.kind == catalog.VEC3:
        col = row.column(align=True)
        col.label(text=item.label, translate=False)
        col.row(align=True).prop(item, field, text="")
    else:
        row.prop(item, field, text=item.label, translate=False)


class EXAMPLES_PT_Examples(Panel):
    bl_label = "Examples"
    bl_idname = "EXAMPLES_PT_Examples"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = get_category_name()
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        global _SYNC_PENDING
        layout = self.layout
        state = get_state(context)
        examples = _load_catalog()

        row = layout.row(align=True)
        if not examples:
            row.label(text="No example scripts found", icon="ERROR")
            row.operator(EXAMPLES_OT_Refresh.bl_idname, text="", icon="FILE_REFRESH")
            return
        row.prop(state, "example", text="")
        row.operator(EXAMPLES_OT_Refresh.bl_idname, text="", icon="FILE_REFRESH")

        example = find_example(state.example)
        if example is None:
            return
        if state.loaded != state.example and not _SYNC_PENDING:
            _SYNC_PENDING = True
            bpy.app.timers.register(_sync_params, first_interval=0.0)

        # `ui_scale` reads 0 in a Blender with no window (background mode on
        # Windows), so it falls back to 1 rather than dividing by it.
        scale = context.preferences.system.ui_scale or 1.0
        width = max(20, int(context.region.width / (7.0 * scale)))
        if example.description:
            col = layout.column(align=True)
            for line in textwrap.wrap(example.description, width, break_on_hyphens=False):
                col.label(text=line, translate=False)
        if example.error:
            layout.label(text=example.error, icon="ERROR", translate=False)
            return

        box = layout.box()
        header = box.row()
        header.label(text="Parameters")
        header.operator(EXAMPLES_OT_ResetParams.bl_idname, text="", icon="LOOP_BACK")
        col = box.column(align=True)
        for item in state.params:
            _draw_param(col, item)
        if example.skipped:
            box.label(
                text=iface_("Kept at the script's default: {names}").format(
                    names=", ".join(example.skipped)),
                icon="INFO",
            )

        layout.row().prop(state, "target", expand=True)
        layout.operator(EXAMPLES_OT_Populate.bl_idname, icon="SCENE_DATA")


classes = (
    EXAMPLES_PG_Param,
    EXAMPLES_PG_State,
    EXAMPLES_OT_Populate,
    EXAMPLES_OT_Refresh,
    EXAMPLES_OT_ResetParams,
    EXAMPLES_PT_Examples,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.WindowManager.ppf_examples = PointerProperty(type=EXAMPLES_PG_State)


def unregister():
    global _LOADED
    if bpy.app.timers.is_registered(_sync_params):
        bpy.app.timers.unregister(_sync_params)
    del bpy.types.WindowManager.ppf_examples
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    _CATALOG.clear()
    _ITEMS.clear()
    _LOADED = False
