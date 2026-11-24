# File: scenarios/bl_examples_populate.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# The Examples panel (`ui/examples.py`) lists every script in
# `examples/blender/`, draws a field for each of its parameters, and its
# Populate button builds the example into a scene the solver would accept.
#
# Assertion-only: no server, no build, no run. `bl_blender_examples` is the
# on-demand scenario that solves each example; this one stops at the payload,
# so it is cheap enough for every sweep and still catches an example the panel
# cannot list, draw or populate.
#
# Checks:
#   `catalog`: one entry per `examples/blender/*.py` that does not opt out
#   with `PANEL = False` (`character-anim.py` does), none with an error, each
#   with at least one parameter, and the panel's state round-trips each
#   script's defaults (the fields pass exactly what the script would use).
#   `draw:<key>`: the panel draws with that example selected (a recording mock
#   layout, since the rig's hidden window does not reliably draw a sidebar):
#   one value field per parameter, the target switch and the Populate button.
#   `confirm_dialog`: Populate pressed by hand asks first. Its dialog names
#   the new scene for NEW, warns that the scene's solver setup is cleared for
#   CURRENT, and offers Save As for both; a script's call runs unasked.
#   `<key>`: Populate into a NEW scene finishes, switches the window to the
#   scene it made, leaves the previous scene's objects as they were, gives
#   every active group at least one object, every one of them in the new
#   scene, and the scene encodes (mesh and parameter payloads both decode
#   non-empty).
#   `failure_cleans_up`: an example whose `build()` raises (a script written
#   for the check and added to the catalog) FAILS CLEANLY: the operator
#   cancels, the scene it made is gone, the window is back on the previous
#   scene and that scene's objects are as they were.
#   `param_applies`: a changed parameter reaches the script (pdrd_chain at
#   three links builds five objects fewer than at its default eight).
#   `current_target`: Populate with Current Scene builds into the current
#   scene and makes no new one.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

BACKENDS = ("real",)


_DRIVER_BODY = r"""
import ast
import glob
import os
import traceback
import types

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
EXAMPLES_DIR = os.path.join(LOCAL_PATH, "examples", "blender")


class MockLayout:
    # Answers every layout call and records what was drawn into one sink
    # shared with every child layout.

    def __init__(self, sink=None):
        self.sink = [] if sink is None else sink
        self.enabled = True
        self.active = True
        self.alignment = "EXPAND"

    def _child(self, *a, **k):
        return MockLayout(self.sink)

    box = row = column = split = _child

    def label(self, *a, **k):
        self.sink.append(("label", k.get("text", "")))

    def separator(self, *a, **k):
        pass

    def prop(self, data, prop_name, *a, **k):
        self.sink.append(("prop", prop_name, getattr(data, "name", "")))

    def operator(self, idname, *a, **k):
        self.sink.append(("operator", idname))


def _objects(scene):
    return {o.name for o in scene.objects}


def _group_report(scene):
    groups = {}
    for group in dh.groups.iterate_active_object_groups(scene):
        names = []
        for assigned in group.assigned_objects:
            obj = dh.uuid_registry.get_object_by_uuid(assigned.uuid) if assigned.uuid else None
            names.append(obj.name if obj is not None else None)
        groups[group.name] = names
    return groups


def _encode(scene):
    with bpy.context.temp_override(scene=scene, view_layer=scene.view_layers[0]):
        data_bytes, param_bytes = dh.encode_payload()
    data = dh.decode_addon_blob(data_bytes)
    param = dh.decode_addon_blob(param_bytes)
    return len(data_bytes), len(param_bytes), bool(data), bool(param)


def _drop_scene(scene, keep):
    # Each check starts from the one base scene: remove what Populate made.
    window = bpy.context.window_manager.windows[0]
    if window.scene == scene:
        window.scene = keep
    for obj in list(scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.scenes.remove(scene)


def _populate(key, target="NEW", overrides=None):
    state = ex.get_state(bpy.context)
    state.example = key
    ex.load_params(state)
    for item in state.params:
        if overrides and item.name in overrides:
            value = overrides[item.name]
            item.use_value = True
            if item.kind == "INT":
                item.int_value = value
            elif item.kind == "STR":
                item.str_value = value
    state.target = target
    try:
        return bpy.ops.examples.populate()
    except RuntimeError as exc:
        # An operator that reports an ERROR raises through `bpy.ops`
        # instead of returning its CANCELLED.
        dh.log(f"populate {key}: {exc}")
        return {"CANCELLED"}


try:
    dh = DriverHelpers(pkg, result)
    dh.uuid_registry = __import__(pkg + ".core.uuid_registry",
                                  fromlist=["get_object_by_uuid"])
    ex = __import__(pkg + ".ui.examples", fromlist=["get_state"])
    cat = __import__(pkg + ".core.example_catalog", fromlist=["read_catalog"])
    window = bpy.context.window_manager.windows[0]
    base = window.scene

    # catalog
    scripts, opted_out = [], []
    for p in sorted(glob.glob(os.path.join(EXAMPLES_DIR, "*.py"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        with open(p, encoding="utf-8") as handle:
            offered = cat._offered(ast.parse(handle.read()))
        (scripts if offered else opted_out).append(stem)
    examples = ex._load_catalog(force=True)
    listed = sorted(e.key for e in examples)
    errors = {e.key: e.error for e in examples if e.error}
    no_params = [e.key for e in examples if not e.params]
    mismatched = {}
    state = ex.get_state(bpy.context)
    for e in examples:
        state.example = e.key
        ex.load_params(state)
        kwargs = ex.build_kwargs(state)
        want = {p.name: p.default for p in e.params}
        diff = {k: (kwargs.get(k), v) for k, v in want.items() if kwargs.get(k) != v
                and not (isinstance(v, float) and abs(kwargs.get(k, 0.0) - v) <= 1e-6 * max(1.0, abs(v)))
                and not (isinstance(v, tuple) and all(abs(a - b) <= 1e-6 for a, b in zip(kwargs.get(k), v)))}
        if diff:
            mismatched[e.key] = diff
    dh.record("catalog", listed == scripts and not errors and not no_params and not mismatched, {
        "directory": cat.examples_dir(),
        "scripts": scripts, "opted_out": opted_out, "listed": listed, "errors": errors,
        "no_params": no_params, "default_mismatch": mismatched,
    })

    # draw:<key>
    ctx = types.SimpleNamespace(
        window_manager=bpy.context.window_manager,
        region=types.SimpleNamespace(width=320),
        preferences=bpy.context.preferences,
    )
    for e in examples:
        try:
            state.example = e.key
            ex.load_params(state)
            layout = MockLayout()
            ex.EXAMPLES_PT_Examples.draw(types.SimpleNamespace(layout=layout), ctx)
            fields = [s for s in layout.sink if s[0] == "prop" and s[1].endswith("_value")
                      and s[1] != "use_value"]
            ops = [s[1] for s in layout.sink if s[0] == "operator"]
            props = [s[1] for s in layout.sink if s[0] == "prop"]
            ok = (len(fields) == len(e.params)
                  and "examples.populate" in ops
                  and "target" in props and "example" in props)
            dh.record(f"draw:{e.key}", ok, {
                "fields": len(fields), "params": len(e.params),
                "operators": ops, "props": sorted(set(props)),
            })
        except Exception as exc:
            dh.record(f"draw:{e.key}", False, {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-1500:]})

    # confirm_dialog
    try:
        state.example = examples[0].key
        ex.load_params(state)
        seen = {}
        for target in ("NEW", "CURRENT"):
            state.target = target
            layout = MockLayout()
            ctx = types.SimpleNamespace(
                window_manager=bpy.context.window_manager, scene=bpy.context.scene)
            ex.EXAMPLES_OT_Populate.draw(types.SimpleNamespace(layout=layout), ctx)
            seen[target] = {
                "labels": [x[1] for x in layout.sink if x[0] == "label"],
                "operators": [x[1] for x in layout.sink if x[0] == "operator"],
            }
        new_scene = ex._unique_scene_name(examples[0].title)
        ok = (
            hasattr(ex.EXAMPLES_OT_Populate, "invoke")
            and all("wm.save_as_mainfile" in v["operators"] for v in seen.values())
            and any(new_scene in t for t in seen["NEW"]["labels"])
            and any("solver setup" in t and bpy.context.scene.name in t
                    for t in seen["CURRENT"]["labels"])
            and not any("solver setup" in t for t in seen["NEW"]["labels"])
        )
        dh.record("confirm_dialog", ok, seen)
    except Exception as exc:
        dh.record("confirm_dialog", False, {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:]})
    state.target = "NEW"

    # <key>
    for e in examples:
        before_scenes = set(bpy.data.scenes.keys())
        before_objects = _objects(base)
        try:
            outcome = _populate(e.key, "NEW")
            made = [bpy.data.scenes[n] for n in set(bpy.data.scenes.keys()) - before_scenes]
            scene = made[0] if len(made) == 1 else None
            details = {"outcome": sorted(outcome), "scenes_made": [s.name for s in made]}
            ok = outcome == {"FINISHED"} and scene is not None
            if scene is not None:
                groups = _group_report(scene)
                in_scene = _objects(scene)
                stray = {g: [n for n in names if n is None or n not in in_scene]
                         for g, names in groups.items()}
                stray = {g: v for g, v in stray.items() if v}
                empty = [g for g, names in groups.items() if not names]
                encoded = _encode(scene)
                details.update({
                    "scene": scene.name, "window_scene": window.scene.name,
                    "n_objects": len(in_scene),
                    "groups": {g: len(n) for g, n in groups.items()},
                    "empty_groups": empty, "objects_outside_scene": stray,
                    "base_objects_unchanged": _objects(base) == before_objects,
                    "encoded_bytes": encoded[:2],
                })
                ok = (ok and window.scene == scene and in_scene and groups
                      and not empty and not stray
                      and _objects(base) == before_objects
                      and encoded[2] and encoded[3])
                _drop_scene(scene, base)
            dh.record(e.key, bool(ok), details)
        except Exception as exc:
            dh.record(e.key, False, {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-1500:]})
            for n in set(bpy.data.scenes.keys()) - before_scenes:
                _drop_scene(bpy.data.scenes[n], base)

    # failure_cleans_up
    try:
        broken = os.path.join(os.path.dirname(os.path.abspath(PROBE_DIR)), "broken_example.py")
        with open(broken, "w", encoding="utf-8") as handle:
            handle.write(
                "import bpy\n\n\n"
                "def build(count: int = 2):\n"
                "    '''Broken.\n\n    Adds objects, then raises.\n    '''\n"
                "    for i in range(count):\n"
                "        mesh = bpy.data.meshes.new(f'broken.{i}')\n"
                "        bpy.context.scene.collection.objects.link(\n"
                "            bpy.data.objects.new(f'broken.{i}', mesh))\n"
                "    raise RuntimeError('this example fails on purpose')\n")
        entry = cat.read_example(broken)
        ex._CATALOG.append(entry)
        ex._ITEMS.append((entry.key, entry.title, entry.description))
        before_scenes = set(bpy.data.scenes.keys())
        before_objects = _objects(base)
        outcome = _populate(entry.key, "NEW")
        made = sorted(set(bpy.data.scenes.keys()) - before_scenes)
        dh.record("failure_cleans_up",
                  outcome == {"CANCELLED"} and not made and window.scene == base
                  and _objects(base) == before_objects, {
                      "outcome": sorted(outcome), "scenes_left": made,
                      "window_scene": window.scene.name,
                  })
        for n in made:
            _drop_scene(bpy.data.scenes[n], base)
    except Exception as exc:
        dh.record("failure_cleans_up", False, {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:]})

    # param_applies
    try:
        counts = {}
        for n_links in (None, 3):
            before_scenes = set(bpy.data.scenes.keys())
            outcome = _populate("pdrd_chain", "NEW",
                                {"n_links": n_links} if n_links else None)
            scene = bpy.data.scenes[(set(bpy.data.scenes.keys()) - before_scenes).pop()]
            counts[str(n_links or "default")] = (sorted(outcome), len(scene.objects))
            _drop_scene(scene, base)
        dh.record("param_applies",
                  counts["default"][1] - counts["3"][1] == 5, {"counts": counts})
    except Exception as exc:
        dh.record("param_applies", False, {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:]})

    # current_target
    try:
        work = bpy.data.scenes.new("examples-current-target")
        window.scene = work
        before_scenes = set(bpy.data.scenes.keys())
        outcome = _populate("pdrd_chain", "CURRENT")
        groups = _group_report(work)
        ok = (outcome == {"FINISHED"} and set(bpy.data.scenes.keys()) == before_scenes
              and window.scene == work and len(work.objects) > 0 and bool(groups))
        dh.record("current_target", ok, {
            "outcome": sorted(outcome), "n_objects": len(work.objects),
            "groups": {g: len(n) for g, n in groups.items()},
            "new_scenes": sorted(set(bpy.data.scenes.keys()) - before_scenes),
        })
        _drop_scene(work, base)
    except Exception as exc:
        dh.record("current_target", False, {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:]})
except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE.replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 600.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
