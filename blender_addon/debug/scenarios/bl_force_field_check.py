# File: scenarios/bl_force_field_check.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Compile and Check (owner's request): the add-on asks the RUNNING server to
# compile the force-field script with the frontend's compiler, and shows the
# answer, without a Transfer and without a build.
#
# The request travels the real path: `force_field_check` over the live
# connection's channel, the server's worker running `python -m
# frontend._force_field_ check`, and the answer back on the add-on's own
# thread. Only the modal wrapper around it is skipped, because its poll and its
# completion are the two functions the scenario calls directly.
#
# Subtests:
#   A. disabled_until_connected: with a script but no connection the operator's
#      poll is False and the panel's reason names the connection.
#   B. enabled_when_the_server_runs: after connecting, the poll is True and the
#      reason is empty.
#   C. good_script_compiles: the server answers OK with an instruction count.
#   D. bad_script_names_its_line: an `import os` on line 3 answers not OK,
#      naming line 3.
#   E. no_build_started: the check leaves the solver state where it was.
#   F. answer_is_tied_to_the_text: an answer belongs to the text it was given,
#      and there is none for any other text.
#   G. panel_draws_each_state: the Force Fields box, drawn into a recording
#      stand-in layout, shows the connection reason while disconnected, the
#      estimate line, the check's answer for the text it was given, and asks
#      for a new check once the text is edited.
#   H. python_api_sets_every_kind: solver.param takes the force field's
#      vector, datablock (by object and by name) and scalar settings, the
#      group proxy takes force_field_weight, and check_force_field_script
#      refuses while disconnected.
#   I. python_api_checks_on_the_server: connected, check_force_field_script
#      answers like the panel's button.
#   J. clear_resets_the_pointers: solver.clear() leaves no collection or
#      script behind.
#   K. builtins_are_listed_everywhere: the Script box draws the Built-in
#      Functions button, its popup lists every signature of the shared list,
#      and solver.get_force_field_builtins() returns that list.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True
BACKENDS = ("real",)

_DRIVER_BODY = r"""
import time
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>

GOOD = "import math\n\ndef eval(x, y, z, t):\n    return (math.sin(t), 0.0, -x)\n"
BAD = "def eval(x, y, z, t):\n    a = 1.0\n    import os\n    return (a, 0.0, 0.0)\n"

try:
    dh = DriverHelpers(pkg, result)
    ff = __import__(pkg + ".core.force_field", fromlist=["request_check"])
    ops_mod = __import__(pkg + ".ui.dynamics.force_field_ops",
                         fromlist=["SOLVER_OT_ForceFieldCheck"])
    OpCls = ops_mod.SOLVER_OT_ForceFieldCheck

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(size=1.0)
    sheet = bpy.context.object
    dh.save_blend(PROBE_DIR, "force_field_check.blend")
    root = dh.configure_state(project_name="force_field_check", frame_count=5)
    state = root.state
    dh.api.solver.create_group("Cloth", "SHELL").add(sheet.name)
    # A field object, so the panel has a grid to estimate.
    bpy.ops.object.effector_add(type="TURBULENCE", location=(0.0, 0.0, 0.5))
    turb = bpy.context.object
    text = bpy.data.texts.new("ff_check.py")
    text.from_string(GOOD)
    state.force_field_script = text

    panels = __import__(pkg + ".ui.dynamics.panels", fromlist=["_draw_force_fields"])

    class Recorder:
        # A stand-in UILayout: every call returns a layout and the labels and
        # operators drawn are recorded, so the panel code runs as in Blender.
        def __init__(self, sink):
            self.sink = sink
            self.enabled = True
        def __getattr__(self, name):
            def call(*args, **kwargs):
                if name == "label":
                    self.sink.append(("label", kwargs.get("text", args[0] if args else "")))
                elif name == "operator":
                    self.sink.append(("operator", args[0] if args else "", self.enabled))
                return Recorder(self.sink)
            return call

    def draw_panel():
        sink = []
        state.show_force_field = True
        panels._draw_force_fields(Recorder(sink), bpy.context, state)
        return sink

    drawn_disconnected = draw_panel()

    api = __import__(pkg + ".ops.api", fromlist=["solver"]).solver
    coll = bpy.data.collections.new("FFApiFields")
    bpy.context.scene.collection.children.link(coll)
    coll.objects.link(turb)
    api.param.force_field_preview_resolution = (5, 4, 3)
    api.param.force_field_collection = coll
    api.param.force_field_script = "ff_check.py"
    api.param.force_field_spacing = 0.05
    api.param.force_field_time_samples = 5
    grp = api.get_groups()[0]
    grp.param.force_field_weight = 0.5
    try:
        api.check_force_field_script(timeout=5.0)
        refused = None
    except RuntimeError as e:
        refused = str(e)
    raw_group = next(g for g in __import__(pkg + ".models.groups",
                     fromlist=["iterate_object_groups"]).iterate_object_groups(bpy.context.scene))
    dh.record("H_python_api_sets_every_kind",
              tuple(state.force_field_preview_resolution) == (5, 4, 3)
              and state.force_field_collection == coll
              and state.force_field_script == text
              and abs(state.force_field_spacing - 0.05) < 1e-6
              and state.force_field_time_samples == 5
              and abs(raw_group.force_field_weight - 0.5) < 1e-6
              and refused is not None and "Connect" in refused,
              {"preview_resolution": list(state.force_field_preview_resolution),
               "collection": state.force_field_collection.name
               if state.force_field_collection else None,
               "spacing": state.force_field_spacing,
               "script": state.force_field_script.name if state.force_field_script else None,
               "weight": raw_group.force_field_weight, "refused": refused})

    reason = ops_mod.check_unavailable_reason(bpy.context)
    dh.record("A_disabled_until_connected",
              not OpCls.poll(bpy.context) and "Connect" in reason,
              {"reason": reason})

    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=state.project_name)
    reason = ops_mod.check_unavailable_reason(bpy.context)
    dh.record("B_enabled_when_the_server_runs",
              OpCls.poll(bpy.context) and reason == "", {"reason": reason})
    solver_before = dh.facade.engine.state.solver.name

    def check(source):
        ff.request_check(dh.com.channel_opener(), source, text.name)
        deadline = time.time() + 60.0
        while ff.check_state()["running"] and time.time() < deadline:
            time.sleep(0.05)
        return ff.check_state()["result"] or {}

    good = check(GOOD)
    dh.record("C_good_script_compiles",
              good.get("ok") is True and "instructions" in good.get("summary", ""),
              good)
    bad = check(BAD)
    dh.record("D_bad_script_names_its_line",
              bad.get("ok") is False and bad.get("line") == 3
              and "Import" in bad.get("error", ""), bad)
    dh.facade.engine.dispatch(dh.events.PollTick())
    dh.facade.tick()
    solver_after = dh.facade.engine.state.solver.name
    dh.record("E_no_build_started", solver_before == solver_after,
              {"before": solver_before, "after": solver_after})
    # The last answer was for BAD, so the panel shows it only while the Text
    # holds BAD, and asks for a new check once the Text changes back.
    text.from_string(BAD)
    drawn_connected = draw_panel()
    text.from_string(GOOD)
    drawn_edited = draw_panel()
    labels_off = [e[1] for e in drawn_disconnected if e[0] == "label"]
    labels_on = [e[1] for e in drawn_connected if e[0] == "label"]
    labels_edited = [e[1] for e in drawn_edited if e[0] == "label"]
    dh.record("G_panel_draws_each_state",
              any("Connect" in l for l in labels_off)
              and any("MB estimated" in l for l in labels_on)
              and any(l.startswith("Line 3:") for l in labels_on)
              and any("Not checked" in l for l in labels_edited),
              {"disconnected": labels_off, "connected": labels_on,
               "edited": labels_edited})
    text.from_string(GOOD)
    api_answer = api.check_force_field_script(timeout=60.0)
    dh.record("I_python_api_checks_on_the_server",
              api_answer.get("ok") is True, api_answer)
    tied = ff.check_result_for(GOOD)
    other = ff.check_result_for(BAD)
    dh.record("F_answer_is_tied_to_the_text",
              tied is not None and tied.get("ok") is True and other is None,
              {"for_good": tied, "for_bad": other})
    api.clear()
    dh.record("J_clear_resets_the_pointers",
              state.force_field_script is None
              and state.force_field_collection is None, {})

    script_api = __import__(pkg + ".core.script_api", fromlist=["SECTIONS"])
    popup_sink = []

    class Popup:
        layout = Recorder(popup_sink)

    ops_mod.SCENE_OT_ForceFieldBuiltins.draw(Popup(), bpy.context)
    popup_labels = [e[1] for e in popup_sink if e[0] == "label"]
    listed = [sig for _, entries in script_api.SECTIONS for _, sig, _ in entries]
    api_list = api.get_force_field_builtins()
    drawn_ops = [e[1] for e in drawn_disconnected if e[0] == "operator"]
    dh.record("K_builtins_are_listed_everywhere",
              "scene.force_field_builtins" in drawn_ops
              and all(sig in popup_labels for sig in listed)
              and [r["signature"] for r in api_list] == listed
              and any(r["name"] == "curl_noise" for r in api_list),
              {"missing_in_popup": [s_ for s_ in listed if s_ not in popup_labels],
               "operators": drawn_ops, "api_rows": len(api_list)})


except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
