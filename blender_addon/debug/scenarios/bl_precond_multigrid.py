# File: scenarios/bl_precond_multigrid.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# The Preconditioner picker offers Multigrid beside Block Jacobi and Schwarz,
# and the choice the artist makes is the one the solver runs.
#
# Three things can go wrong between the picker and the solve, each without
# a word. A new item can take a number an older .blend already stored for
# another choice. The encoder can ship a name the picker does not show, which
# is what a two-way `"schwarz" if ... else "block-jacobi"` does to a third
# item: Multigrid would run as Block Jacobi and report nothing. And the
# solver stops a multigrid run that holds a rigid body, after the transfer
# and the build, where the artist learns it last.
#
# Subtests:
#   A. picker_slots_are_stable: Block Jacobi, Schwarz and Multigrid sit at
#      numbers 0, 1 and 2, and the default is still Block Jacobi.
#   B. each_choice_ships_its_solver_name: every item encodes as the name the
#      solver takes for it, checked against a table written here rather
#      than the add-on's own.
#   C. rigid_body_is_refused_by_name: with an active PDRD group holding an
#      object, Multigrid is refused at encode with the sentence the panel
#      shows; Block Jacobi encodes the same scene, and Multigrid encodes it
#      again once the group is gone.
#   D. panel_draws_each_choice: the Linear System Solver box draws the
#      Schwarz level count under Schwarz only, and the refusal line under
#      Multigrid only while a PDRD group is in the scene.
#   E. mcp_sets_and_reports: `set_scene_parameters` takes "multigrid" in any
#      case and `get_scene_parameters` reports it; a name the picker does not
#      offer is refused and leaves the choice where it was.
#   F. solve_hangs_under_multigrid: a cloth pinned along one edge and
#      transferred with Multigrid is built, run and fetched; its free edge has
#      swung down and its pinned edge has not moved.
#   G. solver_ran_the_multigrid (host-side): the solver's literal on-disk
#      input names `multigrid`, and the run left the one record the
#      hierarchy build writes, which no other preconditioner writes.

from __future__ import annotations

import glob
import os

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# Run on the real backend: subtests A to E encode or draw and never reach the
# solver, and F and G solve a cloth of a few hundred vertices, which every
# backend covers.
BACKENDS = ("real",)

_FRAME_COUNT = 8
_FRAME_RATE = 60
# Fine enough that the hierarchy has several levels below the mesh.
_SUBDIVISIONS = 20
# The least the free edge must have dropped over the run for the solve to
# count as having simulated. The run covers nine sixtieths of a second, in
# which free fall is about eleven centimeters.
_MIN_DROP = 0.01
# The most the pinned edge may have moved: a fixed pin holds to round-off.
_PIN_SLACK = 1.0e-4


_DRIVER_BODY = r"""
import traceback

import numpy as np

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
FRAME_COUNT = <<FRAME_COUNT>>
FRAME_RATE = <<FRAME_RATE>>
SUBDIVISIONS = <<SUBDIVISIONS>>
MIN_DROP = <<MIN_DROP>>
PIN_SLACK = <<PIN_SLACK>>

# What the solver calls each choice. Written here, not read from the add-on,
# so a wrong entry in the add-on's table cannot agree with itself.
SOLVER_NAMES = {
    "BLOCK_JACOBI": "block-jacobi",
    "SCHWARZ": "schwarz",
    "MULTIGRID": "multigrid",
}


class MockLayout:
    # Records what a draw function asked for: the properties it drew and the
    # text of its labels. The rig's hidden window does not reliably draw a
    # sidebar, so the box is drawn into this instead.
    def __init__(self, sink=None):
        self.sink = sink if sink is not None else {"props": [], "labels": []}

    def box(self, *a, **k):
        return MockLayout(self.sink)

    def row(self, *a, **k):
        return MockLayout(self.sink)

    def label(self, *a, text="", **k):
        self.sink["labels"].append(text)

    def prop(self, data, prop_name, *a, **k):
        self.sink["props"].append(prop_name)


def drawn_box(panels, state):
    layout = MockLayout()
    panels._draw_linear_system_solver(layout, bpy.context, state)
    return layout.sink


def encode_error():
    try:
        dh.encoder_param.encode_param(bpy.context)
        return ""
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def shipped_precond():
    payload = DriverHelpers.decode_addon_blob(
        dh.encoder_param.encode_param(bpy.context))
    return payload["scene"]["precond"]


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_grid_add(size=1.0, x_subdivisions=SUBDIVISIONS,
                                    y_subdivisions=SUBDIVISIONS,
                                    location=(0.0, 0.0, 1.0))
    cloth = bpy.context.active_object
    cloth.name = "MultigridCloth"
    # One edge is held and the opposite one is free, so the solve has
    # stretch and bending to resolve rather than a rigid fall.
    held = [v.index for v in cloth.data.vertices if v.co.y > 0.5 - 1.0e-6]
    free = [v.index for v in cloth.data.vertices if v.co.y < -0.5 + 1.0e-6]
    cloth.vertex_groups.new(name="Edge").add(held, 1.0, "REPLACE")

    dh.save_blend(PROBE_DIR, "precond_multigrid.blend")
    # The rig's defaults switch gravity off; this scenario needs the cloth
    # to move.
    root = dh.configure_state(project_name="precond_multigrid",
                              frame_count=FRAME_COUNT,
                              frame_rate=FRAME_RATE,
                              gravity=(0.0, 0.0, -9.8))
    state = root.state
    scene = bpy.context.scene

    sheet = dh.api.solver.create_group("Cloth", "SHELL")
    sheet.add(cloth.name)
    sheet.create_pin(cloth.name, "Edge")

    # ---- A: the picker's numbers ------------------------------------
    prop = state.bl_rna.properties["precond"]
    slots = {item.identifier: int(item.value) for item in prop.enum_items}
    dh.record(
        "A_picker_slots_are_stable",
        slots == {"BLOCK_JACOBI": 0, "SCHWARZ": 1, "MULTIGRID": 2}
        and prop.default == "BLOCK_JACOBI"
        and state.precond == "BLOCK_JACOBI",
        {"slots": slots, "default": prop.default,
         "fresh_scene": state.precond},
    )

    # ---- B: every choice ships under its own name -------------------
    shipped = {}
    for identifier in SOLVER_NAMES:
        state.precond = identifier
        shipped[identifier] = shipped_precond()
    dh.record(
        "B_each_choice_ships_its_solver_name",
        shipped == SOLVER_NAMES,
        {"shipped": shipped, "expected": SOLVER_NAMES},
    )

    # ---- C: a rigid body beside the multigrid -----------------------
    bpy.ops.mesh.primitive_cube_add(size=0.2, location=(5.0, 0.0, 1.0))
    block = bpy.context.active_object
    block.name = "Block"
    rigid = dh.api.solver.create_group("Rigid", "PDRD")
    rigid.add(block.name)

    state.precond = "MULTIGRID"
    refused = encode_error()
    sentence = dh.groups.multigrid_refusal(scene, state)
    state.precond = "BLOCK_JACOBI"
    jacobi = encode_error()
    jacobi_sentence = dh.groups.multigrid_refusal(scene, state)

    # ---- D: what the box draws, with the rigid group still present ----
    panels = __import__(pkg + ".ui.dynamics.panels",
                        fromlist=["_draw_linear_system_solver"])
    refusal_line = "Multigrid does not solve PDRD groups; Transfer will refuse"
    state.show_linear_system_solver = True
    drawn = {}
    for identifier in SOLVER_NAMES:
        state.precond = identifier
        drawn[identifier] = drawn_box(panels, state)
    state.show_linear_system_solver = False
    folded = drawn_box(panels, state)
    state.show_linear_system_solver = True
    rigid.delete()
    state.precond = "MULTIGRID"
    cleared = encode_error()
    cleared_sentence = dh.groups.multigrid_refusal(scene, state)
    dh.record(
        "C_rigid_body_is_refused_by_name",
        isinstance(sentence, str)
        and "'Rigid'" in sentence
        and "Block Jacobi or Schwarz" in sentence
        and refused == f"ValueError: {sentence}"
        and jacobi == "" and jacobi_sentence is None
        and cleared == "" and cleared_sentence is None,
        {"refused": refused[:400], "panel_sentence": sentence,
         "block_jacobi": jacobi[:200], "after_group_deleted": cleared[:200]},
    )
    bpy.data.objects.remove(block, do_unlink=True)

    state.precond = "MULTIGRID"
    drawn_alone = drawn_box(panels, state)
    dh.record(
        "D_panel_draws_each_choice",
        all("precond" in drawn[i]["props"] for i in SOLVER_NAMES)
        and ["schwarz_levels" in drawn[i]["props"] for i in SOLVER_NAMES]
        == [False, True, False]
        and [refusal_line in drawn[i]["labels"] for i in SOLVER_NAMES]
        == [False, False, True]
        and refusal_line not in drawn_alone["labels"]
        and "precond" not in folded["props"]
        and refusal_line not in folded["labels"],
        {"with_rigid_group": drawn, "multigrid_without_it": drawn_alone,
         "folded": folded},
    )

    # ---- E: the MCP tool --------------------------------------------
    remote = __import__(pkg + ".mcp.handlers.remote",
                        fromlist=["set_scene_parameters"])
    state.precond = "BLOCK_JACOBI"
    answer = remote.set_scene_parameters({"precond": "multigrid"})
    after_set = state.precond
    reported = remote.get_scene_parameters({}).get("parameters", {})
    unknown = remote.set_scene_parameters({"precond": "amg"})
    after_unknown = state.precond
    dh.record(
        "E_mcp_sets_and_reports",
        answer.get("status") != "error"
        and after_set == "MULTIGRID"
        and reported.get("precond") == "MULTIGRID"
        and unknown.get("status") == "error"
        and "MULTIGRID" in str(unknown.get("message", ""))
        and after_unknown == "MULTIGRID",
        {"set": {k: answer.get(k) for k in ("status", "message")},
         "state_after_set": after_set,
         "reported": reported.get("precond"),
         "unknown": {k: unknown.get(k) for k in ("status", "message")},
         "state_after_unknown": after_unknown},
    )

    # ---- F: a solve under the multigrid ------------------------------
    state.precond = "MULTIGRID"
    data_bytes, param_bytes = dh.encode_payload()
    dh.connect(local_path=LOCAL_PATH, server_port=SERVER_PORT,
               project_name=root.state.project_name)
    dh.log("connected")
    dh.build_and_wait(data_bytes, param_bytes,
                      message="precond_multigrid:build")
    dh.log("built")
    dh.run_and_wait(timeout=240.0)
    solver_after_run = dh.facade.engine.state.solver.name
    dh.log(f"ran solver={solver_after_run}")
    dh.force_frame_query(expected_frames=FRAME_COUNT - 1, timeout=30.0)
    frames_reported = int(dh.facade.engine.state.frame)
    dh.settle_idle(timeout=15.0)
    applied, total = dh.fetch_and_drain()
    dh.log(f"fetch.drained applied={applied}/{total}")

    # The cache holds object-local positions, and the object is unrotated
    # and unscaled, so its third column is height.
    path = dh.find_pc2_for(cloth)
    heights = []
    pin_drift = float("inf")
    finite = False
    if path:
        frames = np.asarray(dh.read_pc2(path), dtype=np.float64)
        finite = bool(np.isfinite(frames).all())
        heights = [float(frame[free, 2].mean()) for frame in frames]
        pin_drift = float(np.abs(frames[:, held, :] - frames[0, held, :]).max())
    drop = heights[0] - heights[-1] if len(heights) >= 2 else 0.0
    dh.record(
        "F_solve_hangs_under_multigrid",
        solver_after_run in ("READY", "RESUMABLE")
        and frames_reported >= FRAME_COUNT - 1
        and len(heights) >= FRAME_COUNT - 1
        and finite
        and drop > MIN_DROP
        and pin_drift < PIN_SLACK,
        {"solver": solver_after_run, "frames_reported": frames_reported,
         "frames_fetched": len(heights), "free_edge_drop": drop,
         "least": MIN_DROP, "pinned_edge_drift": pin_drift,
         "most": PIN_SLACK,
         "free_edge_heights": [round(h, 5) for h in heights], "pc2": path,
         "held": len(held), "free": len(free),
         "error": getattr(dh.facade.engine.state, "error", None)},
    )

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
        .replace("<<FRAME_COUNT>>", str(_FRAME_COUNT))
        .replace("<<FRAME_RATE>>", str(_FRAME_RATE))
        .replace("<<SUBDIVISIONS>>", str(_SUBDIVISIONS))
        .replace("<<MIN_DROP>>", repr(_MIN_DROP))
        .replace("<<PIN_SLACK>>", repr(_PIN_SLACK))
    )


def _check_solver_ran_the_multigrid(workspace: str) -> tuple[bool, dict]:
    """Host-side: what the solver was handed, and what it left behind.

    The parameter file is the solver's literal input, so it settles which
    name crossed the wire. The hierarchy build record settles that the run
    acted on it: `initialize.multigrid_build.out` is written once when the
    multigrid lays its grids and by nothing else, so a run that fell back to
    another preconditioner leaves no such file.
    """
    # A session is reachable by its own path and through a symlink beside
    # it, so every file is counted once, by where it really is.
    def found(name):
        paths = glob.glob(os.path.join(workspace, "**", name), recursive=True)
        return sorted({os.path.realpath(path) for path in paths})

    params = found("param.toml")
    named = []
    for path in params:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    key, _, value = line.partition("=")
                    if key.strip() == "precond":
                        named.append(value.strip().strip('"'))
        except OSError:
            continue
    records = found("initialize.multigrid_build.out")
    builds = []
    for path in records:
        try:
            with open(path, "r", encoding="utf-8") as f:
                builds.append(sum(1 for line in f if line.strip()))
        except OSError:
            continue
    ok = (len(named) > 0 and all(name == "multigrid" for name in named)
          and builds == [1])
    return ok, {"precond_in_param_toml": named, "param_toml_paths": params,
                "hierarchy_build_records": builds,
                "hierarchy_build_paths": records}


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 420.0))
    if err is not None:
        return err
    checks = dict(result.get("checks", {}))
    ok, details = _check_solver_ran_the_multigrid(ctx.workspace)
    checks["G_solver_ran_the_multigrid"] = {"ok": ok, "details": details}
    return r.report_named_checks(checks)
