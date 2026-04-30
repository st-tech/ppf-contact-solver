# File: scenarios/bl_violation_overlay_classification.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Violation-overlay batch builder classification.
#
# ``_build_violation_batches`` (ui/dynamics/overlay_geometry.py) maps a
# server-reported violations list onto GPU batches and 2D labels for the
# 3D viewport. The mapping has two responsibilities:
#
#   1. Classify each violation by its ``type`` field and pick the
#      correct color: red for ``self_intersection`` / ``wall`` /
#      ``runtime_intersection``, orange for ``contact_offset``, purple
#      for ``sphere``. Unknown types fall through every elif branch
#      without producing any batch (silent no-op).
#   2. Convert solver Y-up positions to Blender Z-up via the nested
#      ``_solver_to_blender(pos) = Vector((pos[0], -pos[2], pos[1]))``
#      helper inside ``_build_violation_batches``, so the highlights
#      line up with the viewport meshes.
#
# This scenario bypasses the server entirely: it builds a synthetic
# violations payload covering one of each known type plus an unknown
# ``frobnitz`` entry, dispatches a single ``ServerPolled`` event with
# a fully formed protocol-0.03 response, and asserts:
#
#   A. ``_interpret_response`` writes the payload onto
#      ``engine.state.violations`` verbatim.
#   B. ``_build_violation_batches`` returns one batch per known type
#      with the expected color.
#   C. The unknown ``frobnitz`` type produces no batch (the elif chain
#      in ``_build_violation_batches`` has no ``else`` branch, so an
#      unrecognized type is a silent no-op rather than a fallback
#      batch).
#   D. Every emitted batch's matching label carries the violation's
#      ``count`` field as a leading integer in the label text.
#   E. Positions in the emitted batches reflect the
#      ``_solver_to_blender`` axis swap.
#
# Pure UI scenario: no server, no solver, no transfer.

from __future__ import annotations

from . import _runner as r


NEEDS_BLENDER = True


_DRIVER_TEMPLATE = r"""
import bpy, time, traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def log(msg):
    result["phases"].append((round(time.time(), 3), msg))


def record(name, ok, details):
    result["checks"][name] = {"ok": bool(ok), "details": details}


try:
    log("setup_start")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.active_object
    plane.name = "ViolationMesh"

    facade = __import__(pkg + ".core.facade",
                        fromlist=["engine", "tick"])
    events = __import__(pkg + ".core.events", fromlist=["ServerPolled"])
    # ``_solver_to_blender`` is a nested helper inside
    # ``_build_violation_batches`` (not importable at module scope), so
    # only ``_build_violation_batches`` shows up in the fromlist.
    overlay_geometry = __import__(pkg + ".ui.dynamics.overlay_geometry",
                                  fromlist=["_build_violation_batches"])

    # Build a synthetic violations payload, one entry of each known
    # type plus a deliberately-unknown 'frobnitz' to exercise the
    # unrecognized-type branch in _build_violation_batches.
    #
    # Schemas come straight from overlay_geometry._build_violation_batches:
    #   wall   / sphere       : {"vertices": [{"pos": [x, y, z]}, ...]}
    #   contact_offset        : {"pairs": [{"ei_type": "edge"|"triangle",
    #                                        "ei_pos": [[x,y,z], ...],
    #                                        "ej_type": ..., "ej_pos": ...}, ...]}
    #   self_intersection     : {"tris": [[[3 verts], [3 verts]], ...]}
    #   runtime_intersection  : {"entries": [{"itype": "face_edge"|"edge_edge"|
    #                                                  "collision_mesh",
    #                                          "positions0": [...],
    #                                          "positions1": [...]}, ...]}
    #
    # Each violation also carries an integer 'count' that must surface
    # verbatim in the batch's matching label text.
    synthetic_violations = [
        {
            "type": "self_intersection",
            "count": 7,
            "tris": [
                [
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    [[0.5, 0.5, -0.1], [1.5, 0.5, -0.1], [0.5, 1.5, -0.1]],
                ],
            ],
        },
        {
            "type": "contact_offset",
            "count": 3,
            "pairs": [
                {
                    "ei_type": "edge",
                    "ei_pos": [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                    "ej_type": "triangle",
                    "ej_pos": [[2.0, 0.5, 0.0], [3.0, 0.5, 0.0],
                               [2.5, 0.5, 0.5]],
                },
            ],
        },
        {
            "type": "wall",
            "count": 4,
            "vertices": [
                # Solver-space (4, 2, 3) must map to Blender (4, -3, 2)
                # via _solver_to_blender, then the label adds +0.05 z.
                {"pos": [4.0, 2.0, 3.0]},
                {"pos": [4.0, 0.0, 1.0]},
            ],
        },
        {
            "type": "sphere",
            "count": 2,
            "vertices": [
                # Solver-space (5, 1, 2) -> Blender (5, -2, 1).
                {"pos": [5.0, 1.0, 2.0]},
                {"pos": [5.0, 0.0, 1.0]},
            ],
        },
        {
            "type": "runtime_intersection",
            "count": 5,
            "entries": [
                {
                    "itype": "face_edge",
                    "positions0": [[6.0, 0.0, 0.0], [7.0, 0.0, 0.0],
                                   [6.5, 0.0, 1.0]],
                    "positions1": [[6.5, 0.5, 0.0], [6.5, 0.5, 1.0]],
                },
            ],
        },
        # Unknown type. No 'else' branch in _build_violation_batches,
        # so this should yield zero batches and zero labels.
        {
            "type": "frobnitz",
            "count": 1,
            "note": "deliberately-unrecognized type",
        },
    ]
    log("payload_built")

    # Build a fully-formed protocol-0.03 response so _interpret_response
    # accepts it (missing protocol_version or upload_id triggers an early
    # version_ok=False return that never touches state.violations).
    response = {
        "protocol_version": "0.03",
        "upload_id": "synthetic-upload-id",
        "data_hash": "",
        "param_hash": "",
        "status": "READY",
        "error": "",
        "info": "",
        "root": "",
        "frame": 0,
        "violations": synthetic_violations,
    }
    facade.engine.dispatch(events.ServerPolled(response=response))
    facade.tick()
    log("server_polled_dispatched")

    # ------------------------------------------------------------------
    # A. _interpret_response copies the violations list onto state.
    # ------------------------------------------------------------------
    state_violations = list(facade.engine.state.violations)
    record(
        "A_interpret_response_writes_violations",
        state_violations == synthetic_violations,
        {
            "expected_len": len(synthetic_violations),
            "got_len": len(state_violations),
            "types_got": [v.get("type") for v in state_violations],
        },
    )

    # ------------------------------------------------------------------
    # Build the batches once, reuse the result for B-E.
    # ------------------------------------------------------------------
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    batches, labels = overlay_geometry._build_violation_batches(
        scene, depsgraph, synthetic_violations,
    )
    log(f"built batches={len(batches)} labels={len(labels)}")

    # The contract returns (batch, primitive, color) triples. Index by
    # color so we can tie each batch back to its source violation type.
    expected_colors = {
        "self_intersection": (1.0, 0.1, 0.1, 0.85),
        "contact_offset": (1.0, 0.5, 0.0, 0.85),
        "wall": (1.0, 0.1, 0.1, 0.9),
        "sphere": (0.8, 0.1, 1.0, 0.9),
        "runtime_intersection": (1.0, 0.05, 0.05, 0.9),
    }
    expected_label_text = {
        "self_intersection": "Self-Intersections",
        "contact_offset": "Contact-Offset Violations",
        "wall": "Wall Violations",
        "sphere": "Sphere Violations",
        "runtime_intersection": "Runtime Intersection(s)",
    }
    batch_colors = [tuple(round(c, 4) for c in trip[2]) for trip in batches]
    batch_prims = [trip[1] for trip in batches]

    # ------------------------------------------------------------------
    # B. One batch per known type with the expected color.
    # ------------------------------------------------------------------
    seen_color_for = {}
    for vtype, want in expected_colors.items():
        rounded = tuple(round(c, 4) for c in want)
        seen_color_for[vtype] = rounded in batch_colors
    five_known = all(seen_color_for.values())
    # Wall is points; everything else is triangles. A point primitive on
    # the wall batch is the strongest signal that classification didn't
    # collapse wall and self-intersection (both red).
    wall_color = tuple(round(c, 4) for c in expected_colors["wall"])
    sphere_color = tuple(round(c, 4) for c in expected_colors["sphere"])
    wall_is_points = any(
        batch_prims[i] == "POINTS" and batch_colors[i] == wall_color
        for i in range(len(batches))
    )
    sphere_is_points = any(
        batch_prims[i] == "POINTS" and batch_colors[i] == sphere_color
        for i in range(len(batches))
    )
    record(
        "B_batch_per_known_type",
        five_known and wall_is_points and sphere_is_points,
        {
            "seen_color_for": seen_color_for,
            "batch_count": len(batches),
            "wall_is_points": wall_is_points,
            "sphere_is_points": sphere_is_points,
            "batch_colors": batch_colors,
            "batch_prims": batch_prims,
        },
    )

    # ------------------------------------------------------------------
    # C. Unknown type ('frobnitz') yields no batch / no label.
    # ------------------------------------------------------------------
    # Drive the builder with ONLY the unknown payload to isolate. The
    # full-payload run above already gave us len()=5 batches (one per
    # known type), so the fallthrough is silent in the mixed case too,
    # but a dedicated single-entry call makes the assertion crisp.
    only_unknown = [
        {"type": "frobnitz", "count": 99, "note": "lone unknown"},
    ]
    u_batches, u_labels = overlay_geometry._build_violation_batches(
        scene, depsgraph, only_unknown,
    )
    record(
        "C_unknown_type_fallthrough",
        len(u_batches) == 0 and len(u_labels) == 0 and len(batches) == 5,
        {
            "unknown_only_batches": len(u_batches),
            "unknown_only_labels": len(u_labels),
            "mixed_run_batches": len(batches),
        },
    )

    # ------------------------------------------------------------------
    # D. Every batch's matching label includes the source 'count'.
    # ------------------------------------------------------------------
    # Index labels by their literal text. Wall and self_intersection
    # share an RGB triple (both red), so a color-keyed dict would
    # collapse them into one entry; the LABELS map in
    # _build_violation_batches gives each type a unique text suffix,
    # so the per-text mapping is safe.
    label_text_set = {lab["text"] for lab in labels}
    label_count_ok = {}
    for v in synthetic_violations:
        vtype = v.get("type")
        if vtype not in expected_colors:
            continue
        expected_text = (
            f"{v['count']} {expected_label_text[vtype]}"
        )
        label_count_ok[vtype] = expected_text in label_text_set
    record(
        "D_labels_have_count",
        all(label_count_ok.values()) and len(label_count_ok) == 5,
        {
            "label_count_ok": label_count_ok,
            "label_texts": sorted(label_text_set),
        },
    )

    # ------------------------------------------------------------------
    # E. _solver_to_blender axis swap is reflected in label positions.
    # ------------------------------------------------------------------
    # ``_solver_to_blender`` is defined as a nested function inside
    # ``_build_violation_batches`` and isn't importable at module
    # scope, so we replicate the formula here:
    #     Vector((pos[0], -pos[2], pos[1]))
    # The wall violation's first vertex is solver (4, 2, 3); the swap
    # produces Blender (4, -3, 2) and the label code adds +0.05 z, so
    # we expect (4, -3, 2.05). The sphere's first vertex (5, 1, 2)
    # maps to (5, -2, 1) plus +0.05 z = (5, -2, 1.05). Both expected
    # positions have a non-trivial y (which was zero in solver space)
    # and a swapped z, so a no-op identity mapping would fail.
    eps = 1e-5

    def _approx(actual, expected):
        return all(abs(actual[i] - expected[i]) < eps for i in range(3))

    # Find labels by text (wall and self_intersection share an RGB
    # color, so a color-keyed lookup is ambiguous), then check their
    # 3D positions against the swap formula.
    wall_pos_ok = False
    sphere_pos_ok = False
    wall_actual = None
    sphere_actual = None
    for lab in labels:
        if lab["text"] == "4 Wall Violations":
            wall_actual = tuple(lab["pos_3d"])
            wall_pos_ok = _approx(lab["pos_3d"], (4.0, -3.0, 2.05))
        elif lab["text"] == "2 Sphere Violations":
            sphere_actual = tuple(lab["pos_3d"])
            sphere_pos_ok = _approx(lab["pos_3d"], (5.0, -2.0, 1.05))

    record(
        "E_axis_swap_applied",
        wall_pos_ok and sphere_pos_ok,
        {
            "wall_pos_ok": wall_pos_ok,
            "sphere_pos_ok": sphere_pos_ok,
            "wall_pos_3d": wall_actual,
            "sphere_pos_3d": sphere_actual,
            "expected_wall": (4.0, -3.0, 2.05),
            "expected_sphere": (5.0, -2.0, 1.05),
        },
    )

    log(f"checks={len(result['checks'])} done")

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
