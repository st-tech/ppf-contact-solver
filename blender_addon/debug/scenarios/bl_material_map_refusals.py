# File: scenarios/bl_material_map_refusals.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Everything a spatial material map REFUSES, and the one source it accepts.
#
# Each of these was a defect that shipped a plausible-but-wrong result rather
# than an error, so each check exists to keep a specific silent path loud.
#
#   A_attribute_source_is_read: an ATTRIBUTE source works at all. It is read
#      off the EVALUATED mesh, which is where Geometry Nodes writes one; the
#      original datablock does not carry it.
#   B_non_finite_weight_refused: a NaN weight names the object, the source and
#      the vertex. Clamping made `min(1, max(0, nan))` read as 0.0, which is
#      "use the base value", so the artist got the unmapped result and no word.
#   C_negative_target_refused: a negative target names the parameter. For
#      strain-limit it does not tighten the limit, it switches the limiter off
#      on exactly the painted faces, because the solver's gate is `> 0`.
#   D_pressure_is_not_mappable: `pressure` is refused by name. Its per-face
#      potential is translation-variant, so a varying pressure means something
#      different depending on where the object sits (measured: per-vertex
#      forces change 275% at 1 m and 2202% at 8 m, while a uniform pressure is
#      exactly invariant). The enum id stays reserved.
#   E_rod_map_refused: a rod carries no element table a per-vertex map can be
#      reduced over, so no key is offered for one.
#   F_rod_keyframe_refused: the per-frame material tables are written for
#      TRIANGLES only, so a rod's sampled schedule reached no element table.
#      It was shipped and silently ignored for the whole solve.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True

# RUNS ON THE REAL BACKEND, established by RUNNING it rather than by reading
# it. A full rig sweep against a CPU build passed it, and that run is the
# evidence this line rests on.
BACKENDS = ("real",)


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})


def encode_error():
    try:
        dh.encode_payload()
        return ""
    except Exception as exc:
        return str(exc)


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="MapMesh")
    root = dh.configure_state(project_name="map_refusals", frame_count=4)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    pin = cloth.create_pin(plane.name, "AllPin")
    pin.move_by(delta=(0.05, 0.0, 0.0), frame_start=1, frame_end=3,
                transition="LINEAR")
    group = root.object_group_0

    n_vert = len(plane.data.vertices)
    attr = plane.data.attributes.new(name="Painted", type="FLOAT", domain="POINT")
    for i in range(n_vert):
        attr.data[i].value = i / max(1, n_vert - 1)

    entry = group.material_maps.add()
    entry.parameter = "bend"
    entry.source_type = "ATTRIBUTE"
    entry.source_name = "Painted"
    entry.target_value = 500.0
    group.material_maps_index = 0

    err_a = encode_error()
    blob = dh.decode_addon_blob(dh.encode_payload()[1]) if not err_a else {}
    weights = []
    for e in blob.get("group", []):
        mm = e[0].get("material-maps") or {}
        if "bend" in mm:
            weights = list((mm["bend"].get("weights") or {}).values())[0]
    dh.record(
        "A_attribute_source_is_read",
        err_a == "" and len(weights) == n_vert and max(weights) > 0.9,
        {"error": err_a[:200], "n_weights": len(weights),
         "max": max(weights) if weights else None},
    )

    # ----- B: a non-finite weight is named, not clamped to "use the base"
    attr.data[3].value = float("nan")
    err_b = encode_error()
    dh.record(
        "B_non_finite_weight_refused",
        "not a finite number" in err_b and "vertex 3" in err_b
        and "Painted" in err_b,
        {"error": err_b[:300]},
    )
    attr.data[3].value = 0.5

    # ----- C: a negative target is nonsense for every mappable parameter
    entry.target_value = -1.0
    err_c = encode_error()
    dh.record(
        "C_negative_target_refused",
        "-1" in err_c and "bend" in err_c and "minimum" in err_c,
        {"error": err_c[:300]},
    )
    # A target of zero clears a non-negative test and still aborts the build
    # for a parameter whose own slider floor is above zero.
    entry.parameter = "young-mod"
    entry.target_value = 0.0
    err_c2 = encode_error()
    dh.record(
        "C2_target_below_the_property_floor_refused",
        "young-mod" in err_c2 and "minimum" in err_c2
        and "shell_young_modulus" in err_c2,
        {"error": err_c2[:300]},
    )
    entry.parameter = "bend"
    entry.target_value = 500.0

    # ----- D: pressure keeps its enum id and loses its base property
    models = __import__(pkg + ".models.material_maps", fromlist=["x"])
    entry.parameter = "pressure"
    err_d = encode_error()
    dh.record(
        "D_pressure_is_not_mappable",
        models.base_property("pressure", "SHELL") is None
        and dict((k, n) for k, _l, _d, n in models.MATERIAL_MAP_KEYS)["pressure"] == 6
        and "not available as a map" in err_d,
        {"error": err_d[:300],
         "base_prop": models.base_property("pressure", "SHELL")},
    )
    entry.parameter = "bend"

    # ----- E: a rod has no element table to reduce a map over
    group.object_type = "ROD"
    err_e = encode_error()
    dh.record(
        "E_rod_map_refused",
        "not available as a map" in err_e and "ROD" in err_e,
        {"error": err_e[:300]},
    )
    group.material_maps.clear()
    group.material_maps_index = 0

    # ----- F: a rod's material keyframes reach no element table
    group.bend = 1.0
    group.keyframe_insert(data_path="bend", frame=1)
    group.bend = 9.0
    group.keyframe_insert(data_path="bend", frame=4)
    err_f = encode_error()
    dh.record(
        "F_rod_keyframe_refused",
        "ROD" in err_f and "not animated" in err_f and "bend" in err_f,
        {"error": err_f[:300]},
    )

    # ----- I: a Transfer refused AFTER the geometry encode leaves the
    # topology stamp alone. Stamping at encode time records a scene as
    # transferred that never reached the wire, and the stale-topology warning
    # that would tell the artist to re-transfer never fires.
    encoder_mesh = __import__(pkg + ".core.encoder.mesh",
                              fromlist=["encode_obj"])
    encoder_params = __import__(pkg + ".core.encoder.params",
                                fromlist=["encode_param"])
    state = root.state
    state.set_mesh_hash({"marker": "before"})
    group.enable_strain_limit = False
    doomed = group.material_maps.add()
    doomed.parameter = "strain-limit"
    doomed.source_type = "VERTEX_GROUP"
    doomed.source_name = "Painted"
    doomed.target_value = 9.0
    encoder_mesh.encode_obj(bpy.context)          # succeeds, holds the stamp
    param_failed = ""
    try:
        encoder_params.encode_param(bpy.context)  # refused
    except Exception as param_exc:
        param_failed = str(param_exc)
    dh.record(
        "I_a_refused_transfer_does_not_stamp_the_topology",
        bool(param_failed)
        and state.get_mesh_hash().get("marker") == "before"
        and bool(state.pending_mesh_hash_json),
        {"param_error": param_failed[:200],
         "stamp": state.get_mesh_hash(),
         "held_pending": bool(state.pending_mesh_hash_json)},
    )
    group.material_maps.remove(len(group.material_maps) - 1)
    group.enable_strain_limit = True

    # ----- H: a keyframe Blender offers on a type it is not sampled for ----
    # Every material slider is drawn with Blender's own keyframe control, so a
    # type whose schedule reaches no element table has to say so. A STATIC
    # group's friction was dropped before the F-curve lookup and the solve ran
    # on the slider's static value with nothing reported.
    # The curves live on the SCENE: a group is a PropertyGroup, and its data
    # path is `zozo_contact_solver.object_group_N.<prop>`.
    bpy.context.scene.animation_data_clear()
    group.object_type = "STATIC"
    group.friction = 0.2
    group.keyframe_insert(data_path="friction", frame=1)
    group.friction = 0.8
    group.keyframe_insert(data_path="friction", frame=4)
    err_h = encode_error()
    dh.record(
        "H_static_keyframe_refused",
        "STATIC" in err_h and "friction" in err_h and "not animated" in err_h,
        {"error": err_h[:300]},
    )
    bpy.context.scene.animation_data_clear()
    group.object_type = "SHELL"

    # ----- G: the shrink factors cannot be authored non-positive -----------
    # They scale the rest tangent matrix the solver inverts, so a zero would
    # make it singular and be reported as collinear geometry. The solver
    # refuses one by name; this is the half an artist can reach.
    rna = group.bl_rna.properties
    dh.record(
        "G_shrink_factors_are_clamped_positive",
        rna["shrink_x"].hard_min > 0.0 and rna["shrink_y"].hard_min > 0.0,
        {"shrink_x_min": rna["shrink_x"].hard_min,
         "shrink_y_min": rna["shrink_y"].hard_min},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE.replace("<<LOCAL_PATH>>", REPO_ROOT_POSIX)


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 300.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
