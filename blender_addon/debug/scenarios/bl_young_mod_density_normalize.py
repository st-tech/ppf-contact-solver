# File: scenarios/bl_young_mod_density_normalize.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# End-to-end of the per-group ``young_mod_density_normalized`` toggle.
# The solver consumes "young-mod" as a density-normalized value (Pa/rho).
# When the toggle is ON (default) the encoder sends the field value
# unchanged; when OFF the field is a true Young's modulus in pascals and
# the encoder divides it by density before sending. This scenario proves
# both legs for a SHELL group (the conversion in core/encoder/params.py is
# type-agnostic, so SHELL exercises the same code path as SOLID/ROD):
#
#   1. Send-time witness: decode the exact CBOR param payload that
#      ``encode_param`` produces (what the solver receives over the wire)
#      and assert the "young-mod" value is unchanged with the toggle ON and
#      divided by density with it OFF, while "density" is always unchanged.
#   2. Emulated-solver witness: a successful ``build_and_wait`` for each
#      case. The emulated Rust binary builds the FixedScene from the sent
#      payload, and ``scene.rs`` asserts ``young_mod > 0`` per element while
#      doing so, so a broken conversion (zero/negative) would fail the build.

from __future__ import annotations

from . import _driver_lib as dl
from . import _runner as r
from . import REPO_ROOT_POSIX


NEEDS_BLENDER = True


_DRIVER_BODY = r"""
import traceback

result.setdefault("phases", [])
result.setdefault("errors", [])
result.setdefault("checks", {})
LOCAL_PATH = "<<LOCAL_PATH>>"
SERVER_PORT = <<SERVER_PORT>>
PROJECT_NAME = "<<PROJECT_NAME>>"

FIELD_YM = 2000.0    # value typed into the Young's Modulus field
DENSITY = 4.0        # group density; 2000 / 4 = 500 divides cleanly


def sent_young_mod(dh):
    # Encode the payload exactly as a real upload does and decode the CBOR
    # envelope the solver receives. group_params is a list of
    # (params_dict, objects, uuids); the first group's params_dict carries
    # "young-mod" and "density".
    data_bytes, param_bytes = dh.encode_payload()
    payload = dh.decode_addon_blob(param_bytes)
    groups = payload.get("group", [])
    if not groups:
        raise RuntimeError("param payload has no group entries")
    params = groups[0][0]
    return data_bytes, param_bytes, float(params["young-mod"]), float(params["density"])


def run_pass(dh, group, normalized, label):
    group.young_mod_density_normalized = normalized
    data_bytes, param_bytes, ym, dens = sent_young_mod(dh)
    expected = FIELD_YM if normalized else FIELD_YM / DENSITY
    dh.record(
        "%s_send_time_value" % label,
        abs(ym - expected) < 1e-3 and abs(dens - DENSITY) < 1e-3,
        {"normalized": normalized, "field_young_mod": FIELD_YM,
         "density": DENSITY, "sent_young_mod": ym, "sent_density": dens,
         "expected_young_mod": expected},
    )
    # A successful build witnesses that the emulated solver deserialized and
    # accepted the (possibly converted) young-mod that was sent.
    dh.build_and_wait(data_bytes, param_bytes,
                      message="young-mod-%s" % label)
    dh.record("%s_emulated_build_ok" % label, True,
              {"sent_young_mod": ym})
    return ym


try:
    dh = DriverHelpers(pkg, result)
    dh.log("setup_start")
    plane = dh.reset_scene_to_pinned_plane(name="YoungModMesh")
    dh.save_blend(PROBE_DIR, "young_mod_density.blend")
    root = dh.configure_state(project_name=PROJECT_NAME, frame_count=2)

    cloth = dh.api.solver.create_group("Cloth", "SHELL")
    cloth.add(plane.name)
    cloth.create_pin(plane.name, "AllPin")

    group = root.object_group_0
    group.shell_young_modulus = FIELD_YM
    group.shell_density = DENSITY
    dh.log("configured young_mod=%g density=%g" % (FIELD_YM, DENSITY))

    dh.connect_local(local_path=LOCAL_PATH, server_port=SERVER_PORT,
                     project_name=root.state.project_name)
    dh.log("connected")

    # Toggle ON (default): the field is already Pa/rho, sent unchanged.
    ym_on = run_pass(dh, group, True, "normalized_on")
    # Toggle OFF: the field is true Pa, divided by density before send.
    ym_off = run_pass(dh, group, False, "normalized_off")

    # The toggle must actually change the sent value, and each value must
    # land on its expected target.
    dh.record(
        "toggle_changes_sent_value",
        abs(ym_on - ym_off) > 1e-3
        and abs(ym_on - FIELD_YM) < 1e-3
        and abs(ym_off - FIELD_YM / DENSITY) < 1e-3,
        {"young_mod_on": ym_on, "young_mod_off": ym_off,
         "field": FIELD_YM, "density": DENSITY},
    )

except Exception as exc:
    result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["errors"].append(traceback.format_exc())
"""


_DRIVER_TEMPLATE = dl.DRIVER_LIB + _DRIVER_BODY


def build_driver(ctx: r.ScenarioContext) -> str:
    repo_root = REPO_ROOT_POSIX
    return (
        _DRIVER_TEMPLATE
        .replace("<<LOCAL_PATH>>", repo_root)
        .replace("<<SERVER_PORT>>", str(ctx.server_port))
        .replace("<<PROJECT_NAME>>", ctx.project_name)
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
