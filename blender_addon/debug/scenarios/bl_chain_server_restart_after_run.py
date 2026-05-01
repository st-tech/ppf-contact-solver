# Stop-after-run: locks in the addon-side half of the regression
#
#   "I simulated, stopped the server and started it again, and now
#    Scene Info still shows the simulated frames but Fetch All
#    Animation is disabled and status reads Waiting for Data."
#
# Specifically: when the user clicks Stop after a real simulation,
# the addon must drop its cached server response. Without that, the
# UI keeps rendering the pre-stop ``scene_info`` (with all 180
# simulated frames) right next to a "Waiting for Data" status.
#
# The companion fixes that this scenario does **not** exercise:
#
#   * Server-side ``_load_app`` rehydrating ``state.frame`` from
#     ``vert_*.bin`` on disk.
#   * win_native ``_do_launch_server`` re-spawning ``server.py``,
#     and the wmic -> PowerShell fallback in
#     ``kill_local_server_on_port`` -- Windows only, exercised live
#     against a running Blender on a Windows host.
#
# The local-backend rig cannot drive an end-to-end Stop -> Start
# cycle through the addon's launcher because that path expects the
# production venv at ``$HOME/.local/share/ppf-cts/venv`` and does
# not pass ``--debug`` to the relaunched ``server.py`` -- the rig
# spawns its server with ``--debug`` plus a custom Python interpreter
# from the orchestrator. We keep the scenario tight on what the rig
# can verify and rely on the unit test + the win_native live test
# for the rest.

from __future__ import annotations
from . import _chain_lib as cl
from . import _runner as r

NEEDS_BLENDER = True

SEQUENCE = (
    "connect", "transfer", "verify_idle",
    "run", "fetch", "verify_pc2",
    "stop_server", "verify_response_cache_empty",
)


def build_driver(ctx: r.ScenarioContext) -> str:
    return cl.build_chain_driver(
        ctx, project_name="chain_server_restart_after_run",
        mesh_name="ChainServerRestartMesh", sequence=SEQUENCE,
    )


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx, timeout=max(ctx.timeout, 240.0))
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
