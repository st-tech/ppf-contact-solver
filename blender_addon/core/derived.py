# File: derived.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Pure helpers that classify a raw server response. They never mutate,
# never call bpy.ops, never do I/O.

from __future__ import annotations

from typing import Optional

from .protocol import SERVER_BUSY_STATUSES, SIM_RUNNING_STATUSES


# ---------------------------------------------------------------------------
# Sim running
# ---------------------------------------------------------------------------


def is_sim_running_from_response(response: Optional[dict]) -> bool:
    """True when the raw server response indicates active simulation
    (narrow sense: BUSY or SAVE_AND_QUIT, not BUILDING).

    Used by UI panels that read ``com.response`` directly and need to
    choose between "live statistics" and "final statistics" displays.
    """
    if not response:
        return False
    return response.get("status") in SIM_RUNNING_STATUSES


def is_server_busy_from_response(response: Optional[dict]) -> bool:
    """True when the server is doing anything the client shouldn't
    interrupt: actively simulating OR building.

    Use this in operator ``poll()`` methods that need to refuse user
    input while the server is engaged.  The strict "is it actively
    producing frames" check is ``is_sim_running_from_response``.
    """
    if not response:
        return False
    return response.get("status") in SERVER_BUSY_STATUSES


