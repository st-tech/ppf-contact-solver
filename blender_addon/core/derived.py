# File: derived.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Single source of truth for cross-cutting derived facts.
#
# Before this module, the addon had ~7 independent computations of
# "is the sim running?", 3 for "what is the authoritative frame count?",
# and 3 for "what is the current project name?".  Every new field
# derivation was another drift source.  Move all of that here; every
# other caller just imports and calls.
#
# All functions here are pure: they read from ``AppState`` (and
# optionally a ``ResponseCache``) and return a value.  They never
# mutate, never call bpy.ops, never do I/O.

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from .state import Activity, AppState, Phase, Server, Solver

if TYPE_CHECKING:
    from .cache import ResponseCache


# ---------------------------------------------------------------------------
# Sim running
# ---------------------------------------------------------------------------


def is_sim_running(state: AppState) -> bool:
    """True when the *remote* solver is actively producing frames.

    Combines Phase (must be ONLINE), Server (must be RUNNING), and Solver
    (one of the actively-producing states).  Accepts nothing else — no
    fallback to parsing response dicts.  If a caller can only see the
    raw response, use ``is_sim_running_from_response`` explicitly.
    """
    if state.phase != Phase.ONLINE:
        return False
    if state.server != Server.RUNNING:
        return False
    return state.solver in (Solver.RUNNING, Solver.STARTING, Solver.SAVING)


def is_sim_running_from_response(response: Optional[dict]) -> bool:
    """True when the raw server response indicates active simulation
    (narrow sense: BUSY or SAVE_AND_QUIT, not BUILDING).

    Used by UI panels that read ``com.response`` directly and need to
    choose between "live statistics" and "final statistics" displays.
    """
    if not response:
        return False
    return response.get("status") in {"BUSY", "SAVE_AND_QUIT"}


def is_server_busy_from_response(response: Optional[dict]) -> bool:
    """True when the server is doing anything the client shouldn't
    interrupt: actively simulating OR building.

    Use this in operator ``poll()`` methods that need to refuse user
    input while the server is engaged.  The strict "is it actively
    producing frames" check is ``is_sim_running_from_response``.
    """
    if not response:
        return False
    return response.get("status") in {"BUSY", "SAVE_AND_QUIT", "BUILDING"}


def is_client_busy(state: AppState) -> bool:
    """True when the *client* is blocking on local I/O."""
    return state.activity != Activity.IDLE


# ---------------------------------------------------------------------------
# Frame count
# ---------------------------------------------------------------------------


def authoritative_frame_count(
    state: AppState,
    scene_frame_count: int,
    discovered_frames: Optional[int] = None,
) -> int:
    """Resolve the "how many frames do we expect?" question.

    Policy, in decreasing priority:
    1. If Fetch has discovered actual files on the remote, that count wins.
       Discovery walks the filesystem; it's the most truthful number.
    2. Otherwise, the server's reported frame index (state.frame) if > 0.
    3. Fall back to the scene property ``frame_count`` set by the user.

    The policy exists so that a user who ran a longer sim and resumes
    with a shorter frame_count doesn't get confused by the stale large
    server response.  Fetch discovery wins because it reflects actual
    on-remote files.
    """
    if discovered_frames is not None and discovered_frames > 0:
        return discovered_frames
    if state.frame > 0:
        return state.frame
    return max(scene_frame_count, 0)


# ---------------------------------------------------------------------------
# Project name
# ---------------------------------------------------------------------------


def current_project_name(state: AppState, runner_project_name: Optional[str]) -> str:
    """Resolve the current project name.

    The runner's project name is set by the user via
    ``communicator.set_project_name()`` or via scene-property sync.
    That is authoritative while connected.  If empty, fall back to
    the state's remote_root basename.
    """
    if runner_project_name:
        return runner_project_name
    import os
    if state.remote_root:
        return os.path.basename(state.remote_root.rstrip("/"))
    return ""


# ---------------------------------------------------------------------------
# Session id
# ---------------------------------------------------------------------------


def current_session_id(state: AppState) -> str:
    """Session id of the active connection, or empty string when offline."""
    return state.session_id or ""


def session_mismatch(state: AppState, expected: str) -> bool:
    """True when the state's session id is non-empty AND differs from ``expected``.

    Used by reconcile-on-reconnect logic to spot orphaned sims.
    """
    return bool(state.session_id) and bool(expected) and state.session_id != expected
