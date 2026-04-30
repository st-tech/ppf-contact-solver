# File: effects.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Typed, frozen effect classes for the event-driven state machine.
# Effects are pure *descriptions* of side-effects to perform -- they carry
# no behavior.  The ``EffectRunner`` (effect_runner.py) is the only place
# where effects are actually executed.
#
# Separating "deciding what to do" (transitions.py) from "doing it"
# (effect_runner.py) is the key architectural principle:
#   - transitions.py is a pure function (testable, no I/O)
#   - effect_runner.py is the impure shell (real I/O, threads)

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class Effect:
    """Base class for all effects."""


# ---------------------------------------------------------------------------
# Connection effects (run on background thread)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoConnect(Effect):
    """Open a connection to the remote host."""
    backend_type: str = ""       # "ssh" | "docker" | "local" | "win_native"
    config: dict = field(default_factory=dict)
    server_port: int = 0


@dataclass(frozen=True)
class DoDisconnect(Effect):
    """Close the current connection."""


@dataclass(frozen=True)
class DoValidateRemotePath(Effect):
    """Validate that required paths exist on the remote."""


# ---------------------------------------------------------------------------
# Server lifecycle effects (run on background thread)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoLaunchServer(Effect):
    """Start the remote solver server process."""


@dataclass(frozen=True)
class DoStopServer(Effect):
    """Stop (pkill) the remote solver server process."""


# ---------------------------------------------------------------------------
# Server query effects (run on background thread)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoQuery(Effect):
    """Send a JSON query to the solver server and dispatch ServerPolled."""
    request: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DoPollAfter(Effect):
    """Schedule a ``PollTick`` event after *delay* seconds.

    This replaces the ``while True: time.sleep(0.25); query()`` loops
    in the old ``_update_status()`` with a non-blocking timer.
    """
    delay: float = 0.25


# ---------------------------------------------------------------------------
# Data transfer effects (run on background thread)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoSendData(Effect):
    """Send binary data to *remote_path* on the server."""
    remote_path: str = ""
    data: bytes = b""


@dataclass(frozen=True)
class DoUploadAtomic(Effect):
    """Ship ``data.pickle`` and ``param.pickle`` to the server as one
    atomic transaction; on success dispatches ``UploadPipelineComplete``.

    Either payload may be empty to skip the corresponding file, but at
    least one must be non-empty. ``param_hash`` is the client-computed
    fingerprint of the param payload (see
    ``encoder.params.compute_param_hash``); the server stores it
    alongside ``upload_id.txt`` and echoes it on every status response.
    """
    project_root: str = ""
    data: bytes = b""
    param: bytes = b""
    data_hash: str = ""
    param_hash: str = ""


@dataclass(frozen=True)
class DoReceiveData(Effect):
    """Receive binary data from *remote_path* on the server.

    The result is dispatched as ``ReceiveDataComplete(data=...)`` so that the
    caller can pick it up.
    """
    remote_path: str = ""
    tag: str = ""   # optional label for the caller to identify the response


# ---------------------------------------------------------------------------
# Fetch / animation effects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoFetchMap(Effect):
    """Download map.pickle and surface_map.pickle from the remote."""
    root: str = ""


@dataclass(frozen=True)
class DoFetchFrames(Effect):
    """Download vertex data for a range of frames.

    If *only_latest* is True, only the most recent frame is fetched (used
    during live simulation polling).
    """
    root: str = ""
    frame_count: int = 0
    already_fetched: list[int] = field(default_factory=list)
    only_latest: bool = False


# ---------------------------------------------------------------------------
# Exec effects (run on background thread)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoExec(Effect):
    """Execute a shell command on the remote."""
    command: str = ""
    shell: bool = False


# ---------------------------------------------------------------------------
# Abort / interrupt effects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoSetInterrupt(Effect):
    """Set the interrupt flag so that in-flight I/O operations abort."""


@dataclass(frozen=True)
class DoClearInterrupt(Effect):
    """Clear the interrupt flag."""


# ---------------------------------------------------------------------------
# Animation buffer effects (main thread)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoClearAnimation(Effect):
    """Clear the animation frame buffer."""


@dataclass(frozen=True)
class DoResetAnimationBuffer(Effect):
    """Reset every animation buffer slot to pre-fetch state: drop queued
    frames, zero the applied/total counters, forget the animation map.
    Emitted at the start of a fetch so stale state from a previous
    (possibly aborted) run can't leak through."""


# ---------------------------------------------------------------------------
# Terminate / save-and-quit effects (background thread)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoTerminate(Effect):
    """Send terminate request to solver and wait for it to stop."""


@dataclass(frozen=True)
class DoSaveAndQuit(Effect):
    """Send save_and_quit request to solver."""


# ---------------------------------------------------------------------------
# UI / logging effects (main thread, immediate)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoLog(Effect):
    """Write a message to the addon console."""
    message: str = ""


@dataclass(frozen=True)
class DoRedrawUI(Effect):
    """Tag Blender 3D viewport areas for redraw."""
