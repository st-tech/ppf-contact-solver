# File: events.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Typed, frozen event classes for the event-driven state machine.
# Every external stimulus (user action, server response, background I/O
# completion, error) is modeled as an ``Event`` subclass.  Events carry
# data but have *no* behavior -- they are pure descriptions of "what happened."

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class Event:
    """Base class for all events in the state machine."""


# ---------------------------------------------------------------------------
# Connection events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConnectRequested(Event):
    """User requested a connection (SSH, Docker, local, win_native)."""
    backend_type: str = ""       # "ssh" | "docker" | "local" | "win_native"
    config: dict = field(default_factory=dict)
    server_port: int = 0


@dataclass(frozen=True)
class Connected(Event):
    """Background thread successfully established the connection."""
    remote_root: str = ""


@dataclass(frozen=True)
class ConnectionFailed(Event):
    """Background connect attempt failed."""
    error: str = ""


@dataclass(frozen=True)
class DisconnectRequested(Event):
    """User explicitly requested disconnection."""


@dataclass(frozen=True)
class Disconnected(Event):
    """Connection was closed (either by request or loss)."""


@dataclass(frozen=True)
class ConnectionLost(Event):
    """Connection dropped unexpectedly (e.g. SSH transport dead)."""


# ---------------------------------------------------------------------------
# Server lifecycle events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StartServerRequested(Event):
    """User requested remote server start."""


@dataclass(frozen=True)
class ServerLaunched(Event):
    """Server process is up and responding to queries."""


@dataclass(frozen=True)
class StopServerRequested(Event):
    """User requested remote server stop."""


@dataclass(frozen=True)
class ServerStopped(Event):
    """Server process has been killed."""


@dataclass(frozen=True)
class ServerLost(Event):
    """Server stopped responding (empty response to query)."""


# ---------------------------------------------------------------------------
# Server response
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ServerPolled(Event):
    """A server query returned a response dict.

    This is the central event that replaces the old ``_update_status()``
    conditional.  The pure ``transition()`` function interprets the response
    and derives the new solver state.
    """
    response: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Solver operation events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildRequested(Event):
    """User requested a scene build."""


@dataclass(frozen=True)
class RunRequested(Event):
    """User requested simulation start."""


@dataclass(frozen=True)
class ResumeRequested(Event):
    """User requested simulation resume."""


@dataclass(frozen=True)
class AbortRequested(Event):
    """User requested abort of current operation."""


@dataclass(frozen=True)
class TerminateRequested(Event):
    """User requested solver termination (kill simulation)."""


@dataclass(frozen=True)
class SaveAndQuitRequested(Event):
    """User requested save-and-quit."""


# ---------------------------------------------------------------------------
# Data transfer events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SendDataRequested(Event):
    """User requested sending data to the remote."""
    remote_path: str = ""
    data: bytes = b""
    message: str = ""


@dataclass(frozen=True)
class SendDataComplete(Event):
    """Data send finished successfully."""


@dataclass(frozen=True)
class BuildPipelineRequested(Event):
    """Atomic (upload data+param) → build pipeline.

    Replaces the old SOLVER_OT_Transfer modal orchestration: a single
    event carries the two pickle payloads, the engine drives upload and
    build as sequenced effects, and the UI just watches ``activity``.

    Either payload may be empty to skip that file (params-only update),
    but at least one must be non-empty. ``param_hash`` is the
    quick-fingerprint of the param payload computed via
    ``encoder.params.compute_param_hash``; the server stores it and
    echoes it on every status response so the UI can detect drift.
    """
    data: bytes = b""
    param: bytes = b""
    data_hash: str = ""
    param_hash: str = ""
    message: str = ""


@dataclass(frozen=True)
class UploadOnlyRequested(Event):
    """Atomic upload of data.pickle / param.pickle without chaining a build.

    Used by debug-only transfer tests that want to isolate the upload
    path from the build path.
    """
    data: bytes = b""
    param: bytes = b""
    data_hash: str = ""
    param_hash: str = ""
    message: str = ""


@dataclass(frozen=True)
class UploadPipelineComplete(Event):
    """Atomic upload finished. Whether a build follows is driven by
    ``state.pending_build`` which the transition sets from whichever
    upstream event (BuildPipelineRequested vs UploadOnlyRequested)
    initiated the upload."""


@dataclass(frozen=True)
class ReceiveDataRequested(Event):
    """User requested receiving data from the remote."""
    remote_path: str = ""
    message: str = ""


@dataclass(frozen=True)
class ReceiveDataComplete(Event):
    """Data receive finished successfully."""
    data: bytes = b""


# ---------------------------------------------------------------------------
# Fetch / animation events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FetchRequested(Event):
    """User requested fetching animation frames."""


@dataclass(frozen=True)
class FetchMapComplete(Event):
    """Animation map and surface_map downloads finished."""
    map_data: Any = None      # dict[str, numpy.ndarray]
    surface_map: Any = None   # dict


@dataclass(frozen=True)
class FrameFetched(Event):
    """One animation frame was downloaded from the remote."""
    frame_number: int = 0
    vertices: Any = None   # numpy.ndarray


@dataclass(frozen=True)
class FetchComplete(Event):
    """All requested frames have been downloaded."""
    total_frames: int = 0


@dataclass(frozen=True)
class FetchFailed(Event):
    """A fetch or apply step failed; state must leave FETCHING/APPLYING.

    Dispatched by worker-thread fetch paths (on precondition miss or
    exception), by the apply side (on exception), and by the liveness
    watchdog (on stuck progress).
    """
    reason: str = ""


@dataclass(frozen=True)
class AllFramesApplied(Event):
    """Main-thread animation application is done (buffer drained)."""


# ---------------------------------------------------------------------------
# Exec events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExecRequested(Event):
    """User requested command execution on the remote."""
    command: str = ""
    shell: bool = False


@dataclass(frozen=True)
class ExecComplete(Event):
    """Remote command execution finished."""
    output: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Query events (explicit user-initiated queries)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryRequested(Event):
    """User-initiated server query with optional payload."""
    request: dict[str, Any] = field(default_factory=dict)
    message: str = ""


# ---------------------------------------------------------------------------
# Progress / status events (from background threads)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProgressUpdated(Event):
    """Progress callback from a background I/O operation."""
    progress: float = 0.0
    traffic: str = ""


@dataclass(frozen=True)
class ErrorOccurred(Event):
    """An error happened in a background operation."""
    error: str = ""
    source: str = ""   # which subsystem raised it


# ---------------------------------------------------------------------------
# Internal scheduling events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PollTick(Event):
    """Scheduled re-query of the server (replaces the sleep-poll loop)."""
