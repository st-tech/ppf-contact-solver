# File: server/events.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Typed, frozen event classes for the server state machine.
# Events come from three sources:
#   1. Client requests (parsed from socket protocol)
#   2. Build thread (progress callbacks, completion, failure)
#   3. Solver monitor (frame updates, exit detection, crash detection)

from __future__ import annotations

from dataclasses import dataclass, field


class Event:
    """Base class for all server events."""


# ---------------------------------------------------------------------------
# Client request events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProjectSelected(Event):
    """Client connected with a project name. Sets up project context.

    *upload_id* reconciles the on-disk identity: when the server restarts,
    or when a different process (Jupyter) wrote data to the project dir,
    the id is read from upload_id.txt and stamped here without disturbing
    the build state (the existing build artifact is still valid for that
    upload).
    """
    name: str = ""
    root: str = ""
    has_data: bool = False
    has_param: bool = False
    has_app: bool = False
    is_resumable: bool = False
    upload_id: str = ""
    data_hash: str = ""
    param_hash: str = ""


@dataclass(frozen=True)
class BuildRequested(Event):
    """Client requested a scene build."""


@dataclass(frozen=True)
class CancelBuildRequested(Event):
    """Client requested cancellation of an in-progress build."""


@dataclass(frozen=True)
class StartRequested(Event):
    """Client requested simulation start."""


@dataclass(frozen=True)
class ResumeRequested(Event):
    """Client requested simulation resume from checkpoint."""


@dataclass(frozen=True)
class TerminateRequested(Event):
    """Client requested solver termination."""


@dataclass(frozen=True)
class SaveAndQuitRequested(Event):
    """Client requested save-and-quit."""


@dataclass(frozen=True)
class DeleteRequested(Event):
    """Client requested project data deletion."""


@dataclass(frozen=True)
class UploadLanded(Event):
    """A data.pickle / param.pickle write completed. Fresh upload_id is
    stamped onto the state so any in-flight builds from other clients
    know their target has been replaced. ``data_hash`` and
    ``param_hash`` are the client-computed fingerprints that travelled
    in the upload header; the transition stores them so make_response
    can echo on every poll."""
    upload_id: str = ""
    data_hash: str = ""
    param_hash: str = ""
    has_data: bool = True
    has_param: bool = True


# ---------------------------------------------------------------------------
# Build thread events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildProgress(Event):
    """Build thread reports progress update."""
    progress: float = 0.0
    info: str = ""


@dataclass(frozen=True)
class BuildCompleted(Event):
    """Build thread finished successfully."""


@dataclass(frozen=True)
class BuildFailed(Event):
    """Build thread encountered an error."""
    error: str = ""
    violations: list = field(default_factory=list)


@dataclass(frozen=True)
class BuildCancelledEvent(Event):
    """Build thread was cancelled by user."""


@dataclass(frozen=True)
class GPUCheckFailed(Event):
    """GPU availability check failed before build."""
    error: str = ""


# ---------------------------------------------------------------------------
# Solver monitor events (from background monitor thread)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SolverFrameUpdated(Event):
    """Monitor detected a new frame in the output directory."""
    frame: int = 0


@dataclass(frozen=True)
class SolverFinished(Event):
    """Monitor detected solver subprocess exited cleanly."""
    resumable: bool = False


@dataclass(frozen=True)
class SolverCrashed(Event):
    """Monitor detected solver crash via error logs."""
    error: str = ""
    violations: list = field(default_factory=list)


@dataclass(frozen=True)
class SolverSaving(Event):
    """Monitor detected save_and_quit flag file."""


# ---------------------------------------------------------------------------
# Error events
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ErrorOccurred(Event):
    """A generic error from any subsystem."""
    error: str = ""
