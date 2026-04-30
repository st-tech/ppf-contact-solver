# File: state.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Immutable application state and orthogonal phase enums for the
# event-driven state machine.  The entire addon state is captured by
# a single frozen ``AppState`` dataclass whose fields are only ever
# changed inside the pure ``transition()`` function (see transitions.py).

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Orthogonal state dimensions
# ---------------------------------------------------------------------------

class Phase(Enum):
    """Connection lifecycle."""
    OFFLINE = auto()
    CONNECTING = auto()
    ONLINE = auto()


class Server(Enum):
    """Remote server process status."""
    UNKNOWN = auto()
    LAUNCHING = auto()
    RUNNING = auto()
    STOPPING = auto()


class Solver(Enum):
    """Solver readiness as reported by the remote server."""
    NO_DATA = auto()
    NO_BUILD = auto()
    BUILDING = auto()
    READY = auto()
    RESUMABLE = auto()
    STARTING = auto()
    RUNNING = auto()
    SAVING = auto()
    FAILED = auto()


class Activity(Enum):
    """What the *client* is currently doing (local blocking operation)."""
    IDLE = auto()
    BUILDING = auto()
    SENDING = auto()
    RECEIVING = auto()
    FETCHING = auto()
    APPLYING = auto()
    EXECUTING = auto()
    ABORTING = auto()


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AppState:
    """Complete, immutable snapshot of the addon state.

    Every field is replaced atomically via ``dataclasses.replace()`` inside the
    pure ``transition()`` function.  No part of the code base should mutate
    an ``AppState`` directly.
    """

    phase: Phase = Phase.OFFLINE
    server: Server = Server.UNKNOWN
    solver: Solver = Solver.NO_DATA
    activity: Activity = Activity.IDLE

    # Context carried across transitions
    remote_root: str = ""
    # Session id: stamped on every artifact this run produces (PC2 headers,
    # MESH_CACHE binds, remote directories, project manifest).  Fresh value
    # is generated on each Connected event, cleared on Disconnected.
    session_id: str = ""
    # Last upload_id reported by the server in its status response.
    # Mirrors what's on disk in the project's upload_id.txt.
    server_upload_id: str = ""
    # upload_id that the current in-flight build is tied to. Pinned on
    # the first poll after BuildRequested; any subsequent poll where
    # server_upload_id differs means the server state was reset
    # (restart, external deletion, or another client's upload) and the
    # build target is gone. Cleared when activity returns to IDLE.
    active_upload_id: str = ""
    # True when an upload is in flight whose successful completion
    # should immediately kick off a build. Set by BuildPipelineRequested,
    # cleared by UploadPipelineComplete (once the build request is
    # dispatched) or by any error / abort path that resets activity.
    pending_build: bool = False
    # Quick fingerprints of the data.pickle / param.pickle currently
    # on the server, mirrored from each status response. The Run /
    # UpdateParams operators recompute the live scene's hashes at
    # click time and compare against these to detect drift -- there
    # is no client-side cache. ``UpdateParams`` is always clickable
    # (subject to base connection / state guards) and just does the
    # right thing if the scene happens to be in sync; ``Run``
    # refuses with a "Click Update Params" message when params drift,
    # and a "Click Transfer" message when geometry drifts.
    server_data_hash: str = ""
    server_param_hash: str = ""
    error: str = ""
    server_error: str = ""
    violations: list = field(default_factory=list)
    message: str = ""
    progress: float = 0.0
    traffic: str = ""
    frame: int = 0
    version_ok: bool = True

    # -- derived helpers (no mutation) --

    @property
    def busy(self) -> bool:
        return self.activity != Activity.IDLE

    @property
    def can_operate(self) -> bool:
        """True when the user may issue solver commands."""
        return (
            self.phase == Phase.ONLINE
            and self.server == Server.RUNNING
            and not self.busy
            and self.version_ok
        )

    # -- backward compatibility --

    def to_remote_status(self) -> "RemoteStatus":
        """Map the composite state to the legacy flat ``RemoteStatus`` enum.

        This keeps existing UI panels, operator ``poll()`` methods, and MCP
        handlers working while the migration is in progress.
        """
        from .status import RemoteStatus

        # Phase takes priority
        if self.phase == Phase.OFFLINE:
            return RemoteStatus.DISCONNECTED
        if self.phase == Phase.CONNECTING:
            return RemoteStatus.CONNECTING
        if not self.version_ok:
            return RemoteStatus.PROTOCOL_VERSION_MISMATCH

        # Server lifecycle
        if self.server == Server.UNKNOWN:
            return RemoteStatus.SERVER_NOT_RUNNING
        if self.server == Server.LAUNCHING:
            return RemoteStatus.SERVER_LAUNCHING
        if self.server == Server.STOPPING:
            return RemoteStatus.STOPPING_SERVER

        # Activity overlay (client-side blocking operation)
        _ACTIVITY_MAP = {
            Activity.BUILDING: RemoteStatus.BUILDING,
            Activity.SENDING: RemoteStatus.DATA_SENDING,
            Activity.RECEIVING: RemoteStatus.DATA_RECEIVING,
            Activity.FETCHING: RemoteStatus.FETCHING,
            Activity.APPLYING: RemoteStatus.APPLYING_DOWNLOADED_ANIM,
            Activity.EXECUTING: RemoteStatus.EXECUTING_COMMAND,
            Activity.ABORTING: RemoteStatus.ABORTING,
        }
        if self.activity in _ACTIVITY_MAP:
            return _ACTIVITY_MAP[self.activity]

        # Solver readiness
        _SOLVER_MAP = {
            Solver.NO_DATA: RemoteStatus.WAITING_FOR_DATA,
            Solver.NO_BUILD: RemoteStatus.WAITING_FOR_BUILD,
            Solver.BUILDING: RemoteStatus.BUILDING,
            Solver.READY: RemoteStatus.READY,
            Solver.RESUMABLE: RemoteStatus.RESUMABLE,
            Solver.STARTING: RemoteStatus.STARTING_SOLVER,
            Solver.RUNNING: RemoteStatus.SIMULATION_IN_PROGRESS,
            Solver.SAVING: RemoteStatus.SAVING_IN_PROGRESS,
            Solver.FAILED: RemoteStatus.SIMULATION_FAILED,
        }
        if self.solver in _SOLVER_MAP:
            return _SOLVER_MAP[self.solver]

        # Fallback
        if self.error:
            return RemoteStatus.ERROR
        return RemoteStatus.UNKNOWN
