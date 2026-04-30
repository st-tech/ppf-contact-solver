# File: server/state.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Immutable server state and orthogonal phase enums.
# The entire server state is a single frozen ``ServerState`` dataclass
# whose fields only change inside the pure ``server_transition()`` function.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class Data(Enum):
    """Whether scene data (data.pickle + param.pickle) has been uploaded."""
    EMPTY = auto()
    UPLOADED = auto()


class Build(Enum):
    """App build lifecycle."""
    NONE = auto()         # Never built or deleted
    BUILDING = auto()     # Build thread running
    BUILT = auto()        # App ready
    FAILED = auto()       # Build error


class Solver(Enum):
    """Solver subprocess lifecycle."""
    IDLE = auto()         # Not running
    RUNNING = auto()      # Subprocess active
    SAVING = auto()       # save_and_quit in progress
    FAILED = auto()       # Crashed


@dataclass(frozen=True)
class ServerState:
    """Complete, immutable snapshot of the server state.

    Replaced atomically via ``dataclasses.replace()`` inside the pure
    ``server_transition()`` function.  Nothing else should construct or
    mutate a ``ServerState``.
    """

    # Project identity
    name: str = ""
    root: str = ""

    # Orthogonal phases
    data: Data = Data.EMPTY
    build: Build = Build.NONE
    solver: Solver = Solver.IDLE

    # Upload identity: generated when data.pickle / param.pickle land; cleared
    # when the project is deleted. Persisted to upload_id.txt alongside the
    # data files so it survives server restarts as long as the files do.
    # The client pins this value when a build is in flight and uses it to
    # detect server-state loss (restart, external deletion, new upload from
    # another client) without relying on brittle status-string heuristics.
    upload_id: str = ""

    # Quick fingerprints of the last uploaded ``data.pickle`` and
    # ``param.pickle``. The client computes them (see
    # ``encoder.mesh.compute_data_hash`` /
    # ``encoder.params.compute_param_hash``), ships them on the
    # upload, and the server persists each next to ``upload_id.txt``.
    # Echoed on every status response so the UI can detect drift
    # between the live scene and what's currently on the server, and
    # so the "Run" click-time check can refuse a run when geometry or
    # parameters diverged.
    data_hash: str = ""
    param_hash: str = ""

    # Context
    resumable: bool = False
    frame: int = 0
    build_progress: float = 0.0
    build_info: str = ""
    error: str = ""
    violations: list = field(default_factory=list)

    # -- derived (no mutation) --

    @property
    def status_string(self) -> str:
        """Map to the protocol-level status string sent to clients.

        This replaces the old ``ServerState._status()`` method that
        checked the filesystem, globals, and log files on every request.
        """
        if self.build == Build.BUILDING:
            return "BUILDING"
        if self.solver == Solver.RUNNING:
            return "BUSY"
        if self.solver == Solver.SAVING:
            return "SAVE_AND_QUIT"
        if self.data == Data.EMPTY:
            return "NO_DATA"
        if self.build in (Build.NONE, Build.FAILED):
            return "NO_BUILD"
        if self.solver == Solver.FAILED:
            return "FAILED"
        if self.error:
            return "FAILED"
        if self.resumable:
            return "RESUMABLE"
        return "READY"

    @property
    def data_string(self) -> str:
        """Protocol-level data availability string."""
        return "READY" if self.data == Data.UPLOADED else "NO_DATA"
