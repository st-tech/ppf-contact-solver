# File: status.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Status enums, dataclasses, and utilities extracted from core/client.py.

import time

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy

from ..models.defaults import DEFAULT_SERVER_PORT


class BytesPerSecondCalculator:
    def __init__(self, window_seconds: float = 3.0):
        self.window_seconds = window_seconds
        self.samples = []

    def add_sample(self, bytes_processed: int):
        now = time.time()
        self.samples.append((now, bytes_processed))
        self.samples = [
            (t, b) for t, b in self.samples if now - t <= self.window_seconds
        ]

    def get_bytes_per_second(self) -> float:
        if len(self.samples) <= 1:
            return 0.0
        t0, b0 = self.samples[0]
        t1, b1 = self.samples[-1]
        elapsed = t1 - t0
        return (b1 - b0) / elapsed if elapsed > 0 else 0.0


# One-line summary per crash cause, keyed by the server's ``crash_kind`` tag
# (``ppf_cts_formats::status::CrashKind::tag``).
#
# Every value is a translation key, so it must stay free of numbers, paths,
# process ids and error strings. Those belong in the detail line the panel
# draws untranslated beneath it; folding them in here would make the key set
# unbounded and leave every locale falling back to English.
#
# A tag with no entry falls back to the generic line, so a newer server can
# name a cause this addon does not know without the panel going blank.
CRASH_CAUSE_SUMMARY: dict[str, str] = {
    "intersection": "Intersection detected",
    "ccd": "Continuous collision detection failed",
    "cg": "Linear solver failed to converge",
    "newton_stall": "Newton solve made no progress (over-constrained configuration)",
    "pin_infeasible": "A pinned vertex is driven into a collider it cannot yield to",
    "overlapping_start": "Two surfaces start the step already touching or overlapping",
    "init_intersection": "Intersection in the initial configuration",
    "oom": "Out of GPU memory",
    "cuda_driver": "Unrecoverable CUDA runtime or driver error",
    "watchdog_timeout": "A GPU kernel ran past the operating system's watchdog timeout",
    "panic": "Solver host panicked",
    "solver_invariant": "Solver stopped on a failed internal check",
    "device_assert": "A solver invariant failed on the GPU",
    "killed_by_signal": "The solver process was killed before it could report",
    "library_load_failed": "A required library could not be loaded",
    "launch_failed": "The solver exited before it started",
    "unknown_abrupt": "Solver exited abnormally without reporting a cause",
}

# Shown for a tag this build does not recognize.
CRASH_CAUSE_FALLBACK = "Solver exited abnormally without reporting a cause"


def crash_cause_summary(kind: str) -> str:
    """Untranslated one-line summary for a ``crash_kind`` tag."""
    return CRASH_CAUSE_SUMMARY.get(kind, CRASH_CAUSE_FALLBACK)


class RemoteStatus(Enum):
    DISCONNECTED = "Disconnected"
    CONNECTING = "Connecting..."
    WAITING_FOR_DATA = "Waiting for Data"
    WAITING_FOR_BUILD = "Waiting for Build"
    SERVER_NOT_RUNNING = "Waiting for Server Start..."
    SERVER_LAUNCHING = "Server Launching..."
    STOPPING_SERVER = "Stopping Server..."
    PROTOCOL_VERSION_MISMATCH = "Protocol Version Mismatch"
    BUILDING = "Building Scene..."
    SIMULATION_IN_PROGRESS = "Simulation Running..."
    FETCHING = "Fetching Animation..."
    APPLYING_DOWNLOADED_ANIM = "Applying Downloaded Animation..."
    DATA_SENDING = "Data Sending..."
    DATA_RECEIVING = "Data Receiving..."
    EXECUTING_COMMAND = "Executing Command..."
    SAVING_IN_PROGRESS = "Saving In Progress..."
    READY = "Ready to Run"
    RESUMABLE = "Resumable"
    STARTING_SOLVER = "Initializing..."
    SIMULATION_FAILED = "Simulation Failed"
    ERROR = "Error"
    ABORTING = "Aborting..."
    UNKNOWN = "Unknown Status"

    def in_progress(self):
        return self in {
            RemoteStatus.BUILDING,
            RemoteStatus.SIMULATION_IN_PROGRESS,
            RemoteStatus.SAVING_IN_PROGRESS,
            RemoteStatus.DATA_SENDING,
            RemoteStatus.DATA_RECEIVING,
            RemoteStatus.FETCHING,
            RemoteStatus.APPLYING_DOWNLOADED_ANIM,
            RemoteStatus.STARTING_SOLVER,
        }

    def abortable(self):
        return self in {
            RemoteStatus.DATA_SENDING,
            RemoteStatus.DATA_RECEIVING,
            RemoteStatus.FETCHING,
            RemoteStatus.APPLYING_DOWNLOADED_ANIM,
        }

    def ready(self):
        """Check if the protocol version is compatible."""
        return self not in {
            RemoteStatus.PROTOCOL_VERSION_MISMATCH,
            RemoteStatus.ERROR,
            RemoteStatus.UNKNOWN,
        }

    @property
    def icon(self):
        """Return the icon name associated with the status."""
        icons = {
            RemoteStatus.DISCONNECTED: "UNLINKED",
            RemoteStatus.WAITING_FOR_DATA: "LINKED",
            RemoteStatus.BUILDING: "SETTINGS",
            RemoteStatus.SIMULATION_IN_PROGRESS: "PLAY",
            RemoteStatus.DATA_SENDING: "EXPORT",
            RemoteStatus.DATA_RECEIVING: "IMPORT",
            RemoteStatus.SIMULATION_FAILED: "ERROR",
            RemoteStatus.ERROR: "ERROR",
            RemoteStatus.UNKNOWN: "ERROR",
            RemoteStatus.PROTOCOL_VERSION_MISMATCH: "ERROR",
        }
        return icons.get(self, "INFO_LARGE")


@dataclass
class ConnectionInfo:
    type: str
    current_directory: str
    remote_root: str
    instance: Any
    server_running: bool
    container: str
    server_port: int

    def __init__(self):
        self.clear()

    def clear(self):
        """Clear the connection information."""
        self.type = ""
        self.current_directory = ""
        self.remote_root = ""
        self.instance = None
        self.server_running = False
        self.container = ""
        self.server_port = DEFAULT_SERVER_PORT  # Updated when server starts


@dataclass
class CommunicatorInfo:
    status: RemoteStatus = RemoteStatus.DISCONNECTED
    message: str = ""
    error: str = ""
    server_error: str = ""
    violations: list = field(default_factory=list)
    response: dict = field(default_factory=dict)
    progress: float = 0.0
    traffic: str = ""
    def clear_traffic(self):
        self.traffic = ""
        self.progress = 0.0


@dataclass
class AnimationData:
    map: dict[str, numpy.ndarray]
    frame: list[tuple[int, numpy.ndarray]]
    surface_map: dict
    total_frames: int = 0
    applied_frames: int = 0

    def clear(self):
        """Clear the animation data."""
        self.map.clear()
        self.frame.clear()
        self.surface_map.clear()
        self.total_frames = 0
        self.applied_frames = 0
