# File: status.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Status enums, dataclasses, and utilities extracted from core/client.py.

import threading
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


# NOTE: is_running / is_simulating have been removed.  Use
# ``core.derived.is_server_busy_from_response`` (server is doing
# anything) or ``core.derived.is_sim_running_from_response`` (narrow:
# actively simulating) instead, or the AppState-aware
# ``core.derived.is_sim_running(state)`` when a state snapshot is in
# scope.


@dataclass
class ConnectionInfo:
    type: str
    current_directory: str
    remote_root: str
    instance: Any
    server_running: bool
    container: str
    thread: threading.Thread | None
    server_thread: threading.Thread | None
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
class CommunicatorLocks:
    """Domain-specific locks for the Communicator.

    Each lock protects a group of related fields:
        task: _task, _interrupt
        status: _com (CommunicatorInfo -- progress, status, errors, response)
        connection: _connection (ConnectionInfo)
        animation: _animation, _fetched
        data: _data, _exec_output

    When acquiring multiple locks, always use alphabetical order
    to prevent deadlocks: animation -> connection -> data -> status -> task
    """

    task: threading.RLock
    status: threading.RLock
    connection: threading.RLock
    animation: threading.RLock
    data: threading.RLock

    def __init__(self):
        self.task = threading.RLock()
        self.status = threading.RLock()
        self.connection = threading.RLock()
        self.animation = threading.RLock()
        self.data = threading.RLock()


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
