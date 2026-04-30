# File: server/effects.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Typed, frozen effect classes for the server state machine.
# Effects describe side-effects to perform -- they carry no behavior.
# The ``EffectExecutor`` in engine.py is the only place they are executed.

from __future__ import annotations

from dataclasses import dataclass


class Effect:
    """Base class for all server effects."""


# ---------------------------------------------------------------------------
# Build effects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoCheckGPU(Effect):
    """Verify GPU availability before starting build."""


@dataclass(frozen=True)
class DoSpawnBuild(Effect):
    """Spawn background build thread."""


@dataclass(frozen=True)
class DoCancelBuild(Effect):
    """Signal the build thread to stop."""


# ---------------------------------------------------------------------------
# Solver effects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoLaunchSolver(Effect):
    """Start the solver subprocess.

    *resume_from*: None for fresh start, -1 for latest checkpoint.
    """
    resume_from: int | None = None


@dataclass(frozen=True)
class DoKillSolver(Effect):
    """Terminate the solver subprocess."""


@dataclass(frozen=True)
class DoRequestSaveAndQuit(Effect):
    """Create the save_and_quit flag file for the solver to detect."""


# ---------------------------------------------------------------------------
# Disk I/O effects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoLoadApp(Effect):
    """Deserialize app_state from disk."""
    name: str = ""
    root: str = ""


@dataclass(frozen=True)
class DoDeleteProjectData(Effect):
    """Delete the entire project directory."""
    root: str = ""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DoLog(Effect):
    """Write a message to the server log."""
    message: str = ""
