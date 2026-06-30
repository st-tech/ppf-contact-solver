# File: core/encode_progress.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Progress state for the synchronous scene-encode phase.

Encoding the scene (validation, mesh extraction, CBOR) and the click-time
drift check run on Blender's main thread inside the Transfer / Run / Resume
operators, so they block the UI while they run. Before the upload/build is
even dispatched there is no server status to drive the panel's progress bar,
which is why a heavy scene used to show a frozen cursor with no feedback.

The staged-modal driver in :class:`core.async_op.AsyncOperator` publishes a
``(done, total, label)`` snapshot here between stages; the solver panel renders
it as a labeled progress bar. Only one encode runs at a time: the action
operators gate ``poll()`` on ``not is_active()`` here (``com.busy()`` alone is
not enough, since the engine is only dispatched on the final stage and stays
un-busy for the encode window), so this single-writer module global needs no
locking.
"""

_state = {"active": False, "done": 0, "total": 1, "label": ""}


def begin(total: int, label: str = "") -> None:
    """Start a new encode-progress run with ``total`` stages."""
    _state.update(active=True, done=0, total=max(1, int(total)), label=str(label))


def update(done: int, label: "str | None" = None) -> None:
    """Advance to ``done`` completed stages; optionally set the status label."""
    _state["done"] = int(done)
    if label is not None:
        _state["label"] = str(label)


def end() -> None:
    """Clear the run so the panel stops drawing the encode bar."""
    _state.update(active=False, done=0, total=1, label="")


def is_active() -> bool:
    """True while a staged encode is publishing progress."""
    return bool(_state["active"])


def snapshot() -> "tuple[int, int, str]":
    """``(done, total, label)`` for the UI; ``done`` is stages completed."""
    return (int(_state["done"]), int(_state["total"]), str(_state["label"]))
