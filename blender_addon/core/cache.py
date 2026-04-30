# File: cache.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Ephemeral caches that *derive from* the state machine but are NOT part
# of the immutable ``AppState`` snapshot.  These are the home for data
# that's only meaningful for UI display or transient bookkeeping —
# raw server response dicts, animation buffers, overlay batches — and
# should never flow back into a transition arm as "state".
#
# Separating cache from state matters because the transition function
# is pure: replaying a prior (state, event) pair must produce the same
# result.  If we carried the raw response dict in state, a stale retry
# could re-interpret yesterday's data as today's truth.  Keeping
# derived/display data out here makes that impossible.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResponseCache:
    """Holds the most recent raw server response for UI display.

    Mutable on purpose — this is a scratch pad, not part of the state
    machine.  Readers should treat it as "last-seen" with no freshness
    guarantee.  If ``empty`` is True, no response has been received
    since the runner was constructed or last cleared.
    """

    last_response: dict[str, Any] = field(default_factory=dict)
    empty: bool = True

    def record(self, response: dict[str, Any]) -> None:
        """Store the latest response.  Caller passes a shallow dict; we
        take a shallow copy so later mutations by the producer don't
        change our snapshot."""
        self.last_response = dict(response) if response else {}
        self.empty = not self.last_response

    def clear(self) -> None:
        """Drop the cached response (on disconnect, reset, reload)."""
        self.last_response = {}
        self.empty = True

    def get(self, key: str, default: Any = None) -> Any:
        """Shortcut for ``cache.last_response.get(key, default)``."""
        return self.last_response.get(key, default)
