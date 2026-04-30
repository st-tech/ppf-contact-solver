# File: session.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Session identity.  Every connection run gets a fresh, short id that is
# stamped on downstream artifacts (PC2 headers, MESH_CACHE modifier binds,
# remote output directories, project manifest) so the addon can tell
# "this sim" from "my previous sim" from "someone else's sim" when it
# reconnects, reopens a .blend, or recovers from a Blender crash.
#
# The id is not a secret or a crypto token — it only needs to be unique
# per connect so reconciliation can do UUID-equality comparison.

from __future__ import annotations

import uuid as _uuid

# Length is a readability tradeoff: 12 hex chars ~= 48 bits, vastly more
# than any user will generate in a single machine's lifetime and short
# enough to fit in a PC2 header's reserved region and in an error
# message without wrapping.
_ID_LEN = 12


def new_session_id() -> str:
    """Return a freshly-generated 12-hex-char session id."""
    return _uuid.uuid4().hex[:_ID_LEN]


def is_valid(session_id: str) -> bool:
    """True if *session_id* is well-formed (right length, all hex)."""
    if not isinstance(session_id, str):
        return False
    if len(session_id) != _ID_LEN:
        return False
    try:
        int(session_id, 16)
    except ValueError:
        return False
    return True
