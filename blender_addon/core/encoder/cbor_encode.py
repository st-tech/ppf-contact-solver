# File: blender_addon/core/encoder/cbor_encode.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
"""CBOR upload encoder.

Used by ``encode_obj`` (mesh.py) and ``encode_param`` (params.py).
Output wraps the producer dict in the schema-version envelope
``{version, kind, payload}`` defined by
``crates/ppf-cts-formats/src/envelope.rs`` and consumed by the Rust
server.

cbor2 is imported lazily so addon load doesn't fail on installs where
the wheel isn't yet vendored under ``blender_addon/lib/``. Users
install it through the existing UI install-ops flow (see
ui/install_ops.py) the same way paramiko / docker get installed.
"""

from __future__ import annotations

import numpy as np

SCHEMA_VERSION = 1


def _numpy_default(encoder, value):
    """cbor2 fallback for non-natively-encodable types.

    The producer dicts contain numpy arrays (vert / face / edge /
    transform) and numpy scalars (np.float32(...) for sim params).
    cbor2 doesn't know either, so we route through ``.tolist()`` /
    ``.item()`` to get plain Python types — which is also what the
    Rust serde schema decodes into (Vec<[f32; N]>, primitives).

    Tuples are already CBOR-encodable as arrays without help, so we
    don't override them here.
    """
    if isinstance(value, np.ndarray):
        encoder.encode(value.tolist())
        return
    if isinstance(value, np.generic):
        encoder.encode(value.item())
        return
    raise TypeError(
        f"CBOR encoder cannot serialize {type(value).__name__} "
        f"(value: {value!r})"
    )


def dumps_envelope(kind: str, payload) -> bytes:
    """Wrap ``payload`` in the schema-version envelope and emit CBOR bytes.

    Args:
        kind: payload type tag, e.g. ``"Scene"`` or ``"Param"``. Must
            match the consumer's expected kind or it'll be rejected.
        payload: producer dict (data tree from ``_build_obj_data`` or
            ``_build_param_dict``). Numpy arrays and scalars inside
            are converted via ``_numpy_default``.

    Returns:
        CBOR-encoded bytes ready for ``upload_atomic``.
    """
    from ..module import get_cbor2

    cbor2 = get_cbor2()

    envelope = {"version": SCHEMA_VERSION, "kind": kind, "payload": payload}
    return cbor2.dumps(envelope, default=_numpy_default)
