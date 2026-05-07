# File: encoder/__init__.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from ..transform import _normalize_and_scale, _swap_axes, _to_solver

from .mesh import (  # noqa: E402
    compute_data_hash,
    compute_mesh_hash,
    detect_stitch_edges,
    encode_obj,
    encode_obj_with_hash,
)
from .params import (  # noqa: E402
    compute_param_hash,
    encode_param,
    encode_param_with_hash,
)

__all__ = [
    "_swap_axes",
    "_to_solver",
    "_normalize_and_scale",
    "encode_obj",
    "compute_data_hash",
    "compute_mesh_hash",
    "detect_stitch_edges",
    "encode_param",
    "compute_param_hash",
    "prepare_upload",
]


def prepare_upload(
    context,
    *,
    want_data: bool = True,
    want_param: bool = True,
) -> tuple[bytes, bytes, str, str]:
    """Single source of truth for what gets sent up the wire.

    Builds each payload tree once, encodes to CBOR, and hashes the
    encoded bytes so the upload-time hash and the click-time drift
    hash use the same algorithm. The server echoes the hashes on every
    status response; ``SOLVER_OT_Run`` and ``SOLVER_OT_UpdateParams``
    re-compute against the live scene to decide whether the user has
    drifted from the last upload.

    Returns ``(data, param, data_hash, param_hash)``. Either payload
    is ``b""`` (and its hash ``""``) when its ``want_*`` flag is False.
    """
    data, data_hash = encode_obj_with_hash(context) if want_data else (b"", "")
    param, param_hash = encode_param_with_hash(context) if want_param else (b"", "")
    return data, param, data_hash, param_hash
