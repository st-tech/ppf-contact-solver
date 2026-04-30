# File: encoder/__init__.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

from ..transform import _normalize_and_scale, _swap_axes, _to_solver

from .mesh import compute_data_hash, compute_mesh_hash, detect_stitch_edges, encode_obj  # noqa: E402
from .params import compute_param_hash, encode_param  # noqa: E402

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

    Encodes the requested payloads and computes the matching SHA-256
    fingerprint over the same dict tree. Both hashes ride along with
    the upload so the server can echo them back on every status
    response; the click-time handlers in ``SOLVER_OT_Run`` and
    ``SOLVER_OT_UpdateParams`` re-compute fresh hashes against the
    live scene to decide whether the user has drifted from the last
    upload.

    Returns ``(data, param, data_hash, param_hash)``. Either payload
    is ``b""`` (and its hash ``""``) when its ``want_*`` flag is False.
    """
    data = encode_obj(context) if want_data else b""
    param = encode_param(context) if want_param else b""
    data_hash = compute_data_hash(context) if data else ""
    param_hash = compute_param_hash(context) if param else ""
    return data, param, data_hash, param_hash
