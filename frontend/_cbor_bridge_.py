# File: frontend/_cbor_bridge_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
"""CBOR adapter for SceneDecoder / ParamDecoder.

The Blender addon emits the upload payload as a CBOR envelope
``{version, kind, payload}`` (see crates/ppf-cts-formats/src/envelope.rs).
The Python decoders in :mod:`frontend._decoder_` consume pickled
numpy-laden dicts; this adapter loads CBOR bytes and rehydrates only
the few fields that the decoder treats as numpy arrays.
"""

from __future__ import annotations

import pickle

import cbor2

import numpy as np

SCHEMA_VERSION = 1

KIND_SCENE = "Scene"
KIND_PARAM = "Param"
# Frontend-side producer kinds. ``KIND_VERTEX_MAP`` /
# ``KIND_SURFACE_MAP`` carry plain dicts of numpy arrays (``map.pickle``
# / ``surface_map.pickle`` payloads in ``frontend/_scene_.py:export_fixed``).
# ``KIND_APP_STATE`` (``app.pickle`` from ``App.save``) and
# ``KIND_FIXED_SESSION`` (``fixed_session.pickle`` from
# ``Session._save_fixed_session``) carry a structured CBOR map
# payload that exposes inspectable metadata (project name, info path,
# cmd path) alongside a ``pickle_blob`` bytes field used to rehydrate
# the deep Python class graph (``Session`` / ``ParamHolder`` /
# ``FixedScene`` / manager objects) which has no schema-level CBOR
# representation today. Older on-disk files (payload = raw bytes,
# produced by what is now the test-only :func:`dumps_pickled_envelope`)
# are still read back through the same :func:`loads_pickle_blob`
# dispatcher: it inspects the payload shape and accepts either bytes
# or a dict, so the byte-sniff at the call site does not need to know
# which producer wrote the file.
KIND_VERTEX_MAP = "VertexMap"
KIND_SURFACE_MAP = "SurfaceMap"
KIND_APP_STATE = "AppState"
KIND_FIXED_SESSION = "FixedSession"

# Inner format version of the surface-map payload (the frame-embedding
# maps produced by ``_bvh_.frame_mapping``). Bumped when that math changes
# in an incompatible way. v2: switched from in-plane barycentric weights to
# frame-embedding coefs. The frontend producers/consumers
# (``_mesh_._tetrahedralize`` cache, ``_scene_.FixedScene.export_fixed`` envelope,
# and :func:`loads_surface_map` below) all reference this constant. The addon
# consumer ``blender_addon/core/effect_runner.py`` does not import frontend/,
# so it keeps its own literal ``2`` that must be bumped in lockstep.
SURFACE_MAP_VERSION = 2


class CborSchemaError(RuntimeError):
    """Raised when a CBOR payload doesn't match the expected envelope."""


def is_cbor(blob: bytes) -> bool:
    """Cheap content sniff used by the decoders to dispatch.

    Pickle protocols 2+ start with the ``\\x80`` opcode. Pickle protocols
    0-1 start with ``(`` / ``]`` etc., but the addon hasn't emitted those
    in years; CBOR maps with at most 23 keys start with ``0xa0``-``0xb7``,
    well clear of any pickle prefix used in this codebase.
    """
    return bool(blob) and blob[0] != 0x80


def _numpy_default(encoder, value):
    """cbor2 fallback for numpy types.

    Mirrors ``blender_addon/core/encoder/cbor_encode.py`` so the
    producer dicts on the frontend side encode the same way the addon
    does: ndarray via ``.tolist()``, scalars via ``.item()``.
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
    """Wrap ``payload`` in the canonical envelope and emit CBOR bytes.

    Twin of :func:`blender_addon.core.encoder.cbor_encode.dumps_envelope`
    for use on the frontend side. Numpy arrays / scalars inside the
    payload are converted to plain Python types via ``_numpy_default``.
    """
    envelope = {"version": SCHEMA_VERSION, "kind": kind, "payload": payload}
    return cbor2.dumps(envelope, default=_numpy_default)


def loads_envelope(blob: bytes, expected_kind: str):
    """Decode a CBOR envelope and return its raw payload. **Test-only reader.**

    No type-specific rehydration; callers handle rehydration themselves
    (e.g. unpickling opaque bytes for ``KIND_APP_STATE``, casting
    ndarray-shaped lists for ``KIND_VERTEX_MAP``).

    Live decode paths use the kind-specific loaders instead
    (:func:`loads_scene`, :func:`loads_param`, :func:`loads_vertex_map`,
    :func:`loads_surface_map`, :func:`loads_pickle_blob`); the only
    callers of this generic reader are the round-trip tests, where it
    pairs with :func:`dumps_envelope`. Note that :func:`dumps_envelope`
    itself is NOT test-only: it is the live producer used by the
    frontend writers.
    """
    env = cbor2.loads(blob)
    return _unwrap(env, expected_kind)


def _unwrap(env: object, expected_kind: str) -> object:
    if not isinstance(env, dict):
        raise CborSchemaError(
            f"top-level CBOR must be a map, got {type(env).__name__}"
        )
    version = env.get("version")
    kind = env.get("kind")
    if version != SCHEMA_VERSION:
        raise CborSchemaError(
            f"schema version mismatch: payload={version}, this build expects "
            f"{SCHEMA_VERSION}"
        )
    if kind != expected_kind:
        raise CborSchemaError(
            f"kind mismatch: payload={kind!r}, expected={expected_kind!r}"
        )
    if "payload" not in env:
        raise CborSchemaError("envelope missing 'payload' key")
    return env["payload"]


def _rehydrate_scene_object(obj: dict) -> None:
    """Convert nested-list ndarray fields back into the dtypes the
    legacy SceneDecoder expects.

    Pickle preserved numpy dtypes; CBOR via cbor2 only carries plain
    Python lists, so the addon-side encoder calls ``arr.tolist()`` on
    every ndarray and we re-cast here. Keys not on the list are left
    untouched (string names, UUIDs, plain Python ints, etc.).
    """
    if obj.get("vert") is not None:
        obj["vert"] = np.asarray(obj["vert"], dtype=np.float32)
    if obj.get("bend_rest_vert") is not None:
        obj["bend_rest_vert"] = np.asarray(obj["bend_rest_vert"], dtype=np.float32)
    if obj.get("face") is not None:
        obj["face"] = np.asarray(obj["face"], dtype=np.uint32)
    if obj.get("edge") is not None:
        obj["edge"] = np.asarray(obj["edge"], dtype=np.uint32)
    if obj.get("transform") is not None:
        obj["transform"] = np.asarray(obj["transform"], dtype=np.float64)

    stitch = obj.get("stitch")
    if stitch is not None:
        # Producer encoded as a 2-element CBOR array (tuple-equivalent).
        # detect_stitch_edges in the addon returns
        #   (edge_array: int, weight_array: float).
        edges, weights = stitch
        edge_arr = (
            np.asarray(edges, dtype=np.uint32)
            if edges
            else np.zeros((0, 2), dtype=np.uint32)
        )
        weight_arr = (
            np.asarray(weights, dtype=np.float32)
            if weights
            else np.zeros((0,), dtype=np.float32)
        )
        obj["stitch"] = (edge_arr, weight_arr)


def loads_scene(blob: bytes) -> list:
    """Decode a CBOR scene envelope and rehydrate ndarray fields."""
    env = cbor2.loads(blob)
    payload = _unwrap(env, KIND_SCENE)
    if not isinstance(payload, list):
        raise CborSchemaError("scene payload must be a list of groups")
    for group in payload:
        if not isinstance(group, dict):
            raise CborSchemaError("scene group must be a map")
        for obj in group.get("object", []) or []:
            if isinstance(obj, dict):
                _rehydrate_scene_object(obj)
    return payload


def loads_param(blob: bytes) -> dict:
    """Decode a CBOR param envelope.

    The legacy ParamDecoder consumes plain Python types throughout
    (``np.asarray`` is invoked on demand inside ``apply_*`` methods);
    no broad rehydration is needed here. We just unwrap the envelope
    and pass the payload through.
    """
    env = cbor2.loads(blob)
    payload = _unwrap(env, KIND_PARAM)
    if not isinstance(payload, dict):
        raise CborSchemaError("param payload must be a map")
    return payload


def load_param_file(path: str) -> dict:
    """Read a ``param.pickle`` file and return its decoded payload.

    Owns the open + read + byte-sniff + dispatch that the decoder sites
    would otherwise repeat: CBOR envelopes route through
    :func:`loads_param`, legacy raw-pickle files through ``pickle.loads``.
    Extension validation stays at the call sites (it is not uniform: the
    populate() ftetwild peek does not validate), so this helper does not
    fold it in.
    """
    with open(path, "rb") as f:
        blob = f.read()
    if is_cbor(blob):
        return loads_param(blob)
    return pickle.loads(blob)


def load_scene_file(path: str) -> list:
    """Read a ``data.pickle`` file and return its decoded scene payload.

    Scene-side twin of :func:`load_param_file`: CBOR envelopes route
    through :func:`loads_scene`, legacy raw-pickle files through
    ``pickle.loads``. As with the param reader, extension validation is
    left to the call site.
    """
    with open(path, "rb") as f:
        blob = f.read()
    if is_cbor(blob):
        return loads_scene(blob)
    return pickle.loads(blob)


def loads_vertex_map(blob: bytes) -> dict:
    """Decode a CBOR vertex-map envelope (``map.pickle`` payload).

    The producer is ``FixedScene.export_fixed``: ``dict[str, ndarray]``
    keyed by object name, each ndarray a per-vertex int index map.
    On the wire, ndarrays become nested Python lists (cbor2 has no
    ndarray type); we re-cast to ``int64`` because the consumers
    (``effect_runner._ensure_anim_map``, ``_pin_fidelity_common``) treat
    these as integer index arrays.
    """
    env = cbor2.loads(blob)
    payload = _unwrap(env, KIND_VERTEX_MAP)
    if not isinstance(payload, dict):
        raise CborSchemaError("vertex-map payload must be a map")
    return {k: np.asarray(v, dtype=np.int64) for k, v in payload.items()}


def loads_surface_map(blob: bytes) -> dict:
    """Decode a CBOR surface-map envelope (``surface_map.pickle`` payload).

    The producer wraps a ``{"version": 2, "maps": dict}`` shape. Each
    map entry is a ``(tri_indices, coefs, surf_tri)`` tuple of numpy
    arrays. We rehydrate each tuple back to (int64, float64, int64)
    so downstream code that does index lookup still works.
    """
    env = cbor2.loads(blob)
    payload = _unwrap(env, KIND_SURFACE_MAP)
    if not isinstance(payload, dict):
        raise CborSchemaError("surface-map payload must be a map")
    inner_version = payload.get("version")
    maps = payload.get("maps")
    if inner_version != SURFACE_MAP_VERSION or not isinstance(maps, dict):
        raise CborSchemaError(
            f"surface-map inner format unsupported: version={inner_version!r}"
        )
    rehydrated: dict = {}
    for name, entry in maps.items():
        tri_indices, coefs, surf_tri = entry
        rehydrated[name] = (
            np.asarray(tri_indices, dtype=np.int64),
            np.asarray(coefs, dtype=np.float64),
            np.asarray(surf_tri, dtype=np.int64),
        )
    return {"version": inner_version, "maps": rehydrated}


def dumps_pickled_envelope(kind: str, pickled: bytes) -> bytes:
    """Wrap already-pickled bytes in a CBOR envelope. **Test-only producer.**

    Not used by any live writer in ``frontend/`` or ``blender_addon/``;
    grep confirms the only callers live in
    ``tests/test_cbor_producer_roundtrip.py``. Retained so the
    raw-bytes on-disk format (payload = bytes under ``KIND_APP_STATE``
    / ``KIND_FIXED_SESSION``) can be synthesized in tests that prove
    :func:`loads_pickle_blob` still reads files written by older
    ``App.save`` / ``Session._save_fixed_session`` versions. Live
    writers use :func:`dumps_envelope` directly with a structured map
    payload.
    """
    envelope = {"version": SCHEMA_VERSION, "kind": kind, "payload": pickled}
    return cbor2.dumps(envelope)


def loads_pickled_envelope(blob: bytes, expected_kind: str) -> bytes:
    """Unwrap a CBOR envelope carrying opaque pickled bytes. **Test-only reader.**

    Live consumers route through :func:`loads_pickle_blob`, which
    handles both the dict-shaped payloads emitted by current writers
    and the raw-bytes payloads emitted by :func:`dumps_pickled_envelope`.
    Retained for symmetry with the test-only producer above.
    """
    env = cbor2.loads(blob)
    payload = _unwrap(env, expected_kind)
    if not isinstance(payload, (bytes, bytearray)):
        raise CborSchemaError(
            f"{expected_kind} payload must be bytes, got {type(payload).__name__}"
        )
    return bytes(payload)


def loads_pickle_blob(blob: bytes, expected_kind: str) -> bytes:
    """Extract pickle bytes from an ``AppState`` / ``FixedSession`` envelope.

    Dispatches on payload shape so that one consumer call site handles
    both wire formats:

    * Native CBOR map payloads (current producer) carry the pickle
      bytes under the ``"pickle_blob"`` key alongside structured
      metadata fields. Returns those bytes.
    * Raw CBOR ``bytes`` payloads (older :func:`dumps_pickled_envelope`
      writer, still on disk for saves predating the structured-map
      producer) are returned as-is.

    The bytes path is the original behavior and stays correct forever:
    it is not a fallback, it's the read path for files written before
    the producer switched.
    """
    env = cbor2.loads(blob)
    payload = _unwrap(env, expected_kind)
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    if isinstance(payload, dict):
        pickled = payload.get("pickle_blob")
        if not isinstance(pickled, (bytes, bytearray)):
            raise CborSchemaError(
                f"{expected_kind} map payload missing 'pickle_blob' bytes"
            )
        return bytes(pickled)
    raise CborSchemaError(
        f"{expected_kind} payload must be bytes or map, got {type(payload).__name__}"
    )


def load_pickle_payload(path: str, expected_kind: str):
    """Read a saved pickle file (CBOR-enveloped or raw) and unpickle it.

    ``loads_pickle_blob`` handles both the current dict-shaped payload
    and the older raw-bytes envelopes left on disk by earlier builds, so
    a single sniff routes both formats. Files predating the CBOR
    envelope are still plain pickle and are read back directly.
    """
    with open(path, "rb") as f:
        blob = f.read()
    if is_cbor(blob):
        return pickle.loads(loads_pickle_blob(blob, expected_kind))
    return pickle.loads(blob)
