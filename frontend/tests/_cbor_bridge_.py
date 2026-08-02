# File: _cbor_bridge_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Test module for the CBOR envelope schema (frontend/_cbor_bridge_.py)."""

import pickle

import numpy as np

from .._cbor_bridge_ import (
    KIND_APP_STATE,
    KIND_PARAM,
    KIND_SCENE,
    PICKLE_CHUNK_BYTES,
    CborSchemaError,
    chunk_pickle_blob,
    dumps_envelope,
    dumps_pickled_envelope,
    loads_envelope,
    loads_pickle_blob,
)


def test_envelope_roundtrip():
    """A payload survives dumps_envelope -> loads_envelope unchanged."""
    print("  Testing envelope round-trip...")

    payload = {"name": "demo", "values": [1, 2, 3], "flag": True}
    blob = dumps_envelope(KIND_PARAM, payload)
    assert isinstance(blob, (bytes, bytearray)), (
        f"dumps_envelope must return bytes, got {type(blob)}"
    )
    assert len(blob) > 0, "dumps_envelope returned empty bytes"

    recovered = loads_envelope(blob, KIND_PARAM)
    assert recovered == payload, f"Payload changed: {recovered!r} vs {payload!r}"

    print("    Envelope round-trip: PASS")


def test_envelope_numpy_payload():
    """Numpy arrays in the payload encode cleanly via the default hook."""
    print("  Testing numpy payload encoding...")

    payload = {"verts": np.array([[1.0, 2.0, 3.0]], dtype=np.float32)}
    blob = dumps_envelope(KIND_SCENE, payload)
    recovered = loads_envelope(blob, KIND_SCENE)
    assert recovered == {"verts": [[1.0, 2.0, 3.0]]}, (
        f"Numpy payload mismatch: {recovered!r}"
    )

    print("    Numpy payload encoding: PASS")


def test_envelope_wrong_kind_raises():
    """loads_envelope with a kind mismatch raises CborSchemaError with a
    message identifying the mismatch."""
    print("  Testing wrong-kind detection...")

    blob = dumps_envelope(KIND_PARAM, {"x": 1})
    try:
        loads_envelope(blob, KIND_SCENE)
    except CborSchemaError as e:
        msg = str(e)
        assert "kind mismatch" in msg, f"Unexpected error message: {msg}"
        assert KIND_PARAM in msg or repr(KIND_PARAM) in msg, (
            f"Error message must mention payload kind: {msg}"
        )
        assert KIND_SCENE in msg or repr(KIND_SCENE) in msg, (
            f"Error message must mention expected kind: {msg}"
        )
        print("    Wrong-kind error: PASS")
        return
    raise AssertionError("loads_envelope did not raise on kind mismatch")


def test_pickle_blob_chunk_roundtrip():
    """A chunked ``pickle_blob`` payload rehydrates to the original object."""
    print("  Testing chunked pickle_blob round-trip...")

    original = {"frames": list(range(1000)), "name": "session"}
    pickled = pickle.dumps(original)
    chunks = chunk_pickle_blob(pickled)

    assert all(isinstance(c, bytes) for c in chunks), (
        "chunks must be bytes; cbor2 encodes a memoryview as an array of ints"
    )
    assert b"".join(chunks) == pickled, "chunks must reassemble to the original"
    assert max(len(c) for c in chunks) <= PICKLE_CHUNK_BYTES

    blob = dumps_envelope(KIND_APP_STATE, {"name": "x", "pickle_blob": chunks})
    assert pickle.loads(loads_pickle_blob(blob, KIND_APP_STATE)) == original
    print(f"    {len(pickled)} bytes in {len(chunks)} chunk(s): PASS")


def test_pickle_blob_splits_into_bounded_chunks():
    """A blob spanning several chunks splits into bounded pieces and rejoins.

    A session graph pickles to gigabytes (1.7 GB for a 3.2M-vertex scene) and
    is read back through this path, so the chunk size is what keeps that read
    linear rather than quadratic; see :data:`PICKLE_CHUNK_BYTES` for the
    decode measurements behind the value. Both producers (``App.save`` and
    ``Session._save_fixed_session``) reach the chunk list only through
    :func:`chunk_pickle_blob`, so bounding the split here covers both.

    The guard is structural rather than a timed decode because wall-clock cost
    per byte is not constant across payload sizes: a buffer past the
    allocator's mmap threshold is re-faulted on every pass, and one past L3
    loses cache residency. Measured on cbor2 6.0.1, the chunked path costs
    0.96 ms/MB at 8 MB and 1.33 ms/MB at 32 MB on a quiet Linux host, and
    0.35 -> 1.19 ms/MB on a shared CI runner. Quadrupling the payload
    therefore takes 5.2x to 13.4x rather than the 4x of the underlying linear
    decode, which is the same order as the growth a timing test would be
    trying to detect, so no threshold separates the two. Comparing the two
    payload SHAPES at one size does cancel those constants (chunking measures
    6.6x to 120.7x faster than a single byte string from 8 MB up), but that
    separation exists only while the installed cbor2 has the quadratic, and
    6.1.3 fixed it upstream.
    """
    print("  Testing pickle_blob chunk bounds...")

    assert (64 << 10) <= PICKLE_CHUNK_BYTES <= (4 << 20), (
        f"PICKLE_CHUNK_BYTES is {PICKLE_CHUNK_BYTES} bytes, outside the "
        "64 KiB - 4 MiB region measured flat; above it the quadratic decode "
        "returns and a gigabyte-scale session reads as a hang"
    )

    # Two full chunks plus a short remainder, so the split is exercised at
    # both boundaries instead of dividing evenly.
    tail = 7
    pickled = b"\xa5" * (2 * PICKLE_CHUNK_BYTES + tail)
    chunks = chunk_pickle_blob(pickled)

    assert [len(c) for c in chunks] == [
        PICKLE_CHUNK_BYTES,
        PICKLE_CHUNK_BYTES,
        tail,
    ], f"unexpected chunk sizes: {[len(c) for c in chunks]}"
    assert all(isinstance(c, bytes) for c in chunks), (
        "chunks must be bytes; cbor2 encodes a memoryview as an array of ints"
    )

    # Prove the multi-chunk list rejoins through the real reader, not just
    # through b"".join.
    blob = dumps_envelope(KIND_APP_STATE, {"pickle_blob": chunks})
    assert loads_pickle_blob(blob, KIND_APP_STATE) == pickled, (
        "multi-chunk payload did not rejoin to the original bytes"
    )

    print(
        f"    {len(pickled)} bytes -> {len(chunks)} chunks of "
        f"{PICKLE_CHUNK_BYTES} bytes: PASS"
    )


def test_pickle_blob_accepts_every_saved_shape():
    """Files saved before chunking still read back."""
    print("  Testing legacy pickle_blob shapes...")

    original = {"kind": "legacy"}
    pickled = pickle.dumps(original)

    # A single byte string under the key: correct, just slow at scale.
    single = dumps_envelope(KIND_APP_STATE, {"pickle_blob": pickled})
    assert pickle.loads(loads_pickle_blob(single, KIND_APP_STATE)) == original

    # The oldest shape: the payload IS the pickle bytes, with no map.
    raw = dumps_pickled_envelope(KIND_APP_STATE, pickled)
    assert pickle.loads(loads_pickle_blob(raw, KIND_APP_STATE)) == original
    print("    Single-blob and raw-bytes payloads: PASS")


def test_pickle_blob_rejects_a_bad_chunk():
    """A chunk that is not bytes names itself instead of failing in pickle."""
    print("  Testing malformed chunk rejection...")

    blob = dumps_envelope(KIND_APP_STATE, {"pickle_blob": [b"ok", 42]})
    try:
        loads_pickle_blob(blob, KIND_APP_STATE)
    except CborSchemaError as e:
        assert "chunk 1" in str(e), f"error must name the bad chunk: {e}"
        print(f"    Rejected: {e}: PASS")
        return
    raise AssertionError("loads_pickle_blob accepted a non-bytes chunk")


def run_tests() -> bool:
    """Run all CBOR bridge tests. Returns True if all tests pass."""
    print("=" * 50)
    print("CBOR Bridge Tests")
    print("=" * 50)

    try:
        test_envelope_roundtrip()
        test_envelope_numpy_payload()
        test_envelope_wrong_kind_raises()
        test_pickle_blob_chunk_roundtrip()
        test_pickle_blob_splits_into_bounded_chunks()
        test_pickle_blob_accepts_every_saved_shape()
        test_pickle_blob_rejects_a_bad_chunk()
        print("\nAll CBOR bridge tests PASSED!")
        return True
    except AssertionError as e:
        print(f"\nTest FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nTest ERROR: {e}")
        return False
