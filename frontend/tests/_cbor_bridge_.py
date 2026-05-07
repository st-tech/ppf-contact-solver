# File: _cbor_bridge_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Test module for the CBOR envelope schema (frontend/_cbor_bridge_.py)."""

import numpy as np

from .._cbor_bridge_ import (
    KIND_PARAM,
    KIND_SCENE,
    CborSchemaError,
    dumps_envelope,
    loads_envelope,
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


def run_tests() -> bool:
    """Run all CBOR bridge tests. Returns True if all tests pass."""
    print("=" * 50)
    print("CBOR Bridge Tests")
    print("=" * 50)

    try:
        test_envelope_roundtrip()
        test_envelope_numpy_payload()
        test_envelope_wrong_kind_raises()
        print("\nAll CBOR bridge tests PASSED!")
        return True
    except AssertionError as e:
        print(f"\nTest FAILED: {e}")
        return False
    except Exception as e:
        print(f"\nTest ERROR: {e}")
        return False
