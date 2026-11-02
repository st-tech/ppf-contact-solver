# File: scenarios/bl_cbor_twin_agrees.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: GPL-3.0-or-later
#
# The CBOR envelope encoder exists twice: `core/encoder/cbor_encode.py`
# (GPL-3.0-or-later, what the add-on uploads with) and
# `frontend/_cbor_bridge_.py` (Apache-2.0, what the frontend and
# `crates/ppf-cts-formats/tests/scripts/gen_fixtures.py` encode with). The two
# are not byte-identical files, so this compares what they EMIT. A drift would
# make the checked-in CBOR fixtures describe bytes the add-on never sends.
#
#   A. schema_versions_agree: both spell the same SCHEMA_VERSION.
#   B. envelopes_are_byte_identical: a payload of numpy arrays and scalars
#      encodes to the same bytes under kind "Scene" and kind "Param".
#   C. unencodable_values_are_refused_by_both: an object neither knows raises
#      TypeError in both, so neither silently writes a plausible envelope.
#
# The frontend file is read from the tree the loaded add-on sits in, so the
# comparison is between two files of one checkout. Pure introspection: no
# server, no solver, no transfer.

from __future__ import annotations

from . import _runner as r


NEEDS_BLENDER = True

BACKENDS = ("real",)


_DRIVER_TEMPLATE = r"""
import importlib.util, os, sys, traceback
import numpy as np
result.setdefault("checks", {})
result.setdefault("errors", [])


def record(name, ok, details=None):
    result["checks"][name] = {"ok": bool(ok), "details": details or {}}


def payload():
    return {
        "name": "cloth",
        "vert": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        "face": np.array([[0, 1, 1]], dtype=np.uint32),
        "transform": np.eye(4, dtype=np.float64),
        "gravity": np.float32(-9.8),
        "frames": np.int64(120),
        "nested": {"pin": [0, 1], "stitch": (np.array([[0, 1]], dtype=np.uint32),)},
    }


try:
    addon = __import__(pkg + ".core.encoder.cbor_encode", fromlist=["dumps_envelope"])
    # `import_module`, not `__import__`: the latter returns the TOP package,
    # which under Blender's extension layout is `bl_ext`, not the add-on.
    root = os.path.dirname(os.path.dirname(os.path.realpath(
        importlib.import_module(pkg).__file__)))
    path = os.path.join(root, "frontend", "_cbor_bridge_.py")
    spec = importlib.util.spec_from_file_location("ppf_cts_cbor_bridge_twin", path)
    bridge = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bridge
    spec.loader.exec_module(bridge)

    record("A_schema_versions_agree",
           addon.SCHEMA_VERSION == bridge.SCHEMA_VERSION,
           {"addon": addon.SCHEMA_VERSION, "frontend": bridge.SCHEMA_VERSION})

    differing = [k for k in ("Scene", "Param")
                 if addon.dumps_envelope(k, payload())
                 != bridge.dumps_envelope(k, payload())]
    record("B_envelopes_are_byte_identical", not differing,
           {"differing_kinds": differing, "frontend_file": path})

    refused = {}
    for label, fn in (("addon", addon.dumps_envelope),
                      ("frontend", bridge.dumps_envelope)):
        try:
            fn("Param", {"bad": object()})
            refused[label] = False
        except TypeError:
            refused[label] = True
    record("C_unencodable_values_are_refused_by_both", all(refused.values()),
           refused)

except Exception as exc:
    result["errors"].append(type(exc).__name__ + ": " + str(exc))
    result["errors"].append(traceback.format_exc())
"""


def build_driver(ctx: r.ScenarioContext) -> str:
    return _DRIVER_TEMPLATE


def run(ctx: r.ScenarioContext) -> dict:
    result, err = r.wait_blender_result(ctx)
    if err is not None:
        return err
    return r.report_named_checks(result.get("checks", {}))
