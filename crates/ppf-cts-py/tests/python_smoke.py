"""
Smoke tests for the `_ppf_cts_py` extension module.

These tests exercise one trivial call per binding module so the
extension's top-level surface is exercised end-to-end. Coverage is
deliberately shallow: per-kernel correctness lives in the Rust unit
tests.

Invocation (after `cargo build --release` builds the cdylib):

    pytest crates/ppf-cts-py/tests/

We obtain `_ppf_cts_py` through the `frontend` package, which loads the
cdylib built into `target/release/` by absolute path and registers it as
`_ppf_cts_py`. The repo root must be importable for that to work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from frontend import _rust
except Exception as e:  # pragma: no cover - environment-dependent
    pytest.skip(f"_ppf_cts_py cdylib unavailable: {e}", allow_module_level=True)


def test_smoke_top_level():
    assert isinstance(_rust.version(), str)
    assert isinstance(_rust.schema_version(), int)


def test_smoke_app():
    base = _rust.get_data_dirpath("/tmp/x")
    assert isinstance(base, str) and base
    # The data dir is namespaced under the input root; "/tmp/x" must
    # appear somewhere in the resolved path.
    assert "/tmp/x" in base or base.startswith("/tmp/x")


def test_smoke_asset():
    reg = _rust.AssetRegistry()
    # An empty registry should report zero entries through whatever
    # listing accessor it exposes; we don't assume a particular method
    # name beyond "you can construct one".
    assert reg is not None


def test_smoke_decoder():
    # Phase C.2 made the error class concrete; assert on it instead of
    # the bare Exception base.
    with pytest.raises(ValueError):
        _rust.validate_pickle_extension("foo.txt")
    _rust.validate_pickle_extension("foo.pickle")


def test_smoke_extra():
    assert callable(_rust.sparse_clone)


def test_smoke_kernels():
    import numpy as np
    verts = np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    pinned = np.array([False, False], dtype=np.bool_)
    pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    n = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    out = _rust.check_wall_violations_single(verts, pinned, pos, n)
    assert len(out) == 1 and out[0][0] == 0


def test_smoke_mesh():
    verts, faces = _rust.mesh_icosphere(1.0, 0)
    assert verts.shape[1] == 3 and faces.shape[1] == 3
    assert verts.shape[0] == 12 and faces.shape[0] == 20


def test_smoke_param():
    holder = _rust.app_param()
    # The default param holder ships with a non-trivial set of keys;
    # an empty key_list would mean app_param returned the wrong thing.
    keys = holder.key_list()
    assert isinstance(keys, list) and len(keys) > 0


def test_smoke_pin():
    # Constructing the holder verifies the pyclass surface is wired,
    # not just exposed as a name.
    h = _rust.PinHolder()
    assert h is not None


def test_smoke_render():
    m = _rust.ortho_matrix(-1.0, 1.0, -1.0, 1.0, 0.0, 1.0)
    assert m.shape == (4, 4)
    # Orthographic projection sends the +z axis (back) to z=-1 and the
    # -z axis (front) to z=+1 with our convention; cheap sanity that
    # the matrix isn't an identity / zero.
    import numpy as np
    assert not np.allclose(m, np.eye(4))


def test_smoke_scene():
    assert _rust.scene_axis_letter_to_index("x") == 0
    assert _rust.scene_axis_letter_to_index("Y") == 1


def test_smoke_session():
    s = _rust.convert_time(1500.0)
    assert isinstance(s, str) and s


def test_smoke_utils():
    assert isinstance(_rust.solver_busy(), bool)
    assert isinstance(_rust.process_name(), str)
    assert isinstance(_rust.get_cache_dir(), str)
