# File: __init__.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

"""Entry point for the frontend package.

Import :class:`App` from this package to start the application; the other
names listed in ``__all__`` expose the scene, session, mesh, plotting,
asset, and utility APIs.
"""

# Resolve the Rust extension module exactly once, at the top of the
# package. Submodules import it as ``from . import _rust`` so a missing
# wheel surfaces this single, actionable error instead of a cascade of
# ``ModuleNotFoundError`` from whichever submodule loaded first.
#
# Tree-local PyO3 lookup. The compiled ``_ppf_cts_py`` wheel is installed
# into ``<tree-root>/.tree-pyo3/`` by build-all.sh, so two worktrees
# (e.g. one on `dev`, one on `main`) on the same host don't clobber each
# other's PyO3 module through a shared venv site-packages. We prepend
# the tree-local dir to ``sys.path`` if it exists, falling back to the
# usual import resolution (system Python / Blender bundle / shared venv)
# when it doesn't. This is intentionally relative to *this file*, so
# the right tree's build wins regardless of caller cwd.
import os as _os
import sys as _sys
_TREE_PYO3 = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".tree-pyo3"))
if _os.path.isdir(_TREE_PYO3) and _TREE_PYO3 not in _sys.path:
    _sys.path.insert(0, _TREE_PYO3)
try:
    import _ppf_cts_py as _rust  # noqa: N816
except ImportError as e:
    raise ImportError(
        "_ppf_cts_py extension module not found. Build with "
        "`build-all.sh` from the repo root (writes the wheel into "
        f"{_TREE_PYO3}), or install the prebuilt wheel into the "
        "active Python environment. Original error: " + str(e)
    ) from e

__all__ = [
    "App",
    "get_cache_dir",
    "AssetManager",
    "AssetFetcher",
    "AssetUploader",
    "SceneManager",
    "Scene",
    "SceneInfo",
    "ObjectAdder",
    "FixedScene",
    "Object",
    "InvisibleAdder",
    "Wall",
    "Sphere",
    "Extra",
    "MeshManager",
    "CreateManager",
    "Rod",
    "TetMesh",
    "TriMesh",
    "PlotManager",
    "Plot",
    "SessionManager",
    "Session",
    "FixedSession",
    "SessionInfo",
    "SessionExport",
    "SessionOutput",
    "SessionGet",
    "CppRustDocStringParser",
    "ParamManager",
    "Utils",
    "BlenderApp",
    "sdf",
]

from . import _sdf_ as sdf
from ._app_ import App
from ._asset_ import AssetFetcher, AssetManager, AssetUploader
from ._decoder_ import BlenderApp
from ._extra_ import Extra
from ._mesh_ import CreateManager, MeshManager, Rod, TetMesh, TriMesh
from ._parse_ import CppRustDocStringParser
from ._plot_ import Plot, PlotManager
from ._scene_ import (
    FixedScene,
    InvisibleAdder,
    Object,
    ObjectAdder,
    Scene,
    SceneInfo,
    SceneManager,
    Sphere,
    Wall,
)
from ._session_ import (
    FixedSession,
    ParamManager,
    Session,
    SessionExport,
    SessionGet,
    SessionInfo,
    SessionManager,
    SessionOutput,
)
from ._utils_ import Utils, get_cache_dir
