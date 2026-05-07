# File: _extra_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Wraps `core::extra` from the maturin wheel `_ppf_cts_py`, which is a
# hard requirement; there is no Python fallback.

from . import _rust  # type: ignore[attr-defined]


class Extra:
    """Collection of auxiliary helpers that do not belong to any manager.

    ``load_CIPC_stitch_mesh(path)`` loads a CIPC-format stitch mesh and
    returns ``(vertices, faces, (stitch_index, stitch_weight))``.
    ``sparse_clone(url, dest, paths, delete_exist=False)`` fetches a
    git repository using sparse-checkout. See ``_rust.load_cipc_stitch_mesh``
    and ``_rust.sparse_clone`` for full signatures.

    Example:
        Access the helpers via :attr:`App.extra` to sparse-clone an
        external dataset and load one of its stitch meshes::

            import os
            from frontend import App, get_cache_dir

            app = App.create("fitting")
            dest = os.path.join(get_cache_dir(), "Codim-IPC")
            app.extra.sparse_clone(
                "https://github.com/ipc-sim/Codim-IPC",
                dest,
                ["Projects/FEMShell/input/dress_knife"],
            )
            stage = os.path.join(dest, "Projects/FEMShell/input/dress_knife/stage.obj")
            V, F, S = app.extra.load_CIPC_stitch_mesh(stage)
    """

    load_CIPC_stitch_mesh = staticmethod(_rust.load_cipc_stitch_mesh)
    sparse_clone = staticmethod(_rust.sparse_clone)
