# File: frontend/_debug_runtime_.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Test-rig-only runtime patches for the production ``frontend`` module.
#
# When ``frontend.build_worker`` is launched by the orchestrator-driven
# rig (or any caller that sets ``PPF_CTS_DATA_ROOT``), production
# behaviors that assume a real GPU + the canonical
# ``~/.local/share/ppf-cts/...`` data root need to be relaxed so the
# emulated solver and per-worker shadow trees work. This module owns
# those overrides so ``build_worker`` doesn't reach into the
# ``frontend`` package's private attrs at runtime.
#
# Production runs leave ``PPF_CTS_DATA_ROOT`` unset, so
# ``install_debug_patches`` short-circuits and the ``frontend`` module
# is left untouched.

import os


def install_debug_patches() -> None:
    """When the test rig launches the build worker,
    ``PPF_CTS_DATA_ROOT`` shadows the per-worker project root and we
    need to apply two patches to the production ``frontend`` module:
    skip the GPU check and redirect ``BlenderApp._root`` to the shadow
    tree. Production runs leave the env var unset and this function is
    a no-op.
    """
    shadow = os.environ.get("PPF_CTS_DATA_ROOT")
    if not shadow:
        return

    import frontend  # type: ignore

    # ``Utils.check_gpu`` raises when nvidia-smi is missing; the
    # emulated solver doesn't need it. Same for ``get_driver_version``,
    # whose return value is checked against a min-version floor.
    frontend.Utils.check_gpu = staticmethod(lambda: None)
    frontend.Utils.get_driver_version = staticmethod(lambda: 999)

    # Per-process ``Utils.busy`` so a foreign worker's solver doesn't
    # trip "Solver is already running" in parallel mode.
    try:
        import psutil as _psutil  # type: ignore
    except ImportError:
        _psutil = None
    if _psutil is not None:
        def _busy_local() -> bool:
            try:
                me = _psutil.Process(os.getpid())
                children = me.children(recursive=True)
            except (_psutil.NoSuchProcess, _psutil.AccessDenied, OSError):
                return False
            for child in children:
                try:
                    name = child.name()
                    status = child.status()
                except (_psutil.NoSuchProcess, _psutil.AccessDenied, OSError):
                    continue
                if "ppf-contact" in name and status != _psutil.STATUS_ZOMBIE:
                    return True
            return False
        frontend.Utils.busy = staticmethod(_busy_local)

    # Shadow the data root so per-worker isolation under the
    # orchestrator's temp tree works (production hardcodes
    # ~/.local/share/ppf-cts/git-<branch>/...). The Rust
    # ppf-cts-server stores uploads at <shadow>/<name>, so we mirror
    # that flat layout here.
    original_init = frontend.BlenderApp.__init__

    def _patched_init(self, name, *args, **kwargs):  # type: ignore[no-untyped-def]
        original_init(self, name, *args, **kwargs)
        self._data_dirpath = shadow
        self._root = os.path.join(shadow, name)
        cache_root = os.path.join(self._root, ".cash")
        os.makedirs(cache_root, exist_ok=True)
        if hasattr(self, "_mesh_manager"):
            self._mesh_manager.set_cache_dir(cache_root)

    frontend.BlenderApp.__init__ = _patched_init  # type: ignore[assignment]
