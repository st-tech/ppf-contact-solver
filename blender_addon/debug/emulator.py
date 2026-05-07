# File: server/emulator.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Debug-mode patching for the solver server.
#
# The debug rig's whole point is to exercise the **real** production
# pipeline -- real encoder roundtrip in the addon, real ``frontend``
# package on the server, real Rust binary -- on hosts that lack a CUDA
# GPU. We achieve that by:
#
#   1. Patching ``frontend.Utils.check_gpu`` so it doesn't raise when
#      ``nvidia-smi`` is missing.
#   2. Honoring ``PPF_CTS_DATA_ROOT`` inside ``frontend.BlenderApp`` so
#      per-worker isolation in the orchestrator's temp tree works (the
#      real frontend hardcodes ``~/.local/share/ppf-cts/...`` otherwise).
#
# The Rust binary at ``target/release/ppf-contact-solver`` must be
# built with ``--features emulated --no-default-features``: that build
# stubs the CUDA FFI and applies pin kinematics directly to the vertex
# buffer per frame (see src/backend.rs::apply_kinematic_constraint).

from __future__ import annotations

import json
import logging
import os


def install() -> None:
    """Patch the production ``frontend`` module for CUDA-free runs.

    Idempotent: subsequent calls are no-ops.
    """
    import frontend  # type: ignore

    if getattr(frontend.Utils, "_PPF_DEBUG_PATCHED", False):
        return

    # ---- 1) Skip GPU + driver checks ----
    # ``check_gpu`` raises RuntimeError when nvidia-smi is missing.
    # ``get_driver_version`` returns None when nvidia-smi is missing,
    # which then trips a ValueError("Driver version could not be
    # detected.") inside session.start. Replace both.
    def _check_gpu_noop():
        return None

    def _driver_version_stub():
        # Return a value above session.start's min (520).
        return 999

    # ``Utils.busy()`` does a global ``psutil.process_iter`` for any
    # process whose name contains ``"ppf-contact"``. Under the debug
    # orchestrator's parallel mode, every worker has its own solver
    # subprocess named ``ppf-contact-solver``; a foreign worker's
    # process trips this check and ``session.start`` bails out with
    # "Solver is already running" before our Popen runs. Restrict the
    # check to descendants of THIS server process so each worker only
    # sees its own solver.
    import os as _os
    import psutil as _psutil  # type: ignore

    def _busy_local() -> bool:
        try:
            me = _psutil.Process(_os.getpid())
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

    frontend.Utils.check_gpu = staticmethod(_check_gpu_noop)  # type: ignore[assignment]
    frontend.Utils.get_driver_version = staticmethod(_driver_version_stub)  # type: ignore[assignment]
    frontend.Utils.busy = staticmethod(_busy_local)  # type: ignore[assignment]
    frontend.Utils._PPF_DEBUG_PATCHED = True  # type: ignore[attr-defined]

    # ---- 2) Honor PPF_CTS_DATA_ROOT inside BlenderApp ----
    if not getattr(frontend.BlenderApp, "_PPF_DEBUG_PATCHED", False):
        original_init = frontend.BlenderApp.__init__

        def _patched_init(self, name: str, verbose: bool = False,
                          progress_callback=None):
            original_init(self, name, verbose, progress_callback=progress_callback)
            shadow = os.environ.get("PPF_CTS_DATA_ROOT")
            if shadow:
                self._data_dirpath = os.path.join(shadow, "git-debug")
                self._root = os.path.join(self._data_dirpath, name)
                cache_root = os.path.join(self._root, ".cash")
                os.makedirs(cache_root, exist_ok=True)
                if hasattr(self, "_mesh_manager"):
                    self._mesh_manager._cache_root = cache_root  # type: ignore[attr-defined]

        frontend.BlenderApp.__init__ = _patched_init  # type: ignore[assignment]
        frontend.BlenderApp._PPF_DEBUG_PATCHED = True  # type: ignore[attr-defined]

    # ---- 3) Trace subprocess.Popen calls so we can see the Rust
    # binary's PID, args, env, and exit. Helps diagnose
    # "Popen returned but no output" issues.
    import subprocess as _sp
    if not getattr(_sp.Popen, "_PPF_DEBUG_PATCHED", False):
        _OrigPopen = _sp.Popen

        class _TracedPopen(_OrigPopen):
            def __init__(self, *args, **kwargs):
                logging.info(
                    f"Popen: args={args[0] if args else kwargs.get('args')!r} "
                    f"cwd={kwargs.get('cwd')!r} "
                    f"shell={kwargs.get('shell')} "
                    f"start_new_session={kwargs.get('start_new_session')}"
                )
                super().__init__(*args, **kwargs)
                logging.info(f"Popen: started pid={self.pid}")

        _TracedPopen._PPF_DEBUG_PATCHED = True  # type: ignore[attr-defined]
        _sp.Popen = _TracedPopen  # type: ignore[assignment]

    # ---- 4) PPF_EMULATED_VIOLATIONS knob ----
    # When set, the next make_response merges the parsed JSON list into
    # the response's ``violations`` field. Single-shot: the env var is
    # cleared after one successful injection so follow-up responses are
    # clean. Malformed JSON is logged and ignored.
    from server import engine as _engine_mod
    if not getattr(_engine_mod.ServerEngine, "_PPF_DEBUG_VIOLATIONS_PATCHED", False):
        _orig_make_response = _engine_mod.ServerEngine.make_response

        def _patched_make_response(self):  # type: ignore[no-untyped-def]
            response = _orig_make_response(self)
            raw = os.environ.get("PPF_EMULATED_VIOLATIONS")
            if raw:
                try:
                    parsed = json.loads(raw)
                except (TypeError, ValueError) as exc:
                    logging.error(
                        "PPF_EMULATED_VIOLATIONS: malformed JSON, "
                        "ignoring (%s)", exc,
                    )
                    os.environ.pop("PPF_EMULATED_VIOLATIONS", None)
                    return response
                if not isinstance(parsed, list):
                    logging.error(
                        "PPF_EMULATED_VIOLATIONS: expected a list, "
                        "got %s, ignoring", type(parsed).__name__,
                    )
                    os.environ.pop("PPF_EMULATED_VIOLATIONS", None)
                    return response
                response["violations"] = parsed
                os.environ.pop("PPF_EMULATED_VIOLATIONS", None)
                logging.info(
                    "PPF_EMULATED_VIOLATIONS: injected %d violation(s)",
                    len(parsed),
                )
            return response

        _engine_mod.ServerEngine.make_response = _patched_make_response  # type: ignore[assignment]
        _engine_mod.ServerEngine._PPF_DEBUG_VIOLATIONS_PATCHED = True  # type: ignore[attr-defined]

    logging.info(
        "debug emulator: real frontend in use, check_gpu patched, "
        "PPF_CTS_DATA_ROOT honored. Rust binary expected at "
        "target/release/ppf-contact-solver built with --features emulated."
    )
