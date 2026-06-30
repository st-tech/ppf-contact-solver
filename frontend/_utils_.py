# File: _utils_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Thin Python shim over the Rust `_ppf_cts_py` utility surface
# (process discovery, fast-check flag, cache + CI path arithmetic,
# HTML table rendering). The methods that stay in Python use IPython
# (`Utils.in_jupyter_notebook`) or shell out to `nvidia-smi` via
# subprocess (`Utils.get_gpu_count`, `Utils.get_driver_version`).

import os
import subprocess

from typing import Optional

import numpy as np

from . import _rust  # type: ignore[attr-defined]


def _as_c(arr, dtype):
    """Return a C-contiguous view of ``arr`` with the given dtype.

    The Rust kernels take ``PyReadonlyArray`` parameters and re-check
    C-contiguity, so the contiguous copy is load-bearing; the ``dtype``
    keeps the existing silent-cast behavior so callers that build
    indices with the platform-default int width keep working. ``None``
    passes through unchanged for the optional array arguments.
    """
    return None if arr is None else np.ascontiguousarray(arr, dtype=dtype)


def get_cache_dir() -> str:
    """Get the ppf-cts cache directory."""
    return _rust.get_cache_dir()


def get_export_base_path() -> str:
    """Resolve the export base path, honoring fast-check mode."""
    return _rust.get_export_base_path()


def dict_to_html_table(data: dict, classes: str = "table") -> str:
    """Render a column-oriented mapping to an HTML table."""
    columns = [(str(k), [str(x) for x in v]) for k, v in data.items()]
    return _rust.dict_to_html_table(columns, classes)


class Utils:
    """Utility class for frontend.

    Example:
        Check whether the solver is running and stop it if so before
        kicking off a new simulation::

            from frontend import Utils

            if Utils.busy():
                Utils.terminate()
            print("gpus:", Utils.get_gpu_count())
    """

    @staticmethod
    def in_jupyter_notebook() -> bool:
        """Determine if the code is running in a Jupyter notebook."""
        dirpath = os.path.dirname(os.path.abspath(__file__))
        if _rust.has_cli_or_ci_marker(dirpath):
            return False
        try:
            from IPython import get_ipython  # type: ignore

            shell = get_ipython().__class__.__name__
            return shell == "ZMQInteractiveShell"
        except (NameError, ImportError):
            return False

    @staticmethod
    def ci_name() -> Optional[str]:
        """Determine if the code is running in a CI environment."""
        dirpath = os.path.dirname(os.path.abspath(__file__))
        return _rust.ci_name(dirpath)

    @staticmethod
    def is_fast_check() -> bool:
        """Determine if fast check mode is enabled."""
        return _rust.is_fast_check()

    @staticmethod
    def platform_which() -> str:
        """Return the platform discriminator ('windows' or 'unix') used by the Rust launcher helpers."""
        import platform

        return "windows" if platform.system() == "Windows" else "unix"

    @staticmethod
    def set_fast_check(enabled: bool = True):
        """Set fast check mode."""
        _rust.set_fast_check(enabled)

    @staticmethod
    def get_ci_root() -> str:
        """Get the path to the CI directory."""
        return _rust.get_ci_root()

    @staticmethod
    def get_ci_dir() -> str:
        """Get the path to the CI local directory."""
        ci_name = Utils.ci_name()
        assert ci_name is not None
        return _rust.get_ci_dir(ci_name)

    @staticmethod
    def get_gpu_count() -> int:
        """Number of NVIDIA GPUs visible to nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            lines = [
                l for l in result.stdout.strip().split("\n") if l.startswith("GPU ")
            ]
            return len(lines)
        except subprocess.CalledProcessError as e:
            print("Error occurred while running nvidia-smi:", e)
            return 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("nvidia-smi not found. Is NVIDIA driver installed?")
            return 0

    @staticmethod
    def get_driver_version() -> Optional[int]:
        """Major NVIDIA driver version, or None if nvidia-smi is unavailable."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return int(result.stdout.strip().split(".")[0])
        except subprocess.CalledProcessError as e:
            print("Error occurred while running nvidia-smi:", e)
            return None
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("nvidia-smi not found. Is NVIDIA driver installed?")
            return None

    @staticmethod
    def check_gpu():
        """Check that an NVIDIA GPU with sufficient compute capability is present."""
        _rust.check_gpu()

    terminate = staticmethod(_rust.terminate_solver)
    busy = staticmethod(_rust.solver_busy)
