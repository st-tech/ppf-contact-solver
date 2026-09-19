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

import http.client
import os
import random
import socket
import ssl
import subprocess
import time
import urllib.error
import urllib.request

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


def _tree_root() -> str:
    """The root of the tree this frontend was imported from.

    Resolved the same way ``App.get_data_dirpath`` resolves it, from this
    package's own file, because that is the only thing that reliably names the
    tree: the working directory is wherever the user happened to start, and a
    packaged tree keeps its cache and its data inside ITSELF, so handing the
    wrong root would put a distribution's state in the home directory it was
    packaged to stay out of.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_cache_dir() -> str:
    """Get the ppf-cts cache directory."""
    return _rust.get_cache_dir(_tree_root())


def fetch_asset(
    url: str,
    dest_path: str,
    attempts: int = 4,
    base_delay: float = 2.0,
    timeout: float = 10,
    ssl_context: Optional[ssl.SSLContext] = None,
) -> None:
    """Download ``url`` to ``dest_path``, retrying transient failures.

    A hosted asset (a GitHub release, for instance) sits behind a gateway
    that occasionally answers a request with a 502, 503 or 504 under load,
    and a dropped connection or a timeout is just as transient, so both
    are worth a bounded number of retries with backoff. An HTTP 4xx
    response, most commonly a 404, means the asset genuinely is not at
    this URL, so it is raised immediately instead of retried: retrying it
    would only delay a correct failure by the whole backoff budget. The
    file is written to a temporary path next to ``dest_path`` and moved
    into place only once the download completes, so an interrupted
    attempt can never leave a truncated file where a later run would
    treat it as a cached asset.

    Raises:
        RuntimeError: every attempt failed. The message names the URL,
            the attempt count, and the last error.
    """
    tmp_path = dest_path + ".part"
    last_error = None
    for attempt in range(attempts):
        try:
            with (
                urllib.request.urlopen(
                    url, timeout=timeout, context=ssl_context
                ) as response,
                open(tmp_path, "wb") as out_file,
            ):
                out_file.write(response.read())
            os.replace(tmp_path, dest_path)
            return
        except urllib.error.HTTPError as e:
            if e.code < 500:
                raise
            last_error = e
        except (
            urllib.error.URLError,
            # A CONNECTION DROPPED MID-BODY DOES NOT ARRIVE AS A URLError.
            # `urlopen` has already returned by then, so the failure surfaces
            # from `response.read()` as an http.client exception
            # (IncompleteRead, RemoteDisconnected) or an ssl.SSLError. Those are
            # the shape a flaky CDN actually produces, so leaving them out made
            # this retry cover the case that fails least often.
            http.client.HTTPException,
            ssl.SSLError,
            socket.timeout,
            TimeoutError,
            ConnectionError,
            OSError,
        ) as e:
            last_error = e
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if attempt < attempts - 1:
            delay = base_delay * (2**attempt) + random.uniform(0, 0.5)
            print(
                f"fetch of {url} failed ({last_error}), "
                f"retrying in {delay:.1f}s [attempt {attempt + 2}/{attempts}]"
            )
            time.sleep(delay)
    raise RuntimeError(f"failed to fetch {url} after {attempts} attempts: {last_error}")


def get_export_base_path() -> str:
    """Resolve the export base path, honoring fast-check mode."""
    return _rust.get_export_base_path(_tree_root())


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
        return _rust.get_ci_root(_tree_root())

    @staticmethod
    def get_ci_dir() -> str:
        """Get the path to the CI local directory."""
        ci_name = Utils.ci_name()
        assert ci_name is not None
        return _rust.get_ci_dir(_tree_root(), ci_name)

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
        """Check that the backend a run uses can run on this machine.

        Asks that backend's own solver (``ppf-contact-solver --probe``), whose
        answer carries the backend's precise refusal, and raises with it.
        """
        from . import _require_usable_backend

        _require_usable_backend()

    terminate = staticmethod(_rust.terminate_solver)
    busy = staticmethod(_rust.solver_busy)
