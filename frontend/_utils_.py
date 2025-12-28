# File: _utils_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import contextlib
import os
import platform
import subprocess

from typing import Optional

import psutil  # pyright: ignore[reportMissingModuleSource]

PROCESS_NAME = "ppf-contact"


def get_cache_dir() -> str:
    """Get the ppf-cts cache directory.

    Returns a platform-appropriate cache directory for ppf-cts.
    - Linux/Mac: ~/.cache/ppf-cts
    - Windows: <project>/.cache/ppf-cts (project-relative)

    Returns:
        str: Path to the ppf-cts cache directory.
    """
    if platform.system() == "Windows":
        # Use project-relative cache on Windows
        frontend_dir = os.path.dirname(os.path.realpath(__file__))
        base_dir = os.path.dirname(frontend_dir)
        cache_dir = os.path.join(base_dir, "cache", "ppf-cts")
    else:
        # Use ~/.cache/ppf-cts on Linux/Mac
        cache_dir = os.path.expanduser(os.path.join("~", ".cache", "ppf-cts"))

    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_export_base_path() -> str:
    """Get the base path for export directories.

    When fast_check mode is enabled, exports go to the cache directory
    to avoid leaving artifacts in the examples directory.
    Otherwise, exports go to the 'export' directory relative to project root.

    Returns:
        str: The base path for export directories.
    """
    if Utils.is_fast_check():
        # Use cache directory for fast_check mode
        return os.path.join(get_cache_dir(), "export")
    else:
        # Use relative 'export' directory for normal operation
        return "export"


def dict_to_html_table(data: dict, classes: str = "table", index: bool = False) -> str:
    """Convert a dictionary to an HTML table.

    Replacement for pandas DataFrame.to_html() to avoid pandas dependency.

    Args:
        data: Dictionary where keys are column names and values are lists of values.
        classes: CSS classes to add to the table element.
        index: Whether to include row index (ignored, kept for API compatibility).

    Returns:
        HTML string representation of the table.
    """
    if not data:
        return "<table></table>"

    # Get column names and number of rows
    columns = list(data.keys())
    num_rows = len(next(iter(data.values()))) if data else 0

    # Build HTML
    html_parts = [f'<table class="{classes}">']

    # Header
    html_parts.append("<thead><tr>")
    for col in columns:
        html_parts.append(f"<th>{col}</th>")
    html_parts.append("</tr></thead>")

    # Body
    html_parts.append("<tbody>")
    for i in range(num_rows):
        html_parts.append("<tr>")
        for col in columns:
            val = data[col][i] if i < len(data[col]) else ""
            html_parts.append(f"<td>{val}</td>")
        html_parts.append("</tr>")
    html_parts.append("</tbody>")

    html_parts.append("</table>")
    return "".join(html_parts)


class Utils:
    """Utility class for frontend."""

    @staticmethod
    def in_jupyter_notebook():
        """Determine if the code is running in a Jupyter notebook."""
        dirpath = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(os.path.join(dirpath, ".CLI")) or os.path.exists(
            os.path.join(dirpath, ".CI")
        ):
            return False
        try:
            from IPython import get_ipython  # type: ignore

            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                return True
            elif shell == "TerminalInteractiveShell":
                return False
            else:
                return False
        except (NameError, ImportError):
            return False

    @staticmethod
    def ci_name() -> Optional[str]:
        """Determine if the code is running in a CI environment.

        Returns:
            name (str): The name of the CI environment, or an empty string if not in a CI environment.
        """
        dirpath = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(dirpath, ".CI")
        if os.path.exists(path):
            with open(path) as f:
                lines = f.readlines()
                last_line = ""
                if len(lines) > 0:
                    last_line = lines[-1].strip()
                if last_line == "":
                    raise ValueError(
                        "The .CI file is empty. Please add the name of the CI environment."
                    )
                else:
                    return last_line
        else:
            return None

    # Module-level flag for fast check mode (set by App.set_fast_check())
    _fast_check_enabled = False

    @staticmethod
    def is_fast_check() -> bool:
        """Determine if fast check mode is enabled.

        Fast check mode forces simulations to run for only 1 frame,
        enabling quick validation of all examples.

        Returns:
            bool: True if fast check mode is enabled.
        """
        return Utils._fast_check_enabled

    @staticmethod
    def set_fast_check(enabled: bool = True):
        """Set fast check mode.

        Args:
            enabled: Whether to enable fast check mode.
        """
        Utils._fast_check_enabled = enabled

    @staticmethod
    def get_ci_root() -> str:
        """Get the path to the CI directory."""
        return os.path.join(get_cache_dir(), "ci")

    @staticmethod
    def get_ci_dir() -> str:
        """Get the path to the CI local directory."""
        ci_name = Utils.ci_name()
        assert ci_name is not None
        return os.path.join(Utils.get_ci_root(), ci_name)

    @staticmethod
    def get_gpu_count():
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"], capture_output=True, text=True, check=True
            )
            gpu_count = len(result.stdout.strip().split("\n"))
            return gpu_count
        except subprocess.CalledProcessError as e:
            print("Error occurred while running nvidia-smi:", e)
            return 0
        except FileNotFoundError:
            print("nvidia-smi not found. Is NVIDIA driver installed?")
            return 0

    @staticmethod
    def get_driver_version() -> Optional[int]:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            )
            driver_version = result.stdout.strip()
            return int(driver_version.split(".")[0])
        except subprocess.CalledProcessError as e:
            print("Error occurred while running nvidia-smi:", e)
            return None
        except FileNotFoundError:
            print("nvidia-smi not found. Is NVIDIA driver installed?")
            return None

    @staticmethod
    def terminate():
        """Terminate the solver."""
        for proc in psutil.process_iter(["pid", "name", "status"]):
            if (
                PROCESS_NAME in proc.info["name"]
                and proc.info["status"] != psutil.STATUS_ZOMBIE
            ):
                with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                    proc.terminate()

    @staticmethod
    def busy() -> bool:
        """Check if the solver is running.

        Returns:
            bool: True if the solver is running, False otherwise.
        """
        for proc in psutil.process_iter(["pid", "name", "status"]):
            if (
                PROCESS_NAME in proc.info["name"]
                and proc.info["status"] != psutil.STATUS_ZOMBIE
            ):
                return True
        return False
