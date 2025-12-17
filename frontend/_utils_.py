# File: _utils_.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import os
import subprocess
import tempfile

import psutil  # pyright: ignore[reportMissingModuleSource]

PROCESS_NAME = "ppf-contact"


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
    def ci_name() -> str | None:
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

    @staticmethod
    def get_ci_root() -> str:
        """Get the path to the CI directory."""
        return os.path.join(tempfile.gettempdir(), "ci")

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
    def get_driver_version() -> int | None:
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
                try:
                    proc.terminate()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass

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
