import os
import subprocess
from typing import Optional


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
    def in_CI() -> bool:
        """Determine if the code is running in a CI environment."""
        dirpath = os.path.dirname(os.path.abspath(__file__))
        return os.path.exists(os.path.join(dirpath, ".CI"))

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
