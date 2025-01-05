import os


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
