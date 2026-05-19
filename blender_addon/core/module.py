# File: module.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import importlib
import os
import subprocess
import sys
import threading

from ..models.console import console


def get_install_target():
    """Return the directory where ``install_module`` drops new packages.

    Uses Blender's user ``scripts/addons/modules`` dir: it is already
    on ``sys.path``, and being outside any extension directory it
    doesn't trigger Blender 5's "Policy violation with top level
    module" warning that fires for every module loaded from inside an
    extension's tree.
    """
    import bpy  # pyright: ignore
    path = bpy.utils.user_resource("SCRIPTS", path="addons/modules", create=True)
    return os.path.normpath(path)


def import_module(name):
    """Import a module by dotted name from Blender's ``sys.path``.

    The install target (``scripts/addons/modules``) is already on
    Blender's ``sys.path``, so a plain ``importlib.import_module`` is
    all the resolution needed.
    """
    return importlib.import_module(name)


def get_cbor2():
    """Return the cbor2 module bundled via ``blender_manifest.toml``."""
    import cbor2
    return cbor2


def module_exists(packages):
    """
    Check whether each package is installed at the install target.

    Args:
        packages (list): List of package names to check.

    Returns:
        bool: True if all packages exist, False otherwise.
    """
    target = get_install_target()
    return all(
        os.path.exists(os.path.join(target, package, "__init__.py"))
        for package in packages
    )


is_installing = False
install_result = None  # None: not run, True: success, False: failed
install_error_message = ""
install_mutex = threading.Lock()


def install_module(packages):
    """
    Install a list of Python packages.

    Args:
        packages (list): List of package names to install.
    """

    def install_task():
        global is_installing, install_result, install_error_message

        with install_mutex:
            is_installing = True
            install_result = None
            install_error_message = ""

        try:
            target_path = get_install_target()
            console.write(f"Installing packages to {target_path}: {', '.join(packages)}")

            subprocess.check_call(
                [sys.executable, "-m", "ensurepip", "--upgrade"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            process = subprocess.Popen(
                [sys.executable, "-m", "pip", "install", "--target", target_path]
                + packages,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate()

            if stdout:
                console.write(stdout.strip())
            if stderr:
                console.write(stderr.strip())

            if process.returncode != 0:
                error_msg = f"Failed to install {', '.join(packages)}"
                if stderr:
                    error_msg += f": {stderr.strip()}"
                console.write(error_msg, timestamp=True)

                with install_mutex:
                    install_result = False
                    install_error_message = error_msg
            else:
                # Python's PathFinder caches a "no such package" answer
                # the first time a name is looked up under each
                # sys.path entry. Blender starts the addon with
                # scripts/addons/modules empty (or missing the names
                # we're about to install), so without invalidating
                # those caches a subsequent import_module(name) keeps
                # returning ModuleNotFoundError until the user
                # restarts Blender. Drop any partial or negatively-
                # cached sys.modules entries for the just-installed
                # names too, so a prior failed `import paramiko`
                # doesn't shadow the fresh install.
                importlib.invalidate_caches()
                for pkg in packages:
                    sys.modules.pop(pkg, None)
                success_msg = f"Successfully installed {', '.join(packages)}"
                console.write(success_msg, timestamp=True)

                with install_mutex:
                    install_result = True
                    install_error_message = ""

        except Exception as e:
            error_msg = f"Error during installation: {e}"
            console.write(error_msg, timestamp=True)

            with install_mutex:
                install_result = False
                install_error_message = error_msg
        finally:
            with install_mutex:
                is_installing = False

    threading.Thread(target=install_task, daemon=True).start()


def get_installing_status():
    """Safely get the current installation status."""
    with install_mutex:
        return is_installing


def get_install_result():
    """Safely get the last installation result.

    Returns:
        None: Installation never run or currently in progress
        True: Last installation succeeded
        False: Last installation failed
    """
    with install_mutex:
        return install_result


def get_install_error_message():
    """Safely get the last installation error message."""
    with install_mutex:
        return install_error_message


def clear_install_status():
    """Clear the installation status and error message."""
    global install_result, install_error_message
    with install_mutex:
        install_result = None
        install_error_message = ""
