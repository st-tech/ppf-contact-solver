# File: module.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import importlib.util
import os
import subprocess
import sys
import threading

from ..models.console import console


def get_lib_dir():
    """Get the directory of the add-on lib."""
    addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(addon_dir, "lib")


def import_module(name):
    """
    Import a module from the add-on lib directory, supporting dotted names (e.g., 'module.submodule').

    Args:
        name (str): The module name, possibly with dots.

    Returns:
        module: The imported module object.
    """
    lib_dir = get_lib_dir()
    parts = name.split(".")
    if len(parts) == 1:
        path = os.path.join(lib_dir, parts[0], "__init__.py")
    else:
        path = os.path.join(lib_dir, *parts[:-1], parts[-1] + ".py")
        if not os.path.isfile(path):
            path = os.path.join(lib_dir, *parts, "__init__.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such file: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None, f"Cannot find spec for {name}"
    if spec.loader is None:
        raise AttributeError(f"The loader for {name} is None")
    module = importlib.util.module_from_spec(spec)
    save_path = sys.path.copy()
    sys.path.insert(0, lib_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path = save_path
    return module


def module_exists(packages):
    """
    Check if a list of modules exist.

    Args:
        packages (list): List of package names to check.

    Returns:
        bool: True if all packages exist, False otherwise.
    """
    lib_dir = get_lib_dir()
    for package in packages:
        if not os.path.exists(os.path.join(lib_dir, package, "__init__.py")):
            return False
    return True


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
            lib_path = get_lib_dir()
            console.write(f"Installing packages to {lib_path}: {', '.join(packages)}")

            subprocess.check_call(
                [sys.executable, "-m", "ensurepip", "--upgrade"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            process = subprocess.Popen(
                [sys.executable, "-m", "pip", "install", "--target", lib_path]
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
