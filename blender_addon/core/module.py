# File: module.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0

import importlib
import importlib.util
import os
import subprocess
import sys
import threading

from ..models.console import console


def get_lib_dir():
    """Get the directory of the add-on lib (legacy install target)."""
    addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(addon_dir, "lib")


def get_install_target():
    """Return the directory where ``install_module`` drops new packages.

    Prefer Blender's user ``scripts/addons/modules`` dir: it is already
    on ``sys.path``, and being outside any extension directory it doesn't
    trigger Blender 5's "Policy violation with top level module" warning
    that fires for every module loaded from inside an extension's tree.
    Falls back to the addon's ``lib/`` directory when bpy is unavailable
    (pytest, standalone debug-rig harnesses) so the install path still
    resolves to *something* writable.
    """
    try:
        import bpy  # pyright: ignore
        path = bpy.utils.user_resource("SCRIPTS", path="addons/modules", create=True)
        if path:
            return os.path.normpath(path)
    except Exception:
        pass
    return get_lib_dir()


def import_module(name):
    """Import a module by dotted name, preferring the normal sys.path.

    Args:
        name (str): The module name, possibly with dots.

    Returns:
        module: The imported module object.

    Order of resolution:
    1. ``importlib.import_module`` -- finds the package in
       ``scripts/addons/modules`` (the new install target) or anywhere
       else already on ``sys.path``.
    2. Legacy fallback: load directly from ``<addon>/lib/``. Kept so
       users whose old ``install_module`` runs landed inside the
       extension dir still work until they re-click Install Modules.
       This path triggers Blender 5's policy-violation warning.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        pass

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


def get_cbor2():
    """Resolve cbor2 from ``blender_addon/lib/`` when sideloaded, falling
    back to a plain import for the wheel-installed path (Extensions UI)
    and standalone harnesses (pytest, build_worker.py, the test rig)."""
    try:
        return import_module("cbor2")
    except (ImportError, FileNotFoundError):
        import cbor2

        return cbor2


def module_exists(packages):
    """
    Check whether each package is importable.

    Looks first in the new install target (``scripts/addons/modules``)
    and falls back to the legacy ``<addon>/lib/`` location, so both old
    and new installs are recognized.

    Args:
        packages (list): List of package names to check.

    Returns:
        bool: True if all packages exist, False otherwise.
    """
    candidates = [get_install_target(), get_lib_dir()]
    for package in packages:
        if not any(
            os.path.exists(os.path.join(d, package, "__init__.py"))
            for d in candidates
        ):
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
