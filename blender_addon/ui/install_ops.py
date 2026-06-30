# File: install_ops.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# cbor2, Paramiko, and Docker installation operators.

from ..core.async_op import AsyncOperator
from ..core.utils import redraw_all_areas
from ..core.module import clear_install_status, get_installing_status, install_module


class REMOTE_OT_InstallCbor2(AsyncOperator):
    """Install the cbor2 library.

    Recovery path for installs where the manifest's bundled cbor2 wheel
    was never installed (the addon was copied in manually, or a
    5.0->5.1 settings migration carried it over without reinstalling its
    wheels). The normal path is the wheel in ``blender_manifest.toml``;
    this button pip-installs the matching wheel for the running
    interpreter into the addons/modules directory, which is on
    ``sys.path``. Pinned to 6.0.1 to match the bundled wheel set."""

    bl_idname = "ssh.install_cbor2"
    bl_label = "Install cbor2 to Add-on Directory"

    timeout: float = 120.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, _):
        return get_installing_status() is False

    def execute(self, context):
        clear_install_status()
        install_module(["cbor2==6.0.1"])
        self.setup_modal(context)
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        return not get_installing_status()

    def on_complete(self, context):
        redraw_all_areas(context)


class REMOTE_OT_InstallParamiko(AsyncOperator):
    """Install the Paramiko library."""

    bl_idname = "ssh.install_paramiko"
    bl_label = "Install Paramiko to Add-on Directory"

    timeout: float = 120.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, _):
        return get_installing_status() is False

    def execute(self, context):
        clear_install_status()
        install_module(["paramiko"])
        self.setup_modal(context)
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        return not get_installing_status()

    def on_complete(self, context):
        redraw_all_areas(context)


class REMOTE_OT_InstallDocker(AsyncOperator):
    """Install the Docker library."""

    bl_idname = "ssh.install_docker"
    bl_label = "Install Docker-Py to Add-on Directory"

    timeout: float = 120.0
    auto_redraw: bool = True

    @classmethod
    def poll(cls, _):
        return get_installing_status() is False

    def execute(self, context):
        clear_install_status()
        install_module(["docker"])
        self.setup_modal(context)
        return {"RUNNING_MODAL"}

    def is_complete(self) -> bool:
        return not get_installing_status()

    def on_complete(self, context):
        redraw_all_areas(context)


classes = [
    REMOTE_OT_InstallCbor2,
    REMOTE_OT_InstallParamiko,
    REMOTE_OT_InstallDocker,
]
