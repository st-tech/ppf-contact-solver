"""Fill the Backend Communicator with a representative connection.

Passed to ``blender_addon/capture.sh --pre-python`` and steered by
``PPF_CAPTURE_CONNECTION`` (one of the ``server_type`` enum values in
``blender_addon/ui/state.py``, default ``CUSTOM`` = plain SSH)::

    PPF_CAPTURE_CONNECTION=DOCKER \\
    bash blender_addon/capture.sh -o docs/blender_addon/images/connections \\
        --pre-python docs/tools/capture_connection.py \\
        "MAIN_PT_RemotePanel:Connect"

Why fill anything at all: the connection pages show which fields a given
transport asks for, and an all-empty panel makes the rows hard to tell
apart at documentation width. The values below are deliberately generic
placeholders — a reader should see the SHAPE of the form, not a working
host, and not whoever regenerated the screenshot last. That matters here
in particular, because the previous captures went out carrying their
author's real home directory in the SSH Key field.
"""

import os

import bpy

from bl_ext.user_default.ppf_contact_solver.models.groups import get_addon_data

# Only the fields the panel draws for the chosen type need to be set;
# the rest stay at their defaults and are not drawn.
PLACEHOLDERS = {
    "host": "gpu-host.example.com",
    "username": "ubuntu",
    "key_path": "~/.ssh/id_rsa",
    "ssh_remote_path": "/home/ubuntu/ppf-contact-solver",
    "command": "ssh gpu-host",
    "docker_path": "/root/ppf-contact-solver",
}


def main() -> None:
    server_type = os.environ.get("PPF_CAPTURE_CONNECTION", "CUSTOM")
    state = get_addon_data(bpy.context.scene).ssh_state

    state.server_type = server_type
    for field, value in PLACEHOLDERS.items():
        if hasattr(state, field):
            setattr(state, field, value)

    print(f"capture_connection: server_type={server_type}")


main()
