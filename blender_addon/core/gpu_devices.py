# File: gpu_devices.py
# Code: Claude Code
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# CUDA device enumeration and selection for the solver server.
#
# The add-on starts the solver server on every backend, so on every backend the
# server's environment is out of the user's reach and a machine with more than
# one GPU has nowhere to say which one to use. This module names the devices so
# the choice can be made by GPU rather than by a bare number, and turns a choice
# into the CUDA_VISIBLE_DEVICES the launch carries.
#
# One rule covers every backend: the devices offered are those of the machine
# that will run the server, enumerated by running nvidia-smi THROUGH the
# connection to it. That machine is the local one for Local and Windows Native,
# another host for SSH, and a container for Docker, and none of those
# distinctions reach this module: they are all just "the solver host", reached
# the same way. nvidia-smi is the same tool the server and check_gpu already
# use, so a device listed here is one the solver can be pointed at.

from __future__ import annotations

import shlex
from typing import NamedTuple

# Sentinel for "the add-on sets nothing", which leaves whatever the solver host
# already puts in the server's environment. It is the default.
AUTOMATIC = -1

# The identity columns come first and the free-text name last, so a comma inside
# a marketing name cannot shift the columns that are read by position.
# `ppf_cts_core::utils::parse_gpu_rows` reads the same shape on the server side.
_QUERY = "--query-gpu=index,uuid,name"
_FORMAT = "--format=csv,noheader"
NVIDIA_SMI_ARGS = ("nvidia-smi", _QUERY, _FORMAT)

# What to run on the solver host to enumerate it. One string rather than an
# argv, because every backend's ``exec_command`` takes a shell command.
NVIDIA_SMI_COMMAND = " ".join(NVIDIA_SMI_ARGS)
PROBE_TIMEOUT_SECONDS = 5.0
STALE_SELECTION_ID = 2_147_483_646


class GpuProbeError(RuntimeError):
    """nvidia-smi could not be run, or answered with something unreadable."""


class GpuDevice(NamedTuple):
    """One CUDA device as ``nvidia-smi`` reports it.

    ``index`` is the value saved in the scene and shown in the panel. ``uuid``
    is the launch identity because CUDA's numeric ordering can differ from
    nvidia-smi's.
    """

    index: int
    name: str
    uuid: str


def parse_nvidia_smi_devices(text: str) -> list[GpuDevice]:
    """Parse ``nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader``.

    Raises:
        GpuProbeError: if *text* holds no readable device row. An empty answer
            means no NVIDIA device is visible, which the caller must not read
            as "the machine has one GPU and it is fine".
    """
    devices: list[GpuDevice] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # The name is whatever remains after the two identity columns, so it
        # survives a comma of its own intact.
        parts = line.split(",", 2)
        if len(parts) != 3:
            continue
        try:
            index = int(parts[0].strip())
        except ValueError:
            continue
        devices.append(
            GpuDevice(index=index, name=parts[2].strip(), uuid=parts[1].strip())
        )
    if not devices:
        raise GpuProbeError(
            "nvidia-smi listed no CUDA device. An NVIDIA GPU is required to "
            "run the solver."
        )
    return devices


# ---------------------------------------------------------------------------
# Session cache
#
# The dropdown asks for this list on every redraw, and the answer costs a
# command on the solver host, so it is read once per connection and reused. The
# outcome is kept whether it succeeded or failed, so a host with no NVIDIA
# driver is not re-probed on every redraw. Refresh is what re-runs it.

_devices: list[GpuDevice] = []
_probe_error: str = ""
_probed: bool = False


def load_devices(text: str) -> list[GpuDevice]:
    """Install the device list parsed from nvidia-smi output *text*."""
    global _devices, _probe_error, _probed
    try:
        _devices = parse_nvidia_smi_devices(text)
        _probe_error = ""
    except GpuProbeError as exc:
        _devices = []
        _probe_error = str(exc)
    _probed = True
    return _devices


def record_probe_failure(message: str) -> None:
    """Record that the solver host could not be enumerated, with the reason.

    Used when the probe fails before any output exists, which is what an
    unreachable host or a backend command error looks like.
    """
    global _devices, _probe_error, _probed
    _devices = []
    _probe_error = message
    _probed = True


def forget_devices() -> None:
    """Drop what is known, so nothing is offered until the next probe.

    Called when a connection ends: the next one may reach a different machine,
    where a list left over from this one would name GPUs that are not there.
    """
    global _devices, _probe_error, _probed
    _devices = []
    _probe_error = ""
    _probed = False


def cached_gpu_devices() -> list[GpuDevice]:
    """The solver host's devices, or an empty list before it has been probed.

    Never probes: reaching the solver host means a command over the backend,
    which belongs on the worker thread that owns the connection, not in the
    panel draw that calls this.
    """
    return _devices


def gpu_probe_error() -> str:
    """The message from the last probe, or "" if it succeeded or none has run."""
    return _probe_error


def has_probed() -> bool:
    """True once the solver host has been probed for this connection."""
    return _probed


# ---------------------------------------------------------------------------
# Selection


def find_device(index: int, devices: list[GpuDevice] | None = None) -> GpuDevice | None:
    """Return the enumerated device with CUDA index *index*, or None."""
    if index < 0:
        return None
    for device in cached_gpu_devices() if devices is None else devices:
        if device.index == index:
            return device
    return None


def find_device_by_uuid(
    uuid: str, devices: list[GpuDevice] | None = None
) -> GpuDevice | None:
    """Return the enumerated device with exactly matching *uuid*, or None."""
    if not uuid:
        return None
    for device in cached_gpu_devices() if devices is None else devices:
        if device.uuid == uuid:
            return device
    return None


def selected_device(
    index: int, uuid: str = "", devices: list[GpuDevice] | None = None
) -> GpuDevice | None:
    """Resolve a saved selection, preferring its stable UUID."""
    if uuid:
        return find_device_by_uuid(uuid, devices)
    return find_device(index, devices)


def selection_token(index: int, uuid: str = "") -> str:
    """CUDA visibility token for a saved GPU selection."""
    device = selected_device(index, uuid)
    if device is not None:
        return device.uuid
    return uuid or str(int(index))


def describe_launch(index: int, launched: bool, uuid: str = "") -> str:
    """One console line recording which GPU a server start used.

    A run's GPU is otherwise only visible while the panel is open, so a session
    read back later has no way to tell which device produced it. *launched* is
    False when the add-on attached to a server that was already running, where
    the selection reached nothing and the device is whatever that server
    started on.
    """
    if not launched:
        return (
            "Solver server: attached to one that was already running, so its "
            "GPU is the one it started with. Remote Hardware in the panel "
            "names it."
        )
    if index == AUTOMATIC and not uuid:
        return (
            "Solver server started with GPU Automatic, so it keeps whatever "
            "CUDA_VISIBLE_DEVICES its own host sets"
        )
    device = selected_device(index, uuid)
    display_index = device.index if device is not None else index
    named = f" ({device.name})" if device is not None else ""
    token = selection_token(index, uuid)
    return (
        f"Solver server started on GPU {display_index}{named}, "
        f"CUDA_VISIBLE_DEVICES={token}"
    )


def shell_prefix(index: int, uuid: str = "") -> str:
    """CUDA visibility assignment to prepend to a server command, or "".

    The backends that start the server through a shell deliver the variable as
    an assignment in front of the binary. ``AUTOMATIC`` contributes nothing, so
    the launch keeps whatever the solver host's own shell sets.
    """
    if index == AUTOMATIC and not uuid:
        return ""
    return f"CUDA_VISIBLE_DEVICES={shlex.quote(selection_token(index, uuid))} "


def apply_selection(env: dict, index: int, uuid: str = "") -> dict:
    """Write the device selection for *index* into environment mapping *env*.

    The counterpart of :func:`shell_prefix` for the backend that spawns the
    server as a child process, where there is no shell to carry an assignment.
    ``AUTOMATIC`` leaves *env* untouched, so a ``CUDA_VISIBLE_DEVICES`` the
    add-on's own process inherited passes through. Any other value replaces it:
    the panel's choice is the more specific one, and a selection that silently
    lost to an inherited variable would put the solver on a GPU the panel does
    not name.

    Returns *env* so a caller can chain it.
    """
    if index != AUTOMATIC or uuid:
        env["CUDA_VISIBLE_DEVICES"] = selection_token(index, uuid)
    return env


def validate_selection(index: int, uuid: str = "") -> None:
    """Raise if *index* names a device the solver host does not have.

    A stale selection (a .blend saved against a multi-GPU host, opened against
    a smaller one) would otherwise reach CUDA as a device set with nothing in
    it, and the solver would fail later with an error naming no GPU. Checked
    only when the devices could actually be enumerated: with no list there is
    no evidence to contradict what the user asked for, and inventing one would
    be worse than honoring the request.

    Raises:
        ValueError: naming the selected index and what is present.
    """
    if index == AUTOMATIC and not uuid:
        return
    devices = cached_gpu_devices()
    if not devices:
        return
    if selected_device(index, uuid, devices) is not None:
        return
    present = ", ".join(f"{d.index} ({d.name})" for d in devices)
    wanted = f"GPU {index}" if not uuid else f"GPU {index} ({uuid})"
    raise ValueError(
        f"{wanted} is not present on the solver host. Detected: {present}."
    )
