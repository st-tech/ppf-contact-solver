# File: protocol.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Low-level socket/transport operations extracted from client.py.

import json

# Must match crates/ppf-cts-server/src/lib.rs:PROTOCOL_VERSION.
# A mismatch trips transitions.py's strict-equality protocol check.
#
# BUMP THIS whenever the data / param payload schema OR the
# addon-encoder / frontend-decoder contract changes. The handshake is
# the only thing that catches an addon paired with a server whose
# frontend decoder disagrees about payload shape; an un-bumped version
# there silently mis-decodes instead of erroring.
#
# 0.09: additive scene / dyn-param fields for the per-object bending
# reference rest shape and the angular velocity overwrite. ObjectInfo
# gains an optional bend_rest_vert (guarded by a has_bend_rest_vert
# count) carrying per-object reference rest positions for hinge rest
# angles, and the dyn-param table gains angular_velocity:<dmap> /
# angular_velocity_world:<dmap> keyframe streams (principal-axis spins
# resolved from live geometry, and fixed world-axis spins). The new
# fields are optional, so an old decoder silently drops the bending
# reference and the spins instead of erroring; the handshake forces the
# matching pair.
#
# 0.08: co-located transfer. When the addon and server share a
# machine (local / win_native backends), the addon writes
# data.pickle / param.pickle straight to the project root on disk and
# then sends a lightweight upload_notify JSON request (carrying the
# addon-minted upload_id, the data / param hashes, and has_data /
# has_param) instead of streaming the payloads through the socket via
# upload_atomic. The server stamps the supplied id, then dispatches
# the same UploadLanded event the streamed path does. Set
# PPF_FORCE_TCP_TRANSFER=1 to keep local / win_native on the streamed
# path (the test rig does this so every scenario but the dedicated
# direct-disk one still exercises the wire handlers). An old server
# paired with a new addon rejects upload_notify as an unknown
# request; both directions are caught by this handshake.
#
# 0.07: moving STATIC colliders join the output vertex map. The
# pin-shells produced by transform_animation (Case 1) and
# static_deform_animation (Case 3) are no longer flagged
# `_exclude_from_output`, so their simulator-projected per-frame
# positions stream back as a regular PC2 + ContactSolverCache
# modifier on the static object. An old client paired with a new
# server would receive vertex frames for static UUIDs its decoder
# wasn't expecting; a new client paired with an old server would
# silently keep displaying the input animation while the cloth
# resolves against the soft-pinned positions.
#
# 0.06: deforming STATIC mesh colliders. A STATIC object whose
# modifier stack deforms vertices (Armature, MeshDeform, Lattice,
# shape keys, etc.) can now carry a static_deform_animation payload
# alongside vert / transform: a per-frame absolute vertex buffer in
# solver world space, captured from Blender's depsgraph via the new
# Capture Deformation operator. The decoder builds a zero-stiffness
# pin shell whose every vertex is driven by MoveByOperation segments
# derived from consecutive frames, just like the per-vertex pin
# animation in 0.05 but spanning the whole mesh. Mutually exclusive
# with transform_animation and static_ops; an old decoder would
# ignore the new field and play the rest-pose mesh.
#
# 0.05: keyframed pin animation. param.pin_config[uuid] stays
# {vertex_index: PinData}, but each PinData now carries its own
# single-entry pin_anim ({that_vertex: PinAnim}) and the frontend
# decoder builds genuine per-vertex MoveByOperation deltas from it
# instead of broadcasting one vertex's track. An old decoder would
# treat the new payload as a rigid translation.
#
# 0.04: TCMD requests carry a 4-byte big-endian length prefix between
# the b"TCMD" header and the payload, replacing the prior wire that
# relied on shutdown(SHUT_WR) as the end-of-input signal. Windows
# tokio did not deliver that half-close to the server's AsyncRead, so
# the server hung in its read loop and connections piled up in
# FIN_WAIT_2 until the server stopped responding entirely.
PROTOCOL_VERSION = "0.11"
HEADER_TEXT_CMD = b"TCMD"
HEADER_JSON_DATA = b"JSON"
DEFAULT_CHUNK_SIZE = 32 * 1024

# Wire status tokens. These are the raw ``status`` strings the server
# stamps on every query response (see crates/ppf-cts-server). Defining
# them once here lets derived.py, transitions.py, and effect_runner.py
# reference the same constants, so a server-side rename only needs to be
# tracked in one place. These are the protocol-level tokens; the
# human-readable display strings live in core/status.py.
STATUS_NO_DATA = "NO_DATA"
STATUS_NO_BUILD = "NO_BUILD"
STATUS_BUILDING = "BUILDING"
STATUS_READY = "READY"
STATUS_RESUMABLE = "RESUMABLE"
STATUS_FAILED = "FAILED"
STATUS_BUSY = "BUSY"
STATUS_SAVE_AND_QUIT = "SAVE_AND_QUIT"

# Curated status sets. ``SIM_RUNNING_STATUSES`` is the narrow "actively
# producing frames" check (BUSY or SAVE_AND_QUIT); ``SERVER_BUSY_STATUSES``
# also includes BUILDING for the broader "do not interrupt the server"
# check. Frozen so callers cannot mutate the shared module-level sets.
SIM_RUNNING_STATUSES = frozenset({STATUS_BUSY, STATUS_SAVE_AND_QUIT})
SERVER_BUSY_STATUSES = frozenset(
    {STATUS_BUSY, STATUS_SAVE_AND_QUIT, STATUS_BUILDING}
)

# Rejection message for a data_send aimed at the scene pickles. The same
# invariant is enforced server-side in crates/ppf-cts-server/src/wire/data.rs
# (byte-identical text); keeping the one ``{basename}`` template here lets the
# client paths reuse it so the two wordings cannot drift apart.
DATA_SEND_PICKLE_REJECT = (
    "data_send no longer accepts {basename}; "
    "use upload_atomic for scene uploads."
)


def format_traffic(bytes_per_second):
    """Format the traffic string based on the speed in B/s."""
    if bytes_per_second is None or bytes_per_second == 0.0:
        return ""
    if bytes_per_second < 1024:
        return f"{bytes_per_second:.2f} B/s"
    elif bytes_per_second < 1024 * 1024:
        kb_per_second = bytes_per_second / 1024
        return f"{kb_per_second:.2f} KB/s"
    else:
        mb_per_second = bytes_per_second / (1024 * 1024)
        return f"{mb_per_second:.2f} MB/s"


def _send_json_header(sock, request_data):
    """Send the ``JSON`` header byte tag plus a newline-terminated JSON header.

    Args:
        sock: Socket or SSH channel exposing ``sendall``/``send``.
        request_data: JSON-serializable dict sent as the header.
    """
    sock.sendall(HEADER_JSON_DATA)
    header = json.dumps(request_data).encode() + b"\n"
    sent = 0
    while sent < len(header):
        n = sock.send(header[sent:])
        if n == 0:
            raise RuntimeError("Socket connection broken during header send")
        sent += n


def _read_ok_response(sock, fail_label="Server did not confirm upload"):
    """Read the server's newline-terminated confirmation and raise on failure.

    Args:
        sock: Socket or SSH channel exposing ``recv``.
        fail_label: Message used when the server neither confirms with
            ``OK`` nor sends a parseable ``{"error": ...}`` payload.

    Raises:
        Exception if the response does not contain ``OK``. The server's
        error payload is bubbled up when present; otherwise the raw
        response is truncated into the message.
    """
    response = b""
    while b"\n" not in response:
        chunk = sock.recv(1024)
        if not chunk:
            break
        response += chunk

    if b"OK" not in response:
        # Bubble up the server's error payload if it sent one.
        try:
            msg = json.loads(response.decode().strip())
            raise Exception(msg.get("error", fail_label))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise Exception(f"{fail_label}: {response[:200]!r}")


def socket_data_send(
    sock,
    request_data,
    data,
    chunk_size,
    progress_callback,
    interrupt_callback,
    bps_calculator,
):
    """Send data via socket connection.

    Args:
        sock: Socket or SSH channel.
        request_data: JSON-serializable dict sent as the header.
        data: Raw bytes to send.
        chunk_size: Size of each chunk.
        progress_callback: ``fn(progress_float, traffic_str)`` or None.
        interrupt_callback: ``fn() -> bool`` or None.
        bps_calculator: A BytesPerSecondCalculator instance.
    """
    _send_json_header(sock, request_data)

    total_sent = 0

    for i in range(0, len(data), chunk_size):
        if interrupt_callback and interrupt_callback():
            raise Exception("Data send interrupted.")

        chunk = data[i : i + chunk_size]
        chunk_sent = 0

        while chunk_sent < len(chunk):
            sent = sock.send(chunk[chunk_sent:])
            if sent == 0:
                raise RuntimeError("Socket connection broken during send")
            chunk_sent += sent
            total_sent += sent

        bps_calculator.add_sample(total_sent)
        bps = bps_calculator.get_bytes_per_second()

        if progress_callback:
            progress_callback(
                min(1.0, total_sent / len(data)), format_traffic(bps)
            )

    response = b""
    while b"\n" not in response:
        chunk = sock.recv(1024)
        if not chunk:
            break
        response += chunk

    if b"OK" not in response:
        raise Exception("Server did not confirm data receipt")


def socket_upload_atomic(
    sock,
    request_data,
    data,
    param,
    chunk_size,
    progress_callback,
    interrupt_callback,
    bps_calculator,
):
    """Send a ``upload_atomic`` request and the two payloads back-to-back.

    Args:
        sock: Socket or SSH channel, already opened to the server.
        request_data: JSON dict sent as the header; must carry
            ``{"request": "upload_atomic", "name": ..., "path": ...,
              "data_size": <int>, "param_size": <int>}``. Either size may
            be zero (update just one file), but at least one must be > 0.
        data: Raw bytes for ``data.pickle``. Pass ``b""`` to skip.
        param: Raw bytes for ``param.pickle``. Pass ``b""`` to skip.
        chunk_size: Network chunk size in bytes.
        progress_callback: ``fn(progress_float, traffic_str)`` or None.
            Progress is the combined fraction across both payloads.
        interrupt_callback: ``fn() -> bool`` or None. Interrupts raise.
        bps_calculator: A BytesPerSecondCalculator instance.

    Raises:
        Exception if the server does not confirm with ``OK``.
    """
    _send_json_header(sock, request_data)

    total_size = len(data) + len(param)
    total_sent = 0

    def _send_payload(payload):
        nonlocal total_sent
        for i in range(0, len(payload), chunk_size):
            if interrupt_callback and interrupt_callback():
                raise Exception("Upload interrupted.")
            chunk = payload[i : i + chunk_size]
            chunk_sent = 0
            while chunk_sent < len(chunk):
                sent = sock.send(chunk[chunk_sent:])
                if sent == 0:
                    raise RuntimeError("Socket connection broken during send")
                chunk_sent += sent
                total_sent += sent
            bps_calculator.add_sample(total_sent)
            bps = bps_calculator.get_bytes_per_second()
            if progress_callback and total_size > 0:
                progress_callback(
                    min(1.0, total_sent / total_size), format_traffic(bps)
                )

    if data:
        _send_payload(data)
    if param:
        _send_payload(param)

    _read_ok_response(sock)


def socket_data_receive(
    sock, request_data, chunk_size, progress_callback, interrupt_callback, bps_calculator
):
    """Receive data via socket connection.

    Args:
        sock: Socket or SSH channel.
        request_data: JSON-serializable dict sent as the header.
        chunk_size: Size of each receive chunk.
        progress_callback: ``fn(progress_float, traffic_str)`` or None.
        interrupt_callback: ``fn() -> bool`` or None.
        bps_calculator: A BytesPerSecondCalculator instance.

    Returns:
        The received bytes.
    """
    _send_json_header(sock, request_data)

    metadata_response = b""
    while b"\n" not in metadata_response:
        chunk = sock.recv(1024)
        if not chunk:
            raise Exception("Connection closed while receiving metadata")
        metadata_response += chunk

    newline_pos = metadata_response.find(b"\n")
    metadata_json = metadata_response[:newline_pos]
    remaining_data = metadata_response[newline_pos + 1 :]

    try:
        metadata = json.loads(metadata_json.decode().strip())
        if "error" in metadata:
            raise Exception(f"Server error: {metadata['error']}")
        total_size = metadata.get("size", 0)
    except UnicodeDecodeError as e:
        raise Exception(
            f"UTF-8 decode error in metadata: {e}. Raw bytes: {metadata_json[:50]}"
        ) from e
    except json.JSONDecodeError as e:
        raise Exception(
            f"Invalid metadata response from server. Raw bytes: {metadata_json[:50]}"
        ) from e

    data = remaining_data
    bytes_received = len(remaining_data)

    while bytes_received < total_size:
        if interrupt_callback and interrupt_callback():
            raise Exception("Data receive interrupted.")

        remaining = total_size - bytes_received
        recv_size = min(chunk_size, remaining)
        chunk = sock.recv(recv_size)

        if not chunk:
            raise Exception("Connection closed during data transfer")

        data += chunk
        bytes_received += len(chunk)
        bps_calculator.add_sample(bytes_received)
        bps = bps_calculator.get_bytes_per_second()

        if progress_callback and total_size > 0:
            progress_callback(
                min(1.0, bytes_received / total_size),
                format_traffic(bps),
            )

    return data
