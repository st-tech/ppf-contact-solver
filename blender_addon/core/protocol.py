# File: protocol.py
# Code: Claude Code and Codex
# Review: Ryoichi Ando (ryoichi.ando@zozo.com)
# License: Apache v2.0
#
# Low-level socket/transport operations extracted from client.py.

import json
import socket
import subprocess

PROTOCOL_VERSION = "0.03"
HEADER_TEXT_CMD = b"TCMD"
HEADER_BINARY_DATA = b"BDAT"
HEADER_JSON_DATA = b"JSON"
DEFAULT_CHUNK_SIZE = 32 * 1024


def open_server_channel(connection):
    """Open a socket or SSH channel to the server, depending on connection type.

    For SSH: returns a paramiko channel (must be closed by caller).
    For docker/local: returns a connected socket (must be closed by caller).
    """
    if connection.type == "ssh":
        transport = connection.instance.get_transport()
        return transport.open_channel(
            kind="direct-tcpip",
            dest_addr=("localhost", connection.server_port),
            src_addr=("localhost", 0),
        )
    elif connection.type in ("docker", "local"):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(("localhost", connection.server_port))
        return s
    else:
        raise ValueError(f"Unsupported connection type: {connection.type}")


def _shutdown_write(channel):
    """Shutdown write side of socket or SSH channel."""
    if hasattr(channel, "shutdown_write"):
        channel.shutdown_write()  # paramiko channel
    else:
        channel.shutdown(socket.SHUT_WR)  # regular socket


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


def exec_command(command, connection, shell=False, cwd=None):
    """Execute a command on the remote server or container.

    Args:
        command: The command string to execute.
        connection: A ConnectionInfo instance.
        shell: Whether to wrap the command with /bin/sh -c.
        cwd: Working directory override; defaults to connection.current_directory.

    Returns:
        dict with keys ``exit_code``, ``stdout`` (list of str), ``stderr`` (list of str).
    """
    if not cwd:
        cwd = connection.current_directory
    exit_code = 0
    output = ""
    error_output = ""
    if shell:
        command = f"/bin/sh -c '{command}'"
    try:
        if connection.type == "ssh":
            if connection.container and not command.startswith("docker"):
                command = f"docker exec -w {cwd} {connection.container} {command}"
            elif not connection.container:
                command = f"cd {cwd} && {command}"
            __stdin__, stdout, stderr = connection.instance.exec_command(command)
            exit_code = stdout.channel.recv_exit_status()
            output = stdout.read().decode().strip()
            error_output = stderr.read().decode().strip()
        elif connection.type == "docker":
            exit_code, stdout = connection.instance.exec_run(command, workdir=cwd)
            if exit_code == 0:
                output = stdout.decode().strip()
            else:
                error_output = stdout.decode().strip()
        elif connection.type == "local":
            try:
                process = subprocess.Popen(
                    command,
                    shell=shell,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, stderr = process.communicate()
                exit_code = process.returncode
                output = stdout.decode().strip()
                error_output = stderr.decode().strip()
            except Exception as e:
                return {
                    "exit_code": 1,
                    "stdout": [],
                    "stderr": [str(e)],
                }
        else:
            raise Exception("No active connection to execute the command.")
    except Exception as e:
        return {
            "exit_code": 1,
            "stdout": [],
            "stderr": [str(e)],
        }

    return {
        "exit_code": exit_code,
        "stdout": output.splitlines(),
        "stderr": error_output.splitlines(),
    }


def send_query(args, connection, project_name, chunk_size=DEFAULT_CHUNK_SIZE):
    """Send a query to the remote server and return the response dict.

    Args:
        args: Dict of query arguments (e.g. ``{"request": "build"}``).
        connection: A ConnectionInfo instance.
        project_name: The project name to include in the query.
        chunk_size: Network chunk size in bytes.

    Returns:
        A tuple ``(response, server_running)`` where *response* is the parsed
        JSON dict and *server_running* is a bool indicating whether the server
        responded successfully.
    """
    if args is None:
        args = {}
    response = {}
    server_running = False

    if connection.instance:
        if project_name is None:
            return response, server_running
        else:
            args["name"] = project_name

        flattened = ""
        for key, value in args.items():
            flattened += f"--{key} {value} "

        channel = None
        try:
            channel = open_server_channel(connection)
            channel.sendall(HEADER_TEXT_CMD)
            data = flattened.encode()
            total_sent = 0
            while total_sent < len(data):
                sent = channel.send(
                    data[total_sent : total_sent + chunk_size]
                )
                if sent == 0:
                    raise RuntimeError("Socket connection broken during send")
                total_sent += sent
            _shutdown_write(channel)
            response_data = b""
            while True:
                data = channel.recv(chunk_size)
                if not data:
                    break
                response_data += data
            if not response_data:
                raise Exception("Empty JSON response.")
            response = json.loads(response_data.decode())
            server_running = True
        except Exception:
            server_running = False
        finally:
            if channel:
                channel.close()

    return response, server_running


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
    sock.sendall(HEADER_JSON_DATA)

    header = json.dumps(request_data).encode() + b"\n"
    header_sent = 0
    while header_sent < len(header):
        sent = sock.send(header[header_sent:])
        if sent == 0:
            raise RuntimeError("Socket connection broken during header send")
        header_sent += sent

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
    sock.sendall(HEADER_JSON_DATA)

    header = json.dumps(request_data).encode() + b"\n"
    header_sent = 0
    while header_sent < len(header):
        sent = sock.send(header[header_sent:])
        if sent == 0:
            raise RuntimeError("Socket connection broken during header send")
        header_sent += sent

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
            raise Exception(msg.get("error", "Server did not confirm upload"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise Exception(f"Server did not confirm upload: {response[:200]!r}")


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
    sock.sendall(HEADER_JSON_DATA)

    header = json.dumps(request_data).encode() + b"\n"
    header_sent = 0
    while header_sent < len(header):
        sent = sock.send(header[header_sent:])
        if sent == 0:
            raise RuntimeError("Socket connection broken during header send")
        header_sent += sent

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
