"""Cross-platform utility functions for MCP server operations."""

import socket
import sys
import time


def is_port_available(port, host="localhost"):
    """Check if a port is available for binding.

    On Windows, ``SO_REUSEADDR`` allows a new socket to bind on top of an
    already-listening one, which would make this check return True even
    when a listener is active. The check must behave like the real
    HTTPServer bind (no reuse) so that a running server is correctly
    detected as "port in use".
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sys.platform != "win32":
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            return True
    except OSError:
        return False


def wait_for_port_release(port, host="localhost", max_attempts=10, initial_delay=0.1):
    """Wait for a port to be released with exponential backoff.

    Args:
        port: Port number to check
        host: Host address (default: localhost)
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay in seconds

    Returns:
        bool: True if port becomes available, False if timeout
    """
    delay = initial_delay

    for attempt in range(max_attempts):
        if is_port_available(port, host):
            if attempt > 0:
                print(f"Port {port} became available after {attempt + 1} attempts")
            return True

        if attempt < max_attempts - 1:  # Don't sleep on last attempt
            print(
                f"Port {port} still occupied, waiting {delay:.2f}s... (attempt {attempt + 1}/{max_attempts})"
            )
            time.sleep(delay)
            delay = min(delay * 1.5, 2.0)  # Exponential backoff with max 2s

    return False
