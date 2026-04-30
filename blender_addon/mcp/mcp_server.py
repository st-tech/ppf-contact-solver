"""Pure Socket-based MCP Server for Blender integration."""

import contextlib
import socket
import threading
import time

from http.server import ThreadingHTTPServer

from ..core.utils import get_timer_wait_time
from ..models.defaults import DEFAULT_MCP_PORT
from .http_handler import MCPRequestHandler
from .server_utils import is_port_available, wait_for_port_release
from .sessions import clear_all_sessions


class BlenderMCPServer:
    """Pure socket-based MCP Server for exposing Blender functionality via HTTP."""

    def __init__(self):
        self.server = None
        self.thread = None
        self.running = False
        self.port = DEFAULT_MCP_PORT
        self.shutdown_event = threading.Event()

    def is_running(self):
        """Check if the MCP server is currently running."""
        return self.running and self.thread and self.thread.is_alive()

    def start(self, port=DEFAULT_MCP_PORT):
        """Start the MCP server in a separate thread with enhanced cleanup."""
        # First, try to stop any existing instance gracefully
        if self.is_running():
            print("MCP Server: Stopping existing instance...")
            self.stop()
            time.sleep(0.5)  # Wait for graceful shutdown

        self.port = port

        # Check if port is available
        if not is_port_available(port):
            print(f"MCP Server: Port {port} is not available")

            # Wait for port to become available with exponential backoff
            if wait_for_port_release(port, max_attempts=5, initial_delay=0.2):
                print(f"MCP Server: Port {port} is now available")
            else:
                # Try alternative ports
                print("MCP Server: Searching for alternative port...")
                original_port = port
                for alt_port in range(port + 1, port + 10):
                    if is_port_available(alt_port):
                        print(
                            f"MCP Server: Using alternative port {alt_port} (original {original_port} unavailable)"
                        )
                        port = alt_port
                        self.port = port
                        break
                else:
                    raise Exception(
                        f"MCP Server: Could not find available port in range {original_port}-{original_port + 9}"
                    )

        self.running = True
        self.shutdown_event.clear()  # Reset shutdown event

        def run_server():
            try:
                self.server = ThreadingHTTPServer(
                    ("localhost", port), MCPRequestHandler
                )

                # Configure server socket for better shutdown behavior
                self.server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if hasattr(socket, "SO_REUSEPORT"):
                    with contextlib.suppress(OSError):
                        self.server.socket.setsockopt(
                            socket.SOL_SOCKET, socket.SO_REUSEPORT, 1
                        )

                print(f"MCP Server: Successfully started on localhost:{port}")

                # Use timeout to make shutdown more responsive
                self.server.timeout = 1.0

                while self.running and not self.shutdown_event.is_set():
                    try:
                        self.server.handle_request()
                    except OSError:
                        # Socket was closed, exit gracefully
                        break

            except Exception as e:
                print(f"MCP Server error: {e}")
            finally:
                self.running = False
                self.shutdown_event.set()
                print("MCP Server: Server thread exiting")

        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()

        # Give the server a moment to start and verify it's running
        time.sleep(get_timer_wait_time())

        # Verify the server actually started
        if not is_port_available(port):
            # Server should have bound to the port
            print(f"MCP Server: Confirmed running on port {port}")
        else:
            # Something went wrong
            self.running = False
            raise Exception(f"MCP Server: Failed to bind to port {port}")

    def stop(self):
        """Stop the MCP server with proper coordination and error handling."""
        if not self.running:
            print("MCP Server: Already stopped")
            return

        print("MCP Server: Initiating shutdown...")

        # Step 1: Signal shutdown to prevent new requests
        self.running = False
        self.shutdown_event.set()

        # Close any live Streamable HTTP sessions so their SSE threads exit.
        clear_all_sessions()

        # Step 2: Close server socket to stop accepting connections
        if self.server:
            try:
                print("MCP Server: Closing server socket...")
                self.server.server_close()
                print("MCP Server: Server socket closed")
            except Exception as e:
                print(f"MCP Server: Error closing server socket: {e}")

        # Step 3: Wait for server thread to finish with increased timeout
        if self.thread and self.thread.is_alive():
            print("MCP Server: Waiting for server thread to finish...")
            self.thread.join(timeout=5.0)

            if self.thread.is_alive():
                print(
                    "MCP Server: Warning - server thread did not terminate within 5 seconds"
                )
            else:
                print("MCP Server: Server thread terminated successfully")

        # Step 4: Clean up references
        self.server = None
        self.thread = None

        # Step 5: Wait for port to be released
        if hasattr(self, "port"):
            print(f"MCP Server: Waiting for port {self.port} to be released...")

            if wait_for_port_release(self.port, max_attempts=8, initial_delay=0.1):
                print(f"MCP Server: Port {self.port} successfully released")
            else:
                print(f"MCP Server: Warning - port {self.port} may still be occupied")
                print(
                    "MCP Server: This is usually temporary and will resolve automatically"
                )

        print("MCP Server: Shutdown complete")


# Global MCP server instance
mcp_server = BlenderMCPServer()


def get_mcp_server():
    """Get the global MCP server instance."""
    return mcp_server


def is_mcp_running():
    """Check if MCP server is currently running."""
    return mcp_server.is_running()


def start_mcp_server(port=DEFAULT_MCP_PORT):
    """Start the MCP server."""
    return mcp_server.start(port)


def stop_mcp_server():
    """Stop the MCP server."""
    return mcp_server.stop()


def cleanup_mcp_server():
    """Cleanup MCP server during addon unregister."""
    try:
        if mcp_server.is_running():
            print("MCP Server: Cleaning up during addon unregister...")
            stop_mcp_server()
        else:
            # Even if we think it's not running, check the port
            if hasattr(mcp_server, "port") and not is_port_available(mcp_server.port):
                print(
                    f"MCP Server: Found process on port {mcp_server.port}, waiting for release..."
                )
                wait_for_port_release(
                    mcp_server.port, max_attempts=5, initial_delay=0.2
                )
    except Exception as e:
        print(f"MCP Server: Error during cleanup: {e}")
    try:
        from .task_system import clear_task_state
        clear_task_state()
    except Exception:
        pass
