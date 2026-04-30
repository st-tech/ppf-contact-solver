"""MCP Task Processing System for thread-safe Blender operations with integrated decorator support."""

import threading
import time

# Import handlers to trigger decorator registration
from .handlers import connection, console, debug, group, remote, simulation  # noqa: F401
from .integration import get_integrated_handlers, initialize_integrated_system

_get_integrated_handlers = get_integrated_handlers
_initialize_integrated_system = initialize_integrated_system


# MCP Task Processing System (keep existing task system)
_mcp_task_queue = []
_mcp_results = {}
_mcp_lock = threading.Lock()
_mcp_task_id_counter = 0


def post_mcp_task(task_type, args):
    """Post a task to be processed by the Blender timer in the main thread."""
    global _mcp_task_id_counter

    with _mcp_lock:
        _mcp_task_id_counter += 1
        task_id = f"mcp_task_{_mcp_task_id_counter}"

        task = {
            "id": task_id,
            "type": task_type,
            "args": args,
            "timestamp": time.time(),
        }

        _mcp_task_queue.append(task)

    return task_id


def get_mcp_result(task_id, timeout=5.0):
    """Wait for and retrieve the result of a posted task."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        with _mcp_lock:
            if task_id in _mcp_results:
                result = _mcp_results[task_id]
                del _mcp_results[task_id]  # Clean up
                return result

        time.sleep(0.01)  # Small sleep to prevent busy waiting

    # Timeout occurred
    return {
        "status": "error",
        "message": f"Task {task_id} timed out after {timeout} seconds",
    }


def process_mcp_tasks():
    """Process MCP tasks in the main thread. Called by Blender timer."""
    tasks_to_process = []

    # Get all pending tasks
    with _mcp_lock:
        tasks_to_process = _mcp_task_queue[:]
        _mcp_task_queue.clear()

    # Process each task
    for task in tasks_to_process:
        task_id = task["id"]
        task_type = task["type"]
        args = task["args"]

        try:
            result = _execute_blender_task(task_type, args)
        except Exception as e:
            result = {"status": "error", "message": str(e)}

        # Store result
        with _mcp_lock:
            _mcp_results[task_id] = result


def _execute_blender_task(task_type, args):
    """Execute a Blender task in the main thread where bpy access is safe."""
    handlers = _get_integrated_handlers()
    handler = handlers.get(task_type)
    if handler:
        return handler(args)
    return {"status": "error", "message": f"Unknown task type: {task_type}"}


def clear_task_state():
    """Drop any pending tasks and results. Called on addon unregister so
    a subsequent reload doesn't try to run handlers that point into freed
    module namespaces."""
    with _mcp_lock:
        _mcp_task_queue.clear()
        _mcp_results.clear()


# Initialize the integrated system on module load
_initialize_integrated_system()
