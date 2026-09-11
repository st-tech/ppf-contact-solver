"""MCP Task Processing System for thread-safe Blender operations with integrated decorator support."""

import threading
import time

from .integration import get_integrated_handlers, initialize_integrated_system

_get_integrated_handlers = get_integrated_handlers
_initialize_integrated_system = initialize_integrated_system


# MCP Task Processing System (keep existing task system)
_mcp_task_queue = []
_mcp_results = {}
_mcp_result_times = {}
_mcp_lock = threading.Lock()
_mcp_task_id_counter = 0

# Tasks whose reader has gone away, mapped to when the cancellation was
# recorded. Closing the response stream is the transport's cancellation
# signal, and a cancelled task must not leave a result behind for nobody.
#
# An entry is only needed while the task can still finish and store a result,
# so it is dropped as soon as that happens, and reaped on the same schedule as
# an abandoned result in case it never does.
_cancelled_tasks: dict = {}

# Reap results that were never collected (client disconnected or the
# get_mcp_result timeout already fired before the handler finished). The
# default get_mcp_result timeout is 5.0s, so 2x that is comfortably past
# the point any live reader would still be waiting, and a waiting reader
# always consumes its own entry within its timeout window first.
_RESULT_REAP_SECONDS = 10.0


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


def try_get_mcp_result(task_id):
    """Take the result of *task_id* if it is ready.

    Returns ``(True, result)`` once, then ``(False, None)`` forever after, so
    a caller that polls can tell "not finished yet" from "finished and already
    collected". This is what lets a transport wait on a long tool call without
    committing to a fixed deadline up front.
    """
    with _mcp_lock:
        if task_id in _mcp_results:
            result = _mcp_results.pop(task_id)
            _mcp_result_times.pop(task_id, None)
            return True, result
    return False, None


def cancel_mcp_task(task_id):
    """Drop *task_id*, whether it is queued, running, or already finished.

    A queued task is removed before it runs; a task already in flight is
    allowed to finish on the main thread, but its result is discarded rather
    than left for a reader that has gone away.
    """
    with _mcp_lock:
        removed_from_queue = False
        for index, task in enumerate(_mcp_task_queue):
            if task["id"] == task_id:
                del _mcp_task_queue[index]
                removed_from_queue = True
                break
        _mcp_results.pop(task_id, None)
        _mcp_result_times.pop(task_id, None)
        # A task taken off the queue never runs, so there is no later result
        # to suppress and nothing to remember. Recording it anyway would grow
        # the map by one entry for every cancelled call, forever.
        if not removed_from_queue:
            _cancelled_tasks[task_id] = time.time()


def get_mcp_result(task_id, timeout=5.0):
    """Wait up to *timeout* seconds for the result of a posted task.

    Suits a caller that knows the work is short. A caller that cannot bound
    the work should poll ``try_get_mcp_result`` instead, so a slow task is
    reported as still running rather than as failed.
    """
    start_time = time.time()

    while time.time() - start_time < timeout:
        found, result = try_get_mcp_result(task_id)
        if found:
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
        cancelled = set(_cancelled_tasks)
    tasks_to_process = [t for t in tasks_to_process if t["id"] not in cancelled]

    # Process each task
    for task in tasks_to_process:
        task_id = task["id"]
        task_type = task["type"]
        args = task["args"]

        try:
            result = _execute_blender_task(task_type, args)
        except Exception as e:
            result = {"status": "error", "message": str(e)}

        # Store result, unless the reader has gone away.
        with _mcp_lock:
            if task_id in _cancelled_tasks:
                del _cancelled_tasks[task_id]
                continue
            _mcp_results[task_id] = result
            _mcp_result_times[task_id] = time.time()

    # Reap any results abandoned by a reader that already timed out or
    # disconnected, so _mcp_results does not grow without bound over a long
    # session. A live get_mcp_result deletes its own entry well within its
    # timeout window, so this only evicts entries no reader will ever read.
    now = time.time()
    with _mcp_lock:
        stale_ids = [
            tid
            for tid, stored_at in _mcp_result_times.items()
            if now - stored_at > _RESULT_REAP_SECONDS
        ]
        for tid in stale_ids:
            _mcp_results.pop(tid, None)
            del _mcp_result_times[tid]
        # A cancellation whose task never came back is reaped on the same
        # schedule, so the map cannot grow without bound over a long session.
        for tid in [
            tid
            for tid, marked_at in _cancelled_tasks.items()
            if now - marked_at > _RESULT_REAP_SECONDS
        ]:
            del _cancelled_tasks[tid]


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
        _mcp_result_times.clear()
        _cancelled_tasks.clear()


# Initialize the integrated system on module load
_initialize_integrated_system()
