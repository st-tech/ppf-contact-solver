"""Session store for the Streamable HTTP MCP transport.

The server answers every JSON-RPC request synchronously over the POST
response, so there is no server-initiated push. Each session keeps a queue
used only to deliver a None sentinel from close(), which unblocks the GET
SSE stream's blocking get() so it can shut down promptly.
"""

import queue
import threading
import time
import uuid


class Session:
    def __init__(self, session_id: str):
        self.id = session_id
        self.created_at = time.monotonic()
        self.event_queue: "queue.Queue" = queue.Queue()
        self.closed = False

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.event_queue.put(None)


_sessions: dict = {}
_sessions_lock = threading.Lock()


def create_session() -> Session:
    session_id = uuid.uuid4().hex
    session = Session(session_id)
    with _sessions_lock:
        _sessions[session_id] = session
    return session


def get_session(session_id: str):
    if not session_id:
        return None
    with _sessions_lock:
        return _sessions.get(session_id)


def delete_session(session_id: str) -> bool:
    with _sessions_lock:
        session = _sessions.pop(session_id, None)
    if session is None:
        return False
    session.close()
    return True


def clear_all_sessions() -> None:
    with _sessions_lock:
        sessions = list(_sessions.values())
        _sessions.clear()
    for session in sessions:
        session.close()
