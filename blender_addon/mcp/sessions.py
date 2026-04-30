"""Session store for the Streamable HTTP MCP transport.

Each session owns a queue of server-to-client SSE events plus a small ring
buffer so a reconnecting client can resume from Last-Event-ID.
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
        self._next_event_id = 0
        self._recent: list = []
        self._recent_max = 256
        self._lock = threading.Lock()
        self.closed = False

    def emit(self, payload: str) -> str:
        with self._lock:
            event_id = str(self._next_event_id)
            self._next_event_id += 1
            self._recent.append((event_id, payload))
            if len(self._recent) > self._recent_max:
                self._recent = self._recent[-self._recent_max:]
        self.event_queue.put((event_id, payload))
        return event_id

    def replay_after(self, last_event_id: str):
        with self._lock:
            snapshot = list(self._recent)
        seen = False
        for eid, payload in snapshot:
            if seen:
                yield eid, payload
            elif eid == last_event_id:
                seen = True

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
