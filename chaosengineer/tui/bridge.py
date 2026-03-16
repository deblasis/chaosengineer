"""EventBridge — in-memory event store connecting coordinator to TUI."""
from __future__ import annotations

import queue
import threading
from collections import deque


class EventBridge:
    """Thread-safe event store with ring buffer (history) and live notification.

    The coordinator thread calls publish(). The TUI subscribes for live events
    and calls snapshot() for replay on toggle.

    Uses queue.Queue (stdlib thread-safe), NOT asyncio.Queue which is not
    thread-safe across event loops.
    """

    def __init__(self, capacity: int = 200):
        self._buffer: deque[dict] = deque(maxlen=capacity)
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()

    def publish(self, event: dict) -> None:
        """Append event to ring buffer and notify all subscribers."""
        with self._lock:
            self._buffer.append(event)
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

    def snapshot(self) -> list[dict]:
        """Return a copy of the ring buffer for replay."""
        with self._lock:
            return list(self._buffer)

    def subscribe(self) -> queue.Queue[dict]:
        """Register a new subscriber. Returns a thread-safe queue for live events."""
        q: queue.Queue[dict] = queue.Queue(maxsize=500)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        """Remove a subscriber."""
        with self._lock:
            self._subscribers.remove(q)
