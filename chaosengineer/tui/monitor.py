"""Monitor client -- connects to chaos-bus and feeds events to local EventBridge."""
from __future__ import annotations

import json
import logging
import threading
import urllib.request

from chaosengineer.tui.bridge import EventBridge

logger = logging.getLogger(__name__)


class MonitorClient:
    """Connects to a remote chaos-bus HTTP endpoint and streams events into an EventBridge.

    The bus exposes ``/events`` as a streaming NDJSON endpoint.
    MonitorClient reads the stream in a background thread and publishes each
    event to a local :class:`EventBridge` so the TUI can display them.
    """

    def __init__(self, bus_url: str, run_id: str = ""):
        self._bus_url = bus_url.rstrip("/")
        self._run_id = run_id
        self.bridge = EventBridge()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background streaming thread."""
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the streaming thread to stop."""
        self._stop.set()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _stream_loop(self) -> None:
        """Read streaming NDJSON from the bus ``/events`` endpoint."""
        while not self._stop.is_set():
            try:
                url = f"{self._bus_url}/events"
                if self._run_id:
                    url += f"?run_id={self._run_id}"
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    for line_bytes in resp:
                        if self._stop.is_set():
                            break
                        line = line_bytes.decode("utf-8").strip()
                        if not line:
                            continue
                        event = json.loads(line)
                        self.bridge.publish(event)
            except Exception:
                logger.debug("Monitor stream failed, reconnecting...", exc_info=True)
            self._stop.wait(timeout=1.0)
