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

    The bus exposes ``/events?offset=N`` which returns ``{"events": [...]}`` batches.
    MonitorClient polls that endpoint in a background thread and publishes each
    event to a local :class:`EventBridge` so the TUI can display them.
    """

    def __init__(self, bus_url: str, run_id: str = ""):
        self._bus_url = bus_url.rstrip("/")
        self._run_id = run_id
        self.bridge = EventBridge()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the background polling thread."""
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the polling thread to stop."""
        self._stop.set()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _poll_loop(self) -> None:
        """Poll the bus ``/events`` endpoint for new events."""
        seen = 0
        while not self._stop.is_set():
            try:
                url = f"{self._bus_url}/events?offset={seen}"
                if self._run_id:
                    url += f"&run_id={self._run_id}"
                resp = urllib.request.urlopen(url, timeout=5)
                data = json.loads(resp.read())
                for event in data.get("events", []):
                    self.bridge.publish(event)
                    seen += 1
            except Exception:
                logger.debug("Monitor poll failed, retrying...", exc_info=True)
            self._stop.wait(timeout=1.0)
