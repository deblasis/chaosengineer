# chaosengineer/metrics/publisher.py
"""Event publisher — writes to file and optionally to an in-process EventBridge."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chaosengineer.metrics.logger import Event, EventLogger


class EventPublisher:
    """Publishes events to file (always) and to an EventBridge (if provided).

    Same log(Event) interface as EventLogger — the coordinator doesn't change.
    """

    def __init__(self, path: Path, bridge: "EventBridge | None" = None):
        self._path = path
        self._logger = EventLogger(path)
        self._bridge = bridge
        self._run_id: str | None = None

    def log(self, event: Event) -> None:
        """Write event to file and publish to bridge."""
        ts = event.ts or datetime.now(timezone.utc).isoformat()
        payload: dict[str, Any] = {"ts": ts, "event": event.event, **event.data}

        # Track run_id from run_started events
        if event.event == "run_started" and "run_id" in event.data:
            self._run_id = event.data["run_id"]

        # Include run_id on all events if known
        if "run_id" not in payload and self._run_id:
            payload["run_id"] = self._run_id

        self._logger.log(event)

        if self._bridge is not None:
            self._bridge.publish(payload)

    def read_events(self, event_type: str | None = None) -> list[dict]:
        """Read events from the JSONL file."""
        if not self._path.exists():
            return []
        events = []
        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if event_type is None or record.get("event") == event_type:
                    events.append(record)
        return events
