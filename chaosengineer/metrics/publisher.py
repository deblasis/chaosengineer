# chaosengineer/metrics/publisher.py
"""Event publisher that posts to the message bus with fallback to EventLogger."""
from __future__ import annotations

import json
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chaosengineer.metrics.logger import Event, EventLogger


class EventPublisher:
    """Publishes events to the message bus, falling back to direct file writes.

    Same log(Event) interface as EventLogger — the coordinator doesn't change.
    """

    def __init__(self, bus_url: str | None, fallback_path: Path | None = None, bridge: "EventBridge | None" = None):
        self._bus_url = bus_url
        self._fallback_path = fallback_path
        self._fallback: EventLogger | None = None
        self._bus_available = False
        self._run_id: str | None = None
        self._bridge = bridge

        if bus_url:
            try:
                req = urllib.request.Request(f"{bus_url}/healthz")
                urllib.request.urlopen(req, timeout=2)
                self._bus_available = True
            except Exception:
                pass

        if not self._bus_available:
            if fallback_path:
                self._fallback = EventLogger(fallback_path)
            else:
                raise RuntimeError(
                    "Bus unreachable and no fallback path provided"
                )

    def log(self, event: Event) -> None:
        """Publish an event to the bus, or fall back to direct file write."""
        ts = event.ts or datetime.now(timezone.utc).isoformat()
        payload: dict[str, Any] = {"ts": ts, "event": event.event, **event.data}

        # Track run_id from run_started events
        if event.event == "run_started" and "run_id" in event.data:
            self._run_id = event.data["run_id"]

        # Include run_id on all events if known
        if "run_id" not in payload and self._run_id:
            payload["run_id"] = self._run_id

        if self._bridge is not None:
            self._bridge.publish(payload)

        if not self._bus_available:
            if self._fallback:
                self._fallback.log(event)
            return

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self._bus_url}/publish",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            # Create fallback lazily on first bus failure
            if self._fallback is None and self._fallback_path:
                self._fallback = EventLogger(self._fallback_path)
            if self._fallback:
                self._fallback.log(event)
            else:
                import sys
                print(
                    f"Warning: failed to publish event {event.event}",
                    file=sys.stderr,
                )

    def poll_commands(self) -> list[dict]:
        """Poll the bus for pending commands. Returns empty list on error."""
        if not self._bus_available:
            return []
        try:
            req = urllib.request.Request(f"{self._bus_url}/commands")
            resp = urllib.request.urlopen(req, timeout=2)
            return json.loads(resp.read())
        except Exception:
            return []

    def read_events(self, event_type: str | None = None) -> list[dict]:
        """Read events from the JSONL file (written by bus or fallback).

        Always reads from the file — works whether the bus or EventLogger wrote it.
        """
        if self._fallback_path is None or not self._fallback_path.exists():
            return []
        events = []
        with open(self._fallback_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if event_type is None or record.get("event") == event_type:
                    events.append(record)
        return events
