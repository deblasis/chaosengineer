"""JSONL event logger for ChaosEngineer runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Event:
    """A single event to log."""
    event: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: str | None = None  # ISO timestamp, auto-generated if None


class EventLogger:
    """Append-only JSONL event logger.

    Events are stored in flat format matching the spec:
    {"ts": "...", "event": "run_started", "workload": "test", ...}
    Event-specific fields are merged at the top level (no 'data' envelope).
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def log(self, event: Event) -> None:
        ts = event.ts or datetime.now(timezone.utc).isoformat()
        record = {"ts": ts, "event": event.event, **event.data}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def read_events(self, event_type: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        events = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if event_type is None or record.get("event") == event_type:
                    events.append(record)
        return events
