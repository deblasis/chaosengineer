"""Tests for JSONL event logger."""

import json
import time
from pathlib import Path

import pytest
from chaosengineer.metrics.logger import EventLogger, Event


class TestEventLogger:
    def test_log_creates_file(self, tmp_output_dir):
        logger = EventLogger(tmp_output_dir / "events.jsonl")
        logger.log(Event(event="run_started", data={"workload": "test"}))
        assert (tmp_output_dir / "events.jsonl").exists()

    def test_log_writes_valid_jsonl(self, tmp_output_dir):
        logger = EventLogger(tmp_output_dir / "events.jsonl")
        logger.log(Event(event="run_started", data={"workload": "test"}))
        logger.log(Event(event="iteration_started", data={"dimension": "lr"}))

        lines = (tmp_output_dir / "events.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

        event1 = json.loads(lines[0])
        assert event1["event"] == "run_started"
        assert event1["workload"] == "test"  # flat format, no 'data' envelope
        assert "ts" in event1

        event2 = json.loads(lines[1])
        assert event2["event"] == "iteration_started"

    def test_read_events(self, tmp_output_dir):
        logger = EventLogger(tmp_output_dir / "events.jsonl")
        logger.log(Event(event="a", data={}))
        logger.log(Event(event="b", data={"x": 1}))

        events = logger.read_events()
        assert len(events) == 2
        assert events[0]["event"] == "a"
        assert events[1]["x"] == 1  # flat format

    def test_read_events_filtered(self, tmp_output_dir):
        logger = EventLogger(tmp_output_dir / "events.jsonl")
        logger.log(Event(event="run_started", data={}))
        logger.log(Event(event="worker_completed", data={"w": 1}))
        logger.log(Event(event="worker_completed", data={"w": 2}))

        events = logger.read_events(event_type="worker_completed")
        assert len(events) == 2
        assert all(e["event"] == "worker_completed" for e in events)

    def test_empty_file_read(self, tmp_output_dir):
        logger = EventLogger(tmp_output_dir / "events.jsonl")
        events = logger.read_events()
        assert events == []

    def test_timestamp_is_iso(self, tmp_output_dir):
        logger = EventLogger(tmp_output_dir / "events.jsonl")
        logger.log(Event(event="test", data={}))

        events = logger.read_events()
        ts = events[0]["ts"]
        # Should be parseable as ISO format
        from datetime import datetime
        datetime.fromisoformat(ts)
