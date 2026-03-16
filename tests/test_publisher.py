# tests/test_publisher.py
"""Tests for EventPublisher."""
import json

import pytest

from chaosengineer.bus import EventBridge
from chaosengineer.metrics.logger import Event
from chaosengineer.metrics.publisher import EventPublisher


class TestEventPublisher:
    def test_log_writes_to_file(self, tmp_path):
        log_path = tmp_path / "events.jsonl"
        pub = EventPublisher(path=log_path)
        pub.log(Event(event="run_started", data={"run_id": "run-1", "workload": "test"}))

        records = [json.loads(line) for line in log_path.read_text().strip().split("\n")]
        assert len(records) == 1
        assert records[0]["event"] == "run_started"
        assert records[0]["run_id"] == "run-1"

    def test_log_publishes_to_bridge(self, tmp_path):
        bridge = EventBridge()
        pub = EventPublisher(path=tmp_path / "events.jsonl", bridge=bridge)
        pub.log(Event(event="run_started", data={"run_id": "r1"}))

        snap = bridge.snapshot()
        assert len(snap) == 1
        assert snap[0]["event"] == "run_started"
        assert snap[0]["run_id"] == "r1"

    def test_log_includes_run_id_from_run_started(self, tmp_path):
        bridge = EventBridge()
        pub = EventPublisher(path=tmp_path / "events.jsonl", bridge=bridge)
        pub.log(Event(event="run_started", data={"run_id": "run-abc"}))
        pub.log(Event(event="worker_completed", data={"metric": 2.5}))

        snap = bridge.snapshot()
        assert snap[1]["run_id"] == "run-abc"

    def test_bridge_receives_all_events(self, tmp_path):
        bridge = EventBridge()
        pub = EventPublisher(path=tmp_path / "events.jsonl", bridge=bridge)
        pub.log(Event("run_started", data={"run_id": "r1"}))
        pub.log(Event("iteration_started", data={"iteration": 0}))
        pub.log(Event("worker_completed", data={"experiment_id": "e1"}))

        assert len(bridge.snapshot()) == 3

    def test_no_bridge_still_writes_file(self, tmp_path):
        log_path = tmp_path / "events.jsonl"
        pub = EventPublisher(path=log_path)
        pub.log(Event("run_started", data={"run_id": "r1"}))
        assert log_path.exists()

    def test_read_events_from_file(self, tmp_path):
        log_path = tmp_path / "events.jsonl"
        log_path.write_text(
            '{"ts":"2026-01-01T00:00:00+00:00","event":"run_started","run_id":"r1"}\n'
            '{"ts":"2026-01-01T00:00:01+00:00","event":"worker_completed","metric":1.0}\n'
        )
        pub = EventPublisher(path=log_path)

        all_events = pub.read_events()
        assert len(all_events) == 2

        starts = pub.read_events("run_started")
        assert len(starts) == 1
        assert starts[0]["run_id"] == "r1"

    def test_read_events_empty_when_no_file(self, tmp_path):
        pub = EventPublisher(path=tmp_path / "missing.jsonl")
        assert pub.read_events() == []
