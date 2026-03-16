# tests/test_publisher.py
"""Tests for EventPublisher."""
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from unittest.mock import patch

import pytest

from chaosengineer.metrics.logger import Event
from chaosengineer.metrics.publisher import EventPublisher


class FakeBusHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler that records published events and serves commands."""

    events = []
    commands = []

    def do_POST(self):
        if self.path == "/publish":
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            FakeBusHandler.events.append(body)
            self.send_response(200)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/healthz":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        elif self.path == "/commands":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            cmds = FakeBusHandler.commands
            FakeBusHandler.commands = []
            self.wfile.write(json.dumps(cmds).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress request logging


@pytest.fixture()
def fake_bus():
    """Start a fake bus HTTP server on a random port."""
    FakeBusHandler.events = []
    FakeBusHandler.commands = []
    server = HTTPServer(("127.0.0.1", 0), FakeBusHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


class TestEventPublisher:
    def test_log_posts_to_bus(self, fake_bus, tmp_path):
        pub = EventPublisher(bus_url=fake_bus, fallback_path=tmp_path / "events.jsonl")
        pub.log(Event(event="run_started", data={"run_id": "run-1", "workload": "test"}))

        assert len(FakeBusHandler.events) == 1
        evt = FakeBusHandler.events[0]
        assert evt["event"] == "run_started"
        assert evt["run_id"] == "run-1"
        assert evt["workload"] == "test"
        assert "ts" in evt

    def test_log_includes_run_id_from_run_started(self, fake_bus, tmp_path):
        pub = EventPublisher(bus_url=fake_bus, fallback_path=tmp_path / "events.jsonl")
        pub.log(Event(event="run_started", data={"run_id": "run-abc"}))
        pub.log(Event(event="worker_completed", data={"metric": 2.5}))

        # Second event should include run_id from run_started
        evt = FakeBusHandler.events[1]
        assert evt["run_id"] == "run-abc"

    def test_fallback_when_bus_unavailable(self, tmp_path):
        log_path = tmp_path / "events.jsonl"
        pub = EventPublisher(bus_url=None, fallback_path=log_path)
        pub.log(Event(event="test_event", data={"key": "value"}))

        assert log_path.exists()
        records = [json.loads(line) for line in log_path.read_text().strip().split("\n")]
        assert len(records) == 1
        assert records[0]["event"] == "test_event"

    def test_fallback_on_connection_error(self, tmp_path):
        log_path = tmp_path / "events.jsonl"
        # Point to a port that nothing listens on
        pub = EventPublisher(
            bus_url="http://127.0.0.1:1",
            fallback_path=log_path,
        )
        pub.log(Event(event="test_event", data={"key": "value"}))

        assert log_path.exists()

    def test_poll_commands(self, fake_bus, tmp_path):
        FakeBusHandler.commands = [
            {"command": "pause", "run_id": "run-1"},
        ]
        pub = EventPublisher(bus_url=fake_bus, fallback_path=tmp_path / "events.jsonl")
        cmds = pub.poll_commands()

        assert len(cmds) == 1
        assert cmds[0]["command"] == "pause"

    def test_poll_commands_empty_when_no_bus(self, tmp_path):
        pub = EventPublisher(bus_url=None, fallback_path=tmp_path / "events.jsonl")
        assert pub.poll_commands() == []

    def test_read_events_from_file(self, tmp_path):
        log_path = tmp_path / "events.jsonl"
        log_path.write_text(
            '{"ts":"2026-01-01T00:00:00+00:00","event":"run_started","run_id":"r1"}\n'
            '{"ts":"2026-01-01T00:00:01+00:00","event":"worker_completed","metric":1.0}\n'
        )
        pub = EventPublisher(bus_url=None, fallback_path=log_path)

        all_events = pub.read_events()
        assert len(all_events) == 2

        starts = pub.read_events("run_started")
        assert len(starts) == 1
        assert starts[0]["run_id"] == "r1"

    def test_read_events_empty_when_no_file(self, tmp_path):
        pub = EventPublisher(bus_url=None, fallback_path=tmp_path / "missing.jsonl")
        assert pub.read_events() == []

    def test_raises_when_no_bus_and_no_fallback(self):
        with pytest.raises(RuntimeError):
            EventPublisher(bus_url=None, fallback_path=None)


class TestEventPublisherBridge:
    def test_publishes_to_bridge_when_provided(self, tmp_path):
        from chaosengineer.bus import EventBridge

        bridge = EventBridge()
        publisher = EventPublisher(bus_url=None, fallback_path=tmp_path / "events.jsonl", bridge=bridge)
        publisher.log(Event("run_started", data={"run_id": "r1"}))

        snap = bridge.snapshot()
        assert len(snap) == 1
        assert snap[0]["event"] == "run_started"
        assert snap[0]["run_id"] == "r1"

    def test_bridge_receives_all_events(self, tmp_path):
        from chaosengineer.bus import EventBridge

        bridge = EventBridge()
        publisher = EventPublisher(bus_url=None, fallback_path=tmp_path / "events.jsonl", bridge=bridge)
        publisher.log(Event("run_started", data={"run_id": "r1"}))
        publisher.log(Event("iteration_started", data={"iteration": 0}))
        publisher.log(Event("worker_completed", data={"experiment_id": "e1"}))

        assert len(bridge.snapshot()) == 3

    def test_no_bridge_still_works(self, tmp_path):
        """Without bridge, publisher works exactly as before."""
        publisher = EventPublisher(bus_url=None, fallback_path=tmp_path / "events.jsonl")
        publisher.log(Event("run_started", data={"run_id": "r1"}))
        assert (tmp_path / "events.jsonl").exists()
