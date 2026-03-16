"""Tests for the detached monitor mode."""
from __future__ import annotations

import http.server
import json
import threading
import time
from unittest.mock import MagicMock

import pytest

from chaosengineer.tui.app import ChaosApp
from chaosengineer.tui.bridge import EventBridge
from chaosengineer.tui.monitor import MonitorClient
from chaosengineer.tui.pause_gate import PauseGate
from textual.widgets import RichLog


# ---------------------------------------------------------------------------
# Helpers: tiny HTTP server that mimics the bus /events endpoint
# ---------------------------------------------------------------------------

class _FakeBusHandler(http.server.BaseHTTPRequestHandler):
    """Serves canned events from ``self.server.events``."""

    def do_GET(self):
        if self.path.startswith("/events"):
            # Parse offset from query string
            offset = 0
            if "?" in self.path:
                qs = self.path.split("?", 1)[1]
                for param in qs.split("&"):
                    if param.startswith("offset="):
                        offset = int(param.split("=")[1])

            events = self.server.events[offset:]  # type: ignore[attr-defined]
            body = json.dumps({"events": events}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/healthz":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass  # silence request logs during tests


@pytest.fixture()
def fake_bus():
    """Start a fake bus HTTP server and yield its URL + event list."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _FakeBusHandler)
    server.events = []  # type: ignore[attr-defined]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    port = server.server_address[1]
    yield f"http://127.0.0.1:{port}", server.events
    server.shutdown()


# ---------------------------------------------------------------------------
# MonitorClient tests
# ---------------------------------------------------------------------------

class TestMonitorClient:
    def test_publishes_events_to_bridge(self, fake_bus):
        """MonitorClient should forward events from the bus to its bridge."""
        url, events = fake_bus
        events.extend([
            {"event": "run_started", "run_id": "r1", "ts": "2026-01-01T00:00:00Z"},
            {"event": "iteration_started", "dimension": "lr", "iteration": 0,
             "tasks": [], "ts": "2026-01-01T00:00:01Z"},
        ])

        client = MonitorClient(bus_url=url)
        client.start()
        # Give the poll loop time to fetch
        time.sleep(2.5)
        client.stop()

        snap = client.bridge.snapshot()
        assert len(snap) == 2
        assert snap[0]["event"] == "run_started"
        assert snap[1]["event"] == "iteration_started"

    def test_incremental_polling(self, fake_bus):
        """MonitorClient should only fetch new events on subsequent polls."""
        url, events = fake_bus
        events.append({"event": "run_started", "run_id": "r1"})

        client = MonitorClient(bus_url=url)
        client.start()
        time.sleep(2.5)

        # Add more events while client is running
        events.append({"event": "budget_checkpoint", "spent_usd": 1.0})
        time.sleep(2.5)
        client.stop()

        snap = client.bridge.snapshot()
        assert len(snap) == 2

    def test_handles_connection_error_gracefully(self):
        """MonitorClient should not crash if the bus is unreachable."""
        client = MonitorClient(bus_url="http://127.0.0.1:1")  # nothing listening
        client.start()
        time.sleep(2.5)
        client.stop()
        # No exception should propagate; bridge simply stays empty
        assert client.bridge.snapshot() == []

    def test_run_id_filter(self, fake_bus):
        """When run_id is specified, MonitorClient should include it in the URL."""
        url, events = fake_bus
        events.append({"event": "run_started", "run_id": "specific-run"})

        client = MonitorClient(bus_url=url, run_id="specific-run")
        client.start()
        time.sleep(2.5)
        client.stop()

        snap = client.bridge.snapshot()
        assert len(snap) == 1

    def test_stop_terminates_polling(self, fake_bus):
        """After stop(), the polling thread should finish."""
        url, events = fake_bus
        client = MonitorClient(bus_url=url)
        client.start()
        assert client.is_running
        client.stop()
        time.sleep(2.0)
        assert not client.is_running


# ---------------------------------------------------------------------------
# Read-only ChaosApp tests
# ---------------------------------------------------------------------------

@pytest.fixture
def readonly_app():
    bridge = EventBridge()
    gate = PauseGate()
    coordinator = MagicMock()
    pause_controller = MagicMock()
    pause_controller.pause_requested = False
    return ChaosApp(
        bridge=bridge,
        pause_gate=gate,
        coordinator=coordinator,
        pause_controller=pause_controller,
        readonly=True,
    )


@pytest.fixture
def readwrite_app():
    bridge = EventBridge()
    gate = PauseGate()
    coordinator = MagicMock()
    pause_controller = MagicMock()
    pause_controller.pause_requested = False
    return ChaosApp(
        bridge=bridge,
        pause_gate=gate,
        coordinator=coordinator,
        pause_controller=pause_controller,
        readonly=False,
    )


class TestReadonlyMode:
    async def test_readonly_pause_shows_message(self, readonly_app):
        """Pressing P in readonly mode should show read-only message, not pause."""
        async with readonly_app.run_test() as pilot:
            readonly_app.action_pause()
            log = readonly_app.query_one("#event-log", RichLog)
            assert len(log.lines) > 0

    async def test_readonly_extend_shows_message(self, readonly_app):
        """Pressing E in readonly mode should show read-only message, not extend."""
        async with readonly_app.run_test() as pilot:
            readonly_app.action_extend()
            log = readonly_app.query_one("#event-log", RichLog)
            assert len(log.lines) > 0
            # Coordinator's extend_budget should NOT be called
            readonly_app._coordinator.extend_budget.assert_not_called()

    async def test_readonly_does_not_call_pause_controller(self, readonly_app):
        """In readonly mode, action_pause should not touch pause_controller."""
        async with readonly_app.run_test() as pilot:
            readonly_app.action_pause()
            assert readonly_app._pause_controller.pause_requested is False

    async def test_readonly_command_bar_shows_quit_only(self, readonly_app):
        """Readonly command bar should show only Quit."""
        async with readonly_app.run_test() as pilot:
            bar = readonly_app.query_one("#command-bar")
            rendered = str(bar.render())
            assert "read-only" in rendered
            assert "uit" in rendered  # [Q] is consumed by Rich markup

    async def test_readwrite_command_bar_shows_all_actions(self, readwrite_app):
        """Non-readonly command bar should show Pause, Extend, Quit."""
        async with readwrite_app.run_test() as pilot:
            bar = readwrite_app.query_one("#command-bar")
            rendered = str(bar.render())
            assert "ause" in rendered
            assert "xtend" in rendered
            assert "uit" in rendered

    async def test_readonly_quit_works(self, readonly_app):
        """Quit should work in readonly mode without touching pause_gate."""
        async with readonly_app.run_test() as pilot:
            # Should not raise
            readonly_app.action_quit_tui()
