"""Tests for the detached monitor mode."""
from __future__ import annotations

import io
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from chaosengineer.tui.app import ChaosApp
from chaosengineer.tui.bridge import EventBridge
from chaosengineer.tui.monitor import MonitorClient
from chaosengineer.tui.pause_gate import PauseGate
from textual.widgets import RichLog


# ---------------------------------------------------------------------------
# Helpers: fake streaming HTTP response
# ---------------------------------------------------------------------------

class FakeStreamResponse:
    """Simulates a streaming HTTP response with newline-delimited JSON."""

    def __init__(self, events):
        lines = [json.dumps(e) + "\n" for e in events]
        self._data = "".join(lines).encode()
        self._stream = io.BytesIO(self._data)
        self.status = 200

    def read(self, amt=-1):
        return self._stream.read(amt)

    def readline(self):
        return self._stream.readline()

    def __iter__(self):
        return iter(self._stream)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# MonitorClient streaming tests
# ---------------------------------------------------------------------------

class TestMonitorClientStreaming:
    def test_streams_events_to_bridge(self):
        """MonitorClient should read streaming /events and publish to bridge."""
        events = [
            {"event": "run_started", "run_id": "r1"},
            {"event": "worker_completed", "run_id": "r1", "metric": 0.5},
        ]
        fake_resp = FakeStreamResponse(events)

        with patch("urllib.request.urlopen", return_value=fake_resp):
            client = MonitorClient(bus_url="http://fake:1234")
            client.start()
            time.sleep(0.3)
            client.stop()

        snapshot = client.bridge.snapshot()
        assert len(snapshot) == 2
        assert snapshot[0]["event"] == "run_started"
        assert snapshot[1]["event"] == "worker_completed"

    def test_reconnects_on_error(self):
        """MonitorClient should reconnect after stream errors."""
        call_count = 0
        events = [{"event": "run_started", "run_id": "r1"}]

        def fake_urlopen(req, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("refused")
            return FakeStreamResponse(events)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client = MonitorClient(bus_url="http://fake:1234")
            client.start()
            time.sleep(1.5)  # enough for retry
            client.stop()

        assert call_count >= 2

    def test_run_id_filter(self):
        """MonitorClient passes run_id to the URL."""
        captured_urls = []

        def capture_urlopen(req, **kwargs):
            url = req.full_url if hasattr(req, 'full_url') else str(req)
            captured_urls.append(url)
            raise ConnectionError("stop")

        with patch("urllib.request.urlopen", side_effect=capture_urlopen):
            client = MonitorClient(bus_url="http://fake:1234", run_id="run-abc")
            client.start()
            time.sleep(0.3)
            client.stop()

        assert any("run_id=run-abc" in u for u in captured_urls)


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
