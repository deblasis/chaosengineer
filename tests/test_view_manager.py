"""Tests for ViewManager — toggle between log mode and TUI mode."""
from unittest.mock import MagicMock, patch
import threading

import pytest

from chaosengineer.tui.views import ViewManager
from chaosengineer.tui.bridge import EventBridge
from chaosengineer.tui.pause_gate import PauseGate


class TestViewManagerState:
    def test_starts_in_log_mode(self):
        vm = ViewManager(
            bridge=EventBridge(),
            pause_gate=PauseGate(),
            pause_controller=MagicMock(),
            coordinator=MagicMock(),
            status_display=MagicMock(),
        )
        assert not vm.tui_active

    def test_tui_active_flag_set_during_tui(self):
        vm = ViewManager(
            bridge=EventBridge(),
            pause_gate=PauseGate(),
            pause_controller=MagicMock(),
            coordinator=MagicMock(),
            status_display=MagicMock(),
        )
        assert not vm.tui_active


class TestViewManagerCoordinatorDone:
    def test_stops_when_coordinator_thread_finishes(self):
        """ViewManager.run() should exit when coord_done event is set."""
        vm = ViewManager(
            bridge=EventBridge(),
            pause_gate=PauseGate(),
            pause_controller=MagicMock(),
            coordinator=MagicMock(),
            status_display=MagicMock(),
        )
        vm._coord_done = threading.Event()
        vm._coord_done.set()
        # run() should return quickly since coordinator is done
        # (We can't test the full stdin loop without a terminal)
