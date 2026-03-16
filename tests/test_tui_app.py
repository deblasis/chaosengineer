"""Tests for the ChaosEngineer TUI app."""
from unittest.mock import MagicMock

import pytest

from chaosengineer.tui.app import ChaosApp
from chaosengineer.tui.bridge import EventBridge
from chaosengineer.tui.pause_gate import PauseGate


@pytest.fixture
def app():
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
    )


class TestChaosAppMounts:
    async def test_app_has_budget_bar(self, app):
        """App should have a budget bar widget."""
        async with app.run_test() as pilot:
            assert app.query_one("#budget-bar") is not None

    async def test_app_has_experiment_table(self, app):
        """App should have an experiment data table."""
        async with app.run_test() as pilot:
            assert app.query_one("#experiment-table") is not None

    async def test_app_has_event_log(self, app):
        """App should have an event log."""
        async with app.run_test() as pilot:
            assert app.query_one("#event-log") is not None

    async def test_app_has_command_bar(self, app):
        """App should have a footer command bar."""
        async with app.run_test() as pilot:
            footer = app.query_one("#command-bar")
            assert footer is not None
