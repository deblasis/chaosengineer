"""Tests for the ChaosEngineer TUI app."""
from unittest.mock import MagicMock

import pytest

from chaosengineer.tui.app import ChaosApp
from chaosengineer.tui.bridge import EventBridge
from chaosengineer.tui.pause_gate import PauseGate
from textual.widgets import DataTable, RichLog


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


class TestChaosAppEventHandling:
    async def test_iteration_started_adds_table_rows(self, app):
        """iteration_started event should add rows to experiment table."""
        async with app.run_test() as pilot:
            app._handle_event({
                "ts": "2026-03-16T14:00:00Z",
                "event": "iteration_started",
                "dimension": "lr",
                "num_workers": 2,
                "iteration": 0,
                "tasks": [
                    {"experiment_id": "exp-0-0", "params": {"lr": 0.01}},
                    {"experiment_id": "exp-0-1", "params": {"lr": 0.1}},
                ],
            })
            table = app.query_one("#experiment-table", DataTable)
            assert table.row_count == 2

    async def test_worker_completed_updates_status(self, app):
        """worker_completed event should update row status."""
        async with app.run_test() as pilot:
            app._handle_event({
                "event": "iteration_started",
                "dimension": "lr",
                "iteration": 0,
                "tasks": [{"experiment_id": "exp-0-0"}],
            })
            app._handle_event({
                "event": "worker_completed",
                "experiment_id": "exp-0-0",
                "metric": 0.85,
                "cost_usd": 0.42,
            })
            table = app.query_one("#experiment-table", DataTable)
            row = table.get_row("exp-0-0")
            assert "done" in str(row)

    async def test_event_log_receives_entries(self, app):
        """Events should appear in the log widget."""
        async with app.run_test() as pilot:
            app._handle_event({
                "ts": "2026-03-16T14:00:00Z",
                "event": "run_started",
                "run_id": "r1",
            })
            log = app.query_one("#event-log", RichLog)
            assert len(log.lines) > 0

    async def test_run_completed_shows_banner(self, app):
        """run_completed event should show completion banner."""
        async with app.run_test() as pilot:
            app._handle_event({
                "event": "run_completed",
                "best_metric": 0.95,
                "total_experiments": 10,
                "total_cost_usd": 4.20,
            })
            log = app.query_one("#event-log", RichLog)
            assert len(log.lines) > 0
