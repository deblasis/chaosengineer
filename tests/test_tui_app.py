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


class TestCollapsibleIterations:
    """Tests for collapsible past-iteration groups in the DataTable."""

    def _start_iteration(self, app, iteration, tasks, dimension="lr"):
        """Helper: fire an iteration_started event."""
        app._handle_event({
            "ts": "2026-03-16T14:00:00Z",
            "event": "iteration_started",
            "dimension": dimension,
            "num_workers": len(tasks),
            "iteration": iteration,
            "tasks": tasks,
        })

    def _complete_worker(self, app, exp_id, metric=0.85, cost=0.42):
        """Helper: fire a worker_completed event."""
        app._handle_event({
            "event": "worker_completed",
            "experiment_id": exp_id,
            "metric": metric,
            "cost_usd": cost,
        })

    def _fail_worker(self, app, exp_id):
        """Helper: fire a worker_failed event."""
        app._handle_event({
            "event": "worker_failed",
            "experiment_id": exp_id,
        })

    async def test_new_iteration_collapses_previous(self, app):
        """Starting iteration 1 should collapse iteration 0 into a summary row."""
        async with app.run_test() as pilot:
            self._start_iteration(app, 0, [
                {"experiment_id": "exp-0-0"},
                {"experiment_id": "exp-0-1"},
            ])
            self._complete_worker(app, "exp-0-0", metric=0.85, cost=0.42)
            self._complete_worker(app, "exp-0-1", metric=0.90, cost=0.38)
            table = app.query_one("#experiment-table", DataTable)
            assert table.row_count == 2

            # Start iteration 1 — should collapse iteration 0
            self._start_iteration(app, 1, [
                {"experiment_id": "exp-1-0"},
            ])
            # iteration 0: 2 rows collapsed into 1 summary; iteration 1: 1 row
            assert table.row_count == 2
            assert 0 in app._collapsed

    async def test_summary_row_has_correct_aggregated_stats(self, app):
        """The summary row should show correct count, status, cost, and best metric."""
        async with app.run_test() as pilot:
            self._start_iteration(app, 0, [
                {"experiment_id": "exp-0-0"},
                {"experiment_id": "exp-0-1"},
                {"experiment_id": "exp-0-2"},
            ], dimension="batch_size")
            self._complete_worker(app, "exp-0-0", metric=0.80, cost=1.00)
            self._complete_worker(app, "exp-0-1", metric=0.95, cost=2.00)
            self._complete_worker(app, "exp-0-2", metric=0.70, cost=0.50)

            # Start next iteration to trigger collapse
            self._start_iteration(app, 1, [{"experiment_id": "exp-1-0"}])

            table = app.query_one("#experiment-table", DataTable)
            summary_row = table.get_row("iter-0")
            # summary_row is a tuple: (#, Worker, Dimension, Status, Cost, Delta)
            assert summary_row[0] == "iter-0"
            assert "3 experiments" in summary_row[1]
            assert summary_row[2] == "batch_size"
            assert summary_row[3] == "done"
            assert summary_row[4] == "$3.50"
            assert summary_row[5] == "0.95"

    async def test_summary_shows_mixed_status_on_failure(self, app):
        """Summary should show 'mixed' status when any worker failed."""
        async with app.run_test() as pilot:
            self._start_iteration(app, 0, [
                {"experiment_id": "exp-0-0"},
                {"experiment_id": "exp-0-1"},
            ])
            self._complete_worker(app, "exp-0-0", metric=0.85, cost=0.42)
            self._fail_worker(app, "exp-0-1")

            self._start_iteration(app, 1, [{"experiment_id": "exp-1-0"}])

            table = app.query_one("#experiment-table", DataTable)
            summary_row = table.get_row("iter-0")
            assert summary_row[3] == "mixed"

    async def test_expand_collapse_toggle(self, app):
        """Pressing X on a summary row should expand it, pressing again should collapse."""
        async with app.run_test() as pilot:
            self._start_iteration(app, 0, [
                {"experiment_id": "exp-0-0"},
                {"experiment_id": "exp-0-1"},
            ])
            self._complete_worker(app, "exp-0-0", metric=0.85, cost=0.42)
            self._complete_worker(app, "exp-0-1", metric=0.90, cost=0.38)

            # Start iteration 1 to collapse iteration 0
            self._start_iteration(app, 1, [{"experiment_id": "exp-1-0"}])
            table = app.query_one("#experiment-table", DataTable)
            assert table.row_count == 2  # 1 summary + 1 current
            assert 0 in app._collapsed

            # Move cursor to the summary row (row 0) and press X to expand
            table.cursor_coordinate = table.cursor_coordinate._replace(row=0)
            app.action_toggle_expand()
            # Now iteration 0 is expanded: 2 individual rows + 1 current row
            assert table.row_count == 3
            assert 0 not in app._collapsed

            # Verify individual rows are back
            row_0 = table.get_row("exp-0-0")
            assert "done" in str(row_0)

            # Move cursor to an expanded past-iteration row and press X to re-collapse
            # After expand, rows are: exp-1-0 (current), exp-0-0, exp-0-1
            # So expanded rows start at index 1
            table.cursor_coordinate = table.cursor_coordinate._replace(row=1)
            app.action_toggle_expand()
            assert table.row_count == 2
            assert 0 in app._collapsed

    async def test_current_iteration_not_collapsed(self, app):
        """The current iteration should never be collapsed."""
        async with app.run_test() as pilot:
            self._start_iteration(app, 0, [
                {"experiment_id": "exp-0-0"},
                {"experiment_id": "exp-0-1"},
            ])
            table = app.query_one("#experiment-table", DataTable)
            assert table.row_count == 2
            assert 0 not in app._collapsed
            assert app._current_iteration == 0

    async def test_multiple_iterations_collapse(self, app):
        """Multiple past iterations should each get their own summary row."""
        async with app.run_test() as pilot:
            # Iteration 0
            self._start_iteration(app, 0, [
                {"experiment_id": "exp-0-0"},
            ])
            self._complete_worker(app, "exp-0-0", metric=0.80, cost=1.00)

            # Iteration 1
            self._start_iteration(app, 1, [
                {"experiment_id": "exp-1-0"},
            ])
            self._complete_worker(app, "exp-1-0", metric=0.90, cost=1.50)

            # Iteration 2 — collapses iteration 1 (iteration 0 already collapsed)
            self._start_iteration(app, 2, [
                {"experiment_id": "exp-2-0"},
            ])

            table = app.query_one("#experiment-table", DataTable)
            # 2 summary rows + 1 current row
            assert table.row_count == 3
            assert 0 in app._collapsed
            assert 1 in app._collapsed

    async def test_worker_data_tracked_in_iteration_data(self, app):
        """Worker completion data should be tracked in _iteration_data."""
        async with app.run_test() as pilot:
            self._start_iteration(app, 0, [
                {"experiment_id": "exp-0-0"},
            ])
            self._complete_worker(app, "exp-0-0", metric=0.85, cost=0.42)

            data = app._iteration_data[0]["exp-0-0"]
            assert data["status"] == "done"
            assert data["cost"] == 0.42
            assert data["metric"] == 0.85
