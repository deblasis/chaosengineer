"""State-snapshot tests for the ChaosEngineer TUI app.

Each test renders the app, feeds a sequence of events, then verifies the
complete rendered state of every widget — budget bar, experiment table,
event log, and command bar.
"""
from unittest.mock import MagicMock

import pytest

from chaosengineer.tui.app import BudgetBar, ChaosApp
from chaosengineer.bus import EventBridge
from chaosengineer.tui.pause_gate import PauseGate
from textual.widgets import DataTable, RichLog, Static


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _table_rows(table: DataTable) -> list[list[str]]:
    """Extract all table rows as lists of stringified cell values."""
    rows = []
    for row_key in table.rows:
        row = table.get_row(row_key)
        rows.append([str(cell) for cell in row])
    return rows


def _log_text(log: RichLog) -> str:
    """Get all log content as a single plain-text string."""
    parts = []
    for line in log.lines:
        # Textual RichLog lines are Strip objects with a .text property
        if hasattr(line, "text"):
            parts.append(line.text)
        else:
            parts.append(str(line))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 1. Initial state — app just mounted, nothing processed
# ---------------------------------------------------------------------------

class TestSnapshotInitialState:
    """Verify the app's complete widget state immediately after mount."""

    async def test_budget_bar_shows_defaults(self, app):
        async with app.run_test(size=(120, 40)):
            bar = app.query_one("#budget-bar", BudgetBar)
            # Before any budget_checkpoint, the bar should not have been
            # rendered with values yet — its renderable is the initial Static
            # content (empty string from Static default).
            assert bar._cost == 0.0
            assert bar._max_cost is None
            assert bar._experiments == 0
            assert bar._max_experiments is None
            assert bar._elapsed == "00:00:00"

    async def test_experiment_table_empty_with_columns(self, app):
        async with app.run_test(size=(120, 40)):
            table = app.query_one("#experiment-table", DataTable)
            # Table has 6 columns: #, Worker, Dimension, Status, Cost, Delta
            assert len(table.columns) == 6
            column_labels = [str(col.label) for col in table.columns.values()]
            assert column_labels == ["#", "Worker", "Dimension", "Status", "Cost", "Delta"]
            # No rows yet
            assert table.row_count == 0

    async def test_event_log_empty(self, app):
        async with app.run_test(size=(120, 40)):
            log = app.query_one("#event-log", RichLog)
            assert len(log.lines) == 0

    async def test_command_bar_shows_keybindings(self, app):
        async with app.run_test(size=(120, 40)):
            footer = app.query_one("#command-bar", Static)
            text = str(footer.content)
            assert "[P]ause" in text
            assert "[E]xtend budget" in text
            assert "[Q]uit TUI" in text


# ---------------------------------------------------------------------------
# 2. Mid-run — iteration started, one worker completed, budget updated
# ---------------------------------------------------------------------------

class TestSnapshotMidRun:
    """Verify state after a realistic sequence: iteration -> worker done -> budget."""

    @pytest.fixture
    def mid_run_events(self):
        return [
            {
                "ts": "2026-03-16T14:00:00Z",
                "event": "iteration_started",
                "dimension": "lr",
                "num_workers": 2,
                "iteration": 0,
                "tasks": [
                    {"experiment_id": "exp-0-0", "params": {"lr": 0.01}},
                    {"experiment_id": "exp-0-1", "params": {"lr": 0.1}},
                ],
            },
            {
                "ts": "2026-03-16T14:01:30Z",
                "event": "worker_completed",
                "experiment_id": "exp-0-0",
                "dimension": "lr",
                "metric": 0.85,
                "cost_usd": 0.42,
            },
            {
                "ts": "2026-03-16T14:01:45Z",
                "event": "budget_checkpoint",
                "spent_usd": 0.42,
                "remaining_cost": 9.58,
                "experiments_run": 1,
                "remaining_experiments": 9,
                "elapsed_seconds": 105,
            },
        ]

    async def test_table_has_two_rows(self, app, mid_run_events):
        async with app.run_test(size=(120, 40)):
            for ev in mid_run_events:
                app._handle_event(ev)

            table = app.query_one("#experiment-table", DataTable)
            assert table.row_count == 2

    async def test_completed_worker_shows_done(self, app, mid_run_events):
        async with app.run_test(size=(120, 40)):
            for ev in mid_run_events:
                app._handle_event(ev)

            table = app.query_one("#experiment-table", DataTable)
            row0 = table.get_row("exp-0-0")
            row0_strs = [str(c) for c in row0]
            assert "done" in row0_strs
            assert "$0.42" in row0_strs
            assert "0.85" in row0_strs

    async def test_running_worker_still_running(self, app, mid_run_events):
        async with app.run_test(size=(120, 40)):
            for ev in mid_run_events:
                app._handle_event(ev)

            table = app.query_one("#experiment-table", DataTable)
            row1 = table.get_row("exp-0-1")
            row1_strs = [str(c) for c in row1]
            assert "running" in row1_strs

    async def test_budget_bar_updated(self, app, mid_run_events):
        async with app.run_test(size=(120, 40)):
            for ev in mid_run_events:
                app._handle_event(ev)

            bar = app.query_one("#budget-bar", BudgetBar)
            assert bar._cost == pytest.approx(0.42)
            assert bar._max_cost == pytest.approx(10.0)
            assert bar._experiments == 1
            assert bar._max_experiments == 10
            assert bar._elapsed == "00:01:45"

    async def test_log_has_three_entries(self, app, mid_run_events):
        async with app.run_test(size=(120, 40)):
            for ev in mid_run_events:
                app._handle_event(ev)

            log = app.query_one("#event-log", RichLog)
            # One log line per event
            assert len(log.lines) == 3

    async def test_log_content_reflects_events(self, app, mid_run_events):
        async with app.run_test(size=(120, 40)):
            for ev in mid_run_events:
                app._handle_event(ev)

            log_text = _log_text(app.query_one("#event-log", RichLog))
            assert "iteration_started" in log_text
            assert "worker_completed" in log_text
            assert "budget_checkpoint" in log_text

    async def test_full_table_snapshot(self, app, mid_run_events):
        """Verify the exact cell values for every row in the table."""
        async with app.run_test(size=(120, 40)):
            for ev in mid_run_events:
                app._handle_event(ev)

            table = app.query_one("#experiment-table", DataTable)
            rows = _table_rows(table)
            # Row 0: exp-0-0, completed
            assert rows[0] == ["exp-0-0", "W1", "lr", "done", "$0.42", "0.85"]
            # Row 1: exp-0-1, still running
            assert rows[1] == ["exp-0-1", "W2", "lr", "running", "-", "-"]


# ---------------------------------------------------------------------------
# 3. Pause modal — after a pause_decision_needed event
# ---------------------------------------------------------------------------

class TestSnapshotPauseModal:
    """Verify state after a pause_decision_needed event."""

    async def test_log_shows_pause_notification(self, app):
        async with app.run_test(size=(120, 40)):
            app._handle_event({
                "ts": "2026-03-16T14:05:00Z",
                "event": "pause_decision_needed",
                "reason": "budget_threshold",
            })

            log_text = _log_text(app.query_one("#event-log", RichLog))
            assert "PAUSE REQUESTED" in log_text

    async def test_log_shows_user_instructions(self, app):
        async with app.run_test(size=(120, 40)):
            app._handle_event({
                "ts": "2026-03-16T14:05:00Z",
                "event": "pause_decision_needed",
                "reason": "budget_threshold",
            })

            log_text = _log_text(app.query_one("#event-log", RichLog))
            # Instructions should reference the P key
            assert "P" in log_text

    async def test_pause_log_entry_count(self, app):
        async with app.run_test(size=(120, 40)):
            app._handle_event({
                "ts": "2026-03-16T14:05:00Z",
                "event": "pause_decision_needed",
                "reason": "budget_threshold",
            })

            log = app.query_one("#event-log", RichLog)
            # Two lines: the generic event log + the PAUSE REQUESTED banner
            assert len(log.lines) == 2

    async def test_table_unaffected_by_pause(self, app):
        async with app.run_test(size=(120, 40)):
            app._handle_event({
                "ts": "2026-03-16T14:05:00Z",
                "event": "pause_decision_needed",
                "reason": "budget_threshold",
            })

            table = app.query_one("#experiment-table", DataTable)
            assert table.row_count == 0

    async def test_budget_bar_unaffected_by_pause(self, app):
        async with app.run_test(size=(120, 40)):
            app._handle_event({
                "ts": "2026-03-16T14:05:00Z",
                "event": "pause_decision_needed",
                "reason": "budget_threshold",
            })

            bar = app.query_one("#budget-bar", BudgetBar)
            assert bar._cost == 0.0
            assert bar._experiments == 0


# ---------------------------------------------------------------------------
# 4. Run complete — after run_completed with specific values
# ---------------------------------------------------------------------------

class TestSnapshotRunComplete:
    """Verify state after run_completed event."""

    @pytest.fixture
    def run_completed_event(self):
        return {
            "ts": "2026-03-16T15:00:00Z",
            "event": "run_completed",
            "best_metric": 0.95,
            "total_experiments": 10,
            "total_cost_usd": 4.20,
        }

    async def test_log_shows_completion_banner(self, app, run_completed_event):
        async with app.run_test(size=(120, 40)):
            app._handle_event(run_completed_event)

            log_text = _log_text(app.query_one("#event-log", RichLog))
            assert "RUN COMPLETE" in log_text

    async def test_log_shows_best_metric(self, app, run_completed_event):
        async with app.run_test(size=(120, 40)):
            app._handle_event(run_completed_event)

            log_text = _log_text(app.query_one("#event-log", RichLog))
            assert "0.95" in log_text

    async def test_log_shows_experiment_count(self, app, run_completed_event):
        async with app.run_test(size=(120, 40)):
            app._handle_event(run_completed_event)

            log_text = _log_text(app.query_one("#event-log", RichLog))
            assert "10" in log_text

    async def test_log_shows_total_cost(self, app, run_completed_event):
        async with app.run_test(size=(120, 40)):
            app._handle_event(run_completed_event)

            log_text = _log_text(app.query_one("#event-log", RichLog))
            assert "$4.20" in log_text

    async def test_log_entry_count(self, app, run_completed_event):
        async with app.run_test(size=(120, 40)):
            app._handle_event(run_completed_event)

            log = app.query_one("#event-log", RichLog)
            # Two lines: the generic event line + the RUN COMPLETE banner
            assert len(log.lines) == 2

    async def test_command_bar_unchanged(self, app, run_completed_event):
        async with app.run_test(size=(120, 40)):
            app._handle_event(run_completed_event)

            footer = app.query_one("#command-bar", Static)
            text = str(footer.content)
            assert "[P]ause" in text
            assert "[Q]uit TUI" in text
