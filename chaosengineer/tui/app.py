"""ChaosEngineer TUI — Textual application."""
from __future__ import annotations

import queue as _queue_mod
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.widgets import DataTable, Label, RichLog, Static

if TYPE_CHECKING:
    from chaosengineer.core.pause import PauseController
    from chaosengineer.tui.bridge import EventBridge
    from chaosengineer.tui.pause_gate import PauseGate


class BudgetBar(Static):
    """Budget gauges: cost, experiments, time, elapsed clock."""

    DEFAULT_CSS = """
    BudgetBar {
        height: 3;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cost = 0.0
        self._max_cost = None
        self._experiments = 0
        self._max_experiments = None
        self._elapsed = "00:00:00"

    def update_budget(self, cost: float, max_cost: float | None,
                      experiments: int, max_experiments: int | None,
                      elapsed: str) -> None:
        self._cost = cost
        self._max_cost = max_cost
        self._experiments = experiments
        self._max_experiments = max_experiments
        self._elapsed = elapsed
        self._render_content()

    def _render_content(self) -> None:
        cost_str = f"${self._cost:.2f}"
        if self._max_cost is not None:
            cost_str += f"/${self._max_cost:.0f}"

        exp_str = str(self._experiments)
        if self._max_experiments is not None:
            exp_str += f"/{self._max_experiments}"

        self.update(f"Cost: {cost_str}  Experiments: {exp_str}  [{self._elapsed}]")


class ChaosApp(App):
    """Main TUI application for ChaosEngineer."""

    CSS = """
    #budget-bar {
        dock: top;
        height: 3;
        background: $surface;
        border-bottom: solid $primary;
    }
    #experiment-table {
        height: 1fr;
        min-height: 5;
    }
    #event-log {
        height: 1fr;
        min-height: 5;
        border-top: solid $primary;
    }
    #command-bar {
        dock: bottom;
        height: 1;
        background: $surface;
        border-top: solid $primary;
    }
    """

    BINDINGS = [
        ("p", "pause", "Pause"),
        ("e", "extend", "Extend Budget"),
        ("x", "toggle_expand", "Expand/Collapse"),
        ("q", "quit_tui", "Quit TUI"),
        ("escape", "quit_tui", "Quit TUI"),
        ("ctrl+c", "pause", "Pause"),
    ]

    def __init__(self, bridge: "EventBridge", pause_gate: "PauseGate",
                 coordinator, pause_controller: "PauseController"):
        super().__init__()
        self._bridge = bridge
        self._pause_gate = pause_gate
        self._coordinator = coordinator
        self._pause_controller = pause_controller
        self._event_queue: "_queue_mod.Queue | None" = None
        # Iteration tracking for collapsible groups
        self._current_iteration: int | None = None
        self._iteration_rows: dict[int, list[str]] = {}
        self._iteration_data: dict[int, dict[str, dict]] = {}
        self._iteration_summary: dict[int, dict] = {}
        self._collapsed: set[int] = set()

    def compose(self) -> ComposeResult:
        yield BudgetBar(id="budget-bar")
        yield DataTable(id="experiment-table")
        yield RichLog(id="event-log", highlight=True, markup=True)
        yield Static("[P]ause  [E]xtend budget  [X] Expand/Collapse  [Q]uit TUI", id="command-bar")

    def on_mount(self) -> None:
        table = self.query_one("#experiment-table", DataTable)
        table.add_columns(
            ("#", "#"),
            ("Worker", "Worker"),
            ("Dimension", "Dimension"),
            ("Status", "Status"),
            ("Cost", "Cost"),
            ("Delta", "Delta"),
        )

        # Replay history FIRST, then subscribe for live events.
        for event in self._bridge.snapshot():
            self._handle_event(event)

        self._event_queue = self._bridge.subscribe()
        self.set_interval(0.1, self._poll_events)

    def _poll_events(self) -> None:
        """Drain events from the thread-safe queue."""
        if self._event_queue is None:
            return
        while not self._event_queue.empty():
            try:
                event = self._event_queue.get_nowait()
                self._handle_event(event)
            except _queue_mod.Empty:
                break

    def _handle_event(self, event: dict) -> None:
        """Process a single event and update widgets."""
        event_type = event.get("event", "")
        log = self.query_one("#event-log", RichLog)
        ts = event.get("ts", "")
        if isinstance(ts, str) and "T" in ts:
            ts = ts.split("T")[1][:8]

        log.write(f"{ts} {event_type} {self._event_summary(event)}")

        if event_type == "iteration_started":
            self._on_iteration_started(event)
        elif event_type == "worker_completed":
            self._on_worker_completed(event)
        elif event_type == "worker_failed":
            self._on_worker_failed(event)
        elif event_type == "budget_checkpoint":
            self._on_budget_checkpoint(event)
        elif event_type == "run_completed":
            self._on_run_completed(event)
        elif event_type == "run_failed":
            log.write("[bold red]RUN FAILED[/bold red]")
        elif event_type == "pause_decision_needed":
            self._on_pause_decision_needed(event)

    def _event_summary(self, event: dict) -> str:
        """One-line summary of event data for the log."""
        etype = event.get("event", "")
        if etype == "worker_completed":
            return f"dim={event.get('dimension', '?')} metric={event.get('metric', '?')}"
        if etype == "iteration_started":
            return f"dim={event.get('dimension', '?')} workers={event.get('num_workers', '?')}"
        if etype == "breakthrough":
            return f"new_best={event.get('new_best', '?')}"
        return ""

    def _collapse_iteration(self, iteration: int) -> None:
        """Replace individual rows of *iteration* with a single summary row."""
        table = self.query_one("#experiment-table", DataTable)
        row_keys = self._iteration_rows.get(iteration, [])
        if not row_keys:
            return

        # Build summary from tracked data
        data = self._iteration_data.get(iteration, {})
        count = len(row_keys)
        statuses = [d.get("status", "running") for d in data.values()]
        if all(s == "done" for s in statuses):
            agg_status = "done"
        elif any(s == "FAILED" for s in statuses):
            agg_status = "mixed"
        else:
            agg_status = "running"

        total_cost = sum(d.get("cost", 0.0) for d in data.values())
        metrics = [d["metric"] for d in data.values() if "metric" in d]
        best_metric = max(metrics) if metrics else "-"
        dimension = next(
            (d.get("dimension", "?") for d in data.values()), "?"
        )

        self._iteration_summary[iteration] = {
            "count": count,
            "dimension": dimension,
            "status": agg_status,
            "cost": total_cost,
            "best_metric": best_metric,
        }

        # Remove individual rows
        for key in row_keys:
            try:
                table.remove_row(key)
            except Exception:
                pass

        # Add summary row
        summary_key = f"iter-{iteration}"
        cost_str = f"${total_cost:.2f}" if isinstance(total_cost, (int, float)) else "-"
        table.add_row(
            summary_key,
            f"{count} experiments",
            dimension,
            agg_status,
            cost_str,
            str(best_metric),
            key=summary_key,
        )
        self._collapsed.add(iteration)

    def _expand_iteration(self, iteration: int) -> None:
        """Replace summary row with individual experiment rows."""
        table = self.query_one("#experiment-table", DataTable)
        summary_key = f"iter-{iteration}"
        try:
            table.remove_row(summary_key)
        except Exception:
            pass

        row_keys = self._iteration_rows.get(iteration, [])
        data = self._iteration_data.get(iteration, {})
        for i, exp_id in enumerate(row_keys):
            d = data.get(exp_id, {})
            cost_val = d.get("cost", 0.0)
            cost_str = f"${cost_val:.2f}" if cost_val else "-"
            metric_str = str(d.get("metric", "-"))
            table.add_row(
                exp_id,
                d.get("worker", f"W{i+1}"),
                d.get("dimension", "?"),
                d.get("status", "running"),
                cost_str,
                metric_str,
                key=exp_id,
            )
        self._collapsed.discard(iteration)

    def _on_iteration_started(self, event: dict) -> None:
        table = self.query_one("#experiment-table", DataTable)
        tasks = event.get("tasks", [])
        iteration = event.get("iteration", "?")

        # Collapse previous iteration if there is one
        if self._current_iteration is not None and self._current_iteration != iteration:
            self._collapse_iteration(self._current_iteration)

        self._current_iteration = iteration
        self._iteration_rows.setdefault(iteration, [])
        self._iteration_data.setdefault(iteration, {})

        for i, task in enumerate(tasks):
            exp_id = task.get("experiment_id", f"exp-{iteration}-{i}")
            table.add_row(
                exp_id, f"W{i+1}", event.get("dimension", "?"),
                "running", "-", "-",
                key=exp_id,
            )
            self._iteration_rows[iteration].append(exp_id)
            self._iteration_data[iteration][exp_id] = {
                "worker": f"W{i+1}",
                "dimension": event.get("dimension", "?"),
                "status": "running",
                "cost": 0.0,
            }

    def _on_worker_completed(self, event: dict) -> None:
        table = self.query_one("#experiment-table", DataTable)
        exp_id = event.get("experiment_id", "")
        metric = event.get("metric", "?")
        cost = event.get("cost_usd", 0)
        # Update tracked data for this experiment
        for it_data in self._iteration_data.values():
            if exp_id in it_data:
                it_data[exp_id]["status"] = "done"
                it_data[exp_id]["cost"] = cost
                it_data[exp_id]["metric"] = metric
                break
        try:
            table.update_cell(exp_id, "Status", "done")
            table.update_cell(exp_id, "Cost", f"${cost:.2f}")
            table.update_cell(exp_id, "Delta", f"{metric}")
        except Exception:
            pass

    def _on_worker_failed(self, event: dict) -> None:
        table = self.query_one("#experiment-table", DataTable)
        exp_id = event.get("experiment_id", "")
        # Update tracked data for this experiment
        for it_data in self._iteration_data.values():
            if exp_id in it_data:
                it_data[exp_id]["status"] = "FAILED"
                break
        try:
            table.update_cell(exp_id, "Status", "FAILED")
        except Exception:
            pass

    def _on_budget_checkpoint(self, event: dict) -> None:
        bar = self.query_one("#budget-bar", BudgetBar)
        spent = event.get("spent_usd", 0)
        remaining_cost = event.get("remaining_cost")
        max_cost = (spent + remaining_cost) if remaining_cost is not None else None
        exp_run = event.get("experiments_run", 0)
        remaining_exp = event.get("remaining_experiments")
        max_exp = (exp_run + remaining_exp) if remaining_exp is not None else None
        bar.update_budget(
            cost=spent,
            max_cost=max_cost,
            experiments=exp_run,
            max_experiments=max_exp,
            elapsed=self._format_elapsed(event.get("elapsed_seconds", 0)),
        )

    def _on_run_completed(self, event: dict) -> None:
        log = self.query_one("#event-log", RichLog)
        best = event.get("best_metric", "?")
        total = event.get("total_experiments", "?")
        cost = event.get("total_cost_usd", 0)
        log.write(f"[bold green]RUN COMPLETE[/bold green] best={best} experiments={total} cost=${cost:.2f}")

    def _on_pause_decision_needed(self, event: dict) -> None:
        """Show notification and let user decide via keybindings."""
        log = self.query_one("#event-log", RichLog)
        log.write("[bold yellow]PAUSE REQUESTED[/bold yellow] — Press [P] to pause or [Q] to continue and exit TUI")

    def action_toggle_expand(self) -> None:
        """Handle X key — expand or collapse a past iteration summary row."""
        table = self.query_one("#experiment-table", DataTable)
        if table.cursor_row is None:
            return
        try:
            row_key = table.get_row_at(table.cursor_row)
        except Exception:
            return
        # Get the actual RowKey from the coordinate
        try:
            row_key_obj = list(table.rows.keys())[table.cursor_row]
            key_value = str(row_key_obj.value)
        except (IndexError, AttributeError):
            return

        # Check if it's a summary row (iter-N format)
        if key_value.startswith("iter-"):
            try:
                iteration = int(key_value.split("-", 1)[1])
            except (ValueError, IndexError):
                return
            if iteration in self._collapsed:
                self._expand_iteration(iteration)
            return

        # Check if the cursor is on an individual row from a past iteration
        # and collapse it back
        for iteration, row_keys in self._iteration_rows.items():
            if key_value in row_keys and iteration != self._current_iteration:
                if iteration not in self._collapsed:
                    self._collapse_iteration(iteration)
                return

    def action_pause(self) -> None:
        """Handle P key — pause the coordinator."""
        self._pause_controller.pause_requested = True
        if self._pause_gate.decision_needed.is_set():
            self._pause_gate.submit_decision("pause")
        log = self.query_one("#event-log", RichLog)
        log.write("[yellow]Pause submitted[/yellow]")

    def action_extend(self) -> None:
        """Handle E key — extend budget. For now, fixed increment."""
        self._coordinator.extend_budget(add_cost=5.0, add_experiments=5)
        log = self.query_one("#event-log", RichLog)
        log.write("[green]Budget extended: +$5.00, +5 experiments[/green]")

    def action_quit_tui(self) -> None:
        """Handle Q/Esc — exit TUI mode."""
        if self._pause_gate.decision_needed.is_set():
            self._pause_gate.submit_decision("continue")
        if self._event_queue is not None:
            self._bridge.unsubscribe(self._event_queue)
        self.exit()

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
