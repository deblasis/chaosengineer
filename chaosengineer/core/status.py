"""StatusDisplay — real-time progress output to stderr."""

from __future__ import annotations

import sys
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from chaosengineer.core.interfaces import ExperimentTask
    from chaosengineer.core.models import BudgetConfig, ExperimentResult


class StatusDisplay:
    """Prints run progress to stderr."""

    def __init__(self) -> None:
        self._iteration: int = 0
        self._cost: float = 0.0
        self._start_time: float | None = None

    def on_run_start(self, budget_config: "BudgetConfig") -> None:
        self._start_time = time.monotonic()
        parts = []
        if budget_config.max_experiments is not None:
            parts.append(f"max_experiments={budget_config.max_experiments}")
        if budget_config.max_api_cost is not None:
            parts.append(f"max_cost=${budget_config.max_api_cost:.2f}")
        if budget_config.max_wall_time_seconds is not None:
            parts.append(f"max_time={budget_config.max_wall_time_seconds}s")
        budget_str = ", ".join(parts) if parts else "unlimited"
        print(f"Budget: {budget_str}", file=sys.stderr)

    def on_worker_done(
        self,
        task: "ExperimentTask",
        result: "ExperimentResult",
        completed: int,
        total: int,
    ) -> None:
        self._cost += getattr(result, "cost_usd", 0.0) or 0.0
        elapsed = self._elapsed()
        line = self._format_progress(
            iteration=self._iteration,
            completed=completed,
            total=total,
            cost=self._cost,
            elapsed=elapsed,
        )
        print(f"\r{line}", end="", file=sys.stderr)

    def on_iteration_done(self, iteration: int, best_metric: float) -> None:
        self._iteration = iteration + 1
        elapsed = self._elapsed()
        line = self._format_progress(
            iteration=iteration,
            completed=0,
            total=0,
            cost=self._cost,
            elapsed=elapsed,
        )
        print(f"\r{line} | best={best_metric}", file=sys.stderr)

    def _elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    def _format_progress(
        self,
        iteration: int,
        completed: int,
        total: int,
        cost: float,
        elapsed: float,
    ) -> str:
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        time_str = f"{h:02d}:{m:02d}:{s:02d}"

        parts = [f"iter {iteration}"]
        if total > 0:
            parts.append(f"{completed}/{total} workers done")
        parts.append(f"${cost:.2f}")
        parts.append(time_str)

        line = "[" + " | ".join(parts) + "]"
        line += " Ctrl+C to pause"

        return line
