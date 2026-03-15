"""Budget tracking and enforcement."""

from __future__ import annotations

import time
from typing import Any

from chaosengineer.core.models import BudgetConfig


class BudgetTracker:
    """Tracks spending against budget limits."""

    def __init__(self, config: BudgetConfig):
        self.config = config
        self.spent_usd: float = 0.0
        self.experiments_run: int = 0
        self.consecutive_no_improvement: int = 0
        self._start_time: float | None = None

    def start(self) -> None:
        self._start_time = time.monotonic()

    def add_cost(self, cost_usd: float) -> None:
        self.spent_usd += cost_usd

    def record_experiment(self) -> None:
        self.experiments_run += 1

    def record_no_improvement(self) -> None:
        self.consecutive_no_improvement += 1

    def record_improvement(self) -> None:
        self.consecutive_no_improvement = 0

    @property
    def remaining_cost(self) -> float | None:
        if self.config.max_api_cost is None:
            return None
        return max(0.0, self.config.max_api_cost - self.spent_usd)

    @property
    def remaining_experiments(self) -> int | None:
        if self.config.max_experiments is None:
            return None
        return max(0, self.config.max_experiments - self.experiments_run)

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def remaining_time(self) -> float | None:
        if self.config.max_wall_time_seconds is None:
            return None
        if self._start_time is None:
            return self.config.max_wall_time_seconds
        return max(0.0, self.config.max_wall_time_seconds - self.elapsed_seconds)

    def is_exhausted(self) -> bool:
        if self.config.max_api_cost is not None and self.spent_usd >= self.config.max_api_cost:
            return True
        if self.config.max_experiments is not None and self.experiments_run >= self.config.max_experiments:
            return True
        if (
            self.config.max_wall_time_seconds is not None
            and self._start_time is not None
            and self.elapsed_seconds >= self.config.max_wall_time_seconds
        ):
            return True
        if (
            self.config.max_plateau_iterations is not None
            and self.consecutive_no_improvement >= self.config.max_plateau_iterations
        ):
            return True
        return False

    def snapshot(self) -> dict[str, Any]:
        return {
            "spent_usd": self.spent_usd,
            "remaining_cost": self.remaining_cost,
            "experiments_run": self.experiments_run,
            "remaining_experiments": self.remaining_experiments,
            "elapsed_seconds": self.elapsed_seconds,
            "remaining_time": self.remaining_time,
            "consecutive_no_improvement": self.consecutive_no_improvement,
            "is_exhausted": self.is_exhausted(),
        }
