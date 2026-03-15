"""RunSnapshot: reconstructed state from event log for resume."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from chaosengineer.core.interfaces import ExperimentTask
from chaosengineer.core.models import Baseline, BudgetConfig


class StopReason(Enum):
    PAUSED = "paused"
    COMPLETED = "completed"
    CRASHED = "crashed"


@dataclass
class ExperimentSummary:
    experiment_id: str
    dimension: str
    params: dict[str, Any]
    metric: float | None
    status: str  # "completed" or "failed"
    cost_usd: float


@dataclass
class DimensionOutcome:
    name: str
    values_tested: list[str]
    winner: str | None
    metric_improvement: float | None


@dataclass
class IncompleteIteration:
    dimension: str
    total_workers: int
    completed_experiments: list[ExperimentSummary]
    missing_experiment_ids: list[str]
    missing_tasks: list[ExperimentTask]


@dataclass
class RunSnapshot:
    run_id: str
    workload_name: str
    workload_spec_hash: str
    budget_config: BudgetConfig
    mode: str

    # Restored state
    active_baselines: list[Baseline]
    baseline_history: list[Baseline]
    dimensions_explored: list[DimensionOutcome]
    discovered_dimensions: dict[str, list[str]]
    experiments: list[ExperimentSummary]
    history: list[dict[str, Any]]
    total_cost_usd: float
    total_experiments_run: int
    elapsed_time: float
    consecutive_no_improvement: int

    # Incomplete work
    incomplete_iteration: IncompleteIteration | None
    stop_reason: StopReason
