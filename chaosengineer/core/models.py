"""Core data models for ChaosEngineer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExperimentStatus(Enum):
    PLANNED = "planned"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


class WorkerStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    TERMINATED = "terminated"


class DimensionType(Enum):
    DIRECTIONAL = "directional"
    ENUM = "enum"
    DIVERSE = "diverse"


@dataclass
class DimensionSpec:
    """A dimension of the experiment space."""
    name: str
    dim_type: DimensionType
    current_value: Any = None
    options: list[str] | None = None
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dim_type": self.dim_type.value,
            "current_value": self.current_value,
            "options": self.options,
            "description": self.description,
        }


@dataclass
class ExperimentResult:
    """Result of a completed experiment."""
    primary_metric: float
    secondary_metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    commit_hash: str | None = None
    duration_seconds: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error_message: str | None = None

    def to_dict(self) -> dict:
        return {
            "primary_metric": self.primary_metric,
            "secondary_metrics": self.secondary_metrics,
            "artifacts": self.artifacts,
            "commit_hash": self.commit_hash,
            "duration_seconds": self.duration_seconds,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "error_message": self.error_message,
        }


@dataclass
class Experiment:
    """A single experiment with specific parameters."""
    experiment_id: str
    dimension: str
    params: dict[str, Any]
    baseline_commit: str
    status: ExperimentStatus = ExperimentStatus.PLANNED
    worker_id: str | None = None
    result: ExperimentResult | None = None
    branch_id: str | None = None  # for beam search branching
    is_stale: bool = False  # marked when a breakthrough invalidates this experiment

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "dimension": self.dimension,
            "params": self.params,
            "baseline_commit": self.baseline_commit,
            "status": self.status.value,
            "worker_id": self.worker_id,
            "result": self.result.to_dict() if self.result else None,
            "branch_id": self.branch_id,
        }


@dataclass
class WorkerState:
    """Tracks the state of a worker."""
    worker_id: str
    resource: str = ""
    status: WorkerStatus = WorkerStatus.IDLE
    current_experiment_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "resource": self.resource,
            "status": self.status.value,
            "current_experiment_id": self.current_experiment_id,
        }


@dataclass
class Baseline:
    """A known-good state to branch experiments from."""
    commit: str
    metric_value: float
    metric_name: str
    branch_id: str | None = None  # for beam search

    def to_dict(self) -> dict:
        return {
            "commit": self.commit,
            "metric_value": self.metric_value,
            "metric_name": self.metric_name,
            "branch_id": self.branch_id,
        }


@dataclass
class BudgetConfig:
    """Budget constraints for a run."""
    max_api_cost: float | None = None
    max_experiments: int | None = None
    max_wall_time_seconds: float | None = None
    max_plateau_iterations: int | None = None  # stop after N iterations with no improvement (None = no limit)

    def to_dict(self) -> dict:
        return {
            "max_api_cost": self.max_api_cost,
            "max_experiments": self.max_experiments,
            "max_wall_time_seconds": self.max_wall_time_seconds,
            "max_plateau_iterations": self.max_plateau_iterations,
        }


@dataclass
class Run:
    """A complete experimentation session."""
    run_id: str
    workload_name: str
    budget: BudgetConfig
    mode: str = "parallel"  # "sequential" | "parallel"
    experiments: list[Experiment] = field(default_factory=list)
    workers: list[WorkerState] = field(default_factory=list)
    baselines: list[Baseline] = field(default_factory=list)
    dimensions_explored: list[str] = field(default_factory=list)
    current_iteration: int = 0
    total_cost_usd: float = 0.0
    total_experiments_run: int = 0
    start_time: float | None = None
    end_time: float | None = None
