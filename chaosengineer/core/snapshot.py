"""RunSnapshot: reconstructed state from event log for resume."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
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


def build_snapshot(events_path: Path) -> RunSnapshot:
    """Replay events.jsonl to reconstruct run state."""
    events = _read_events(events_path)

    # Initial state — filled by run_started
    run_id = ""
    workload_name = ""
    workload_spec_hash = ""
    metric_direction = "lower"
    budget_config = BudgetConfig()
    mode = "parallel"
    active_baselines: list[Baseline] = []
    baseline_history: list[Baseline] = []
    dimensions_explored: list[DimensionOutcome] = []
    discovered_dimensions: dict[str, list[str]] = {}
    experiments: list[ExperimentSummary] = []
    history: list[dict] = []
    total_cost_usd = 0.0
    total_experiments_run = 0
    elapsed_time = 0.0
    consecutive_no_improvement = 0
    stop_reason: StopReason | None = None

    # Iteration tracking for incomplete detection
    current_iteration_dim: str | None = None
    current_iteration_worker_count = 0
    current_iteration_tasks: list[dict] = []
    current_iteration_completed: list[ExperimentSummary] = []
    current_iteration_completed_ids: set[str] = set()
    iteration_finalized = True

    for entry in events:
        event_type = entry.get("event", "")

        if event_type == "run_started":
            run_id = entry.get("run_id", "")
            workload_name = entry.get("workload", "")  # coordinator uses "workload" not "workload_name"
            workload_spec_hash = entry.get("workload_spec_hash", "")
            metric_direction = entry.get("metric_direction", "lower")
            budget_data = entry.get("budget", {})
            budget_config = BudgetConfig(
                max_api_cost=budget_data.get("max_api_cost"),
                max_experiments=budget_data.get("max_experiments"),
                max_wall_time_seconds=budget_data.get("max_wall_time_seconds"),
                max_plateau_iterations=budget_data.get("max_plateau_iterations"),
            )
            mode = entry.get("mode", "parallel")
            bl = entry.get("baseline", {})
            initial_baseline = Baseline(
                commit=bl.get("commit", ""),
                metric_value=bl.get("metric_value", 0.0),
                metric_name=bl.get("metric_name", ""),
            )
            active_baselines = [initial_baseline]
            baseline_history = [initial_baseline]

        elif event_type == "iteration_started":
            if not iteration_finalized and current_iteration_dim:
                _finalize_iteration(
                    current_iteration_dim, current_iteration_completed,
                    dimensions_explored, metric_direction,
                )
            current_iteration_dim = entry.get("dimension", "")
            current_iteration_worker_count = entry.get("num_workers", 0)  # coordinator uses "num_workers"
            current_iteration_tasks = entry.get("tasks", [])
            current_iteration_completed = []
            current_iteration_completed_ids = set()
            iteration_finalized = False

        elif event_type in ("worker_completed", "worker_failed"):
            exp_id = entry.get("experiment_id", "")
            dim = entry.get("dimension", "")
            params = entry.get("params", {})
            metric = entry.get("metric")
            cost = entry.get("cost_usd", 0.0)
            status = "completed" if event_type == "worker_completed" else "failed"

            summary = ExperimentSummary(exp_id, dim, params, metric, status, cost)
            experiments.append(summary)
            total_experiments_run += 1
            total_cost_usd += cost

            history.append({
                "experiment_id": exp_id,
                "dimension": dim,
                "params": params,
                "metric": metric,
                "status": status,
            })

            current_iteration_completed.append(summary)
            current_iteration_completed_ids.add(exp_id)

        elif event_type == "breakthrough":
            new_bl = Baseline(
                commit=entry.get("commit", ""),
                metric_value=entry.get("metric", entry.get("new_best", 0.0)),
                metric_name=active_baselines[0].metric_name if active_baselines else "",
            )
            active_baselines = [new_bl]
            baseline_history.append(new_bl)
            consecutive_no_improvement = 0

        elif event_type == "diverse_discovered":
            dim_name = entry.get("dimension", "")
            options = entry.get("options", [])
            discovered_dimensions[dim_name] = options

        elif event_type == "budget_checkpoint":
            elapsed_time = entry.get("elapsed_seconds", elapsed_time)
            consecutive_no_improvement = entry.get(
                "consecutive_no_improvement", consecutive_no_improvement
            )

        elif event_type == "run_paused":
            stop_reason = StopReason.PAUSED
            budget_state = entry.get("budget_state", {})
            elapsed_time = budget_state.get("elapsed_seconds", elapsed_time)
            bl_list = entry.get("active_baselines", [])
            if bl_list:
                active_baselines = [
                    Baseline(b["commit"], b["metric_value"], b["metric_name"])
                    for b in bl_list
                ]
            if not iteration_finalized and current_iteration_dim:
                if len(current_iteration_completed) >= current_iteration_worker_count:
                    _finalize_iteration(
                        current_iteration_dim, current_iteration_completed,
                        dimensions_explored, metric_direction,
                    )
                    iteration_finalized = True

        elif event_type == "run_completed":
            stop_reason = StopReason.COMPLETED
            if not iteration_finalized and current_iteration_dim:
                _finalize_iteration(
                    current_iteration_dim, current_iteration_completed,
                    dimensions_explored, metric_direction,
                )
                iteration_finalized = True

        elif event_type == "run_resumed":
            stop_reason = None

        elif event_type == "iteration_gap_completed":
            pass

    if stop_reason is None:
        stop_reason = StopReason.CRASHED

    incomplete_iteration = None
    if not iteration_finalized and current_iteration_dim:
        completed_count = len(current_iteration_completed)
        if completed_count < current_iteration_worker_count:
            missing_tasks = []
            missing_ids = []
            for task_data in current_iteration_tasks:
                tid = task_data["experiment_id"]
                if tid not in current_iteration_completed_ids:
                    missing_ids.append(tid)
                    missing_tasks.append(ExperimentTask(
                        experiment_id=tid,
                        params=task_data["params"],
                        command=task_data["command"],
                        baseline_commit=task_data["baseline_commit"],
                    ))
            incomplete_iteration = IncompleteIteration(
                dimension=current_iteration_dim,
                total_workers=current_iteration_worker_count,
                completed_experiments=current_iteration_completed,
                missing_experiment_ids=missing_ids,
                missing_tasks=missing_tasks,
            )
        else:
            _finalize_iteration(
                current_iteration_dim, current_iteration_completed,
                dimensions_explored, metric_direction,
            )

    return RunSnapshot(
        run_id=run_id,
        workload_name=workload_name,
        workload_spec_hash=workload_spec_hash,
        budget_config=budget_config,
        mode=mode,
        active_baselines=active_baselines,
        baseline_history=baseline_history,
        dimensions_explored=dimensions_explored,
        discovered_dimensions=discovered_dimensions,
        experiments=experiments,
        history=history,
        total_cost_usd=total_cost_usd,
        total_experiments_run=total_experiments_run,
        elapsed_time=elapsed_time,
        consecutive_no_improvement=consecutive_no_improvement,
        incomplete_iteration=incomplete_iteration,
        stop_reason=stop_reason,
    )


def _read_events(events_path: Path) -> list[dict]:
    """Read JSONL event file."""
    events = []
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _finalize_iteration(
    dimension: str,
    completed: list[ExperimentSummary],
    dimensions_explored: list[DimensionOutcome],
    metric_direction: str = "lower",
):
    """Record a completed iteration as a DimensionOutcome."""
    values_tested = []
    best_metric = None
    best_value = None
    for exp in completed:
        val = str(list(exp.params.values())[0]) if exp.params else "?"
        values_tested.append(val)
        if exp.metric is not None:
            if best_metric is None:
                best_metric = exp.metric
                best_value = val
            elif metric_direction == "lower" and exp.metric < best_metric:
                best_metric = exp.metric
                best_value = val
            elif metric_direction == "higher" and exp.metric > best_metric:
                best_metric = exp.metric
                best_value = val

    dimensions_explored.append(DimensionOutcome(
        name=dimension,
        values_tested=values_tested,
        winner=best_value,
        metric_improvement=None,
    ))
