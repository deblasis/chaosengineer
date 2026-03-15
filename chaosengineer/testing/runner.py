"""Scenario runner: loads YAML scenarios, wires up simulator+executor, runs coordinator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.models import Baseline, BudgetConfig, ExperimentResult
from chaosengineer.metrics.logger import EventLogger
from chaosengineer.testing.executor import ScriptedExecutor
from chaosengineer.testing.simulator import DimensionPlan, ScriptedDecisionMaker
from chaosengineer.workloads.parser import WorkloadSpec


@dataclass
class ScenarioResult:
    """Result of running a scenario."""
    scenario_name: str
    passed: bool
    final_best_metric: float
    total_experiments: int
    event_log_path: Path
    expected: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def load_scenario(
    path: Path | str | None = None,
    content: str | None = None,
) -> dict:
    """Load a YAML scenario file or string."""
    if content is None:
        if path is None:
            raise ValueError("Either path or content must be provided")
        content = Path(path).read_text()
    return yaml.safe_load(content)


def _build_workload_spec(workload_data: dict) -> WorkloadSpec:
    """Build a WorkloadSpec from scenario YAML data."""
    budget_data = workload_data.get("budget", {})
    budget = BudgetConfig(
        max_api_cost=budget_data.get("max_api_cost"),
        max_experiments=budget_data.get("max_experiments"),
        max_wall_time_seconds=budget_data.get("max_wall_time_seconds"),
        max_plateau_iterations=budget_data.get("max_plateau_iterations"),
    )
    return WorkloadSpec(
        name=workload_data.get("name", "scenario-test"),
        primary_metric=workload_data.get("primary_metric", "metric"),
        metric_direction=workload_data.get("metric_direction", "lower"),
        execution_command=workload_data.get("execution_command", "echo test"),
        workers_available=workload_data.get("workers_available", 1),
        budget=budget,
    )


def _build_plans(plans_data: list[dict]) -> list[DimensionPlan]:
    """Build DimensionPlan list from scenario YAML data."""
    return [
        DimensionPlan(
            dimension_name=p["dimension_name"],
            values=p["values"],
        )
        for p in plans_data
    ]


def _build_results(results_data: dict) -> dict[str, ExperimentResult]:
    """Build ExperimentResult map from scenario YAML data."""
    results = {}
    for exp_id, data in results_data.items():
        results[exp_id] = ExperimentResult(
            primary_metric=data["primary_metric"],
            duration_seconds=data.get("duration_seconds", 0),
            error_message=data.get("error_message"),
        )
    return results


class ScenarioRunner:
    """Runs test scenarios without LLM calls or real experiment execution."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def run_scenario(
        self,
        path: Path | str | None = None,
        content: str | None = None,
    ) -> ScenarioResult:
        """Run a scenario and return the result."""
        scenario = load_scenario(path=path, content=content)
        scenario_name = scenario.get("scenario", "unnamed")

        # Build components from scenario data
        spec = _build_workload_spec(scenario["workload"])
        plans = _build_plans(scenario["plans"])
        results = _build_results(scenario["results"])
        initial_baseline = Baseline(
            commit=scenario["initial_baseline"]["commit"],
            metric_value=scenario["initial_baseline"]["metric_value"],
            metric_name=scenario["initial_baseline"]["metric_name"],
        )

        event_log_path = self.output_dir / f"{scenario_name.replace(' ', '_')}.jsonl"

        tie_threshold_pct = scenario.get("tie_threshold_pct", 1.0)

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(event_log_path),
            budget=BudgetTracker(spec.budget),
            initial_baseline=initial_baseline,
            tie_threshold_pct=tie_threshold_pct,
        )

        coordinator.run()

        # Check expected outcomes
        expected = scenario.get("expected", {})
        errors = []

        if "final_best_metric" in expected:
            actual = coordinator.best_baseline.metric_value
            exp_val = expected["final_best_metric"]
            if abs(actual - exp_val) > 1e-6:
                errors.append(
                    f"final_best_metric: expected {exp_val}, got {actual}"
                )

        if "total_experiments" in expected:
            actual = coordinator.budget.experiments_run
            exp_val = expected["total_experiments"]
            if actual != exp_val:
                errors.append(
                    f"total_experiments: expected {exp_val}, got {actual}"
                )

        if "breakthroughs" in expected:
            events = coordinator.logger.read_events(event_type="breakthrough")
            actual = len(events)
            exp_val = expected["breakthroughs"]
            if actual != exp_val:
                errors.append(
                    f"breakthroughs: expected {exp_val}, got {actual}"
                )

        return ScenarioResult(
            scenario_name=scenario_name,
            passed=len(errors) == 0,
            final_best_metric=coordinator.best_baseline.metric_value,
            total_experiments=coordinator.budget.experiments_run,
            event_log_path=event_log_path,
            expected=expected,
            errors=errors,
        )
