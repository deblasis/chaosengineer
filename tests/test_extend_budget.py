"""Tests for Coordinator.extend_budget()."""
from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.models import Baseline, BudgetConfig
from chaosengineer.metrics.logger import EventLogger
from chaosengineer.testing.executor import ScriptedExecutor
from chaosengineer.testing.simulator import ScriptedDecisionMaker
from chaosengineer.workloads.parser import WorkloadSpec


def _make_coordinator(tmp_path, budget=None):
    spec = WorkloadSpec(
        name="test", primary_metric="loss", metric_direction="lower",
        execution_command="echo 1", workers_available=1,
        budget=budget or BudgetConfig(max_api_cost=10.0, max_experiments=5),
    )
    return Coordinator(
        spec=spec, decision_maker=ScriptedDecisionMaker([]),
        executor=ScriptedExecutor({}),
        logger=EventLogger(tmp_path / "events.jsonl"),
        budget=BudgetTracker(spec.budget),
        initial_baseline=Baseline("HEAD", 3.0, "loss"),
    )


class TestExtendBudget:
    def test_extend_cost(self, tmp_path):
        c = _make_coordinator(tmp_path)
        c.extend_budget(add_cost=5.0)
        assert c.budget.config.max_api_cost == 15.0

    def test_extend_experiments(self, tmp_path):
        c = _make_coordinator(tmp_path)
        c.extend_budget(add_experiments=3)
        assert c.budget.config.max_experiments == 8

    def test_extend_time(self, tmp_path):
        c = _make_coordinator(tmp_path, BudgetConfig(max_wall_time_seconds=60.0))
        c.extend_budget(add_time=30.0)
        assert c.budget.config.max_wall_time_seconds == 90.0

    def test_extend_preserves_none_fields(self, tmp_path):
        c = _make_coordinator(tmp_path, BudgetConfig(max_api_cost=10.0))
        c.extend_budget(add_experiments=5)
        assert c.budget.config.max_api_cost == 10.0
        assert c.budget.config.max_experiments is None  # was None, stays None

    def test_extend_preserves_plateau(self, tmp_path):
        c = _make_coordinator(tmp_path, BudgetConfig(max_api_cost=10.0, max_plateau_iterations=3))
        c.extend_budget(add_cost=5.0)
        assert c.budget.config.max_plateau_iterations == 3
