"""Integration tests for coordinator with batch run_experiments."""

import pytest
from chaosengineer.core.models import (
    BudgetConfig, Baseline, ExperimentResult, ExperimentStatus,
)
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.budget import BudgetTracker
from chaosengineer.metrics.logger import EventLogger
from chaosengineer.core.interfaces import DimensionPlan, ExperimentTask
from chaosengineer.testing.simulator import ScriptedDecisionMaker
from chaosengineer.testing.executor import ScriptedExecutor
from chaosengineer.workloads.parser import WorkloadSpec


def _make_spec(**overrides) -> WorkloadSpec:
    defaults = dict(
        name="test",
        primary_metric="val_bpb",
        metric_direction="lower",
        execution_command="echo test",
        workers_available=4,
        budget=BudgetConfig(max_experiments=100),
    )
    defaults.update(overrides)
    return WorkloadSpec(**defaults)


class TestBatchRunExperiments:
    """Verify coordinator uses run_experiments and handles batch results."""

    def test_batch_called_with_correct_tasks(self, tmp_output_dir):
        """The coordinator should build ExperimentTask objects and call run_experiments."""
        spec = _make_spec()
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}, {"lr": 0.08}]),
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.91),
            "exp-0-1": ExperimentResult(primary_metric=0.95),
        }

        # Track calls to run_experiments
        calls = []

        class TrackingExecutor(ScriptedExecutor):
            def run_experiments(self, tasks, on_worker_done=None):
                calls.append(tasks)
                return super().run_experiments(tasks, on_worker_done=on_worker_done)

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=TrackingExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="abc", metric_value=0.97, metric_name="val_bpb"),
        )

        coordinator.run()

        assert len(calls) == 1
        assert len(calls[0]) == 2
        assert isinstance(calls[0][0], ExperimentTask)
        assert calls[0][0].experiment_id == "exp-0-0"
        assert calls[0][0].params == {"lr": 0.02}
        assert calls[0][1].experiment_id == "exp-0-1"

    def test_mixed_success_failure_in_batch(self, tmp_output_dir):
        spec = _make_spec()
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}, {"lr": 0.08}]),
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.0, error_message="OOM"),
            "exp-0-1": ExperimentResult(primary_metric=0.93),
        }
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="abc", metric_value=0.97, metric_name="val_bpb"),
        )

        coordinator.run()

        assert coordinator.best_baseline.metric_value == 0.93
        failed = [
            e for e in coordinator.run_state.experiments
            if e.status == ExperimentStatus.FAILED
        ]
        assert len(failed) == 1

    def test_budget_tracking_across_batches(self, tmp_output_dir):
        spec = _make_spec()
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}]),
            DimensionPlan(dimension_name="depth", values=[{"depth": 6}]),
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.91, cost_usd=0.10),
            "exp-1-0": ExperimentResult(primary_metric=0.89, cost_usd=0.15),
        }
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="abc", metric_value=0.97, metric_name="val_bpb"),
        )

        coordinator.run()

        assert coordinator.budget.experiments_run == 2
        assert coordinator.budget.spent_usd == pytest.approx(0.25)

    def test_all_experiments_started_before_batch(self, tmp_output_dir):
        """All experiments should be in RUNNING state before batch call."""
        spec = _make_spec()
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}, {"lr": 0.08}]),
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.91),
            "exp-0-1": ExperimentResult(primary_metric=0.95),
        }

        pre_batch_statuses = []

        class InspectingExecutor(ScriptedExecutor):
            def __init__(self, results, coordinator_ref):
                super().__init__(results)
                self._coordinator_ref = coordinator_ref

            def run_experiments(self, tasks, on_worker_done=None):
                # Capture experiment statuses before execution
                for exp in self._coordinator_ref.run_state.experiments:
                    pre_batch_statuses.append(exp.status)
                return super().run_experiments(tasks, on_worker_done=on_worker_done)

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),  # placeholder
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="abc", metric_value=0.97, metric_name="val_bpb"),
        )
        # Replace executor with inspecting version
        coordinator.executor = InspectingExecutor(results, coordinator)

        coordinator.run()

        assert all(s == ExperimentStatus.RUNNING for s in pre_batch_statuses)
