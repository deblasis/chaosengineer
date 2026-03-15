"""Tests for coordinator logic."""

import pytest
from chaosengineer.core.models import (
    BudgetConfig, Baseline, DimensionSpec, DimensionType,
    Experiment, ExperimentResult, ExperimentStatus,
)
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.budget import BudgetTracker
from chaosengineer.metrics.logger import EventLogger
from chaosengineer.core.interfaces import DimensionPlan
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


class TestCoordinatorBasicFlow:
    """Test a simple 1-dimension, 2-worker directional sweep."""

    def test_single_iteration_picks_winner(self, tmp_output_dir):
        spec = _make_spec()
        plans = [
            DimensionPlan(
                dimension_name="lr",
                values=[{"lr": 0.02}, {"lr": 0.08}],
            ),
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.91),
            "exp-0-1": ExperimentResult(primary_metric=0.95),
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

        # Best baseline should be updated to 0.91 (lower is better)
        assert coordinator.best_baseline.metric_value == 0.91
        # 2 experiments should have run
        assert coordinator.budget.experiments_run == 2
        # Events should be logged
        events = coordinator.logger.read_events()
        assert any(e["event"] == "run_started" for e in events)
        assert any(e["event"] == "breakthrough" for e in events)

    def test_no_improvement_records_plateau(self, tmp_output_dir):
        spec = _make_spec()
        plans = [
            DimensionPlan(
                dimension_name="lr",
                values=[{"lr": 0.02}, {"lr": 0.08}],
            ),
        ]
        # Both results worse than baseline
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.99),
            "exp-0-1": ExperimentResult(primary_metric=0.98),
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

        # Baseline should not change
        assert coordinator.best_baseline.metric_value == 0.97
        assert coordinator.budget.consecutive_no_improvement == 1


class TestCoordinatorMultiIteration:
    """Test multiple iterations across dimensions."""

    def test_two_dimensions(self, tmp_output_dir):
        spec = _make_spec()
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}, {"lr": 0.08}]),
            DimensionPlan(dimension_name="depth", values=[{"depth": 6}, {"depth": 12}]),
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.91),
            "exp-0-1": ExperimentResult(primary_metric=0.95),
            "exp-1-0": ExperimentResult(primary_metric=0.89),
            "exp-1-1": ExperimentResult(primary_metric=0.92),
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

        assert coordinator.best_baseline.metric_value == 0.89
        assert coordinator.budget.experiments_run == 4


class TestCoordinatorBeamSearch:
    """Test branching on ties."""

    def test_tie_creates_branches(self, tmp_output_dir):
        spec = _make_spec(
            budget=BudgetConfig(max_experiments=100),
        )
        plans = [
            DimensionPlan(
                dimension_name="activation",
                values=[{"act": "A"}, {"act": "B"}, {"act": "C"}],
            ),
            # After tie between A and B, coordinator plans next dim for each branch
            DimensionPlan(dimension_name="depth_branchA", values=[{"depth": 6}]),
            DimensionPlan(dimension_name="depth_branchB", values=[{"depth": 6}]),
        ]
        results = {
            # A and B tie (within threshold), C is worse
            "exp-0-0": ExperimentResult(primary_metric=0.91),
            "exp-0-1": ExperimentResult(primary_metric=0.911),  # within 1% of 0.91
            "exp-0-2": ExperimentResult(primary_metric=0.98),
            # Branch explorations
            "exp-1-0": ExperimentResult(primary_metric=0.88),
            "exp-2-0": ExperimentResult(primary_metric=0.90),
        }
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="abc", metric_value=0.97, metric_name="val_bpb"),
            tie_threshold_pct=1.0,  # 1% threshold
        )

        coordinator.run()

        # Best should be 0.88 from branch A exploration
        assert coordinator.best_baseline.metric_value == 0.88


class TestCoordinatorBudgetEnforcement:
    """Test that the coordinator stops when budget is exhausted."""

    def test_stops_at_experiment_limit(self, tmp_output_dir):
        spec = _make_spec(budget=BudgetConfig(max_experiments=2))
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}, {"lr": 0.08}]),
            # This should never execute
            DimensionPlan(dimension_name="depth", values=[{"depth": 6}]),
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.93),
            "exp-0-1": ExperimentResult(primary_metric=0.95),
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
        # Second dimension should NOT have been explored
        events = coordinator.logger.read_events(event_type="iteration_started")
        assert len(events) == 1  # only one iteration

    def test_stops_on_plateau(self, tmp_output_dir):
        spec = _make_spec(budget=BudgetConfig(max_plateau_iterations=2))
        plans = [
            DimensionPlan(dimension_name="a", values=[{"a": 1}]),
            DimensionPlan(dimension_name="b", values=[{"b": 1}]),
            DimensionPlan(dimension_name="c", values=[{"c": 1}]),  # should not run
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.99),  # worse
            "exp-1-0": ExperimentResult(primary_metric=0.98),  # still worse
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
        assert coordinator.budget.consecutive_no_improvement == 2


class TestCoordinatorFailedExperiment:
    """Test handling of failed experiments."""

    def test_failure_does_not_crash_coordinator(self, tmp_output_dir):
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

        # Should pick the non-failed result
        assert coordinator.best_baseline.metric_value == 0.93
        # Failed experiment should be recorded
        failed = [
            e for e in coordinator.run_state.experiments
            if e.status == ExperimentStatus.FAILED
        ]
        assert len(failed) == 1


class TestFindTies:
    def test_ties_with_zero_best_metric(self, tmp_output_dir):
        spec = _make_spec()
        plans = [
            DimensionPlan(
                dimension_name="lr",
                values=[{"lr": 0.01}, {"lr": 0.02}, {"lr": 0.03}],
            ),
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.0),
            "exp-0-1": ExperimentResult(primary_metric=0.005),
            "exp-0-2": ExperimentResult(primary_metric=0.5),
        }
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="abc", metric_value=0.97, metric_name="val_bpb"),
            tie_threshold_pct=1.0,
        )

        sorted_results = [
            (Experiment(experiment_id="e0", dimension="lr", params={}, baseline_commit="abc"),
             ExperimentResult(primary_metric=0.0)),
            (Experiment(experiment_id="e1", dimension="lr", params={}, baseline_commit="abc"),
             ExperimentResult(primary_metric=0.005)),
            (Experiment(experiment_id="e2", dimension="lr", params={}, baseline_commit="abc"),
             ExperimentResult(primary_metric=0.5)),
        ]

        tied = coordinator._find_ties(sorted_results, best_metric=0.0)
        assert len(tied) >= 2
        assert all(r.primary_metric < 0.1 for _, r in tied)


class TestCoordinatorBudgetTrimming:
    def test_budget_trim_logs_warning(self, tmp_output_dir):
        spec = _make_spec(budget=BudgetConfig(max_experiments=3))
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}, {"lr": 0.08}]),
            DimensionPlan(dimension_name="depth", values=[{"d": 6}, {"d": 12}]),
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.93),
            "exp-0-1": ExperimentResult(primary_metric=0.95),
            "exp-1-0": ExperimentResult(primary_metric=0.90),
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

        trim_events = coordinator.logger.read_events(event_type="budget_trim")
        assert len(trim_events) == 1
        assert trim_events[0]["original_count"] == 2
        assert trim_events[0]["trimmed_count"] == 1


class TestCoordinatorRunId:
    def test_custom_run_id(self, tmp_output_dir):
        spec = _make_spec()
        plans = [DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}])]
        results = {"exp-0-0": ExperimentResult(primary_metric=0.91)}
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="abc", metric_value=0.97, metric_name="val_bpb"),
            run_id="my-custom-run",
        )
        assert coordinator.run_state.run_id == "my-custom-run"

    def test_auto_generated_run_id(self, tmp_output_dir):
        spec = _make_spec()
        plans = [DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}])]
        results = {"exp-0-0": ExperimentResult(primary_metric=0.91)}
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="abc", metric_value=0.97, metric_name="val_bpb"),
        )
        assert coordinator.run_state.run_id != "run-001"
        assert len(coordinator.run_state.run_id) > 0


class TestCoordinatorExceptionLogging:
    """Test that exceptions during execution are logged as worker_failed events."""

    def test_exception_logs_worker_failed_event(self, tmp_output_dir):
        """When executor raises an exception (not just returns error_message),
        the coordinator should still log a worker_failed event."""
        spec = _make_spec()
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}]),
        ]
        results = {}  # empty — will raise KeyError
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="abc", metric_value=0.97, metric_name="val_bpb"),
        )

        coordinator.run()

        failed_events = coordinator.logger.read_events(event_type="worker_failed")
        assert len(failed_events) == 1
        assert "exp-0-0" in failed_events[0]["experiment_id"]
