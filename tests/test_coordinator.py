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


class TestRunStateSync:
    def test_run_state_reflects_execution(self, tmp_output_dir):
        spec = _make_spec()
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}, {"lr": 0.08}]),
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.91, cost_usd=0.10),
            "exp-0-1": ExperimentResult(primary_metric=0.95, cost_usd=0.15),
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

        assert coordinator.run_state.total_experiments_run == 2
        assert coordinator.run_state.total_cost_usd == pytest.approx(0.25)
        assert coordinator.run_state.current_iteration == 1
        assert coordinator.run_state.start_time is not None
        assert coordinator.run_state.end_time is not None
        assert "lr" in coordinator.run_state.dimensions_explored


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


class TestCoordinatorHistoryEfficiency:
    def test_history_passed_to_decision_maker_without_full_reread(self, tmp_output_dir):
        """The coordinator should not re-read the entire log file each iteration."""
        spec = _make_spec()
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}]),
            DimensionPlan(dimension_name="depth", values=[{"depth": 6}]),
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.91),
            "exp-1-0": ExperimentResult(primary_metric=0.89),
        }

        call_count = 0
        original_read = EventLogger.read_events

        def counting_read(self_logger, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_read(self_logger, *args, **kwargs)

        logger = EventLogger(tmp_output_dir / "events.jsonl")
        logger.read_events = lambda *a, **kw: counting_read(logger, *a, **kw)

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=logger,
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="abc", metric_value=0.97, metric_name="val_bpb"),
        )

        coordinator.run()

        # With in-memory history, read_events should not be called during run
        assert call_count == 0


class TestDiverseDimensionDiscovery:
    """Test _discover_diverse_dimensions integration."""

    def test_discovers_options_for_diverse_dims(self, tmp_output_dir):
        spec = _make_spec(
            dimensions=[
                DimensionSpec(name="optimizer", dim_type=DimensionType.DIVERSE),
            ],
            budget=BudgetConfig(max_experiments=1),
        )
        plans = [
            DimensionPlan(dimension_name="optimizer", values=[{"optimizer": "adam"}]),
        ]
        results = {"exp-0-0": ExperimentResult(primary_metric=0.9)}
        dm = ScriptedDecisionMaker(
            plans, diverse_options={"optimizer": ["adam", "sgd", "rmsprop"]},
        )
        coordinator = Coordinator(
            spec=spec,
            decision_maker=dm,
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="HEAD", metric_value=1.0, metric_name="val_bpb"),
        )
        coordinator.run()

        # Verify options were discovered
        dim = spec.dimensions[0]
        assert dim.options == ["adam", "sgd", "rmsprop"]

        # Verify event was logged
        events = coordinator.logger.read_events(event_type="diverse_discovered")
        assert len(events) == 1
        assert events[0]["dimension"] == "optimizer"
        assert events[0]["options"] == ["adam", "sgd", "rmsprop"]

    def test_skips_non_diverse_dims(self, tmp_output_dir):
        spec = _make_spec(
            dimensions=[
                DimensionSpec(name="lr", dim_type=DimensionType.DIRECTIONAL, current_value=0.04),
                DimensionSpec(name="act", dim_type=DimensionType.ENUM, options=["relu", "gelu"]),
            ],
            budget=BudgetConfig(max_experiments=1),
        )
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}]),
        ]
        results = {"exp-0-0": ExperimentResult(primary_metric=0.9)}
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="HEAD", metric_value=1.0, metric_name="val_bpb"),
        )
        coordinator.run()

        # No diverse_discovered events
        events = coordinator.logger.read_events(event_type="diverse_discovered")
        assert len(events) == 0

        # Dimensions unchanged
        assert spec.dimensions[0].options is None  # DIRECTIONAL has no options
        assert spec.dimensions[1].options == ["relu", "gelu"]  # ENUM kept its options

    def test_skips_diverse_with_existing_options(self, tmp_output_dir):
        spec = _make_spec(
            dimensions=[
                DimensionSpec(
                    name="optimizer", dim_type=DimensionType.DIVERSE,
                    options=["adam", "sgd"],  # already populated
                ),
            ],
            budget=BudgetConfig(max_experiments=1),
        )
        plans = [
            DimensionPlan(dimension_name="optimizer", values=[{"optimizer": "adam"}]),
        ]
        results = {"exp-0-0": ExperimentResult(primary_metric=0.9)}
        dm = ScriptedDecisionMaker(
            plans, diverse_options={"optimizer": ["adam", "sgd", "rmsprop", "adagrad"]},
        )
        coordinator = Coordinator(
            spec=spec,
            decision_maker=dm,
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="HEAD", metric_value=1.0, metric_name="val_bpb"),
        )
        coordinator.run()

        # Options should NOT have been overwritten
        assert spec.dimensions[0].options == ["adam", "sgd"]
        events = coordinator.logger.read_events(event_type="diverse_discovered")
        assert len(events) == 0

    def test_noop_when_no_diverse_dims(self, tmp_output_dir):
        spec = _make_spec(
            dimensions=[
                DimensionSpec(name="lr", dim_type=DimensionType.DIRECTIONAL, current_value=0.04),
            ],
            budget=BudgetConfig(max_experiments=1),
        )
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}]),
        ]
        results = {"exp-0-0": ExperimentResult(primary_metric=0.9)}
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="HEAD", metric_value=1.0, metric_name="val_bpb"),
        )
        coordinator.run()

        events = coordinator.logger.read_events(event_type="diverse_discovered")
        assert len(events) == 0
        events = coordinator.logger.read_events(event_type="diverse_discovery_failed")
        assert len(events) == 0

    def test_logs_failure_on_empty_options(self, tmp_output_dir):
        spec = _make_spec(
            dimensions=[
                DimensionSpec(name="optimizer", dim_type=DimensionType.DIVERSE),
            ],
            budget=BudgetConfig(max_experiments=1),
        )
        plans = [
            DimensionPlan(dimension_name="optimizer", values=[{"optimizer": "adam"}]),
        ]
        results = {"exp-0-0": ExperimentResult(primary_metric=0.9)}
        # Empty diverse_options for "optimizer" — ScriptedDecisionMaker returns []
        dm = ScriptedDecisionMaker(plans, diverse_options={})
        coordinator = Coordinator(
            spec=spec,
            decision_maker=dm,
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="HEAD", metric_value=1.0, metric_name="val_bpb"),
        )
        coordinator.run()

        # Options should remain None
        assert spec.dimensions[0].options is None

        # Failed event logged
        events = coordinator.logger.read_events(event_type="diverse_discovery_failed")
        assert len(events) == 1
        assert events[0]["dimension"] == "optimizer"

    def test_logs_failure_on_exception(self, tmp_output_dir):
        spec = _make_spec(
            dimensions=[
                DimensionSpec(name="optimizer", dim_type=DimensionType.DIVERSE),
                DimensionSpec(name="strategy", dim_type=DimensionType.DIVERSE),
            ],
            budget=BudgetConfig(max_experiments=1),
        )
        plans = [
            DimensionPlan(dimension_name="strategy", values=[{"strategy": "A"}]),
        ]
        results = {"exp-0-0": ExperimentResult(primary_metric=0.9)}

        class FailingDecisionMaker(ScriptedDecisionMaker):
            def discover_diverse_options(self, dimension_name, context):
                if dimension_name == "optimizer":
                    raise ValueError("LLM returned garbage")
                return ["A", "B"]

        dm = FailingDecisionMaker(plans)
        coordinator = Coordinator(
            spec=spec,
            decision_maker=dm,
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="HEAD", metric_value=1.0, metric_name="val_bpb"),
        )
        coordinator.run()

        # optimizer failed, strategy succeeded
        assert spec.dimensions[0].options is None
        assert spec.dimensions[1].options == ["A", "B"]

        failed = coordinator.logger.read_events(event_type="diverse_discovery_failed")
        assert len(failed) == 1
        assert failed[0]["dimension"] == "optimizer"

        discovered = coordinator.logger.read_events(event_type="diverse_discovered")
        assert len(discovered) == 1
        assert discovered[0]["dimension"] == "strategy"


class TestEnrichedEvents:
    """Verify events contain fields needed by build_snapshot()."""

    def _run_simple_coordinator(self, tmp_path, plans, results, budget_config):
        spec = WorkloadSpec(
            name="test", primary_metric="loss", metric_direction="lower",
            execution_command="echo 1", workers_available=2,
            budget=budget_config,
        )
        log_path = tmp_path / "events.jsonl"
        coord = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(log_path),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("abc", 3.0, "loss"),
        )
        coord.run()
        return EventLogger(log_path)

    def test_run_started_has_resume_fields(self, tmp_path):
        plans = [DimensionPlan("lr", [{"lr": 0.1}])]
        results = {"exp-0-0": ExperimentResult(primary_metric=2.0)}
        logger = self._run_simple_coordinator(
            tmp_path, plans, results, BudgetConfig(max_experiments=1))
        events = logger.read_events("run_started")
        e = events[0]
        assert "run_id" in e
        assert "mode" in e
        assert "metric_direction" in e
        assert "workload_spec_hash" in e

    def test_iteration_started_has_tasks(self, tmp_path):
        plans = [DimensionPlan("lr", [{"lr": 0.01}, {"lr": 0.1}])]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=2.0),
            "exp-0-1": ExperimentResult(primary_metric=2.5),
        }
        logger = self._run_simple_coordinator(
            tmp_path, plans, results, BudgetConfig(max_experiments=2))
        events = logger.read_events("iteration_started")
        tasks = events[0].get("tasks", [])
        assert len(tasks) == 2
        assert all("experiment_id" in t for t in tasks)
        assert all("params" in t for t in tasks)
        assert all("command" in t for t in tasks)
        assert all("baseline_commit" in t for t in tasks)

    def test_worker_completed_has_dimension_and_top_level_metric(self, tmp_path):
        plans = [DimensionPlan("lr", [{"lr": 0.1}])]
        results = {"exp-0-0": ExperimentResult(primary_metric=2.0, cost_usd=0.5)}
        logger = self._run_simple_coordinator(
            tmp_path, plans, results, BudgetConfig(max_experiments=1))
        events = logger.read_events("worker_completed")
        e = events[0]
        assert e["dimension"] == "lr"
        assert e["metric"] == 2.0
        assert e["cost_usd"] == 0.5

    def test_worker_failed_has_dimension_and_params(self, tmp_path):
        plans = [DimensionPlan("lr", [{"lr": 0.1}])]
        results = {"exp-0-0": ExperimentResult(primary_metric=0.0, error_message="OOM")}
        logger = self._run_simple_coordinator(
            tmp_path, plans, results, BudgetConfig(max_experiments=1))
        events = logger.read_events("worker_failed")
        e = events[0]
        assert e["dimension"] == "lr"
        assert "params" in e
        assert "cost_usd" in e

    def test_breakthrough_has_commit_and_metric(self, tmp_path):
        plans = [DimensionPlan("lr", [{"lr": 0.1}])]
        results = {"exp-0-0": ExperimentResult(primary_metric=2.0, commit_hash="newcommit")}
        logger = self._run_simple_coordinator(
            tmp_path, plans, results, BudgetConfig(max_experiments=1))
        events = logger.read_events("breakthrough")
        assert len(events) == 1
        assert events[0]["metric"] == 2.0
        assert events[0]["commit"] == "newcommit"
