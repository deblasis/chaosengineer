"""Tests for Coordinator.resume_from_snapshot()."""

from __future__ import annotations

from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.interfaces import DimensionPlan, ExperimentTask
from chaosengineer.core.models import (
    Baseline, BudgetConfig, ExperimentResult,
)
from chaosengineer.core.snapshot import (
    DimensionOutcome, ExperimentSummary, IncompleteIteration, RunSnapshot, StopReason,
)
from chaosengineer.metrics.logger import EventLogger
from chaosengineer.testing.executor import ScriptedExecutor
from chaosengineer.testing.simulator import ScriptedDecisionMaker
from chaosengineer.workloads.parser import WorkloadSpec


def _make_spec(budget: BudgetConfig) -> WorkloadSpec:
    return WorkloadSpec(
        name="test",
        primary_metric="loss",
        metric_direction="lower",
        execution_command="echo 1",
        workers_available=2,
        budget=budget,
    )


class TestResumeAtCleanBoundary:
    def test_resume_continues_from_explored_dimensions(self, tmp_path):
        """Resume after 1 dimension explored, coordinator picks next."""
        spec = _make_spec(BudgetConfig(max_experiments=10))

        plans = [DimensionPlan("bs", [{"bs": 32}, {"bs": 64}])]
        results = {
            "exp-1-0": ExperimentResult(primary_metric=2.0),
            "exp-1-1": ExperimentResult(primary_metric=2.2),
        }

        snapshot = RunSnapshot(
            run_id="run-resume-1",
            workload_name="test",
            workload_spec_hash="sha256:abc",
            budget_config=BudgetConfig(max_experiments=10),
            mode="parallel",
            active_baselines=[Baseline("bbb", 2.5, "loss")],
            baseline_history=[Baseline("aaa", 3.0, "loss"), Baseline("bbb", 2.5, "loss")],
            dimensions_explored=[
                DimensionOutcome("lr", ["0.01", "0.1"], "0.01", 0.5),
            ],
            discovered_dimensions={},
            experiments=[
                ExperimentSummary("exp-0-0", "lr", {"lr": 0.01}, 2.5, "completed", 0.5),
                ExperimentSummary("exp-0-1", "lr", {"lr": 0.1}, 2.8, "completed", 0.5),
            ],
            history=[
                {"experiment_id": "exp-0-0", "dimension": "lr", "params": {"lr": 0.01}, "metric": 2.5, "status": "completed"},
                {"experiment_id": "exp-0-1", "dimension": "lr", "params": {"lr": 0.1}, "metric": 2.8, "status": "completed"},
            ],
            total_cost_usd=1.0,
            total_experiments_run=2,
            elapsed_time=60.0,
            consecutive_no_improvement=0,
            incomplete_iteration=None,
            stop_reason=StopReason.PAUSED,
        )

        log_path = tmp_path / "events.jsonl"
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(log_path),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
        )

        coordinator.resume_from_snapshot(snapshot)

        events = EventLogger(log_path).read_events("iteration_started")
        assert len(events) >= 1
        assert coordinator.budget.experiments_run >= 4  # 2 prior + 2 new


class TestResumeWithPartialIteration:
    def test_completes_missing_workers(self, tmp_path):
        """Resume with 1/3 workers done, should run remaining 2."""
        spec = _make_spec(BudgetConfig(max_experiments=10))

        plans = []  # No new dimensions after gap completion
        results = {
            "exp-0-1": ExperimentResult(primary_metric=2.6),
            "exp-0-2": ExperimentResult(primary_metric=2.9),
        }

        snapshot = RunSnapshot(
            run_id="run-resume-2",
            workload_name="test",
            workload_spec_hash="sha256:abc",
            budget_config=BudgetConfig(max_experiments=10),
            mode="parallel",
            active_baselines=[Baseline("aaa", 3.0, "loss")],
            baseline_history=[Baseline("aaa", 3.0, "loss")],
            dimensions_explored=[],
            discovered_dimensions={},
            experiments=[
                ExperimentSummary("exp-0-0", "lr", {"lr": 0.01}, 2.5, "completed", 0.5),
            ],
            history=[
                {"experiment_id": "exp-0-0", "dimension": "lr", "params": {"lr": 0.01}, "metric": 2.5, "status": "completed"},
            ],
            total_cost_usd=0.5,
            total_experiments_run=1,
            elapsed_time=30.0,
            consecutive_no_improvement=0,
            incomplete_iteration=IncompleteIteration(
                dimension="lr",
                total_workers=3,
                completed_experiments=[
                    ExperimentSummary("exp-0-0", "lr", {"lr": 0.01}, 2.5, "completed", 0.5),
                ],
                missing_experiment_ids=["exp-0-1", "exp-0-2"],
                missing_tasks=[
                    ExperimentTask("exp-0-1", {"lr": 0.1}, "echo 1", "aaa"),
                    ExperimentTask("exp-0-2", {"lr": 1.0}, "echo 1", "aaa"),
                ],
            ),
            stop_reason=StopReason.PAUSED,
        )

        log_path = tmp_path / "events.jsonl"
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(log_path),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
        )

        coordinator.resume_from_snapshot(snapshot)

        gap_events = EventLogger(log_path).read_events("iteration_gap_completed")
        assert len(gap_events) == 1
        assert gap_events[0]["original_completed"] == 1
        assert gap_events[0]["gap_filled"] == 2


class TestExitBranching:
    def test_run_completed_when_dimensions_exhausted(self, tmp_path):
        """When decision maker returns None (no more dims), emit run_completed."""
        spec = _make_spec(BudgetConfig(max_experiments=10))
        plans = []  # No dimensions available
        results = {}

        log_path = tmp_path / "events.jsonl"
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(log_path),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
        )

        coordinator.run()

        completed = EventLogger(log_path).read_events("run_completed")
        paused = EventLogger(log_path).read_events("run_paused")
        assert len(completed) == 1
        assert len(paused) == 0

    def test_run_paused_when_budget_exhausted(self, tmp_path):
        """When budget runs out mid-run, emit run_paused."""
        spec = _make_spec(BudgetConfig(max_experiments=2))
        plans = [
            DimensionPlan("lr", [{"lr": 0.01}, {"lr": 0.1}]),
            DimensionPlan("bs", [{"bs": 32}]),  # Should NOT be reached
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=2.5),
            "exp-0-1": ExperimentResult(primary_metric=2.8),
        }

        log_path = tmp_path / "events.jsonl"
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(log_path),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
        )

        coordinator.run()

        completed = EventLogger(log_path).read_events("run_completed")
        paused = EventLogger(log_path).read_events("run_paused")
        assert len(completed) == 0
        assert len(paused) == 1
        assert paused[0]["reason"] == "budget_exhausted"
