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


class TestRestartIteration:
    def test_restart_iteration_discards_partial_and_re_picks(self, tmp_path):
        """--restart-iteration discards partial results, LLM re-picks dimension."""
        spec = _make_spec(BudgetConfig(max_experiments=10))

        # LLM will pick lr again (or a different one), then run 2 experiments
        plans = [DimensionPlan("lr", [{"lr": 0.01}, {"lr": 0.1}])]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=2.3),
            "exp-0-1": ExperimentResult(primary_metric=2.6),
        }

        snapshot = RunSnapshot(
            run_id="run-restart",
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

        coordinator.resume_from_snapshot(snapshot, restart_iteration=True)

        # Should NOT have iteration_gap_completed (partial was discarded)
        gap_events = EventLogger(log_path).read_events("iteration_gap_completed")
        assert len(gap_events) == 0

        # Should have run_resumed with restart_iteration=True
        resumed = EventLogger(log_path).read_events("run_resumed")
        assert len(resumed) == 1
        assert resumed[0]["restart_iteration"] is True

        # LLM picked lr dimension fresh and ran 2 experiments
        iter_events = EventLogger(log_path).read_events("iteration_started")
        assert len(iter_events) >= 1


class TestBeamSearchStateRoundTrip:
    def test_multiple_active_baselines_survive_resume(self, tmp_path):
        """Multiple active baselines (beam search ties) survive snapshot and resume."""
        spec = _make_spec(BudgetConfig(max_experiments=10))

        # LLM picks next dimension with 2 experiments
        plans = [DimensionPlan("dropout", [{"dropout": 0.1}, {"dropout": 0.3}])]
        results = {
            "exp-1-0": ExperimentResult(primary_metric=2.0),
            "exp-1-1": ExperimentResult(primary_metric=2.2),
        }

        # Snapshot has 2 active baselines from a beam search tie
        snapshot = RunSnapshot(
            run_id="run-beam",
            workload_name="test",
            workload_spec_hash="sha256:abc",
            budget_config=BudgetConfig(max_experiments=10),
            mode="parallel",
            active_baselines=[
                Baseline("bbb", 2.5, "loss", branch_id="exp-0-0"),
                Baseline("ccc", 2.51, "loss", branch_id="exp-0-1"),
            ],
            baseline_history=[
                Baseline("aaa", 3.0, "loss"),
                Baseline("bbb", 2.5, "loss"),
                Baseline("ccc", 2.51, "loss"),
            ],
            dimensions_explored=[
                DimensionOutcome("lr", ["0.01", "0.1"], "0.01", 0.5),
            ],
            discovered_dimensions={},
            experiments=[
                ExperimentSummary("exp-0-0", "lr", {"lr": 0.01}, 2.5, "completed", 0.5),
                ExperimentSummary("exp-0-1", "lr", {"lr": 0.1}, 2.51, "completed", 0.5),
            ],
            history=[
                {"experiment_id": "exp-0-0", "dimension": "lr", "params": {"lr": 0.01}, "metric": 2.5, "status": "completed"},
                {"experiment_id": "exp-0-1", "dimension": "lr", "params": {"lr": 0.1}, "metric": 2.51, "status": "completed"},
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

        # After resume, the loop ran dropout experiments (2.0, 2.2).
        # Since metric_direction="lower", 2.0 beats the snapshot baseline of 2.5.
        assert coordinator.best_baseline.metric_value == 2.0

        # Verify the resumed event recorded the snapshot state
        resumed = EventLogger(log_path).read_events("run_resumed")
        assert len(resumed) == 1
        assert resumed[0]["snapshot_summary"]["experiments_completed"] == 2

        # The new iteration should have run (proves baselines were restored)
        iter_events = EventLogger(log_path).read_events("iteration_started")
        assert len(iter_events) >= 1
