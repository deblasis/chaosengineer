"""Tests for Coordinator + PauseController integration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.interfaces import DimensionPlan, ExperimentTask
from chaosengineer.core.models import (
    Baseline, BudgetConfig, ExperimentResult,
)
from chaosengineer.core.pause import PauseController
from chaosengineer.core.status import StatusDisplay
from chaosengineer.metrics.logger import EventLogger
from chaosengineer.testing.executor import ScriptedExecutor
from chaosengineer.testing.simulator import ScriptedDecisionMaker
from chaosengineer.workloads.parser import WorkloadSpec


def _make_spec(budget: BudgetConfig | None = None) -> WorkloadSpec:
    return WorkloadSpec(
        name="test",
        primary_metric="loss",
        metric_direction="lower",
        execution_command="echo 1",
        workers_available=2,
        budget=budget or BudgetConfig(max_experiments=10),
    )


class TestPauseBeforeIteration:
    def test_pause_requested_logs_run_paused(self, tmp_path):
        """When pause_requested before iteration, log run_paused with user_requested."""
        spec = _make_spec()
        plans = [DimensionPlan("lr", [{"lr": 0.01}, {"lr": 0.1}])]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=2.5),
            "exp-0-1": ExperimentResult(primary_metric=2.8),
        }

        pc = PauseController()
        pc.pause_requested = True
        pc.show_post_iteration_menu = MagicMock(return_value="pause")

        log_path = tmp_path / "events.jsonl"
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(log_path),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
            pause_controller=pc,
        )

        coordinator.run()

        paused = EventLogger(log_path).read_events("run_paused")
        assert len(paused) == 1
        assert paused[0]["reason"] == "user_requested"

    def test_continue_resumes_normally(self, tmp_path):
        """When user picks 'continue', run proceeds."""
        spec = _make_spec(BudgetConfig(max_experiments=2))
        plans = [DimensionPlan("lr", [{"lr": 0.01}, {"lr": 0.1}])]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=2.5),
            "exp-0-1": ExperimentResult(primary_metric=2.8),
        }

        pc = PauseController()
        pc.pause_requested = True
        pc.show_post_iteration_menu = MagicMock(return_value="continue")

        log_path = tmp_path / "events.jsonl"
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(log_path),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
            pause_controller=pc,
        )

        coordinator.run()

        assert coordinator.budget.experiments_run == 2


class TestPauseAfterIteration:
    def test_wait_then_ask_shows_menu_after_iteration(self, tmp_path):
        """wait_then_ask causes menu to appear after iteration completes."""
        spec = _make_spec(BudgetConfig(max_experiments=4))
        plans = [
            DimensionPlan("lr", [{"lr": 0.01}, {"lr": 0.1}]),
            DimensionPlan("bs", [{"bs": 32}]),
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=2.5),
            "exp-0-1": ExperimentResult(primary_metric=2.8),
            "exp-1-0": ExperimentResult(primary_metric=2.3),
        }

        pc = PauseController()
        pc.wait_then_ask = True
        pc.show_post_iteration_menu = MagicMock(return_value="pause")

        log_path = tmp_path / "events.jsonl"
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(log_path),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
            pause_controller=pc,
        )

        coordinator.run()

        paused = EventLogger(log_path).read_events("run_paused")
        assert len(paused) == 1
        assert paused[0]["reason"] == "user_requested"
        assert coordinator.budget.experiments_run == 2


class TestPauseDuringResume:
    def test_pause_works_during_resumed_run(self, tmp_path):
        """Pause checks work in _run_loop() regardless of entry point."""
        from chaosengineer.core.snapshot import (
            RunSnapshot, StopReason, DimensionOutcome, ExperimentSummary,
        )
        spec = _make_spec(BudgetConfig(max_experiments=10))
        plans = [DimensionPlan("bs", [{"bs": 32}])]
        results = {"exp-1-0": ExperimentResult(primary_metric=2.0)}

        pc = PauseController()
        pc.pause_requested = True
        pc.show_post_iteration_menu = MagicMock(return_value="pause")

        snapshot = RunSnapshot(
            run_id="run-pause-resume",
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
            ],
            history=[],
            total_cost_usd=0.5,
            total_experiments_run=1,
            elapsed_time=30.0,
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
            pause_controller=pc,
        )

        coordinator.resume_from_snapshot(snapshot)

        paused = EventLogger(log_path).read_events("run_paused")
        assert len(paused) == 1
        assert paused[0]["reason"] == "user_requested"


class TestNoPauseController:
    def test_none_pause_controller_works(self, tmp_path):
        """Coordinator without pause_controller runs normally."""
        spec = _make_spec(BudgetConfig(max_experiments=2))
        plans = [DimensionPlan("lr", [{"lr": 0.01}, {"lr": 0.1}])]
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
        assert coordinator.budget.experiments_run == 2


class TestKillResumeRoundTrip:
    def test_kill_mid_iteration_produces_resumable_snapshot(self, tmp_path):
        """Kill mid-iteration → kill_issued auto-pauses → snapshot → resume completes it."""
        from chaosengineer.core.snapshot import build_snapshot, StopReason

        spec = _make_spec(BudgetConfig(max_experiments=10))

        # First run: 2 experiments planned, second fails (simulating kill)
        plans_run1 = [DimensionPlan("lr", [{"lr": 0.01}, {"lr": 0.1}])]
        results_run1 = {
            "exp-0-0": ExperimentResult(primary_metric=2.5),
            "exp-0-1": ExperimentResult(primary_metric=0.0, error_message="killed"),
        }

        pc = PauseController()
        # Mock mid-iteration menu to return "kill"
        pc.show_mid_iteration_menu = MagicMock(return_value="kill")

        # Simulate Ctrl+C arriving after the first worker completes: use a
        # ScriptedExecutor subclass that sets pause_requested after exp-0-0.
        class KillAfterFirstExecutor(ScriptedExecutor):
            def run_experiment(self, experiment_id, params, command, baseline_commit, resource=""):
                result = super().run_experiment(experiment_id, params, command, baseline_commit, resource)
                if experiment_id == "exp-0-0":
                    pc.pause_requested = True
                return result

        log_path = tmp_path / "events.jsonl"
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans_run1),
            executor=KillAfterFirstExecutor(results_run1),
            logger=EventLogger(log_path),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
            pause_controller=pc,
        )

        coordinator.run()

        # Verify kill_issued caused auto-pause
        assert pc.kill_issued is True

        # Verify run_paused event exists
        paused = EventLogger(log_path).read_events("run_paused")
        assert len(paused) == 1
        assert paused[0]["reason"] == "user_requested"

        # Build snapshot and verify it's resumable
        snapshot = build_snapshot(log_path)
        assert snapshot.stop_reason == StopReason.PAUSED

        # Resume: complete the missing work
        plans_run2 = []  # No new dims after gap fill
        results_run2 = {
            "exp-0-1": ExperimentResult(primary_metric=2.6),
        }

        log_path2 = tmp_path / "events2.jsonl"
        import shutil
        shutil.copy(log_path, log_path2)

        coordinator2 = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans_run2),
            executor=ScriptedExecutor(results_run2),
            logger=EventLogger(log_path2),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
        )
        coordinator2.resume_from_snapshot(snapshot)
