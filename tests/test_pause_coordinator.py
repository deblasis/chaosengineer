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
