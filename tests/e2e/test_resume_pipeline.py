# tests/e2e/test_resume_pipeline.py
"""End-to-end test: run a scenario, pause it, resume it."""

from __future__ import annotations

import json
from pathlib import Path

from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.interfaces import DimensionPlan
from chaosengineer.core.models import Baseline, BudgetConfig, ExperimentResult
from chaosengineer.core.snapshot import build_snapshot, StopReason
from chaosengineer.metrics.logger import EventLogger
from chaosengineer.testing.executor import ScriptedExecutor
from chaosengineer.testing.simulator import ScriptedDecisionMaker
from chaosengineer.workloads.parser import WorkloadSpec


class TestResumePipeline:
    def test_pause_and_resume_full_cycle(self, tmp_path):
        """Run 1 dimension, hit budget, resume with more budget, run 2nd dimension."""
        spec = WorkloadSpec(
            name="resume-e2e",
            primary_metric="loss",
            metric_direction="lower",
            execution_command="echo 1",
            workers_available=2,
            budget=BudgetConfig(max_experiments=2),
        )

        # Phase 1: Run with budget for only 2 experiments
        plans_phase1 = [
            DimensionPlan("lr", [{"lr": 0.01}, {"lr": 0.1}]),
            DimensionPlan("bs", [{"bs": 32}, {"bs": 64}]),  # Won't reach this
        ]
        results_phase1 = {
            "exp-0-0": ExperimentResult(primary_metric=2.5),
            "exp-0-1": ExperimentResult(primary_metric=2.8),
        }

        log_path = tmp_path / "events.jsonl"
        coord = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans_phase1),
            executor=ScriptedExecutor(results_phase1),
            logger=EventLogger(log_path),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
        )
        coord.run()

        # Verify phase 1 paused (not completed)
        snapshot = build_snapshot(log_path)
        assert snapshot.stop_reason == StopReason.PAUSED
        assert snapshot.total_experiments_run == 2
        assert len(snapshot.dimensions_explored) == 1

        # Phase 2: Resume with extended budget
        snapshot.budget_config = BudgetConfig(max_experiments=4)

        plans_phase2 = [
            DimensionPlan("bs", [{"bs": 32}, {"bs": 64}]),
        ]
        results_phase2 = {
            "exp-1-0": ExperimentResult(primary_metric=2.0),
            "exp-1-1": ExperimentResult(primary_metric=2.3),
        }

        coord2 = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans_phase2),
            executor=ScriptedExecutor(results_phase2),
            logger=EventLogger(log_path),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
        )
        coord2.resume_from_snapshot(snapshot)

        # Verify phase 2 completed
        final_snapshot = build_snapshot(log_path)
        assert final_snapshot.total_experiments_run == 4
        assert len(final_snapshot.dimensions_explored) == 2

        # Verify events timeline
        logger = EventLogger(log_path)
        resumed = logger.read_events("run_resumed")
        assert len(resumed) == 1

    def test_resume_with_partial_iteration(self, tmp_path):
        """Run stops mid-iteration, resume completes missing workers."""
        # Write a crafted event log simulating a partial stop
        log_path = tmp_path / "events.jsonl"
        events = [
            {"event": "run_started", "run_id": "run-partial", "workload": "test",
             "workload_spec_hash": "sha256:abc", "budget": {"max_experiments": 10},
             "mode": "parallel", "baseline": {"commit": "aaa", "metric_value": 3.0, "metric_name": "loss"},
             "ts": "2026-01-01T00:00:00Z"},
            {"event": "iteration_started", "iteration": 0, "dimension": "lr",
             "num_workers": 3, "tasks": [
                 {"experiment_id": "exp-0-0", "params": {"lr": 0.01}, "command": "echo 1", "baseline_commit": "aaa"},
                 {"experiment_id": "exp-0-1", "params": {"lr": 0.1}, "command": "echo 1", "baseline_commit": "aaa"},
                 {"experiment_id": "exp-0-2", "params": {"lr": 1.0}, "command": "echo 1", "baseline_commit": "aaa"},
             ], "ts": "2026-01-01T00:00:01Z"},
            {"event": "worker_completed", "experiment_id": "exp-0-0", "dimension": "lr",
             "params": {"lr": 0.01}, "metric": 2.5, "cost_usd": 0.50, "ts": "2026-01-01T00:00:10Z"},
            {"event": "run_paused", "reason": "user_interrupt", "last_iteration": 0,
             "budget_state": {"spent_usd": 0.5, "experiments_run": 1, "elapsed_seconds": 10},
             "active_baselines": [{"commit": "aaa", "metric_value": 3.0, "metric_name": "loss"}],
             "ts": "2026-01-01T00:00:11Z"},
        ]
        with open(log_path, "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        snapshot = build_snapshot(log_path)
        assert snapshot.incomplete_iteration is not None
        assert len(snapshot.incomplete_iteration.missing_tasks) == 2

        # Resume: provide results for the missing workers
        spec = WorkloadSpec(
            name="test", primary_metric="loss", metric_direction="lower",
            execution_command="echo 1", workers_available=3,
            budget=BudgetConfig(max_experiments=10),
        )
        results = {
            "exp-0-1": ExperimentResult(primary_metric=2.6),
            "exp-0-2": ExperimentResult(primary_metric=2.9),
        }

        coord = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker([]),
            executor=ScriptedExecutor(results),
            logger=EventLogger(log_path),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
        )
        coord.resume_from_snapshot(snapshot)

        # Check gap was completed
        logger = EventLogger(log_path)
        gap_events = logger.read_events("iteration_gap_completed")
        assert len(gap_events) == 1
        assert gap_events[0]["gap_filled"] == 2
