"""Tests for RunSnapshot data models and build_snapshot replay."""

from __future__ import annotations

from chaosengineer.core.snapshot import (
    DimensionOutcome,
    ExperimentSummary,
    IncompleteIteration,
    RunSnapshot,
    StopReason,
)
from chaosengineer.core.models import Baseline, BudgetConfig
from chaosengineer.core.interfaces import ExperimentTask


class TestSnapshotDataModels:
    def test_stop_reason_values(self):
        assert StopReason.PAUSED.value == "paused"
        assert StopReason.COMPLETED.value == "completed"
        assert StopReason.CRASHED.value == "crashed"

    def test_experiment_summary_creation(self):
        s = ExperimentSummary(
            experiment_id="exp-0-0",
            dimension="learning_rate",
            params={"learning_rate": 0.001},
            metric=2.41,
            status="completed",
            cost_usd=0.50,
        )
        assert s.experiment_id == "exp-0-0"
        assert s.metric == 2.41

    def test_dimension_outcome_creation(self):
        d = DimensionOutcome(
            name="learning_rate",
            values_tested=["0.001", "0.01", "0.1"],
            winner="0.001",
            metric_improvement=0.12,
        )
        assert d.name == "learning_rate"
        assert d.winner == "0.001"

    def test_incomplete_iteration_creation(self):
        inc = IncompleteIteration(
            dimension="batch_size",
            total_workers=3,
            completed_experiments=[
                ExperimentSummary("exp-2-0", "batch_size", {"batch_size": 32}, 2.3, "completed", 0.5),
            ],
            missing_experiment_ids=["exp-2-1", "exp-2-2"],
            missing_tasks=[
                ExperimentTask("exp-2-1", {"batch_size": 64}, "python train.py", "abc123"),
                ExperimentTask("exp-2-2", {"batch_size": 128}, "python train.py", "abc123"),
            ],
        )
        assert inc.total_workers == 3
        assert len(inc.missing_tasks) == 2

    def test_run_snapshot_creation(self):
        snap = RunSnapshot(
            run_id="run-abc",
            workload_name="test",
            workload_spec_hash="sha256:abc",
            budget_config=BudgetConfig(max_experiments=10),
            mode="parallel",
            active_baselines=[Baseline("abc", 3.0, "loss")],
            baseline_history=[Baseline("abc", 3.0, "loss")],
            dimensions_explored=[],
            discovered_dimensions={},
            experiments=[],
            history=[],
            total_cost_usd=0.0,
            total_experiments_run=0,
            elapsed_time=0.0,
            consecutive_no_improvement=0,
            incomplete_iteration=None,
            stop_reason=StopReason.PAUSED,
        )
        assert snap.run_id == "run-abc"
        assert snap.stop_reason == StopReason.PAUSED
