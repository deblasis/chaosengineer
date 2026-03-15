"""Tests for RunSnapshot data models and build_snapshot replay."""

from __future__ import annotations

import json
from pathlib import Path

from chaosengineer.core.snapshot import (
    DimensionOutcome,
    ExperimentSummary,
    IncompleteIteration,
    RunSnapshot,
    StopReason,
    build_snapshot,
)
from chaosengineer.core.models import Baseline, BudgetConfig
from chaosengineer.core.interfaces import ExperimentTask


def _write_events(path: Path, events: list[dict]):
    """Write raw event dicts to a JSONL file."""
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


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


class TestBuildSnapshot:
    def test_completed_run(self, tmp_path):
        """Snapshot of a fully completed run."""
        events_path = tmp_path / "events.jsonl"
        _write_events(events_path, [
            {"event": "run_started", "run_id": "run-1", "workload": "test",
             "workload_spec_hash": "sha256:abc", "budget": {"max_experiments": 10},
             "mode": "parallel", "baseline": {"commit": "aaa", "metric_value": 3.0, "metric_name": "loss"},
             "ts": "2026-01-01T00:00:00Z"},
            {"event": "iteration_started", "iteration": 0, "dimension": "lr",
             "num_workers": 2, "tasks": [
                 {"experiment_id": "exp-0-0", "params": {"lr": 0.01}, "command": "echo", "baseline_commit": "aaa"},
                 {"experiment_id": "exp-0-1", "params": {"lr": 0.1}, "command": "echo", "baseline_commit": "aaa"},
             ], "ts": "2026-01-01T00:00:01Z"},
            {"event": "worker_completed", "experiment_id": "exp-0-0", "dimension": "lr",
             "params": {"lr": 0.01}, "metric": 2.5, "cost_usd": 0.50, "ts": "2026-01-01T00:00:10Z"},
            {"event": "worker_completed", "experiment_id": "exp-0-1", "dimension": "lr",
             "params": {"lr": 0.1}, "metric": 2.8, "cost_usd": 0.50, "ts": "2026-01-01T00:00:11Z"},
            {"event": "breakthrough", "metric": 2.5, "experiment_id": "exp-0-0",
             "commit": "bbb", "ts": "2026-01-01T00:00:12Z"},
            {"event": "budget_checkpoint", "spent_usd": 1.0, "experiments_run": 2,
             "ts": "2026-01-01T00:00:13Z"},
            {"event": "run_completed", "best_metric": 2.5, "total_experiments": 2,
             "total_cost_usd": 1.0, "ts": "2026-01-01T00:00:14Z"},
        ])

        snap = build_snapshot(events_path)
        assert snap.run_id == "run-1"
        assert snap.stop_reason == StopReason.COMPLETED
        assert len(snap.active_baselines) == 1
        assert snap.active_baselines[0].metric_value == 2.5
        assert snap.total_experiments_run == 2
        assert snap.total_cost_usd == 1.0
        assert len(snap.dimensions_explored) == 1
        assert snap.dimensions_explored[0].name == "lr"
        assert snap.incomplete_iteration is None

    def test_paused_run(self, tmp_path):
        """Snapshot of a run paused due to budget exhaustion."""
        events_path = tmp_path / "events.jsonl"
        _write_events(events_path, [
            {"event": "run_started", "run_id": "run-2", "workload": "test",
             "workload_spec_hash": "sha256:def", "budget": {"max_experiments": 2},
             "mode": "parallel", "baseline": {"commit": "aaa", "metric_value": 3.0, "metric_name": "loss"},
             "ts": "2026-01-01T00:00:00Z"},
            {"event": "iteration_started", "iteration": 0, "dimension": "lr",
             "num_workers": 2, "tasks": [
                 {"experiment_id": "exp-0-0", "params": {"lr": 0.01}, "command": "echo", "baseline_commit": "aaa"},
                 {"experiment_id": "exp-0-1", "params": {"lr": 0.1}, "command": "echo", "baseline_commit": "aaa"},
             ], "ts": "2026-01-01T00:00:01Z"},
            {"event": "worker_completed", "experiment_id": "exp-0-0", "dimension": "lr",
             "params": {"lr": 0.01}, "metric": 2.5, "cost_usd": 0.50, "ts": "2026-01-01T00:00:10Z"},
            {"event": "worker_completed", "experiment_id": "exp-0-1", "dimension": "lr",
             "params": {"lr": 0.1}, "metric": 2.8, "cost_usd": 0.50, "ts": "2026-01-01T00:00:11Z"},
            {"event": "breakthrough", "metric": 2.5, "experiment_id": "exp-0-0",
             "commit": "bbb", "ts": "2026-01-01T00:00:12Z"},
            {"event": "run_paused", "reason": "budget_exhausted", "last_iteration": 0,
             "budget_state": {"spent_usd": 1.0, "experiments_run": 2, "elapsed_seconds": 14},
             "active_baselines": [{"commit": "bbb", "metric_value": 2.5, "metric_name": "loss"}],
             "ts": "2026-01-01T00:00:14Z"},
        ])

        snap = build_snapshot(events_path)
        assert snap.stop_reason == StopReason.PAUSED
        assert snap.total_cost_usd == 1.0
        assert snap.total_experiments_run == 2

    def test_crashed_run_inferred(self, tmp_path):
        """No terminal event means crash is inferred."""
        events_path = tmp_path / "events.jsonl"
        _write_events(events_path, [
            {"event": "run_started", "run_id": "run-3", "workload": "test",
             "workload_spec_hash": "sha256:ghi", "budget": {"max_experiments": 10},
             "mode": "parallel", "baseline": {"commit": "aaa", "metric_value": 3.0, "metric_name": "loss"},
             "ts": "2026-01-01T00:00:00Z"},
            {"event": "iteration_started", "iteration": 0, "dimension": "lr",
             "num_workers": 3, "tasks": [
                 {"experiment_id": "exp-0-0", "params": {"lr": 0.01}, "command": "echo", "baseline_commit": "aaa"},
                 {"experiment_id": "exp-0-1", "params": {"lr": 0.1}, "command": "echo", "baseline_commit": "aaa"},
                 {"experiment_id": "exp-0-2", "params": {"lr": 1.0}, "command": "echo", "baseline_commit": "aaa"},
             ], "ts": "2026-01-01T00:00:01Z"},
            {"event": "worker_completed", "experiment_id": "exp-0-0", "dimension": "lr",
             "params": {"lr": 0.01}, "metric": 2.5, "cost_usd": 0.50, "ts": "2026-01-01T00:00:10Z"},
        ])

        snap = build_snapshot(events_path)
        assert snap.stop_reason == StopReason.CRASHED
        assert snap.incomplete_iteration is not None
        assert snap.incomplete_iteration.dimension == "lr"
        assert snap.incomplete_iteration.total_workers == 3
        assert len(snap.incomplete_iteration.completed_experiments) == 1
        assert len(snap.incomplete_iteration.missing_tasks) == 2
        assert snap.incomplete_iteration.missing_experiment_ids == ["exp-0-1", "exp-0-2"]

    def test_paused_run_with_partial_iteration(self, tmp_path):
        """Paused mid-iteration: some workers completed, some didn't."""
        events_path = tmp_path / "events.jsonl"
        _write_events(events_path, [
            {"event": "run_started", "run_id": "run-4", "workload": "test",
             "workload_spec_hash": "sha256:jkl", "budget": {"max_experiments": 10},
             "mode": "parallel", "baseline": {"commit": "aaa", "metric_value": 3.0, "metric_name": "loss"},
             "ts": "2026-01-01T00:00:00Z"},
            {"event": "iteration_started", "iteration": 0, "dimension": "lr",
             "num_workers": 3, "tasks": [
                 {"experiment_id": "exp-0-0", "params": {"lr": 0.01}, "command": "echo", "baseline_commit": "aaa"},
                 {"experiment_id": "exp-0-1", "params": {"lr": 0.1}, "command": "echo", "baseline_commit": "aaa"},
                 {"experiment_id": "exp-0-2", "params": {"lr": 1.0}, "command": "echo", "baseline_commit": "aaa"},
             ], "ts": "2026-01-01T00:00:01Z"},
            {"event": "worker_completed", "experiment_id": "exp-0-0", "dimension": "lr",
             "params": {"lr": 0.01}, "metric": 2.5, "cost_usd": 0.50, "ts": "2026-01-01T00:00:10Z"},
            {"event": "worker_completed", "experiment_id": "exp-0-1", "dimension": "lr",
             "params": {"lr": 0.1}, "metric": 2.8, "cost_usd": 0.50, "ts": "2026-01-01T00:00:11Z"},
            {"event": "run_paused", "reason": "user_interrupt", "last_iteration": 0,
             "budget_state": {"spent_usd": 1.0, "experiments_run": 2, "elapsed_seconds": 12},
             "active_baselines": [{"commit": "aaa", "metric_value": 3.0, "metric_name": "loss"}],
             "ts": "2026-01-01T00:00:12Z"},
        ])

        snap = build_snapshot(events_path)
        assert snap.stop_reason == StopReason.PAUSED
        assert snap.incomplete_iteration is not None
        assert snap.incomplete_iteration.total_workers == 3
        assert len(snap.incomplete_iteration.completed_experiments) == 2
        assert len(snap.incomplete_iteration.missing_tasks) == 1
        assert snap.incomplete_iteration.missing_experiment_ids == ["exp-0-2"]

    def test_diverse_dimensions_captured(self, tmp_path):
        """Discovered DIVERSE options are stored in snapshot."""
        events_path = tmp_path / "events.jsonl"
        _write_events(events_path, [
            {"event": "run_started", "run_id": "run-5", "workload": "test",
             "workload_spec_hash": "sha256:mno", "budget": {"max_experiments": 10},
             "mode": "parallel", "baseline": {"commit": "aaa", "metric_value": 3.0, "metric_name": "loss"},
             "ts": "2026-01-01T00:00:00Z"},
            {"event": "diverse_discovered", "dimension": "augmentation",
             "options": ["cutmix", "mixup", "randaugment"],
             "ts": "2026-01-01T00:00:05Z"},
            {"event": "run_completed", "best_metric": 3.0, "total_experiments": 0,
             "total_cost_usd": 0.0, "ts": "2026-01-01T00:00:10Z"},
        ])

        snap = build_snapshot(events_path)
        assert "augmentation" in snap.discovered_dimensions
        assert snap.discovered_dimensions["augmentation"] == ["cutmix", "mixup", "randaugment"]

    def test_history_reconstructed(self, tmp_path):
        """_history list reconstructed from worker events."""
        events_path = tmp_path / "events.jsonl"
        _write_events(events_path, [
            {"event": "run_started", "run_id": "run-6", "workload": "test",
             "workload_spec_hash": "sha256:pqr", "budget": {"max_experiments": 10},
             "mode": "parallel", "baseline": {"commit": "aaa", "metric_value": 3.0, "metric_name": "loss"},
             "ts": "2026-01-01T00:00:00Z"},
            {"event": "iteration_started", "iteration": 0, "dimension": "lr",
             "num_workers": 1, "tasks": [
                 {"experiment_id": "exp-0-0", "params": {"lr": 0.01}, "command": "echo", "baseline_commit": "aaa"},
             ], "ts": "2026-01-01T00:00:01Z"},
            {"event": "worker_completed", "experiment_id": "exp-0-0", "dimension": "lr",
             "params": {"lr": 0.01}, "metric": 2.5, "cost_usd": 0.50, "ts": "2026-01-01T00:00:10Z"},
            {"event": "run_completed", "best_metric": 2.5, "total_experiments": 1,
             "total_cost_usd": 0.5, "ts": "2026-01-01T00:00:12Z"},
        ])

        snap = build_snapshot(events_path)
        assert len(snap.history) >= 1
        assert any(h.get("dimension") == "lr" for h in snap.history)

    def test_resumed_run_replayed(self, tmp_path):
        """A previously resumed run replays correctly through both segments."""
        events_path = tmp_path / "events.jsonl"
        _write_events(events_path, [
            {"event": "run_started", "run_id": "run-7", "workload": "test",
             "workload_spec_hash": "sha256:stu", "budget": {"max_experiments": 10},
             "mode": "parallel", "baseline": {"commit": "aaa", "metric_value": 3.0, "metric_name": "loss"},
             "ts": "2026-01-01T00:00:00Z"},
            {"event": "iteration_started", "iteration": 0, "dimension": "lr",
             "num_workers": 1, "tasks": [
                 {"experiment_id": "exp-0-0", "params": {"lr": 0.01}, "command": "echo", "baseline_commit": "aaa"},
             ], "ts": "2026-01-01T00:00:01Z"},
            {"event": "worker_completed", "experiment_id": "exp-0-0", "dimension": "lr",
             "params": {"lr": 0.01}, "metric": 2.5, "cost_usd": 0.50, "ts": "2026-01-01T00:00:10Z"},
            {"event": "run_paused", "reason": "budget_exhausted", "last_iteration": 0,
             "budget_state": {"spent_usd": 0.5, "experiments_run": 1, "elapsed_seconds": 10},
             "active_baselines": [{"commit": "bbb", "metric_value": 2.5, "metric_name": "loss"}],
             "ts": "2026-01-01T00:00:10Z"},
            # --- resumed ---
            {"event": "run_resumed", "original_run_id": "run-7",
             "budget_extensions": {"add_experiments": 5},
             "ts": "2026-01-01T01:00:00Z"},
            {"event": "iteration_started", "iteration": 1, "dimension": "bs",
             "num_workers": 1, "tasks": [
                 {"experiment_id": "exp-1-0", "params": {"bs": 32}, "command": "echo", "baseline_commit": "bbb"},
             ], "ts": "2026-01-01T01:00:01Z"},
            {"event": "worker_completed", "experiment_id": "exp-1-0", "dimension": "bs",
             "params": {"bs": 32}, "metric": 2.3, "cost_usd": 0.60, "ts": "2026-01-01T01:00:10Z"},
            {"event": "breakthrough", "metric": 2.3, "experiment_id": "exp-1-0",
             "commit": "ccc", "ts": "2026-01-01T01:00:11Z"},
            {"event": "run_completed", "best_metric": 2.3, "total_experiments": 2,
             "total_cost_usd": 1.1, "ts": "2026-01-01T01:00:12Z"},
        ])

        snap = build_snapshot(events_path)
        assert snap.stop_reason == StopReason.COMPLETED
        assert snap.total_experiments_run == 2
        assert snap.total_cost_usd == 1.1
        assert len(snap.dimensions_explored) == 2
        assert snap.active_baselines[0].metric_value == 2.3

    def test_beam_search_multiple_baselines_via_run_paused(self, tmp_path):
        """Multiple active baselines from beam search survive snapshot round-trip."""
        events_path = tmp_path / "events.jsonl"
        _write_events(events_path, [
            {"event": "run_started", "run_id": "run-beam", "workload": "test",
             "workload_spec_hash": "sha256:abc", "budget": {"max_experiments": 10},
             "mode": "parallel", "baseline": {"commit": "aaa", "metric_value": 3.0, "metric_name": "loss"},
             "ts": "2026-01-01T00:00:00Z"},
            {"event": "iteration_started", "iteration": 0, "dimension": "lr",
             "num_workers": 2, "tasks": [
                 {"experiment_id": "exp-0-0", "params": {"lr": 0.01}, "command": "echo", "baseline_commit": "aaa"},
                 {"experiment_id": "exp-0-1", "params": {"lr": 0.1}, "command": "echo", "baseline_commit": "aaa"},
             ], "ts": "2026-01-01T00:00:01Z"},
            {"event": "worker_completed", "experiment_id": "exp-0-0", "dimension": "lr",
             "params": {"lr": 0.01}, "metric": 2.5, "cost_usd": 0.50, "ts": "2026-01-01T00:00:10Z"},
            {"event": "worker_completed", "experiment_id": "exp-0-1", "dimension": "lr",
             "params": {"lr": 0.1}, "metric": 2.51, "cost_usd": 0.50, "ts": "2026-01-01T00:00:11Z"},
            {"event": "breakthrough", "metric": 2.5, "experiment_id": "exp-0-0",
             "commit": "bbb", "ts": "2026-01-01T00:00:12Z"},
            {"event": "run_paused", "reason": "budget_exhausted", "last_iteration": 0,
             "budget_state": {"spent_usd": 1.0, "experiments_run": 2, "elapsed_seconds": 12},
             "active_baselines": [
                 {"commit": "bbb", "metric_value": 2.5, "metric_name": "loss"},
                 {"commit": "ccc", "metric_value": 2.51, "metric_name": "loss"},
             ],
             "ts": "2026-01-01T00:00:13Z"},
        ])

        snap = build_snapshot(events_path)
        assert snap.stop_reason == StopReason.PAUSED
        assert len(snap.active_baselines) == 2
        assert snap.active_baselines[0].metric_value == 2.5
        assert snap.active_baselines[1].metric_value == 2.51
        assert snap.active_baselines[0].commit == "bbb"
        assert snap.active_baselines[1].commit == "ccc"

    def test_workload_spec_hash_stored(self, tmp_path):
        """Workload spec hash from run_started is stored in snapshot."""
        events_path = tmp_path / "events.jsonl"
        _write_events(events_path, [
            {"event": "run_started", "run_id": "run-hash", "workload": "test",
             "workload_spec_hash": "sha256:deadbeef1234",
             "budget": {"max_experiments": 10}, "mode": "parallel",
             "baseline": {"commit": "aaa", "metric_value": 3.0, "metric_name": "loss"},
             "ts": "2026-01-01T00:00:00Z"},
            {"event": "run_completed", "best_metric": 3.0, "total_experiments": 0,
             "total_cost_usd": 0.0, "ts": "2026-01-01T00:00:01Z"},
        ])

        snap = build_snapshot(events_path)
        assert snap.workload_spec_hash == "sha256:deadbeef1234"
