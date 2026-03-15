"""Tests for TaskPacketBuilder."""

import pytest
from pathlib import Path

from chaosengineer.core.interfaces import ExperimentTask
from chaosengineer.execution.task_packet import TaskPacketBuilder
from chaosengineer.workloads.parser import WorkloadSpec


def _make_spec(**overrides) -> WorkloadSpec:
    defaults = dict(
        name="test",
        primary_metric="val_bpb",
        metric_direction="lower",
        execution_command="python train.py",
        modifiable_files=["train.py", "config.yaml"],
        constraints_text="Do not modify prepare.py",
    )
    defaults.update(overrides)
    return WorkloadSpec(**defaults)


class TestTaskPacketBuild:
    def test_writes_task_file(self, tmp_path):
        spec = _make_spec()
        task = ExperimentTask("exp-0-0", {"lr": 0.02}, "python train.py", "abc123")
        builder = TaskPacketBuilder()

        result_file = tmp_path / "result.json"
        worktree_path = Path("/tmp/worktrees/exp-0-0")

        task_file = builder.build(
            task, spec, worktree_path, result_file,
            run_id="run-abcd1234",
            output_dir=tmp_path,
        )

        assert task_file.exists()
        content = task_file.read_text()

        # Verify key sections present
        assert "# Experiment: exp-0-0" in content
        assert "lr: 0.02" in content or "lr" in content
        assert "python train.py" in content
        assert "val_bpb" in content
        assert "lower" in content
        assert "train.py" in content
        assert "config.yaml" in content
        assert "Do not modify prepare.py" in content
        assert str(result_file) in content

    def test_includes_metric_parse_command(self, tmp_path):
        spec = _make_spec(metric_parse_command='grep "val_bpb" output.log')
        task = ExperimentTask("exp-0-0", {"lr": 0.02}, "python train.py", "abc123")
        builder = TaskPacketBuilder()

        task_file = builder.build(
            task, spec, Path("/tmp/wt"), tmp_path / "result.json",
            run_id="run-x", output_dir=tmp_path,
        )
        content = task_file.read_text()

        assert 'grep "val_bpb" output.log' in content

    def test_no_modifiable_files_says_any(self, tmp_path):
        spec = _make_spec(modifiable_files=[])
        task = ExperimentTask("exp-0-0", {"lr": 0.02}, "echo", "abc123")
        builder = TaskPacketBuilder()

        task_file = builder.build(
            task, spec, Path("/tmp/wt"), tmp_path / "result.json",
            run_id="run-x", output_dir=tmp_path,
        )
        content = task_file.read_text()

        assert "Any files" in content

    def test_no_constraints_says_none(self, tmp_path):
        spec = _make_spec(constraints_text="")
        task = ExperimentTask("exp-0-0", {"lr": 0.02}, "echo", "abc123")
        builder = TaskPacketBuilder()

        task_file = builder.build(
            task, spec, Path("/tmp/wt"), tmp_path / "result.json",
            run_id="run-x", output_dir=tmp_path,
        )
        content = task_file.read_text()

        assert "None" in content

    def test_time_budget_included_when_set(self, tmp_path):
        spec = _make_spec(time_budget_seconds=300)
        task = ExperimentTask("exp-0-0", {"lr": 0.02}, "echo", "abc123")
        builder = TaskPacketBuilder()

        task_file = builder.build(
            task, spec, Path("/tmp/wt"), tmp_path / "result.json",
            run_id="run-x", output_dir=tmp_path,
        )
        content = task_file.read_text()

        assert "300" in content

    def test_no_time_limit_when_none(self, tmp_path):
        spec = _make_spec(time_budget_seconds=None)
        task = ExperimentTask("exp-0-0", {"lr": 0.02}, "echo", "abc123")
        builder = TaskPacketBuilder()

        task_file = builder.build(
            task, spec, Path("/tmp/wt"), tmp_path / "result.json",
            run_id="run-x", output_dir=tmp_path,
        )
        content = task_file.read_text()

        assert "No time limit" in content

    def test_branch_name_includes_run_id(self, tmp_path):
        spec = _make_spec()
        task = ExperimentTask("exp-0-0", {"lr": 0.02}, "echo", "abc123")
        builder = TaskPacketBuilder()

        task_file = builder.build(
            task, spec, Path("/tmp/wt"), tmp_path / "result.json",
            run_id="run-abcd1234", output_dir=tmp_path,
        )
        content = task_file.read_text()

        assert "chaosengineer/run-abcd1234/exp-0-0" in content
