"""Tests for SubagentExecutor."""

import json
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY, call

import pytest

from chaosengineer.core.interfaces import ExperimentTask
from chaosengineer.core.models import ExperimentResult
from chaosengineer.execution.subagent import SubagentExecutor
from chaosengineer.execution.worktree import WorktreeManager
from chaosengineer.workloads.parser import WorkloadSpec


def _make_spec(**overrides) -> WorkloadSpec:
    defaults = dict(
        name="test",
        primary_metric="val_bpb",
        metric_direction="lower",
        execution_command="python train.py",
        time_budget_seconds=300,
    )
    defaults.update(overrides)
    return WorkloadSpec(**defaults)


def _git_success(*args, **kwargs):
    """Return success for any git worktree command."""
    return subprocess.CompletedProcess(args=[], returncode=0)


_RESULT_STDOUT = json.dumps({
    "type": "result",
    "subtype": "success",
    "total_cost_usd": 0.42,
    "usage": {"input_tokens": 5000, "output_tokens": 200},
})


def _make_popen_mock(stdout="", stderr="", returncode=0, side_effect=None):
    """Build a mock Popen instance whose communicate() returns (stdout, stderr)."""
    proc = MagicMock()
    proc.returncode = returncode
    if side_effect is not None:
        proc.communicate.side_effect = side_effect
    else:
        proc.communicate.return_value = (stdout, stderr)
    proc.__enter__ = lambda s: s
    proc.__exit__ = MagicMock(return_value=False)
    return proc


class TestSubagentRunExperiment:
    """Test single experiment execution with mocked subprocess."""

    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_successful_experiment(self, mock_popen, mock_run, tmp_path):
        def git_dispatch(cmd, *args, **kwargs):
            return subprocess.CompletedProcess(args=[], returncode=0)

        mock_run.side_effect = git_dispatch

        result_dir = tmp_path / "output" / "exp-0-0"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "result.json").write_text(json.dumps({
            "primary_metric": 0.91,
            "commit_hash": "def456",
        }))
        mock_popen.return_value = _make_popen_mock(stdout="", stderr="", returncode=0)

        spec = _make_spec()
        executor = SubagentExecutor(spec, tmp_path / "output", "sequential", run_id="run-test", repo_root=tmp_path)

        result = executor.run_experiment(
            experiment_id="exp-0-0",
            params={"lr": 0.02},
            command="python train.py",
            baseline_commit="abc123",
        )

        assert result.primary_metric == 0.91
        assert result.commit_hash == "def456"
        assert result.error_message is None
        assert result.duration_seconds > 0

    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_subprocess_failure_returns_error(self, mock_popen, mock_run, tmp_path):
        mock_run.side_effect = lambda cmd, *a, **kw: subprocess.CompletedProcess(args=[], returncode=0)
        mock_popen.return_value = _make_popen_mock(stdout="", stderr="claude error", returncode=1)

        spec = _make_spec()
        executor = SubagentExecutor(spec, tmp_path / "output", "sequential", run_id="run-test", repo_root=tmp_path)

        result = executor.run_experiment(
            experiment_id="exp-0-0",
            params={"lr": 0.02},
            command="python train.py",
            baseline_commit="abc123",
        )

        assert result.error_message is not None
        assert "failed" in result.error_message.lower() or "exit" in result.error_message.lower()

    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_timeout_returns_error(self, mock_popen, mock_run, tmp_path):
        mock_run.side_effect = lambda cmd, *a, **kw: subprocess.CompletedProcess(args=[], returncode=0)

        proc = _make_popen_mock(side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=360))
        proc.wait.return_value = None
        mock_popen.return_value = proc

        spec = _make_spec(time_budget_seconds=300)
        executor = SubagentExecutor(spec, tmp_path / "output", "sequential", run_id="run-test", repo_root=tmp_path)

        result = executor.run_experiment(
            experiment_id="exp-0-0",
            params={"lr": 0.02},
            command="python train.py",
            baseline_commit="abc123",
        )

        assert result.error_message is not None
        assert "timed out" in result.error_message.lower()

    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_no_timeout_when_budget_is_none(self, mock_popen, mock_run, tmp_path):
        mock_run.side_effect = lambda cmd, *a, **kw: subprocess.CompletedProcess(args=[], returncode=0)
        mock_popen.return_value = _make_popen_mock(stdout="", stderr="", returncode=0)

        spec = _make_spec(time_budget_seconds=None)
        executor = SubagentExecutor(spec, tmp_path / "output", "sequential", run_id="run-test", repo_root=tmp_path)

        executor.run_experiment(
            experiment_id="exp-0-0",
            params={"lr": 0.02},
            command="python train.py",
            baseline_commit="abc123",
        )

        # communicate() should have been called with timeout=None
        proc = mock_popen.return_value
        proc.communicate.assert_called_once_with(timeout=None)

    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_successful_experiment_has_cost(self, mock_popen, mock_run, tmp_path):
        mock_run.side_effect = lambda cmd, *a, **kw: subprocess.CompletedProcess(args=[], returncode=0)

        result_dir = tmp_path / "output" / "exp-0-0"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "result.json").write_text(json.dumps({"primary_metric": 0.91}))
        mock_popen.return_value = _make_popen_mock(stdout=_RESULT_STDOUT, stderr="", returncode=0)

        spec = _make_spec()
        executor = SubagentExecutor(spec, tmp_path / "output", "sequential", run_id="run-test", repo_root=tmp_path)

        result = executor.run_experiment(
            experiment_id="exp-0-0",
            params={"lr": 0.02},
            command="python train.py",
            baseline_commit="abc123",
        )

        assert result.cost_usd == 0.42
        assert result.tokens_in == 5000
        assert result.tokens_out == 200

    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_failed_experiment_still_has_cost(self, mock_popen, mock_run, tmp_path):
        mock_run.side_effect = lambda cmd, *a, **kw: subprocess.CompletedProcess(args=[], returncode=0)
        mock_popen.return_value = _make_popen_mock(stdout=_RESULT_STDOUT, stderr="error", returncode=1)

        spec = _make_spec()
        executor = SubagentExecutor(spec, tmp_path / "output", "sequential", run_id="run-test", repo_root=tmp_path)

        result = executor.run_experiment(
            experiment_id="exp-0-0",
            params={"lr": 0.02},
            command="python train.py",
            baseline_commit="abc123",
        )

        assert result.error_message is not None
        assert result.cost_usd == 0.42

    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_missing_result_file_still_has_cost(self, mock_popen, mock_run, tmp_path):
        """Subagent ran (incurred cost) but failed to write result.json."""
        mock_run.side_effect = lambda cmd, *a, **kw: subprocess.CompletedProcess(args=[], returncode=0)
        # Don't write result.json — subagent failed to produce output
        mock_popen.return_value = _make_popen_mock(stdout=_RESULT_STDOUT, stderr="", returncode=0)

        spec = _make_spec()
        executor = SubagentExecutor(spec, tmp_path / "output", "sequential", run_id="run-test", repo_root=tmp_path)

        result = executor.run_experiment(
            experiment_id="exp-0-0",
            params={"lr": 0.02},
            command="python train.py",
            baseline_commit="abc123",
        )

        assert result.error_message is not None  # ResultParser returns error
        assert result.cost_usd == 0.42  # But cost is still tracked


class TestSubagentRunExperiments:
    """Test batch execution."""

    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_sequential_batch(self, mock_popen, mock_run, tmp_path):
        call_count = 0

        def popen_dispatch(cmd, *args, **kwargs):
            nonlocal call_count
            exp_id = f"exp-0-{call_count}"
            result_dir = tmp_path / "output" / exp_id
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "result.json").write_text(json.dumps({
                "primary_metric": 0.9 + call_count * 0.01,
            }))
            call_count += 1
            return _make_popen_mock(stdout="", stderr="", returncode=0)

        mock_run.side_effect = lambda cmd, *a, **kw: subprocess.CompletedProcess(args=[], returncode=0)
        mock_popen.side_effect = popen_dispatch

        spec = _make_spec()
        executor = SubagentExecutor(spec, tmp_path / "output", "sequential", run_id="run-test", repo_root=tmp_path)

        tasks = [
            ExperimentTask("exp-0-0", {"lr": 0.02}, "echo", "abc"),
            ExperimentTask("exp-0-1", {"lr": 0.08}, "echo", "abc"),
        ]
        results = executor.run_experiments(tasks)

        assert len(results) == 2
        assert all(r.error_message is None or r.primary_metric > 0 for r in results)

    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_parallel_batch(self, mock_popen, mock_run, tmp_path):
        def popen_dispatch(cmd, *args, **kwargs):
            prompt = cmd[2] if len(cmd) > 2 else ""
            for exp_id in ["exp-0-0", "exp-0-1"]:
                if exp_id in prompt:
                    result_dir = tmp_path / "output" / exp_id
                    result_dir.mkdir(parents=True, exist_ok=True)
                    (result_dir / "result.json").write_text(json.dumps({
                        "primary_metric": 0.91,
                    }))
                    break
            return _make_popen_mock(stdout="", stderr="", returncode=0)

        mock_run.side_effect = lambda cmd, *a, **kw: subprocess.CompletedProcess(args=[], returncode=0)
        mock_popen.side_effect = popen_dispatch

        spec = _make_spec()
        executor = SubagentExecutor(spec, tmp_path / "output", "parallel", run_id="run-test", repo_root=tmp_path)

        tasks = [
            ExperimentTask("exp-0-0", {"lr": 0.02}, "echo", "abc"),
            ExperimentTask("exp-0-1", {"lr": 0.08}, "echo", "abc"),
        ]
        results = executor.run_experiments(tasks)

        assert len(results) == 2

    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_thread_crash_returns_error_result(self, mock_popen, mock_run, tmp_path):
        mock_run.side_effect = lambda cmd, *a, **kw: subprocess.CompletedProcess(args=[], returncode=0)
        mock_popen.side_effect = RuntimeError("unexpected crash")

        spec = _make_spec()
        executor = SubagentExecutor(spec, tmp_path / "output", "parallel", run_id="run-test", repo_root=tmp_path)

        tasks = [ExperimentTask("exp-0-0", {"lr": 0.02}, "echo", "abc")]
        results = executor.run_experiments(tasks)

        assert len(results) == 1
        assert results[0].error_message is not None


class TestKillActive:
    def test_kill_active_terminates_processes(self):
        """kill_active() calls terminate() on all stored processes."""
        from chaosengineer.execution.subagent import SubagentExecutor

        executor = MagicMock(spec=SubagentExecutor)
        executor._active_processes = []
        executor._process_lock = threading.Lock()

        proc1 = MagicMock()
        proc2 = MagicMock()
        executor._active_processes = [proc1, proc2]

        SubagentExecutor.kill_active(executor)

        proc1.terminate.assert_called_once()
        proc2.terminate.assert_called_once()

    def test_kill_active_empty_is_noop(self):
        """kill_active() with no active processes does nothing."""
        from chaosengineer.execution.subagent import SubagentExecutor

        executor = MagicMock(spec=SubagentExecutor)
        executor._active_processes = []
        executor._process_lock = threading.Lock()

        SubagentExecutor.kill_active(executor)  # Should not raise


class TestSubagentResourceHandling:
    @patch("subprocess.run")
    @patch("subprocess.Popen")
    def test_gpu_resource_sets_env(self, mock_popen, mock_run, tmp_path):
        popen_kwargs = {}

        def popen_dispatch(cmd, *args, **kwargs):
            popen_kwargs.update(kwargs)
            return _make_popen_mock(stdout="", stderr="", returncode=0)

        mock_run.side_effect = lambda cmd, *a, **kw: subprocess.CompletedProcess(args=[], returncode=0)
        mock_popen.side_effect = popen_dispatch

        spec = _make_spec()
        executor = SubagentExecutor(spec, tmp_path / "output", "sequential", run_id="run-test", repo_root=tmp_path)

        executor.run_experiment(
            experiment_id="exp-0-0",
            params={"lr": 0.02},
            command="echo",
            baseline_commit="abc",
            resource="gpu:2",
        )

        assert popen_kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2"
