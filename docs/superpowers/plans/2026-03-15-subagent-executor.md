# SubagentExecutor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `SubagentExecutor` — a real experiment executor that spawns Claude Code subagents in isolated git worktrees, plus the batch execution API, coordinator integration, and CLI wiring.

**Architecture:** Layered executor with four components (TaskPacketBuilder, WorktreeManager, ResultParser, SubagentExecutor) in a new `chaosengineer/execution/` package. The `ExperimentExecutor` ABC gains a batch `run_experiments()` method. The coordinator's `_run_iteration` is refactored into prepare/execute/handle phases. CLI gets `--executor` and `--mode` flags.

**Tech Stack:** Python 3.10+, subprocess (for `claude -p`), concurrent.futures.ThreadPoolExecutor, git worktrees, PyYAML, pytest.

**Spec:** `docs/superpowers/specs/2026-03-15-subagent-executor-design.md`

---

## Chunk 1: ABC Changes + Parser Fix

### Task 1: Add ExperimentTask dataclass and batch method to ExperimentExecutor ABC

**Files:**
- Modify: `chaosengineer/core/interfaces.py`
- Test: `tests/test_interfaces.py`

- [ ] **Step 1: Write failing tests for ExperimentTask and run_experiments**

Add to `tests/test_interfaces.py`:

```python
from chaosengineer.core.interfaces import (
    DecisionMaker,
    DimensionPlan,
    ExperimentExecutor,
    ExperimentTask,
)
from chaosengineer.core.models import ExperimentResult


class TestExperimentTask:
    def test_creation(self):
        task = ExperimentTask(
            experiment_id="exp-0-0",
            params={"lr": 0.02},
            command="echo test",
            baseline_commit="abc123",
        )
        assert task.experiment_id == "exp-0-0"
        assert task.params == {"lr": 0.02}
        assert task.command == "echo test"
        assert task.baseline_commit == "abc123"
        assert task.resource == ""

    def test_creation_with_resource(self):
        task = ExperimentTask(
            experiment_id="exp-0-0",
            params={"lr": 0.02},
            command="echo test",
            baseline_commit="abc123",
            resource="gpu:0",
        )
        assert task.resource == "gpu:0"


class TestRunExperimentsDefault:
    """Test that the default run_experiments calls run_experiment sequentially."""

    def test_default_sequential_batch(self):
        from chaosengineer.testing.executor import ScriptedExecutor

        results_map = {
            "exp-0-0": ExperimentResult(primary_metric=0.91),
            "exp-0-1": ExperimentResult(primary_metric=0.95),
        }
        executor = ScriptedExecutor(results_map)

        tasks = [
            ExperimentTask("exp-0-0", {"lr": 0.02}, "echo", "abc"),
            ExperimentTask("exp-0-1", {"lr": 0.08}, "echo", "abc"),
        ]
        results = executor.run_experiments(tasks)

        assert len(results) == 2
        assert results[0].primary_metric == 0.91
        assert results[1].primary_metric == 0.95
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_interfaces.py -v`
Expected: ImportError for `ExperimentTask`, AttributeError for `run_experiments`

- [ ] **Step 3: Add ExperimentTask and run_experiments to interfaces.py**

In `chaosengineer/core/interfaces.py`, add the `ExperimentTask` dataclass after `DimensionPlan` and add the `run_experiments` method to `ExperimentExecutor`:

```python
@dataclass
class ExperimentTask:
    """Input packet for a single experiment."""
    experiment_id: str
    params: dict[str, Any]
    command: str
    baseline_commit: str
    resource: str = ""


class ExperimentExecutor(ABC):
    """Interface for running experiments.

    In real mode, this runs commands in worktrees. In test mode, returns
    scripted results instantly.
    """

    @abstractmethod
    def run_experiment(
        self,
        experiment_id: str,
        params: dict[str, Any],
        command: str,
        baseline_commit: str,
        resource: str = "",
    ) -> ExperimentResult:
        """Run an experiment and return its result."""

    def run_experiments(self, tasks: list[ExperimentTask]) -> list[ExperimentResult]:
        """Run a batch of experiments. Default: sequential.

        Postcondition: always returns exactly one ExperimentResult per input task.
        Failures are captured as ExperimentResult with error_message set, never raised.
        """
        results = []
        for t in tasks:
            try:
                results.append(
                    self.run_experiment(
                        t.experiment_id, t.params, t.command, t.baseline_commit, t.resource
                    )
                )
            except Exception as e:
                results.append(ExperimentResult(
                    primary_metric=0.0,
                    error_message=f"Executor error for {t.experiment_id}: {e}",
                ))
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_interfaces.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/ -v --ignore=tests/test_runner.py --ignore=tests/test_shipped_scenarios.py`
Expected: All 117 tests PASS (ignoring the 2 with yaml collection errors)

- [ ] **Step 6: Commit**

```bash
git add chaosengineer/core/interfaces.py tests/test_interfaces.py
git commit -m "feat: add ExperimentTask dataclass and batch run_experiments to ExperimentExecutor ABC"
```

### Task 2: Change time_budget_seconds default from 300 to None

**Files:**
- Modify: `chaosengineer/workloads/parser.py`
- Modify: `tests/test_parser.py`

- [ ] **Step 1: Update test to verify None default when Time budget absent**

In `tests/test_parser.py`, add a test and update the existing one:

```python
class TestTimeBudgetDefault:
    def test_absent_time_budget_is_none(self):
        md = """# Workload: No Budget
## Experiment Space
## Execution
- Command: `echo`
## Evaluation
- Metric: score (lower is better)
## Resources
- Available: 1
"""
        spec = parse_workload_spec(content=md)
        assert spec.time_budget_seconds is None

    def test_explicit_time_budget_is_parsed(self):
        md = """# Workload: With Budget
## Experiment Space
## Execution
- Command: `echo`
- Time budget per experiment: 5 minutes
## Evaluation
- Metric: score (lower is better)
## Resources
- Available: 1
"""
        spec = parse_workload_spec(content=md)
        assert spec.time_budget_seconds == 300
```

The existing `test_parse_execution` test at line 36 already asserts `== 300` for the sample workload which has an explicit "Time budget: 5 minutes" — it will keep passing.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_parser.py::TestTimeBudgetDefault -v`
Expected: FAIL — `test_absent_time_budget_is_none` gets 300.0 instead of None

- [ ] **Step 3: Update parser to return None when Time budget is absent**

In `chaosengineer/workloads/parser.py`:

Change `WorkloadSpec` field:
```python
    time_budget_seconds: float | None = None
```

Change `_parse_time_budget` to return `None` instead of `300`:
```python
def _parse_time_budget(text: str) -> float | None:
    """Parse time budget, returns seconds or None if absent."""
    match = re.search(r"Time budget.*?:\s*(\d+)\s*(minutes?|seconds?|hours?)", text, re.IGNORECASE)
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("hour"):
        return value * 3600
    if unit.startswith("minute"):
        return value * 60
    return float(value)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_parser.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/ -v --ignore=tests/test_runner.py --ignore=tests/test_shipped_scenarios.py`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add chaosengineer/workloads/parser.py tests/test_parser.py
git commit -m "feat: change time_budget_seconds default from 300 to None for unlimited experiments"
```

## Chunk 2: Execution Package — ResultParser and WorktreeManager

### Task 3: Create ResultParser

**Files:**
- Create: `chaosengineer/execution/__init__.py` (empty for now)
- Create: `chaosengineer/execution/result_parser.py`
- Create: `tests/test_result_parser.py`

- [ ] **Step 1: Create the execution package with empty __init__.py**

```bash
mkdir -p chaosengineer/execution
```

Write `chaosengineer/execution/__init__.py`:
```python
"""Experiment execution backends."""
```

- [ ] **Step 2: Write failing tests for ResultParser**

Create `tests/test_result_parser.py`:

```python
"""Tests for ResultParser."""

import json
import pytest
from pathlib import Path

from chaosengineer.execution.result_parser import ResultParser


class TestResultParserValidJSON:
    def test_parse_complete_result(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps({
            "primary_metric": 0.91,
            "secondary_metrics": {"train_loss": 1.5},
            "artifacts": ["model.pt"],
            "commit_hash": "abc123",
            "error_message": None,
        }))

        parser = ResultParser()
        result = parser.parse(result_file, "exp-0-0", duration_seconds=42.5)

        assert result.primary_metric == 0.91
        assert result.secondary_metrics == {"train_loss": 1.5}
        assert result.artifacts == ["model.pt"]
        assert result.commit_hash == "abc123"
        assert result.error_message is None
        assert result.duration_seconds == 42.5

    def test_parse_minimal_result(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps({"primary_metric": 0.91}))

        parser = ResultParser()
        result = parser.parse(result_file, "exp-0-0", duration_seconds=10.0)

        assert result.primary_metric == 0.91
        assert result.secondary_metrics == {}
        assert result.artifacts == []
        assert result.commit_hash is None
        assert result.error_message is None

    def test_parse_error_result(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps({
            "primary_metric": 0.0,
            "error_message": "OOM during training",
        }))

        parser = ResultParser()
        result = parser.parse(result_file, "exp-0-0", duration_seconds=5.0)

        assert result.error_message == "OOM during training"

    def test_extra_fields_ignored(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps({
            "primary_metric": 0.91,
            "some_unknown_field": "ignored",
        }))

        parser = ResultParser()
        result = parser.parse(result_file, "exp-0-0", duration_seconds=1.0)

        assert result.primary_metric == 0.91


class TestResultParserErrors:
    def test_missing_file_returns_error_result(self, tmp_path):
        result_file = tmp_path / "nonexistent.json"

        parser = ResultParser()
        result = parser.parse(result_file, "exp-0-0", duration_seconds=0.0)

        assert result.primary_metric == 0.0
        assert result.error_message is not None
        assert "not found" in result.error_message.lower() or "missing" in result.error_message.lower()

    def test_malformed_json_returns_error_result(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text("not valid json {{{")

        parser = ResultParser()
        result = parser.parse(result_file, "exp-0-0", duration_seconds=0.0)

        assert result.primary_metric == 0.0
        assert result.error_message is not None

    def test_missing_primary_metric_returns_error_result(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps({"secondary_metrics": {}}))

        parser = ResultParser()
        result = parser.parse(result_file, "exp-0-0", duration_seconds=0.0)

        assert result.primary_metric == 0.0
        assert result.error_message is not None
        assert "primary_metric" in result.error_message.lower()

    def test_primary_metric_not_numeric_returns_error(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps({"primary_metric": "not a number"}))

        parser = ResultParser()
        result = parser.parse(result_file, "exp-0-0", duration_seconds=0.0)

        assert result.error_message is not None
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_result_parser.py -v`
Expected: ImportError for `ResultParser`

- [ ] **Step 4: Implement ResultParser**

Create `chaosengineer/execution/result_parser.py`:

```python
"""Reads and validates experiment result JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from chaosengineer.core.models import ExperimentResult


class ResultParser:
    """Reads result.json, validates, and builds ExperimentResult."""

    def parse(
        self,
        result_file: Path,
        experiment_id: str,
        duration_seconds: float,
    ) -> ExperimentResult:
        """Parse a result file into an ExperimentResult.

        Returns an error ExperimentResult if the file is missing, malformed,
        or missing required fields. Never raises.
        """
        if not result_file.exists():
            return ExperimentResult(
                primary_metric=0.0,
                duration_seconds=duration_seconds,
                error_message=f"Result file not found for {experiment_id}: {result_file}",
            )

        try:
            raw = result_file.read_text()
            data = json.loads(raw)
        except (json.JSONDecodeError, OSError) as e:
            return ExperimentResult(
                primary_metric=0.0,
                duration_seconds=duration_seconds,
                error_message=f"Failed to parse result for {experiment_id}: {e}",
            )

        if "primary_metric" not in data:
            return ExperimentResult(
                primary_metric=0.0,
                duration_seconds=duration_seconds,
                error_message=f"Missing primary_metric in result for {experiment_id}",
            )

        try:
            primary = float(data["primary_metric"])
        except (TypeError, ValueError) as e:
            return ExperimentResult(
                primary_metric=0.0,
                duration_seconds=duration_seconds,
                error_message=f"Invalid primary_metric for {experiment_id}: {e}",
            )

        return ExperimentResult(
            primary_metric=primary,
            secondary_metrics=data.get("secondary_metrics", {}),
            artifacts=data.get("artifacts", []),
            commit_hash=data.get("commit_hash"),
            duration_seconds=duration_seconds,
            error_message=data.get("error_message"),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_result_parser.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add chaosengineer/execution/__init__.py chaosengineer/execution/result_parser.py tests/test_result_parser.py
git commit -m "feat: add ResultParser for experiment result JSON validation"
```

### Task 4: Create WorktreeManager

**Files:**
- Create: `chaosengineer/execution/worktree.py`
- Create: `tests/test_worktree.py`

- [ ] **Step 1: Write failing tests for WorktreeManager**

Create `tests/test_worktree.py`:

```python
"""Tests for WorktreeManager."""

import subprocess
from pathlib import Path
from unittest.mock import patch, call

import pytest

from chaosengineer.execution.worktree import WorktreeManager


class TestWorktreeCreate:
    @patch("chaosengineer.execution.worktree.subprocess.run")
    def test_creates_worktree_with_branch(self, mock_run, tmp_path):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        mgr = WorktreeManager(repo_root=repo_root)

        path = mgr.create("abc123", "run-abcd1234", "exp-0-0")

        assert path == repo_root / ".chaosengineer" / "worktrees" / "exp-0-0"
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "worktree" in cmd
        assert "add" in cmd
        assert "-b" in cmd
        assert "chaosengineer/run-abcd1234/exp-0-0" in cmd
        assert "abc123" in cmd
        # git commands run from repo root
        assert mock_run.call_args[1].get("cwd") == str(repo_root)

    @patch("chaosengineer.execution.worktree.subprocess.run")
    def test_create_raises_on_git_failure(self, mock_run, tmp_path):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stderr="fatal: error"
        )
        mgr = WorktreeManager(repo_root=tmp_path)

        with pytest.raises(RuntimeError, match="Failed to create worktree"):
            mgr.create("abc123", "run-abcd1234", "exp-0-0")


class TestWorktreeCleanup:
    @patch("chaosengineer.execution.worktree.subprocess.run")
    def test_removes_worktree(self, mock_run, tmp_path):
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        mgr = WorktreeManager(repo_root=tmp_path)
        worktree_path = tmp_path / ".chaosengineer" / "worktrees" / "exp-0-0"

        mgr.cleanup(worktree_path)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert "worktree" in cmd
        assert "remove" in cmd
        assert str(worktree_path) in cmd

    @patch("chaosengineer.execution.worktree.subprocess.run")
    def test_cleanup_ignores_missing_worktree(self, mock_run, tmp_path):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stderr="not a working tree"
        )
        mgr = WorktreeManager(repo_root=tmp_path)
        worktree_path = tmp_path / ".chaosengineer" / "worktrees" / "exp-0-0"

        # Should not raise
        mgr.cleanup(worktree_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_worktree.py -v`
Expected: ImportError for `WorktreeManager`

- [ ] **Step 3: Implement WorktreeManager**

Create `chaosengineer/execution/worktree.py`:

```python
"""Git worktree lifecycle management for experiment isolation."""

from __future__ import annotations

import subprocess
from pathlib import Path


class WorktreeManager:
    """Creates and cleans up git worktrees for experiment isolation."""

    def __init__(self, repo_root: Path):
        self._repo_root = repo_root
        self._worktrees_dir = repo_root / ".chaosengineer" / "worktrees"

    def create(
        self, baseline_commit: str, run_id: str, experiment_id: str
    ) -> Path:
        """Create a worktree with a named branch.

        Returns the worktree path.
        """
        worktree_path = self._worktrees_dir / experiment_id
        branch_name = f"chaosengineer/{run_id}/{experiment_id}"

        self._worktrees_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [
                "git", "worktree", "add",
                str(worktree_path),
                "-b", branch_name,
                baseline_commit,
            ],
            capture_output=True,
            text=True,
            cwd=str(self._repo_root),
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to create worktree for {experiment_id}: {result.stderr}"
            )

        return worktree_path

    def cleanup(self, worktree_path: Path) -> None:
        """Remove a worktree. The branch persists."""
        subprocess.run(
            ["git", "worktree", "remove", str(worktree_path), "--force"],
            capture_output=True,
            text=True,
            cwd=str(self._repo_root),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_worktree.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/execution/worktree.py tests/test_worktree.py
git commit -m "feat: add WorktreeManager for git worktree lifecycle"
```

### Task 5: Create TaskPacketBuilder

**Files:**
- Create: `chaosengineer/execution/task_packet.py`
- Create: `tests/test_task_packet.py`

- [ ] **Step 1: Write failing tests for TaskPacketBuilder**

Create `tests/test_task_packet.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_task_packet.py -v`
Expected: ImportError for `TaskPacketBuilder`

- [ ] **Step 3: Implement TaskPacketBuilder**

Create `chaosengineer/execution/task_packet.py`:

```python
"""Constructs task markdown files for Claude Code subagents."""

from __future__ import annotations

from pathlib import Path

from chaosengineer.core.interfaces import ExperimentTask
from chaosengineer.workloads.parser import WorkloadSpec


class TaskPacketBuilder:
    """Generates task markdown files that instruct subagents."""

    def build(
        self,
        task: ExperimentTask,
        spec: WorkloadSpec,
        worktree_path: Path,
        result_file: Path,
        run_id: str,
        output_dir: Path,
    ) -> Path:
        """Build a task markdown file and write it to the output directory.

        Returns the path to the written task file.
        """
        params_block = "\n".join(
            f"  {k}: {v}" for k, v in task.params.items()
        )

        if spec.modifiable_files:
            files_block = "\n".join(f"- {f}" for f in spec.modifiable_files)
        else:
            files_block = "Any files"

        constraints_block = spec.constraints_text if spec.constraints_text else "None"

        if spec.metric_parse_command:
            metric_step = f'5. Parse metrics by running: `{spec.metric_parse_command}`'
        else:
            metric_step = f"5. Extract the primary metric '{spec.primary_metric}' from the command output"

        if spec.time_budget_seconds is not None:
            time_block = f"Complete within {spec.time_budget_seconds} seconds"
        else:
            time_block = "No time limit — run to completion"

        branch_name = f"chaosengineer/{run_id}/{task.experiment_id}"
        param_summary = ", ".join(f"{k}={v}" for k, v in task.params.items())

        content = f"""# Experiment: {task.experiment_id}

## Objective
Apply the following parameter changes and run the experiment command.
Report the results as a JSON file.

## Parameters
{params_block}

## Working Directory
You are working in a git worktree at: {worktree_path}
Branch: {branch_name}

## Modifiable Files
{files_block}

## Constraints
{constraints_block}

## Instructions
1. Study the codebase to understand how the parameters above map to code
2. Modify the relevant files to apply these parameter values
3. Commit your changes with message: "experiment {task.experiment_id}: {param_summary}"
4. Run the experiment command: `{task.command}`
{metric_step}
6. Write results to: {result_file}

## Time Budget
{time_block}

## Result Format
Write ONLY this JSON to {result_file}:

```json
{{
  "primary_metric": <float>,
  "secondary_metrics": {{"metric_name": <float>, ...}},
  "artifacts": ["path/to/artifact", ...],
  "commit_hash": "<git commit hash of your changes>",
  "error_message": null
}}
```

If the experiment fails, set error_message to a description of what went wrong.
Primary metric name: {spec.primary_metric} ({spec.metric_direction} is better)
"""
        task_file = output_dir / task.experiment_id / "task.md"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text(content)
        return task_file
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_task_packet.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/execution/task_packet.py tests/test_task_packet.py
git commit -m "feat: add TaskPacketBuilder for subagent task markdown generation"
```

## Chunk 3: SubagentExecutor + Factory

### Task 6: Create SubagentExecutor

**Files:**
- Create: `chaosengineer/execution/subagent.py`
- Create: `tests/test_subagent_executor.py`

- [ ] **Step 1: Write failing tests for SubagentExecutor**

Create `tests/test_subagent_executor.py`:

```python
"""Tests for SubagentExecutor."""

import json
import subprocess
import time
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

import pytest

from chaosengineer.core.interfaces import ExperimentTask
from chaosengineer.core.models import ExperimentResult
from chaosengineer.execution.subagent import SubagentExecutor
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


class TestSubagentRunExperiment:
    """Test single experiment execution with mocked subprocess."""

    @patch("chaosengineer.execution.subagent.subprocess.run")
    @patch("chaosengineer.execution.worktree.subprocess.run")
    def test_successful_experiment(self, mock_wt_run, mock_sub_run, tmp_path):
        # Mock worktree creation and cleanup
        mock_wt_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

        # Mock claude subprocess — write result file as side effect
        def write_result(*args, **kwargs):
            # Find the output dir and write result.json
            result_dir = tmp_path / "output" / "exp-0-0"
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "result.json").write_text(json.dumps({
                "primary_metric": 0.91,
                "commit_hash": "def456",
            }))
            return subprocess.CompletedProcess(args=[], returncode=0)

        mock_sub_run.side_effect = write_result

        spec = _make_spec()
        executor = SubagentExecutor(spec, tmp_path / "output", "sequential", run_id="run-test")

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

    @patch("chaosengineer.execution.subagent.subprocess.run")
    @patch("chaosengineer.execution.worktree.subprocess.run")
    def test_subprocess_failure_returns_error(self, mock_wt_run, mock_sub_run, tmp_path):
        mock_wt_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        mock_sub_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=1, stderr="claude error"
        )

        spec = _make_spec()
        executor = SubagentExecutor(spec, tmp_path / "output", "sequential", run_id="run-test")

        result = executor.run_experiment(
            experiment_id="exp-0-0",
            params={"lr": 0.02},
            command="python train.py",
            baseline_commit="abc123",
        )

        assert result.error_message is not None
        assert "failed" in result.error_message.lower() or "exit" in result.error_message.lower()

    @patch("chaosengineer.execution.subagent.subprocess.run")
    @patch("chaosengineer.execution.worktree.subprocess.run")
    def test_timeout_returns_error(self, mock_wt_run, mock_sub_run, tmp_path):
        mock_wt_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        mock_sub_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=360)

        spec = _make_spec(time_budget_seconds=300)
        executor = SubagentExecutor(spec, tmp_path / "output", "sequential", run_id="run-test")

        result = executor.run_experiment(
            experiment_id="exp-0-0",
            params={"lr": 0.02},
            command="python train.py",
            baseline_commit="abc123",
        )

        assert result.error_message is not None
        assert "timeout" in result.error_message.lower()

    @patch("chaosengineer.execution.subagent.subprocess.run")
    @patch("chaosengineer.execution.worktree.subprocess.run")
    def test_no_timeout_when_budget_is_none(self, mock_wt_run, mock_sub_run, tmp_path):
        mock_wt_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        mock_sub_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

        spec = _make_spec(time_budget_seconds=None)
        executor = SubagentExecutor(spec, tmp_path / "output", "sequential", run_id="run-test")

        executor.run_experiment(
            experiment_id="exp-0-0",
            params={"lr": 0.02},
            command="python train.py",
            baseline_commit="abc123",
        )

        # Check that subprocess.run was called with timeout=None
        call_kwargs = mock_sub_run.call_args[1]
        assert call_kwargs.get("timeout") is None


class TestSubagentRunExperiments:
    """Test batch execution."""

    @patch("chaosengineer.execution.subagent.subprocess.run")
    @patch("chaosengineer.execution.worktree.subprocess.run")
    def test_sequential_batch(self, mock_wt_run, mock_sub_run, tmp_path):
        mock_wt_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

        call_count = 0

        def write_result(*args, **kwargs):
            nonlocal call_count
            exp_id = f"exp-0-{call_count}"
            result_dir = tmp_path / "output" / exp_id
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "result.json").write_text(json.dumps({
                "primary_metric": 0.9 + call_count * 0.01,
            }))
            call_count += 1
            return subprocess.CompletedProcess(args=[], returncode=0)

        mock_sub_run.side_effect = write_result

        spec = _make_spec()
        executor = SubagentExecutor(spec, tmp_path / "output", "sequential", run_id="run-test")

        tasks = [
            ExperimentTask("exp-0-0", {"lr": 0.02}, "echo", "abc"),
            ExperimentTask("exp-0-1", {"lr": 0.08}, "echo", "abc"),
        ]
        results = executor.run_experiments(tasks)

        assert len(results) == 2
        assert all(r.error_message is None or r.primary_metric > 0 for r in results)

    @patch("chaosengineer.execution.subagent.subprocess.run")
    @patch("chaosengineer.execution.worktree.subprocess.run")
    def test_parallel_batch(self, mock_wt_run, mock_sub_run, tmp_path):
        mock_wt_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

        def write_result(*args, **kwargs):
            # Extract experiment_id from the prompt text
            prompt = args[0][2] if len(args[0]) > 2 else ""
            for exp_id in ["exp-0-0", "exp-0-1"]:
                if exp_id in prompt:
                    result_dir = tmp_path / "output" / exp_id
                    result_dir.mkdir(parents=True, exist_ok=True)
                    (result_dir / "result.json").write_text(json.dumps({
                        "primary_metric": 0.91,
                    }))
                    break
            return subprocess.CompletedProcess(args=[], returncode=0)

        mock_sub_run.side_effect = write_result

        spec = _make_spec()
        executor = SubagentExecutor(spec, tmp_path / "output", "parallel", run_id="run-test")

        tasks = [
            ExperimentTask("exp-0-0", {"lr": 0.02}, "echo", "abc"),
            ExperimentTask("exp-0-1", {"lr": 0.08}, "echo", "abc"),
        ]
        results = executor.run_experiments(tasks)

        assert len(results) == 2

    @patch("chaosengineer.execution.subagent.subprocess.run")
    @patch("chaosengineer.execution.worktree.subprocess.run")
    def test_thread_crash_returns_error_result(self, mock_wt_run, mock_sub_run, tmp_path):
        mock_wt_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        mock_sub_run.side_effect = RuntimeError("unexpected crash")

        spec = _make_spec()
        executor = SubagentExecutor(spec, tmp_path / "output", "parallel", run_id="run-test")

        tasks = [ExperimentTask("exp-0-0", {"lr": 0.02}, "echo", "abc")]
        results = executor.run_experiments(tasks)

        assert len(results) == 1
        assert results[0].error_message is not None


class TestSubagentResourceHandling:
    @patch("chaosengineer.execution.subagent.subprocess.run")
    @patch("chaosengineer.execution.worktree.subprocess.run")
    def test_gpu_resource_sets_env(self, mock_wt_run, mock_sub_run, tmp_path):
        mock_wt_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        mock_sub_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

        spec = _make_spec()
        executor = SubagentExecutor(spec, tmp_path / "output", "sequential", run_id="run-test")

        executor.run_experiment(
            experiment_id="exp-0-0",
            params={"lr": 0.02},
            command="echo",
            baseline_commit="abc",
            resource="gpu:2",
        )

        call_kwargs = mock_sub_run.call_args[1]
        assert call_kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "2"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_subagent_executor.py -v`
Expected: ImportError for `SubagentExecutor`

- [ ] **Step 3: Implement SubagentExecutor**

Create `chaosengineer/execution/subagent.py`:

```python
"""SubagentExecutor — spawns Claude Code subagents for real experiments."""

from __future__ import annotations

import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from chaosengineer.core.interfaces import ExperimentExecutor, ExperimentTask
from chaosengineer.core.models import ExperimentResult
from chaosengineer.execution.result_parser import ResultParser
from chaosengineer.execution.task_packet import TaskPacketBuilder
from chaosengineer.execution.worktree import WorktreeManager
from chaosengineer.workloads.parser import WorkloadSpec

_GRACE_SECONDS = 60


def _get_repo_root() -> Path:
    """Find the git repo root from CWD."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return Path.cwd()
    return Path(result.stdout.strip())


class SubagentExecutor(ExperimentExecutor):
    """Runs experiments by spawning Claude Code subagents in git worktrees."""

    def __init__(
        self,
        spec: WorkloadSpec,
        output_dir: Path,
        mode: str = "sequential",
        run_id: str = "run-unknown",
    ):
        self._spec = spec
        self._output_dir = output_dir
        self._mode = mode
        self._run_id = run_id
        self._worktree_mgr = WorktreeManager(repo_root=_get_repo_root())
        self._task_builder = TaskPacketBuilder()
        self._result_parser = ResultParser()

    def run_experiment(
        self,
        experiment_id: str,
        params: dict[str, Any],
        command: str,
        baseline_commit: str,
        resource: str = "",
    ) -> ExperimentResult:
        """Run a single experiment via Claude Code subagent."""
        task = ExperimentTask(experiment_id, params, command, baseline_commit, resource)
        return self._run_single(task)

    def run_experiments(self, tasks: list[ExperimentTask]) -> list[ExperimentResult]:
        """Run a batch of experiments.

        Postcondition: always returns exactly one ExperimentResult per input task.
        Failures are captured as ExperimentResult with error_message set, never raised.
        """
        max_workers = len(tasks) if self._mode == "parallel" else 1

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(self._run_single, task) for task in tasks]
            results = []
            for future, task in zip(futures, tasks):
                try:
                    results.append(future.result())
                except Exception as e:
                    results.append(ExperimentResult(
                        primary_metric=0.0,
                        error_message=f"Thread crashed for {task.experiment_id}: {e}",
                    ))
        return results

    def _run_single(self, task: ExperimentTask) -> ExperimentResult:
        """Execute the full pipeline for one experiment."""
        result_file = self._output_dir / task.experiment_id / "result.json"

        # 1. Create worktree
        worktree_path = self._worktree_mgr.create(
            task.baseline_commit, self._run_id, task.experiment_id
        )

        try:
            # 2. Build task file
            task_file = self._task_builder.build(
                task, self._spec, worktree_path, result_file,
                run_id=self._run_id, output_dir=self._output_dir,
            )

            # 3. Invoke claude subprocess
            start = time.monotonic()
            invoke_result = self._invoke(
                task_file, worktree_path, task.resource
            )
            duration = time.monotonic() - start

            if isinstance(invoke_result, ExperimentResult):
                return invoke_result

            # 4. Check for non-zero exit code
            if invoke_result.returncode != 0:
                return ExperimentResult(
                    primary_metric=0.0,
                    duration_seconds=duration,
                    error_message=(
                        f"Claude process failed for {task.experiment_id} "
                        f"(exit {invoke_result.returncode}): "
                        f"{(invoke_result.stderr or '')[:500]}"
                    ),
                )

            # 5. Parse result
            return self._result_parser.parse(
                result_file, task.experiment_id, duration
            )
        finally:
            # 6. Cleanup worktree
            self._worktree_mgr.cleanup(worktree_path)

    def _invoke(
        self,
        task_file: Path,
        worktree_path: Path,
        resource: str,
    ) -> subprocess.CompletedProcess | ExperimentResult:
        """Spawn claude -p subprocess using the task file.

        Returns CompletedProcess on success, or an error ExperimentResult
        on timeout.
        """
        env = {**os.environ}
        if resource:
            gpu_id = _parse_gpu_id(resource)
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id

        timeout = None
        if self._spec.time_budget_seconds is not None:
            timeout = self._spec.time_budget_seconds + _GRACE_SECONDS

        prompt = task_file.read_text()
        cmd = [
            "claude", "-p", prompt,
            "--allowedTools", "Edit,Write,Bash,Read",
        ]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(worktree_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result
        except subprocess.TimeoutExpired:
            return ExperimentResult(
                primary_metric=0.0,
                error_message=(
                    f"Experiment timed out after {timeout}s "
                    f"(budget: {self._spec.time_budget_seconds}s + {_GRACE_SECONDS}s grace)"
                ),
            )


def _parse_gpu_id(resource: str) -> str | None:
    """Extract GPU device ID from resource string like 'gpu:2'."""
    match = re.match(r"gpu:(\d+)", resource)
    return match.group(1) if match else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_subagent_executor.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/execution/subagent.py tests/test_subagent_executor.py
git commit -m "feat: add SubagentExecutor with parallel batch execution"
```

### Task 7: Create executor factory

**Files:**
- Modify: `chaosengineer/execution/__init__.py`
- Create: `tests/test_executor_factory.py`

- [ ] **Step 1: Write failing tests for create_executor**

Create `tests/test_executor_factory.py`:

```python
"""Tests for executor factory."""

import json
import pytest
from pathlib import Path

import yaml

from chaosengineer.execution import create_executor
from chaosengineer.execution.subagent import SubagentExecutor
from chaosengineer.testing.executor import ScriptedExecutor
from chaosengineer.core.models import ExperimentResult
from chaosengineer.workloads.parser import WorkloadSpec


def _make_spec() -> WorkloadSpec:
    return WorkloadSpec(
        name="test",
        primary_metric="val_bpb",
        metric_direction="lower",
        execution_command="echo test",
    )


class TestCreateExecutor:
    def test_subagent_backend(self, tmp_path):
        executor = create_executor("subagent", _make_spec(), tmp_path, "sequential")
        assert isinstance(executor, SubagentExecutor)

    def test_scripted_from_file(self, tmp_path):
        results_file = tmp_path / "results.yaml"
        results_file.write_text(yaml.dump({
            "exp-0-0": {"primary_metric": 0.91},
            "exp-0-1": {"primary_metric": 0.95},
        }))

        executor = create_executor(
            "scripted", _make_spec(), tmp_path, "sequential",
            scripted_results=results_file,
        )
        assert isinstance(executor, ScriptedExecutor)

        result = executor.run_experiment("exp-0-0", {}, "", "")
        assert result.primary_metric == 0.91

    def test_scripted_from_folder(self, tmp_path):
        folder = tmp_path / "results"
        folder.mkdir()
        (folder / "batch1.yaml").write_text(yaml.dump({
            "exp-0-0": {"primary_metric": 0.91},
        }))
        (folder / "batch2.yaml").write_text(yaml.dump({
            "exp-0-1": {"primary_metric": 0.95},
        }))

        executor = create_executor(
            "scripted", _make_spec(), tmp_path, "sequential",
            scripted_results=folder,
        )
        assert isinstance(executor, ScriptedExecutor)

        r0 = executor.run_experiment("exp-0-0", {}, "", "")
        r1 = executor.run_experiment("exp-0-1", {}, "", "")
        assert r0.primary_metric == 0.91
        assert r1.primary_metric == 0.95

    def test_scripted_requires_path(self):
        with pytest.raises(ValueError, match="scripted_results"):
            create_executor("scripted", _make_spec(), Path("/tmp"), "sequential")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            create_executor("unknown", _make_spec(), Path("/tmp"), "sequential")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_executor_factory.py -v`
Expected: ImportError for `create_executor`

- [ ] **Step 3: Implement create_executor factory**

Replace `chaosengineer/execution/__init__.py`:

```python
"""Experiment execution backends."""

from __future__ import annotations

from pathlib import Path

import yaml

from chaosengineer.core.interfaces import ExperimentExecutor
from chaosengineer.core.models import ExperimentResult
from chaosengineer.workloads.parser import WorkloadSpec


def create_executor(
    backend: str,
    spec: WorkloadSpec,
    output_dir: Path,
    mode: str = "sequential",
    scripted_results: Path | None = None,
    run_id: str = "run-unknown",
) -> ExperimentExecutor:
    """Create an executor with the specified backend.

    Args:
        backend: "subagent" (default) or "scripted" (for testing/demos)
        spec: Workload specification
        output_dir: Directory for experiment output
        mode: "sequential" or "parallel"
        scripted_results: Path to YAML file or folder (required for scripted)
        run_id: Run identifier (used by subagent for branch naming)
    """
    if backend == "subagent":
        from chaosengineer.execution.subagent import SubagentExecutor
        return SubagentExecutor(spec, output_dir, mode, run_id=run_id)

    elif backend == "scripted":
        if scripted_results is None:
            raise ValueError(
                "scripted_results path is required when using --executor=scripted"
            )
        results = _load_scripted_results(scripted_results)
        from chaosengineer.testing.executor import ScriptedExecutor
        return ScriptedExecutor(results)

    else:
        raise ValueError(
            f"Unknown executor backend: '{backend}'. Use 'subagent' or 'scripted'."
        )


def _load_scripted_results(path: Path) -> dict[str, ExperimentResult]:
    """Load scripted results from a YAML file or folder of YAML files."""
    if path.is_dir():
        merged: dict[str, dict] = {}
        for yaml_file in sorted(path.glob("*.yaml")):
            data = yaml.safe_load(yaml_file.read_text())
            if data:
                merged.update(data)
        raw = merged
    else:
        raw = yaml.safe_load(path.read_text()) or {}

    results = {}
    for exp_id, data in raw.items():
        results[exp_id] = ExperimentResult(
            primary_metric=data["primary_metric"],
            secondary_metrics=data.get("secondary_metrics", {}),
            artifacts=data.get("artifacts", []),
            commit_hash=data.get("commit_hash"),
            duration_seconds=data.get("duration_seconds", 0),
            error_message=data.get("error_message"),
        )
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_executor_factory.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/execution/__init__.py tests/test_executor_factory.py
git commit -m "feat: add create_executor factory with subagent and scripted backends"
```

## Chunk 4: Coordinator Batch Refactor + CLI Wiring

### Task 8: Refactor coordinator _run_iteration to use batch API

**Files:**
- Modify: `chaosengineer/core/coordinator.py`
- Create: `tests/test_batch_coordinator.py`

- [ ] **Step 1: Write integration tests for batch coordinator**

Create `tests/test_batch_coordinator.py`:

```python
"""Integration tests for coordinator with batch run_experiments."""

import pytest
from chaosengineer.core.models import (
    BudgetConfig, Baseline, ExperimentResult, ExperimentStatus,
)
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.budget import BudgetTracker
from chaosengineer.metrics.logger import EventLogger
from chaosengineer.core.interfaces import DimensionPlan, ExperimentTask
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


class TestBatchRunExperiments:
    """Verify coordinator uses run_experiments and handles batch results."""

    def test_batch_called_with_correct_tasks(self, tmp_output_dir):
        """The coordinator should build ExperimentTask objects and call run_experiments."""
        spec = _make_spec()
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}, {"lr": 0.08}]),
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.91),
            "exp-0-1": ExperimentResult(primary_metric=0.95),
        }

        # Track calls to run_experiments
        calls = []

        class TrackingExecutor(ScriptedExecutor):
            def run_experiments(self, tasks):
                calls.append(tasks)
                return super().run_experiments(tasks)

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=TrackingExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="abc", metric_value=0.97, metric_name="val_bpb"),
        )

        coordinator.run()

        assert len(calls) == 1
        assert len(calls[0]) == 2
        assert isinstance(calls[0][0], ExperimentTask)
        assert calls[0][0].experiment_id == "exp-0-0"
        assert calls[0][0].params == {"lr": 0.02}
        assert calls[0][1].experiment_id == "exp-0-1"

    def test_mixed_success_failure_in_batch(self, tmp_output_dir):
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

        assert coordinator.best_baseline.metric_value == 0.93
        failed = [
            e for e in coordinator.run_state.experiments
            if e.status == ExperimentStatus.FAILED
        ]
        assert len(failed) == 1

    def test_budget_tracking_across_batches(self, tmp_output_dir):
        spec = _make_spec()
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}]),
            DimensionPlan(dimension_name="depth", values=[{"depth": 6}]),
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.91, cost_usd=0.10),
            "exp-1-0": ExperimentResult(primary_metric=0.89, cost_usd=0.15),
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
        assert coordinator.budget.spent_usd == pytest.approx(0.25)

    def test_all_experiments_started_before_batch(self, tmp_output_dir):
        """All experiments should be in RUNNING state before batch call."""
        spec = _make_spec()
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}, {"lr": 0.08}]),
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.91),
            "exp-0-1": ExperimentResult(primary_metric=0.95),
        }

        pre_batch_statuses = []

        class InspectingExecutor(ScriptedExecutor):
            def __init__(self, results, coordinator_ref):
                super().__init__(results)
                self._coordinator_ref = coordinator_ref

            def run_experiments(self, tasks):
                # Capture experiment statuses before execution
                for exp in self._coordinator_ref.run_state.experiments:
                    pre_batch_statuses.append(exp.status)
                return super().run_experiments(tasks)

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),  # placeholder
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="abc", metric_value=0.97, metric_name="val_bpb"),
        )
        # Replace executor with inspecting version
        coordinator.executor = InspectingExecutor(results, coordinator)

        coordinator.run()

        assert all(s == ExperimentStatus.RUNNING for s in pre_batch_statuses)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_batch_coordinator.py -v`
Expected: FAIL — coordinator doesn't call `run_experiments`

- [ ] **Step 3: Refactor _run_iteration to use batch API**

In `chaosengineer/core/coordinator.py`, add `ExperimentTask` to imports:

```python
from chaosengineer.core.interfaces import DecisionMaker, DimensionPlan, ExperimentExecutor, ExperimentTask
```

Replace the `_run_iteration` method (lines 168-234) with the batch version:

```python
    def _run_iteration(
        self, plan: DimensionPlan, baseline: Baseline
    ) -> list[tuple[Experiment, ExperimentResult | None]]:
        """Run all experiments for one dimension sweep from a given baseline."""
        # Phase 1: Build Experiment objects and task list
        tasks: list[ExperimentTask] = []
        experiment_workers: list[tuple[Experiment, WorkerState]] = []

        for i, params in enumerate(plan.values):
            exp_id = f"exp-{self._iteration}-{i}"
            exp = Experiment(
                experiment_id=exp_id,
                dimension=plan.dimension_name,
                params=params,
                baseline_commit=baseline.commit,
                branch_id=baseline.branch_id,
            )
            self.run_state.experiments.append(exp)

            worker = WorkerState(worker_id=f"w-{self._iteration}-{i}")
            assign_experiment(exp, worker.worker_id)
            assign_worker(worker, exp.experiment_id)
            start_experiment(exp)

            tasks.append(ExperimentTask(
                exp_id, params, self.spec.execution_command, baseline.commit,
            ))
            experiment_workers.append((exp, worker))

        # Phase 2: Execute batch
        batch_results = self.executor.run_experiments(tasks)

        # Phase 3: Handle results
        results: list[tuple[Experiment, ExperimentResult | None]] = []
        for (exp, worker), result in zip(experiment_workers, batch_results):
            if result.error_message:
                fail_experiment(exp, result)
                self._log(Event(
                    event="worker_failed",
                    data={"experiment_id": exp.experiment_id, "error": result.error_message},
                ))
            else:
                complete_experiment(exp, result)
                self._log(Event(
                    event="worker_completed",
                    data={
                        "experiment_id": exp.experiment_id,
                        "params": exp.params,
                        "result": result.to_dict(),
                    },
                ))
            release_worker(worker)
            self.budget.record_experiment()
            self.budget.add_cost(result.cost_usd)

            results.append((exp, result))

        return results
```

- [ ] **Step 4: Run batch coordinator tests**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_batch_coordinator.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/ -v --ignore=tests/test_runner.py --ignore=tests/test_shipped_scenarios.py`
Expected: All existing tests PASS (the coordinator refactor is backward-compatible — `ScriptedExecutor` inherits `run_experiments` default)

- [ ] **Step 6: Commit**

```bash
git add chaosengineer/core/coordinator.py tests/test_batch_coordinator.py
git commit -m "feat: refactor coordinator _run_iteration to use batch run_experiments API"
```

### Task 9: Wire CLI with --executor and --mode flags

**Files:**
- Modify: `chaosengineer/cli.py`
- Create: `tests/test_cli_run.py`

- [ ] **Step 1: Write failing tests for CLI run command**

Create `tests/test_cli_run.py`:

```python
"""Tests for CLI run command argument parsing and validation."""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch

from chaosengineer.cli import main


class TestCliRunArgs:
    """Test that argparse accepts and defaults the new flags correctly.
    We test argument parsing by intercepting at the argparse level."""

    def test_executor_flag_default(self):
        """--executor defaults to 'subagent'."""
        import argparse
        with patch("sys.argv", ["chaosengineer", "run", "workload.md"]):
            # Intercept parse_args to check values without executing
            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = argparse.Namespace(
                    command="run", workload=Path("workload.md"),
                    executor="subagent", mode="sequential",
                    llm_backend="claude-code", scripted_results=None,
                    output_dir=Path(".chaosengineer/output"),
                )
                # Patch _execute_run to prevent actual execution
                with patch("chaosengineer.cli._execute_run"):
                    main()

    def test_executor_choices_validated(self):
        """argparse rejects invalid --executor values."""
        with patch("sys.argv", [
            "chaosengineer", "run", "workload.md",
            "--executor", "invalid",
        ]):
            with pytest.raises(SystemExit):
                main()

    def test_mode_choices_validated(self):
        """argparse rejects invalid --mode values."""
        with patch("sys.argv", [
            "chaosengineer", "run", "workload.md",
            "--mode", "invalid",
        ]):
            with pytest.raises(SystemExit):
                main()

    def test_scripted_results_accepted(self):
        """--scripted-results is accepted as a Path argument."""
        import argparse
        with patch("sys.argv", [
            "chaosengineer", "run", "workload.md",
            "--executor", "scripted",
            "--scripted-results", "results.yaml",
        ]):
            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = argparse.Namespace(
                    command="run", workload=Path("workload.md"),
                    executor="scripted", mode="sequential",
                    llm_backend="claude-code",
                    scripted_results=Path("results.yaml"),
                    output_dir=Path(".chaosengineer/output"),
                )
                with patch("chaosengineer.cli._execute_run"):
                    main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_cli_run.py -v`
Expected: FAIL — `--executor`, `--mode` flags don't exist, `_execute_run` doesn't exist

- [ ] **Step 3: Update CLI with new flags and execution wiring**

Replace the `run` command section in `chaosengineer/cli.py`. Add the new args to `run_parser` and extract the run logic into a helper:

```python
    # Run command: execute a workload
    run_parser = subparsers.add_parser("run", help="Run a workload")
    run_parser.add_argument(
        "workload",
        type=Path,
        help="Path to workload spec markdown file",
    )
    run_parser.add_argument(
        "--llm-backend",
        choices=["claude-code", "sdk"],
        default="claude-code",
        help="LLM backend for coordinator decisions (default: claude-code)",
    )
    run_parser.add_argument(
        "--executor",
        choices=["subagent", "scripted"],
        default="subagent",
        help="Executor backend (default: subagent)",
    )
    run_parser.add_argument(
        "--mode",
        choices=["sequential", "parallel"],
        default="sequential",
        help="Execution mode (default: sequential)",
    )
    run_parser.add_argument(
        "--scripted-results",
        type=Path,
        help="YAML file or folder with canned results (required for --executor=scripted)",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".chaosengineer/output"),
        help="Directory for run output",
    )
```

Add `_execute_run` function and update the `elif args.command == "run":` block:

```python
    elif args.command == "run":
        _execute_run(args)
```

Add the function before `_print_scenario_result`:

```python
def _execute_run(args):
    """Execute a workload run with the specified backends."""
    import uuid
    from chaosengineer.workloads.parser import parse_workload_spec
    from chaosengineer.llm import create_decision_maker
    from chaosengineer.execution import create_executor
    from chaosengineer.core.coordinator import Coordinator
    from chaosengineer.core.budget import BudgetTracker
    from chaosengineer.metrics.logger import EventLogger
    from chaosengineer.core.models import Baseline

    if args.executor == "scripted" and args.scripted_results is None:
        print("Error: --scripted-results is required when using --executor=scripted", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = parse_workload_spec(args.workload)

    # Generate a single run_id for both coordinator and executor
    run_id = f"run-{uuid.uuid4().hex[:8]}"

    llm_dir = args.output_dir / "llm_decisions"
    llm_dir.mkdir(parents=True, exist_ok=True)

    dm = create_decision_maker(args.llm_backend, spec, llm_dir)
    executor = create_executor(
        args.executor, spec, args.output_dir, args.mode,
        scripted_results=args.scripted_results,
        run_id=run_id,
    )
    logger = EventLogger(args.output_dir / "events.jsonl")
    budget = BudgetTracker(spec.budget)

    # TODO: initial baseline should come from workload spec or be auto-detected
    initial_baseline = Baseline(
        commit="HEAD",
        metric_value=float("inf") if spec.metric_direction == "lower" else float("-inf"),
        metric_name=spec.primary_metric,
    )

    coordinator = Coordinator(
        spec=spec,
        decision_maker=dm,
        executor=executor,
        logger=logger,
        budget=budget,
        initial_baseline=initial_baseline,
        run_id=run_id,
    )

    print(f"Starting run: {spec.name}")
    print(f"  LLM backend: {args.llm_backend}")
    print(f"  Executor: {args.executor} ({args.mode})")
    print(f"  Output: {args.output_dir}")

    coordinator.run()

    print(f"\nRun complete:")
    print(f"  Best metric: {coordinator.best_baseline.metric_value}")
    print(f"  Experiments: {coordinator.budget.experiments_run}")
    print(f"  Cost: ${coordinator.budget.spent_usd:.2f}")
```

- [ ] **Step 4: Run CLI tests**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_cli_run.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/ -v --ignore=tests/test_runner.py --ignore=tests/test_shipped_scenarios.py`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add chaosengineer/cli.py tests/test_cli_run.py
git commit -m "feat: wire CLI run command with --executor and --mode flags"
```

## Chunk 5: End-to-End Tests

### Task 10: Fix test collection errors (yaml dependency)

**Files:**
- Modify: `tests/test_runner.py` (if needed)
- Modify: `tests/test_shipped_scenarios.py` (if needed)

- [ ] **Step 1: Verify the collection errors**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_runner.py tests/test_shipped_scenarios.py --co 2>&1 | head -20`
Expected: `ModuleNotFoundError: No module named 'yaml'` — this means `pyyaml` is declared in pyproject.toml but not installed in the environment

- [ ] **Step 2: Install the missing dependency**

Run: `cd /Users/alex/CODE/OSS/autoresearch && uv sync`

If that doesn't resolve it (e.g., torch dependency issues on macOS), try:
Run: `cd /Users/alex/CODE/OSS/autoresearch && pip install pyyaml`

- [ ] **Step 3: Verify test collection works**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_runner.py tests/test_shipped_scenarios.py --co`
Expected: Tests collected without errors

- [ ] **Step 4: Run the previously-failing tests**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_runner.py tests/test_shipped_scenarios.py -v`
Expected: All PASS

- [ ] **Step 5: Commit if any file changes were needed**

```bash
# Only if files were modified (e.g., uv.lock updated)
git add uv.lock
git commit -m "fix: sync dependencies to resolve yaml import errors in tests"
```

### Task 11: Create end-to-end tests with scripted executor

**Files:**
- Create: `tests/e2e/` directory
- Create: `tests/e2e/__init__.py`
- Create: `tests/e2e/test_scripted_pipeline.py`
- Create: `tests/e2e/fixtures/simple_results.yaml`
- Create: `tests/e2e/fixtures/simple_workload.md`
- Create: `tests/e2e/fixtures/results_folder/batch1.yaml`
- Create: `tests/e2e/fixtures/results_folder/batch2.yaml`

- [ ] **Step 1: Create test fixtures**

Create `tests/e2e/__init__.py` (empty).

Create `tests/e2e/fixtures/simple_workload.md`:

```markdown
# Workload: Simple E2E Test

## Context
A simple workload for end-to-end testing.

## Experiment Space
- Directional: "lr" (currently 0.04)

## Execution
- Command: `echo test`

## Evaluation
- Type: automatic
- Metric: val_bpb (lower is better)

## Resources
- Available: 2

## Budget
- Max experiments: 4
```

Create `tests/e2e/fixtures/simple_results.yaml`:

```yaml
"exp-0-0":
  primary_metric: 0.91
"exp-0-1":
  primary_metric: 0.95
```

Create `tests/e2e/fixtures/results_folder/batch1.yaml`:

```yaml
"exp-0-0":
  primary_metric: 0.91
  secondary_metrics:
    train_loss: 1.5
```

Create `tests/e2e/fixtures/results_folder/batch2.yaml`:

```yaml
"exp-0-1":
  primary_metric: 0.95
```

- [ ] **Step 2: Write end-to-end tests**

Create `tests/e2e/test_scripted_pipeline.py`:

```python
"""End-to-end tests: full pipeline with scripted executor."""

import json
import pytest
from pathlib import Path

from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.interfaces import DimensionPlan
from chaosengineer.core.models import Baseline, BudgetConfig
from chaosengineer.execution import create_executor
from chaosengineer.llm import create_decision_maker
from chaosengineer.metrics.logger import EventLogger
from chaosengineer.testing.simulator import ScriptedDecisionMaker
from chaosengineer.workloads.parser import parse_workload_spec


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestScriptedPipelineFromFile:
    """Full pipeline using scripted results from a single YAML file."""

    def test_full_run(self, tmp_path):
        spec = parse_workload_spec(FIXTURES_DIR / "simple_workload.md")
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}, {"lr": 0.08}]),
        ]

        executor = create_executor(
            "scripted", spec, tmp_path, "sequential",
            scripted_results=FIXTURES_DIR / "simple_results.yaml",
        )

        logger = EventLogger(tmp_path / "events.jsonl")
        budget = BudgetTracker(spec.budget)

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=executor,
            logger=logger,
            budget=budget,
            initial_baseline=Baseline(commit="abc", metric_value=0.97, metric_name="val_bpb"),
        )

        coordinator.run()

        # Verify results
        assert coordinator.best_baseline.metric_value == 0.91
        assert coordinator.budget.experiments_run == 2

        # Verify event log
        events = logger.read_events()
        assert any(e["event"] == "run_started" for e in events)
        assert any(e["event"] == "run_completed" for e in events)
        assert any(e["event"] == "breakthrough" for e in events)
        assert any(e["event"] == "worker_completed" for e in events)


class TestScriptedPipelineFromFolder:
    """Full pipeline using scripted results from a folder of YAML files."""

    def test_full_run_from_folder(self, tmp_path):
        spec = parse_workload_spec(FIXTURES_DIR / "simple_workload.md")
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}, {"lr": 0.08}]),
        ]

        executor = create_executor(
            "scripted", spec, tmp_path, "sequential",
            scripted_results=FIXTURES_DIR / "results_folder",
        )

        logger = EventLogger(tmp_path / "events.jsonl")
        budget = BudgetTracker(spec.budget)

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=executor,
            logger=logger,
            budget=budget,
            initial_baseline=Baseline(commit="abc", metric_value=0.97, metric_name="val_bpb"),
        )

        coordinator.run()

        assert coordinator.best_baseline.metric_value == 0.91
        assert coordinator.budget.experiments_run == 2


class TestBudgetExhaustion:
    """Verify budget limits terminate the run correctly."""

    def test_max_experiments_stops_run(self, tmp_path):
        spec = parse_workload_spec(FIXTURES_DIR / "simple_workload.md")
        # Budget is max 4 experiments from workload, but only provide 2 results
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}, {"lr": 0.08}]),
        ]

        executor = create_executor(
            "scripted", spec, tmp_path, "sequential",
            scripted_results=FIXTURES_DIR / "simple_results.yaml",
        )

        logger = EventLogger(tmp_path / "events.jsonl")
        budget = BudgetTracker(BudgetConfig(max_experiments=2))

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=executor,
            logger=logger,
            budget=budget,
            initial_baseline=Baseline(commit="abc", metric_value=0.97, metric_name="val_bpb"),
        )

        coordinator.run()

        assert coordinator.budget.experiments_run == 2
        assert coordinator.budget.is_exhausted()

        # Verify run_completed event
        completed = logger.read_events(event_type="run_completed")
        assert len(completed) == 1


class TestEventLogCompleteness:
    """Verify the event log contains all expected events in order."""

    def test_event_sequence(self, tmp_path):
        spec = parse_workload_spec(FIXTURES_DIR / "simple_workload.md")
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}, {"lr": 0.08}]),
        ]

        executor = create_executor(
            "scripted", spec, tmp_path, "sequential",
            scripted_results=FIXTURES_DIR / "simple_results.yaml",
        )

        logger = EventLogger(tmp_path / "events.jsonl")
        budget = BudgetTracker(spec.budget)

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=executor,
            logger=logger,
            budget=budget,
            initial_baseline=Baseline(commit="abc", metric_value=0.97, metric_name="val_bpb"),
        )

        coordinator.run()

        events = logger.read_events()
        event_types = [e["event"] for e in events]

        # Must start with run_started and end with run_completed
        assert event_types[0] == "run_started"
        assert event_types[-1] == "run_completed"

        # Must have iteration_started
        assert "iteration_started" in event_types

        # Must have worker_completed for each successful experiment
        completed = [e for e in events if e["event"] == "worker_completed"]
        assert len(completed) == 2
```

- [ ] **Step 3: Create Autoresearch scenario E2E test**

This validates the full pipeline against the original use case. Create `tests/e2e/fixtures/autoresearch_results.yaml` with scripted results matching the sample workload's dimension space:

```yaml
# Scripted results for the sample autoresearch workload
"exp-0-0":
  primary_metric: 0.92
  secondary_metrics: { train_loss: 1.3, perplexity: 2.5 }
"exp-0-1":
  primary_metric: 0.95
  secondary_metrics: { train_loss: 1.5, perplexity: 2.8 }
```

Add to `tests/e2e/test_scripted_pipeline.py`:

```python
class TestAutoresearchScenario:
    """The original autoresearch workload run e2e with scripted results."""

    def test_autoresearch_workload(self, tmp_path):
        sample_workload = Path(__file__).parents[2] / "tests" / "fixtures" / "sample_workload.md"
        spec = parse_workload_spec(sample_workload)
        plans = [
            DimensionPlan(
                dimension_name="learning_rate",
                values=[{"learning_rate": 0.02}, {"learning_rate": 0.08}],
            ),
        ]

        executor = create_executor(
            "scripted", spec, tmp_path, "sequential",
            scripted_results=FIXTURES_DIR / "autoresearch_results.yaml",
        )

        logger = EventLogger(tmp_path / "events.jsonl")
        budget = BudgetTracker(spec.budget)

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=executor,
            logger=logger,
            budget=budget,
            initial_baseline=Baseline(
                commit="abc", metric_value=0.97, metric_name="val_bpb",
            ),
        )

        coordinator.run()

        assert coordinator.best_baseline.metric_value == 0.92
        assert coordinator.budget.experiments_run == 2

        # Verify secondary metrics are in the event log
        completed = logger.read_events(event_type="worker_completed")
        assert any(
            "train_loss" in e.get("result", {}).get("secondary_metrics", {})
            for e in completed
        )
```

- [ ] **Step 4: Run end-to-end tests**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/e2e/ -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/ -v --ignore=tests/test_runner.py --ignore=tests/test_shipped_scenarios.py`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add tests/e2e/
git commit -m "feat: add end-to-end tests for scripted executor pipeline"
```

### Task 12: Final verification and cleanup

- [ ] **Step 1: Run complete test suite including scenario tests**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/ -v`
Expected: All tests PASS (including test_runner.py and test_shipped_scenarios.py if yaml was fixed)

- [ ] **Step 2: Verify test count**

Expected: ~150 total tests (117 existing + ~33 new)

- [ ] **Step 3: Update memory with Phase 2C status**

Update the project memory file to reflect Phase 2C completion.

- [ ] **Step 4: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "chore: Phase 2C SubagentExecutor complete"
```
