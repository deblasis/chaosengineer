# ChaosEngineer Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a general-purpose parallel experimentation framework that evolves autoresearch into deblasis/chaosengineer, supporting coordinator/worker architecture, generic workload specs, a testing framework, and metrics/observability.

**Architecture:** Clean-room Python package (`chaosengineer/`) alongside existing autoresearch files. All orchestration logic lives in Python with a pluggable "decision maker" interface — real mode calls an LLM, test mode uses scripted responses. This allows the entire coordinator flow to be validated without LLM calls via a scenario-based testing framework.

**Tech Stack:** Python 3.10+, pytest, PyYAML, dataclasses, pathlib. No new heavy dependencies. Uses `uv` as the package manager (already configured).

**Spec:** `docs/superpowers/specs/2026-03-15-chaosengineer-design.md`

---

## File Structure

```
chaosengineer/
  __init__.py                  # Package version and public API
  core/
    __init__.py
    models.py                  # Data classes: Experiment, WorkerState, Run, Baseline, DimensionSpec
    state.py                   # State machine transitions with validation
    coordinator.py             # Coordinator loop logic: dimension planning, allocation, beam search
    budget.py                  # Budget tracking and enforcement (cost, experiments, time)
  metrics/
    __init__.py
    logger.py                  # JSONL event logger (append-only, structured events)
    summary.py                 # Per-run summary generation (human + machine readable)
  workloads/
    __init__.py
    parser.py                  # Markdown workload spec parser
  testing/
    __init__.py
    simulator.py               # DecisionMaker interface + ScriptedDecisionMaker for tests
    executor.py                # ExperimentExecutor interface + ScriptedExecutor for tests
    runner.py                  # Scenario runner: loads YAML, wires simulator+executor, runs coordinator
    scenarios/                 # Shipped YAML test scenarios
      breakthrough.yaml
      tie_branching.yaml
      budget_exhaustion.yaml
  cli.py                       # CLI entry point (--mode sequential|parallel)

tests/
  __init__.py
  conftest.py                  # Shared fixtures (tmp dirs, sample specs, etc.)
  test_models.py               # Data model construction, serialization, equality
  test_state.py                # State machine transition validation
  test_logger.py               # JSONL event writing and reading
  test_budget.py               # Budget tracking, enforcement, edge cases
  test_parser.py               # Workload spec parsing from markdown
  test_coordinator.py          # Coordinator logic with scripted decision maker
  test_runner.py               # End-to-end scenario runner tests
```

---

## Chunk 1: Project Setup & Data Models

### Task 1: Initialize Python package structure

**Files:**
- Create: `chaosengineer/__init__.py`
- Create: `chaosengineer/core/__init__.py`
- Create: `chaosengineer/metrics/__init__.py`
- Create: `chaosengineer/workloads/__init__.py`
- Create: `chaosengineer/testing/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Update pyproject.toml to add test dependencies and package config**

```toml
[project]
name = "chaosengineer"
version = "0.1.0"
description = "General-purpose parallel experimentation framework"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "kernels>=0.11.7",
    "matplotlib>=3.10.8",
    "numpy>=2.2.6",
    "pandas>=2.3.3",
    "pyarrow>=21.0.0",
    "pyyaml>=6.0",
    "requests>=2.32.0",
    "rustbpe>=0.1.0",
    "tiktoken>=0.11.0",
    "torch==2.9.1",
]

[project.optional-dependencies]
test = ["pytest>=8.0"]

[tool.uv.sources]
torch = [
    { index = "pytorch-cu128" },
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

Note: Only changes are `name`, `description`, adding `pyyaml`, and adding `[project.optional-dependencies]` for test. Keep the existing `[tool.uv.sources]` and `[[tool.uv.index]]` sections exactly as they are.

- [ ] **Step 2: Create package __init__ files**

`chaosengineer/__init__.py`:
```python
"""ChaosEngineer: General-purpose parallel experimentation framework."""

__version__ = "0.1.0"
```

`chaosengineer/core/__init__.py`:
```python
"""Core data models and orchestration logic."""
```

`chaosengineer/metrics/__init__.py`:
```python
"""Metrics logging and observability."""
```

`chaosengineer/workloads/__init__.py`:
```python
"""Workload specification loading and parsing."""
```

`chaosengineer/testing/__init__.py`:
```python
"""Testing framework: simulators, scenario runner, shipped scenarios."""
```

`tests/__init__.py`: empty file

`tests/conftest.py`:
```python
"""Shared test fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary directory for test outputs (logs, results, etc.)."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
```

- [ ] **Step 3: Verify the package installs and pytest runs**

Run: `cd /Users/alex/CODE/OSS/autoresearch && uv pip install -e ".[test]" && uv run pytest tests/ -v --co`
Expected: Package installs, pytest collects 0 tests (no test files yet), exits cleanly.

- [ ] **Step 4: Commit**

```bash
git add chaosengineer/ tests/ pyproject.toml
git commit -m "feat: initialize chaosengineer package structure with test setup"
```

---

### Task 2: Define core data models

**Files:**
- Create: `chaosengineer/core/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests for data models**

`tests/test_models.py`:
```python
"""Tests for core data models."""

import pytest
from chaosengineer.core.models import (
    ExperimentStatus,
    WorkerStatus,
    DimensionType,
    DimensionSpec,
    Experiment,
    WorkerState,
    Baseline,
    BudgetConfig,
    Run,
)


class TestDimensionSpec:
    def test_directional(self):
        d = DimensionSpec(
            name="learning_rate",
            dim_type=DimensionType.DIRECTIONAL,
            current_value=0.04,
        )
        assert d.name == "learning_rate"
        assert d.dim_type == DimensionType.DIRECTIONAL
        assert d.current_value == 0.04
        assert d.options is None

    def test_enum(self):
        d = DimensionSpec(
            name="activation",
            dim_type=DimensionType.ENUM,
            options=["GeLU", "SiLU", "ReLU"],
        )
        assert d.dim_type == DimensionType.ENUM
        assert len(d.options) == 3

    def test_diverse(self):
        d = DimensionSpec(
            name="prompt_strategy",
            dim_type=DimensionType.DIVERSE,
        )
        assert d.dim_type == DimensionType.DIVERSE
        assert d.options is None  # discovered at runtime


class TestExperiment:
    def test_creation(self):
        exp = Experiment(
            experiment_id="exp-001",
            dimension="learning_rate",
            params={"learning_rate": 0.08},
            baseline_commit="abc1234",
        )
        assert exp.status == ExperimentStatus.PLANNED
        assert exp.result is None
        assert exp.worker_id is None

    def test_to_dict_roundtrip(self):
        exp = Experiment(
            experiment_id="exp-001",
            dimension="learning_rate",
            params={"learning_rate": 0.08},
            baseline_commit="abc1234",
        )
        d = exp.to_dict()
        assert d["experiment_id"] == "exp-001"
        assert d["status"] == "planned"
        assert d["params"] == {"learning_rate": 0.08}


class TestWorkerState:
    def test_creation(self):
        w = WorkerState(worker_id="w1", resource="CUDA_VISIBLE_DEVICES=0")
        assert w.status == WorkerStatus.IDLE
        assert w.current_experiment_id is None

    def test_to_dict(self):
        w = WorkerState(worker_id="w1", resource="CUDA_VISIBLE_DEVICES=0")
        d = w.to_dict()
        assert d["worker_id"] == "w1"
        assert d["status"] == "idle"


class TestBaseline:
    def test_creation(self):
        b = Baseline(commit="abc1234", metric_value=0.95, metric_name="val_bpb")
        assert b.commit == "abc1234"
        assert b.metric_value == 0.95


class TestBudgetConfig:
    def test_defaults(self):
        b = BudgetConfig()
        assert b.max_api_cost is None
        assert b.max_experiments is None
        assert b.max_wall_time_seconds is None
        assert b.max_plateau_iterations is None

    def test_with_limits(self):
        b = BudgetConfig(max_api_cost=50.0, max_experiments=100, max_wall_time_seconds=28800)
        assert b.max_api_cost == 50.0
        assert b.max_experiments == 100
        assert b.max_wall_time_seconds == 28800


class TestRun:
    def test_creation(self):
        budget = BudgetConfig(max_experiments=10)
        run = Run(
            run_id="run-001",
            workload_name="nn-arch-search",
            budget=budget,
        )
        assert run.experiments == []
        assert run.workers == []
        assert len(run.baselines) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chaosengineer.core.models'`

- [ ] **Step 3: Implement data models**

`chaosengineer/core/models.py`:
```python
"""Core data models for ChaosEngineer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExperimentStatus(Enum):
    PLANNED = "planned"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


class WorkerStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    TERMINATED = "terminated"


class DimensionType(Enum):
    DIRECTIONAL = "directional"
    ENUM = "enum"
    DIVERSE = "diverse"


@dataclass
class DimensionSpec:
    """A dimension of the experiment space."""
    name: str
    dim_type: DimensionType
    current_value: Any = None
    options: list[str] | None = None
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dim_type": self.dim_type.value,
            "current_value": self.current_value,
            "options": self.options,
            "description": self.description,
        }


@dataclass
class ExperimentResult:
    """Result of a completed experiment."""
    primary_metric: float
    secondary_metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    commit_hash: str | None = None
    duration_seconds: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error_message: str | None = None

    def to_dict(self) -> dict:
        return {
            "primary_metric": self.primary_metric,
            "secondary_metrics": self.secondary_metrics,
            "artifacts": self.artifacts,
            "commit_hash": self.commit_hash,
            "duration_seconds": self.duration_seconds,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cost_usd": self.cost_usd,
            "error_message": self.error_message,
        }


@dataclass
class Experiment:
    """A single experiment with specific parameters."""
    experiment_id: str
    dimension: str
    params: dict[str, Any]
    baseline_commit: str
    status: ExperimentStatus = ExperimentStatus.PLANNED
    worker_id: str | None = None
    result: ExperimentResult | None = None
    branch_id: str | None = None  # for beam search branching
    is_stale: bool = False  # marked when a breakthrough invalidates this experiment

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "dimension": self.dimension,
            "params": self.params,
            "baseline_commit": self.baseline_commit,
            "status": self.status.value,
            "worker_id": self.worker_id,
            "result": self.result.to_dict() if self.result else None,
            "branch_id": self.branch_id,
        }


@dataclass
class WorkerState:
    """Tracks the state of a worker."""
    worker_id: str
    resource: str = ""
    status: WorkerStatus = WorkerStatus.IDLE
    current_experiment_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "resource": self.resource,
            "status": self.status.value,
            "current_experiment_id": self.current_experiment_id,
        }


@dataclass
class Baseline:
    """A known-good state to branch experiments from."""
    commit: str
    metric_value: float
    metric_name: str
    branch_id: str | None = None  # for beam search

    def to_dict(self) -> dict:
        return {
            "commit": self.commit,
            "metric_value": self.metric_value,
            "metric_name": self.metric_name,
            "branch_id": self.branch_id,
        }


@dataclass
class BudgetConfig:
    """Budget constraints for a run."""
    max_api_cost: float | None = None
    max_experiments: int | None = None
    max_wall_time_seconds: float | None = None
    max_plateau_iterations: int | None = None  # stop after N iterations with no improvement (None = no limit)

    def to_dict(self) -> dict:
        return {
            "max_api_cost": self.max_api_cost,
            "max_experiments": self.max_experiments,
            "max_wall_time_seconds": self.max_wall_time_seconds,
            "max_plateau_iterations": self.max_plateau_iterations,
        }


@dataclass
class Run:
    """A complete experimentation session."""
    run_id: str
    workload_name: str
    budget: BudgetConfig
    mode: str = "parallel"  # "sequential" | "parallel"
    experiments: list[Experiment] = field(default_factory=list)
    workers: list[WorkerState] = field(default_factory=list)
    baselines: list[Baseline] = field(default_factory=list)
    dimensions_explored: list[str] = field(default_factory=list)
    current_iteration: int = 0
    total_cost_usd: float = 0.0
    total_experiments_run: int = 0
    start_time: float | None = None
    end_time: float | None = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/core/models.py tests/test_models.py
git commit -m "feat: add core data models for experiments, workers, runs, dimensions"
```

---

### Task 3: Implement state machine transitions

**Files:**
- Create: `chaosengineer/core/state.py`
- Create: `tests/test_state.py`

- [ ] **Step 1: Write failing tests for state transitions**

`tests/test_state.py`:
```python
"""Tests for state machine transitions."""

import pytest
from chaosengineer.core.models import (
    Experiment, ExperimentStatus, WorkerState, WorkerStatus,
)
from chaosengineer.core.state import (
    InvalidTransitionError,
    assign_experiment,
    start_experiment,
    complete_experiment,
    fail_experiment,
    kill_experiment,
    assign_worker,
    release_worker,
    terminate_worker,
)


class TestExperimentTransitions:
    def _make_experiment(self, status=ExperimentStatus.PLANNED):
        exp = Experiment(
            experiment_id="exp-001",
            dimension="lr",
            params={"lr": 0.08},
            baseline_commit="abc1234",
        )
        exp.status = status
        return exp

    def test_assign_from_planned(self):
        exp = self._make_experiment()
        assign_experiment(exp, worker_id="w1")
        assert exp.status == ExperimentStatus.ASSIGNED
        assert exp.worker_id == "w1"

    def test_assign_from_running_raises(self):
        exp = self._make_experiment(ExperimentStatus.RUNNING)
        with pytest.raises(InvalidTransitionError):
            assign_experiment(exp, worker_id="w1")

    def test_start_from_assigned(self):
        exp = self._make_experiment(ExperimentStatus.ASSIGNED)
        start_experiment(exp)
        assert exp.status == ExperimentStatus.RUNNING

    def test_start_from_planned_raises(self):
        exp = self._make_experiment(ExperimentStatus.PLANNED)
        with pytest.raises(InvalidTransitionError):
            start_experiment(exp)

    def test_complete_from_running(self):
        from chaosengineer.core.models import ExperimentResult
        exp = self._make_experiment(ExperimentStatus.RUNNING)
        result = ExperimentResult(primary_metric=0.93)
        complete_experiment(exp, result)
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.result.primary_metric == 0.93

    def test_fail_from_running(self):
        from chaosengineer.core.models import ExperimentResult
        exp = self._make_experiment(ExperimentStatus.RUNNING)
        result = ExperimentResult(primary_metric=0.0, error_message="OOM")
        fail_experiment(exp, result)
        assert exp.status == ExperimentStatus.FAILED

    def test_kill_from_running(self):
        exp = self._make_experiment(ExperimentStatus.RUNNING)
        kill_experiment(exp)
        assert exp.status == ExperimentStatus.KILLED

    def test_kill_from_assigned(self):
        exp = self._make_experiment(ExperimentStatus.ASSIGNED)
        kill_experiment(exp)
        assert exp.status == ExperimentStatus.KILLED

    def test_complete_from_planned_raises(self):
        from chaosengineer.core.models import ExperimentResult
        exp = self._make_experiment(ExperimentStatus.PLANNED)
        with pytest.raises(InvalidTransitionError):
            complete_experiment(exp, ExperimentResult(primary_metric=0.0))


class TestWorkerTransitions:
    def _make_worker(self, status=WorkerStatus.IDLE):
        w = WorkerState(worker_id="w1", resource="GPU:0")
        w.status = status
        return w

    def test_assign_from_idle(self):
        w = self._make_worker()
        assign_worker(w, experiment_id="exp-001")
        assert w.status == WorkerStatus.BUSY
        assert w.current_experiment_id == "exp-001"

    def test_assign_from_busy_raises(self):
        w = self._make_worker(WorkerStatus.BUSY)
        with pytest.raises(InvalidTransitionError):
            assign_worker(w, experiment_id="exp-002")

    def test_release_from_busy(self):
        w = self._make_worker(WorkerStatus.BUSY)
        w.current_experiment_id = "exp-001"
        release_worker(w)
        assert w.status == WorkerStatus.IDLE
        assert w.current_experiment_id is None

    def test_terminate_from_busy(self):
        w = self._make_worker(WorkerStatus.BUSY)
        terminate_worker(w)
        assert w.status == WorkerStatus.TERMINATED

    def test_terminate_from_idle(self):
        w = self._make_worker(WorkerStatus.IDLE)
        terminate_worker(w)
        assert w.status == WorkerStatus.TERMINATED

    def test_assign_terminated_raises(self):
        w = self._make_worker(WorkerStatus.TERMINATED)
        with pytest.raises(InvalidTransitionError):
            assign_worker(w, experiment_id="exp-001")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_state.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement state machine**

`chaosengineer/core/state.py`:
```python
"""State machine transitions for experiments and workers."""

from __future__ import annotations

from chaosengineer.core.models import (
    Experiment,
    ExperimentResult,
    ExperimentStatus,
    WorkerState,
    WorkerStatus,
)


class InvalidTransitionError(Exception):
    """Raised when a state transition is not allowed."""

    def __init__(self, entity_id: str, current: str, target: str):
        super().__init__(
            f"Invalid transition for {entity_id}: {current} -> {target}"
        )


# Valid transitions for experiments
_EXPERIMENT_TRANSITIONS: dict[ExperimentStatus, set[ExperimentStatus]] = {
    ExperimentStatus.PLANNED: {ExperimentStatus.ASSIGNED, ExperimentStatus.KILLED},
    ExperimentStatus.ASSIGNED: {ExperimentStatus.RUNNING, ExperimentStatus.KILLED},
    ExperimentStatus.RUNNING: {
        ExperimentStatus.COMPLETED,
        ExperimentStatus.FAILED,
        ExperimentStatus.KILLED,
    },
    ExperimentStatus.COMPLETED: set(),
    ExperimentStatus.FAILED: set(),
    ExperimentStatus.KILLED: set(),
}

# Valid transitions for workers
_WORKER_TRANSITIONS: dict[WorkerStatus, set[WorkerStatus]] = {
    WorkerStatus.IDLE: {WorkerStatus.BUSY, WorkerStatus.TERMINATED},
    WorkerStatus.BUSY: {WorkerStatus.IDLE, WorkerStatus.TERMINATED},
    WorkerStatus.TERMINATED: set(),
}


def _check_experiment_transition(
    exp: Experiment, target: ExperimentStatus
) -> None:
    allowed = _EXPERIMENT_TRANSITIONS.get(exp.status, set())
    if target not in allowed:
        raise InvalidTransitionError(exp.experiment_id, exp.status.value, target.value)


def _check_worker_transition(
    worker: WorkerState, target: WorkerStatus
) -> None:
    allowed = _WORKER_TRANSITIONS.get(worker.status, set())
    if target not in allowed:
        raise InvalidTransitionError(worker.worker_id, worker.status.value, target.value)


# --- Experiment transitions ---


def assign_experiment(exp: Experiment, worker_id: str) -> None:
    _check_experiment_transition(exp, ExperimentStatus.ASSIGNED)
    exp.status = ExperimentStatus.ASSIGNED
    exp.worker_id = worker_id


def start_experiment(exp: Experiment) -> None:
    _check_experiment_transition(exp, ExperimentStatus.RUNNING)
    exp.status = ExperimentStatus.RUNNING


def complete_experiment(exp: Experiment, result: ExperimentResult) -> None:
    _check_experiment_transition(exp, ExperimentStatus.COMPLETED)
    exp.status = ExperimentStatus.COMPLETED
    exp.result = result


def fail_experiment(exp: Experiment, result: ExperimentResult) -> None:
    _check_experiment_transition(exp, ExperimentStatus.FAILED)
    exp.status = ExperimentStatus.FAILED
    exp.result = result


def kill_experiment(exp: Experiment) -> None:
    _check_experiment_transition(exp, ExperimentStatus.KILLED)
    exp.status = ExperimentStatus.KILLED


# --- Worker transitions ---


def assign_worker(worker: WorkerState, experiment_id: str) -> None:
    _check_worker_transition(worker, WorkerStatus.BUSY)
    worker.status = WorkerStatus.BUSY
    worker.current_experiment_id = experiment_id


def release_worker(worker: WorkerState) -> None:
    _check_worker_transition(worker, WorkerStatus.IDLE)
    worker.status = WorkerStatus.IDLE
    worker.current_experiment_id = None


def terminate_worker(worker: WorkerState) -> None:
    _check_worker_transition(worker, WorkerStatus.TERMINATED)
    worker.status = WorkerStatus.TERMINATED
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_state.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/core/state.py tests/test_state.py
git commit -m "feat: add state machine transitions for experiments and workers"
```

---

## Chunk 2: Event Logger & Budget Tracker

### Task 4: Implement JSONL event logger

**Files:**
- Create: `chaosengineer/metrics/logger.py`
- Create: `tests/test_logger.py`

- [ ] **Step 1: Write failing tests for the event logger**

`tests/test_logger.py`:
```python
"""Tests for JSONL event logger."""

import json
import time
from pathlib import Path

import pytest
from chaosengineer.metrics.logger import EventLogger, Event


class TestEventLogger:
    def test_log_creates_file(self, tmp_output_dir):
        logger = EventLogger(tmp_output_dir / "events.jsonl")
        logger.log(Event(event="run_started", data={"workload": "test"}))
        assert (tmp_output_dir / "events.jsonl").exists()

    def test_log_writes_valid_jsonl(self, tmp_output_dir):
        logger = EventLogger(tmp_output_dir / "events.jsonl")
        logger.log(Event(event="run_started", data={"workload": "test"}))
        logger.log(Event(event="iteration_started", data={"dimension": "lr"}))

        lines = (tmp_output_dir / "events.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

        event1 = json.loads(lines[0])
        assert event1["event"] == "run_started"
        assert event1["workload"] == "test"  # flat format, no 'data' envelope
        assert "ts" in event1

        event2 = json.loads(lines[1])
        assert event2["event"] == "iteration_started"

    def test_read_events(self, tmp_output_dir):
        logger = EventLogger(tmp_output_dir / "events.jsonl")
        logger.log(Event(event="a", data={}))
        logger.log(Event(event="b", data={"x": 1}))

        events = logger.read_events()
        assert len(events) == 2
        assert events[0]["event"] == "a"
        assert events[1]["x"] == 1  # flat format

    def test_read_events_filtered(self, tmp_output_dir):
        logger = EventLogger(tmp_output_dir / "events.jsonl")
        logger.log(Event(event="run_started", data={}))
        logger.log(Event(event="worker_completed", data={"w": 1}))
        logger.log(Event(event="worker_completed", data={"w": 2}))

        events = logger.read_events(event_type="worker_completed")
        assert len(events) == 2
        assert all(e["event"] == "worker_completed" for e in events)

    def test_empty_file_read(self, tmp_output_dir):
        logger = EventLogger(tmp_output_dir / "events.jsonl")
        events = logger.read_events()
        assert events == []

    def test_timestamp_is_iso(self, tmp_output_dir):
        logger = EventLogger(tmp_output_dir / "events.jsonl")
        logger.log(Event(event="test", data={}))

        events = logger.read_events()
        ts = events[0]["ts"]
        # Should be parseable as ISO format
        from datetime import datetime
        datetime.fromisoformat(ts)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_logger.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the event logger**

`chaosengineer/metrics/logger.py`:
```python
"""JSONL event logger for ChaosEngineer runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class Event:
    """A single event to log."""
    event: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: str | None = None  # ISO timestamp, auto-generated if None


class EventLogger:
    """Append-only JSONL event logger.

    Events are stored in flat format matching the spec:
    {"ts": "...", "event": "run_started", "workload": "test", ...}
    Event-specific fields are merged at the top level (no 'data' envelope).
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def log(self, event: Event) -> None:
        ts = event.ts or datetime.now(timezone.utc).isoformat()
        record = {"ts": ts, "event": event.event, **event.data}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def read_events(self, event_type: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        events = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if event_type is None or record.get("event") == event_type:
                    events.append(record)
        return events
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_logger.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/metrics/logger.py tests/test_logger.py
git commit -m "feat: add JSONL event logger for run observability"
```

---

### Task 5: Implement budget tracker

**Files:**
- Create: `chaosengineer/core/budget.py`
- Create: `tests/test_budget.py`

- [ ] **Step 1: Write failing tests for budget tracking**

`tests/test_budget.py`:
```python
"""Tests for budget tracking and enforcement."""

import time
import pytest
from chaosengineer.core.models import BudgetConfig
from chaosengineer.core.budget import BudgetTracker


class TestBudgetTracker:
    def test_no_limits(self):
        tracker = BudgetTracker(BudgetConfig())
        assert not tracker.is_exhausted()

    def test_cost_tracking(self):
        tracker = BudgetTracker(BudgetConfig(max_api_cost=10.0))
        tracker.add_cost(3.0)
        assert tracker.spent_usd == 3.0
        assert tracker.remaining_cost == 7.0
        assert not tracker.is_exhausted()

    def test_cost_exhaustion(self):
        tracker = BudgetTracker(BudgetConfig(max_api_cost=10.0))
        tracker.add_cost(10.0)
        assert tracker.is_exhausted()

    def test_experiment_count_tracking(self):
        tracker = BudgetTracker(BudgetConfig(max_experiments=5))
        tracker.record_experiment()
        tracker.record_experiment()
        assert tracker.experiments_run == 2
        assert tracker.remaining_experiments == 3
        assert not tracker.is_exhausted()

    def test_experiment_count_exhaustion(self):
        tracker = BudgetTracker(BudgetConfig(max_experiments=2))
        tracker.record_experiment()
        tracker.record_experiment()
        assert tracker.is_exhausted()

    def test_time_exhaustion(self):
        tracker = BudgetTracker(BudgetConfig(max_wall_time_seconds=1.0))
        tracker.start()
        # Simulate elapsed time by backdating start
        tracker._start_time = time.monotonic() - 2.0
        assert tracker.is_exhausted()

    def test_time_not_started(self):
        tracker = BudgetTracker(BudgetConfig(max_wall_time_seconds=100.0))
        # Not started yet, should not be exhausted
        assert not tracker.is_exhausted()

    def test_plateau_tracking(self):
        tracker = BudgetTracker(BudgetConfig(max_plateau_iterations=3))
        tracker.record_no_improvement()
        tracker.record_no_improvement()
        assert not tracker.is_exhausted()
        tracker.record_no_improvement()
        assert tracker.is_exhausted()

    def test_plateau_resets_on_improvement(self):
        tracker = BudgetTracker(BudgetConfig(max_plateau_iterations=3))
        tracker.record_no_improvement()
        tracker.record_no_improvement()
        tracker.record_improvement()
        assert tracker.consecutive_no_improvement == 0
        tracker.record_no_improvement()
        assert not tracker.is_exhausted()

    def test_multiple_limits_any_exhausts(self):
        tracker = BudgetTracker(BudgetConfig(max_api_cost=100.0, max_experiments=2))
        tracker.record_experiment()
        tracker.record_experiment()
        # Experiments exhausted, even though cost is fine
        assert tracker.is_exhausted()

    def test_snapshot(self):
        tracker = BudgetTracker(BudgetConfig(max_api_cost=50.0, max_experiments=100))
        tracker.add_cost(5.0)
        tracker.record_experiment()
        snap = tracker.snapshot()
        assert snap["spent_usd"] == 5.0
        assert snap["remaining_cost"] == 45.0
        assert snap["experiments_run"] == 1
        assert snap["remaining_experiments"] == 99
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_budget.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement budget tracker**

`chaosengineer/core/budget.py`:
```python
"""Budget tracking and enforcement."""

from __future__ import annotations

import time
from typing import Any

from chaosengineer.core.models import BudgetConfig


class BudgetTracker:
    """Tracks spending against budget limits."""

    def __init__(self, config: BudgetConfig):
        self.config = config
        self.spent_usd: float = 0.0
        self.experiments_run: int = 0
        self.consecutive_no_improvement: int = 0
        self._start_time: float | None = None

    def start(self) -> None:
        self._start_time = time.monotonic()

    def add_cost(self, cost_usd: float) -> None:
        self.spent_usd += cost_usd

    def record_experiment(self) -> None:
        self.experiments_run += 1

    def record_no_improvement(self) -> None:
        self.consecutive_no_improvement += 1

    def record_improvement(self) -> None:
        self.consecutive_no_improvement = 0

    @property
    def remaining_cost(self) -> float | None:
        if self.config.max_api_cost is None:
            return None
        return max(0.0, self.config.max_api_cost - self.spent_usd)

    @property
    def remaining_experiments(self) -> int | None:
        if self.config.max_experiments is None:
            return None
        return max(0, self.config.max_experiments - self.experiments_run)

    @property
    def elapsed_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    @property
    def remaining_time(self) -> float | None:
        if self.config.max_wall_time_seconds is None:
            return None
        if self._start_time is None:
            return self.config.max_wall_time_seconds
        return max(0.0, self.config.max_wall_time_seconds - self.elapsed_seconds)

    def is_exhausted(self) -> bool:
        if self.config.max_api_cost is not None and self.spent_usd >= self.config.max_api_cost:
            return True
        if self.config.max_experiments is not None and self.experiments_run >= self.config.max_experiments:
            return True
        if (
            self.config.max_wall_time_seconds is not None
            and self._start_time is not None
            and self.elapsed_seconds >= self.config.max_wall_time_seconds
        ):
            return True
        if (
            self.config.max_plateau_iterations is not None
            and self.consecutive_no_improvement >= self.config.max_plateau_iterations
        ):
            return True
        return False

    def snapshot(self) -> dict[str, Any]:
        return {
            "spent_usd": self.spent_usd,
            "remaining_cost": self.remaining_cost,
            "experiments_run": self.experiments_run,
            "remaining_experiments": self.remaining_experiments,
            "elapsed_seconds": self.elapsed_seconds,
            "remaining_time": self.remaining_time,
            "consecutive_no_improvement": self.consecutive_no_improvement,
            "is_exhausted": self.is_exhausted(),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_budget.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/core/budget.py tests/test_budget.py
git commit -m "feat: add budget tracker with cost, experiment, time, and plateau limits"
```

---

## Chunk 3: Workload Spec Parser

### Task 6: Implement markdown workload spec parser

**Files:**
- Create: `chaosengineer/workloads/parser.py`
- Create: `tests/test_parser.py`

- [ ] **Step 1: Write a sample workload spec for testing**

Create `tests/fixtures/sample_workload.md`:
```markdown
# Workload: Neural Network Architecture Search

## Context
Training a small language model on climbmix-400b dataset. The goal is to find
the best hyperparameters and architecture choices to minimize val_bpb within
a fixed 5-minute training budget.

## Experiment Space
- Directional: "learning_rate" (currently 0.04)
- Directional: "depth" (currently 8)
- Enum: "activation" options: GeLU, SiLU, ReLU
- Diverse: "attention_mechanism"
Constraint: depth * 64 must stay under 1024 for memory

## Execution
- Command: `uv run train.py > run.log 2>&1`
- Time budget per experiment: 5 minutes

## Evaluation
- Type: automatic
- Metric: val_bpb (lower is better)
- Parse: `grep "^val_bpb:" run.log`
- Secondary metrics: peak_vram_mb

## Resources
- Per worker: 1 GPU
- Available: 4

## Budget
- Max API cost: $50
- Max experiments: 100
- Max wall time: 8h

## Constraints
- Files workers may modify: train.py
- Do not modify prepare.py
```

- [ ] **Step 2: Write failing tests for the parser**

`tests/test_parser.py`:
```python
"""Tests for workload spec parser."""

import pytest
from pathlib import Path
from chaosengineer.workloads.parser import parse_workload_spec, WorkloadSpec
from chaosengineer.core.models import DimensionType


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestParseWorkloadSpec:
    def test_parse_name(self):
        spec = parse_workload_spec(FIXTURES_DIR / "sample_workload.md")
        assert spec.name == "Neural Network Architecture Search"

    def test_parse_context(self):
        spec = parse_workload_spec(FIXTURES_DIR / "sample_workload.md")
        assert "climbmix-400b" in spec.context

    def test_parse_dimensions(self):
        spec = parse_workload_spec(FIXTURES_DIR / "sample_workload.md")
        dims = {d.name: d for d in spec.dimensions}
        assert "learning_rate" in dims
        assert dims["learning_rate"].dim_type == DimensionType.DIRECTIONAL
        assert dims["learning_rate"].current_value == 0.04
        assert "depth" in dims
        assert dims["depth"].dim_type == DimensionType.DIRECTIONAL
        assert dims["depth"].current_value == 8.0
        assert "activation" in dims
        assert dims["activation"].dim_type == DimensionType.ENUM
        assert dims["activation"].options == ["GeLU", "SiLU", "ReLU"]
        assert "attention_mechanism" in dims
        assert dims["attention_mechanism"].dim_type == DimensionType.DIVERSE

    def test_parse_execution(self):
        spec = parse_workload_spec(FIXTURES_DIR / "sample_workload.md")
        assert "uv run train.py" in spec.execution_command
        assert spec.time_budget_seconds == 300  # 5 minutes

    def test_parse_evaluation(self):
        spec = parse_workload_spec(FIXTURES_DIR / "sample_workload.md")
        assert spec.evaluation_type == "automatic"
        assert spec.primary_metric == "val_bpb"
        assert spec.metric_direction == "lower"
        assert 'grep "^val_bpb:" run.log' in spec.metric_parse_command

    def test_parse_resources(self):
        spec = parse_workload_spec(FIXTURES_DIR / "sample_workload.md")
        assert spec.workers_available == 4

    def test_parse_budget(self):
        spec = parse_workload_spec(FIXTURES_DIR / "sample_workload.md")
        assert spec.budget.max_api_cost == 50.0
        assert spec.budget.max_experiments == 100
        assert spec.budget.max_wall_time_seconds == 28800  # 8h

    def test_parse_constraints(self):
        spec = parse_workload_spec(FIXTURES_DIR / "sample_workload.md")
        assert "train.py" in spec.modifiable_files
        assert "prepare.py" in spec.constraints_text

    def test_parse_from_string(self):
        md = """# Workload: Simple Test

## Context
A simple test workload.

## Experiment Space
- Directional: "value" (currently 10)

## Execution
- Command: `echo hello`
- Time budget per experiment: 1 minute

## Evaluation
- Type: automatic
- Metric: score (higher is better)
- Parse: `grep score output.txt`

## Resources
- Available: 2

## Budget
- Max experiments: 5
"""
        spec = parse_workload_spec(content=md)
        assert spec.name == "Simple Test"
        assert len(spec.dimensions) == 1
        assert spec.metric_direction == "higher"
        assert spec.workers_available == 2

    def test_is_better_lower(self):
        spec = parse_workload_spec(FIXTURES_DIR / "sample_workload.md")
        assert spec.is_better(0.91, 0.95)
        assert not spec.is_better(0.95, 0.91)

    def test_is_better_higher(self):
        md = """# Workload: Score Test

## Experiment Space

## Execution
- Command: `echo`

## Evaluation
- Type: automatic
- Metric: accuracy (higher is better)

## Resources
- Available: 1
"""
        spec = parse_workload_spec(content=md)
        assert spec.is_better(0.95, 0.91)
        assert not spec.is_better(0.91, 0.95)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_parser.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement the parser**

`chaosengineer/workloads/parser.py`:
```python
"""Markdown workload spec parser."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from chaosengineer.core.models import BudgetConfig, DimensionSpec, DimensionType


@dataclass
class WorkloadSpec:
    """Parsed workload specification."""
    name: str = ""
    context: str = ""
    dimensions: list[DimensionSpec] = field(default_factory=list)
    execution_command: str = ""
    time_budget_seconds: float = 300
    evaluation_type: str = "automatic"  # "automatic" | "human"
    primary_metric: str = ""
    metric_direction: str = "lower"  # "lower" | "higher"
    metric_parse_command: str = ""
    secondary_metrics: list[str] = field(default_factory=list)
    workers_available: int = 1
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    modifiable_files: list[str] = field(default_factory=list)
    constraints_text: str = ""
    raw_markdown: str = ""

    def is_better(self, new_value: float, old_value: float) -> bool:
        if self.metric_direction == "lower":
            return new_value < old_value
        return new_value > old_value


def _extract_sections(markdown: str) -> dict[str, str]:
    """Split markdown into {heading: content} pairs."""
    sections: dict[str, str] = {}
    current_heading = ""
    current_lines: list[str] = []

    for line in markdown.split("\n"):
        heading_match = re.match(r"^##\s+(.+)$", line)
        if heading_match:
            if current_heading:
                sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = heading_match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_heading:
        sections[current_heading] = "\n".join(current_lines).strip()

    return sections


def _parse_name(markdown: str) -> str:
    match = re.search(r"^#\s+Workload:\s*(.+)$", markdown, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _parse_dimensions(text: str) -> list[DimensionSpec]:
    dims = []
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("-"):
            continue

        line = line.lstrip("- ").strip()

        # Directional: "name" (currently value)
        dir_match = re.match(
            r'Directional:\s*"(\w+)"\s*\(currently\s+([\d.]+)\)', line
        )
        if dir_match:
            dims.append(DimensionSpec(
                name=dir_match.group(1),
                dim_type=DimensionType.DIRECTIONAL,
                current_value=float(dir_match.group(2)),
            ))
            continue

        # Enum: "name" options: A, B, C
        enum_match = re.match(
            r'Enum:\s*"(\w+)"\s*options:\s*(.+)', line
        )
        if enum_match:
            options = [o.strip() for o in enum_match.group(2).split(",")]
            dims.append(DimensionSpec(
                name=enum_match.group(1),
                dim_type=DimensionType.ENUM,
                options=options,
            ))
            continue

        # Diverse: "name"
        diverse_match = re.match(r'Diverse:\s*"(\w+)"', line)
        if diverse_match:
            dims.append(DimensionSpec(
                name=diverse_match.group(1),
                dim_type=DimensionType.DIVERSE,
            ))
            continue

    return dims


def _parse_time_budget(text: str) -> float:
    """Parse time budget, returns seconds."""
    match = re.search(r"Time budget.*?:\s*(\d+)\s*(minutes?|seconds?|hours?)", text, re.IGNORECASE)
    if not match:
        return 300  # default 5 minutes
    value = int(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("hour"):
        return value * 3600
    if unit.startswith("minute"):
        return value * 60
    return float(value)


def _parse_command(text: str) -> str:
    match = re.search(r"Command:\s*`([^`]+)`", text)
    return match.group(1) if match else ""


def _parse_evaluation(text: str) -> tuple[str, str, str, str]:
    """Returns (eval_type, metric_name, direction, parse_command)."""
    eval_type = "automatic"
    if re.search(r"Type:\s*human", text, re.IGNORECASE):
        eval_type = "human"

    metric_name = ""
    direction = "lower"
    metric_match = re.search(r"Metric:\s*(\w+)\s*\((\w+)\s+is\s+better\)", text)
    if metric_match:
        metric_name = metric_match.group(1)
        direction = metric_match.group(2).lower()

    parse_cmd = ""
    parse_match = re.search(r"Parse:\s*`([^`]+)`", text)
    if parse_match:
        parse_cmd = parse_match.group(1)

    return eval_type, metric_name, direction, parse_cmd


def _parse_workers_available(text: str) -> int:
    match = re.search(r"Available:\s*(\d+)", text)
    return int(match.group(1)) if match else 1


def _parse_budget(text: str) -> BudgetConfig:
    max_cost = None
    cost_match = re.search(r"Max API cost:\s*\$?([\d.]+)", text)
    if cost_match:
        max_cost = float(cost_match.group(1))

    max_experiments = None
    exp_match = re.search(r"Max experiments:\s*(\d+)", text)
    if exp_match:
        max_experiments = int(exp_match.group(1))

    max_time = None
    time_match = re.search(r"Max wall time:\s*(\d+)\s*(h|hours?|m|minutes?|s|seconds?)", text, re.IGNORECASE)
    if time_match:
        value = int(time_match.group(1))
        unit = time_match.group(2).lower()
        if unit.startswith("h"):
            max_time = value * 3600.0
        elif unit.startswith("m"):
            max_time = value * 60.0
        else:
            max_time = float(value)

    return BudgetConfig(
        max_api_cost=max_cost,
        max_experiments=max_experiments,
        max_wall_time_seconds=max_time,
    )


def _parse_modifiable_files(text: str) -> list[str]:
    files = []
    match = re.search(r"Files workers may modify:\s*(.+)", text)
    if match:
        files = [f.strip() for f in match.group(1).split(",")]
    return files


def parse_workload_spec(
    path: Path | str | None = None,
    content: str | None = None,
) -> WorkloadSpec:
    """Parse a markdown workload spec from a file path or string content."""
    if content is None:
        if path is None:
            raise ValueError("Either path or content must be provided")
        content = Path(path).read_text()

    sections = _extract_sections(content)

    # Execution
    exec_section = sections.get("Execution", "")
    command = _parse_command(exec_section)
    time_budget = _parse_time_budget(exec_section)

    # Evaluation
    eval_section = sections.get("Evaluation", "")
    eval_type, metric_name, direction, parse_cmd = _parse_evaluation(eval_section)

    # Resources
    resources_section = sections.get("Resources", "")
    workers = _parse_workers_available(resources_section)

    # Budget
    budget_section = sections.get("Budget", "")
    budget = _parse_budget(budget_section)

    # Constraints
    constraints_section = sections.get("Constraints", "")
    modifiable_files = _parse_modifiable_files(constraints_section)

    return WorkloadSpec(
        name=_parse_name(content),
        context=sections.get("Context", ""),
        dimensions=_parse_dimensions(sections.get("Experiment Space", "")),
        execution_command=command,
        time_budget_seconds=time_budget,
        evaluation_type=eval_type,
        primary_metric=metric_name,
        metric_direction=direction,
        metric_parse_command=parse_cmd,
        workers_available=workers,
        budget=budget,
        modifiable_files=modifiable_files,
        constraints_text=constraints_section,
        raw_markdown=content,
    )
```

- [ ] **Step 5: Create the fixture directory and file**

```bash
mkdir -p tests/fixtures
```

Then write the sample workload spec from Step 1 to `tests/fixtures/sample_workload.md`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_parser.py -v`
Expected: All tests PASS.

- [ ] **Step 7: Commit**

```bash
git add chaosengineer/workloads/parser.py tests/test_parser.py tests/fixtures/
git commit -m "feat: add markdown workload spec parser"
```

---

## Chunk 4: Testing Framework (Simulator & Executor Interfaces)

### Task 7: Implement decision maker and executor interfaces

**Files:**
- Create: `chaosengineer/testing/simulator.py`
- Create: `chaosengineer/testing/executor.py`
- Create: `tests/test_simulator.py`

- [ ] **Step 1: Write failing tests**

`tests/test_simulator.py`:
```python
"""Tests for decision maker and executor interfaces."""

import pytest
from chaosengineer.core.models import DimensionType, DimensionSpec, ExperimentResult
from chaosengineer.testing.simulator import (
    DecisionMaker,
    ScriptedDecisionMaker,
    DimensionPlan,
)
from chaosengineer.testing.executor import (
    ExperimentExecutor,
    ScriptedExecutor,
)


class TestScriptedDecisionMaker:
    def test_returns_scripted_dimensions(self):
        plans = [
            DimensionPlan(
                dimension_name="lr",
                values=[{"lr": 0.02}, {"lr": 0.08}],
            ),
            DimensionPlan(
                dimension_name="depth",
                values=[{"depth": 6}, {"depth": 12}],
            ),
        ]
        dm = ScriptedDecisionMaker(plans)
        plan = dm.pick_next_dimension(
            dimensions=[], baselines=[], history=[]
        )
        assert plan.dimension_name == "lr"
        assert len(plan.values) == 2

    def test_exhausted_returns_none(self):
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}]),
        ]
        dm = ScriptedDecisionMaker(plans)
        dm.pick_next_dimension([], [], [])
        result = dm.pick_next_dimension([], [], [])
        assert result is None

    def test_pick_diverse_options(self):
        dm = ScriptedDecisionMaker(
            plans=[],
            diverse_options={"strategy": ["A", "B", "C"]},
        )
        options = dm.discover_diverse_options("strategy", context="")
        assert options == ["A", "B", "C"]


class TestScriptedExecutor:
    def test_returns_scripted_result(self):
        results = {
            "exp-001": ExperimentResult(primary_metric=0.93, duration_seconds=300),
            "exp-002": ExperimentResult(primary_metric=0.95, duration_seconds=300),
        }
        executor = ScriptedExecutor(results)
        result = executor.run_experiment(
            experiment_id="exp-001",
            params={"lr": 0.02},
            command="echo",
            baseline_commit="abc",
        )
        assert result.primary_metric == 0.93

    def test_missing_experiment_raises(self):
        executor = ScriptedExecutor({})
        with pytest.raises(KeyError):
            executor.run_experiment("unknown", {}, "echo", "abc")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_simulator.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement simulator and executor**

`chaosengineer/testing/simulator.py`:
```python
"""Decision maker interface and scripted implementation for testing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from chaosengineer.core.models import Baseline, DimensionSpec


@dataclass
class DimensionPlan:
    """A plan to explore one dimension."""
    dimension_name: str
    values: list[dict[str, Any]]  # one dict per worker


class DecisionMaker(ABC):
    """Interface for making experiment planning decisions.

    In real mode, this calls an LLM. In test mode, returns scripted responses.
    """

    @abstractmethod
    def pick_next_dimension(
        self,
        dimensions: list[DimensionSpec],
        baselines: list[Baseline],
        history: list[dict],
    ) -> DimensionPlan | None:
        """Pick the next dimension to explore. Returns None if done."""

    @abstractmethod
    def discover_diverse_options(
        self, dimension_name: str, context: str
    ) -> list[str]:
        """Discover the saturated set for a diverse dimension."""


class ScriptedDecisionMaker(DecisionMaker):
    """Returns pre-scripted decisions for testing."""

    def __init__(
        self,
        plans: list[DimensionPlan],
        diverse_options: dict[str, list[str]] | None = None,
    ):
        self._plans = list(plans)
        self._plan_index = 0
        self._diverse_options = diverse_options or {}

    def pick_next_dimension(
        self,
        dimensions: list[DimensionSpec],
        baselines: list[Baseline],
        history: list[dict],
    ) -> DimensionPlan | None:
        if self._plan_index >= len(self._plans):
            return None
        plan = self._plans[self._plan_index]
        self._plan_index += 1
        return plan

    def discover_diverse_options(
        self, dimension_name: str, context: str
    ) -> list[str]:
        return self._diverse_options.get(dimension_name, [])
```

`chaosengineer/testing/executor.py`:
```python
"""Experiment executor interface and scripted implementation for testing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from chaosengineer.core.models import ExperimentResult


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


class ScriptedExecutor(ExperimentExecutor):
    """Returns pre-scripted results for testing."""

    def __init__(self, results: dict[str, ExperimentResult]):
        self._results = results

    def run_experiment(
        self,
        experiment_id: str,
        params: dict[str, Any],
        command: str,
        baseline_commit: str,
        resource: str = "",
    ) -> ExperimentResult:
        if experiment_id not in self._results:
            raise KeyError(f"No scripted result for experiment {experiment_id}")
        return self._results[experiment_id]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_simulator.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/testing/simulator.py chaosengineer/testing/executor.py tests/test_simulator.py
git commit -m "feat: add decision maker and executor interfaces with scripted test implementations"
```

---

## Chunk 5: Coordinator Logic

### Task 8: Implement coordinator core loop

This is the central piece — the coordinator that picks dimensions, allocates workers, collects results, manages baselines, and handles beam search on ties.

**Files:**
- Create: `chaosengineer/core/coordinator.py`
- Create: `tests/test_coordinator.py`

- [ ] **Step 1: Write failing tests for basic coordinator flow**

`tests/test_coordinator.py`:
```python
"""Tests for coordinator logic."""

import pytest
from chaosengineer.core.models import (
    BudgetConfig, Baseline, DimensionSpec, DimensionType,
    ExperimentResult, ExperimentStatus,
)
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.budget import BudgetTracker
from chaosengineer.metrics.logger import EventLogger
from chaosengineer.testing.simulator import ScriptedDecisionMaker, DimensionPlan
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


class TestCoordinatorBasicFlow:
    """Test a simple 1-dimension, 2-worker directional sweep."""

    def test_single_iteration_picks_winner(self, tmp_output_dir):
        spec = _make_spec()
        plans = [
            DimensionPlan(
                dimension_name="lr",
                values=[{"lr": 0.02}, {"lr": 0.08}],
            ),
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.91),
            "exp-0-1": ExperimentResult(primary_metric=0.95),
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

        # Best baseline should be updated to 0.91 (lower is better)
        assert coordinator.best_baseline.metric_value == 0.91
        # 2 experiments should have run
        assert coordinator.budget.experiments_run == 2
        # Events should be logged
        events = coordinator.logger.read_events()
        assert any(e["event"] == "run_started" for e in events)
        assert any(e["event"] == "breakthrough" for e in events)

    def test_no_improvement_records_plateau(self, tmp_output_dir):
        spec = _make_spec()
        plans = [
            DimensionPlan(
                dimension_name="lr",
                values=[{"lr": 0.02}, {"lr": 0.08}],
            ),
        ]
        # Both results worse than baseline
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.99),
            "exp-0-1": ExperimentResult(primary_metric=0.98),
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

        # Baseline should not change
        assert coordinator.best_baseline.metric_value == 0.97
        assert coordinator.budget.consecutive_no_improvement == 1


class TestCoordinatorMultiIteration:
    """Test multiple iterations across dimensions."""

    def test_two_dimensions(self, tmp_output_dir):
        spec = _make_spec()
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}, {"lr": 0.08}]),
            DimensionPlan(dimension_name="depth", values=[{"depth": 6}, {"depth": 12}]),
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.91),
            "exp-0-1": ExperimentResult(primary_metric=0.95),
            "exp-1-0": ExperimentResult(primary_metric=0.89),
            "exp-1-1": ExperimentResult(primary_metric=0.92),
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

        assert coordinator.best_baseline.metric_value == 0.89
        assert coordinator.budget.experiments_run == 4


class TestCoordinatorBeamSearch:
    """Test branching on ties."""

    def test_tie_creates_branches(self, tmp_output_dir):
        spec = _make_spec(
            budget=BudgetConfig(max_experiments=100),
        )
        plans = [
            DimensionPlan(
                dimension_name="activation",
                values=[{"act": "A"}, {"act": "B"}, {"act": "C"}],
            ),
            # After tie between A and B, coordinator plans next dim for each branch
            DimensionPlan(dimension_name="depth_branchA", values=[{"depth": 6}]),
            DimensionPlan(dimension_name="depth_branchB", values=[{"depth": 6}]),
        ]
        results = {
            # A and B tie (within threshold), C is worse
            "exp-0-0": ExperimentResult(primary_metric=0.91),
            "exp-0-1": ExperimentResult(primary_metric=0.911),  # within 1% of 0.91
            "exp-0-2": ExperimentResult(primary_metric=0.98),
            # Branch explorations
            "exp-1-0": ExperimentResult(primary_metric=0.88),
            "exp-2-0": ExperimentResult(primary_metric=0.90),
        }
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="abc", metric_value=0.97, metric_name="val_bpb"),
            tie_threshold_pct=1.0,  # 1% threshold
        )

        coordinator.run()

        # Best should be 0.88 from branch A exploration
        assert coordinator.best_baseline.metric_value == 0.88


class TestCoordinatorBudgetEnforcement:
    """Test that the coordinator stops when budget is exhausted."""

    def test_stops_at_experiment_limit(self, tmp_output_dir):
        spec = _make_spec(budget=BudgetConfig(max_experiments=2))
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}, {"lr": 0.08}]),
            # This should never execute
            DimensionPlan(dimension_name="depth", values=[{"depth": 6}]),
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.93),
            "exp-0-1": ExperimentResult(primary_metric=0.95),
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
        # Second dimension should NOT have been explored
        events = coordinator.logger.read_events(event_type="iteration_started")
        assert len(events) == 1  # only one iteration

    def test_stops_on_plateau(self, tmp_output_dir):
        spec = _make_spec(budget=BudgetConfig(max_plateau_iterations=2))
        plans = [
            DimensionPlan(dimension_name="a", values=[{"a": 1}]),
            DimensionPlan(dimension_name="b", values=[{"b": 1}]),
            DimensionPlan(dimension_name="c", values=[{"c": 1}]),  # should not run
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.99),  # worse
            "exp-1-0": ExperimentResult(primary_metric=0.98),  # still worse
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
        assert coordinator.budget.consecutive_no_improvement == 2


class TestCoordinatorFailedExperiment:
    """Test handling of failed experiments."""

    def test_failure_does_not_crash_coordinator(self, tmp_output_dir):
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

        # Should pick the non-failed result
        assert coordinator.best_baseline.metric_value == 0.93
        # Failed experiment should be recorded
        failed = [
            e for e in coordinator.run_state.experiments
            if e.status == ExperimentStatus.FAILED
        ]
        assert len(failed) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_coordinator.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the coordinator**

`chaosengineer/core/coordinator.py`:
```python
"""Coordinator: the central orchestration loop."""

from __future__ import annotations

from chaosengineer.core.models import (
    Baseline,
    Experiment,
    ExperimentResult,
    ExperimentStatus,
    Run,
    WorkerState,
)
from chaosengineer.core.state import (
    assign_experiment,
    assign_worker,
    complete_experiment,
    fail_experiment,
    release_worker,
    start_experiment,
)
from chaosengineer.core.budget import BudgetTracker
from chaosengineer.metrics.logger import EventLogger, Event
from chaosengineer.testing.simulator import DecisionMaker, DimensionPlan
from chaosengineer.testing.executor import ExperimentExecutor
from chaosengineer.workloads.parser import WorkloadSpec


class Coordinator:
    """Runs the experiment loop: pick dimension, allocate workers, collect results."""

    def __init__(
        self,
        spec: WorkloadSpec,
        decision_maker: DecisionMaker,
        executor: ExperimentExecutor,
        logger: EventLogger,
        budget: BudgetTracker,
        initial_baseline: Baseline,
        tie_threshold_pct: float = 1.0,
    ):
        self.spec = spec
        self.decision_maker = decision_maker
        self.executor = executor
        self.logger = logger
        self.budget = budget
        self.best_baseline = initial_baseline
        self.tie_threshold_pct = tie_threshold_pct
        self.run_state = Run(
            run_id="run-001",
            workload_name=spec.name,
            budget=spec.budget,
            baselines=[initial_baseline],
        )
        self._iteration = 0

    def run(self) -> None:
        """Execute the coordinator loop until budget or dimensions exhausted."""
        self.logger.log(Event(
            event="run_started",
            data={
                "workload": self.spec.name,
                "budget": self.budget.config.to_dict(),
                "baseline": self.best_baseline.to_dict(),
            },
        ))
        self.budget.start()

        active_baselines = [self.best_baseline]

        while not self.budget.is_exhausted():
            # For each active baseline (beam search: may be >1 after ties),
            # ask the decision maker for a plan and run it.
            next_active = []
            for baseline in active_baselines:
                if self.budget.is_exhausted():
                    break

                plan = self.decision_maker.pick_next_dimension(
                    dimensions=self.spec.dimensions,
                    baselines=[baseline],
                    history=self.logger.read_events(),
                )
                if plan is None:
                    continue  # this branch has no more dimensions

                # Check if budget can accommodate this iteration
                if (
                    self.budget.config.max_experiments is not None
                    and self.budget.experiments_run + len(plan.values)
                    > self.budget.config.max_experiments
                ):
                    remaining = (
                        self.budget.config.max_experiments
                        - self.budget.experiments_run
                    )
                    if remaining <= 0:
                        break
                    plan = DimensionPlan(
                        dimension_name=plan.dimension_name,
                        values=plan.values[:remaining],
                    )

                self.logger.log(Event(
                    event="iteration_started",
                    data={
                        "dimension": plan.dimension_name,
                        "num_workers": len(plan.values),
                        "iteration": self._iteration,
                        "branch_id": baseline.branch_id,
                    },
                ))

                iteration_results = self._run_iteration(plan, baseline)

                branch_baselines = self._evaluate_iteration(
                    plan, iteration_results, [baseline]
                )
                next_active.extend(branch_baselines)

                self.logger.log(Event(
                    event="budget_checkpoint",
                    data=self.budget.snapshot(),
                ))

                self._iteration += 1

            if not next_active:
                break  # all branches exhausted
            active_baselines = next_active

        self.logger.log(Event(
            event="run_completed",
            data={
                "best_metric": self.best_baseline.metric_value,
                "total_experiments": self.budget.experiments_run,
                "total_cost_usd": self.budget.spent_usd,
            },
        ))

    def _run_iteration(
        self, plan: DimensionPlan, baseline: Baseline
    ) -> list[tuple[Experiment, ExperimentResult | None]]:
        """Run all experiments for one dimension sweep from a given baseline."""
        results = []
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

            # Create a temporary worker
            worker = WorkerState(worker_id=f"w-{self._iteration}-{i}")

            # State transitions
            assign_experiment(exp, worker.worker_id)
            assign_worker(worker, exp.experiment_id)
            start_experiment(exp)

            # Execute
            try:
                result = self.executor.run_experiment(
                    experiment_id=exp_id,
                    params=params,
                    command=self.spec.execution_command,
                    baseline_commit=baseline.commit,
                )
                if result.error_message:
                    fail_experiment(exp, result)
                    self.logger.log(Event(
                        event="worker_failed",
                        data={"experiment_id": exp_id, "error": result.error_message},
                    ))
                else:
                    complete_experiment(exp, result)
                    self.logger.log(Event(
                        event="worker_completed",
                        data={
                            "experiment_id": exp_id,
                            "params": params,
                            "result": result.to_dict(),
                        },
                    ))
            except Exception as e:
                error_result = ExperimentResult(
                    primary_metric=0.0, error_message=str(e)
                )
                fail_experiment(exp, error_result)
                result = None

            release_worker(worker)
            self.budget.record_experiment()
            self.budget.add_cost(result.cost_usd if result else 0.0)

            results.append((exp, result))

        return results

    def _evaluate_iteration(
        self,
        plan: DimensionPlan,
        results: list[tuple[Experiment, ExperimentResult | None]],
        active_baselines: list[Baseline],
    ) -> list[Baseline]:
        """Evaluate iteration results. Returns updated active baselines."""
        # Filter to successful experiments (completed, no error)
        valid = [
            (exp, res)
            for exp, res in results
            if exp.status == ExperimentStatus.COMPLETED
            and res is not None
            and res.error_message is None
        ]

        if not valid:
            self.budget.record_no_improvement()
            return active_baselines

        # Find the best result(s)
        if self.spec.metric_direction == "lower":
            valid.sort(key=lambda x: x[1].primary_metric)
        else:
            valid.sort(key=lambda x: -x[1].primary_metric)

        best_exp, best_result = valid[0]
        best_metric = best_result.primary_metric

        # Check if best beats current baseline
        if self.spec.is_better(best_metric, self.best_baseline.metric_value):
            new_baseline = Baseline(
                commit=best_result.commit_hash or best_exp.baseline_commit,
                metric_value=best_metric,
                metric_name=self.spec.primary_metric,
            )
            self.best_baseline = new_baseline
            self.run_state.baselines.append(new_baseline)
            self.budget.record_improvement()

            self.logger.log(Event(
                event="breakthrough",
                data={
                    "new_best": best_metric,
                    "previous_best": active_baselines[0].metric_value
                    if active_baselines
                    else None,
                    "from_experiment": best_exp.experiment_id,
                },
            ))

            # Check for ties (beam search)
            tied = self._find_ties(valid, best_metric)
            if len(tied) > 1:
                # Multiple tied winners become parallel baselines
                return [
                    Baseline(
                        commit=r.commit_hash or e.baseline_commit,
                        metric_value=r.primary_metric,
                        metric_name=self.spec.primary_metric,
                        branch_id=e.experiment_id,
                    )
                    for e, r in tied
                ]

            return [new_baseline]
        else:
            self.budget.record_no_improvement()
            return active_baselines

    def _find_ties(
        self,
        sorted_results: list[tuple[Experiment, ExperimentResult]],
        best_metric: float,
    ) -> list[tuple[Experiment, ExperimentResult]]:
        """Find results within tie threshold of the best."""
        if self.tie_threshold_pct <= 0 or not sorted_results:
            return [sorted_results[0]]

        threshold = abs(best_metric) * (self.tie_threshold_pct / 100.0)
        tied = []
        for exp, result in sorted_results:
            if abs(result.primary_metric - best_metric) <= threshold:
                tied.append((exp, result))
        return tied if tied else [sorted_results[0]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_coordinator.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/core/coordinator.py tests/test_coordinator.py
git commit -m "feat: add coordinator with dimension sweep, beam search, and budget enforcement"
```

---

## Chunk 6: Scenario Runner & CLI

### Task 9: Implement YAML scenario runner

**Files:**
- Create: `chaosengineer/testing/runner.py`
- Create: `chaosengineer/testing/scenarios/breakthrough.yaml`
- Create: `tests/test_runner.py`

- [ ] **Step 1: Create a shipped test scenario**

`chaosengineer/testing/scenarios/breakthrough.yaml`:
```yaml
scenario: "breakthrough triggers baseline update"
description: >
  Two workers test a directional dimension. One finds an improvement,
  triggering a new baseline. The coordinator should advance and continue.

initial_baseline:
  commit: "abc1234"
  metric_value: 0.97
  metric_name: "val_bpb"

workload:
  name: "test-workload"
  primary_metric: "val_bpb"
  metric_direction: "lower"
  execution_command: "echo test"
  workers_available: 2
  budget:
    max_experiments: 4

plans:
  - dimension_name: "lr"
    values:
      - { lr: 0.02 }
      - { lr: 0.08 }
  - dimension_name: "depth"
    values:
      - { depth: 6 }
      - { depth: 12 }

results:
  "exp-0-0": { primary_metric: 0.91 }
  "exp-0-1": { primary_metric: 0.95 }
  "exp-1-0": { primary_metric: 0.88 }
  "exp-1-1": { primary_metric: 0.90 }

expected:
  final_best_metric: 0.88
  total_experiments: 4
  breakthroughs: 2
```

- [ ] **Step 2: Write failing tests for the scenario runner**

`tests/test_runner.py`:
```python
"""Tests for the scenario runner."""

import pytest
from pathlib import Path
from chaosengineer.testing.runner import ScenarioRunner, load_scenario, ScenarioResult

SCENARIOS_DIR = Path(__file__).parent.parent / "chaosengineer" / "testing" / "scenarios"


class TestLoadScenario:
    def test_load_breakthrough_scenario(self):
        scenario = load_scenario(SCENARIOS_DIR / "breakthrough.yaml")
        assert scenario["scenario"] == "breakthrough triggers baseline update"
        assert len(scenario["plans"]) == 2
        assert len(scenario["results"]) == 4

    def test_load_from_string(self):
        yaml_str = """
scenario: "test"
initial_baseline:
  commit: "abc"
  metric_value: 1.0
  metric_name: "score"
workload:
  name: "test"
  primary_metric: "score"
  metric_direction: "lower"
  execution_command: "echo"
  workers_available: 1
  budget:
    max_experiments: 1
plans:
  - dimension_name: "x"
    values:
      - { x: 1 }
results:
  "exp-0-0": { primary_metric: 0.9 }
expected:
  final_best_metric: 0.9
  total_experiments: 1
"""
        scenario = load_scenario(content=yaml_str)
        assert scenario["scenario"] == "test"


class TestScenarioRunner:
    def test_run_breakthrough_scenario(self, tmp_output_dir):
        runner = ScenarioRunner(output_dir=tmp_output_dir)
        result = runner.run_scenario(SCENARIOS_DIR / "breakthrough.yaml")

        assert result.passed
        assert result.final_best_metric == 0.88
        assert result.total_experiments == 4

    def test_run_scenario_with_expected_checks(self, tmp_output_dir):
        runner = ScenarioRunner(output_dir=tmp_output_dir)
        result = runner.run_scenario(SCENARIOS_DIR / "breakthrough.yaml")

        assert result.passed
        # Check expectations from YAML
        assert result.final_best_metric == result.expected["final_best_metric"]
        assert result.total_experiments == result.expected["total_experiments"]

    def test_run_produces_event_log(self, tmp_output_dir):
        runner = ScenarioRunner(output_dir=tmp_output_dir)
        result = runner.run_scenario(SCENARIOS_DIR / "breakthrough.yaml")

        assert result.event_log_path.exists()
        # Should have events
        import json
        events = [json.loads(line) for line in result.event_log_path.read_text().strip().split("\n")]
        assert len(events) > 0
        assert events[0]["event"] == "run_started"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_runner.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement the scenario runner**

`chaosengineer/testing/runner.py`:
```python
"""Scenario runner: loads YAML scenarios, wires up simulator+executor, runs coordinator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.models import Baseline, BudgetConfig, ExperimentResult
from chaosengineer.metrics.logger import EventLogger
from chaosengineer.testing.executor import ScriptedExecutor
from chaosengineer.testing.simulator import DimensionPlan, ScriptedDecisionMaker
from chaosengineer.workloads.parser import WorkloadSpec


@dataclass
class ScenarioResult:
    """Result of running a scenario."""
    scenario_name: str
    passed: bool
    final_best_metric: float
    total_experiments: int
    event_log_path: Path
    expected: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def load_scenario(
    path: Path | str | None = None,
    content: str | None = None,
) -> dict:
    """Load a YAML scenario file or string."""
    if content is None:
        if path is None:
            raise ValueError("Either path or content must be provided")
        content = Path(path).read_text()
    return yaml.safe_load(content)


def _build_workload_spec(workload_data: dict) -> WorkloadSpec:
    """Build a WorkloadSpec from scenario YAML data."""
    budget_data = workload_data.get("budget", {})
    budget = BudgetConfig(
        max_api_cost=budget_data.get("max_api_cost"),
        max_experiments=budget_data.get("max_experiments"),
        max_wall_time_seconds=budget_data.get("max_wall_time_seconds"),
        max_plateau_iterations=budget_data.get("max_plateau_iterations"),
    )
    return WorkloadSpec(
        name=workload_data.get("name", "scenario-test"),
        primary_metric=workload_data.get("primary_metric", "metric"),
        metric_direction=workload_data.get("metric_direction", "lower"),
        execution_command=workload_data.get("execution_command", "echo test"),
        workers_available=workload_data.get("workers_available", 1),
        budget=budget,
    )


def _build_plans(plans_data: list[dict]) -> list[DimensionPlan]:
    """Build DimensionPlan list from scenario YAML data."""
    return [
        DimensionPlan(
            dimension_name=p["dimension_name"],
            values=p["values"],
        )
        for p in plans_data
    ]


def _build_results(results_data: dict) -> dict[str, ExperimentResult]:
    """Build ExperimentResult map from scenario YAML data."""
    results = {}
    for exp_id, data in results_data.items():
        results[exp_id] = ExperimentResult(
            primary_metric=data["primary_metric"],
            duration_seconds=data.get("duration_seconds", 0),
            error_message=data.get("error_message"),
        )
    return results


class ScenarioRunner:
    """Runs test scenarios without LLM calls or real experiment execution."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def run_scenario(
        self,
        path: Path | str | None = None,
        content: str | None = None,
    ) -> ScenarioResult:
        """Run a scenario and return the result."""
        scenario = load_scenario(path=path, content=content)
        scenario_name = scenario.get("scenario", "unnamed")

        # Build components from scenario data
        spec = _build_workload_spec(scenario["workload"])
        plans = _build_plans(scenario["plans"])
        results = _build_results(scenario["results"])
        initial_baseline = Baseline(
            commit=scenario["initial_baseline"]["commit"],
            metric_value=scenario["initial_baseline"]["metric_value"],
            metric_name=scenario["initial_baseline"]["metric_name"],
        )

        event_log_path = self.output_dir / f"{scenario_name.replace(' ', '_')}.jsonl"

        tie_threshold_pct = scenario.get("tie_threshold_pct", 1.0)

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(event_log_path),
            budget=BudgetTracker(spec.budget),
            initial_baseline=initial_baseline,
            tie_threshold_pct=tie_threshold_pct,
        )

        coordinator.run()

        # Check expected outcomes
        expected = scenario.get("expected", {})
        errors = []

        if "final_best_metric" in expected:
            actual = coordinator.best_baseline.metric_value
            exp_val = expected["final_best_metric"]
            if abs(actual - exp_val) > 1e-6:
                errors.append(
                    f"final_best_metric: expected {exp_val}, got {actual}"
                )

        if "total_experiments" in expected:
            actual = coordinator.budget.experiments_run
            exp_val = expected["total_experiments"]
            if actual != exp_val:
                errors.append(
                    f"total_experiments: expected {exp_val}, got {actual}"
                )

        if "breakthroughs" in expected:
            events = coordinator.logger.read_events(event_type="breakthrough")
            actual = len(events)
            exp_val = expected["breakthroughs"]
            if actual != exp_val:
                errors.append(
                    f"breakthroughs: expected {exp_val}, got {actual}"
                )

        return ScenarioResult(
            scenario_name=scenario_name,
            passed=len(errors) == 0,
            final_best_metric=coordinator.best_baseline.metric_value,
            total_experiments=coordinator.budget.experiments_run,
            event_log_path=event_log_path,
            expected=expected,
            errors=errors,
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_runner.py -v`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add chaosengineer/testing/runner.py chaosengineer/testing/scenarios/ tests/test_runner.py
git commit -m "feat: add YAML scenario runner for testing coordinator without LLM calls"
```

---

### Task 10: Add additional shipped scenarios

**Files:**
- Create: `chaosengineer/testing/scenarios/tie_branching.yaml`
- Create: `chaosengineer/testing/scenarios/budget_exhaustion.yaml`
- Create: `tests/test_shipped_scenarios.py`

- [ ] **Step 1: Create tie branching scenario**

`chaosengineer/testing/scenarios/tie_branching.yaml`:
```yaml
scenario: "tie branching creates parallel baselines"
description: >
  Three options are tested. Two tie for best, triggering beam search.
  Each branch is explored, and the best overall result wins.

initial_baseline:
  commit: "abc1234"
  metric_value: 0.97
  metric_name: "val_bpb"

workload:
  name: "tie-test"
  primary_metric: "val_bpb"
  metric_direction: "lower"
  execution_command: "echo test"
  workers_available: 3
  budget:
    max_experiments: 10

plans:
  - dimension_name: "activation"
    values:
      - { act: "GeLU" }
      - { act: "SiLU" }
      - { act: "ReLU" }
  # Branch explorations after tie
  - dimension_name: "depth_from_GeLU"
    values:
      - { depth: 10 }
  - dimension_name: "depth_from_SiLU"
    values:
      - { depth: 10 }

results:
  # GeLU and SiLU tie, ReLU is worse
  "exp-0-0": { primary_metric: 0.91 }
  "exp-0-1": { primary_metric: 0.911 }
  "exp-0-2": { primary_metric: 0.98 }
  # Branch results
  "exp-1-0": { primary_metric: 0.87 }
  "exp-2-0": { primary_metric: 0.90 }

expected:
  final_best_metric: 0.87
  total_experiments: 5
```

- [ ] **Step 2: Create budget exhaustion scenario**

`chaosengineer/testing/scenarios/budget_exhaustion.yaml`:
```yaml
scenario: "budget exhaustion stops execution"
description: >
  Coordinator has budget for 3 experiments. First iteration uses 2,
  second iteration is trimmed to 1, then stops.

initial_baseline:
  commit: "abc1234"
  metric_value: 0.97
  metric_name: "val_bpb"

workload:
  name: "budget-test"
  primary_metric: "val_bpb"
  metric_direction: "lower"
  execution_command: "echo test"
  workers_available: 2
  budget:
    max_experiments: 3

plans:
  - dimension_name: "lr"
    values:
      - { lr: 0.02 }
      - { lr: 0.08 }
  - dimension_name: "depth"
    values:
      - { depth: 6 }
      - { depth: 12 }

results:
  "exp-0-0": { primary_metric: 0.93 }
  "exp-0-1": { primary_metric: 0.95 }
  "exp-1-0": { primary_metric: 0.90 }

expected:
  final_best_metric: 0.90
  total_experiments: 3
```

- [ ] **Step 3: Write tests that run all shipped scenarios**

`tests/test_shipped_scenarios.py`:
```python
"""Tests that run all shipped scenarios and verify expectations pass."""

import pytest
from pathlib import Path
from chaosengineer.testing.runner import ScenarioRunner

SCENARIOS_DIR = Path(__file__).parent.parent / "chaosengineer" / "testing" / "scenarios"


def _scenario_files():
    return sorted(SCENARIOS_DIR.glob("*.yaml"))


@pytest.mark.parametrize("scenario_path", _scenario_files(), ids=lambda p: p.stem)
def test_shipped_scenario(scenario_path, tmp_output_dir):
    runner = ScenarioRunner(output_dir=tmp_output_dir)
    result = runner.run_scenario(scenario_path)

    assert result.passed, (
        f"Scenario '{result.scenario_name}' failed:\n"
        + "\n".join(f"  - {e}" for e in result.errors)
    )
```

- [ ] **Step 4: Run all scenario tests**

Run: `uv run pytest tests/test_shipped_scenarios.py -v`
Expected: All 3 scenarios (breakthrough, tie_branching, budget_exhaustion) PASS.

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/testing/scenarios/ tests/test_shipped_scenarios.py
git commit -m "feat: add shipped test scenarios for breakthrough, tie branching, and budget exhaustion"
```

---

### Task 11: Add per-run summary generation

**Files:**
- Create: `chaosengineer/metrics/summary.py`
- Add tests to: `tests/test_logger.py`

- [ ] **Step 1: Write failing tests for summary generation**

Add to `tests/test_logger.py`:
```python
from chaosengineer.metrics.summary import generate_summary


class TestRunSummary:
    def test_summary_from_events(self, tmp_output_dir):
        logger = EventLogger(tmp_output_dir / "events.jsonl")
        logger.log(Event(event="run_started", data={"workload": "test", "budget": {}}))
        logger.log(Event(event="worker_completed", data={
            "experiment_id": "exp-0-0",
            "result": {"primary_metric": 0.93, "cost_usd": 0.02},
        }))
        logger.log(Event(event="breakthrough", data={
            "new_best": 0.93, "previous_best": 0.97,
        }))
        logger.log(Event(event="run_completed", data={
            "best_metric": 0.93, "total_experiments": 1, "total_cost_usd": 0.02,
        }))

        summary = generate_summary(logger)
        assert summary["best_metric"] == 0.93
        assert summary["total_experiments"] == 1
        assert summary["breakthroughs"] == 1

    def test_summary_as_text(self, tmp_output_dir):
        logger = EventLogger(tmp_output_dir / "events.jsonl")
        logger.log(Event(event="run_completed", data={
            "best_metric": 0.93, "total_experiments": 5, "total_cost_usd": 1.50,
        }))

        summary = generate_summary(logger)
        text = summary_to_text(summary)
        assert "0.93" in text
        assert "5" in text
```

Update the import at the top of the test file:
```python
from chaosengineer.metrics.summary import generate_summary, summary_to_text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_logger.py::TestRunSummary -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement summary generation**

`chaosengineer/metrics/summary.py`:
```python
"""Per-run summary generation."""

from __future__ import annotations

from typing import Any

from chaosengineer.metrics.logger import EventLogger


def generate_summary(logger: EventLogger) -> dict[str, Any]:
    """Generate a summary dict from the event log."""
    events = logger.read_events()

    completed_events = [e for e in events if e["event"] == "run_completed"]
    run_data = completed_events[-1]["data"] if completed_events else {}

    breakthroughs = [e for e in events if e["event"] == "breakthrough"]
    iterations = [e for e in events if e["event"] == "iteration_started"]
    worker_completions = [e for e in events if e["event"] == "worker_completed"]
    failures = [e for e in events if e["event"] == "worker_failed"]

    return {
        "best_metric": run_data.get("best_metric"),
        "total_experiments": run_data.get("total_experiments", len(worker_completions) + len(failures)),
        "total_cost_usd": run_data.get("total_cost_usd", 0),
        "breakthroughs": len(breakthroughs),
        "iterations": len(iterations),
        "failures": len(failures),
    }


def summary_to_text(summary: dict[str, Any]) -> str:
    """Format summary as human-readable text."""
    lines = [
        "=== Run Summary ===",
        f"Best metric:       {summary.get('best_metric', 'N/A')}",
        f"Total experiments:  {summary.get('total_experiments', 0)}",
        f"Breakthroughs:      {summary.get('breakthroughs', 0)}",
        f"Iterations:         {summary.get('iterations', 0)}",
        f"Failures:           {summary.get('failures', 0)}",
        f"Total cost (USD):   ${summary.get('total_cost_usd', 0):.2f}",
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_logger.py -v`
Expected: All tests PASS (both old and new).

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/metrics/summary.py tests/test_logger.py
git commit -m "feat: add per-run summary generation from event logs"
```

---

### Task 12: Add CLI entry point

**Files:**
- Create: `chaosengineer/cli.py`
- Modify: `pyproject.toml` (add script entry point)

- [ ] **Step 1: Add script entry point to pyproject.toml**

Add to `pyproject.toml`:
```toml
[project.scripts]
chaosengineer = "chaosengineer.cli:main"
```

- [ ] **Step 2: Implement the CLI**

`chaosengineer/cli.py`:
```python
"""CLI entry point for ChaosEngineer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from chaosengineer.testing.runner import ScenarioRunner


def main():
    parser = argparse.ArgumentParser(
        prog="chaosengineer",
        description="General-purpose parallel experimentation framework",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Test command: run scenarios
    test_parser = subparsers.add_parser("test", help="Run test scenarios")
    test_parser.add_argument(
        "scenario",
        nargs="?",
        help="Path to scenario YAML file. If omitted, runs all shipped scenarios.",
    )
    test_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".chaosengineer/test-output"),
        help="Directory for test output",
    )

    # Version
    subparsers.add_parser("version", help="Print version")

    args = parser.parse_args()

    if args.command == "version":
        from chaosengineer import __version__
        print(f"chaosengineer {__version__}")

    elif args.command == "test":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        runner = ScenarioRunner(output_dir=args.output_dir)

        if args.scenario:
            result = runner.run_scenario(Path(args.scenario))
            _print_scenario_result(result)
            sys.exit(0 if result.passed else 1)
        else:
            # Run all shipped scenarios
            scenarios_dir = Path(__file__).parent / "testing" / "scenarios"
            all_passed = True
            for path in sorted(scenarios_dir.glob("*.yaml")):
                result = runner.run_scenario(path)
                _print_scenario_result(result)
                if not result.passed:
                    all_passed = False
            sys.exit(0 if all_passed else 1)

    else:
        parser.print_help()


def _print_scenario_result(result):
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {result.scenario_name}")
    print(f"  Best metric: {result.final_best_metric}")
    print(f"  Experiments: {result.total_experiments}")
    if result.errors:
        for error in result.errors:
            print(f"  ERROR: {error}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Install and verify CLI works**

Run: `cd /Users/alex/CODE/OSS/autoresearch && uv pip install -e ".[test]" && uv run chaosengineer version`
Expected: `chaosengineer 0.1.0`

- [ ] **Step 4: Run shipped scenarios via CLI**

Run: `uv run chaosengineer test`
Expected: All shipped scenarios show `[PASS]`.

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/cli.py pyproject.toml
git commit -m "feat: add CLI entry point with test scenario runner command"
```

---

### Task 13: Run full test suite

- [ ] **Step 1: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 2: Verify no test isolation issues**

Run: `uv run pytest tests/ -v --tb=short -x`
Expected: All tests PASS with no flaky failures.

- [ ] **Step 3: Final commit with all tests green**

Only if any fixups were needed:
```bash
git add -A
git commit -m "fix: resolve test issues from full suite run"
```
