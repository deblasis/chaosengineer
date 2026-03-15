# Retry/Resume Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable resuming partially-completed ChaosEngineer runs from the append-only event log.

**Architecture:** A `RunSnapshot` dataclass is reconstructed by replaying `events.jsonl`. The coordinator gains a `resume_from_snapshot()` entry point that restores state (baselines, budget, history, explored dimensions) and re-enters the normal loop. The CLI adds a `resume` subcommand and a run guard that detects existing sessions.

**Tech Stack:** Python 3.12, dataclasses, argparse, JSONL event log, tty/termios for interactive menu

**Spec:** `docs/superpowers/specs/2026-03-15-retry-resume-design.md`

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `chaosengineer/core/snapshot.py` | `RunSnapshot` dataclass + `build_snapshot()` event replay |
| `chaosengineer/core/decision_log.py` | `DecisionLogger` — writes LLM reasoning to `decisions.jsonl` |
| `chaosengineer/cli_menu.py` | `select()` — interactive arrow-key menu utility |
| `tests/test_snapshot.py` | Snapshot reconstruction tests |
| `tests/test_decision_log.py` | Decision log tests |
| `tests/test_cli_menu.py` | Menu utility tests |
| `tests/test_resume_coordinator.py` | Coordinator resume integration tests |
| `tests/e2e/test_resume_pipeline.py` | End-to-end resume scenario |

### Modified Files
| File | Changes |
|------|---------|
| `chaosengineer/core/coordinator.py` | `resume_from_snapshot()`, exit branching (paused vs completed) |
| `chaosengineer/core/budget.py` | `from_snapshot()` classmethod, `_elapsed_offset` field |
| `chaosengineer/core/interfaces.py` | `set_prior_context()` on DecisionMaker ABC |
| `chaosengineer/metrics/logger.py` | New event types: `run_paused`, `run_resumed`, `iteration_gap_completed` |
| `chaosengineer/llm/decision_maker.py` | Override `set_prior_context()`, wire `DecisionLogger` |
| `chaosengineer/testing/simulator.py` | `set_prior_context()` no-op |
| `chaosengineer/cli.py` | `resume` subcommand, run guard with menu |

---

## Chunk 1: Event Enrichment, Snapshot Data Models & Event Replay

### Task 0: Enrich Coordinator Events for Resume Support

The existing coordinator events are missing fields needed by `build_snapshot()`. This task adds those fields **without breaking existing events** — all additions are new keys.

**Files:**
- Modify: `chaosengineer/core/coordinator.py:97-162` (event emissions)
- Modify: `chaosengineer/workloads/parser.py` (add `spec_hash()` method)
- Modify: `tests/test_coordinator.py`

**Current vs needed event fields:**

| Event | Current fields | Fields to add |
|-------|---------------|---------------|
| `run_started` | `workload`, `budget`, `baseline` | `run_id`, `mode`, `metric_direction`, `workload_spec_hash` |
| `iteration_started` | `dimension`, `num_workers`, `iteration`, `branch_id` | `tasks` (list of `{experiment_id, params, command, baseline_commit}`) |
| `worker_completed` | `experiment_id`, `params`, `result` (nested dict) | `dimension`, `metric` (top-level alias), `cost_usd` (top-level alias) |
| `worker_failed` | `experiment_id`, `error` | `dimension`, `params`, `cost_usd` |
| `breakthrough` | `new_best`, `previous_best`, `from_experiment` | `commit`, `metric` (aliases for `from_experiment`'s commit and `new_best`) |

- [ ] **Step 1: Write tests for enriched events**

Add to `tests/test_coordinator.py`:

```python
from chaosengineer.core.models import ExperimentResult


class TestEnrichedEvents:
    """Verify events contain fields needed by build_snapshot()."""

    def _run_simple_coordinator(self, tmp_path, plans, results, budget_config):
        from chaosengineer.workloads.parser import WorkloadSpec
        spec = WorkloadSpec(
            name="test", primary_metric="loss", metric_direction="lower",
            execution_command="echo 1", workers_available=2,
            budget=budget_config,
        )
        log_path = tmp_path / "events.jsonl"
        coord = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(log_path),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("abc", 3.0, "loss"),
        )
        coord.run()
        return EventLogger(log_path)

    def test_run_started_has_resume_fields(self, tmp_path):
        plans = [DimensionPlan("lr", [{"lr": 0.1}])]
        results = {"exp-0-0": ExperimentResult(primary_metric=2.0)}
        logger = self._run_simple_coordinator(
            tmp_path, plans, results, BudgetConfig(max_experiments=1))
        events = logger.read_events("run_started")
        e = events[0]
        assert "run_id" in e
        assert "mode" in e
        assert "metric_direction" in e
        assert "workload_spec_hash" in e

    def test_iteration_started_has_tasks(self, tmp_path):
        plans = [DimensionPlan("lr", [{"lr": 0.01}, {"lr": 0.1}])]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=2.0),
            "exp-0-1": ExperimentResult(primary_metric=2.5),
        }
        logger = self._run_simple_coordinator(
            tmp_path, plans, results, BudgetConfig(max_experiments=2))
        events = logger.read_events("iteration_started")
        tasks = events[0].get("tasks", [])
        assert len(tasks) == 2
        assert all("experiment_id" in t for t in tasks)
        assert all("params" in t for t in tasks)
        assert all("command" in t for t in tasks)
        assert all("baseline_commit" in t for t in tasks)

    def test_worker_completed_has_dimension_and_top_level_metric(self, tmp_path):
        plans = [DimensionPlan("lr", [{"lr": 0.1}])]
        results = {"exp-0-0": ExperimentResult(primary_metric=2.0, cost_usd=0.5)}
        logger = self._run_simple_coordinator(
            tmp_path, plans, results, BudgetConfig(max_experiments=1))
        events = logger.read_events("worker_completed")
        e = events[0]
        assert e["dimension"] == "lr"
        assert e["metric"] == 2.0
        assert e["cost_usd"] == 0.5

    def test_breakthrough_has_commit_and_metric(self, tmp_path):
        plans = [DimensionPlan("lr", [{"lr": 0.1}])]
        results = {"exp-0-0": ExperimentResult(primary_metric=2.0, commit_hash="newcommit")}
        logger = self._run_simple_coordinator(
            tmp_path, plans, results, BudgetConfig(max_experiments=1))
        events = logger.read_events("breakthrough")
        assert len(events) == 1
        assert events[0]["metric"] == 2.0
        assert events[0]["commit"] == "newcommit"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_coordinator.py::TestEnrichedEvents -v`
Expected: FAIL — missing fields in events

- [ ] **Step 3: Add `spec_hash()` to WorkloadSpec**

In `chaosengineer/workloads/parser.py`, add to the `WorkloadSpec` class:

```python
def spec_hash(self) -> str:
    """SHA256 hash of the spec's identifying fields."""
    import hashlib
    content = f"{self.name}|{self.primary_metric}|{self.metric_direction}|{self.execution_command}"
    return "sha256:" + hashlib.sha256(content.encode()).hexdigest()[:16]
```

- [ ] **Step 4: Enrich `run_started` event**

In `chaosengineer/core/coordinator.py`, modify the `run_started` emission at lines 97-104:

```python
self._log(Event(
    event="run_started",
    data={
        "run_id": self.run_state.run_id,
        "workload": self.spec.name,
        "mode": self.run_state.mode,
        "metric_direction": self.spec.metric_direction,
        "budget": self.budget.config.to_dict(),
        "baseline": self.best_baseline.to_dict(),
        "workload_spec_hash": self.spec.spec_hash(),
    },
))
```

- [ ] **Step 5: Enrich `iteration_started` event**

In coordinator.py, modify the `iteration_started` emission at lines 154-162. Insert after building the `tasks` list (after line 224), store a snapshot of tasks for the event, then emit:

```python
self._log(Event(
    event="iteration_started",
    data={
        "dimension": plan.dimension_name,
        "num_workers": len(plan.values),
        "iteration": self._iteration,
        "branch_id": baseline.branch_id,
        "tasks": [
            {
                "experiment_id": t.experiment_id,
                "params": t.params,
                "command": t.command,
                "baseline_commit": t.baseline_commit,
            }
            for t in tasks
        ],
    },
))
```

Note: This means the `iteration_started` event must be moved AFTER the tasks list is built (currently it's emitted before `_run_iteration`). Either move the log call inside `_run_iteration`, or build the task metadata in the main loop and pass it down.

- [ ] **Step 6: Enrich `worker_completed` and `worker_failed` events**

In `_run_iteration()` at lines 234-247, add dimension and top-level metric aliases:

```python
# worker_failed (line 234-237):
self._log(Event(
    event="worker_failed",
    data={
        "experiment_id": exp.experiment_id,
        "dimension": exp.dimension,
        "params": exp.params,
        "error": result.error_message,
        "cost_usd": result.cost_usd,
    },
))

# worker_completed (line 240-247):
self._log(Event(
    event="worker_completed",
    data={
        "experiment_id": exp.experiment_id,
        "dimension": exp.dimension,
        "params": exp.params,
        "metric": result.primary_metric,
        "cost_usd": result.cost_usd,
        "result": result.to_dict(),
    },
))
```

- [ ] **Step 7: Enrich `breakthrough` event**

In `_evaluate_iteration()` at lines 295-303, add commit and metric aliases:

```python
self._log(Event(
    event="breakthrough",
    data={
        "new_best": best_metric,
        "previous_best": active_baselines[0].metric_value
        if active_baselines else None,
        "from_experiment": best_exp.experiment_id,
        "commit": best_result.commit_hash or best_exp.baseline_commit,
        "metric": best_metric,
    },
))
```

- [ ] **Step 8: Run enriched event tests**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_coordinator.py::TestEnrichedEvents -v`
Expected: PASS

- [ ] **Step 9: Run full test suite**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/ -v`
Expected: All existing tests PASS (we only added new fields, didn't remove any)

- [ ] **Step 10: Commit**

```bash
git add chaosengineer/core/coordinator.py chaosengineer/workloads/parser.py tests/test_coordinator.py
git commit -m "feat: enrich coordinator events with fields needed for resume"
```

---

### Task 1: RunSnapshot Data Models

**Files:**
- Create: `chaosengineer/core/snapshot.py`
- Create: `tests/test_snapshot.py`

- [ ] **Step 1: Write tests for snapshot data models**

```python
# tests/test_snapshot.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_snapshot.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chaosengineer.core.snapshot'`

- [ ] **Step 3: Implement snapshot data models**

```python
# chaosengineer/core/snapshot.py
"""RunSnapshot: reconstructed state from event log for resume."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from chaosengineer.core.interfaces import ExperimentTask
from chaosengineer.core.models import Baseline, BudgetConfig


class StopReason(Enum):
    PAUSED = "paused"
    COMPLETED = "completed"
    CRASHED = "crashed"


@dataclass
class ExperimentSummary:
    experiment_id: str
    dimension: str
    params: dict[str, Any]
    metric: float | None
    status: str  # "completed" or "failed"
    cost_usd: float


@dataclass
class DimensionOutcome:
    name: str
    values_tested: list[str]
    winner: str | None
    metric_improvement: float | None


@dataclass
class IncompleteIteration:
    dimension: str
    total_workers: int
    completed_experiments: list[ExperimentSummary]
    missing_experiment_ids: list[str]
    missing_tasks: list[ExperimentTask]


@dataclass
class RunSnapshot:
    run_id: str
    workload_name: str
    workload_spec_hash: str
    budget_config: BudgetConfig
    mode: str

    # Restored state
    active_baselines: list[Baseline]
    baseline_history: list[Baseline]
    dimensions_explored: list[DimensionOutcome]
    discovered_dimensions: dict[str, list[str]]
    experiments: list[ExperimentSummary]
    history: list[dict[str, Any]]
    total_cost_usd: float
    total_experiments_run: int
    elapsed_time: float
    consecutive_no_improvement: int

    # Incomplete work
    incomplete_iteration: IncompleteIteration | None
    stop_reason: StopReason
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_snapshot.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/core/snapshot.py tests/test_snapshot.py
git commit -m "feat: add RunSnapshot data models for resume state"
```

---

### ~~Task 2~~ — Merged into Task 0

Workload spec hash is now added to `run_started` in Task 0.

---

### ~~Task 3~~ — Merged into Task 8

The `run_paused` vs `run_completed` exit branching is now part of the `_run_loop()` refactoring in Task 8. The `exhaustion_reason` property on BudgetTracker is added in Task 5 alongside `from_snapshot()`.

---

### Task 4: `build_snapshot()` — Event Log Replay

**Files:**
- Modify: `chaosengineer/core/snapshot.py`
- Modify: `tests/test_snapshot.py`

- [ ] **Step 1: Write tests for `build_snapshot()`**

Add to `tests/test_snapshot.py`:

```python
import json
from pathlib import Path

from chaosengineer.core.snapshot import build_snapshot, StopReason
from chaosengineer.metrics.logger import EventLogger, Event


def _write_events(path: Path, events: list[dict]):
    """Write raw event dicts to a JSONL file."""
    with open(path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


class TestBuildSnapshot:
    def test_completed_run(self, tmp_path):
        """Snapshot of a fully completed run."""
        events_path = tmp_path / "events.jsonl"
        _write_events(events_path, [
            {"event": "run_started", "run_id": "run-1", "workload_name": "test",
             "workload_spec_hash": "sha256:abc", "budget": {"max_experiments": 10},
             "mode": "parallel", "baseline": {"commit": "aaa", "metric_value": 3.0, "metric_name": "loss"},
             "ts": "2026-01-01T00:00:00Z"},
            {"event": "iteration_started", "iteration": 0, "dimension": "lr",
             "worker_count": 2, "tasks": [
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
            {"event": "run_started", "run_id": "run-2", "workload_name": "test",
             "workload_spec_hash": "sha256:def", "budget": {"max_experiments": 2},
             "mode": "parallel", "baseline": {"commit": "aaa", "metric_value": 3.0, "metric_name": "loss"},
             "ts": "2026-01-01T00:00:00Z"},
            {"event": "iteration_started", "iteration": 0, "dimension": "lr",
             "worker_count": 2, "tasks": [
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
            {"event": "run_started", "run_id": "run-3", "workload_name": "test",
             "workload_spec_hash": "sha256:ghi", "budget": {"max_experiments": 10},
             "mode": "parallel", "baseline": {"commit": "aaa", "metric_value": 3.0, "metric_name": "loss"},
             "ts": "2026-01-01T00:00:00Z"},
            {"event": "iteration_started", "iteration": 0, "dimension": "lr",
             "worker_count": 3, "tasks": [
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
            {"event": "run_started", "run_id": "run-4", "workload_name": "test",
             "workload_spec_hash": "sha256:jkl", "budget": {"max_experiments": 10},
             "mode": "parallel", "baseline": {"commit": "aaa", "metric_value": 3.0, "metric_name": "loss"},
             "ts": "2026-01-01T00:00:00Z"},
            {"event": "iteration_started", "iteration": 0, "dimension": "lr",
             "worker_count": 3, "tasks": [
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
            {"event": "run_started", "run_id": "run-5", "workload_name": "test",
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
            {"event": "run_started", "run_id": "run-6", "workload_name": "test",
             "workload_spec_hash": "sha256:pqr", "budget": {"max_experiments": 10},
             "mode": "parallel", "baseline": {"commit": "aaa", "metric_value": 3.0, "metric_name": "loss"},
             "ts": "2026-01-01T00:00:00Z"},
            {"event": "iteration_started", "iteration": 0, "dimension": "lr",
             "worker_count": 1, "tasks": [
                 {"experiment_id": "exp-0-0", "params": {"lr": 0.01}, "command": "echo", "baseline_commit": "aaa"},
             ], "ts": "2026-01-01T00:00:01Z"},
            {"event": "worker_completed", "experiment_id": "exp-0-0", "dimension": "lr",
             "params": {"lr": 0.01}, "metric": 2.5, "cost_usd": 0.50, "ts": "2026-01-01T00:00:10Z"},
            {"event": "run_completed", "best_metric": 2.5, "total_experiments": 1,
             "total_cost_usd": 0.5, "ts": "2026-01-01T00:00:12Z"},
        ])

        snap = build_snapshot(events_path)
        assert len(snap.history) >= 1
        # History should contain worker result entries usable by decision maker
        assert any(h.get("dimension") == "lr" for h in snap.history)

    def test_resumed_run_replayed(self, tmp_path):
        """A previously resumed run replays correctly through both segments."""
        events_path = tmp_path / "events.jsonl"
        _write_events(events_path, [
            {"event": "run_started", "run_id": "run-7", "workload_name": "test",
             "workload_spec_hash": "sha256:stu", "budget": {"max_experiments": 10},
             "mode": "parallel", "baseline": {"commit": "aaa", "metric_value": 3.0, "metric_name": "loss"},
             "ts": "2026-01-01T00:00:00Z"},
            {"event": "iteration_started", "iteration": 0, "dimension": "lr",
             "worker_count": 1, "tasks": [
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
             "worker_count": 1, "tasks": [
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_snapshot.py::TestBuildSnapshot -v`
Expected: FAIL — `ImportError: cannot import name 'build_snapshot'`

- [ ] **Step 3: Implement `build_snapshot()`**

Add to `chaosengineer/core/snapshot.py`:

```python
import json
from pathlib import Path


def build_snapshot(events_path: Path) -> RunSnapshot:
    """Replay events.jsonl to reconstruct run state."""
    events = _read_events(events_path)

    # Initial state — filled by run_started
    run_id = ""
    workload_name = ""
    workload_spec_hash = ""
    metric_direction = "lower"  # needed for _finalize_iteration winner detection
    budget_config = BudgetConfig()
    mode = "parallel"
    active_baselines: list[Baseline] = []
    baseline_history: list[Baseline] = []
    dimensions_explored: list[DimensionOutcome] = []
    discovered_dimensions: dict[str, list[str]] = {}
    experiments: list[ExperimentSummary] = []
    history: list[dict] = []
    total_cost_usd = 0.0
    total_experiments_run = 0
    elapsed_time = 0.0
    consecutive_no_improvement = 0
    stop_reason: StopReason | None = None

    # Iteration tracking for incomplete detection
    current_iteration_dim: str | None = None
    current_iteration_worker_count = 0
    current_iteration_tasks: list[dict] = []
    current_iteration_completed: list[ExperimentSummary] = []
    current_iteration_completed_ids: set[str] = set()
    iteration_finalized = True  # True until an iteration_started with no matching completion

    for entry in events:
        event_type = entry.get("event", "")

        if event_type == "run_started":
            # Fields added by Task 0. "workload" is the original field name.
            run_id = entry.get("run_id", "")
            workload_name = entry.get("workload", "")
            workload_spec_hash = entry.get("workload_spec_hash", "")
            metric_direction = entry.get("metric_direction", "lower")
            budget_data = entry.get("budget", {})
            budget_config = BudgetConfig(
                max_api_cost=budget_data.get("max_api_cost"),
                max_experiments=budget_data.get("max_experiments"),
                max_wall_time_seconds=budget_data.get("max_wall_time_seconds"),
                max_plateau_iterations=budget_data.get("max_plateau_iterations"),
            )
            mode = entry.get("mode", "parallel")
            bl = entry.get("baseline", {})
            initial_baseline = Baseline(
                commit=bl.get("commit", ""),
                metric_value=bl.get("metric_value", 0.0),
                metric_name=bl.get("metric_name", ""),
            )
            active_baselines = [initial_baseline]
            baseline_history = [initial_baseline]

        elif event_type == "iteration_started":
            # Finalize previous iteration if it was complete
            if not iteration_finalized and current_iteration_dim:
                _finalize_iteration(
                    current_iteration_dim, current_iteration_completed,
                    dimensions_explored, metric_direction,
                )
            current_iteration_dim = entry.get("dimension", "")
            # Field is "num_workers" in the actual coordinator
            current_iteration_worker_count = entry.get("num_workers", 0)
            # "tasks" field added by Task 0
            current_iteration_tasks = entry.get("tasks", [])
            current_iteration_completed = []
            current_iteration_completed_ids = set()
            iteration_finalized = False

        elif event_type in ("worker_completed", "worker_failed"):
            exp_id = entry.get("experiment_id", "")
            # "dimension", "metric", "cost_usd" are top-level aliases added by Task 0
            dim = entry.get("dimension", "")
            params = entry.get("params", {})
            metric = entry.get("metric")  # top-level alias
            cost = entry.get("cost_usd", 0.0)  # top-level alias
            status = "completed" if event_type == "worker_completed" else "failed"

            summary = ExperimentSummary(exp_id, dim, params, metric, status, cost)
            experiments.append(summary)
            total_experiments_run += 1
            total_cost_usd += cost

            # Build history entry for decision maker
            history.append({
                "experiment_id": exp_id,
                "dimension": dim,
                "params": params,
                "metric": metric,
                "status": status,
            })

            # Track iteration completion
            current_iteration_completed.append(summary)
            current_iteration_completed_ids.add(exp_id)

        elif event_type == "breakthrough":
            # "commit" and "metric" are aliases added by Task 0
            new_bl = Baseline(
                commit=entry.get("commit", ""),
                metric_value=entry.get("metric", entry.get("new_best", 0.0)),
                metric_name=active_baselines[0].metric_name if active_baselines else "",
            )
            active_baselines = [new_bl]
            baseline_history.append(new_bl)
            consecutive_no_improvement = 0

        elif event_type == "diverse_discovered":
            dim_name = entry.get("dimension", "")
            options = entry.get("options", [])
            discovered_dimensions[dim_name] = options

        elif event_type == "budget_checkpoint":
            # Cross-check (use as authoritative if present)
            elapsed_time = entry.get("elapsed_seconds", elapsed_time)

        elif event_type == "run_paused":
            stop_reason = StopReason.PAUSED
            budget_state = entry.get("budget_state", {})
            elapsed_time = budget_state.get("elapsed_seconds", elapsed_time)
            # Restore active baselines from paused event
            bl_list = entry.get("active_baselines", [])
            if bl_list:
                active_baselines = [
                    Baseline(b["commit"], b["metric_value"], b["metric_name"])
                    for b in bl_list
                ]
            # Finalize iteration if it was complete
            if not iteration_finalized and current_iteration_dim:
                if len(current_iteration_completed) >= current_iteration_worker_count:
                    _finalize_iteration(
                        current_iteration_dim, current_iteration_completed,
                        dimensions_explored, metric_direction,
                    )
                    iteration_finalized = True

        elif event_type == "run_completed":
            stop_reason = StopReason.COMPLETED
            if not iteration_finalized and current_iteration_dim:
                _finalize_iteration(
                    current_iteration_dim, current_iteration_completed,
                    dimensions_explored, metric_direction,
                )
                iteration_finalized = True

        elif event_type == "run_resumed":
            # Reset stop_reason — we're continuing
            stop_reason = None

        elif event_type == "iteration_gap_completed":
            pass  # Workers already tracked via worker_completed

    # Infer crash if no terminal event
    if stop_reason is None:
        stop_reason = StopReason.CRASHED

    # Detect incomplete iteration
    incomplete_iteration = None
    if not iteration_finalized and current_iteration_dim:
        completed_count = len(current_iteration_completed)
        if completed_count < current_iteration_worker_count:
            # Build missing tasks from iteration_started data
            missing_tasks = []
            missing_ids = []
            for task_data in current_iteration_tasks:
                tid = task_data["experiment_id"]
                if tid not in current_iteration_completed_ids:
                    missing_ids.append(tid)
                    missing_tasks.append(ExperimentTask(
                        experiment_id=tid,
                        params=task_data["params"],
                        command=task_data["command"],
                        baseline_commit=task_data["baseline_commit"],
                    ))
            incomplete_iteration = IncompleteIteration(
                dimension=current_iteration_dim,
                total_workers=current_iteration_worker_count,
                completed_experiments=current_iteration_completed,
                missing_experiment_ids=missing_ids,
                missing_tasks=missing_tasks,
            )
        else:
            # Iteration was actually complete, just not finalized
            _finalize_iteration(
                current_iteration_dim, current_iteration_completed,
                dimensions_explored,
            )

    return RunSnapshot(
        run_id=run_id,
        workload_name=workload_name,
        workload_spec_hash=workload_spec_hash,
        budget_config=budget_config,
        mode=mode,
        active_baselines=active_baselines,
        baseline_history=baseline_history,
        dimensions_explored=dimensions_explored,
        discovered_dimensions=discovered_dimensions,
        experiments=experiments,
        history=history,
        total_cost_usd=total_cost_usd,
        total_experiments_run=total_experiments_run,
        elapsed_time=elapsed_time,
        consecutive_no_improvement=consecutive_no_improvement,
        incomplete_iteration=incomplete_iteration,
        stop_reason=stop_reason,
    )


def _read_events(events_path: Path) -> list[dict]:
    """Read JSONL event file."""
    events = []
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def _finalize_iteration(
    dimension: str,
    completed: list[ExperimentSummary],
    dimensions_explored: list[DimensionOutcome],
    metric_direction: str = "lower",
):
    """Record a completed iteration as a DimensionOutcome."""
    values_tested = []
    best_metric = None
    best_value = None
    for exp in completed:
        val = str(list(exp.params.values())[0]) if exp.params else "?"
        values_tested.append(val)
        if exp.metric is not None:
            if best_metric is None:
                best_metric = exp.metric
                best_value = val
            elif metric_direction == "lower" and exp.metric < best_metric:
                best_metric = exp.metric
                best_value = val
            elif metric_direction == "higher" and exp.metric > best_metric:
                best_metric = exp.metric
                best_value = val

    dimensions_explored.append(DimensionOutcome(
        name=dimension,
        values_tested=values_tested,
        winner=best_value,
        metric_improvement=None,  # Would need baseline comparison
    ))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_snapshot.py -v`
Expected: All snapshot tests PASS

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/core/snapshot.py tests/test_snapshot.py
git commit -m "feat: implement build_snapshot() for event log replay"
```

---

## Chunk 2: Budget, Decision Maker, and Coordinator Resume

### Task 5: `BudgetTracker.from_snapshot()`, Elapsed Offset, and `exhaustion_reason`

**Files:**
- Modify: `chaosengineer/core/budget.py`
- Modify: `tests/test_budget.py`

- [ ] **Step 1: Write tests for `from_snapshot()` and elapsed offset**

Add to `tests/test_budget.py`:

```python
import time

from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.models import BudgetConfig


class TestBudgetFromSnapshot:
    def test_from_snapshot_restores_state(self):
        config = BudgetConfig(max_experiments=20, max_api_cost=10.0)
        tracker = BudgetTracker.from_snapshot(
            config=config,
            experiments_run=5,
            cost_spent=3.50,
            elapsed_offset=120.0,
            consecutive_no_improvement=2,
        )
        assert tracker.experiments_run == 5
        assert tracker.spent_usd == 3.50
        assert tracker.consecutive_no_improvement == 2
        assert tracker.remaining_experiments == 15
        assert tracker.remaining_cost == 6.50

    def test_elapsed_offset_added_to_elapsed_seconds(self):
        config = BudgetConfig(max_wall_time_seconds=300)
        tracker = BudgetTracker.from_snapshot(
            config=config,
            experiments_run=0,
            cost_spent=0.0,
            elapsed_offset=100.0,
            consecutive_no_improvement=0,
        )
        tracker.start()
        # elapsed_seconds should be >= 100 (offset + small real time)
        assert tracker.elapsed_seconds >= 100.0
        assert tracker.remaining_time <= 200.0

    def test_exhausted_with_offset(self):
        config = BudgetConfig(max_wall_time_seconds=100)
        tracker = BudgetTracker.from_snapshot(
            config=config,
            experiments_run=0,
            cost_spent=0.0,
            elapsed_offset=100.0,  # Already spent all time
            consecutive_no_improvement=0,
        )
        tracker.start()
        assert tracker.is_exhausted()


class TestExhaustionReason:
    def test_experiment_exhaustion(self):
        config = BudgetConfig(max_experiments=5)
        tracker = BudgetTracker(config)
        for _ in range(5):
            tracker.record_experiment()
        assert tracker.exhaustion_reason == "budget_exhausted"

    def test_cost_exhaustion(self):
        config = BudgetConfig(max_api_cost=10.0)
        tracker = BudgetTracker(config)
        tracker.add_cost(10.0)
        assert tracker.exhaustion_reason == "budget_exhausted"

    def test_plateau_exhaustion(self):
        config = BudgetConfig(max_plateau_iterations=3)
        tracker = BudgetTracker(config)
        for _ in range(3):
            tracker.record_no_improvement()
        assert tracker.exhaustion_reason == "plateau"

    def test_not_exhausted(self):
        config = BudgetConfig(max_experiments=10)
        tracker = BudgetTracker(config)
        assert tracker.exhaustion_reason is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_budget.py::TestBudgetFromSnapshot -v`
Expected: FAIL — `AttributeError: type object 'BudgetTracker' has no attribute 'from_snapshot'`

- [ ] **Step 3: Implement `from_snapshot()`, elapsed offset, and `exhaustion_reason`**

In `chaosengineer/core/budget.py`, add the `_elapsed_offset` field, classmethod, and property:

```python
# In __init__, add after line 19:
    self._elapsed_offset: float = 0.0

# Add classmethod after __init__:
@classmethod
def from_snapshot(
    cls,
    config: BudgetConfig,
    experiments_run: int,
    cost_spent: float,
    elapsed_offset: float,
    consecutive_no_improvement: int,
) -> "BudgetTracker":
    """Create a tracker pre-loaded with prior run state."""
    tracker = cls(config)
    tracker.experiments_run = experiments_run
    tracker.spent_usd = cost_spent
    tracker._elapsed_offset = elapsed_offset
    tracker.consecutive_no_improvement = consecutive_no_improvement
    return tracker

# Modify elapsed_seconds property (lines 48-52) to include offset:
@property
def elapsed_seconds(self) -> float:
    if self._start_time is None:
        return self._elapsed_offset
    return (time.monotonic() - self._start_time) + self._elapsed_offset

# Add exhaustion_reason property (after is_exhausted method):
@property
def exhaustion_reason(self) -> str | None:
    """Return the reason budget is exhausted, or None."""
    if self.config.max_experiments is not None and self.experiments_run >= self.config.max_experiments:
        return "budget_exhausted"
    if self.config.max_api_cost is not None and self.spent_usd >= self.config.max_api_cost:
        return "budget_exhausted"
    if self.config.max_wall_time_seconds is not None and self._start_time is not None:
        if self.elapsed_seconds >= self.config.max_wall_time_seconds:
            return "time_exhausted"
    if self.config.max_plateau_iterations is not None:
        if self.consecutive_no_improvement >= self.config.max_plateau_iterations:
            return "plateau"
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_budget.py -v`
Expected: All budget tests PASS (existing + new)

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/core/budget.py tests/test_budget.py
git commit -m "feat: add BudgetTracker.from_snapshot() with elapsed time offset"
```

---

### Task 6: `DecisionMaker.set_prior_context()` Interface

**Files:**
- Modify: `chaosengineer/core/interfaces.py:23-42`
- Modify: `chaosengineer/llm/decision_maker.py`
- Modify: `tests/test_simulator.py`

Note: `ScriptedDecisionMaker` in `testing/simulator.py` inherits the no-op default from the ABC — no code change needed there.

- [ ] **Step 1: Write test for `set_prior_context()`**

Add to `tests/test_simulator.py`:

```python
class TestSetPriorContext:
    def test_scripted_decision_maker_accepts_prior_context(self):
        """ScriptedDecisionMaker.set_prior_context() is a no-op but shouldn't error."""
        dm = ScriptedDecisionMaker(plans=[])
        dm.set_prior_context("Prior state: explored lr, bs. Baseline: 2.41")
        # No error, no effect on behavior
        assert dm.pick_next_dimension([], [], []) is None
```

Add to `tests/test_llm_decision_maker.py`:

```python
class TestSetPriorContext:
    def test_prior_context_stored(self, tmp_path):
        """LLMDecisionMaker stores prior context for next prompt."""
        from unittest.mock import MagicMock
        from chaosengineer.llm.decision_maker import LLMDecisionMaker
        from chaosengineer.workloads.parser import WorkloadSpec
        from chaosengineer.core.models import BudgetConfig

        harness = MagicMock()
        spec = WorkloadSpec(
            name="test", primary_metric="loss", metric_direction="lower",
            execution_command="echo", workers_available=1,
            budget=BudgetConfig(max_experiments=1),
        )
        dm = LLMDecisionMaker(harness, spec, tmp_path)
        dm.set_prior_context("Explored: lr, bs. Best: 2.41")
        assert dm._prior_context == "Explored: lr, bs. Best: 2.41"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_simulator.py::TestSetPriorContext tests/test_llm_decision_maker.py::TestSetPriorContext -v`
Expected: FAIL — `set_prior_context` not found

- [ ] **Step 3: Add `set_prior_context()` to DecisionMaker ABC**

In `chaosengineer/core/interfaces.py`, add after the `discover_diverse_options` method (after line 41):

```python
def set_prior_context(self, context: str) -> None:
    """Provide factual summary of prior run state for resume.

    Default is a no-op. LLMDecisionMaker overrides to prepend context
    to subsequent prompts.
    """
    pass
```

- [ ] **Step 4: Override in `LLMDecisionMaker`**

In `chaosengineer/llm/decision_maker.py`, add field and method:

```python
# In __init__, add after line 50:
    self._prior_context: str | None = None

# Add method:
def set_prior_context(self, context: str) -> None:
    self._prior_context = context

# In _build_pick_prompt(), prepend prior context if set:
# At the start of the prompt string, if self._prior_context is not None,
# add: f"\n## Resume Context\n{self._prior_context}\n"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_simulator.py::TestSetPriorContext tests/test_llm_decision_maker.py::TestSetPriorContext -v`
Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add chaosengineer/core/interfaces.py chaosengineer/llm/decision_maker.py tests/test_simulator.py tests/test_llm_decision_maker.py
git commit -m "feat: add set_prior_context() to DecisionMaker interface"
```

---

### Task 7: DecisionLogger

**Files:**
- Create: `chaosengineer/core/decision_log.py`
- Create: `tests/test_decision_log.py`

- [ ] **Step 1: Write tests for DecisionLogger**

```python
# tests/test_decision_log.py
"""Tests for DecisionLogger."""

import json

from chaosengineer.core.decision_log import DecisionLogger


class TestDecisionLogger:
    def test_log_dimension_selected(self, tmp_path):
        logger = DecisionLogger(tmp_path)
        logger.log_dimension_selected(
            dimension="learning_rate",
            reasoning="High potential for improvement",
            alternatives=["batch_size", "dropout"],
        )
        entries = _read_jsonl(tmp_path / "decisions.jsonl")
        assert len(entries) == 1
        assert entries[0]["type"] == "dimension_selected"
        assert entries[0]["dimension"] == "learning_rate"
        assert entries[0]["reasoning"] == "High potential for improvement"
        assert "timestamp" in entries[0]

    def test_log_results_evaluated(self, tmp_path):
        logger = DecisionLogger(tmp_path)
        logger.log_results_evaluated(
            dimension="lr",
            reasoning="0.001 gave best loss",
            winner="0.001",
            metrics={"exp-0-0": 2.5, "exp-0-1": 2.8},
        )
        entries = _read_jsonl(tmp_path / "decisions.jsonl")
        assert entries[0]["type"] == "results_evaluated"
        assert entries[0]["winner"] == "0.001"

    def test_log_diverse_options(self, tmp_path):
        logger = DecisionLogger(tmp_path)
        logger.log_diverse_options(
            dimension="augmentation",
            reasoning="Exploring data augmentation strategies",
            options=["cutmix", "mixup"],
        )
        entries = _read_jsonl(tmp_path / "decisions.jsonl")
        assert entries[0]["type"] == "diverse_options_generated"
        assert entries[0]["options"] == ["cutmix", "mixup"]

    def test_multiple_entries_appended(self, tmp_path):
        logger = DecisionLogger(tmp_path)
        logger.log_dimension_selected("lr", "reason1", [])
        logger.log_dimension_selected("bs", "reason2", [])
        entries = _read_jsonl(tmp_path / "decisions.jsonl")
        assert len(entries) == 2


def _read_jsonl(path):
    entries = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_decision_log.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement DecisionLogger**

```python
# chaosengineer/core/decision_log.py
"""DecisionLogger — persists LLM reasoning for observability."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class DecisionLogger:
    """Writes LLM decision reasoning to decisions.jsonl. Write-only, not used by resume."""

    def __init__(self, output_dir: Path):
        self.path = output_dir / "decisions.jsonl"

    def log_dimension_selected(
        self, dimension: str, reasoning: str, alternatives: list[str]
    ) -> None:
        self._append({
            "type": "dimension_selected",
            "dimension": dimension,
            "reasoning": reasoning,
            "alternatives": alternatives,
        })

    def log_results_evaluated(
        self, dimension: str, reasoning: str, winner: str | None, metrics: dict[str, Any]
    ) -> None:
        self._append({
            "type": "results_evaluated",
            "dimension": dimension,
            "reasoning": reasoning,
            "winner": winner,
            "metrics": metrics,
        })

    def log_diverse_options(
        self, dimension: str, reasoning: str, options: list[str]
    ) -> None:
        self._append({
            "type": "diverse_options_generated",
            "dimension": dimension,
            "reasoning": reasoning,
            "options": options,
        })

    def _append(self, entry: dict[str, Any]) -> None:
        entry["timestamp"] = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_decision_log.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/core/decision_log.py tests/test_decision_log.py
git commit -m "feat: add DecisionLogger for LLM reasoning observability"
```

---

### Task 8: Coordinator `resume_from_snapshot()`

**Files:**
- Modify: `chaosengineer/core/coordinator.py`
- Create: `tests/test_resume_coordinator.py`

- [ ] **Step 1: Write tests for coordinator resume**

```python
# tests/test_resume_coordinator.py
"""Tests for Coordinator.resume_from_snapshot()."""

from __future__ import annotations

import json
from pathlib import Path

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

        # Plans: decision maker will return bs dimension next
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

        # Should have run the bs dimension
        events = EventLogger(log_path).read_events("iteration_started")
        assert len(events) >= 1
        # Budget should reflect prior experiments
        assert coordinator.budget.experiments_run >= 4  # 2 prior + 2 new


class TestResumeWithPartialIteration:
    def test_completes_missing_workers(self, tmp_path):
        """Resume with 1/3 workers done, should run remaining 2."""
        spec = _make_spec(BudgetConfig(max_experiments=10))

        # Only need results for the missing experiments
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

        # Should have an iteration_gap_completed event
        gap_events = EventLogger(log_path).read_events("iteration_gap_completed")
        assert len(gap_events) == 1
        assert gap_events[0]["original_completed"] == 1
        assert gap_events[0]["gap_filled"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_resume_coordinator.py -v`
Expected: FAIL — `AttributeError: 'Coordinator' object has no attribute 'resume_from_snapshot'`

- [ ] **Step 3: Implement `resume_from_snapshot()` on Coordinator**

In `chaosengineer/core/coordinator.py`, add the method. Key implementation details:

```python
def resume_from_snapshot(self, snapshot: RunSnapshot, restart_iteration: bool = False):
    """Resume a run from a reconstructed snapshot."""
    from chaosengineer.core.snapshot import RunSnapshot
    from chaosengineer.metrics.logger import Event

    # Restore budget tracker
    self.budget = BudgetTracker.from_snapshot(
        config=snapshot.budget_config,
        experiments_run=snapshot.total_experiments_run,
        cost_spent=snapshot.total_cost_usd,
        elapsed_offset=snapshot.elapsed_time,
        consecutive_no_improvement=snapshot.consecutive_no_improvement,
    )
    self.budget.start()

    # Restore baselines and iteration state
    active_baselines = list(snapshot.active_baselines)
    self.best_baseline = active_baselines[0] if active_baselines else self.best_baseline
    self._iteration = len(snapshot.dimensions_explored)
    self._history = list(snapshot.history)

    # Restore explored dimensions on the spec
    explored_names = {d.name for d in snapshot.dimensions_explored}

    # Restore DIVERSE discovered options
    for dim in self.spec.dimensions:
        if dim.name in snapshot.discovered_dimensions:
            dim.options = snapshot.discovered_dimensions[dim.name]

    # Set prior context on decision maker
    context_lines = ["Previous run state (resuming):"]
    for d in snapshot.dimensions_explored:
        context_lines.append(f"- Explored {d.name}: winner={d.winner}")
    bl_str = ", ".join(f"{b.metric_value}" for b in active_baselines)
    context_lines.append(f"- Active baselines: {bl_str}")
    context_lines.append(f"- Experiments run: {snapshot.total_experiments_run}")
    context_lines.append(f"- Budget spent: ${snapshot.total_cost_usd:.2f}")
    self.decision_maker.set_prior_context("\n".join(context_lines))

    # Log run_resumed event
    self._log(Event("run_resumed", {
        "original_run_id": snapshot.run_id,
        "restart_iteration": restart_iteration,
        "snapshot_summary": {
            "dimensions_explored": len(snapshot.dimensions_explored),
            "experiments_completed": snapshot.total_experiments_run,
        },
    }))

    # Handle incomplete iteration
    if snapshot.incomplete_iteration and not restart_iteration:
        self._complete_partial_iteration(snapshot.incomplete_iteration, active_baselines)
        self._iteration += 1
    elif snapshot.incomplete_iteration and restart_iteration:
        pass  # Dimension returned to pool, LLM will re-pick

    # Enter normal run loop (skip_init=True bypasses run_started event,
    # budget.start(), and diverse discovery since we already handled those)
    self._run_loop(active_baselines=active_baselines)
```

**Refactoring `run()` into `run()` + `_run_loop()`:**

The existing `run()` method (coordinator.py:95-195) needs to be split. Extract the while loop and its post-loop logic into a new `_run_loop()` method that `run()` calls after init. This way `resume_from_snapshot()` can call `_run_loop()` directly.

```python
def run(self) -> None:
    """Execute the coordinator loop until budget or dimensions exhausted."""
    self._log(Event(
        event="run_started",
        data={
            "run_id": self.run_state.run_id,
            "workload": self.spec.name,
            "mode": self.run_state.mode,
            "metric_direction": self.spec.metric_direction,
            "budget": self.budget.config.to_dict(),
            "baseline": self.best_baseline.to_dict(),
            "workload_spec_hash": self.spec.spec_hash(),
        },
    ))
    self.budget.start()
    self.run_state.start_time = time.time()

    self._discover_diverse_dimensions()

    active_baselines = [self.best_baseline]
    self._run_loop(active_baselines)

def _run_loop(self, active_baselines: list[Baseline]) -> None:
    """Main coordinator loop. Shared by run() and resume_from_snapshot()."""
    all_dimensions_exhausted = True  # assume until proven otherwise

    while not self.budget.is_exhausted():
        next_active: list[Baseline] = []
        for baseline in active_baselines:
            if self.budget.is_exhausted():
                all_dimensions_exhausted = False
                break

            plan = self.decision_maker.pick_next_dimension(
                dimensions=self.spec.dimensions,
                baselines=[baseline],
                history=self._history,
            )
            if plan is None:
                continue  # this branch has no more dimensions

            # ... existing budget trim and iteration logic unchanged ...

            self._iteration += 1
            self.run_state.current_iteration = self._iteration
            self.run_state.dimensions_explored.append(plan.dimension_name)

        if not next_active:
            break
        active_baselines = next_active

    # Exit branching: paused vs completed
    self.run_state.end_time = time.time()
    self.run_state.total_experiments_run = self.budget.experiments_run
    self.run_state.total_cost_usd = self.budget.spent_usd

    if all_dimensions_exhausted and not self.budget.is_exhausted():
        self._log(Event("run_completed", data={
            "best_metric": self.best_baseline.metric_value,
            "total_experiments": self.budget.experiments_run,
            "total_cost_usd": self.budget.spent_usd,
        }))
    else:
        reason = self.budget.exhaustion_reason or "unknown"
        self._log(Event("run_paused", data={
            "reason": reason,
            "last_iteration": self._iteration,
            "budget_state": self.budget.snapshot(),
            "active_baselines": [b.to_dict() for b in active_baselines],
        }))
```

**Important:** The `all_dimensions_exhausted` flag tracks whether the loop exited because all branches returned `plan=None` (true completion) vs budget exhaustion. If `plan is None` for ALL baselines in a cycle, `next_active` will be empty and the loop breaks with `all_dimensions_exhausted=True`. If budget exhaustion breaks the inner loop, the flag is set to `False`.

Also add `_complete_partial_iteration()`:

```python
def _complete_partial_iteration(self, incomplete: IncompleteIteration, active_baselines: list[Baseline]):
    """Run missing workers from an interrupted iteration and evaluate."""
    from chaosengineer.metrics.logger import Event

    new_results = self.executor.run_experiments(incomplete.missing_tasks)

    # Build combined result list
    # Map existing completed experiments to (Experiment, ExperimentResult) tuples.
    # Use baseline_commit from the first missing task (they share the same baseline).
    baseline_commit = incomplete.missing_tasks[0].baseline_commit if incomplete.missing_tasks else ""

    all_pairs = []
    for exp_summary in incomplete.completed_experiments:
        from chaosengineer.core.models import Experiment, ExperimentResult as ER
        exp = Experiment(
            experiment_id=exp_summary.experiment_id,
            dimension=exp_summary.dimension,
            params=exp_summary.params,
            baseline_commit=baseline_commit,
        )
        result = ER(
            primary_metric=exp_summary.metric if exp_summary.metric is not None else 0.0,
            cost_usd=exp_summary.cost_usd,
        )
        all_pairs.append((exp, result))

    for task, result in zip(incomplete.missing_tasks, new_results):
        exp = Experiment(
            experiment_id=task.experiment_id,
            dimension=incomplete.dimension,
            params=task.params,
            baseline_commit=task.baseline_commit,
        )
        all_pairs.append((exp, result))
        # Log as worker_completed/worker_failed
        self.budget.record_experiment()
        self.budget.add_cost(result.cost_usd)
        self._history.append({
            "experiment_id": task.experiment_id,
            "dimension": incomplete.dimension,
            "params": task.params,
            "metric": result.primary_metric,
            "status": "completed" if result.error_message is None else "failed",
        })

    self._log(Event("iteration_gap_completed", {
        "dimension": incomplete.dimension,
        "original_completed": len(incomplete.completed_experiments),
        "gap_filled": len(new_results),
    }))

    # Build a DimensionPlan for evaluation
    plan = DimensionPlan(
        dimension_name=incomplete.dimension,
        values=[task.params for task in incomplete.missing_tasks]
               + [e.params for e in incomplete.completed_experiments],
    )
    active_baselines[:] = self._evaluate_iteration(plan, all_pairs, active_baselines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_resume_coordinator.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add chaosengineer/core/coordinator.py tests/test_resume_coordinator.py
git commit -m "feat: add Coordinator.resume_from_snapshot() with partial iteration completion"
```

---

## Chunk 3: CLI — Menu, Resume Subcommand, Run Guard, E2E

### Task 9: Interactive Menu Utility

**Files:**
- Create: `chaosengineer/cli_menu.py`
- Create: `tests/test_cli_menu.py`

- [ ] **Step 1: Write tests for menu utility**

```python
# tests/test_cli_menu.py
"""Tests for interactive CLI menu."""

from unittest.mock import patch
from chaosengineer.cli_menu import select


class TestMenuNonInteractive:
    """Test the fallback mode (non-interactive terminal)."""

    @patch("chaosengineer.cli_menu._is_interactive", return_value=False)
    @patch("builtins.input", return_value="1")
    def test_fallback_returns_first_option(self, mock_input, mock_interactive):
        result = select("Pick one:", ["Alpha", "Beta", "Gamma"])
        assert result == 0

    @patch("chaosengineer.cli_menu._is_interactive", return_value=False)
    @patch("builtins.input", return_value="2")
    def test_fallback_returns_second_option(self, mock_input, mock_interactive):
        result = select("Pick one:", ["Alpha", "Beta", "Gamma"])
        assert result == 1

    @patch("chaosengineer.cli_menu._is_interactive", return_value=False)
    @patch("builtins.input", return_value="invalid")
    def test_fallback_returns_default_on_invalid(self, mock_input, mock_interactive):
        result = select("Pick one:", ["Alpha", "Beta"], default=0)
        assert result == 0


class TestMenuHelpers:
    def test_format_options_non_interactive(self):
        from chaosengineer.cli_menu import _format_options_text
        text = _format_options_text(["Alpha", "Beta", "Gamma"])
        assert "1) Alpha" in text
        assert "2) Beta" in text
        assert "3) Gamma" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_cli_menu.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement menu utility**

```python
# chaosengineer/cli_menu.py
"""Interactive arrow-key menu for CLI prompts."""

from __future__ import annotations

import sys


def _is_interactive() -> bool:
    """Check if stdin is a terminal."""
    return hasattr(sys.stdin, "isatty") and sys.stdin.isatty()


def _format_options_text(options: list[str]) -> str:
    """Format options as a numbered list for non-interactive mode."""
    return "\n".join(f"  {i + 1}) {opt}" for i, opt in enumerate(options))


def select(prompt: str, options: list[str], default: int = 0) -> int:
    """Show interactive menu, return selected index.

    Uses raw terminal input for arrow key navigation when available.
    Falls back to numbered list with text input for non-interactive terminals.
    """
    if not _is_interactive():
        return _select_text(prompt, options, default)
    return _select_interactive(prompt, options, default)


def _select_text(prompt: str, options: list[str], default: int) -> int:
    """Non-interactive fallback: numbered list with text input."""
    print(f"\n{prompt}\n")
    print(_format_options_text(options))
    try:
        choice = input(f"\nEnter choice [1-{len(options)}]: ").strip()
        idx = int(choice) - 1
        if 0 <= idx < len(options):
            return idx
    except (ValueError, EOFError):
        pass
    return default


def _select_interactive(prompt: str, options: list[str], default: int) -> int:
    """Interactive mode: arrow keys + Enter."""
    import tty
    import termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    selected = default

    try:
        tty.setraw(fd)
        _render(prompt, options, selected)

        while True:
            ch = sys.stdin.read(1)
            if ch == "\r" or ch == "\n":  # Enter
                break
            elif ch == "\x1b":  # Escape sequence
                seq = sys.stdin.read(2)
                if seq == "[A":  # Up arrow
                    selected = (selected - 1) % len(options)
                elif seq == "[B":  # Down arrow
                    selected = (selected + 1) % len(options)
                _render(prompt, options, selected)
            elif ch == "\x03":  # Ctrl+C
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        # Move below the menu
        sys.stdout.write("\n")
        sys.stdout.flush()

    return selected


def _render(prompt: str, options: list[str], selected: int) -> None:
    """Render the menu to stdout."""
    # Move to start and clear
    lines = 2 + len(options)  # prompt + blank line + options
    sys.stdout.write(f"\r\033[{lines}A\033[J" if selected >= 0 else "")
    sys.stdout.write(f"\n{prompt}\n\n")
    for i, opt in enumerate(options):
        marker = "\033[1m  → \033[0m" if i == selected else "    "
        sys.stdout.write(f"{marker}{opt}\n")
    sys.stdout.flush()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_cli_menu.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/cli_menu.py tests/test_cli_menu.py
git commit -m "feat: add interactive arrow-key CLI menu utility"
```

---

### Task 10: Resume Subcommand and Run Guard

**Files:**
- Modify: `chaosengineer/cli.py`
- Modify: `tests/test_cli_run.py`

- [ ] **Step 1: Write tests for resume CLI and run guard**

Add to `tests/test_cli_run.py`:

```python
import json


class TestResumeSubcommand:
    def test_resume_parser_accepts_output_dir(self):
        """Resume subcommand parses output-dir argument."""
        from chaosengineer.cli import main
        import sys

        # Just test argparse setup — resume with non-existent dir should error gracefully
        with pytest.raises(SystemExit):
            sys.argv = ["chaosengineer", "resume", "/nonexistent"]
            main()

    def test_resume_parser_accepts_budget_extensions(self):
        """Resume accepts --add-cost, --add-experiments, --add-time flags."""
        import argparse
        from chaosengineer.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "resume", "/tmp/output",
            "--add-cost", "5.0",
            "--add-experiments", "10",
            "--add-time", "3600",
        ])
        assert args.add_cost == 5.0
        assert args.add_experiments == 10
        assert args.add_time == 3600

    def test_resume_parser_accepts_restart_iteration(self):
        import argparse
        from chaosengineer.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["resume", "/tmp/output", "--restart-iteration"])
        assert args.restart_iteration is True


class TestRunGuard:
    def test_detects_resumable_session(self, tmp_path):
        """Run guard detects events.jsonl with no run_completed."""
        from chaosengineer.cli import _check_resumable_session

        events_path = tmp_path / "events.jsonl"
        events_path.write_text(json.dumps({
            "event": "run_started", "run_id": "run-1",
            "workload_name": "test", "ts": "2026-01-01T00:00:00Z",
        }) + "\n")

        result = _check_resumable_session(tmp_path)
        assert result is not None
        assert result["run_id"] == "run-1"

    def test_no_guard_for_completed_run(self, tmp_path):
        """Run guard returns None for a completed run."""
        from chaosengineer.cli import _check_resumable_session

        events_path = tmp_path / "events.jsonl"
        events_path.write_text(
            json.dumps({"event": "run_started", "run_id": "run-1"}) + "\n"
            + json.dumps({"event": "run_completed", "best_metric": 2.0}) + "\n"
        )

        result = _check_resumable_session(tmp_path)
        assert result is None

    def test_no_guard_without_events_file(self, tmp_path):
        """No events.jsonl means no resumable session."""
        from chaosengineer.cli import _check_resumable_session

        result = _check_resumable_session(tmp_path)
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_cli_run.py::TestResumeSubcommand tests/test_cli_run.py::TestRunGuard -v`
Expected: FAIL — functions not found

- [ ] **Step 3: Implement `_build_parser()`, resume subcommand, run guard, and `_execute_resume()`**

In `chaosengineer/cli.py`, make these changes:

**3a. Extract parser building into `_build_parser()`:**

Move the argparse setup from `main()` into a separate function:

```python
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="chaosengineer")
    subparsers = parser.add_subparsers(dest="command")

    # Existing: test subparser (unchanged)
    test_parser = subparsers.add_parser("test", ...)
    # ... existing arguments ...

    # Existing: run subparser — add --force-fresh flag
    run_parser = subparsers.add_parser("run", ...)
    # ... existing arguments ...
    run_parser.add_argument("--force-fresh", action="store_true",
                            help="Skip run guard prompt, start fresh even if resumable session exists")

    # NEW: resume subparser
    resume_parser = subparsers.add_parser("resume", help="Resume a partially-completed run")
    resume_parser.add_argument("output_dir", type=str,
                               help="Path to output directory with events.jsonl")
    resume_parser.add_argument("--add-cost", type=float, default=0,
                               help="Add USD to cost budget")
    resume_parser.add_argument("--add-experiments", type=int, default=0,
                               help="Add to experiment budget")
    resume_parser.add_argument("--add-time", type=float, default=0,
                               help="Add seconds to time budget")
    resume_parser.add_argument("--restart-iteration", action="store_true",
                               help="Discard partial iteration, restart from scratch")

    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "test":
        _execute_test(args)
    elif args.command == "run":
        _execute_run(args)
    elif args.command == "resume":
        _execute_resume(args)
    else:
        parser.print_help()
```

**3b. Add `_check_resumable_session()`:**

```python
def _check_resumable_session(output_dir: Path) -> dict | None:
    """Check if output_dir has a resumable (non-completed) event log."""
    events_path = output_dir / "events.jsonl"
    if not events_path.exists():
        return None

    run_info = None
    is_completed = False
    with open(events_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("event") == "run_started":
                run_info = entry
            elif entry.get("event") == "run_completed":
                is_completed = True

    if run_info and not is_completed:
        return run_info
    return None
```

**3c. Add run guard to `_execute_run()`:**

At the beginning of `_execute_run(args)`, before any coordinator setup:

```python
def _execute_run(args):
    output_dir = Path(args.output_dir)

    # Run guard: check for resumable session
    if not getattr(args, "force_fresh", False):
        run_info = _check_resumable_session(output_dir)
        if run_info is not None:
            from chaosengineer.cli_menu import select
            from chaosengineer.core.snapshot import build_snapshot
            snap = build_snapshot(output_dir / "events.jsonl")
            dims = len(snap.dimensions_explored)
            best = snap.active_baselines[0].metric_value if snap.active_baselines else "?"

            choice = select(
                f"Found existing run ({dims} dimensions explored, best: {best})",
                [
                    "Resume previous run",
                    "Start fresh (archive existing)",
                    "Cancel",
                ],
            )
            if choice == 0:  # Resume
                print(f"\n  chaosengineer resume {output_dir}\n")
                sys.exit(0)
            elif choice == 1:  # Start fresh
                import shutil
                from datetime import datetime
                bak = output_dir.parent / f"{output_dir.name}.bak-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                shutil.move(str(output_dir), str(bak))
                print(f"Archived existing run to {bak}")
            else:  # Cancel
                sys.exit(0)

    # ... rest of existing _execute_run logic unchanged ...
```

**3d. Add `_execute_resume()`:**

```python
def _execute_resume(args):
    """Execute the resume subcommand."""
    from chaosengineer.core.snapshot import build_snapshot, StopReason

    output_dir = Path(args.output_dir)
    events_path = output_dir / "events.jsonl"

    if not events_path.exists():
        print(f"Error: No events.jsonl found in {output_dir}")
        sys.exit(1)

    snapshot = build_snapshot(events_path)

    if snapshot.stop_reason == StopReason.COMPLETED:
        print("Run already completed. Nothing to resume.")
        sys.exit(0)

    # Apply budget extensions
    bc = snapshot.budget_config
    if args.add_cost > 0:
        bc = BudgetConfig(
            max_api_cost=(bc.max_api_cost or 0) + args.add_cost,
            max_experiments=bc.max_experiments,
            max_wall_time_seconds=bc.max_wall_time_seconds,
            max_plateau_iterations=bc.max_plateau_iterations,
        )
    if args.add_experiments > 0 and bc.max_experiments is not None:
        bc = BudgetConfig(
            max_api_cost=bc.max_api_cost,
            max_experiments=bc.max_experiments + args.add_experiments,
            max_wall_time_seconds=bc.max_wall_time_seconds,
            max_plateau_iterations=bc.max_plateau_iterations,
        )
    if args.add_time > 0 and bc.max_wall_time_seconds is not None:
        bc = BudgetConfig(
            max_api_cost=bc.max_api_cost,
            max_experiments=bc.max_experiments,
            max_wall_time_seconds=bc.max_wall_time_seconds + args.add_time,
            max_plateau_iterations=bc.max_plateau_iterations,
        )
    snapshot.budget_config = bc

    # Check if budget is still exhausted
    budget_tracker = BudgetTracker.from_snapshot(
        config=bc,
        experiments_run=snapshot.total_experiments_run,
        cost_spent=snapshot.total_cost_usd,
        elapsed_offset=snapshot.elapsed_time,
        consecutive_no_improvement=snapshot.consecutive_no_improvement,
    )
    budget_tracker.start()
    if budget_tracker.is_exhausted():
        print("Error: Budget is still exhausted after extensions.")
        print("Use --add-cost, --add-experiments, or --add-time to extend.")
        sys.exit(1)

    # Print resume summary
    dims = len(snapshot.dimensions_explored)
    best = snapshot.active_baselines[0].metric_value if snapshot.active_baselines else "?"
    print(f"Resuming run {snapshot.run_id} — {dims} dimensions explored, best: {best}, ${snapshot.total_cost_usd:.2f} spent")

    # Crash warning
    if snapshot.stop_reason == StopReason.CRASHED:
        print("\nWarning: This run appears to have crashed (no clean stop event).")
        print("Review the event log before continuing.")
        resp = input("Continue? [y/N] ").strip().lower()
        if resp != "y":
            sys.exit(0)

    # Load workload spec — reparse from the original spec file if available,
    # or reconstruct minimal spec from snapshot data.
    # The workload spec path should be stored in output_dir/workload.md or similar.
    # For now, require the user to be in the same repo with the same workload file.
    spec = parse_workload_spec(Path(args.output_dir) / ".." / f"{snapshot.workload_name}.md")

    # Wire up backends (same logic as _execute_run)
    decision_maker = create_decision_maker(
        backend="sdk",  # or read from snapshot metadata
        spec=spec,
        work_dir=output_dir,
    )
    executor = create_executor(
        backend="subagent",
        spec=spec,
        output_dir=output_dir,
        mode="parallel",
        run_id=snapshot.run_id,
    )
    logger = EventLogger(output_dir / "events.jsonl")

    coordinator = Coordinator(
        spec=spec,
        decision_maker=decision_maker,
        executor=executor,
        logger=logger,
        budget=budget_tracker,
        initial_baseline=snapshot.active_baselines[0],
    )
    coordinator.resume_from_snapshot(
        snapshot, restart_iteration=args.restart_iteration
    )
```

Note: The workload spec loading path is a pragmatic approximation — the implementing agent should check how `_execute_run` resolves spec paths and match that pattern. The key point is that `_execute_resume` builds a full Coordinator and calls `resume_from_snapshot()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/test_cli_run.py::TestResumeSubcommand tests/test_cli_run.py::TestRunGuard -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add chaosengineer/cli.py tests/test_cli_run.py
git commit -m "feat: add resume subcommand and run guard with interactive menu"
```

---

### Task 11: End-to-End Resume Scenario

**Files:**
- Create: `tests/e2e/test_resume_pipeline.py`

- [ ] **Step 1: Write E2E test**

This test runs a full scenario, captures its event log, then resumes from that log.

```python
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
            budget=BudgetConfig(max_experiments=2),  # Only 2 experiments
        )

        # Phase 1: Run with budget for 1 iteration (2 experiments)
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
        snapshot.budget_config = BudgetConfig(max_experiments=4)  # Extend to 4

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
            {"event": "run_started", "run_id": "run-partial", "workload_name": "test",
             "workload_spec_hash": "sha256:abc", "budget": {"max_experiments": 10},
             "mode": "parallel", "baseline": {"commit": "aaa", "metric_value": 3.0, "metric_name": "loss"},
             "ts": "2026-01-01T00:00:00Z"},
            {"event": "iteration_started", "iteration": 0, "dimension": "lr",
             "worker_count": 3, "tasks": [
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
            decision_maker=ScriptedDecisionMaker([]),  # No more dimensions
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
```

- [ ] **Step 2: Run E2E test**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/e2e/test_resume_pipeline.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `cd /Users/alex/CODE/OSS/autoresearch && python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/e2e/test_resume_pipeline.py
git commit -m "test: add end-to-end resume pipeline tests"
```

---

### ~~Task 12~~ — Merged into Task 0

The `iteration_started` task enrichment and all other event field additions are now handled by Task 0 at the start of Chunk 1.
