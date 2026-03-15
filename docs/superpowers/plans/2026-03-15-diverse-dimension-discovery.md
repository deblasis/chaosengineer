# Diverse Dimension Discovery Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `discover_diverse_options` into the coordinator so DIVERSE dimensions get their options discovered before experiments begin, then validate with unit and E2E tests.

**Architecture:** Add `_discover_diverse_dimensions()` method to `Coordinator`, called in `run()` after `run_started`. Uses the existing `DecisionMaker.discover_diverse_options()` ABC method. Unit tests via `ScriptedDecisionMaker`, E2E test with inline workload spec.

**Tech Stack:** Python 3.10+, pytest

---

## Chunk 1: Coordinator Integration + Unit Tests

### Task 1: Add `_discover_diverse_dimensions` to Coordinator

**Files:**
- Modify: `chaosengineer/core/coordinator.py:9-16` (add `DimensionType` import)
- Modify: `chaosengineer/core/coordinator.py:68-79` (call discovery in `run()`)
- Test: `tests/test_coordinator.py`

- [ ] **Step 1: Write failing test — discovers options for DIVERSE dims**

In `tests/test_coordinator.py`, add a new test class after the existing ones:

```python
class TestDiverseDimensionDiscovery:
    """Test _discover_diverse_dimensions integration."""

    def test_discovers_options_for_diverse_dims(self, tmp_output_dir):
        spec = _make_spec(
            dimensions=[
                DimensionSpec(name="optimizer", dim_type=DimensionType.DIVERSE),
            ],
            budget=BudgetConfig(max_experiments=1),
        )
        plans = [
            DimensionPlan(dimension_name="optimizer", values=[{"optimizer": "adam"}]),
        ]
        results = {"exp-0-0": ExperimentResult(primary_metric=0.9)}
        dm = ScriptedDecisionMaker(
            plans, diverse_options={"optimizer": ["adam", "sgd", "rmsprop"]},
        )
        coordinator = Coordinator(
            spec=spec,
            decision_maker=dm,
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="HEAD", metric_value=1.0, metric_name="val_bpb"),
        )
        coordinator.run()

        # Verify options were discovered
        dim = spec.dimensions[0]
        assert dim.options == ["adam", "sgd", "rmsprop"]

        # Verify event was logged
        events = coordinator.logger.read_events(event_type="diverse_discovered")
        assert len(events) == 1
        assert events[0]["dimension"] == "optimizer"
        assert events[0]["options"] == ["adam", "sgd", "rmsprop"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_coordinator.py::TestDiverseDimensionDiscovery::test_discovers_options_for_diverse_dims -v`
Expected: FAIL — `DimensionType` not used in coordinator, no `_discover_diverse_dimensions` method

- [ ] **Step 3: Write failing test — skips non-DIVERSE dims**

```python
    def test_skips_non_diverse_dims(self, tmp_output_dir):
        spec = _make_spec(
            dimensions=[
                DimensionSpec(name="lr", dim_type=DimensionType.DIRECTIONAL, current_value=0.04),
                DimensionSpec(name="act", dim_type=DimensionType.ENUM, options=["relu", "gelu"]),
            ],
            budget=BudgetConfig(max_experiments=1),
        )
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}]),
        ]
        results = {"exp-0-0": ExperimentResult(primary_metric=0.9)}
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="HEAD", metric_value=1.0, metric_name="val_bpb"),
        )
        coordinator.run()

        # No diverse_discovered events
        events = coordinator.logger.read_events(event_type="diverse_discovered")
        assert len(events) == 0

        # Dimensions unchanged
        assert spec.dimensions[0].options is None  # DIRECTIONAL has no options
        assert spec.dimensions[1].options == ["relu", "gelu"]  # ENUM kept its options
```

- [ ] **Step 4: Write failing test — skips DIVERSE dims that already have options**

```python
    def test_skips_diverse_with_existing_options(self, tmp_output_dir):
        spec = _make_spec(
            dimensions=[
                DimensionSpec(
                    name="optimizer", dim_type=DimensionType.DIVERSE,
                    options=["adam", "sgd"],  # already populated
                ),
            ],
            budget=BudgetConfig(max_experiments=1),
        )
        plans = [
            DimensionPlan(dimension_name="optimizer", values=[{"optimizer": "adam"}]),
        ]
        results = {"exp-0-0": ExperimentResult(primary_metric=0.9)}
        dm = ScriptedDecisionMaker(
            plans, diverse_options={"optimizer": ["adam", "sgd", "rmsprop", "adagrad"]},
        )
        coordinator = Coordinator(
            spec=spec,
            decision_maker=dm,
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="HEAD", metric_value=1.0, metric_name="val_bpb"),
        )
        coordinator.run()

        # Options should NOT have been overwritten
        assert spec.dimensions[0].options == ["adam", "sgd"]
        events = coordinator.logger.read_events(event_type="diverse_discovered")
        assert len(events) == 0
```

- [ ] **Step 5: Write failing test — no-op when no DIVERSE dims**

```python
    def test_noop_when_no_diverse_dims(self, tmp_output_dir):
        spec = _make_spec(
            dimensions=[
                DimensionSpec(name="lr", dim_type=DimensionType.DIRECTIONAL, current_value=0.04),
            ],
            budget=BudgetConfig(max_experiments=1),
        )
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}]),
        ]
        results = {"exp-0-0": ExperimentResult(primary_metric=0.9)}
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="HEAD", metric_value=1.0, metric_name="val_bpb"),
        )
        coordinator.run()

        events = coordinator.logger.read_events(event_type="diverse_discovered")
        assert len(events) == 0
        events = coordinator.logger.read_events(event_type="diverse_discovery_failed")
        assert len(events) == 0
```

- [ ] **Step 6: Write failing test — logs failure on empty options**

```python
    def test_logs_failure_on_empty_options(self, tmp_output_dir):
        spec = _make_spec(
            dimensions=[
                DimensionSpec(name="optimizer", dim_type=DimensionType.DIVERSE),
            ],
            budget=BudgetConfig(max_experiments=1),
        )
        plans = [
            DimensionPlan(dimension_name="optimizer", values=[{"optimizer": "adam"}]),
        ]
        results = {"exp-0-0": ExperimentResult(primary_metric=0.9)}
        # Empty diverse_options for "optimizer" — ScriptedDecisionMaker returns []
        dm = ScriptedDecisionMaker(plans, diverse_options={})
        coordinator = Coordinator(
            spec=spec,
            decision_maker=dm,
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="HEAD", metric_value=1.0, metric_name="val_bpb"),
        )
        coordinator.run()

        # Options should remain None
        assert spec.dimensions[0].options is None

        # Failed event logged
        events = coordinator.logger.read_events(event_type="diverse_discovery_failed")
        assert len(events) == 1
        assert events[0]["dimension"] == "optimizer"
```

- [ ] **Step 7: Write failing test — logs failure on exception**

```python
    def test_logs_failure_on_exception(self, tmp_output_dir):
        spec = _make_spec(
            dimensions=[
                DimensionSpec(name="optimizer", dim_type=DimensionType.DIVERSE),
                DimensionSpec(name="strategy", dim_type=DimensionType.DIVERSE),
            ],
            budget=BudgetConfig(max_experiments=1),
        )
        plans = [
            DimensionPlan(dimension_name="strategy", values=[{"strategy": "A"}]),
        ]
        results = {"exp-0-0": ExperimentResult(primary_metric=0.9)}

        class FailingDecisionMaker(ScriptedDecisionMaker):
            def discover_diverse_options(self, dimension_name, context):
                if dimension_name == "optimizer":
                    raise ValueError("LLM returned garbage")
                return ["A", "B"]

        dm = FailingDecisionMaker(plans)
        coordinator = Coordinator(
            spec=spec,
            decision_maker=dm,
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_output_dir / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="HEAD", metric_value=1.0, metric_name="val_bpb"),
        )
        coordinator.run()

        # optimizer failed, strategy succeeded
        assert spec.dimensions[0].options is None
        assert spec.dimensions[1].options == ["A", "B"]

        failed = coordinator.logger.read_events(event_type="diverse_discovery_failed")
        assert len(failed) == 1
        assert failed[0]["dimension"] == "optimizer"

        discovered = coordinator.logger.read_events(event_type="diverse_discovered")
        assert len(discovered) == 1
        assert discovered[0]["dimension"] == "strategy"
```

- [ ] **Step 8: Implement `_discover_diverse_dimensions` and wire into `run()`**

In `chaosengineer/core/coordinator.py`, add `DimensionType` to the import from `models` (line 9):

```python
from chaosengineer.core.models import (
    Baseline,
    DimensionType,
    Experiment,
    ExperimentResult,
    ExperimentStatus,
    Run,
    WorkerState,
)
```

Add the method to `Coordinator` (after the `_log` method, before `run`):

```python
def _discover_diverse_dimensions(self) -> None:
    """Discover options for DIVERSE dimensions before the main loop."""
    for dim in self.spec.dimensions:
        if dim.dim_type == DimensionType.DIVERSE and dim.options is None:
            try:
                options = self.decision_maker.discover_diverse_options(
                    dim.name, self.spec.context,
                )
            except Exception as e:
                self._log(Event(
                    event="diverse_discovery_failed",
                    data={"dimension": dim.name, "error": str(e)},
                ))
                continue
            if not options:
                self._log(Event(
                    event="diverse_discovery_failed",
                    data={"dimension": dim.name, "error": "empty options returned"},
                ))
                continue
            dim.options = options
            self._log(Event(
                event="diverse_discovered",
                data={"dimension": dim.name, "options": options, "count": len(options)},
            ))
```

In `run()`, add the call after `self.budget.start()` (after line 79) and before `active_baselines`:

```python
self._discover_diverse_dimensions()
```

- [ ] **Step 9: Run all coordinator tests**

Run: `pytest tests/test_coordinator.py -v`
Expected: All tests PASS

- [ ] **Step 10: Commit**

```bash
git add chaosengineer/core/coordinator.py tests/test_coordinator.py
git commit -m "feat: wire diverse dimension discovery into coordinator"
```

---

## Chunk 2: E2E Test

### Task 2: E2E test with mixed dimension types

**Files:**
- Create: `tests/e2e/test_diverse_discovery.py`

- [ ] **Step 1: Write the E2E test**

Create `tests/e2e/test_diverse_discovery.py`:

```python
"""E2E test: diverse dimension discovery integration.

Validates that the coordinator discovers options for DIVERSE dimensions
before running experiments, using a mixed workload with DIRECTIONAL,
ENUM, and DIVERSE dimension types.
"""

import pytest

from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.models import (
    Baseline,
    BudgetConfig,
    DimensionSpec,
    DimensionType,
    ExperimentResult,
)
from chaosengineer.core.interfaces import DimensionPlan
from chaosengineer.metrics.logger import EventLogger
from chaosengineer.testing.simulator import ScriptedDecisionMaker
from chaosengineer.testing.executor import ScriptedExecutor
from chaosengineer.workloads.parser import WorkloadSpec


def _make_spec():
    return WorkloadSpec(
        name="diverse-discovery-test",
        context="Testing diverse dimension discovery with mixed dimension types.",
        primary_metric="val_bpb",
        metric_direction="lower",
        execution_command="echo test",
        workers_available=1,
        budget=BudgetConfig(max_experiments=5),
        dimensions=[
            DimensionSpec(name="lr", dim_type=DimensionType.DIRECTIONAL, current_value=0.04),
            DimensionSpec(name="activation", dim_type=DimensionType.ENUM, options=["relu", "gelu"]),
            DimensionSpec(name="optimizer", dim_type=DimensionType.DIVERSE),
        ],
    )


PLANS = [
    DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}]),
    DimensionPlan(dimension_name="optimizer", values=[{"optimizer": "adam"}]),
    DimensionPlan(dimension_name="optimizer", values=[{"optimizer": "sgd"}]),
    DimensionPlan(dimension_name="activation", values=[{"activation": "gelu"}]),
    DimensionPlan(dimension_name="optimizer", values=[{"optimizer": "rmsprop"}]),
]

RESULTS = {
    "exp-0-0": ExperimentResult(primary_metric=0.92),  # lr=0.02 — breakthrough
    "exp-1-0": ExperimentResult(primary_metric=0.88),  # adam — breakthrough
    "exp-2-0": ExperimentResult(primary_metric=0.90),  # sgd — no breakthrough
    "exp-3-0": ExperimentResult(primary_metric=0.85),  # gelu — breakthrough
    "exp-4-0": ExperimentResult(primary_metric=0.87),  # rmsprop — no breakthrough
}


class TestDiverseDimensionDiscovery:
    """Full E2E test: diverse discovery + mixed dimension experiments."""

    def _build_coordinator(self, tmp_path):
        spec = _make_spec()
        dm = ScriptedDecisionMaker(
            PLANS,
            diverse_options={"optimizer": ["adam", "sgd", "rmsprop"]},
        )
        logger = EventLogger(tmp_path / "events.jsonl")
        coordinator = Coordinator(
            spec=spec,
            decision_maker=dm,
            executor=ScriptedExecutor(RESULTS),
            logger=logger,
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="HEAD", metric_value=1.0, metric_name="val_bpb"),
        )
        return coordinator, spec, logger

    def test_diverse_options_discovered(self, tmp_path):
        coordinator, spec, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        # DIVERSE dimension got its options populated
        diverse_dim = [d for d in spec.dimensions if d.dim_type == DimensionType.DIVERSE][0]
        assert diverse_dim.options == ["adam", "sgd", "rmsprop"]

    def test_discovery_event_logged(self, tmp_path):
        coordinator, spec, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        events = logger.read_events(event_type="diverse_discovered")
        assert len(events) == 1
        assert events[0]["dimension"] == "optimizer"
        assert events[0]["options"] == ["adam", "sgd", "rmsprop"]
        assert events[0]["count"] == 3

    def test_discovery_event_ordering(self, tmp_path):
        coordinator, spec, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        events = logger.read_events()
        event_types = [e["event"] for e in events]

        run_started_idx = event_types.index("run_started")
        discovered_idx = event_types.index("diverse_discovered")
        first_iteration_idx = event_types.index("iteration_started")

        assert run_started_idx < discovered_idx < first_iteration_idx

    def test_all_experiments_run(self, tmp_path):
        coordinator, spec, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        assert coordinator.budget.experiments_run == 5

    def test_breakthroughs(self, tmp_path):
        coordinator, spec, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        events = logger.read_events(event_type="breakthrough")
        # Breakthroughs: 0.92 < 1.0, 0.88 < 0.92, 0.85 < 0.88
        assert len(events) == 3
        assert coordinator.best_baseline.metric_value == pytest.approx(0.85)

    def test_non_diverse_dims_unchanged(self, tmp_path):
        coordinator, spec, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        lr_dim = [d for d in spec.dimensions if d.name == "lr"][0]
        act_dim = [d for d in spec.dimensions if d.name == "activation"][0]

        # DIRECTIONAL and ENUM dimensions were not touched by discovery
        assert lr_dim.options is None
        assert act_dim.options == ["relu", "gelu"]
```

- [ ] **Step 2: Run E2E tests**

Run: `pytest tests/e2e/test_diverse_discovery.py -v`
Expected: All 6 tests PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/e2e/test_diverse_discovery.py
git commit -m "feat: add E2E test for diverse dimension discovery"
```
