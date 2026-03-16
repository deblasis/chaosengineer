# TUI Dashboard Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an integrated Textual TUI to ChaosEngineer, toggled on/off during execution via `--tui` flag.

**Architecture:** In-memory EventBridge (ring buffer + async queue) connects the coordinator thread to the Textual app on the main thread. PauseGate handles coordinator-to-TUI decision handoff. ViewManager orchestrates toggling between log mode and TUI mode.

**Tech Stack:** Python, Textual >= 0.47, threading, asyncio

**Spec:** `docs/superpowers/specs/2026-03-16-tui-dashboard-design.md`

---

## Chunk 1: Foundation (EventBridge, PauseGate, StatusDisplay suppression)

### Task 1: Add `textual` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add textual to dependencies**

In `pyproject.toml`, add `"textual>=0.47"` to the `dependencies` list.

- [ ] **Step 2: Lock dependencies**

Run: `uv lock`

- [ ] **Step 3: Install**

Run: `uv sync`

- [ ] **Step 4: Verify import**

Run: `uv run python -c "import textual; print(textual.__version__)"`
Expected: Version >= 0.47

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add textual>=0.47 for TUI dashboard"
```

---

### Task 2: EventBridge

**Files:**
- Create: `chaosengineer/tui/__init__.py`
- Create: `chaosengineer/tui/bridge.py`
- Create: `tests/test_bridge.py`

- [ ] **Step 1: Write failing tests for EventBridge**

Create `tests/test_bridge.py`:

```python
"""Tests for EventBridge."""
import queue
import threading

import pytest

from chaosengineer.tui.bridge import EventBridge


class TestEventBridgePublish:
    def test_publish_stores_in_ring_buffer(self):
        bridge = EventBridge(capacity=5)
        bridge.publish({"event": "run_started", "run_id": "r1"})
        assert len(bridge.snapshot()) == 1
        assert bridge.snapshot()[0]["event"] == "run_started"

    def test_ring_buffer_evicts_oldest(self):
        bridge = EventBridge(capacity=3)
        for i in range(5):
            bridge.publish({"event": f"e{i}"})
        snap = bridge.snapshot()
        assert len(snap) == 3
        assert [e["event"] for e in snap] == ["e2", "e3", "e4"]


class TestEventBridgeSubscribe:
    def test_subscriber_receives_live_events(self):
        bridge = EventBridge()
        q = bridge.subscribe()
        bridge.publish({"event": "test"})
        event = q.get(timeout=1.0)
        assert event["event"] == "test"

    def test_unsubscribe_stops_delivery(self):
        bridge = EventBridge()
        q = bridge.subscribe()
        bridge.unsubscribe(q)
        bridge.publish({"event": "test"})
        assert q.empty()

    def test_slow_subscriber_drops_events(self):
        bridge = EventBridge()
        q = bridge.subscribe()
        # Fill the queue (maxsize=500)
        for i in range(600):
            bridge.publish({"event": f"e{i}"})
        # Queue should have 500 (first 500), rest dropped
        assert q.qsize() == 500


class TestEventBridgeSnapshot:
    def test_snapshot_returns_copy(self):
        bridge = EventBridge()
        bridge.publish({"event": "a"})
        snap = bridge.snapshot()
        bridge.publish({"event": "b"})
        assert len(snap) == 1  # copy, not live reference


class TestEventBridgeThreadSafety:
    def test_concurrent_publish_and_snapshot(self):
        bridge = EventBridge(capacity=100)
        errors = []

        def publisher():
            try:
                for i in range(200):
                    bridge.publish({"event": f"e{i}"})
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(200):
                    bridge.snapshot()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=publisher), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_bridge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'chaosengineer.tui'`

- [ ] **Step 3: Implement EventBridge**

Create `chaosengineer/tui/__init__.py` (empty file).

Create `chaosengineer/tui/bridge.py`:

```python
"""EventBridge — in-memory event store connecting coordinator to TUI."""
from __future__ import annotations

import queue
import threading
from collections import deque


class EventBridge:
    """Thread-safe event store with ring buffer (history) and live notification.

    The coordinator thread calls publish(). The TUI subscribes for live events
    and calls snapshot() for replay on toggle.

    Uses queue.Queue (stdlib thread-safe), NOT asyncio.Queue which is not
    thread-safe across event loops.
    """

    def __init__(self, capacity: int = 200):
        self._buffer: deque[dict] = deque(maxlen=capacity)
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()

    def publish(self, event: dict) -> None:
        """Append event to ring buffer and notify all subscribers."""
        with self._lock:
            self._buffer.append(event)
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

    def snapshot(self) -> list[dict]:
        """Return a copy of the ring buffer for replay."""
        with self._lock:
            return list(self._buffer)

    def subscribe(self) -> queue.Queue[dict]:
        """Register a new subscriber. Returns a thread-safe queue for live events."""
        q: queue.Queue[dict] = queue.Queue(maxsize=500)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        """Remove a subscriber."""
        with self._lock:
            self._subscribers.remove(q)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_bridge.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/tui/__init__.py chaosengineer/tui/bridge.py tests/test_bridge.py
git commit -m "feat(tui): add EventBridge with ring buffer and async subscribers"
```

---

### Task 3: PauseGate

**Files:**
- Create: `chaosengineer/tui/pause_gate.py`
- Create: `tests/test_pause_gate.py`

- [ ] **Step 1: Write failing tests for PauseGate**

Create `tests/test_pause_gate.py`:

```python
"""Tests for PauseGate — coordinator <-> TUI decision handoff."""
import threading
import time

import pytest

from chaosengineer.tui.pause_gate import PauseGate


class TestPauseGateRequestDecision:
    def test_blocks_until_decision_submitted(self):
        gate = PauseGate()
        result = []

        def coordinator():
            choice = gate.request_decision(["continue", "pause"])
            result.append(choice)

        t = threading.Thread(target=coordinator)
        t.start()
        # Give coordinator time to block
        time.sleep(0.05)
        assert not result  # still blocked

        gate.submit_decision("pause")
        t.join(timeout=2.0)
        assert result == ["pause"]

    def test_options_are_set_before_blocking(self):
        gate = PauseGate()

        def coordinator():
            gate.request_decision(["a", "b", "c"])

        t = threading.Thread(target=coordinator)
        t.start()
        time.sleep(0.05)
        assert gate.options == ["a", "b", "c"]
        assert gate.decision_needed.is_set()
        gate.submit_decision("a")
        t.join(timeout=2.0)

    def test_decision_needed_cleared_after_decision(self):
        gate = PauseGate()

        def coordinator():
            gate.request_decision(["x"])

        t = threading.Thread(target=coordinator)
        t.start()
        time.sleep(0.05)
        gate.submit_decision("x")
        t.join(timeout=2.0)
        assert not gate.decision_needed.is_set()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pause_gate.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement PauseGate**

Create `chaosengineer/tui/pause_gate.py`:

```python
"""PauseGate — coordinator <-> TUI pause decision handoff."""
from __future__ import annotations

import threading


class PauseGate:
    """Shared object for blocking coordinator until TUI user makes a decision."""

    def __init__(self):
        self.decision: str | None = None
        self.decision_ready = threading.Event()
        self.decision_needed = threading.Event()
        self.options: list[str] = []

    def request_decision(self, options: list[str]) -> str:
        """Called from coordinator thread. Blocks until TUI user decides."""
        self.options = options
        self.decision = None
        self.decision_ready.clear()
        self.decision_needed.set()
        self.decision_ready.wait()
        self.decision_needed.clear()
        return self.decision

    def submit_decision(self, choice: str) -> None:
        """Called from TUI thread when user picks an option."""
        self.decision = choice
        self.decision_ready.set()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pause_gate.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/tui/pause_gate.py tests/test_pause_gate.py
git commit -m "feat(tui): add PauseGate for coordinator-TUI decision handoff"
```

---

### Task 4: StatusDisplay suppression flag

**Files:**
- Modify: `chaosengineer/core/status.py`
- Modify: `tests/test_status.py`

- [ ] **Step 1: Write failing test for suppression**

Add to `tests/test_status.py`:

```python
class TestStatusDisplaySuppression:
    def test_on_worker_done_suppressed(self, capsys):
        from chaosengineer.core.status import StatusDisplay
        from chaosengineer.core.models import BudgetConfig, ExperimentResult
        from chaosengineer.core.interfaces import ExperimentTask

        sd = StatusDisplay()
        sd.suppressed = True
        sd.on_run_start(BudgetConfig(max_experiments=10))
        task = ExperimentTask("e1", {"lr": 0.01}, "echo 1", "HEAD")
        result = ExperimentResult(primary_metric=1.0, cost_usd=0.5)
        sd.on_worker_done(task, result, 1, 2)
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_on_iteration_done_suppressed(self, capsys):
        from chaosengineer.core.status import StatusDisplay

        sd = StatusDisplay()
        sd.suppressed = True
        sd.on_iteration_done(0, 1.5)
        captured = capsys.readouterr()
        assert captured.err == ""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_status.py::TestStatusDisplaySuppression -v`
Expected: FAIL — `AttributeError: 'StatusDisplay' object has no attribute 'suppressed'`

- [ ] **Step 3: Add suppressed flag to StatusDisplay**

In `chaosengineer/core/status.py`, modify `__init__` to add `self.suppressed: bool = False`.

Add `if self.suppressed: return` as the first line in `on_run_start`, `on_worker_done`, and `on_iteration_done`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_status.py -v`
Expected: All PASS (existing + new)

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/core/status.py tests/test_status.py
git commit -m "feat(tui): add suppressed flag to StatusDisplay"
```

---

### Task 5: Coordinator.extend_budget() and _budget_lock

**Files:**
- Modify: `chaosengineer/core/coordinator.py`
- Create: `tests/test_extend_budget.py`

- [ ] **Step 1: Write failing test for extend_budget**

Create `tests/test_extend_budget.py`:

```python
"""Tests for Coordinator.extend_budget()."""
from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.models import Baseline, BudgetConfig
from chaosengineer.metrics.logger import EventLogger
from chaosengineer.testing.executor import ScriptedExecutor
from chaosengineer.testing.simulator import ScriptedDecisionMaker
from chaosengineer.workloads.parser import WorkloadSpec


def _make_coordinator(tmp_path, budget=None):
    spec = WorkloadSpec(
        name="test", primary_metric="loss", metric_direction="lower",
        execution_command="echo 1", workers_available=1,
        budget=budget or BudgetConfig(max_api_cost=10.0, max_experiments=5),
    )
    return Coordinator(
        spec=spec, decision_maker=ScriptedDecisionMaker([]),
        executor=ScriptedExecutor({}),
        logger=EventLogger(tmp_path / "events.jsonl"),
        budget=BudgetTracker(spec.budget),
        initial_baseline=Baseline("HEAD", 3.0, "loss"),
    )


class TestExtendBudget:
    def test_extend_cost(self, tmp_path):
        c = _make_coordinator(tmp_path)
        c.extend_budget(add_cost=5.0)
        assert c.budget.config.max_api_cost == 15.0

    def test_extend_experiments(self, tmp_path):
        c = _make_coordinator(tmp_path)
        c.extend_budget(add_experiments=3)
        assert c.budget.config.max_experiments == 8

    def test_extend_time(self, tmp_path):
        c = _make_coordinator(tmp_path, BudgetConfig(max_wall_time_seconds=60.0))
        c.extend_budget(add_time=30.0)
        assert c.budget.config.max_wall_time_seconds == 90.0

    def test_extend_preserves_none_fields(self, tmp_path):
        c = _make_coordinator(tmp_path, BudgetConfig(max_api_cost=10.0))
        c.extend_budget(add_experiments=5)
        assert c.budget.config.max_api_cost == 10.0
        assert c.budget.config.max_experiments is None  # was None, stays None

    def test_extend_preserves_plateau(self, tmp_path):
        c = _make_coordinator(tmp_path, BudgetConfig(max_api_cost=10.0, max_plateau_iterations=3))
        c.extend_budget(add_cost=5.0)
        assert c.budget.config.max_plateau_iterations == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_extend_budget.py -v`
Expected: FAIL — `AttributeError: 'Coordinator' object has no attribute 'extend_budget'`

- [ ] **Step 3: Add extend_budget to Coordinator**

In `chaosengineer/core/coordinator.py`:

Add `import threading` to the imports.

In `__init__`, add after `self._status_display = status_display`:

```python
self._budget_lock = threading.Lock()
```

Add new method after `_poll_bus_commands`:

```python
def extend_budget(self, add_cost: float = 0, add_experiments: int = 0,
                  add_time: float = 0) -> None:
    """Extend budget limits. Thread-safe — called from TUI or bus command polling."""
    with self._budget_lock:
        cfg = self.budget.config
        self.budget.config = BudgetConfig(
            max_api_cost=(cfg.max_api_cost + add_cost) if cfg.max_api_cost is not None else None,
            max_experiments=(cfg.max_experiments + add_experiments) if cfg.max_experiments is not None else None,
            max_wall_time_seconds=(cfg.max_wall_time_seconds + add_time) if cfg.max_wall_time_seconds is not None else None,
            max_plateau_iterations=cfg.max_plateau_iterations,
        )
```

Refactor `_poll_bus_commands` to use `self.extend_budget()` instead of inline BudgetConfig mutation. Replace the entire `elif cmd.get("command") == "extend_budget":` block (lines 81-104) with:

```python
elif cmd.get("command") == "extend_budget":
    self.extend_budget(
        add_cost=cmd.get("add_cost_usd", 0),
        add_experiments=cmd.get("add_experiments", 0),
        add_time=cmd.get("add_time_seconds", 0),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_extend_budget.py tests/test_bus_commands.py -v`
Expected: All PASS (new + existing bus command tests still work)

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/core/coordinator.py tests/test_extend_budget.py
git commit -m "feat(tui): add thread-safe Coordinator.extend_budget(), refactor _poll_bus_commands"
```

---

### Task 6: Coordinator accepts optional view_manager and pause_gate

**Files:**
- Modify: `chaosengineer/core/coordinator.py`
- Create: `tests/test_tui_coordinator.py`

- [ ] **Step 1: Write failing test for TUI-mode pause handoff**

Create `tests/test_tui_coordinator.py`:

```python
"""Tests for Coordinator integration with TUI (PauseGate, ViewManager)."""
import threading
import time
from unittest.mock import MagicMock

import pytest

from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.interfaces import DimensionPlan
from chaosengineer.core.models import Baseline, BudgetConfig, ExperimentResult
from chaosengineer.core.pause import PauseController
from chaosengineer.metrics.logger import EventLogger
from chaosengineer.testing.executor import ScriptedExecutor
from chaosengineer.testing.simulator import ScriptedDecisionMaker
from chaosengineer.tui.pause_gate import PauseGate
from chaosengineer.workloads.parser import WorkloadSpec


def _make_spec(budget=None):
    return WorkloadSpec(
        name="test", primary_metric="loss", metric_direction="lower",
        execution_command="echo 1", workers_available=1,
        budget=budget or BudgetConfig(max_experiments=10),
    )


class TestCoordinatorPauseGate:
    def test_uses_pause_gate_when_tui_active(self, tmp_path):
        """Coordinator uses PauseGate instead of interactive menu when TUI is active."""
        spec = _make_spec()
        plans = [DimensionPlan("lr", [{"lr": 0.01}])]
        results = {"exp-0-0": ExperimentResult(primary_metric=2.5)}

        pc = PauseController()
        pc.pause_requested = True

        gate = PauseGate()
        view_manager = MagicMock()
        view_manager.tui_active = True

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_path / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
            pause_controller=pc,
            view_manager=view_manager,
            pause_gate=gate,
        )

        # Run coordinator in thread (it will block on PauseGate)
        coord_result = []
        def run_coord():
            coordinator.run()
            coord_result.append("done")

        t = threading.Thread(target=run_coord)
        t.start()

        # Wait for gate to signal decision needed
        gate.decision_needed.wait(timeout=5.0)
        assert gate.decision_needed.is_set()
        assert "pause" in gate.options

        # Submit pause decision
        gate.submit_decision("pause")
        t.join(timeout=5.0)

        assert coord_result == ["done"]

    def test_falls_back_to_menu_when_tui_not_active(self, tmp_path):
        """When view_manager exists but tui_active=False, uses normal menu."""
        spec = _make_spec()
        plans = [DimensionPlan("lr", [{"lr": 0.01}])]
        results = {"exp-0-0": ExperimentResult(primary_metric=2.5)}

        pc = PauseController()
        pc.pause_requested = True
        pc.show_post_iteration_menu = MagicMock(return_value="pause")

        view_manager = MagicMock()
        view_manager.tui_active = False

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_path / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
            pause_controller=pc,
            view_manager=view_manager,
            pause_gate=PauseGate(),
        )

        coordinator.run()
        pc.show_post_iteration_menu.assert_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tui_coordinator.py -v`
Expected: FAIL — `TypeError: Coordinator.__init__() got an unexpected keyword argument 'view_manager'`

- [ ] **Step 3: Add view_manager and pause_gate to Coordinator**

In `chaosengineer/core/coordinator.py`, modify `__init__` signature to add:

```python
view_manager: "ViewManager | None" = None,
pause_gate: "PauseGate | None" = None,
```

Store as `self._view_manager = view_manager` and `self._pause_gate = pause_gate`.

Modify `_run_loop` pause check (the block at line 167-173) to use PauseGate when TUI is active:

```python
if self._pause_controller and self._pause_controller.pause_requested and self._pause_controller.should_show_menu():
    if self._view_manager and self._view_manager.tui_active and self._pause_gate:
        self._log(Event(
            event="pause_decision_needed",
            data={"options": ["continue", "pause"]},
        ))
        choice = self._pause_gate.request_decision(["continue", "pause"])
    else:
        choice = self._pause_controller.show_post_iteration_menu()
    if choice == "pause":
        self._log_user_pause(active_baselines)
        return
    else:
        self._pause_controller.reset()
```

Apply the same pattern to the post-iteration pause check (lines 264-273):

```python
if self._pause_controller and self._pause_controller.should_show_menu():
    if self._view_manager and self._view_manager.tui_active and self._pause_gate:
        self._log(Event(
            event="pause_decision_needed",
            data={"options": ["continue", "pause"]},
        ))
        choice = self._pause_gate.request_decision(["continue", "pause"])
    else:
        choice = self._pause_controller.show_post_iteration_menu(
            f"Iteration {self._iteration - 1} complete. "
            f"{self.spec.primary_metric}={self.best_baseline.metric_value}"
        )
    if choice == "pause":
        self._log_user_pause(active_baselines)
        return
    else:
        self._pause_controller.reset()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tui_coordinator.py tests/test_pause_coordinator.py tests/test_coordinator.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/core/coordinator.py tests/test_tui_coordinator.py
git commit -m "feat(tui): coordinator uses PauseGate when TUI is active"
```

---

## Chunk 2: EventPublisher bridge integration

### Task 7: Wire EventBridge into EventPublisher

**Files:**
- Modify: `chaosengineer/metrics/publisher.py`
- Modify: `tests/test_publisher.py`

- [ ] **Step 1: Write failing test for bridge integration**

Add to `tests/test_publisher.py`:

```python
class TestEventPublisherBridge:
    def test_publishes_to_bridge_when_provided(self, tmp_path):
        from chaosengineer.tui.bridge import EventBridge

        bridge = EventBridge()
        publisher = EventPublisher(bus_url=None, fallback_path=tmp_path / "events.jsonl", bridge=bridge)
        publisher.log(Event("run_started", data={"run_id": "r1"}))

        snap = bridge.snapshot()
        assert len(snap) == 1
        assert snap[0]["event"] == "run_started"
        assert snap[0]["run_id"] == "r1"

    def test_bridge_receives_all_events(self, tmp_path):
        from chaosengineer.tui.bridge import EventBridge

        bridge = EventBridge()
        publisher = EventPublisher(bus_url=None, fallback_path=tmp_path / "events.jsonl", bridge=bridge)
        publisher.log(Event("run_started", data={"run_id": "r1"}))
        publisher.log(Event("iteration_started", data={"iteration": 0}))
        publisher.log(Event("worker_completed", data={"experiment_id": "e1"}))

        assert len(bridge.snapshot()) == 3

    def test_no_bridge_still_works(self, tmp_path):
        """Without bridge, publisher works exactly as before."""
        publisher = EventPublisher(bus_url=None, fallback_path=tmp_path / "events.jsonl")
        publisher.log(Event("run_started", data={"run_id": "r1"}))
        assert (tmp_path / "events.jsonl").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_publisher.py::TestEventPublisherBridge -v`
Expected: FAIL — `TypeError: EventPublisher.__init__() got an unexpected keyword argument 'bridge'`

- [ ] **Step 3: Add bridge parameter to EventPublisher**

In `chaosengineer/metrics/publisher.py`:

Modify `__init__` signature — add `bridge: "EventBridge | None" = None` parameter. Store as `self._bridge = bridge`.

In the `log` method, after building the `payload` dict (line 47) and before the bus-availability check, add:

```python
if self._bridge is not None:
    self._bridge.publish(payload)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_publisher.py -v`
Expected: All PASS (existing + new)

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/metrics/publisher.py tests/test_publisher.py
git commit -m "feat(tui): wire EventBridge into EventPublisher"
```

---

## Chunk 3: Textual TUI App

### Task 8: TUI App — basic layout with budget gauges, experiment table, event log

**Files:**
- Create: `chaosengineer/tui/app.py`
- Create: `tests/test_tui_app.py`

- [ ] **Step 1: Write Textual snapshot test for initial empty state**

Create `tests/test_tui_app.py`:

```python
"""Tests for the ChaosEngineer TUI app."""
import asyncio
from unittest.mock import MagicMock

import pytest

from chaosengineer.tui.app import ChaosApp
from chaosengineer.tui.bridge import EventBridge
from chaosengineer.tui.pause_gate import PauseGate


@pytest.fixture
def app():
    bridge = EventBridge()
    gate = PauseGate()
    coordinator = MagicMock()
    pause_controller = MagicMock()
    pause_controller.pause_requested = False
    return ChaosApp(
        bridge=bridge,
        pause_gate=gate,
        coordinator=coordinator,
        pause_controller=pause_controller,
    )


class TestChaosAppMounts:
    async def test_app_has_budget_bar(self, app):
        """App should have a budget bar widget."""
        async with app.run_test() as pilot:
            assert app.query_one("#budget-bar") is not None

    async def test_app_has_experiment_table(self, app):
        """App should have an experiment data table."""
        async with app.run_test() as pilot:
            assert app.query_one("#experiment-table") is not None

    async def test_app_has_event_log(self, app):
        """App should have an event log."""
        async with app.run_test() as pilot:
            assert app.query_one("#event-log") is not None

    async def test_app_has_command_bar(self, app):
        """App should have a footer command bar."""
        async with app.run_test() as pilot:
            footer = app.query_one("#command-bar")
            assert footer is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tui_app.py -v`
Expected: FAIL — `ModuleNotFoundError` or import error

- [ ] **Step 3: Implement ChaosApp**

Create `chaosengineer/tui/app.py`:

```python
"""ChaosEngineer TUI — Textual application."""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Label, ProgressBar, RichLog, Static

if TYPE_CHECKING:
    from chaosengineer.core.pause import PauseController
    from chaosengineer.tui.bridge import EventBridge
    from chaosengineer.tui.pause_gate import PauseGate


class BudgetBar(Static):
    """Budget gauges: cost, experiments, time, elapsed clock."""

    DEFAULT_CSS = """
    BudgetBar {
        height: 3;
        padding: 0 1;
    }
    """

    def __init__(self):
        super().__init__()
        self._cost = 0.0
        self._max_cost = None
        self._experiments = 0
        self._max_experiments = None
        self._elapsed = "00:00:00"

    def update_budget(self, cost: float, max_cost: float | None,
                      experiments: int, max_experiments: int | None,
                      elapsed: str) -> None:
        self._cost = cost
        self._max_cost = max_cost
        self._experiments = experiments
        self._max_experiments = max_experiments
        self._elapsed = elapsed
        self._render_content()

    def _render_content(self) -> None:
        cost_str = f"${self._cost:.2f}"
        if self._max_cost is not None:
            cost_str += f"/${self._max_cost:.0f}"

        exp_str = str(self._experiments)
        if self._max_experiments is not None:
            exp_str += f"/{self._max_experiments}"

        self.update(f"Cost: {cost_str}  Experiments: {exp_str}  [{self._elapsed}]")


class ChaosApp(App):
    """Main TUI application for ChaosEngineer."""

    CSS = """
    #budget-bar {
        dock: top;
        height: 3;
        background: $surface;
        border-bottom: solid $primary;
    }
    #experiment-table {
        height: 1fr;
        min-height: 5;
    }
    #event-log {
        height: 1fr;
        min-height: 5;
        border-top: solid $primary;
    }
    #command-bar {
        dock: bottom;
        height: 1;
        background: $surface;
        border-top: solid $primary;
    }
    """

    BINDINGS = [
        ("p", "pause", "Pause"),
        ("e", "extend", "Extend Budget"),
        ("q", "quit_tui", "Quit TUI"),
        ("escape", "quit_tui", "Quit TUI"),
    ]

    def __init__(self, bridge: "EventBridge", pause_gate: "PauseGate",
                 coordinator, pause_controller: "PauseController"):
        super().__init__()
        self._bridge = bridge
        self._pause_gate = pause_gate
        self._coordinator = coordinator
        self._pause_controller = pause_controller
        self._event_queue: "queue.Queue | None" = None

    def compose(self) -> ComposeResult:
        yield BudgetBar(id="budget-bar")
        yield DataTable(id="experiment-table")
        yield RichLog(id="event-log", highlight=True, markup=True)
        yield Static("[P]ause  [E]xtend budget  [Q]uit TUI", id="command-bar")

    def on_mount(self) -> None:
        table = self.query_one("#experiment-table", DataTable)
        table.add_columns("#", "Worker", "Dimension", "Status", "Cost", "Delta")

        # Subscribe to bridge and start consuming
        self._event_queue = self._bridge.subscribe()

        # Replay history
        for event in self._bridge.snapshot():
            self._handle_event(event)

        # Start live consumer
        self.set_interval(0.1, self._poll_events)

    def _poll_events(self) -> None:
        """Drain events from the thread-safe queue."""
        import queue as _queue
        if self._event_queue is None:
            return
        while not self._event_queue.empty():
            try:
                event = self._event_queue.get_nowait()
                self._handle_event(event)
            except _queue.Empty:
                break

    def _handle_event(self, event: dict) -> None:
        """Process a single event and update widgets."""
        event_type = event.get("event", "")
        log = self.query_one("#event-log", RichLog)
        ts = event.get("ts", "")
        if isinstance(ts, str) and "T" in ts:
            ts = ts.split("T")[1][:8]

        log.write(f"{ts} {event_type} {self._event_summary(event)}")

        if event_type == "iteration_started":
            self._on_iteration_started(event)
        elif event_type == "worker_completed":
            self._on_worker_completed(event)
        elif event_type == "worker_failed":
            self._on_worker_failed(event)
        elif event_type == "budget_checkpoint":
            self._on_budget_checkpoint(event)
        elif event_type == "run_completed":
            self._on_run_completed(event)
        elif event_type == "run_failed":
            log.write("[bold red]RUN FAILED[/bold red]")
        elif event_type == "pause_decision_needed":
            self._on_pause_decision_needed(event)

    def _event_summary(self, event: dict) -> str:
        """One-line summary of event data for the log."""
        etype = event.get("event", "")
        if etype == "worker_completed":
            return f"dim={event.get('dimension', '?')} metric={event.get('metric', '?')}"
        if etype == "iteration_started":
            return f"dim={event.get('dimension', '?')} workers={event.get('num_workers', '?')}"
        if etype == "breakthrough":
            return f"new_best={event.get('new_best', '?')}"
        return ""

    def _on_iteration_started(self, event: dict) -> None:
        table = self.query_one("#experiment-table", DataTable)
        tasks = event.get("tasks", [])
        iteration = event.get("iteration", "?")
        for i, task in enumerate(tasks):
            exp_id = task.get("experiment_id", f"exp-{iteration}-{i}")
            table.add_row(
                exp_id, f"W{i+1}", event.get("dimension", "?"),
                "running", "-", "-",
                key=exp_id,
            )

    def _on_worker_completed(self, event: dict) -> None:
        table = self.query_one("#experiment-table", DataTable)
        exp_id = event.get("experiment_id", "")
        metric = event.get("metric", "?")
        cost = event.get("cost_usd", 0)
        try:
            row_key = table.get_row(exp_id)
            # Update in place would require row index — simpler to update via key
            table.update_cell(exp_id, "Status", "done")
            table.update_cell(exp_id, "Cost", f"${cost:.2f}")
            table.update_cell(exp_id, "Delta", f"{metric}")
        except Exception:
            pass  # Row might not exist if we missed iteration_started

    def _on_worker_failed(self, event: dict) -> None:
        table = self.query_one("#experiment-table", DataTable)
        exp_id = event.get("experiment_id", "")
        try:
            table.update_cell(exp_id, "Status", "FAILED")
        except Exception:
            pass

    def _on_budget_checkpoint(self, event: dict) -> None:
        bar = self.query_one("#budget-bar", BudgetBar)
        bar.update_budget(
            cost=event.get("spent_usd", 0),
            max_cost=event.get("remaining_cost"),
            experiments=event.get("experiments_run", 0),
            max_experiments=event.get("remaining_experiments"),
            elapsed=self._format_elapsed(event.get("elapsed_seconds", 0)),
        )

    def _on_run_completed(self, event: dict) -> None:
        log = self.query_one("#event-log", RichLog)
        best = event.get("best_metric", "?")
        total = event.get("total_experiments", "?")
        cost = event.get("total_cost_usd", 0)
        log.write(f"[bold green]RUN COMPLETE[/bold green] best={best} experiments={total} cost=${cost:.2f}")

    def _on_pause_decision_needed(self, event: dict) -> None:
        """Show notification and let user decide via keybindings."""
        log = self.query_one("#event-log", RichLog)
        log.write("[bold yellow]PAUSE REQUESTED[/bold yellow] — Press [P] to pause or [C] to continue")
        self._pending_pause = True

    def action_pause(self) -> None:
        """Handle P key — pause the coordinator."""
        self._pause_controller.pause_requested = True
        if self._pause_gate.decision_needed.is_set():
            self._pause_gate.submit_decision("pause")
        log = self.query_one("#event-log", RichLog)
        log.write("[yellow]Pause submitted[/yellow]")

    def action_extend(self) -> None:
        """Handle E key — extend budget. For now, fixed increment."""
        # TODO: Replace with modal input in future iteration
        self._coordinator.extend_budget(add_cost=5.0, add_experiments=5)
        log = self.query_one("#event-log", RichLog)
        log.write("[green]Budget extended: +$5.00, +5 experiments[/green]")

    def action_quit_tui(self) -> None:
        """Handle Q/Esc — exit TUI mode."""
        if self._pause_gate.decision_needed.is_set():
            self._pause_gate.submit_decision("continue")
        if self._event_queue is not None:
            self._bridge.unsubscribe(self._event_queue)
        self.exit()

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tui_app.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/tui/app.py tests/test_tui_app.py
git commit -m "feat(tui): implement ChaosApp with budget bar, experiment table, event log"
```

---

### Task 9: TUI event handling tests — verify widget updates from events

**Files:**
- Modify: `tests/test_tui_app.py`

- [ ] **Step 1: Write tests for event-driven widget updates**

Add to `tests/test_tui_app.py`:

```python
class TestChaosAppEventHandling:
    async def test_iteration_started_adds_table_rows(self, app):
        """iteration_started event should add rows to experiment table."""
        async with app.run_test() as pilot:
            app._handle_event({
                "ts": "2026-03-16T14:00:00Z",
                "event": "iteration_started",
                "dimension": "lr",
                "num_workers": 2,
                "iteration": 0,
                "tasks": [
                    {"experiment_id": "exp-0-0", "params": {"lr": 0.01}},
                    {"experiment_id": "exp-0-1", "params": {"lr": 0.1}},
                ],
            })
            table = app.query_one("#experiment-table", DataTable)
            assert table.row_count == 2

    async def test_worker_completed_updates_status(self, app):
        """worker_completed event should update row status."""
        async with app.run_test() as pilot:
            app._handle_event({
                "event": "iteration_started",
                "dimension": "lr",
                "iteration": 0,
                "tasks": [{"experiment_id": "exp-0-0"}],
            })
            app._handle_event({
                "event": "worker_completed",
                "experiment_id": "exp-0-0",
                "metric": 0.85,
                "cost_usd": 0.42,
            })
            table = app.query_one("#experiment-table", DataTable)
            # Verify the row was updated (status column should say "done")
            row = table.get_row("exp-0-0")
            assert "done" in str(row)

    async def test_event_log_receives_entries(self, app):
        """Events should appear in the log widget."""
        async with app.run_test() as pilot:
            app._handle_event({
                "ts": "2026-03-16T14:00:00Z",
                "event": "run_started",
                "run_id": "r1",
            })
            log = app.query_one("#event-log", RichLog)
            # RichLog should have at least one line
            assert len(log.lines) > 0

    async def test_run_completed_shows_banner(self, app):
        """run_completed event should show completion banner."""
        async with app.run_test() as pilot:
            app._handle_event({
                "event": "run_completed",
                "best_metric": 0.95,
                "total_experiments": 10,
                "total_cost_usd": 4.20,
            })
            log = app.query_one("#event-log", RichLog)
            assert len(log.lines) > 0
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_tui_app.py -v`
Expected: All PASS

- [ ] **Step 3: Fix any issues, then commit**

```bash
git add tests/test_tui_app.py
git commit -m "test(tui): add event handling tests for ChaosApp widgets"
```

---

## Chunk 4: ViewManager and CLI Integration

### Task 10: ViewManager — toggle between log mode and TUI mode

**Files:**
- Create: `chaosengineer/tui/views.py`
- Create: `tests/test_view_manager.py`

- [ ] **Step 1: Write failing tests for ViewManager**

Create `tests/test_view_manager.py`:

```python
"""Tests for ViewManager — toggle between log mode and TUI mode."""
from unittest.mock import MagicMock, patch
import threading

import pytest

from chaosengineer.tui.views import ViewManager
from chaosengineer.tui.bridge import EventBridge
from chaosengineer.tui.pause_gate import PauseGate


class TestViewManagerState:
    def test_starts_in_log_mode(self):
        vm = ViewManager(
            bridge=EventBridge(),
            pause_gate=PauseGate(),
            pause_controller=MagicMock(),
            coordinator=MagicMock(),
            status_display=MagicMock(),
        )
        assert not vm.tui_active

    def test_tui_active_flag_set_during_tui(self):
        vm = ViewManager(
            bridge=EventBridge(),
            pause_gate=PauseGate(),
            pause_controller=MagicMock(),
            coordinator=MagicMock(),
            status_display=MagicMock(),
        )
        assert not vm.tui_active


class TestViewManagerCoordinatorDone:
    def test_stops_when_coordinator_thread_finishes(self):
        """ViewManager.run() should exit when coord_done event is set."""
        vm = ViewManager(
            bridge=EventBridge(),
            pause_gate=PauseGate(),
            pause_controller=MagicMock(),
            coordinator=MagicMock(),
            status_display=MagicMock(),
        )
        vm._coord_done = threading.Event()
        vm._coord_done.set()
        # run() should return quickly since coordinator is done
        # (We can't test the full stdin loop without a terminal)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_view_manager.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement ViewManager**

Create `chaosengineer/tui/views.py`:

```python
"""ViewManager — toggle between log mode and TUI mode."""
from __future__ import annotations

import sys
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chaosengineer.core.coordinator import Coordinator
    from chaosengineer.core.pause import PauseController
    from chaosengineer.core.status import StatusDisplay
    from chaosengineer.tui.bridge import EventBridge
    from chaosengineer.tui.pause_gate import PauseGate


class ViewManager:
    """Manages toggling between log mode (stderr) and TUI mode (Textual)."""

    DEBOUNCE_MS = 500

    def __init__(
        self,
        bridge: "EventBridge",
        pause_gate: "PauseGate",
        pause_controller: "PauseController",
        coordinator: "Coordinator",
        status_display: "StatusDisplay",
    ):
        self._bridge = bridge
        self._pause_gate = pause_gate
        self._pause_controller = pause_controller
        self._coordinator = coordinator
        self._status_display = status_display
        self.tui_active: bool = False
        self._coord_done = threading.Event()
        self._last_toggle: float = 0

    def run(self, coord_done: threading.Event) -> None:
        """Main loop. Runs on the main thread. Blocks until coordinator finishes."""
        self._coord_done = coord_done
        print("Press 't' to open TUI dashboard", file=sys.stderr)

        while not self._coord_done.is_set():
            if self._coord_done.wait(timeout=0.2):
                break
            if self._check_stdin_for_toggle():
                self._enter_tui()

    def _check_stdin_for_toggle(self) -> bool:
        """Check if 't' was pressed on stdin. Uses cbreak mode."""
        if not sys.stdin.isatty():
            return False
        try:
            import select
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if ready:
                ch = sys.stdin.read(1)
                if ch == "t" and self._debounce_ok():
                    return True
        except Exception:
            pass
        return False

    def _debounce_ok(self) -> bool:
        now = time.monotonic() * 1000
        if now - self._last_toggle < self.DEBOUNCE_MS:
            return False
        self._last_toggle = now
        return True

    def _enter_tui(self) -> None:
        """Switch to TUI mode."""
        from chaosengineer.tui.app import ChaosApp

        self.tui_active = True
        self._status_display.suppressed = True

        app = ChaosApp(
            bridge=self._bridge,
            pause_gate=self._pause_gate,
            coordinator=self._coordinator,
            pause_controller=self._pause_controller,
        )
        try:
            app.run()
        except Exception as e:
            print(f"\nTUI error: {e}. Falling back to log mode.", file=sys.stderr)
        finally:
            self.tui_active = False
            self._status_display.suppressed = False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_view_manager.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/tui/views.py tests/test_view_manager.py
git commit -m "feat(tui): add ViewManager for log/TUI mode toggle"
```

---

### Task 11: CLI integration — `--tui` flag and wiring

**Files:**
- Modify: `chaosengineer/cli.py`
- Modify: `tests/test_cli_run.py`

- [ ] **Step 1: Write failing test for --tui flag**

Add to `tests/test_cli_run.py`:

```python
class TestTuiFlag:
    def test_run_parser_accepts_tui_flag(self):
        from chaosengineer.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["run", "workload.md", "--tui"])
        assert args.tui is True

    def test_run_parser_defaults_tui_false(self):
        from chaosengineer.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["run", "workload.md"])
        assert args.tui is False

    def test_resume_parser_accepts_tui_flag(self):
        from chaosengineer.cli import _build_parser
        parser = _build_parser()
        args = parser.parse_args(["resume", "output/", "workload.md", "--tui"])
        assert args.tui is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli_run.py::TestTuiFlag -v`
Expected: FAIL — `error: unrecognized arguments: --tui`

- [ ] **Step 3: Add --tui flag to parser**

In `chaosengineer/cli.py`, in `_build_parser()`:

After the `run_parser` arguments (after `--force-fresh`, around line 54), add:

```python
run_parser.add_argument("--tui", action="store_true", default=False,
                        help="Enable TUI dashboard (toggle with 't' during run)")
```

After the `resume_parser` arguments (after `--restart-iteration`, around line 74), add:

```python
resume_parser.add_argument("--tui", action="store_true", default=False,
                           help="Enable TUI dashboard (toggle with 't' during run)")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli_run.py::TestTuiFlag -v`
Expected: All PASS

- [ ] **Step 5: Wire TUI into _execute_run**

In `_execute_run()`, after `status_display = StatusDisplay()` (line 332) and before the `coordinator = Coordinator(...)` call:

Add imports at the top of the function block:

```python
view_manager = None
if getattr(args, "tui", False):
    from chaosengineer.tui.bridge import EventBridge
    from chaosengineer.tui.pause_gate import PauseGate
    from chaosengineer.tui.views import ViewManager

    bridge = EventBridge()
    pause_gate = PauseGate()
    # Re-create publisher with bridge
    logger = EventPublisher(bus_url=bus_url, fallback_path=log_path, bridge=bridge)
```

Modify the `Coordinator(...)` constructor call to include the new optional parameters:

```python
coordinator = Coordinator(
    spec=spec,
    decision_maker=dm,
    executor=executor,
    logger=logger,
    budget=budget,
    initial_baseline=initial_baseline,
    run_id=run_id,
    pause_controller=pause_controller,
    status_display=status_display,
    view_manager=view_manager if getattr(args, "tui", False) else None,
    pause_gate=pause_gate if getattr(args, "tui", False) else None,
)
```

Replace the `coordinator.run()` call and surrounding try/finally (lines 352-358) with:

```python
pause_controller.install()
try:
    if getattr(args, "tui", False):
        view_manager = ViewManager(bridge, pause_gate, pause_controller,
                                    coordinator, status_display)
        coord_done = threading.Event()

        def run_coordinator():
            try:
                coordinator.run()
            finally:
                coord_done.set()

        coord_thread = threading.Thread(target=run_coordinator, daemon=True)
        coord_thread.start()
        view_manager.run(coord_done)
        coord_thread.join()
    else:
        coordinator.run()
finally:
    pause_controller.uninstall()
    if bus_proc:
        bus_proc.terminate()
```

Add `import threading` to the imports at the top of the function if not already present.

- [ ] **Step 6: Apply same wiring to _execute_resume**

Mirror the same pattern in `_execute_resume()`:
- After creating `logger`, check `args.tui` and create bridge/pause_gate/re-create publisher with bridge
- Pass `view_manager` and `pause_gate` to Coordinator
- Replace `coordinator.resume_from_snapshot()` call with threaded version when `--tui` is set

- [ ] **Step 7: Run full test suite**

Run: `uv run pytest tests/ -v --timeout=60`
Expected: All existing tests pass, new tests pass

- [ ] **Step 8: Commit**

```bash
git add chaosengineer/cli.py tests/test_cli_run.py
git commit -m "feat(tui): wire --tui flag into CLI run and resume commands"
```

---

## Chunk 5: Integration test and polish

### Task 12: Integration test — coordinator + bridge + mock TUI consumer

**Files:**
- Create: `tests/test_tui_integration.py`

- [ ] **Step 1: Write integration test**

Create `tests/test_tui_integration.py`:

```python
"""Integration tests: coordinator thread + EventBridge + consumer."""
import asyncio
import threading
import time

import pytest

from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.interfaces import DimensionPlan
from chaosengineer.core.models import Baseline, BudgetConfig, ExperimentResult
from chaosengineer.core.pause import PauseController
from chaosengineer.core.status import StatusDisplay
from chaosengineer.metrics.publisher import EventPublisher
from chaosengineer.testing.executor import ScriptedExecutor
from chaosengineer.testing.simulator import ScriptedDecisionMaker
from chaosengineer.tui.bridge import EventBridge
from chaosengineer.tui.pause_gate import PauseGate
from chaosengineer.workloads.parser import WorkloadSpec


def _make_spec(budget=None):
    return WorkloadSpec(
        name="test", primary_metric="loss", metric_direction="lower",
        execution_command="echo 1", workers_available=2,
        budget=budget or BudgetConfig(max_experiments=4),
    )


class TestCoordinatorBridgeIntegration:
    def test_events_flow_through_bridge(self, tmp_path):
        """Coordinator -> EventPublisher -> EventBridge -> consumer queue."""
        spec = _make_spec()
        plans = [
            DimensionPlan("lr", [{"lr": 0.01}, {"lr": 0.1}]),
            DimensionPlan("batch", [{"batch": 32}, {"batch": 64}]),
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=2.5, cost_usd=0.5),
            "exp-0-1": ExperimentResult(primary_metric=2.0, cost_usd=0.5),
            "exp-1-0": ExperimentResult(primary_metric=1.8, cost_usd=0.5),
            "exp-1-1": ExperimentResult(primary_metric=1.5, cost_usd=0.5),
        }

        bridge = EventBridge()
        publisher = EventPublisher(
            bus_url=None, fallback_path=tmp_path / "events.jsonl", bridge=bridge,
        )

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=publisher,
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("HEAD", 3.0, "loss"),
            status_display=StatusDisplay(),
        )

        coordinator.run()

        events = bridge.snapshot()
        event_types = [e["event"] for e in events]
        assert "run_started" in event_types
        assert "iteration_started" in event_types
        assert "worker_completed" in event_types

    def test_pause_via_gate_from_thread(self, tmp_path):
        """Coordinator in thread, consumer submits pause via PauseGate."""
        spec = _make_spec(BudgetConfig(max_experiments=20))
        plans = [DimensionPlan("lr", [{"lr": v} for v in [0.01, 0.1, 0.001, 0.5]])] * 5
        results = {
            f"exp-{i}-{j}": ExperimentResult(primary_metric=2.0 - i * 0.1 - j * 0.01, cost_usd=0.1)
            for i in range(5) for j in range(4)
        }

        bridge = EventBridge()
        gate = PauseGate()
        pc = PauseController()
        pc.pause_requested = True

        view_manager_mock = type("VM", (), {"tui_active": True})()

        publisher = EventPublisher(
            bus_url=None, fallback_path=tmp_path / "events.jsonl", bridge=bridge,
        )

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=publisher,
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("HEAD", 3.0, "loss"),
            pause_controller=pc,
            view_manager=view_manager_mock,
            pause_gate=gate,
        )

        coord_done = threading.Event()

        def run_coord():
            coordinator.run()
            coord_done.set()

        t = threading.Thread(target=run_coord)
        t.start()

        # Wait for pause decision request
        gate.decision_needed.wait(timeout=10.0)
        assert gate.decision_needed.is_set()

        gate.submit_decision("pause")
        t.join(timeout=10.0)

        events = bridge.snapshot()
        event_types = [e["event"] for e in events]
        assert "run_paused" in event_types

    def test_extend_budget_from_consumer(self, tmp_path):
        """Consumer extends budget via coordinator.extend_budget()."""
        spec = _make_spec(BudgetConfig(max_api_cost=1.0, max_experiments=2))
        plans = [DimensionPlan("lr", [{"lr": 0.01}, {"lr": 0.1}])]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=2.5, cost_usd=0.3),
            "exp-0-1": ExperimentResult(primary_metric=2.0, cost_usd=0.3),
        }

        bridge = EventBridge()
        publisher = EventPublisher(
            bus_url=None, fallback_path=tmp_path / "events.jsonl", bridge=bridge,
        )

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=publisher,
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("HEAD", 3.0, "loss"),
        )

        # Extend budget before run
        coordinator.extend_budget(add_cost=5.0, add_experiments=10)
        assert coordinator.budget.config.max_api_cost == 6.0
        assert coordinator.budget.config.max_experiments == 12
```

- [ ] **Step 2: Run integration tests**

Run: `uv run pytest tests/test_tui_integration.py -v`
Expected: All PASS

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest tests/ -v --timeout=60`
Expected: All tests pass (existing + new TUI tests)

- [ ] **Step 4: Commit**

```bash
git add tests/test_tui_integration.py
git commit -m "test(tui): add integration tests for coordinator + bridge + pause gate"
```

---

### Task 13: Manual smoke test

- [ ] **Step 1: Build and verify**

Run: `uv run chaosengineer run --help`
Expected: Shows `--tui` flag in help output

- [ ] **Step 2: Run with TUI using scripted executor**

Run a scripted test scenario with `--tui`:

```bash
uv run chaosengineer run workloads/example.md --executor scripted --llm-backend scripted --scripted-results testing/scenarios/... --scripted-plans testing/scenarios/... --tui
```

Verify:
- Log mode shows "Press 't' to open TUI dashboard"
- Pressing `t` opens the TUI with budget bar, experiment table, event log
- Pressing `q` returns to log mode
- Pressing `p` in TUI triggers pause

- [ ] **Step 3: Commit any fixes**

```bash
git add -u
git commit -m "fix(tui): polish from manual smoke test"
```
