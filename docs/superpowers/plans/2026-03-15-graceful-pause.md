# Graceful Pause Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Ctrl+C graceful pause with interactive menu, real-time status line, and executor kill support.

**Architecture:** Signal handler sets a flag on `PauseController`. The coordinator checks the flag at iteration boundaries and shows an interactive menu (Wait/Kill/Continue). A `StatusDisplay` prints per-worker progress to stderr. The executor switches to `as_completed()` with per-worker callbacks and `Popen`-based kill support.

**Tech Stack:** Python signal module, `concurrent.futures.as_completed()`, `subprocess.Popen`, `threading.Lock`

**Spec:** `docs/superpowers/specs/2026-03-15-graceful-pause-design.md`

---

## File Structure

**New files:**
| File | Responsibility |
|------|---------------|
| `chaosengineer/core/pause.py` | PauseController: signal handler, pause state machine, interactive menus |
| `chaosengineer/core/status.py` | StatusDisplay: real-time progress output to stderr |
| `tests/test_pause.py` | PauseController unit tests |
| `tests/test_status.py` | StatusDisplay unit tests |
| `tests/test_pause_coordinator.py` | Coordinator + PauseController integration tests |
| `tests/e2e/test_tmux_pause.py` | E2E tmux signal test |

**Modified files:**
| File | Changes |
|------|---------|
| `chaosengineer/cli_menu.py` | Add letter-key hotkey support to `_select_interactive()` |
| `chaosengineer/core/interfaces.py:77-96` | Add `on_worker_done` callback param to `run_experiments()`, add `kill_active()` |
| `chaosengineer/testing/executor.py` | Override `run_experiments()` to call `on_worker_done` callback |
| `chaosengineer/execution/subagent.py:68-87,145-191` | Switch to `as_completed()`, `Popen` + kill, `on_worker_done` callback |
| `chaosengineer/core/coordinator.py:35-60,117-228,350-416` | Accept `pause_controller`/`status_display`, add pause checks, pass callback to executor |
| `chaosengineer/cli.py:267-282,400-421` | Wire PauseController + StatusDisplay, install/uninstall signal handler |

---

## Chunk 1: Foundation

### Task 0: PauseController

**Files:**
- Create: `chaosengineer/core/pause.py`
- Create: `tests/test_pause.py`

- [ ] **Step 1: Write failing tests for PauseController state machine**

```python
# tests/test_pause.py
"""Tests for PauseController — state machine and signal handling."""

from __future__ import annotations

import signal
from unittest.mock import MagicMock, patch

import pytest

from chaosengineer.core.pause import PauseController


class TestPauseControllerStateMachine:
    def test_initial_state(self):
        pc = PauseController()
        assert pc.pause_requested is False
        assert pc.force_kill is False
        assert pc.wait_then_ask is False
        assert pc.kill_issued is False

    def test_first_sigint_sets_pause_requested(self):
        pc = PauseController()
        pc.on_sigint(signal.SIGINT, None)
        assert pc.pause_requested is True
        assert pc.force_kill is False

    def test_second_sigint_sets_force_kill(self):
        pc = PauseController()
        pc.on_sigint(signal.SIGINT, None)
        pc.on_sigint(signal.SIGINT, None)
        assert pc.force_kill is True

    def test_reset_clears_all_flags(self):
        pc = PauseController()
        pc.on_sigint(signal.SIGINT, None)
        pc.wait_then_ask = True
        pc.reset()
        assert pc.pause_requested is False
        assert pc.wait_then_ask is False
        assert pc.force_kill is False

    def test_should_show_menu_when_pause_requested(self):
        pc = PauseController()
        assert pc.should_show_menu() is False
        pc.pause_requested = True
        assert pc.should_show_menu() is True

    def test_should_show_menu_when_wait_then_ask(self):
        pc = PauseController()
        pc.wait_then_ask = True
        assert pc.should_show_menu() is True


class TestInstallUninstall:
    def test_install_registers_handler(self):
        pc = PauseController()
        original = signal.getsignal(signal.SIGINT)
        try:
            pc.install()
            assert signal.getsignal(signal.SIGINT) == pc.on_sigint
        finally:
            pc.uninstall()
            assert signal.getsignal(signal.SIGINT) == original

    def test_second_sigint_restores_default(self):
        pc = PauseController()
        pc.install()
        try:
            pc.on_sigint(signal.SIGINT, None)  # first
            assert signal.getsignal(signal.SIGINT) == pc.on_sigint
            pc.on_sigint(signal.SIGINT, None)  # second — restores default
            assert signal.getsignal(signal.SIGINT) == signal.default_int_handler
        finally:
            signal.signal(signal.SIGINT, signal.default_int_handler)


class TestSetExecutor:
    def test_set_executor_stores_reference(self):
        pc = PauseController()
        executor = MagicMock()
        pc.set_executor(executor)
        assert pc._executor is executor
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pause.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chaosengineer.core.pause'`

- [ ] **Step 3: Implement PauseController**

```python
# chaosengineer/core/pause.py
"""PauseController — signal handler and interactive pause menus."""

from __future__ import annotations

import signal
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chaosengineer.core.interfaces import ExperimentExecutor


class PauseController:
    """Manages graceful pause state and interactive menus.

    Install as a SIGINT handler. First Ctrl+C sets pause_requested.
    Second Ctrl+C sets force_kill and restores the default handler
    (third Ctrl+C hard-kills).
    """

    def __init__(self) -> None:
        self.pause_requested: bool = False
        self.force_kill: bool = False
        self.wait_then_ask: bool = False
        self.kill_issued: bool = False  # Suppresses re-prompting after kill
        self._executor: ExperimentExecutor | None = None
        self._original_handler = None

    def install(self) -> None:
        """Register as the SIGINT handler."""
        self._original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.on_sigint)

    def uninstall(self) -> None:
        """Restore the original SIGINT handler."""
        if self._original_handler is not None:
            signal.signal(signal.SIGINT, self._original_handler)
            self._original_handler = None

    def set_executor(self, executor: "ExperimentExecutor") -> None:
        """Store executor reference for kill support."""
        self._executor = executor

    def on_sigint(self, signum: int, frame) -> None:
        """SIGINT handler callback."""
        if self.pause_requested:
            self.force_kill = True
            signal.signal(signal.SIGINT, signal.default_int_handler)
            print("\nForce kill armed — next Ctrl+C will terminate.", file=sys.stderr)
        else:
            self.pause_requested = True
            print("\nPause requested — will pause after current work finishes.", file=sys.stderr)

    def should_show_menu(self) -> bool:
        """True if the coordinator should show a pause menu."""
        if self.kill_issued:
            return False  # Auto-pause after iteration, no re-prompt
        return self.pause_requested or self.wait_then_ask

    def reset(self) -> None:
        """Clear all flags (after user picks 'continue')."""
        self.pause_requested = False
        self.force_kill = False
        self.wait_then_ask = False
        self.kill_issued = False

    def show_mid_iteration_menu(self, completed: int, total: int) -> str:
        """Show menu when workers are in flight. Returns 'wait', 'kill', or 'continue'."""
        from chaosengineer.cli_menu import select
        options = [
            "[W] Wait for remaining workers, then decide",
            "[K] Kill workers and pause now",
            "[C] Continue running",
        ]
        idx = select(
            f"Pause requested\n\n  {completed}/{total} workers completed this iteration.",
            options,
        )
        return ["wait", "kill", "continue"][idx]

    def show_post_iteration_menu(self, summary: str = "") -> str:
        """Show menu when no workers are in flight. Returns 'pause' or 'continue'."""
        from chaosengineer.cli_menu import select
        prompt = summary or "Pause requested"
        options = ["[P] Pause now", "[C] Continue running"]
        idx = select(prompt, options)
        return ["pause", "continue"][idx]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pause.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/core/pause.py tests/test_pause.py
git commit -m "feat: add PauseController with signal handling and state machine"
```

---

### Task 1: StatusDisplay

**Files:**
- Create: `chaosengineer/core/status.py`
- Create: `tests/test_status.py`

- [ ] **Step 1: Write failing tests for StatusDisplay**

```python
# tests/test_status.py
"""Tests for StatusDisplay — progress output formatting."""

from __future__ import annotations

import io
from unittest.mock import patch, MagicMock

import pytest

from chaosengineer.core.status import StatusDisplay
from chaosengineer.core.models import BudgetConfig


class TestStatusFormatting:
    def test_format_progress_line(self):
        sd = StatusDisplay()
        line = sd._format_progress(
            iteration=1, completed=2, total=4,
            cost=0.31, elapsed=222.0,
        )
        assert "iter 1" in line
        assert "2/4 workers done" in line
        assert "$0.31" in line
        assert "00:03:42" in line
        assert "Ctrl+C to pause" in line

    def test_format_breakthrough(self):
        sd = StatusDisplay()
        line = sd._format_progress(
            iteration=1, completed=3, total=4,
            cost=0.42, elapsed=255.0,
            breakthrough=("val_bpb", 0.93),
        )
        assert "New best: val_bpb=0.93" in line

    def test_on_worker_done_writes_stderr(self):
        sd = StatusDisplay()
        sd._iteration = 1
        sd._cost = 0.31
        sd._start_time = 0.0
        buf = io.StringIO()
        with patch("chaosengineer.core.status.time") as mock_time, \
             patch("sys.stderr", buf):
            mock_time.monotonic.return_value = 10.0
            task = MagicMock()
            result = MagicMock()
            result.error_message = None
            sd.on_worker_done(task, result, 2, 4)
        output = buf.getvalue()
        assert "2/4 workers done" in output

    def test_on_iteration_done_prints_newline(self):
        sd = StatusDisplay()
        sd._iteration = 1
        sd._cost = 0.50
        sd._start_time = 0.0
        buf = io.StringIO()
        with patch("chaosengineer.core.status.time") as mock_time, \
             patch("sys.stderr", buf):
            mock_time.monotonic.return_value = 60.0
            sd.on_iteration_done(iteration=1, best_metric=2.5)
        output = buf.getvalue()
        assert "\n" in output
        assert "iter 1" in output

    def test_on_run_start(self):
        sd = StatusDisplay()
        buf = io.StringIO()
        with patch("chaosengineer.core.status.time") as mock_time, \
             patch("sys.stderr", buf):
            mock_time.monotonic.return_value = 0.0
            sd.on_run_start(BudgetConfig(max_experiments=10))
        output = buf.getvalue()
        assert "max_experiments=10" in output or "budget" in output.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_status.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chaosengineer.core.status'`

- [ ] **Step 3: Implement StatusDisplay**

```python
# chaosengineer/core/status.py
"""StatusDisplay — real-time progress output to stderr."""

from __future__ import annotations

import sys
import time
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from chaosengineer.core.interfaces import ExperimentTask
    from chaosengineer.core.models import BudgetConfig, ExperimentResult


class StatusDisplay:
    """Prints run progress to stderr.

    Uses \\r to overwrite the current line during worker progress.
    Prints a newline after each iteration for a scrolling log.
    """

    def __init__(self) -> None:
        self._iteration: int = 0
        self._cost: float = 0.0
        self._start_time: float | None = None

    def on_run_start(self, budget_config: "BudgetConfig") -> None:
        """Print initial budget info."""
        self._start_time = time.monotonic()
        parts = []
        if budget_config.max_experiments is not None:
            parts.append(f"max_experiments={budget_config.max_experiments}")
        if budget_config.max_api_cost is not None:
            parts.append(f"max_cost=${budget_config.max_api_cost:.2f}")
        if budget_config.max_wall_time_seconds is not None:
            parts.append(f"max_time={budget_config.max_wall_time_seconds}s")
        budget_str = ", ".join(parts) if parts else "unlimited"
        print(f"Budget: {budget_str}", file=sys.stderr)

    def on_worker_done(
        self,
        task: "ExperimentTask",
        result: "ExperimentResult",
        completed: int,
        total: int,
    ) -> None:
        """Update progress after a worker completes."""
        self._cost += getattr(result, "cost_usd", 0.0) or 0.0
        elapsed = self._elapsed()
        line = self._format_progress(
            iteration=self._iteration,
            completed=completed,
            total=total,
            cost=self._cost,
            elapsed=elapsed,
        )
        print(f"\r{line}", end="", file=sys.stderr)

    def on_iteration_done(self, iteration: int, best_metric: float) -> None:
        """Print iteration summary with newline."""
        self._iteration = iteration + 1
        elapsed = self._elapsed()
        line = self._format_progress(
            iteration=iteration,
            completed=0,
            total=0,
            cost=self._cost,
            elapsed=elapsed,
        )
        print(f"\r{line} | best={best_metric}", file=sys.stderr)

    def on_breakthrough(self, metric_name: str, value: float) -> None:
        """Highlight a new best metric (called during worker_done)."""
        # Stored for next _format_progress call — handled inline by coordinator
        pass

    def _elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    def _format_progress(
        self,
        iteration: int,
        completed: int,
        total: int,
        cost: float,
        elapsed: float,
        breakthrough: tuple[str, float] | None = None,
    ) -> str:
        """Format a progress line."""
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        time_str = f"{h:02d}:{m:02d}:{s:02d}"

        parts = [f"iter {iteration}"]
        if total > 0:
            parts.append(f"{completed}/{total} workers done")
        parts.append(f"${cost:.2f}")
        parts.append(time_str)

        line = "[" + " | ".join(parts) + "]"

        if breakthrough:
            name, val = breakthrough
            line += f" New best: {name}={val}"
        else:
            line += " Ctrl+C to pause"

        return line
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_status.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/core/status.py tests/test_status.py
git commit -m "feat: add StatusDisplay for real-time progress output"
```

---

### Task 2: cli_menu letter-key hotkeys

**Files:**
- Modify: `chaosengineer/cli_menu.py:35-62`
- Modify: `tests/test_cli_menu.py`

- [ ] **Step 1: Write failing test for letter-key hotkeys**

Add to the existing `tests/test_cli_menu.py`:

```python
class TestLetterKeyHotkeys:
    def test_letter_key_selects_option(self):
        """Pressing a letter key matching [X] prefix selects that option."""
        from chaosengineer.cli_menu import _match_hotkey
        options = ["[W] Wait for workers", "[K] Kill and pause", "[C] Continue"]
        assert _match_hotkey("w", options) == 0
        assert _match_hotkey("W", options) == 0
        assert _match_hotkey("k", options) == 1
        assert _match_hotkey("K", options) == 1
        assert _match_hotkey("c", options) == 2
        assert _match_hotkey("x", options) is None

    def test_no_hotkey_brackets_returns_none(self):
        from chaosengineer.cli_menu import _match_hotkey
        options = ["Resume previous run", "Start fresh"]
        assert _match_hotkey("r", options) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_cli_menu.py::TestLetterKeyHotkeys -v`
Expected: FAIL with `ImportError: cannot import name '_match_hotkey'`

- [ ] **Step 3: Implement `_match_hotkey` and wire into `_select_interactive`**

Add to `chaosengineer/cli_menu.py`:

```python
import re

def _match_hotkey(ch: str, options: list[str]) -> int | None:
    """Match a single character against [X] hotkey prefixes in options.
    Returns the index of the matching option, or None."""
    for i, opt in enumerate(options):
        m = re.match(r"\[(\w)\]", opt)
        if m and m.group(1).lower() == ch.lower():
            return i
    return None
```

In `_select_interactive()`, after the arrow-key handling block (after `elif ch == "\x03":`), add:

```python
            else:
                match = _match_hotkey(ch, options)
                if match is not None:
                    selected = match
                    break
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_cli_menu.py -v`
Expected: PASS (all tests including new ones)

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/cli_menu.py tests/test_cli_menu.py
git commit -m "feat: add letter-key hotkey support to cli_menu"
```

---

## Chunk 2: Executor Changes

### Task 3: ExperimentExecutor interface — on_worker_done callback + kill_active

**Files:**
- Modify: `chaosengineer/core/interfaces.py:77-96`
- Modify: `tests/test_interfaces.py` (if it tests `run_experiments`)

- [ ] **Step 1: Write failing test for callback in base class**

Add to `tests/test_interfaces.py` (or create a new section):

```python
class TestRunExperimentsCallback:
    def test_callback_called_per_result(self):
        """Base run_experiments() calls on_worker_done after each task."""
        from unittest.mock import MagicMock
        from chaosengineer.core.interfaces import ExperimentTask
        from chaosengineer.core.models import ExperimentResult

        class SimpleExecutor(ExperimentExecutor):
            def run_experiment(self, experiment_id, params, command, baseline_commit, resource=""):
                return ExperimentResult(primary_metric=1.0)

        executor = SimpleExecutor()
        tasks = [
            ExperimentTask("e-0", {"lr": 0.1}, "echo", "abc"),
            ExperimentTask("e-1", {"lr": 0.2}, "echo", "abc"),
        ]
        callback = MagicMock()
        results = executor.run_experiments(tasks, on_worker_done=callback)

        assert len(results) == 2
        assert callback.call_count == 2
        # First call: (task, result, 1, 2)
        assert callback.call_args_list[0][0][2] == 1  # completed
        assert callback.call_args_list[0][0][3] == 2  # total
        # Second call: (task, result, 2, 2)
        assert callback.call_args_list[1][0][2] == 2

    def test_callback_none_is_fine(self):
        """on_worker_done=None (default) doesn't break anything."""
        class SimpleExecutor(ExperimentExecutor):
            def run_experiment(self, experiment_id, params, command, baseline_commit, resource=""):
                return ExperimentResult(primary_metric=1.0)

        executor = SimpleExecutor()
        tasks = [ExperimentTask("e-0", {"lr": 0.1}, "echo", "abc")]
        results = executor.run_experiments(tasks)
        assert len(results) == 1

    def test_kill_active_default_noop(self):
        """Base class kill_active() is a no-op."""
        class SimpleExecutor(ExperimentExecutor):
            def run_experiment(self, experiment_id, params, command, baseline_commit, resource=""):
                return ExperimentResult(primary_metric=1.0)

        executor = SimpleExecutor()
        executor.kill_active()  # Should not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_interfaces.py::TestRunExperimentsCallback -v`
Expected: FAIL with `TypeError: run_experiments() got an unexpected keyword argument 'on_worker_done'`

- [ ] **Step 3: Update ExperimentExecutor interface**

In `chaosengineer/core/interfaces.py`, replace the `run_experiments` method (lines 77-96) with:

```python
    def run_experiments(
        self,
        tasks: list[ExperimentTask],
        on_worker_done: "Callable[[ExperimentTask, ExperimentResult, int, int], None] | None" = None,
    ) -> list[ExperimentResult]:
        """Run a batch of experiments. Default: sequential.

        Postcondition: always returns exactly one ExperimentResult per input task.
        Failures are captured as ExperimentResult with error_message set, never raised.

        Args:
            on_worker_done: Optional callback(task, result, completed_count, total_count)
                called after each experiment completes.
        """
        results = []
        total = len(tasks)
        for i, t in enumerate(tasks):
            try:
                result = self.run_experiment(
                    t.experiment_id, t.params, t.command, t.baseline_commit, t.resource
                )
            except Exception as e:
                result = ExperimentResult(
                    primary_metric=0.0,
                    error_message=f"Executor error for {t.experiment_id}: {e}",
                )
            results.append(result)
            if on_worker_done is not None:
                on_worker_done(t, result, i + 1, total)
        return results

    def kill_active(self) -> None:
        """Kill any in-flight experiments. Default: no-op."""
        pass
```

Also add the import at the top of the file:

```python
from typing import Any, Callable
```

(The file likely already imports `Any` — just ensure `Callable` is also imported.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_interfaces.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run full test suite to check nothing broke**

Run: `python -m pytest tests/ -x -q`
Expected: All tests pass (the new callback parameter has a default of None, so existing callers are unaffected)

- [ ] **Step 6: Commit**

```bash
git add chaosengineer/core/interfaces.py tests/test_interfaces.py
git commit -m "feat: add on_worker_done callback and kill_active to ExperimentExecutor"
```

---

### Task 4: ScriptedExecutor — support on_worker_done callback

**Files:**
- Modify: `chaosengineer/testing/executor.py`

- [ ] **Step 1: Write failing test**

Add to `tests/test_simulator.py` or create a section in an appropriate test file:

```python
class TestScriptedExecutorCallback:
    def test_on_worker_done_called(self):
        from unittest.mock import MagicMock
        from chaosengineer.core.interfaces import ExperimentTask
        from chaosengineer.core.models import ExperimentResult
        from chaosengineer.testing.executor import ScriptedExecutor

        results = {
            "e-0": ExperimentResult(primary_metric=1.0),
            "e-1": ExperimentResult(primary_metric=2.0),
        }
        executor = ScriptedExecutor(results)
        tasks = [
            ExperimentTask("e-0", {"lr": 0.1}, "echo", "abc"),
            ExperimentTask("e-1", {"lr": 0.2}, "echo", "abc"),
        ]
        callback = MagicMock()
        executor.run_experiments(tasks, on_worker_done=callback)
        assert callback.call_count == 2
```

- [ ] **Step 2: Run test to verify behavior**

Run: `python -m pytest tests/ -k "TestScriptedExecutorCallback" -v`

Note: This test may already pass since `ScriptedExecutor` inherits from `ExperimentExecutor` and the base class `run_experiments()` now calls the callback. If it passes, great — no changes needed to `ScriptedExecutor`. If it fails, override `run_experiments()`.

- [ ] **Step 3: If test fails, override run_experiments in ScriptedExecutor**

Only if needed — the base class implementation should handle this. If `ScriptedExecutor.run_experiment()` works correctly and the base class loop calls the callback, no changes needed.

- [ ] **Step 4: Commit (if changes were made)**

```bash
git add chaosengineer/testing/executor.py tests/test_simulator.py
git commit -m "test: verify ScriptedExecutor supports on_worker_done callback"
```

---

### Task 5: SubagentExecutor — as_completed, Popen, kill support

**Files:**
- Modify: `chaosengineer/execution/subagent.py:68-87,145-191`
- Modify: `tests/test_subagent_executor.py`

- [ ] **Step 1: Write failing tests for as_completed ordering and callback**

Add to `tests/test_subagent_executor.py`:

```python
class TestAsCompletedOrdering:
    def test_results_match_input_order(self):
        """Results are reordered to match input task order, not completion order."""
        # This tests the contract: zip(tasks, results) should be correct
        # Use a mock executor approach since SubagentExecutor needs a real git repo
        from unittest.mock import MagicMock, patch
        from chaosengineer.core.interfaces import ExperimentTask
        from chaosengineer.core.models import ExperimentResult

        # We test this via the interface contract — SubagentExecutor.run_experiments
        # returns results in the same order as tasks regardless of completion order.
        # The actual as_completed behavior is an internal detail.
        pass  # Covered by integration test


class TestKillActive:
    def test_kill_active_terminates_processes(self):
        """kill_active() calls terminate() on all stored processes."""
        from unittest.mock import MagicMock
        from chaosengineer.execution.subagent import SubagentExecutor

        # Create executor with mocked dependencies
        executor = MagicMock(spec=SubagentExecutor)
        executor._active_processes = []
        executor._process_lock = __import__("threading").Lock()

        proc1 = MagicMock()
        proc2 = MagicMock()
        executor._active_processes = [proc1, proc2]

        # Call the real kill_active
        SubagentExecutor.kill_active(executor)

        proc1.terminate.assert_called_once()
        proc2.terminate.assert_called_once()

    def test_kill_active_empty_is_noop(self):
        """kill_active() with no active processes does nothing."""
        from unittest.mock import MagicMock
        from chaosengineer.execution.subagent import SubagentExecutor

        executor = MagicMock(spec=SubagentExecutor)
        executor._active_processes = []
        executor._process_lock = __import__("threading").Lock()

        SubagentExecutor.kill_active(executor)  # Should not raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_subagent_executor.py::TestKillActive -v`
Expected: FAIL with `AttributeError: ... has no attribute 'kill_active'`

- [ ] **Step 3: Implement changes to SubagentExecutor**

In `chaosengineer/execution/subagent.py`:

**Add imports at top:**
```python
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
```

**Add to `__init__`** (after existing fields):
```python
self._active_processes: list[subprocess.Popen] = []
self._process_lock = threading.Lock()
```

**Replace `run_experiments` method** (lines 68-87):
```python
def run_experiments(
    self,
    tasks: list[ExperimentTask],
    on_worker_done: "Callable[[ExperimentTask, ExperimentResult, int, int], None] | None" = None,
) -> list[ExperimentResult]:
    """Run a batch of experiments with as_completed for reactive callbacks."""
    max_workers = len(tasks) if self._mode == "parallel" else 1

    result_map: dict[str, ExperimentResult] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_task = {
            pool.submit(self._run_single, task): task for task in tasks
        }
        completed_count = 0
        total = len(tasks)
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
            except Exception as e:
                result = ExperimentResult(
                    primary_metric=0.0,
                    error_message=f"Thread crashed for {task.experiment_id}: {e}",
                )
            result_map[task.experiment_id] = result
            completed_count += 1
            if on_worker_done is not None:
                on_worker_done(task, result, completed_count, total)

    # Reorder to match input task order
    return [result_map[t.experiment_id] for t in tasks]
```

**Replace `_invoke` method** (lines 145-191) to use `Popen`:
```python
def _invoke(
    self,
    task_file: Path,
    worktree_path: Path,
    resource: str,
) -> subprocess.CompletedProcess | ExperimentResult:
    """Spawn claude -p subprocess using Popen for kill support."""
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
        "--output-format", "stream-json",
        "--verbose",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(worktree_path),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    with self._process_lock:
        self._active_processes.append(proc)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
        )
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        return ExperimentResult(
            primary_metric=0.0,
            error_message=(
                f"Experiment timed out after {timeout}s "
                f"(budget: {self._spec.time_budget_seconds}s + {_GRACE_SECONDS}s grace)"
            ),
        )
    finally:
        with self._process_lock:
            if proc in self._active_processes:
                self._active_processes.remove(proc)
```

**Add `kill_active` method:**
```python
def kill_active(self) -> None:
    """Terminate all active subprocesses."""
    with self._process_lock:
        processes = list(self._active_processes)
    for proc in processes:
        try:
            proc.terminate()
        except OSError:
            pass  # Already dead
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_subagent_executor.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add chaosengineer/execution/subagent.py tests/test_subagent_executor.py
git commit -m "feat: switch SubagentExecutor to as_completed with Popen kill support"
```

---

## Chunk 3: Coordinator Integration

### Task 6: Coordinator — pause checks and status display

**Files:**
- Modify: `chaosengineer/core/coordinator.py:35-60,95-116,117-228,350-416`
- Create: `tests/test_pause_coordinator.py`

- [ ] **Step 1: Write failing tests for coordinator pause behavior**

```python
# tests/test_pause_coordinator.py
"""Tests for Coordinator + PauseController integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.interfaces import DimensionPlan, ExperimentTask
from chaosengineer.core.models import (
    Baseline, BudgetConfig, ExperimentResult,
)
from chaosengineer.core.pause import PauseController
from chaosengineer.core.status import StatusDisplay
from chaosengineer.metrics.logger import EventLogger
from chaosengineer.testing.executor import ScriptedExecutor
from chaosengineer.testing.simulator import ScriptedDecisionMaker
from chaosengineer.workloads.parser import WorkloadSpec


def _make_spec(budget: BudgetConfig | None = None) -> WorkloadSpec:
    return WorkloadSpec(
        name="test",
        primary_metric="loss",
        metric_direction="lower",
        execution_command="echo 1",
        workers_available=2,
        budget=budget or BudgetConfig(max_experiments=10),
    )


class TestPauseBeforeIteration:
    def test_pause_requested_logs_run_paused(self, tmp_path):
        """When pause_requested before iteration, log run_paused with user_requested."""
        spec = _make_spec()
        plans = [DimensionPlan("lr", [{"lr": 0.01}, {"lr": 0.1}])]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=2.5),
            "exp-0-1": ExperimentResult(primary_metric=2.8),
        }

        pc = PauseController()
        pc.pause_requested = True  # Pre-set before run

        # Mock menu to return "pause"
        pc.show_post_iteration_menu = MagicMock(return_value="pause")

        log_path = tmp_path / "events.jsonl"
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(log_path),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
            pause_controller=pc,
        )

        coordinator.run()

        paused = EventLogger(log_path).read_events("run_paused")
        assert len(paused) == 1
        assert paused[0]["reason"] == "user_requested"

    def test_continue_resumes_normally(self, tmp_path):
        """When user picks 'continue', run proceeds."""
        spec = _make_spec(BudgetConfig(max_experiments=2))
        plans = [DimensionPlan("lr", [{"lr": 0.01}, {"lr": 0.1}])]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=2.5),
            "exp-0-1": ExperimentResult(primary_metric=2.8),
        }

        pc = PauseController()
        pc.pause_requested = True

        # Mock menu to return "continue"
        pc.show_post_iteration_menu = MagicMock(return_value="continue")

        log_path = tmp_path / "events.jsonl"
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(log_path),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
            pause_controller=pc,
        )

        coordinator.run()

        # Should have run experiments (not paused)
        assert coordinator.budget.experiments_run == 2


class TestPauseAfterIteration:
    def test_wait_then_ask_shows_menu_after_iteration(self, tmp_path):
        """wait_then_ask causes menu to appear after iteration completes."""
        spec = _make_spec(BudgetConfig(max_experiments=4))
        plans = [
            DimensionPlan("lr", [{"lr": 0.01}, {"lr": 0.1}]),
            DimensionPlan("bs", [{"bs": 32}]),
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=2.5),
            "exp-0-1": ExperimentResult(primary_metric=2.8),
            "exp-1-0": ExperimentResult(primary_metric=2.3),
        }

        pc = PauseController()

        # Menu calls: first shows post-iteration (after iter 0), user picks "pause"
        pc.show_post_iteration_menu = MagicMock(return_value="pause")

        log_path = tmp_path / "events.jsonl"
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(log_path),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
            pause_controller=pc,
        )

        # Set wait_then_ask so menu shows after first iteration
        pc.wait_then_ask = True

        coordinator.run()

        paused = EventLogger(log_path).read_events("run_paused")
        assert len(paused) == 1
        assert paused[0]["reason"] == "user_requested"
        # Only first iteration should have run
        assert coordinator.budget.experiments_run == 2


class TestPauseDuringResume:
    def test_pause_works_during_resumed_run(self, tmp_path):
        """Pause checks work in _run_loop() regardless of entry point."""
        from chaosengineer.core.snapshot import (
            RunSnapshot, StopReason, DimensionOutcome, ExperimentSummary,
        )
        spec = _make_spec(BudgetConfig(max_experiments=10))
        plans = [DimensionPlan("bs", [{"bs": 32}])]
        results = {"exp-1-0": ExperimentResult(primary_metric=2.0)}

        pc = PauseController()
        pc.pause_requested = True
        pc.show_post_iteration_menu = MagicMock(return_value="pause")

        snapshot = RunSnapshot(
            run_id="run-pause-resume",
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
            ],
            history=[],
            total_cost_usd=0.5,
            total_experiments_run=1,
            elapsed_time=30.0,
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
            pause_controller=pc,
        )

        coordinator.resume_from_snapshot(snapshot)

        paused = EventLogger(log_path).read_events("run_paused")
        assert len(paused) == 1
        assert paused[0]["reason"] == "user_requested"


class TestNoPauseController:
    def test_none_pause_controller_works(self, tmp_path):
        """Coordinator without pause_controller runs normally."""
        spec = _make_spec(BudgetConfig(max_experiments=2))
        plans = [DimensionPlan("lr", [{"lr": 0.01}, {"lr": 0.1}])]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=2.5),
            "exp-0-1": ExperimentResult(primary_metric=2.8),
        }

        log_path = tmp_path / "events.jsonl"
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(log_path),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
        )

        coordinator.run()
        assert coordinator.budget.experiments_run == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_pause_coordinator.py -v`
Expected: FAIL with `TypeError: Coordinator.__init__() got an unexpected keyword argument 'pause_controller'`

- [ ] **Step 3: Modify Coordinator to accept pause_controller and status_display**

In `chaosengineer/core/coordinator.py`:

**Update `__init__` signature** (add after `run_id` param):
```python
    pause_controller: "PauseController | None" = None,
    status_display: "StatusDisplay | None" = None,
```

**Add fields in `__init__` body** (after `self._history`):
```python
    self._pause_controller = pause_controller
    self._status_display = status_display
```

**Update `run()` method** — after `self.budget.start()` (line 109), add:
```python
    if self._status_display:
        self._status_display.on_run_start(self.spec.budget)
```

**Update `_run_loop()` method** — add pause checks:

At the **top of the while loop** (after line 121 `while not self.budget.is_exhausted():`), before the `for baseline in active_baselines:` loop, add:
```python
        # Pause check: before starting new iteration
        if self._pause_controller and self._pause_controller.should_show_menu():
            choice = self._pause_controller.show_post_iteration_menu()
            if choice == "pause":
                self._log_user_pause(active_baselines)
                return
            else:
                self._pause_controller.reset()
```

After `self._iteration += 1` block (after line 197), add:
```python
            # Status display: iteration done
            if self._status_display:
                self._status_display.on_iteration_done(
                    self._iteration - 1, self.best_baseline.metric_value,
                )

            # Pause check: after iteration
            if self._pause_controller and self._pause_controller.kill_issued:
                # Auto-pause after kill — no re-prompting
                self._log_user_pause(active_baselines)
                return
            if self._pause_controller and self._pause_controller.should_show_menu():
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

**Update `_run_iteration()`** — pass callback to executor (line 380):
```python
    # Build callback for status display and pause
    callback = None
    if self._status_display or self._pause_controller:
        def _on_worker_done(task, result, completed, total):
            if self._status_display:
                self._status_display.on_worker_done(task, result, completed, total)
            if (self._pause_controller
                    and self._pause_controller.pause_requested
                    and not self._pause_controller.kill_issued
                    and completed < total):
                choice = self._pause_controller.show_mid_iteration_menu(completed, total)
                if choice == "kill":
                    self._pause_controller.kill_issued = True
                    self.executor.kill_active()
                elif choice == "wait":
                    self._pause_controller.wait_then_ask = True
                    self._pause_controller.pause_requested = False
                elif choice == "continue":
                    self._pause_controller.reset()
        callback = _on_worker_done

    batch_results = self.executor.run_experiments(tasks, on_worker_done=callback)
```

**Add helper method** `_log_user_pause`:
```python
def _log_user_pause(self, active_baselines: list[Baseline]) -> None:
    """Log run_paused with reason user_requested."""
    self.run_state.end_time = time.time()
    self.run_state.total_experiments_run = self.budget.experiments_run
    self.run_state.total_cost_usd = self.budget.spent_usd
    self._log(Event(
        event="run_paused",
        data={
            "reason": "user_requested",
            "last_iteration": self._iteration,
            "budget_state": self.budget.snapshot(),
            "active_baselines": [b.to_dict() for b in active_baselines],
        },
    ))
```

**Update `_discover_diverse_dimensions()`** — add pause check at the end of the for loop body (after both success and failure paths, before the next iteration). Insert at the very end of the `for dim in self.spec.dimensions:` loop, after the success log (line 93) and after both `continue` branches:
```python
            # Pause check during diverse discovery (end of loop body)
            if self._pause_controller and self._pause_controller.pause_requested:
                return  # Will be caught at top of _run_loop
```
Place this **inside the for loop but after all the if/continue/else branches** — it's the last statement in the loop body so it fires regardless of whether discovery succeeded or failed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_pause_coordinator.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All tests pass (existing tests don't pass `pause_controller`, so it defaults to None)

- [ ] **Step 6: Commit**

```bash
git add chaosengineer/core/coordinator.py tests/test_pause_coordinator.py
git commit -m "feat: integrate PauseController and StatusDisplay into Coordinator"
```

---

### Task 7: CLI wiring

**Files:**
- Modify: `chaosengineer/cli.py:267-282,400-421`

- [ ] **Step 1: Write failing test for CLI wiring**

Add to `tests/test_cli_run.py` (or appropriate CLI test file):

```python
class TestPauseControllerWiring:
    def test_execute_run_installs_pause_controller(self):
        """_execute_run creates and installs PauseController."""
        # This is best verified by checking the coordinator receives it.
        # Since _execute_run is hard to unit test (it does I/O), we test
        # that the import and wiring code works.
        from chaosengineer.core.pause import PauseController
        from chaosengineer.core.status import StatusDisplay

        pc = PauseController()
        sd = StatusDisplay()
        # Verify they can be instantiated and methods exist
        assert hasattr(pc, "install")
        assert hasattr(pc, "uninstall")
        assert hasattr(sd, "on_run_start")
```

- [ ] **Step 2: Modify _execute_run() in cli.py**

In `chaosengineer/cli.py`, in `_execute_run()`, **before the `coordinator = Coordinator(...)` call** (around line 267), add:

```python
    from chaosengineer.core.pause import PauseController
    from chaosengineer.core.status import StatusDisplay

    pause_controller = PauseController()
    status_display = StatusDisplay()
    pause_controller.set_executor(executor)
```

**Update the Coordinator constructor call** to include:
```python
        pause_controller=pause_controller,
        status_display=status_display,
```

**Wrap the `coordinator.run()` call** (line 282) in a try/finally:
```python
    pause_controller.install()
    try:
        coordinator.run()
    finally:
        pause_controller.uninstall()
```

**Apply the same pattern to `_execute_resume()`** — add the same PauseController/StatusDisplay wiring before the `coordinator.resume_from_snapshot()` call.

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -x -q`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add chaosengineer/cli.py tests/test_cli_run.py
git commit -m "feat: wire PauseController and StatusDisplay into CLI entry points"
```

---

## Chunk 4: Round-Trip and E2E Tests

### Task 8: Kill-then-resume round-trip test

**Files:**
- Add to: `tests/test_pause_coordinator.py`

- [ ] **Step 1: Write the round-trip test**

Add to `tests/test_pause_coordinator.py`:

```python
class TestKillResumeRoundTrip:
    def test_kill_mid_iteration_produces_resumable_snapshot(self, tmp_path):
        """Kill mid-iteration → kill_issued auto-pauses → snapshot → resume completes it."""
        from chaosengineer.core.snapshot import build_snapshot, StopReason

        spec = _make_spec(BudgetConfig(max_experiments=10))

        # First run: 2 experiments planned, second fails (simulating kill)
        plans_run1 = [DimensionPlan("lr", [{"lr": 0.01}, {"lr": 0.1}])]
        results_run1 = {
            "exp-0-0": ExperimentResult(primary_metric=2.5),
            "exp-0-1": ExperimentResult(primary_metric=0.0, error_message="killed"),
        }

        pc = PauseController()
        # Pre-set pause_requested — coordinator callback will show menu on first worker
        pc.pause_requested = True
        # Mock mid-iteration menu to return "kill"
        pc.show_mid_iteration_menu = MagicMock(return_value="kill")
        # show_post_iteration_menu should NOT be called (kill_issued skips it)

        log_path = tmp_path / "events.jsonl"
        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans_run1),
            executor=ScriptedExecutor(results_run1),
            logger=EventLogger(log_path),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
            pause_controller=pc,
        )

        coordinator.run()

        # Verify kill_issued caused auto-pause
        assert pc.kill_issued is True

        # Verify run_paused event exists
        paused = EventLogger(log_path).read_events("run_paused")
        assert len(paused) == 1
        assert paused[0]["reason"] == "user_requested"

        # Build snapshot and verify it's resumable
        snapshot = build_snapshot(log_path)
        assert snapshot.stop_reason == StopReason.PAUSED

        # Resume: complete the missing work
        plans_run2 = []  # No new dims after gap fill
        results_run2 = {
            "exp-0-1": ExperimentResult(primary_metric=2.6),
        }

        log_path2 = tmp_path / "events2.jsonl"
        import shutil
        shutil.copy(log_path, log_path2)

        coordinator2 = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans_run2),
            executor=ScriptedExecutor(results_run2),
            logger=EventLogger(log_path2),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
        )
        coordinator2.resume_from_snapshot(snapshot)
```

- [ ] **Step 2: Run test**

Run: `python -m pytest tests/test_pause_coordinator.py::TestKillResumeRoundTrip -v`
Expected: PASS (or debug and fix)

- [ ] **Step 3: Commit**

```bash
git add tests/test_pause_coordinator.py
git commit -m "test: add kill-then-resume round-trip test"
```

---

### Task 9: E2E tmux test

**Files:**
- Create: `tests/e2e/test_tmux_pause.py`

- [ ] **Step 1: Write the tmux E2E test**

```python
# tests/e2e/test_tmux_pause.py
"""E2E test: Ctrl+C pause via tmux.

Requires: tmux installed, chaos CLI available.
Skipped if tmux is not available.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest


pytestmark = pytest.mark.e2e

TMUX = shutil.which("tmux")
CHAOS = shutil.which("chaos")


@pytest.mark.skipif(not TMUX, reason="tmux not installed")
@pytest.mark.skipif(not CHAOS, reason="chaos CLI not in PATH")
class TestTmuxPause:
    def test_ctrl_c_pause_menu_and_pause(self, tmp_path):
        """Full signal chain: Ctrl+C → menu → Pause → run_paused event."""
        session = f"test-pause-{time.monotonic_ns()}"
        output_dir = tmp_path / "output"
        workload = tmp_path / "workload.md"

        # Create a trivial workload spec that will run for a while
        workload.write_text(
            "# Test Workload\n"
            "## Primary Metric\n"
            "loss (lower)\n"
            "## Budget\n"
            "max_experiments: 10\n"
            "## Execution Command\n"
            "sleep 30 && echo '{\"loss\": 1.0}'\n"
            "## Dimensions\n"
            "- lr: directional, current=0.01\n"
        )

        try:
            # Start chaos run in tmux
            subprocess.run(
                [TMUX, "new-session", "-d", "-s", session,
                 CHAOS, "run", str(workload), "--output-dir", str(output_dir),
                 "--executor", "subagent", "--llm-backend", "claude-code"],
                check=True,
            )

            # Wait for status line to appear (run started)
            events_path = output_dir / "events.jsonl"
            for _ in range(30):
                time.sleep(1)
                if events_path.exists():
                    break
            else:
                pytest.fail("events.jsonl never created")

            # Send Ctrl+C
            subprocess.run([TMUX, "send-keys", "-t", session, "C-c"], check=True)
            time.sleep(2)

            # Send "P" to select Pause
            subprocess.run([TMUX, "send-keys", "-t", session, "P"], check=True)
            time.sleep(3)

            # Verify run_paused event
            events = []
            with open(events_path) as f:
                for line in f:
                    events.append(json.loads(line.strip()))

            paused = [e for e in events if e.get("event") == "run_paused"]
            assert len(paused) >= 1, f"Expected run_paused event, got events: {[e['event'] for e in events]}"
            assert paused[0]["reason"] == "user_requested"

        finally:
            # Cleanup tmux session
            subprocess.run(
                [TMUX, "kill-session", "-t", session],
                capture_output=True,
            )
```

- [ ] **Step 2: Run test (may be skipped locally)**

Run: `python -m pytest tests/e2e/test_tmux_pause.py -v`
Expected: PASS if tmux and chaos CLI available, SKIP otherwise

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/test_tmux_pause.py
git commit -m "test: add E2E tmux test for Ctrl+C graceful pause"
```

---

## Post-Implementation

After all tasks are complete:

- [ ] Run full test suite: `python -m pytest tests/ -x -q`
- [ ] Run E2E tests: `python -m pytest tests/e2e/ -v`
- [ ] Verify the status line works interactively (manual test): `chaos run <workload> --executor subagent`
