# Graceful Pause Design

## Goal

Allow users to cleanly pause a running ChaosEngineer session with Ctrl+C, presenting an interactive menu to wait for in-flight work, kill immediately, or continue. A real-time status line shows progress throughout the run.

## Architecture

Signal handler + coordinator flag pattern. A `PauseController` object owns the SIGINT handler and pause state. The coordinator checks the flag at natural boundaries and delegates to the pause controller for menu display. A `StatusDisplay` prints per-worker progress to stderr. No threading, no async — just flag checks at safe points in the main thread.

## Components

### PauseController

**File:** `chaosengineer/core/pause.py`

Manages pause state and the interactive menu. Passed into the coordinator as an optional dependency. Also holds a reference to the executor (for kill support).

**State:**
- `pause_requested: bool` — set by SIGINT handler
- `force_kill: bool` — set by second SIGINT (any timing)
- `wait_then_ask: bool` — set when user picks "Wait" from menu
- `_executor: ExperimentExecutor | None` — set via `set_executor()`, used for kill

**Signal handler behavior:**
- First Ctrl+C: sets `pause_requested = True`, prints "Pause requested — will pause after current work finishes."
- Second Ctrl+C: sets `force_kill = True`, restores default SIGINT handler. This is a passthrough — the third Ctrl+C hard-kills the process via Python's default handler. No additional cleanup logic runs on `force_kill`; it exists only to give the user an escape hatch.

**Menus:**

Mid-iteration menu (workers in flight):
```
Pause requested

  2/4 workers completed this iteration.

  [W] Wait for remaining workers, then decide
  [K] Kill workers and pause now
  [C] Continue running
```

Post-iteration menu (no workers in flight):
```
Iteration 2 complete. val_bpb=0.93

  [P] Pause now
  [C] Continue running
```

Extend `cli_menu.select()` to support letter-key hotkeys (e.g., pressing "W" selects the option prefixed with `[W]`). Falls back to numbered selection in non-interactive mode.

**Methods:**
- `install()` — register SIGINT handler
- `uninstall()` — restore original SIGINT handler
- `set_executor(executor)` — store executor reference for kill
- `on_sigint(signum, frame)` — handler callback
- `should_show_menu()` — True if `pause_requested` or `wait_then_ask`
- `show_mid_iteration_menu(completed, total) -> str` — returns `"wait"`, `"kill"`, or `"continue"`
- `show_post_iteration_menu(summary) -> str` — returns `"pause"` or `"continue"`
- `reset()` — clear all flags (after user picks "continue")

### StatusDisplay

**File:** `chaosengineer/core/status.py`

Prints progress to stderr. No cursor manipulation beyond `\r` to overwrite the current line during worker progress. Newline after each iteration for a scrolling log.

**Format:**
```
[iter 1/? | 2/4 workers done | $0.31 | 00:03:42] Ctrl+C to pause
```

On breakthrough:
```
[iter 1/? | 3/4 workers done | $0.42 | 00:04:15] New best: val_bpb=0.93
```

**Update points and call sites:**
- `on_run_start(budget_config)` — called by `Coordinator.run()` after `run_started` event
- `on_worker_done(task, result, completed, total)` — called by the `on_worker_done` callback passed to `executor.run_experiments()`
- `on_iteration_done(iteration, best_metric)` — called by `Coordinator._run_loop()` after `_evaluate_iteration()`
- `on_breakthrough(metric_name, value)` — called by `Coordinator._run_loop()` when a new best is found

### Executor Changes

**File:** `chaosengineer/execution/subagent.py`

Three changes to `SubagentExecutor`:

1. **`on_worker_done` callback on `run_experiments()`** — called after each worker finishes. Signature: `Callable[[ExperimentTask, ExperimentResult, int, int], None] | None` where the ints are `(completed_count, total_count)`. The executor tracks these counts internally and passes them to the callback. The callback is purely observational (no return value, cannot influence control flow).

2. **Switch to `as_completed()`** — instead of iterating futures in submission order, use `concurrent.futures.as_completed()` so callbacks fire as soon as each worker finishes. Results are collected into a `dict[str, ExperimentResult]` keyed by experiment ID, then reordered to match the input task order before returning.

3. **Kill support** — switch `_invoke()` from `subprocess.run()` to `subprocess.Popen()` + `communicate()`. Store `Popen` references in `self._active_processes` protected by a `threading.Lock` (worker threads append/remove, main thread reads during kill). A `kill_active()` method acquires the lock, copies the list, then calls `process.terminate()` on each. The corresponding futures will raise exceptions, caught as failed experiments in the `as_completed()` loop. This re-entrant call pattern (kill called from within the `as_completed` loop via the callback) is safe because `Popen.terminate()` is thread-safe and the loop simply sees the remaining futures raise exceptions.

**Interface change:** `ExperimentExecutor.run_experiments()` gains an optional `on_worker_done: Callable[[ExperimentTask, ExperimentResult, int, int], None] | None = None` parameter. The base class default implementation calls it after each sequential result with running counts. `ScriptedExecutor` (test double) calls it per result with counts.

### Coordinator Integration

**File:** `chaosengineer/core/coordinator.py`

The coordinator accepts optional `pause_controller` and `status_display` constructor parameters. When `None` (tests, non-interactive), all pause/status logic is skipped.

**The coordinator owns all pause decisions.** The `on_worker_done` callback (passed to the executor) does two things:
1. Calls `status_display.on_worker_done()` to update the progress line
2. If `pause_controller.pause_requested` and workers remain in flight, calls `pause_controller.show_mid_iteration_menu()`. On "kill": calls `executor.kill_active()`, which terminates subprocesses and causes the remaining futures to fail. On "wait": sets `wait_then_ask`, lets the executor loop continue. On "continue": calls `pause_controller.reset()`.

This callback runs in the main thread (the `as_completed()` loop in `run_experiments()` runs in the calling thread, not a worker thread), so there are no concurrency issues with menu display. Note: when the menu is displayed, the `as_completed()` loop blocks waiting for user input. Workers that complete during this time buffer their results in the futures — the completed count shown in the menu may be slightly stale. This is expected and acceptable.

**Pause check points:**

1. **Before starting a new iteration** — if `should_show_menu()`, show the post-iteration menu. On "pause": log `run_paused` event, return. On "continue": reset flags, proceed.

2. **After iteration completes** — if `should_show_menu()`, show post-iteration menu with same behavior.

3. **During diverse dimension discovery** — if `pause_requested` after returning from each `discover_diverse_options()` call, show post-iteration menu (no workers in flight at this point).

**Event:** `run_paused` with `reason: "user_requested"` (distinct from `"budget_exhausted"`, `"time_exhausted"`, `"plateau"`). The event carries the same data fields as the existing budget-exhaustion `run_paused`: `reason`, `last_iteration`, `budget_state`, `active_baselines`. For a mid-iteration kill, `last_iteration` is the last *completed* iteration (not the interrupted one). The incomplete iteration is inferred by `build_snapshot()` from the mismatch between `iteration_started` worker count and actual `worker_completed`/`worker_failed` events — same as the existing crash detection logic.

**Force-kill propagation:** If the user hits Ctrl+C while a menu is displayed, `cli_menu.select()` catches the `KeyboardInterrupt` and re-raises it. Since `force_kill` has already restored the default SIGINT handler, this `KeyboardInterrupt` propagates up through the coordinator and CLI. The `finally` block in CLI wiring calls `pause_controller.uninstall()` for cleanup. No `run_paused` event is logged — the snapshot system infers `StopReason.CRASHED` on next resume, same as a hard kill.

### CLI Wiring

**File:** `chaosengineer/cli.py`

In `_execute_run()` and `_execute_resume()`:
1. Create `PauseController()` and `StatusDisplay()`
2. Call `pause_controller.set_executor(executor)` after creating executor
3. Pass both into `Coordinator` constructor
4. Call `pause_controller.install()` before `coordinator.run()`
5. Call `pause_controller.uninstall()` after run returns (in a `finally` block)

## Compatibility

### Snapshot & Resume

The existing snapshot system needs no structural changes:
- `build_snapshot()` already reconstructs state from events, including incomplete iterations caused by kills
- `resume_from_snapshot()` already handles partial iterations and budget restoration
- The `run_paused` event already exists; `reason: "user_requested"` is a new value but the same event structure
- The existing `StopReason` enum maps `run_paused` → `StopReason.PAUSED` regardless of reason — no changes needed
- A user-requested pause requires no budget extension on resume (budget wasn't exhausted), but the resume flow already handles this: `budget_extensions` defaults to `None` and is simply not applied

### Event Format

No changes to event format. `run_paused` already accepts arbitrary `reason` strings.

## Testing

### Unit Tests

- **PauseController state machine** — flag transitions: first SIGINT sets `pause_requested`, second sets `force_kill`. Reset clears flags. `should_show_menu()` returns True when `pause_requested` or `wait_then_ask`.
- **StatusDisplay formatting** — verify output strings with known values, capture stderr.
- **Coordinator + mock PauseController** — inject `pause_requested=True`, verify `run_paused` logged with `reason: "user_requested"`.
- **Coordinator "wait then ask"** — inject pause, return "wait", verify menu shown again after iteration completes.
- **Coordinator "continue"** — inject pause, return "continue", verify run continues normally.
- **Coordinator pause during resume** — resume from snapshot with `pause_requested=True`, verify `run_paused` logged (same pause checks apply in `_run_loop()` regardless of entry point).
- **Executor `as_completed` with callback** — verify `on_worker_done` callback fires per worker in completion order.
- **Executor `as_completed` result ordering** — verify returned list matches input task order despite completion-order iteration.
- **Executor kill** — verify `kill_active()` terminates stored processes and results include error entries.
- **Kill mid-iteration then resume round-trip** — run coordinator with pause controller that triggers kill mid-iteration, verify the resulting event stream produces a valid snapshot via `build_snapshot()` with correct `IncompleteIteration`, then verify `resume_from_snapshot()` completes the missing workers.

### E2E Test (tmux)

One test covering the full signal chain in a real terminal:
1. `tmux new-session -d` running `chaos run` with a trivial workload
2. Wait for status line to appear in pane output
3. `tmux send-keys C-c` to trigger SIGINT
4. Wait for pause menu to appear
5. `tmux send-keys "P"` to select pause
6. Verify `events.jsonl` contains `run_paused` with `reason: "user_requested"`
7. Clean up tmux session

Uses a scripted/trivial workload so experiments complete in seconds.
