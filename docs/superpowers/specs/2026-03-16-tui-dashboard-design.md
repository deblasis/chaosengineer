# TUI Dashboard Design

## Overview

An integrated terminal UI for ChaosEngineer that provides live experiment monitoring, budget control, and (future) human-in-the-loop evaluation. The TUI runs in-process, toggled on/off during execution — no external bus connections, no multi-instance conflicts.

## Requirements

- Full operator console: monitor + control + future human-in-the-loop eval
- Integrated mode: `--tui` flag on `run` and `resume` commands
- Toggle with `t` (enter TUI) / `q` or `Esc` (exit to log mode) during execution
- Same process — reads events from an in-memory bridge, not the bus
- TUI owns all interaction when active (pause, extend budget, future eval)
- Ctrl+C = pause shortcut within TUI
- Textual library (Python, async)

## Architecture: Hybrid In-Memory Bridge

Events flow through an in-process `asyncio.Queue`. Commands mutate coordinator state directly (PauseController, BudgetTracker). The event queue interface mirrors the protobuf Event shape (flat JSON dicts matching `events.jsonl` format), so a future detached mode could swap the queue source for a gRPC stream.

No gRPC or HTTP involved — the TUI is a view layer on the coordinator, not a bus consumer.

## Event Bridge

`EventBridge` is a thread-safe event store with two roles: ring buffer (history) and notification (live).

```python
class EventBridge:
    def __init__(self, capacity: int = 200):
        self._buffer: deque[dict] = deque(maxlen=capacity)  # ring buffer
        self._subscribers: list[asyncio.Queue] = []
        self._lock: threading.Lock = threading.Lock()

    def publish(self, event: dict) -> None:
        """Called from coordinator thread. Appends to ring buffer, notifies subscribers."""
        with self._lock:
            self._buffer.append(event)
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass  # slow consumer, skip

    def snapshot(self) -> list[dict]:
        """Returns copy of ring buffer for replay on TUI toggle."""
        with self._lock:
            return list(self._buffer)

    def subscribe(self) -> asyncio.Queue[dict]:
        """TUI calls this on activation. Returns a queue for live events."""
        q: asyncio.Queue[dict] = asyncio.Queue(maxsize=500)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """TUI calls this on deactivation."""
        with self._lock:
            self._subscribers.remove(q)
```

Event shape matches flat JSON format from `events.jsonl`:

```json
{"ts": "...", "event": "run_started", "run_id": "...", ...data}
```

**On TUI toggle:** TUI calls `snapshot()` for replay, then `subscribe()` for live events. No duplicates — snapshot returns history up to the subscribe point. On TUI exit, `unsubscribe()` cleans up.

**Commands** go the other direction — TUI calls methods directly on `PauseController` and `Coordinator.extend_budget()`.

## View Toggle

Two view modes coexist:

**Log mode** (default):
- `StatusDisplay` writes to stderr as today
- Background thread monitors stdin for `t` keypress using **cbreak mode** (not full raw), which preserves SIGINT delivery from Ctrl+C
- Ctrl+C triggers existing PauseController / interactive menu

**TUI mode** (activated by `t`):
- Textual app takes over terminal (alternate screen buffer)
- StatusDisplay output is suppressed via `StatusDisplay.suppressed = True` flag (checked in `on_worker_done` / `on_iteration_done` before writing to stderr)
- All keyboard input owned by Textual
- `q`/`Esc` exits TUI, restores log mode
- Ctrl+C triggers pause via direct PauseController call

A `ViewManager` class owns the switch. On `t`: suspends StatusDisplay, launches Textual app with EventBridge and references to PauseController/BudgetTracker. On TUI exit: Textual returns, ViewManager restores terminal and resumes StatusDisplay.

Textual uses an alternate screen buffer, so original scrollback is preserved on exit.

## TUI Layout

```
┌─────────────────────────────────────────────────┐
│  Budget Gauges                          [00:12:34]│
│  Cost: ████████░░ $4.20/$10  Exp: 6/20  Time: ∞  │
├─────────────────────────────────────────────────┤
│  Experiments                                      │
│  Iteration 3 (3 workers)                          │
│  # │ Worker │ Dimension        │ Status │ Cost │ Δ │
│  7 │ W1     │ error-handling   │ ⟳ run  │$0.41 │...│
│  8 │ W2     │ edge-cases       │ ⟳ run  │$0.33 │...│
│  9 │ W3     │ concurrency      │ ⟳ run  │$0.12 │...│
│                                                   │
│  Iteration 2 (3 workers) ✓                        │
│  4 │ W1     │ logging          │ ✓ done │$0.82 │+12%│
│  5 │ W2     │ input-validation │ ✓ done │$0.91 │ +8%│
│  6 │ W3     │ retries          │ ✓ done │$0.44 │ +3%│
├─────────────────────────────────────────────────┤
│  Event Log                                        │
│  14:23:45 experiment_started dim=error-handling    │
│  14:23:52 experiment_completed score=0.87          │
│  14:24:01 iteration_completed iteration=2          │
├─────────────────────────────────────────────────┤
│  [P]ause  [E]xtend budget  [Q]uit TUI            │
└─────────────────────────────────────────────────┘
```

**Budget Gauges:** Textual `ProgressBar` widgets. Elapsed time clock in the corner.

**Experiments:** `DataTable` grouped by iteration. Current iteration at top, expanded. Past iterations collapsed, expandable with Enter/arrow keys. Worker column shows parallel slot. Sequential mode shows one worker per iteration.

**Event Log:** Scrolling `RichLog`. Auto-scrolls, pauses on manual scroll-up.

**Command Bar:** Footer hotkeys. `P` = pause, `E` = extend budget modal, `Q` = exit TUI.

**Future:** Human-in-the-loop evaluation adds a modal when the coordinator requests human input, showing experiment results and asking for a score/decision. Command bar gains a hotkey for this.

## Threading Model

The coordinator is synchronous. Textual is async. Solution: coordinator runs in a worker thread, main thread owns the view.

**Log mode:**
- Main thread: ViewManager loops on stdin watching for `t`
- Coordinator thread: iteration loop, pushes events to bridge

**TUI mode:**
- Main thread: runs Textual app event loop, consumes EventBridge, handles input
- Coordinator thread: continues running iterations

Refactor: `_execute_run()` spawns the coordinator in a thread, then runs ViewManager in the main thread. ViewManager either loops on stdin (log mode) or runs Textual (TUI mode).

### Pause Decision Handoff

The coordinator needs to ask the user what to do at pause points (post-iteration, mid-iteration). The mechanism differs by view mode.

```python
class PauseGate:
    """Shared object for coordinator <-> TUI pause decision handoff."""
    def __init__(self):
        self.decision: str | None = None          # "continue", "pause", "extend", etc.
        self.decision_ready = threading.Event()    # TUI sets this after user chooses
        self.decision_needed = threading.Event()   # coordinator sets this to request a decision
        self.options: list[str] = []               # choices to present

    def request_decision(self, options: list[str]) -> str:
        """Called from coordinator thread. Blocks until TUI user decides."""
        self.options = options
        self.decision = None
        self.decision_ready.clear()
        self.decision_needed.set()
        self.decision_ready.wait()                 # blocks coordinator
        self.decision_needed.clear()
        return self.decision

    def submit_decision(self, choice: str) -> None:
        """Called from TUI thread when user picks an option."""
        self.decision = choice
        self.decision_ready.set()                  # unblocks coordinator
```

**Coordinator flow (in pause check):**

```python
if self.pause_controller.pause_requested:
    if self.view_manager and self.view_manager.tui_active:
        # Emit event so TUI shows a modal
        self.logger.log(Event("pause_decision_needed", options=["continue", "pause", "extend"]))
        choice = self.pause_gate.request_decision(["continue", "pause", "extend"])
    else:
        # Existing interactive menu on stderr
        choice = self.pause_controller.show_post_iteration_menu()
    # act on choice...
```

**Toggle during pending decision:** If the user exits TUI while a pause modal is showing, the TUI calls `submit_decision("continue")` as a safe default before exiting, unblocking the coordinator. The log-mode view then shows "Run continuing..." so the user knows what happened.

## Integration Points

### CLI (`cli.py`)
- New `--tui` flag added to both `run` and `resume` subparser argument groups
- Assembly and wiring in `_execute_run()` / `_execute_resume()`:

```python
# In _execute_run(), after building coordinator dependencies:
if args.tui:
    bridge = EventBridge()
    pause_gate = PauseGate()
    publisher = EventPublisher(bus_url, fallback_path, bridge=bridge)
    view_manager = ViewManager(bridge, pause_gate, pause_controller, coordinator)
    coordinator = Coordinator(..., logger=publisher, view_manager=view_manager, pause_gate=pause_gate)
    # Run coordinator in background thread
    coord_thread = threading.Thread(target=coordinator.run, daemon=True)
    coord_thread.start()
    # Main thread owns the view
    view_manager.run()  # blocks until coordinator finishes or user quits
    coord_thread.join()
else:
    publisher = EventPublisher(bus_url, fallback_path)
    coordinator = Coordinator(..., logger=publisher)
    coordinator.run()  # existing synchronous path, unchanged
```

### EventPublisher (`metrics/publisher.py`)
- Optional `EventBridge` in constructor
- On `log(event)`: publishes to bus + `bridge.put_nowait(event)` if bridge exists
- Keeps local ring buffer (~200 events) for replay on TUI toggle

### PauseController (`core/pause.py`)
- No changes to existing fields — TUI sets `pause_requested = True` directly
- Coordinator checks `view_manager.tui_active` before showing interactive menus
- When TUI is active: coordinator emits `pause_decision_needed` event, blocks on threading.Event

### Coordinator (`core/coordinator.py`)
- Constructor gains optional `view_manager: ViewManager | None = None` and `pause_gate: PauseGate | None = None`
- New public method extracted from `_poll_bus_commands`:

```python
def extend_budget(self, add_cost: float = 0, add_experiments: int = 0,
                  add_time: float = 0) -> None:
    """Extend budget limits. Thread-safe — called from TUI or bus command polling."""
    with self._budget_lock:  # new threading.Lock, guards config mutation
        cfg = self.budget.config
        self.budget.config = BudgetConfig(
            max_api_cost=(cfg.max_api_cost + add_cost) if cfg.max_api_cost is not None else None,
            max_experiments=(cfg.max_experiments + add_experiments) if cfg.max_experiments is not None else None,
            max_wall_time_seconds=(cfg.max_wall_time_seconds + add_time) if cfg.max_wall_time_seconds is not None else None,
            max_plateau_iterations=cfg.max_plateau_iterations,
        )
```

- Pause check points use `PauseGate` when TUI is active (see Pause Decision Handoff above)
- `_poll_bus_commands` refactored to call `self.extend_budget()` instead of inline mutation

### New files
- `chaosengineer/tui/app.py` — Textual app, layout, event handling
- `chaosengineer/tui/bridge.py` — EventBridge class
- `chaosengineer/tui/views.py` — ViewManager (toggle logic)
- `chaosengineer/tui/widgets.py` — Custom widgets if needed

## Error Handling

**TUI crashes:** ViewManager catches, restores terminal, falls back to log mode. Coordinator unaffected.

**Coordinator finishes while TUI is open:** `run_completed` event → "Run complete" banner with final stats. User reviews, then `q` exits.

**Coordinator errors while TUI is open:** `run_failed` event → highlighted error panel. Same exit flow.

**Toggle during pause decision:** Switching to TUI cancels log-mode menu, re-emits as TUI modal. Exiting TUI during modal transfers back to log-mode menu.

**Rapid toggle:** 500ms debounce in `ViewManager` — tracks `last_toggle_ts` and ignores `t`/`q` keypresses within the window.

**Terminal resize:** Handled natively by Textual. Log mode unaffected.

**Minimum terminal size:** TUI requires at least 80x24. On smaller terminals, Textual shows a "terminal too small" placeholder (built-in behavior).

## Testing Strategy

### Unit tests
- EventBridge: put/get, bounded overflow, replay buffer
- ViewManager: toggle state machine (log→TUI→log), mock Textual app
- TUI widgets: canned events → assert table rows, gauge values

### Integration tests
- Coordinator thread + EventBridge + mock TUI consumer: events flow end-to-end
- Pause decision handoff: coordinator blocks, mock TUI sets decision, coordinator resumes
- Budget extension through TUI: verify BudgetTracker state changes

### Textual snapshot tests
- Initial state, mid-run with parallel experiments, pause modal, run complete banner

### Manual tests
- Toggle in/out during scripted run
- Pause and extend budget from TUI
- Terminal restores cleanly on exit and crash

### Deferred
- Human-in-the-loop evaluation (no RPC yet)
- Detached mode (not implemented)

## Dependencies

- `textual>=0.47` — TUI framework (stable DataTable, RichLog, snapshot testing)
- No new Go dependencies (TUI doesn't use the bus gRPC API)

## Note on Message Bus Spec

The message bus spec (`2026-03-16-message-bus-design.md`) shows the TUI as a gRPC stream subscriber. That reflects the original vision. This spec supersedes that: the integrated TUI uses an in-memory EventBridge, not the bus gRPC API. The bus gRPC Subscribe stream remains available for a future detached monitor mode or web dashboard.

## Deferred Work

- `SubmitEvaluation` RPC and human-in-the-loop eval UI
- Detached monitor mode (`chaosengineer monitor <bus-url>`)
- Web dashboard (separate spec)
