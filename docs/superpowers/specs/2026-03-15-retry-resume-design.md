# Design: Retry/Resume — Resume Partially-Completed Runs

## Problem

ChaosEngineer runs can take hours and consume significant API budget. When a run stops — whether due to budget exhaustion, time limits, user intent, or unexpected interruption — all in-memory coordinator state is lost. The event log and experiment artifacts persist on disk, but there is no mechanism to pick up where the run left off.

Users must currently re-run from scratch, wasting completed experiments, budget, and time. For long-running overnight experiments, this makes any interruption costly.

## Solution

Add a `chaosengineer resume <output-dir>` subcommand that reconstructs coordinator state from the append-only event log, resolves any incomplete work, and re-enters the normal coordinator loop. A decision log provides observability into LLM reasoning for troubleshooting.

## State Reconstruction (`core/snapshot.py`)

A `RunSnapshot` dataclass built by replaying `events.jsonl`. Pure function of the event log — no side effects.

```python
@dataclass
class RunSnapshot:
    run_id: str
    workload_name: str
    workload_spec_hash: str  # SHA256 of original workload spec
    budget_config: BudgetConfig
    mode: str

    # Restored state
    active_baselines: list[Baseline]  # may be >1 due to beam search ties
    baseline_history: list[Baseline]
    dimensions_explored: list[DimensionOutcome]
    discovered_dimensions: dict[str, list[str]]  # DIVERSE dim name → discovered options
    experiments: list[ExperimentSummary]  # completed + failed
    history: list[dict]  # reconstructed _history for decision maker context
    total_cost_usd: float
    total_experiments_run: int
    elapsed_time: float  # wall time consumed before stop
    consecutive_no_improvement: int

    # Incomplete work
    incomplete_iteration: IncompleteIteration | None
    stop_reason: StopReason

class StopReason(Enum):
    PAUSED = "paused"          # clean stop: run_paused event present
    COMPLETED = "completed"    # run finished: run_completed event present
    CRASHED = "crashed"        # inferred: no terminal event (run_paused/run_completed)

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
    missing_tasks: list[ExperimentTask]  # full task objects for re-spawning

@dataclass
class ExperimentSummary:
    experiment_id: str
    dimension: str
    params: dict
    metric: float | None
    status: str  # "completed", "failed"
    cost_usd: float
```

Reconstruction logic:

```python
def build_snapshot(events_path: Path) -> RunSnapshot:
    """Replay events.jsonl to reconstruct run state."""
    events = read_events(events_path)

    # Walk events chronologically
    # Note: existing logger uses "event" as the key, not "type"
    for entry in events:
        match entry["event"]:
            case "run_started":
                # Initialize snapshot fields, store workload_spec_hash
            case "iteration_started":
                # Track current iteration dimension + worker count
                # Store experiment tasks for potential gap-fill
            case "worker_completed":
                # Add to experiments and history, update budget counters
            case "worker_failed":
                # Add to experiments and history as failed
            case "breakthrough":
                # Update active_baselines (may be multiple on ties)
            case "diverse_discovered":
                # Store discovered options: discovered_dimensions[dim] = options
            case "budget_checkpoint":
                # Cross-check budget state
            case "run_paused":
                # Mark stop_reason = StopReason.PAUSED
            case "run_completed":
                # Mark stop_reason = StopReason.COMPLETED (resume is a no-op)
            case "run_resumed":
                # Continue replaying from resume point

    # If no terminal event (run_paused/run_completed), infer crash
    if stop_reason is None:
        stop_reason = StopReason.CRASHED

    # Detect incomplete iteration: iteration_started with fewer
    # worker_completed/worker_failed than expected.
    # Reconstruct missing ExperimentTask objects from iteration_started data.
    detect_incomplete_iteration(snapshot)

    return snapshot
```

## Resume CLI (`cli/resume.py`)

```
chaosengineer resume <output-dir> [--add-cost N] [--add-experiments N] [--add-time N] [--restart-iteration]
```

Flow:

1. Load snapshot from `<output-dir>/events.jsonl`
2. Validate: git repo accessible, workload spec hash matches (warn if changed)
3. If `stop_reason == "completed"`: print "Run already completed" and exit
4. If budget exhausted and no `--add-*` flags: error with message suggesting flags
5. Apply budget extensions to snapshot's budget config
6. Print summary: "Resuming run X — 3/6 dimensions explored, best: 2.41, $4.20 spent"
7. If `stop_reason == "crashed"` (no `run_paused` event): print diagnostics — last event, orphaned worktrees, incomplete workers — and ask for confirmation
8. Log `run_resumed` event
9. Hand off to coordinator via `coordinator.resume_from_snapshot(snapshot)`

Budget extension flags:

- `--add-cost 5.0` — increases `max_api_cost` by $5
- `--add-experiments 10` — increases `max_experiments` by 10
- `--add-time 3600` — increases `max_wall_time_seconds` by 3600

## Run Guard (`cli.py`)

When `chaosengineer run` targets an output dir containing an existing `events.jsonl` with a resumable session (not `run_completed`):

```
Found existing run (3/6 dimensions explored, best: 2.41)

  → Resume previous run
    Start fresh (archive existing)
    Cancel
```

Interactive arrow-key menu (up/down + Enter). Selecting:

- **Resume**: prints the `chaosengineer resume <output-dir>` command and exits (user runs it explicitly, keeping the commands distinct)
- **Start fresh**: moves existing output dir to `<output-dir>.bak/` with timestamp, proceeds with new run
- **Cancel**: exits

Non-interactive fallback (pipes, CI): `--force-fresh` flag skips the prompt.

## Interactive Menu (`cli/menu.py`)

Minimal arrow-key selection utility. No external dependencies.

```python
def select(prompt: str, options: list[str], default: int = 0) -> int:
    """Show interactive menu, return selected index.

    Uses raw terminal input (tty/termios) for arrow key navigation.
    Falls back to numbered list with text input for non-interactive terminals.
    """
```

Display format:
```
{prompt}

  → {highlighted option}
    {other option}
    {other option}
```

Arrow keys navigate, Enter selects. Returns the index of the selected option.

## Coordinator Changes (`core/coordinator.py`)

New method `resume_from_snapshot(snapshot: RunSnapshot)`:

```python
def resume_from_snapshot(self, snapshot: RunSnapshot):
    """Resume a run from a reconstructed snapshot."""
    # Initialize Run from snapshot
    self.run = Run(
        run_id=snapshot.run_id,
        # ... restored fields
    )

    # Budget tracker with elapsed-time offset: adds prior elapsed time
    # to all future elapsed_seconds calculations so wall-time budget
    # accounts for time spent in the original run.
    self.budget_tracker = BudgetTracker.from_snapshot(
        budget_config=snapshot.budget_config,
        experiments_run=snapshot.total_experiments_run,
        cost_spent=snapshot.total_cost_usd,
        elapsed_offset=snapshot.elapsed_time,
        consecutive_no_improvement=snapshot.consecutive_no_improvement,
    )

    self._iteration = len(snapshot.dimensions_explored)
    self.active_baselines = snapshot.active_baselines
    self._history = snapshot.history

    # Restore discovered DIVERSE dimension options so they aren't re-discovered
    for dim in self.workload.dimensions:
        if dim.name in snapshot.discovered_dimensions:
            dim.options = snapshot.discovered_dimensions[dim.name]

    # Build factual context for LLM decision maker
    context = build_resume_context(snapshot)
    self.decision_maker.set_prior_context(context)

    # Handle incomplete iteration
    if snapshot.incomplete_iteration and not self.restart_iteration:
        self._complete_partial_iteration(snapshot.incomplete_iteration)
    elif snapshot.incomplete_iteration and self.restart_iteration:
        # Discard all results from the incomplete iteration.
        # The dimension is returned to the unexplored pool so the LLM
        # can re-pick it (or pick a different one) on the next iteration.
        pass

    # Enter normal loop
    self._run_loop()
```

The factual context block sent to the LLM:

```
Previous run state (resuming):
- Dimensions explored: learning_rate (winner: 0.001, +12%), activation (winner: relu, +3%)
- Active baselines: loss=2.41 (exp-2-3), loss=2.42 (exp-2-1)  [beam search]
- Experiments run: 8 (6 completed, 2 failed)
- Budget remaining: $5.80 / 12 experiments / 45min
- Dimensions remaining: [batch_size, dropout, optimizer]
```

### DecisionMaker Interface Addition

The `DecisionMaker` ABC gains a new method:

```python
class DecisionMaker(ABC):
    # ... existing methods ...

    def set_prior_context(self, context: str) -> None:
        """Provide factual summary of prior run state for resume.

        Called before the first decision after a resume. The context string
        is prepended to subsequent LLM prompts. Default is a no-op for
        implementations that don't need it (e.g. ScriptedDecisionMaker).
        """
        pass  # default no-op; LLMDecisionMaker overrides
```

`ScriptedDecisionMaker` inherits the no-op default. `LLMDecisionMaker` stores the context and prepends it to its next prompt.

### Completing a Partial Iteration

```python
def _complete_partial_iteration(self, incomplete: IncompleteIteration):
    """Run only the missing workers from an interrupted iteration."""
    # Spawn workers using full ExperimentTask objects
    new_results = self.executor.run_experiments(incomplete.missing_tasks)

    # Merge with already-completed experiments
    all_results = incomplete.completed_experiments + new_results

    # Log iteration_gap_completed event
    self.logger.log_iteration_gap_completed(
        dimension=incomplete.dimension,
        original_completed=len(incomplete.completed_experiments),
        gap_filled=len(new_results),
    )

    # Evaluate full set
    self._evaluate_iteration(incomplete.dimension, all_results)
```

### `--restart-iteration` Behavior

When `--restart-iteration` is set and a partial iteration exists:

1. All results from the incomplete iteration are discarded (not counted in budget)
2. The dimension is returned to the unexplored pool
3. The coordinator enters the normal loop, where the LLM picks the next dimension (may re-pick the same one or choose differently)
4. A `run_resumed` event notes `"restart_iteration": true` for auditability

## Decision Log (`core/decision_log.py`)

Separate `decisions.jsonl` file in the output dir. Write-only during runs, read-only by humans. Not used by resume logic.

```python
class DecisionLogger:
    def __init__(self, output_dir: Path):
        self.path = output_dir / "decisions.jsonl"

    def log_dimension_selected(self, dimension: str, reasoning: str,
                                alternatives: list[str]):
        self._append({
            "type": "dimension_selected",
            "dimension": dimension,
            "reasoning": reasoning,
            "alternatives": alternatives,
            "timestamp": time.time(),
        })

    def log_results_evaluated(self, dimension: str, reasoning: str,
                               winner: str | None, metrics: dict):
        self._append({
            "type": "results_evaluated",
            "dimension": dimension,
            "reasoning": reasoning,
            "winner": winner,
            "metrics": metrics,
            "timestamp": time.time(),
        })

    def log_diverse_options(self, dimension: str, reasoning: str,
                            options: list[str]):
        self._append({
            "type": "diverse_options_generated",
            "dimension": dimension,
            "reasoning": reasoning,
            "options": options,
            "timestamp": time.time(),
        })
```

The `LLMDecisionMaker` receives a `DecisionLogger` instance and calls it after each decision. The logger interface is a simple callback — no coupling to the decision-making logic.

## Event Log Additions (`metrics/logger.py`)

Three new event types. Note: uses `"event"` key to match existing logger convention.

### Coordinator exit logic

The coordinator currently always emits `run_completed`. This changes to:

- **`run_completed`** — emitted only when all dimensions are exhausted or the run finishes naturally
- **`run_paused`** — emitted when the loop exits due to budget exhaustion, time limit, or plateau limit

The branching logic in the coordinator loop exit:

```python
if all_dimensions_explored or no_dimensions_remaining:
    self.logger.log_run_completed(...)
else:
    # Budget/time/plateau caused the stop — run is resumable
    self.logger.log_run_paused(reason=budget_tracker.exhaustion_reason, ...)
```

### New events

**`run_paused`** — logged when a run stops but is resumable:
```json
{
    "event": "run_paused",
    "reason": "budget_exhausted",
    "last_iteration": 3,
    "budget_state": {"cost_spent": 9.80, "experiments_run": 14, "elapsed_seconds": 3600},
    "active_baselines": [{"metric": 2.41, "experiment_id": "exp-2-3"}],
    "timestamp": 1706000500
}
```

The `reason` field values: `"budget_exhausted"`, `"time_exhausted"`, `"plateau"`, `"user_interrupt"`.

**`run_resumed`** — logged at resume start:
```json
{
    "event": "run_resumed",
    "original_run_id": "run-abc123",
    "budget_extensions": {"add_cost": 5.0},
    "restart_iteration": false,
    "snapshot_summary": {"dimensions_explored": 3, "experiments_completed": 12},
    "timestamp": 1706001000
}
```

**`iteration_gap_completed`** — logged when partial iteration's missing workers finish:
```json
{
    "event": "iteration_gap_completed",
    "dimension": "learning_rate",
    "original_completed": 3,
    "gap_filled": 2,
    "timestamp": 1706001100
}
```

**Workload spec hash** — new field on `run_started`:
```json
{
    "event": "run_started",
    "workload_spec_hash": "sha256:a1b2c3...",
    ...
}
```

## Budget Tracker Changes (`core/budget.py`)

New class method for restoring from a snapshot:

```python
class BudgetTracker:
    @classmethod
    def from_snapshot(cls, budget_config, experiments_run, cost_spent,
                      elapsed_offset, consecutive_no_improvement):
        """Create a tracker pre-loaded with prior run state."""
        tracker = cls(budget_config)
        tracker._experiments_run = experiments_run
        tracker._cost_spent = cost_spent
        tracker._elapsed_offset = elapsed_offset  # NEW FIELD
        tracker._consecutive_no_improvement = consecutive_no_improvement
        return tracker

    @property
    def elapsed_seconds(self) -> float:
        """Wall time including prior run's elapsed time."""
        current = time.monotonic() - self._start_time if self._start_time else 0
        return current + getattr(self, '_elapsed_offset', 0)
```

The `_elapsed_offset` field is added to the existing `elapsed_seconds` calculation, so wall-time budget correctly accounts for time spent in the original run.

## Testing

1. **Snapshot reconstruction** — unit tests for `build_snapshot()` with crafted event sequences: complete runs, partial iterations, crashes (missing terminal event), multiple resumes, budget-exhausted stops
2. **Incomplete iteration detection** — test that `IncompleteIteration` is correctly populated when `iteration_started` has more workers than `worker_completed + worker_failed`, and that `missing_tasks` contains valid `ExperimentTask` objects
3. **Budget restoration** — verify `BudgetTracker.from_snapshot()` restores exact state (including `elapsed_offset`), and budget extensions apply correctly
4. **Run guard** — test that existing `events.jsonl` triggers the guard, `run_completed` sessions don't, and `--force-fresh` bypasses it
5. **Coordinator resume** — integration tests using `ScriptedDecisionMaker` and `ScriptedExecutor`: resume at clean boundary, resume with partial iteration, resume with `--restart-iteration`
6. **Beam search state** — verify `active_baselines` with >1 entry survives snapshot round-trip, and coordinator resumes with all branches intact
7. **DIVERSE dimension restoration** — verify discovered options are stored in snapshot and restored to dimension specs on resume (no re-discovery)
8. **Decision log** — verify entries written for each decision type, file is valid JSONL
9. **Menu fallback** — test non-interactive fallback for CI environments
10. **Event log round-trip** — write events for a full run, build snapshot, verify all fields match original state (including `_history` reconstruction)
11. **Workload spec validation** — test hash mismatch warning, missing spec handling
12. **Coordinator exit branching** — verify `run_completed` vs `run_paused` emitted correctly based on whether dimensions are exhausted vs budget

## Out of Scope (Future Work)

- **Graceful pause with keyboard menu** — interactive menu during a run (wait for workers / stop now / continue). Separate feature that builds on the resume architecture but doesn't change it.
- **Retryable error classes** — define categories of errors (API timeouts, rate limits, transient infra failures) that should be automatically retried rather than treated as experiment failures. These are infra issues unrelated to the experiment itself. Requires: error classification taxonomy, retry policy config (max retries, backoff), and distinguishing "experiment failed" from "infra failed" in event log.
- **Human-in-the-loop escalation** — flow for when a human needs to perform an action during a run (e.g., evaluating an experiment, providing qualitative judgment, approving a next step) and making results available to ChaosEngineer. Requires: pause-for-input mechanism, result ingestion format, notification/polling.
- **Special multi-resume handling** — multiple resumes should work naturally via event replay, but no special edge-case logic in v1.
