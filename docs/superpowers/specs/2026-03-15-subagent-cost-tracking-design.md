# Subagent Cost Tracking Design

## Problem

`SubagentExecutor` spawns Claude Code subagents via `subprocess.run()` but discards stdout, leaving `ExperimentResult.tokens_in`, `tokens_out`, and `cost_usd` at their default of 0. The downstream infrastructure (`BudgetTracker`, `Coordinator` event logging) already consumes these fields — they just never get populated.

## Solution

Switch the Claude CLI invocation to `--output-format stream-json --verbose`, then parse the final `result` event from stdout to extract cost and token data.

### Claude CLI result event structure

When invoked with `--output-format stream-json --verbose`, the CLI emits NDJSON to stdout. The last event is a `result` object (emitted for both success and error subtypes):

```json
{
  "type": "result",
  "subtype": "success",
  "total_cost_usd": 0.123,
  "num_turns": 3,
  "duration_ms": 34074,
  "usage": {
    "input_tokens": 19671,
    "output_tokens": 4,
    "cache_creation_input_tokens": 19668,
    "cache_read_input_tokens": 0
  },
  "session_id": "..."
}
```

Note: `--verbose` is required when using `--output-format stream-json` with `-p`. It causes additional diagnostic output on stderr; this is addressed in the error-path section below.

## Components

### 1. CLI args (`subagent.py`, `_invoke()`)

Add `"--output-format", "stream-json", "--verbose"` to the `cmd` list. No other invocation changes.

### 2. Stdout parser (`execution/cli_usage.py`, new module)

A frozen dataclass and a single public function:

```python
@dataclass(frozen=True)
class CliUsage:
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

def parse_cli_usage(stdout: str | None) -> CliUsage
```

- Accepts `str | None` — `None` (from `CompletedProcess` without explicit stdout) is treated as empty
- Scans stdout lines in reverse for the last line containing `"type":"result"` (reverse scan matters because stream-json output can be large for long experiments)
- Parses it as JSON, extracts `total_cost_usd`, `usage.input_tokens`, `usage.output_tokens`
- Returns a `CliUsage` frozen dataclass
- On any parse failure (None input, missing line, bad JSON, missing fields), logs a `logger.debug()` message and returns `CliUsage()` (all zeros)
- Never raises — cost tracking must not break experiment execution

We intentionally skip `num_turns` and `duration_ms` from the result event — we already track duration via `time.monotonic()` in `_run_single()`. Cache token fields (`cache_creation_input_tokens`, `cache_read_input_tokens`) are also omitted for now; they can be added to `CliUsage` later without breaking anything.

### 3. Wire into result (`subagent.py`, `_run_single()`)

Extract a helper `_attach_usage(result, stdout)` that populates cost fields on any `ExperimentResult`:

```python
def _attach_usage(result: ExperimentResult, stdout: str | None) -> ExperimentResult:
    usage = parse_cli_usage(stdout)
    result.tokens_in = usage.tokens_in
    result.tokens_out = usage.tokens_out
    result.cost_usd = usage.cost_usd
    return result
```

Apply in all code paths that have a `CompletedProcess`:

- **Happy path** (after `ResultParser.parse()`): attach usage from `invoke_result.stdout`
- **Non-zero exit code path**: attach usage to the error `ExperimentResult` — the CLI may have done significant work before failing, and the result event is still emitted
- **Timeout path** (`TimeoutExpired`): skip — there is no `CompletedProcess`, so no stdout to parse

### 4. Stderr in error messages

With `--verbose`, stderr may contain routine diagnostic output. The existing error message at the non-zero exit code path truncates stderr to 500 chars. This is acceptable — verbose stderr noise in error messages is a minor cosmetic issue and doesn't affect functionality. No change needed.

### 5. Tests

- Update existing subprocess mocks in `test_subagent_executor.py` to include `stdout` with a stream-json result line (mocks currently produce `stdout=None`)
- Add unit tests for `parse_cli_usage()`:
  - Valid result event → correct cost/tokens
  - `None` input → `CliUsage()` zeros
  - Empty string → `CliUsage()` zeros
  - Malformed JSON → `CliUsage()` zeros
  - Valid JSON but missing fields → `CliUsage()` zeros
  - Multiple events including non-result lines → finds last result line
- Verify non-zero exit code path also populates cost fields

## What doesn't change

- **`ResultParser`** — still reads experiment metrics from `result.json`
- **`BudgetTracker` / `Coordinator`** — already wired to use `result.cost_usd`
- **`ExperimentResult` model** — fields already exist
- **Error handling** — cost parse failures logged at debug level, never block experiments

## Data flow

```
subprocess stdout (stream-json) --> parse_cli_usage() --> CliUsage(cost, tokens_in, tokens_out)
                                                                  |
result.json --> ResultParser.parse() --> ExperimentResult <-- cost/tokens injected
                                              |
                                    Coordinator --> BudgetTracker.add_cost()
```
