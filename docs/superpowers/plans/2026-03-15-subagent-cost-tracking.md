# Subagent Cost Tracking Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parse Claude CLI stream-json output to populate ExperimentResult cost/token fields that currently default to 0.

**Architecture:** New `cli_usage.py` module with a frozen `CliUsage` dataclass and `parse_cli_usage()` function. SubagentExecutor adds `--output-format stream-json --verbose` to CLI args and calls `parse_cli_usage()` on stdout to populate cost fields on ExperimentResult in both success and error paths.

**Tech Stack:** Python 3.10+, pytest, dataclasses, json stdlib

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `chaosengineer/execution/cli_usage.py` | `CliUsage` dataclass + `parse_cli_usage()` |
| Create | `tests/test_cli_usage.py` | Unit tests for parser |
| Modify | `chaosengineer/execution/subagent.py:114-129,156-159` | Wire usage into `_run_single()` result paths + add CLI flags in `_invoke()` |
| Modify | `tests/test_subagent_executor.py` | Update mocks to include stdout, add cost assertion tests |

---

## Chunk 1: CLI Usage Parser

### Task 1: CliUsage dataclass and parse_cli_usage() — tests first

**Files:**
- Create: `tests/test_cli_usage.py`
- Create: `chaosengineer/execution/cli_usage.py`

- [ ] **Step 1: Write failing tests for parse_cli_usage**

```python
"""Tests for CLI usage parser."""

import json

import pytest

from chaosengineer.execution.cli_usage import CliUsage, parse_cli_usage


def _result_line(cost: float = 0.5, tokens_in: int = 1000, tokens_out: int = 200) -> str:
    return json.dumps({
        "type": "result",
        "subtype": "success",
        "total_cost_usd": cost,
        "num_turns": 3,
        "usage": {
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    })


class TestCliUsageDefaults:
    def test_frozen_dataclass(self):
        usage = CliUsage()
        assert usage.cost_usd == 0.0
        assert usage.tokens_in == 0
        assert usage.tokens_out == 0
        with pytest.raises(AttributeError):
            usage.cost_usd = 1.0


class TestParseCliUsage:
    def test_valid_result_event(self):
        stdout = _result_line(cost=0.123, tokens_in=5000, tokens_out=300)
        usage = parse_cli_usage(stdout)
        assert usage.cost_usd == 0.123
        assert usage.tokens_in == 5000
        assert usage.tokens_out == 300

    def test_result_among_other_events(self):
        lines = [
            '{"type":"system","subtype":"init"}',
            '{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}',
            _result_line(cost=0.5),
        ]
        usage = parse_cli_usage("\n".join(lines))
        assert usage.cost_usd == 0.5

    def test_error_subtype_still_has_cost(self):
        line = json.dumps({
            "type": "result",
            "subtype": "error",
            "total_cost_usd": 0.08,
            "usage": {"input_tokens": 100, "output_tokens": 10},
        })
        usage = parse_cli_usage(line)
        assert usage.cost_usd == 0.08
        assert usage.tokens_in == 100

    def test_none_input(self):
        assert parse_cli_usage(None) == CliUsage()

    def test_empty_string(self):
        assert parse_cli_usage("") == CliUsage()

    def test_no_result_event(self):
        stdout = '{"type":"assistant","message":{}}\n{"type":"system"}'
        assert parse_cli_usage(stdout) == CliUsage()

    def test_malformed_json(self):
        assert parse_cli_usage("not json at all") == CliUsage()

    def test_result_missing_cost_fields(self):
        line = json.dumps({"type": "result", "subtype": "success"})
        usage = parse_cli_usage(line)
        assert usage.cost_usd == 0.0
        assert usage.tokens_in == 0

    def test_uses_last_result_event(self):
        lines = [
            _result_line(cost=0.1),
            _result_line(cost=0.5),
        ]
        usage = parse_cli_usage("\n".join(lines))
        assert usage.cost_usd == 0.5
```

Write this to `tests/test_cli_usage.py`.

- [ ] **Step 2: Run tests — verify they fail**

Run: `python -m pytest tests/test_cli_usage.py -v`
Expected: ImportError — `chaosengineer.execution.cli_usage` does not exist

- [ ] **Step 3: Implement cli_usage.py**

```python
"""Parse Claude CLI stream-json output for usage/cost data."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CliUsage:
    """Cost and token usage extracted from Claude CLI output."""

    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


def parse_cli_usage(stdout: str | None) -> CliUsage:
    """Extract cost/token data from Claude CLI stream-json stdout.

    Scans lines in reverse for the last ``{"type":"result",...}`` event.
    Returns ``CliUsage()`` (all zeros) on any failure — never raises.
    """
    if not stdout:
        return CliUsage()

    # Scan in reverse — result event is the last line
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        if '"type":"result"' not in line and '"type": "result"' not in line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("Failed to parse CLI result line: %s", line[:200])
            continue
        if data.get("type") != "result":
            continue
        usage = data.get("usage", {})
        return CliUsage(
            cost_usd=float(data.get("total_cost_usd", 0.0)),
            tokens_in=int(usage.get("input_tokens", 0)),
            tokens_out=int(usage.get("output_tokens", 0)),
        )

    logger.debug("No result event found in CLI output (%d bytes)", len(stdout))
    return CliUsage()
```

Write this to `chaosengineer/execution/cli_usage.py`.

- [ ] **Step 4: Run tests — verify they pass**

Run: `python -m pytest tests/test_cli_usage.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/execution/cli_usage.py tests/test_cli_usage.py
git commit -m "feat: add CLI usage parser for subagent cost tracking"
```

---

## Chunk 2: Wire into SubagentExecutor

### Task 2: Add CLI flags and attach usage to results

**Files:**
- Modify: `chaosengineer/execution/subagent.py:114-129,156-159`
- Modify: `tests/test_subagent_executor.py`

- [ ] **Step 1: Update test mocks to include stdout with result event**

In `tests/test_subagent_executor.py`, add the import and a helper at the top of the file (after line 15):

```python
from chaosengineer.execution.cli_usage import CliUsage
```

Add a module-level helper (after `_git_success`):

```python
_RESULT_STDOUT = json.dumps({
    "type": "result",
    "subtype": "success",
    "total_cost_usd": 0.42,
    "usage": {"input_tokens": 5000, "output_tokens": 200},
})
```

Update `test_successful_experiment` (line 50) — change the `CompletedProcess` to include `stdout`:

```python
return subprocess.CompletedProcess(args=[], returncode=0, stdout=_RESULT_STDOUT)
```

Add a new test for cost extraction in `TestSubagentRunExperiment`:

```python
@patch("subprocess.run")
def test_successful_experiment_has_cost(self, mock_run, tmp_path):
    def dispatch(cmd, *args, **kwargs):
        if cmd[0] == "git":
            return subprocess.CompletedProcess(args=[], returncode=0)
        result_dir = tmp_path / "output" / "exp-0-0"
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "result.json").write_text(json.dumps({
            "primary_metric": 0.91,
        }))
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=_RESULT_STDOUT)

    mock_run.side_effect = dispatch

    spec = _make_spec()
    executor = SubagentExecutor(spec, tmp_path / "output", "sequential", run_id="run-test", repo_root=tmp_path)

    result = executor.run_experiment(
        experiment_id="exp-0-0",
        params={"lr": 0.02},
        command="python train.py",
        baseline_commit="abc123",
    )

    assert result.cost_usd == 0.42
    assert result.tokens_in == 5000
    assert result.tokens_out == 200
```

Add a test for cost on non-zero exit code:

```python
@patch("subprocess.run")
def test_failed_experiment_still_has_cost(self, mock_run, tmp_path):
    def dispatch(cmd, *args, **kwargs):
        if cmd[0] == "git":
            return subprocess.CompletedProcess(args=[], returncode=0)
        return subprocess.CompletedProcess(
            args=[], returncode=1, stderr="error", stdout=_RESULT_STDOUT,
        )

    mock_run.side_effect = dispatch

    spec = _make_spec()
    executor = SubagentExecutor(spec, tmp_path / "output", "sequential", run_id="run-test", repo_root=tmp_path)

    result = executor.run_experiment(
        experiment_id="exp-0-0",
        params={"lr": 0.02},
        command="python train.py",
        baseline_commit="abc123",
    )

    assert result.error_message is not None
    assert result.cost_usd == 0.42
```

Add a test for cost when subagent exits 0 but produces no result.json:

```python
@patch("subprocess.run")
def test_missing_result_file_still_has_cost(self, mock_run, tmp_path):
    """Subagent ran (incurred cost) but failed to write result.json."""
    def dispatch(cmd, *args, **kwargs):
        if cmd[0] == "git":
            return subprocess.CompletedProcess(args=[], returncode=0)
        # Don't write result.json — subagent failed to produce output
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=_RESULT_STDOUT)

    mock_run.side_effect = dispatch

    spec = _make_spec()
    executor = SubagentExecutor(spec, tmp_path / "output", "sequential", run_id="run-test", repo_root=tmp_path)

    result = executor.run_experiment(
        experiment_id="exp-0-0",
        params={"lr": 0.02},
        command="python train.py",
        baseline_commit="abc123",
    )

    assert result.error_message is not None  # ResultParser returns error
    assert result.cost_usd == 0.42  # But cost is still tracked
```

Note: The existing `test_successful_experiment` mock does not include `stdout`, so `invoke_result.stdout` will be `None`. This is safe — `parse_cli_usage(None)` returns `CliUsage()` with zeros. The existing test only asserts on `primary_metric`, `commit_hash`, `error_message`, and `duration_seconds`, so it passes unchanged. Updating it to include `stdout` is optional.

Note: The timeout path (`TimeoutExpired`) and the thread-crash path (`run_experiments` `except Exception`) have no `CompletedProcess` and therefore no stdout to parse. Cost remains 0 for these paths by design.

- [ ] **Step 2: Run tests — verify new tests fail**

Run: `python -m pytest tests/test_subagent_executor.py::TestSubagentRunExperiment::test_successful_experiment_has_cost tests/test_subagent_executor.py::TestSubagentRunExperiment::test_failed_experiment_still_has_cost tests/test_subagent_executor.py::TestSubagentRunExperiment::test_missing_result_file_still_has_cost -v`
Expected: FAIL — `cost_usd` is still 0.0

- [ ] **Step 3: Add CLI flags to _invoke()**

In `chaosengineer/execution/subagent.py`, add the import at line 15 (after the `ResultParser` import):

```python
from chaosengineer.execution.cli_usage import parse_cli_usage
```

Change the `cmd` list at lines 156-159 from:

```python
        cmd = [
            "claude", "-p", prompt,
            "--allowedTools", "Edit,Write,Bash,Read",
        ]
```

to:

```python
        cmd = [
            "claude", "-p", prompt,
            "--allowedTools", "Edit,Write,Bash,Read",
            "--output-format", "stream-json",
            "--verbose",
        ]
```

- [ ] **Step 4: Wire usage into _run_single() result paths**

In `chaosengineer/execution/subagent.py`, replace `_run_single()` lines 114-129 (from the non-zero exit code check through the result parse) with:

```python
            # 4. Check for non-zero exit code
            if invoke_result.returncode != 0:
                error_result = ExperimentResult(
                    primary_metric=0.0,
                    duration_seconds=duration,
                    error_message=(
                        f"Claude process failed for {task.experiment_id} "
                        f"(exit {invoke_result.returncode}): "
                        f"{(invoke_result.stderr or '')[:500]}"
                    ),
                )
                usage = parse_cli_usage(invoke_result.stdout)
                error_result.tokens_in = usage.tokens_in
                error_result.tokens_out = usage.tokens_out
                error_result.cost_usd = usage.cost_usd
                return error_result

            # 5. Parse result
            result = self._result_parser.parse(
                result_file, task.experiment_id, duration
            )
            usage = parse_cli_usage(invoke_result.stdout)
            result.tokens_in = usage.tokens_in
            result.tokens_out = usage.tokens_out
            result.cost_usd = usage.cost_usd
            return result
```

Note: `ExperimentResult` is NOT frozen, so direct attribute assignment works.

- [ ] **Step 5: Run all tests — verify they pass**

Run: `python -m pytest tests/test_subagent_executor.py tests/test_cli_usage.py -v`
Expected: All tests PASS

- [ ] **Step 6: Run full test suite to check for regressions**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add chaosengineer/execution/subagent.py tests/test_subagent_executor.py
git commit -m "feat: wire CLI usage parsing into SubagentExecutor"
```
