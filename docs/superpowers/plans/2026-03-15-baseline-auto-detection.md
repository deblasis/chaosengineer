# Baseline Auto-Detection Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded worst-possible baseline in `_execute_run` with a three-tier resolution: CLI flag > workload spec > auto-detect.

**Architecture:** Add `baseline_metric_value` field to `WorkloadSpec` with parser support for a `## Baseline` section. Add `detect_baseline(spec)` function to `cli.py` that runs the workload's execution + parse commands via subprocess. Wire resolution logic in `_execute_run` with `--initial-baseline` CLI flag as override.

**Tech Stack:** Python 3.10+, subprocess, pytest, argparse

---

## Chunk 1: Parser Extension

### Task 1: Parse baseline from workload spec

**Files:**
- Modify: `chaosengineer/workloads/parser.py:12-29` (add field to `WorkloadSpec`)
- Modify: `chaosengineer/workloads/parser.py:201-251` (parse `## Baseline` section)
- Test: `tests/test_parser.py`

- [ ] **Step 1: Write failing test — baseline parsed from spec**

In `tests/test_parser.py`, add to `TestParseWorkloadSpec`:

```python
def test_parse_baseline_metric_value(self):
    md = """# Workload: Test
## Experiment Space
## Execution
- Command: `echo`
## Evaluation
- Metric: score (lower is better)
## Baseline
- Metric value: 2.08
## Resources
- Available: 1
"""
    spec = parse_workload_spec(content=md)
    assert spec.baseline_metric_value == pytest.approx(2.08)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_parser.py::TestParseWorkloadSpec::test_parse_baseline_metric_value -v`
Expected: FAIL — `WorkloadSpec` has no `baseline_metric_value` attribute

- [ ] **Step 3: Write failing test — absent baseline is None**

```python
def test_absent_baseline_is_none(self):
    md = """# Workload: Test
## Experiment Space
## Execution
- Command: `echo`
## Evaluation
- Metric: score (lower is better)
## Resources
- Available: 1
"""
    spec = parse_workload_spec(content=md)
    assert spec.baseline_metric_value is None
```

- [ ] **Step 4: Write failing test — negative baseline value**

```python
def test_parse_negative_baseline_metric_value(self):
    md = """# Workload: Test
## Experiment Space
## Execution
- Command: `echo`
## Evaluation
- Metric: loglik (higher is better)
## Baseline
- Metric value: -1.23
## Resources
- Available: 1
"""
    spec = parse_workload_spec(content=md)
    assert spec.baseline_metric_value == pytest.approx(-1.23)
```

- [ ] **Step 5: Write failing test — scientific notation baseline**

```python
def test_parse_scientific_notation_baseline(self):
    md = """# Workload: Test
## Experiment Space
## Execution
- Command: `echo`
## Evaluation
- Metric: loss (lower is better)
## Baseline
- Metric value: 1.5e-3
## Resources
- Available: 1
"""
    spec = parse_workload_spec(content=md)
    assert spec.baseline_metric_value == pytest.approx(0.0015)
```

- [ ] **Step 6: Implement parser changes**

In `chaosengineer/workloads/parser.py`, add field to `WorkloadSpec` (after line 28):

```python
baseline_metric_value: float | None = None
```

Add parser helper (after `_parse_modifiable_files`):

```python
def _parse_baseline_metric(text: str) -> float | None:
    match = re.search(
        r"Metric\s+value:\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)", text, re.IGNORECASE
    )
    if not match:
        return None
    return float(match.group(1))
```

In `parse_workload_spec`, add after the constraints parsing (before the `return`):

```python
# Baseline
baseline_section = sections.get("Baseline", "")
baseline_value = _parse_baseline_metric(baseline_section)
```

Add `baseline_metric_value=baseline_value` to the returned `WorkloadSpec`.

- [ ] **Step 7: Run all parser tests**

Run: `pytest tests/test_parser.py -v`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add chaosengineer/workloads/parser.py tests/test_parser.py
git commit -m "feat: parse optional baseline metric value from workload spec"
```

---

## Chunk 2: CLI Baseline Resolution

### Task 2: Add `--initial-baseline` CLI flag

**Files:**
- Modify: `chaosengineer/cli.py:33-73` (add argparse argument)
- Test: `tests/test_cli_run.py`

- [ ] **Step 1: Write failing test — argparse accepts `--initial-baseline`**

In `tests/test_cli_run.py`, add to `TestCliRunArgs`:

```python
def test_initial_baseline_flag_accepted(self):
    """--initial-baseline is accepted as a float argument."""
    import argparse
    with patch("sys.argv", [
        "chaosengineer", "run", "workload.md",
        "--initial-baseline", "2.08",
    ]):
        with patch("argparse.ArgumentParser.parse_args") as mock_parse:
            mock_parse.return_value = argparse.Namespace(
                command="run", workload=Path("workload.md"),
                executor="subagent", mode="sequential",
                llm_backend="claude-code", scripted_results=None,
                scripted_plans=None, initial_baseline=2.08,
                output_dir=Path(".chaosengineer/output"),
            )
            with patch("chaosengineer.cli._execute_run"):
                main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli_run.py::TestCliRunArgs::test_initial_baseline_flag_accepted -v`
Expected: FAIL (argparse doesn't know `--initial-baseline`)

- [ ] **Step 3: Add `--initial-baseline` to argparse**

In `chaosengineer/cli.py`, add after the `--output-dir` argument (after line 73):

```python
run_parser.add_argument(
    "--initial-baseline",
    type=float,
    default=None,
    help="Override initial baseline metric value",
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli_run.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/cli.py tests/test_cli_run.py
git commit -m "feat: add --initial-baseline CLI flag"
```

### Task 3: Implement `detect_baseline` and resolution logic

**Files:**
- Modify: `chaosengineer/cli.py:110-157` (`_execute_run` and new `detect_baseline`)
- Test: `tests/test_cli_scripted.py`

- [ ] **Step 1: Write failing test — `detect_baseline` runs execution and parse commands**

In `tests/test_cli_scripted.py`, add a new test class:

```python
from unittest.mock import patch, MagicMock
from chaosengineer.workloads.parser import WorkloadSpec


class TestDetectBaseline:
    """Tests for detect_baseline subprocess auto-detection."""

    def _make_spec(self):
        return WorkloadSpec(
            name="test",
            execution_command="echo test",
            primary_metric="val_bpb",
            metric_direction="lower",
            metric_parse_command="echo 2.08",
        )

    def test_detect_baseline_runs_commands(self):
        from chaosengineer.cli import detect_baseline

        spec = self._make_spec()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="2.08\n")
            result = detect_baseline(spec)

        assert result == pytest.approx(2.08)
        assert mock_run.call_count == 2  # execution + parse

    def test_detect_baseline_exits_on_execution_failure(self):
        from chaosengineer.cli import detect_baseline

        spec = self._make_spec()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="error")
            with pytest.raises(SystemExit):
                detect_baseline(spec)

    def test_detect_baseline_exits_on_unparseable_metric(self):
        from chaosengineer.cli import detect_baseline

        spec = self._make_spec()
        with patch("subprocess.run") as mock_run:
            # First call (execution) succeeds, second call (parse) returns garbage
            mock_run.side_effect = [
                MagicMock(returncode=0),
                MagicMock(returncode=0, stdout="not-a-number\n"),
            ]
            with pytest.raises(SystemExit):
                detect_baseline(spec)

    def test_detect_baseline_exits_on_parse_failure(self):
        from chaosengineer.cli import detect_baseline

        spec = self._make_spec()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0),
                MagicMock(returncode=1, stderr="parse error"),
            ]
            with pytest.raises(SystemExit):
                detect_baseline(spec)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli_scripted.py::TestDetectBaseline -v`
Expected: FAIL — `detect_baseline` doesn't exist

- [ ] **Step 3: Implement `detect_baseline`**

In `chaosengineer/cli.py`, add before `_execute_run`:

```python
def detect_baseline(spec) -> float:
    """Run the workload once to measure the initial baseline metric."""
    import subprocess

    print(f"No baseline specified. Running workload to measure initial baseline...")

    result = subprocess.run(
        spec.execution_command, shell=True, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error: baseline execution failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    result = subprocess.run(
        spec.metric_parse_command, shell=True, capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Error: baseline metric parse failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    try:
        value = float(result.stdout.strip())
    except ValueError:
        print(
            f"Error: could not parse baseline metric from output: {result.stdout.strip()!r}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Initial baseline: {spec.primary_metric} = {value}")
    return value
```

- [ ] **Step 4: Run `detect_baseline` tests to verify they pass**

Run: `pytest tests/test_cli_scripted.py::TestDetectBaseline -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/cli.py tests/test_cli_scripted.py
git commit -m "feat: add detect_baseline for auto-detecting initial metric"
```

### Task 4: Wire resolution logic in `_execute_run`

**Files:**
- Modify: `chaosengineer/cli.py:110-157` (replace hardcoded baseline)
- Test: `tests/test_cli_scripted.py`

- [ ] **Step 1: Write failing test — resolution uses CLI flag over spec**

In `tests/test_cli_scripted.py`, add to `TestScriptedBackend`:

```python
def test_cli_baseline_overrides_spec(self, tmp_path):
    """--initial-baseline flag takes priority over spec baseline."""
    workload, plans, results = self._write_fixtures(tmp_path)
    # Add baseline section to workload
    content = workload.read_text()
    workload.write_text(content + "\n## Baseline\n- Metric value: 99.0\n")

    args = FakeArgs(tmp_path, plans, results, workload)
    args.initial_baseline = 5.0  # CLI override

    _execute_run(args)

    events_file = args.output_dir / "events.jsonl"
    events = [json.loads(line) for line in events_file.read_text().splitlines()]
    run_started = [e for e in events if e["event"] == "run_started"][0]
    assert run_started["baseline"]["metric_value"] == 5.0
```

- [ ] **Step 2: Write failing test — resolution uses spec baseline**

```python
def test_spec_baseline_used_when_no_cli_flag(self, tmp_path):
    """Workload spec baseline used when no --initial-baseline flag."""
    workload, plans, results = self._write_fixtures(tmp_path)
    content = workload.read_text()
    workload.write_text(content + "\n## Baseline\n- Metric value: 3.14\n")

    args = FakeArgs(tmp_path, plans, results, workload)
    args.initial_baseline = None

    _execute_run(args)

    events_file = args.output_dir / "events.jsonl"
    events = [json.loads(line) for line in events_file.read_text().splitlines()]
    run_started = [e for e in events if e["event"] == "run_started"][0]
    assert run_started["baseline"]["metric_value"] == pytest.approx(3.14)
```

- [ ] **Step 3: Write failing test — scripted executor errors without baseline**

(Test code is in Step 5 below — it needs the updated `_write_fixtures` to work correctly.)

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_cli_scripted.py::TestScriptedBackend::test_cli_baseline_overrides_spec tests/test_cli_scripted.py::TestScriptedBackend::test_spec_baseline_used_when_no_cli_flag tests/test_cli_scripted.py::TestScriptedBackend::test_scripted_executor_requires_baseline -v`
Expected: FAIL — `FakeArgs` has no `initial_baseline`, resolution logic not wired

- [ ] **Step 5: Update `FakeArgs` and `_write_fixtures` for baseline support**

In `tests/test_cli_scripted.py`, add to `FakeArgs.__init__`:

```python
self.initial_baseline = None
```

Also update `_write_fixtures` to include a `## Baseline` section in the workload markdown, so existing tests don't hit the scripted executor guard:

```python
workload.write_text(dedent("""\
    # Workload: Test

    ## Context
    Test workload.

    ## Experiment Space
    - Directional: "lr" (currently 0.04)

    ## Execution
    - Command: `echo test`

    ## Evaluation
    - Type: automatic
    - Metric: val_bpb (lower is better)

    ## Baseline
    - Metric value: 1.0

    ## Budget
    - Max experiments: 2
"""))
```

Update `test_scripted_executor_requires_baseline` to use a workload **without** a baseline (write a separate fixture inline):

```python
def test_scripted_executor_requires_baseline(self, tmp_path):
    """--executor=scripted without baseline in spec or CLI flag should error."""
    workload, plans, results = self._write_fixtures(tmp_path)
    # Overwrite workload without baseline section
    workload.write_text(dedent("""\
        # Workload: Test

        ## Context
        Test workload.

        ## Experiment Space
        - Directional: "lr" (currently 0.04)

        ## Execution
        - Command: `echo test`

        ## Evaluation
        - Type: automatic
        - Metric: val_bpb (lower is better)

        ## Budget
        - Max experiments: 2
    """))
    args = FakeArgs(tmp_path, plans, results, workload)
    args.initial_baseline = None

    with pytest.raises(SystemExit):
        _execute_run(args)
```

- [ ] **Step 6: Replace baseline logic in `_execute_run`**

In `chaosengineer/cli.py`, replace lines 152-157 (the TODO and hardcoded baseline) with:

```python
# Resolve initial baseline: CLI flag > workload spec > auto-detect
if args.initial_baseline is not None:
    metric_value = args.initial_baseline
elif spec.baseline_metric_value is not None:
    metric_value = spec.baseline_metric_value
elif args.executor == "scripted":
    print(
        "Error: --executor=scripted requires a baseline. "
        "Use --initial-baseline or add ## Baseline to the workload spec.",
        file=sys.stderr,
    )
    sys.exit(1)
else:
    metric_value = detect_baseline(spec)

initial_baseline = Baseline(
    commit="HEAD",
    metric_value=metric_value,
    metric_name=spec.primary_metric,
)
```

- [ ] **Step 7: Run all CLI scripted tests**

Run: `pytest tests/test_cli_scripted.py -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add chaosengineer/cli.py tests/test_cli_scripted.py
git commit -m "feat: wire three-tier baseline resolution in _execute_run"
```

---

## Chunk 3: Workload Spec Update & E2E Fix

### Task 5: Add baseline to Irish music workload and fix E2E test

**Files:**
- Modify: `workloads/autoresearch-irish-music.md` (add `## Baseline` section)
- Modify: `tests/e2e/test_irish_music_pipeline.py:122-145` (fix breakthrough count)

- [ ] **Step 1: Add `## Baseline` section to Irish music workload**

In `workloads/autoresearch-irish-music.md`, add after the `## Evaluation` section (before `## Resources`):

```markdown
## Baseline
- Metric value: 2.08
```

- [ ] **Step 2: Update `test_run_via_cli_scripted` — add `initial_baseline` to Args and assert 6 breakthroughs**

In `tests/e2e/test_irish_music_pipeline.py`, replace the `test_run_via_cli_scripted` method (lines 122-145) with:

```python
def test_run_via_cli_scripted(self, tmp_path):
    """Verify the full CLI path works for scripted runs.

    The workload spec includes a baseline of 2.08, so the CLI path
    now produces the same 6 breakthroughs as the direct coordinator tests.
    """
    from chaosengineer.cli import _execute_run

    class Args:
        workload = WORKLOADS_DIR / "autoresearch-irish-music.md"
        llm_backend = "scripted"
        scripted_plans = IRISH_MUSIC_DIR / "scripted_plans.yaml"
        executor = "scripted"
        scripted_results = IRISH_MUSIC_DIR / "scripted_results.yaml"
        mode = "sequential"
        output_dir = tmp_path / "output"
        initial_baseline = None

    _execute_run(Args())

    events_file = Args.output_dir / "events.jsonl"
    assert events_file.exists()

    import json
    events = [json.loads(line) for line in events_file.read_text().splitlines()]
    breakthroughs = [e for e in events if e["event"] == "breakthrough"]
    assert len(breakthroughs) == 6
```

- [ ] **Step 3: Run E2E tests**

Run: `pytest tests/e2e/test_irish_music_pipeline.py -v`
Expected: All 7 tests PASS (including the fixed `test_run_via_cli_scripted`)

- [ ] **Step 4: Run full test suite**

Run: `pytest -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add workloads/autoresearch-irish-music.md tests/e2e/test_irish_music_pipeline.py
git commit -m "feat: add baseline to Irish music spec and fix E2E breakthrough count"
```
