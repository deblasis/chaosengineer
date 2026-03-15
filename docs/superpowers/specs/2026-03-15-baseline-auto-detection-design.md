# Design: Initial Baseline Auto-Detection

## Problem

`_execute_run` in `cli.py` hardcodes the initial baseline to the worst possible value (`inf` or `-inf`), meaning the very first experiment is always a "breakthrough." The E2E test documents this: the CLI path produces 7 breakthroughs instead of 6 because experiment 1 (metric 2.15) beats `inf`, whereas the correct baseline of 2.08 would reject it.

## Solution

Three-tier baseline resolution: CLI flag > workload spec > auto-detect.

## Workload Spec Extension

Add an optional `## Baseline` section to the workload markdown:

```markdown
## Baseline
- Metric value: 2.08
```

Commit defaults to `HEAD`. If the section is omitted, auto-detection kicks in.

In the parser, add one field to `WorkloadSpec`:

```python
baseline_metric_value: float | None = None
```

Just the value — the parser doesn't depend on core models.

## Auto-Detection

When no baseline is in the spec and no `--initial-baseline` CLI flag is given, `_execute_run` runs the workload to measure the real baseline:

1. Print: `"No baseline specified. Running workload to measure initial baseline..."`
2. Run `spec.execution_command` via subprocess in the current working directory
3. Run `spec.metric_parse_command` via subprocess to extract the metric value
4. Print: `"Initial baseline: {metric_name} = {value}"`
5. Use that value with `commit="HEAD"` as the `Baseline`

This logic lives in a `detect_baseline(spec) -> float` function in `cli.py` (~15 lines).

If the execution or parse command fails, print an error and exit. No silent fallback to inf.

## Resolution Order & CLI Override

Priority:

1. `--initial-baseline 2.08` CLI flag (highest)
2. Workload spec `## Baseline` section
3. Auto-detect by running the workload

In `_execute_run`:

```python
if args.initial_baseline is not None:
    metric_value = args.initial_baseline
elif spec.baseline_metric_value is not None:
    metric_value = spec.baseline_metric_value
else:
    metric_value = detect_baseline(spec)

initial_baseline = Baseline(
    commit="HEAD",
    metric_value=metric_value,
    metric_name=spec.primary_metric,
)
```

The existing `float("inf")` / `float("-inf")` fallback is removed entirely.

## Testing

1. **Parser tests** — `baseline_metric_value` parsed from spec markdown, `None` when absent
2. **Resolution logic tests** — CLI flag wins over spec, spec wins over auto-detect, auto-detect runs when both absent
3. **`detect_baseline` tests** — mock subprocess calls, verify metric extraction, verify exit on failure
4. **E2E fix** — update Irish music pipeline workload fixture to include `## Baseline` with `2.08`; test should now produce 6 breakthroughs matching the direct coordinator test
