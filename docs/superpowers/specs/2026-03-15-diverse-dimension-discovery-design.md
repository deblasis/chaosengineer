# Design: Diverse Dimension Discovery E2E

## Problem

`discover_diverse_options` is implemented in `LLMDecisionMaker` and `ScriptedDecisionMaker`, but the coordinator never calls it. DIVERSE dimensions are parsed from workload specs with `options=None`, and the LLM never gets a chance to discover the saturated set before experiments begin. The feature is wired but untested end-to-end.

## Solution

Wire discovery into the coordinator at run start, then write an E2E test proving the full flow.

## Coordinator Discovery Phase

Add `_discover_diverse_dimensions()` to `Coordinator`, called in `run()` after `run_started` and `budget.start()` but before the main loop:

```python
def _discover_diverse_dimensions(self):
    for dim in self.spec.dimensions:
        if dim.dim_type == DimensionType.DIVERSE and dim.options is None:
            try:
                options = self.decision_maker.discover_diverse_options(
                    dim.name, self.spec.context,
                )
            except (ValueError, Exception) as e:
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

Mutates `dim.options` in-place so the LLM sees discovered options in all subsequent `pick_next_dimension` prompts (the prompt builder already includes `d.options`). Uses `self._log(Event(...))` matching the coordinator's existing logging pattern.

On failure (exception or empty list), logs a `diverse_discovery_failed` event and skips that dimension — leaves `options` as `None`. The LLM can still pick the dimension but won't have pre-discovered options.

Note: this wires the existing single-call `discover_diverse_options` implementation. The parent design spec describes a multi-round generate/critique/saturate protocol — that is deferred. The `LLMDecisionMaker` currently does a single LLM call that asks for saturation in one shot, which is a reasonable v1 simplification.

## Event Ordering

```
run_started
diverse_discovered (for each DIVERSE dim with options=None)
iteration_started (loop begins)
...
```

Discovery happens after `run_started` and `budget.start()` so it falls within the run's timing window, consistent with the parent spec's note that discovery tokens count toward the API budget. Discovery is cheap (text generation, no experiments). If there are no DIVERSE dimensions, the method is a no-op.

## Unit Tests

Add to existing coordinator tests:

1. **Discovers options for DIVERSE dims** — options=None becomes populated
2. **Skips non-DIVERSE dims** — DIRECTIONAL and ENUM untouched
3. **Skips DIVERSE dims that already have options** — no re-discovery
4. **No-op when no DIVERSE dims** — no `diverse_discovered` events logged
5. **Logs failure on empty options** — `diverse_discovery_failed` event, `dim.options` stays `None`
6. **Logs failure on exception** — `diverse_discovery_failed` event, continues to next dim

All use `ScriptedDecisionMaker` with `diverse_options`.

## E2E Test

New file `tests/e2e/test_diverse_discovery.py` with `TestDiverseDimensionDiscovery`:

- Inline workload spec: 2 DIRECTIONAL dims + 1 DIVERSE dim, 5-iteration budget
- `ScriptedDecisionMaker` with `diverse_options={"optimizer": ["adam", "sgd", "rmsprop"]}`
- 5 scripted plans (mix of directional and diverse dimension exploration)
- 5 scripted results with breakthroughs

Assertions:
1. `diverse_discovered` event logged with 3 options, appearing after `run_started` and before first `iteration_started`
2. The DIVERSE dimension's `options` field is populated after the run
3. All 5 experiments run
4. Correct breakthrough count
5. Self-contained — no external fixture files
