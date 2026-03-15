# Design: Diverse Dimension Discovery E2E

## Problem

`discover_diverse_options` is implemented in `LLMDecisionMaker` and `ScriptedDecisionMaker`, but the coordinator never calls it. DIVERSE dimensions are parsed from workload specs with `options=None`, and the LLM never gets a chance to discover the saturated set before experiments begin. The feature is wired but untested end-to-end.

## Solution

Wire discovery into the coordinator at run start, then write an E2E test proving the full flow.

## Coordinator Discovery Phase

Add `_discover_diverse_dimensions()` to `Coordinator`, called at the start of `run()` before the main loop:

```python
def _discover_diverse_dimensions(self):
    for dim in self.spec.dimensions:
        if dim.dim_type == DimensionType.DIVERSE and dim.options is None:
            options = self.decision_maker.discover_diverse_options(
                dim.name, self.spec.context,
            )
            dim.options = options
            self.logger.log_event(
                "diverse_discovered",
                dimension=dim.name,
                options=options,
                count=len(options),
            )
```

Mutates `dim.options` in-place so the LLM sees discovered options in all subsequent `pick_next_dimension` prompts (the prompt builder already includes `d.options`).

## Event Ordering

```
diverse_discovered (for each DIVERSE dim with options=None)
run_started
iteration_started (loop begins)
...
```

Discovery is cheap (text generation, no experiments). No budget tracking on discovery calls for now — cost tracking is a separate follow-up item. If there are no DIVERSE dimensions, the method is a no-op.

## Unit Tests

Add to existing coordinator tests:

1. **Discovers options for DIVERSE dims** — options=None becomes populated
2. **Skips non-DIVERSE dims** — DIRECTIONAL and ENUM untouched
3. **Skips DIVERSE dims that already have options** — no re-discovery
4. **No-op when no DIVERSE dims** — no events logged

All use `ScriptedDecisionMaker` with `diverse_options`.

## E2E Test

New file `tests/e2e/test_diverse_discovery.py` with `TestDiverseDimensionDiscovery`:

- Inline workload spec: 2 DIRECTIONAL dims + 1 DIVERSE dim, 5-iteration budget
- `ScriptedDecisionMaker` with `diverse_options={"optimizer": ["adam", "sgd", "rmsprop"]}`
- 5 scripted plans (mix of directional and diverse dimension exploration)
- 5 scripted results with breakthroughs

Assertions:
1. `diverse_discovered` event logged with 3 options
2. The DIVERSE dimension's `options` field is populated after the run
3. All 5 experiments run
4. Correct breakthrough count
5. Self-contained — no external fixture files
