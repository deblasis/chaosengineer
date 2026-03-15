# LLM-Backed DecisionMaker Design Spec

**Project:** ChaosEngineer Phase 2, Sub-project B
**Date:** 2026-03-15
**Status:** Approved
**Depends on:** Phase 2A (complete)

## Goal

Build `LLMDecisionMaker`, a real implementation of the `DecisionMaker` ABC that calls an LLM to make experiment planning decisions. The LLM harness is pluggable — defaulting to Claude Code (`claude -p`) so subscription users pay nothing extra, with an SDK backend for API-key users and alternative providers.

## Architecture

```
DecisionMaker (ABC)                    # core/interfaces.py (exists)
├── ScriptedDecisionMaker              # testing/simulator.py (exists)
└── LLMDecisionMaker                   # llm/decision_maker.py (new)
    └── uses LLMHarness (ABC)          # llm/harness.py (new)
        ├── ClaudeCodeHarness          # llm/claude_code.py (new)
        └── SDKHarness                 # llm/sdk.py (new)
```

### LLMHarness ABC

```python
class LLMHarness(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, output_file: Path) -> dict:
        """Send prompt to LLM, return parsed JSON dict."""
```

Minimal interface: system message, user message, output file path. Returns parsed JSON. All harnesses must write the response JSON to `output_file` — this serves as an audit trail and debugging aid. The caller guarantees the parent directory exists. Each backend handles transport differently but the contract is the same.

### ClaudeCodeHarness (default)

- Constructs a combined prompt from system + user messages
- Appends instruction: "Write your JSON answer to {output_file}"
- Runs `claude -p "..."` via `subprocess.run`
- Reads and parses `output_file`
- Accepts optional `model` parameter to pass `--model` flag; by default inherits the user's Claude Code configuration
- No cost tracking — decision-making is "free" under subscription

### SDKHarness

- Uses `anthropic.Anthropic()` client
- Reads configuration from constructor args or environment variables:
  - `ANTHROPIC_API_KEY` — API key
  - `ANTHROPIC_BASE_URL` — base URL for alternative providers (Z.AI, OpenRouter, etc.)
  - `ANTHROPIC_MODEL` — model override (default: `claude-sonnet-4-20250514`)
- Calls `client.messages.create()` with system/user messages
- Parses JSON from response text, writes to `output_file` for logging/debugging
- Exposes `last_usage` property with `tokens_in`, `tokens_out`, `cost_usd` from API response
- Enables alternative Anthropic-compatible providers with zero code changes

### LLMDecisionMaker

```python
class LLMDecisionMaker(DecisionMaker):
    def __init__(self, harness: LLMHarness, spec: WorkloadSpec, work_dir: Path):
        self.harness = harness
        self.spec = spec          # workload context for prompts
        self.work_dir = work_dir  # temp dir for output files
```

Owns all prompt construction, response parsing, and validation. The harness is a transport layer only. The stored `spec` provides richer workload context for prompts (domain description, constraints, metric names) beyond the `dimensions` list already passed via method args. `work_dir` is created by the caller (factory function) and cleaned up when the run completes. Output files accumulate here — one per LLM call — named sequentially (`decision_001.json`, `decision_002.json`, etc.) for post-hoc debugging.

#### pick_next_dimension(dimensions, baselines, history) -> DimensionPlan | None

1. Builds system prompt: coordinator role, experiment framework context
2. Builds user prompt serializing: available dimensions (name, type, current_value, options, description), active baselines (metric_value, metric_name, commit), condensed history (past iterations, results, breakthroughs)
3. Instructs LLM to output JSON: `{"dimension_name": "...", "values": [{"param": val}, ...]}` or `{"done": true}` when no dimensions remain worth exploring
4. Calls `harness.complete(system, user, output_file)`
5. Validates: dimension_name must exist in dimensions list, values must be non-empty list of dicts
6. Returns `DimensionPlan`, or `None` if LLM returns `{"done": true}`

#### discover_diverse_options(dimension_name, context) -> list[str]

1. Builds system prompt: generate maximally diverse, non-overlapping options
2. Builds user prompt: dimension name, workload context, saturation instruction
3. Instructs output as: `{"options": ["opt1", "opt2", ...], "saturated": true}`
4. Single LLM call — saturation loop runs inside LLM reasoning (cheaper than multi-round). This is a v1 simplification of the parent spec's multi-round generate/critique protocol. Multi-round can be added later if single-call quality is insufficient.
5. Returns options list

#### Prompt templates

Module-level string constants in `decision_maker.py`. Colocated with parsing logic — no external template files.

#### JSON extraction

LLM responses may wrap JSON in markdown code fences or add explanatory prose. Both harnesses extract the first `{...}` block from the response text before parsing. This handles the common case of ````json\n{...}\n```` wrapping.

#### Error handling

Invalid JSON or failed validation (unknown dimension, empty values) raises a descriptive error. No retries at this layer for v1.

## Configuration

**CLI flag:**

```
--llm-backend claude-code|sdk   (default: claude-code)
```

Operational choice about how to run, not a property of the workload.

**Factory function:**

```python
# chaosengineer/llm/__init__.py

def create_decision_maker(
    backend: str,           # "claude-code" or "sdk"
    spec: WorkloadSpec,
    work_dir: Path,
) -> LLMDecisionMaker:
```

Instantiates the correct harness and returns a wired `LLMDecisionMaker`.

## Cost Tracking

- **Claude Code backend:** No cost tracking for coordinator reasoning (subscription = flat rate). Budget tracker only tracks experiment execution costs. Per the parent spec, discovery phase token costs "count toward the run's API budget" — this requirement is only satisfiable with the SDK backend; with Claude Code, discovery is effectively free.
- **SDK backend:** `SDKHarness.last_usage` exposes per-call token/cost data. `LLMDecisionMaker` exposes a `last_cost_usd` property (delegates to `harness.last_usage.cost_usd`, returns `0.0` for Claude Code harness). The coordinator reads this after each `pick_next_dimension` / `discover_diverse_options` call and feeds it to `BudgetTracker.add_cost()`.

## File Layout

```
chaosengineer/llm/
    __init__.py              # create_decision_maker factory, exports
    harness.py               # LLMHarness ABC
    claude_code.py           # ClaudeCodeHarness
    sdk.py                   # SDKHarness
    decision_maker.py        # LLMDecisionMaker + prompt templates

tests/
    test_llm_decision_maker.py   # prompt construction, response parsing/validation
    test_harness.py              # extract_json, ClaudeCodeHarness, SDKHarness tests
```

## Testing Strategy

- **test_llm_decision_maker.py**: `FakeHarness` (in-memory, returns canned dicts). Tests prompt construction from dimensions/baselines/history, valid responses → correct `DimensionPlan`, invalid responses raise errors, "no more dimensions" → `None`.
- **test_harness_claude_code.py**: Mocks `subprocess.run`. Writes temp JSON to simulate Claude Code output. Tests happy path and errors (missing file, invalid JSON, non-zero exit).
- **test_harness_sdk.py**: Mocks `anthropic.Anthropic().messages.create()`. Verifies system/user messages passed correctly. Tests cost/token tracking. Tests API errors.
- **Existing tests untouched.** `ScriptedDecisionMaker` and all coordinator tests remain as-is.
- **No real LLM tests** — integration testing with real backends is manual.

## Integration

No changes to the coordinator. It already takes `DecisionMaker` via constructor injection. The wiring happens upstream — CLI runner or test setup picks the right implementation.
