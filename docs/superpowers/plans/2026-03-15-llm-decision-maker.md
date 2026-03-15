# LLM-Backed DecisionMaker Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pluggable LLM-backed DecisionMaker with ClaudeCode and SDK harness backends.

**Architecture:** `LLMDecisionMaker` implements the existing `DecisionMaker` ABC and delegates LLM transport to an `LLMHarness` ABC with two implementations: `ClaudeCodeHarness` (subprocess `claude -p`, default) and `SDKHarness` (Anthropic Python SDK). A factory function wires it up based on a `--llm-backend` CLI flag.

**Tech Stack:** Python 3.10+, `anthropic` SDK (optional dependency), `subprocess` for Claude Code, `pytest` for testing.

**Spec:** `docs/superpowers/specs/2026-03-15-llm-decision-maker-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `chaosengineer/llm/__init__.py` | Package init, `create_decision_maker` factory, exports |
| Create | `chaosengineer/llm/harness.py` | `LLMHarness` ABC, `extract_json` helper |
| Create | `chaosengineer/llm/claude_code.py` | `ClaudeCodeHarness` — subprocess transport |
| Create | `chaosengineer/llm/sdk.py` | `SDKHarness` — Anthropic SDK transport |
| Create | `chaosengineer/llm/decision_maker.py` | `LLMDecisionMaker`, prompt templates, response parsing |
| Create | `tests/test_harness.py` | Tests for LLMHarness ABC, `extract_json`, `ClaudeCodeHarness`, `SDKHarness` |
| Create | `tests/test_llm_decision_maker.py` | Tests for `LLMDecisionMaker` with `FakeHarness` |
| Modify | `chaosengineer/cli.py` | Add `run` subcommand with `--llm-backend` flag |
| Modify | `pyproject.toml` | Add `anthropic` as optional dependency under `[project.optional-dependencies]` |

---

## Chunk 1: LLMHarness ABC + JSON Extraction

### Task 1: LLMHarness ABC and extract_json helper

**Files:**
- Create: `chaosengineer/llm/__init__.py`
- Create: `chaosengineer/llm/harness.py`
- Create: `tests/test_harness.py`

- [ ] **Step 1: Create the llm package with empty init**

```python
# chaosengineer/llm/__init__.py
"""LLM harness abstraction for decision making."""
```

- [ ] **Step 2: Write failing tests for extract_json**

```python
# tests/test_harness.py
"""Tests for LLM harness utilities."""

import json
import pytest

from chaosengineer.llm.harness import extract_json


class TestExtractJson:
    def test_pure_json(self):
        raw = '{"dimension_name": "lr", "values": [{"lr": 0.02}]}'
        assert extract_json(raw) == {"dimension_name": "lr", "values": [{"lr": 0.02}]}

    def test_json_in_code_fence(self):
        raw = 'Here is my answer:\n```json\n{"done": true}\n```\n'
        assert extract_json(raw) == {"done": True}

    def test_json_with_surrounding_prose(self):
        raw = 'I think we should explore lr.\n{"dimension_name": "lr", "values": [{"lr": 0.1}]}\nGood luck!'
        assert extract_json(raw) == {"dimension_name": "lr", "values": [{"lr": 0.1}]}

    def test_nested_json(self):
        raw = '{"options": ["a", "b"], "meta": {"count": 2}}'
        result = extract_json(raw)
        assert result == {"options": ["a", "b"], "meta": {"count": 2}}

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            extract_json("no json here at all")

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            extract_json("{not valid json at all}")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && uv run pytest tests/test_harness.py -v`
Expected: ImportError — `chaosengineer.llm.harness` does not exist yet.

- [ ] **Step 4: Implement LLMHarness ABC and extract_json**

```python
# chaosengineer/llm/harness.py
"""LLM harness abstraction — transport layer for LLM calls."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Usage:
    """Token usage and cost from an LLM call."""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


class LLMHarness(ABC):
    """Abstract base for LLM transport.

    Implementations handle how to send a prompt and get a response.
    All harnesses must write the response JSON to output_file for audit.
    The caller guarantees the parent directory of output_file exists.
    """

    @abstractmethod
    def complete(self, system: str, user: str, output_file: Path) -> dict:
        """Send prompt to LLM, return parsed JSON dict."""

    @property
    def last_usage(self) -> Usage:
        """Token/cost data from the most recent call. Override in subclasses that track cost."""
        return Usage()


def extract_json(text: str) -> dict:
    """Extract the first JSON object from text that may contain prose or code fences.

    Raises ValueError if no valid JSON object is found.
    """
    # Try parsing the whole string first (fast path for pure JSON)
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Try extracting from code fences
    fence_match = re.search(r"```(?:json)?\s*\n(\{.*?\})\s*\n```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Find the first { and try to parse from there
    for match in re.finditer(r"\{", text):
        start = match.start()
        # Try increasingly larger substrings
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"No JSON object found in response: {text[:200]}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/alex/CODE/OSS/autoresearch && uv run pytest tests/test_harness.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add chaosengineer/llm/__init__.py chaosengineer/llm/harness.py tests/test_harness.py
git commit -m "feat: add LLMHarness ABC and extract_json helper"
```

---

## Chunk 2: ClaudeCodeHarness

### Task 2: ClaudeCodeHarness implementation

**Files:**
- Create: `chaosengineer/llm/claude_code.py`
- Modify: `tests/test_harness.py` (append)

- [ ] **Step 1: Write failing tests for ClaudeCodeHarness**

Append to `tests/test_harness.py`:

```python
from unittest.mock import patch, MagicMock
from chaosengineer.llm.claude_code import ClaudeCodeHarness


class TestClaudeCodeHarness:
    def test_happy_path(self, tmp_path):
        output_file = tmp_path / "decision_001.json"
        response_data = {"dimension_name": "lr", "values": [{"lr": 0.02}]}

        def fake_run(args, **kwargs):
            # Simulate Claude Code writing the output file
            output_file.write_text(json.dumps(response_data))
            return MagicMock(returncode=0, stdout="Done", stderr="")

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            harness = ClaudeCodeHarness()
            result = harness.complete(
                system="You are a coordinator.",
                user="Pick a dimension.",
                output_file=output_file,
            )

        assert result == response_data
        assert mock_run.called
        # Verify claude -p was called
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "claude"
        assert "-p" in call_args

    def test_model_override(self, tmp_path):
        output_file = tmp_path / "decision_001.json"
        response_data = {"done": True}

        def fake_run(args, **kwargs):
            output_file.write_text(json.dumps(response_data))
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            harness = ClaudeCodeHarness(model="claude-opus-4-6")
            harness.complete("sys", "usr", output_file)

        call_args = mock_run.call_args[0][0]
        assert "--model" in call_args
        assert "claude-opus-4-6" in call_args

    def test_missing_output_file_raises(self, tmp_path):
        output_file = tmp_path / "decision_001.json"
        # Claude Code runs but doesn't write the file

        with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="", stderr="")):
            harness = ClaudeCodeHarness()
            with pytest.raises(FileNotFoundError):
                harness.complete("sys", "usr", output_file)

    def test_nonzero_exit_raises(self, tmp_path):
        output_file = tmp_path / "decision_001.json"

        with patch("subprocess.run", return_value=MagicMock(returncode=1, stdout="", stderr="error")):
            harness = ClaudeCodeHarness()
            with pytest.raises(RuntimeError, match="Claude Code failed"):
                harness.complete("sys", "usr", output_file)

    def test_timeout_propagates(self, tmp_path):
        import subprocess as sp
        output_file = tmp_path / "decision_001.json"

        with patch("subprocess.run", side_effect=sp.TimeoutExpired(cmd="claude", timeout=300)):
            harness = ClaudeCodeHarness()
            with pytest.raises(sp.TimeoutExpired):
                harness.complete("sys", "usr", output_file)

    def test_last_usage_always_zero(self):
        harness = ClaudeCodeHarness()
        usage = harness.last_usage
        assert usage.cost_usd == 0.0
        assert usage.tokens_in == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && uv run pytest tests/test_harness.py::TestClaudeCodeHarness -v`
Expected: ImportError — `chaosengineer.llm.claude_code` does not exist yet.

- [ ] **Step 3: Implement ClaudeCodeHarness**

```python
# chaosengineer/llm/claude_code.py
"""ClaudeCodeHarness — runs claude -p as a subprocess."""

from __future__ import annotations

import subprocess
from pathlib import Path

from chaosengineer.llm.harness import LLMHarness, extract_json


class ClaudeCodeHarness(LLMHarness):
    """Sends prompts via Claude Code's -p flag.

    Default backend — uses the user's Claude Code subscription.
    No cost tracking (flat rate).
    """

    def __init__(self, model: str | None = None):
        self._model = model

    def complete(self, system: str, user: str, output_file: Path) -> dict:
        prompt = (
            f"{system}\n\n{user}\n\n"
            f"Write ONLY a JSON object to the file: {output_file}\n"
            f"Do not write any other text to the file — only valid JSON."
        )

        cmd = ["claude", "-p", prompt, "--allowedTools", "Write"]
        if self._model:
            cmd.extend(["--model", self._model])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            raise RuntimeError(
                f"Claude Code failed (exit {result.returncode}): {result.stderr[:500]}"
            )

        if not output_file.exists():
            raise FileNotFoundError(
                f"Claude Code did not write output file: {output_file}"
            )

        raw = output_file.read_text()
        return extract_json(raw)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alex/CODE/OSS/autoresearch && uv run pytest tests/test_harness.py::TestClaudeCodeHarness -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/llm/claude_code.py tests/test_harness.py
git commit -m "feat: add ClaudeCodeHarness (claude -p subprocess backend)"
```

---

## Chunk 3: SDKHarness

### Task 3: SDKHarness implementation

**Files:**
- Create: `chaosengineer/llm/sdk.py`
- Modify: `tests/test_harness.py` (append)
- Modify: `pyproject.toml` (add optional dep)

- [ ] **Step 1: Add anthropic as optional dependency**

In `pyproject.toml`, change the `[project.optional-dependencies]` section to:

```toml
[project.optional-dependencies]
test = ["pytest>=8.0"]
sdk = ["anthropic>=0.40.0"]
```

- [ ] **Step 2: Write failing tests for SDKHarness**

Append to `tests/test_harness.py`:

```python
from chaosengineer.llm.sdk import SDKHarness


class TestSDKHarness:
    def _mock_response(self, text, input_tokens=100, output_tokens=50):
        """Build a mock Anthropic API response."""
        content_block = MagicMock()
        content_block.text = text
        response = MagicMock()
        response.content = [content_block]
        response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
        return response

    def test_happy_path(self, tmp_path):
        output_file = tmp_path / "decision_001.json"
        response_json = '{"dimension_name": "lr", "values": [{"lr": 0.02}]}'
        mock_response = self._mock_response(response_json)

        with patch("chaosengineer.llm.sdk.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = mock_client

            harness = SDKHarness(api_key="test-key")
            result = harness.complete("system prompt", "user prompt", output_file)

        assert result == {"dimension_name": "lr", "values": [{"lr": 0.02}]}
        assert output_file.read_text()  # file was written for audit

    def test_env_vars(self, tmp_path):
        output_file = tmp_path / "decision_001.json"
        mock_response = self._mock_response('{"done": true}')

        with patch("chaosengineer.llm.sdk.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = mock_client

            with patch.dict("os.environ", {
                "ANTHROPIC_API_KEY": "env-key",
                "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
                "ANTHROPIC_MODEL": "glm-4.7",
            }):
                harness = SDKHarness()
                harness.complete("sys", "usr", output_file)

            # Check Anthropic client was created with env base_url
            mock_anthropic.Anthropic.assert_called_with(
                api_key="env-key",
                base_url="https://api.z.ai/api/anthropic",
            )
            # Check model from env was used
            create_call = mock_client.messages.create.call_args
            assert create_call.kwargs["model"] == "glm-4.7"

    def test_cost_tracking(self, tmp_path):
        output_file = tmp_path / "decision_001.json"
        mock_response = self._mock_response('{"done": true}', input_tokens=500, output_tokens=200)

        with patch("chaosengineer.llm.sdk.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = mock_client

            harness = SDKHarness(api_key="test-key")
            harness.complete("sys", "usr", output_file)

        usage = harness.last_usage
        assert usage.tokens_in == 500
        assert usage.tokens_out == 200
        assert usage.cost_usd > 0  # some estimated cost

    def test_no_api_key_raises(self, tmp_path):
        with patch("chaosengineer.llm.sdk.anthropic") as mock_anthropic:
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="No API key"):
                    SDKHarness()

    def test_api_error_propagates(self, tmp_path):
        output_file = tmp_path / "decision_001.json"

        with patch("chaosengineer.llm.sdk.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = Exception("API rate limited")
            mock_anthropic.Anthropic.return_value = mock_client

            harness = SDKHarness(api_key="test-key")
            with pytest.raises(Exception, match="API rate limited"):
                harness.complete("sys", "usr", output_file)

    def test_explicit_params_override_env(self, tmp_path):
        output_file = tmp_path / "decision_001.json"
        mock_response = self._mock_response('{"done": true}')

        with patch("chaosengineer.llm.sdk.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = mock_client

            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key", "ANTHROPIC_MODEL": "env-model"}):
                harness = SDKHarness(api_key="explicit-key", model="explicit-model")
                harness.complete("sys", "usr", output_file)

            mock_anthropic.Anthropic.assert_called_with(api_key="explicit-key", base_url=None)
            create_call = mock_client.messages.create.call_args
            assert create_call.kwargs["model"] == "explicit-model"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && uv run pytest tests/test_harness.py::TestSDKHarness -v`
Expected: ImportError — `chaosengineer.llm.sdk` does not exist yet.

- [ ] **Step 4: Implement SDKHarness**

```python
# chaosengineer/llm/sdk.py
"""SDKHarness — uses Anthropic Python SDK for LLM calls."""

from __future__ import annotations

import json
import os
from pathlib import Path

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

from chaosengineer.llm.harness import LLMHarness, Usage, extract_json


# Rough cost per token for estimation (Claude Sonnet 4 pricing).
# Users on alternative providers may have different rates — this is best-effort.
_INPUT_COST_PER_TOKEN = 3.0 / 1_000_000   # $3 per 1M input tokens
_OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000  # $15 per 1M output tokens


class SDKHarness(LLMHarness):
    """Sends prompts via the Anthropic Python SDK.

    Reads configuration from constructor args or environment variables:
    - ANTHROPIC_API_KEY
    - ANTHROPIC_BASE_URL (for alternative providers like Z.AI, OpenRouter)
    - ANTHROPIC_MODEL (default: claude-sonnet-4-20250514)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        if anthropic is None:
            raise ImportError(
                "The anthropic package is required for SDKHarness. "
                "Install it with: uv pip install 'chaosengineer[sdk]'"
            )

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No API key provided. Pass api_key= or set ANTHROPIC_API_KEY."
            )
        resolved_base = base_url or os.environ.get("ANTHROPIC_BASE_URL") or None

        self._client = anthropic.Anthropic(
            api_key=resolved_key,
            base_url=resolved_base,
        )
        self._model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        self._last_usage = Usage()

    @property
    def last_usage(self) -> Usage:
        return self._last_usage

    def complete(self, system: str, user: str, output_file: Path) -> dict:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        text = response.content[0].text
        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        cost = (tokens_in * _INPUT_COST_PER_TOKEN) + (tokens_out * _OUTPUT_COST_PER_TOKEN)

        self._last_usage = Usage(
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
        )

        parsed = extract_json(text)

        # Write to output_file for audit trail
        output_file.write_text(json.dumps(parsed, indent=2))

        return parsed
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/alex/CODE/OSS/autoresearch && uv run pytest tests/test_harness.py::TestSDKHarness -v`
Expected: All 6 tests PASS.

- [ ] **Step 6: Run all harness tests together**

Run: `cd /Users/alex/CODE/OSS/autoresearch && uv run pytest tests/test_harness.py -v`
Expected: All 18 tests PASS (6 extract_json + 6 ClaudeCode + 6 SDK).

- [ ] **Step 7: Commit**

```bash
git add chaosengineer/llm/sdk.py tests/test_harness.py pyproject.toml
git commit -m "feat: add SDKHarness (Anthropic SDK backend with alt-provider support)"
```

---

## Chunk 4: LLMDecisionMaker

### Task 4: LLMDecisionMaker with prompt templates and response parsing

**Files:**
- Create: `chaosengineer/llm/decision_maker.py`
- Create: `tests/test_llm_decision_maker.py`

- [ ] **Step 1: Write failing tests for pick_next_dimension**

```python
# tests/test_llm_decision_maker.py
"""Tests for LLMDecisionMaker — prompt construction and response parsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from chaosengineer.core.interfaces import DimensionPlan
from chaosengineer.core.models import (
    Baseline,
    BudgetConfig,
    DimensionSpec,
    DimensionType,
)
from chaosengineer.llm.decision_maker import LLMDecisionMaker
from chaosengineer.llm.harness import LLMHarness, Usage
from chaosengineer.workloads.parser import WorkloadSpec


class FakeHarness(LLMHarness):
    """Returns canned responses for testing."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self._call_index = 0
        self.calls: list[dict[str, Any]] = []  # record calls for assertion

    def complete(self, system: str, user: str, output_file: Path) -> dict:
        self.calls.append({"system": system, "user": user, "output_file": output_file})
        if self._call_index >= len(self._responses):
            raise RuntimeError("FakeHarness exhausted")
        resp = self._responses[self._call_index]
        self._call_index += 1
        output_file.write_text(json.dumps(resp))
        return resp


def _make_spec(**overrides) -> WorkloadSpec:
    defaults = dict(
        name="test-workload",
        primary_metric="val_bpb",
        metric_direction="lower",
        execution_command="echo test",
        context="Training a small language model.",
        budget=BudgetConfig(max_experiments=100),
    )
    defaults.update(overrides)
    return WorkloadSpec(**defaults)


def _make_dimensions() -> list[DimensionSpec]:
    return [
        DimensionSpec(name="learning_rate", dim_type=DimensionType.DIRECTIONAL, current_value=0.04),
        DimensionSpec(name="activation", dim_type=DimensionType.ENUM, options=["GeLU", "SiLU", "ReLU"]),
        DimensionSpec(name="attention_design", dim_type=DimensionType.DIVERSE),
    ]


def _make_baselines() -> list[Baseline]:
    return [Baseline(commit="abc1234", metric_value=0.97, metric_name="val_bpb")]


class TestPickNextDimension:
    def test_returns_dimension_plan(self, tmp_path):
        harness = FakeHarness([
            {"dimension_name": "learning_rate", "values": [{"learning_rate": 0.02}, {"learning_rate": 0.08}]}
        ])
        dm = LLMDecisionMaker(harness, _make_spec(), tmp_path)
        plan = dm.pick_next_dimension(_make_dimensions(), _make_baselines(), [])

        assert isinstance(plan, DimensionPlan)
        assert plan.dimension_name == "learning_rate"
        assert len(plan.values) == 2

    def test_done_signal_returns_none(self, tmp_path):
        harness = FakeHarness([{"done": True}])
        dm = LLMDecisionMaker(harness, _make_spec(), tmp_path)
        plan = dm.pick_next_dimension(_make_dimensions(), _make_baselines(), [])
        assert plan is None

    def test_unknown_dimension_raises(self, tmp_path):
        harness = FakeHarness([
            {"dimension_name": "nonexistent", "values": [{"x": 1}]}
        ])
        dm = LLMDecisionMaker(harness, _make_spec(), tmp_path)
        with pytest.raises(ValueError, match="Unknown dimension"):
            dm.pick_next_dimension(_make_dimensions(), _make_baselines(), [])

    def test_empty_values_raises(self, tmp_path):
        harness = FakeHarness([
            {"dimension_name": "learning_rate", "values": []}
        ])
        dm = LLMDecisionMaker(harness, _make_spec(), tmp_path)
        with pytest.raises(ValueError, match="empty"):
            dm.pick_next_dimension(_make_dimensions(), _make_baselines(), [])

    def test_prompt_contains_dimensions(self, tmp_path):
        harness = FakeHarness([{"done": True}])
        dm = LLMDecisionMaker(harness, _make_spec(), tmp_path)
        dm.pick_next_dimension(_make_dimensions(), _make_baselines(), [])

        user_prompt = harness.calls[0]["user"]
        assert "learning_rate" in user_prompt
        assert "activation" in user_prompt
        assert "0.97" in user_prompt  # baseline metric

    def test_prompt_contains_history(self, tmp_path):
        harness = FakeHarness([{"done": True}])
        dm = LLMDecisionMaker(harness, _make_spec(), tmp_path)
        history = [
            {"event": "breakthrough", "new_best": 0.93, "from_experiment": "exp-0-0"},
        ]
        dm.pick_next_dimension(_make_dimensions(), _make_baselines(), history)

        user_prompt = harness.calls[0]["user"]
        assert "breakthrough" in user_prompt
        assert "0.93" in user_prompt

    def test_sequential_calls_use_incrementing_filenames(self, tmp_path):
        harness = FakeHarness([{"done": True}, {"done": True}])
        dm = LLMDecisionMaker(harness, _make_spec(), tmp_path)
        dm.pick_next_dimension(_make_dimensions(), _make_baselines(), [])
        dm.pick_next_dimension(_make_dimensions(), _make_baselines(), [])

        files = [c["output_file"].name for c in harness.calls]
        assert files == ["decision_001.json", "decision_002.json"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && uv run pytest tests/test_llm_decision_maker.py::TestPickNextDimension -v`
Expected: ImportError — `chaosengineer.llm.decision_maker` does not exist yet.

- [ ] **Step 3: Implement LLMDecisionMaker with pick_next_dimension**

```python
# chaosengineer/llm/decision_maker.py
"""LLMDecisionMaker — real LLM-backed implementation of DecisionMaker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chaosengineer.core.interfaces import DecisionMaker, DimensionPlan
from chaosengineer.core.models import Baseline, DimensionSpec
from chaosengineer.llm.harness import LLMHarness
from chaosengineer.workloads.parser import WorkloadSpec

PICK_SYSTEM_PROMPT = """\
You are a coordinator for an automated experimentation framework. Your job is to \
analyze the experiment space and history, then decide which dimension to explore next.

The framework uses coordinate descent: one dimension is explored per iteration, with \
multiple values tested in parallel from the same baseline. You pick the dimension that \
is most likely to yield improvement given what has been tried so far.

Respond with ONLY a JSON object — no other text. Either:
- {"dimension_name": "<name>", "values": [{"<param>": <val>}, ...]} to explore a dimension
- {"done": true} if no dimensions remain worth exploring"""

DISCOVER_SYSTEM_PROMPT = """\
You are generating maximally diverse options for an experiment dimension. Your goal is \
to produce a saturated set: options that are genuinely different from each other, covering \
the full space of reasonable approaches.

Think through what options exist. For each, check: is it truly distinct from the others, \
or just a variation? Keep only those that represent fundamentally different approaches. \
Stop when you cannot think of a genuinely novel option.

Respond with ONLY a JSON object — no other text:
{"options": ["option1", "option2", ...], "saturated": true}"""


class LLMDecisionMaker(DecisionMaker):
    """Real LLM-backed decision maker.

    Owns prompt construction, response parsing, and validation.
    Delegates LLM transport to an LLMHarness.
    """

    def __init__(self, harness: LLMHarness, spec: WorkloadSpec, work_dir: Path):
        self.harness = harness
        self.spec = spec
        self.work_dir = work_dir
        self._call_count = 0

    @property
    def last_cost_usd(self) -> float:
        """Cost of the most recent LLM call. 0.0 for ClaudeCode harness."""
        return self.harness.last_usage.cost_usd

    def _next_output_file(self) -> Path:
        self._call_count += 1
        return self.work_dir / f"decision_{self._call_count:03d}.json"

    def pick_next_dimension(
        self,
        dimensions: list[DimensionSpec],
        baselines: list[Baseline],
        history: list[dict],
    ) -> DimensionPlan | None:
        user_prompt = self._build_pick_prompt(dimensions, baselines, history)
        output_file = self._next_output_file()

        response = self.harness.complete(PICK_SYSTEM_PROMPT, user_prompt, output_file)

        if response.get("done"):
            return None

        return self._validate_pick_response(response, dimensions)

    def discover_diverse_options(
        self, dimension_name: str, context: str
    ) -> list[str]:
        user_prompt = (
            f"Dimension: {dimension_name}\n\n"
            f"Context:\n{context}\n\n"
            f"Workload: {self.spec.name}\n"
            f"{self.spec.context}"
        )
        output_file = self._next_output_file()

        response = self.harness.complete(DISCOVER_SYSTEM_PROMPT, user_prompt, output_file)

        options = response.get("options", [])
        if not options:
            raise ValueError(f"LLM returned no options for dimension '{dimension_name}'")
        return options

    def _build_pick_prompt(
        self,
        dimensions: list[DimensionSpec],
        baselines: list[Baseline],
        history: list[dict],
    ) -> str:
        parts = []

        parts.append(f"Workload: {self.spec.name}")
        if self.spec.context:
            parts.append(f"Context: {self.spec.context}")
        parts.append(f"Metric: {self.spec.primary_metric} ({self.spec.metric_direction} is better)")
        parts.append("")

        parts.append("## Available Dimensions")
        for d in dimensions:
            line = f"- {d.name} (type: {d.dim_type.value})"
            if d.current_value is not None:
                line += f", current: {d.current_value}"
            if d.options:
                line += f", options: {d.options}"
            if d.description:
                line += f" — {d.description}"
            parts.append(line)
        parts.append("")

        parts.append("## Active Baselines")
        for b in baselines:
            parts.append(f"- {b.metric_name}={b.metric_value} (commit: {b.commit})")
        parts.append("")

        if history:
            parts.append("## Experiment History")
            parts.append(json.dumps(history, indent=2, default=str))

        return "\n".join(parts)

    def _validate_pick_response(
        self, response: dict, dimensions: list[DimensionSpec]
    ) -> DimensionPlan:
        dim_name = response.get("dimension_name")
        values = response.get("values")

        known_names = {d.name for d in dimensions}
        if dim_name not in known_names:
            raise ValueError(
                f"Unknown dimension '{dim_name}' in LLM response. "
                f"Known dimensions: {known_names}"
            )

        if not values:
            raise ValueError(
                f"LLM returned empty values for dimension '{dim_name}'"
            )

        return DimensionPlan(dimension_name=dim_name, values=values)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alex/CODE/OSS/autoresearch && uv run pytest tests/test_llm_decision_maker.py::TestPickNextDimension -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Write failing tests for discover_diverse_options**

Append to `tests/test_llm_decision_maker.py`:

```python
class TestDiscoverDiverseOptions:
    def test_returns_options_list(self, tmp_path):
        harness = FakeHarness([
            {"options": ["chain-of-thought", "few-shot", "zero-shot"], "saturated": True}
        ])
        dm = LLMDecisionMaker(harness, _make_spec(), tmp_path)
        options = dm.discover_diverse_options("prompt_strategy", "LLM evaluation task")

        assert options == ["chain-of-thought", "few-shot", "zero-shot"]

    def test_empty_options_raises(self, tmp_path):
        harness = FakeHarness([{"options": [], "saturated": True}])
        dm = LLMDecisionMaker(harness, _make_spec(), tmp_path)
        with pytest.raises(ValueError, match="no options"):
            dm.discover_diverse_options("prompt_strategy", "context")

    def test_prompt_contains_dimension_and_context(self, tmp_path):
        harness = FakeHarness([
            {"options": ["a"], "saturated": True}
        ])
        dm = LLMDecisionMaker(harness, _make_spec(), tmp_path)
        dm.discover_diverse_options("attention_design", "Transformer attention mechanisms")

        user_prompt = harness.calls[0]["user"]
        assert "attention_design" in user_prompt
        assert "Transformer attention mechanisms" in user_prompt


class TestLastCostUsd:
    def test_delegates_to_harness(self, tmp_path):
        harness = FakeHarness([{"done": True}])
        dm = LLMDecisionMaker(harness, _make_spec(), tmp_path)
        # FakeHarness inherits default last_usage (all zeros)
        assert dm.last_cost_usd == 0.0
```

- [ ] **Step 6: Run all decision maker tests**

Run: `cd /Users/alex/CODE/OSS/autoresearch && uv run pytest tests/test_llm_decision_maker.py -v`
Expected: All 11 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add chaosengineer/llm/decision_maker.py tests/test_llm_decision_maker.py
git commit -m "feat: add LLMDecisionMaker with prompt templates and validation"
```

---

## Chunk 5: Factory, CLI, and Integration

### Task 5: Factory function and package exports

**Files:**
- Modify: `chaosengineer/llm/__init__.py`
- Append: `tests/test_llm_decision_maker.py`

- [ ] **Step 1: Write failing test for create_decision_maker**

Append to `tests/test_llm_decision_maker.py`:

```python
from unittest.mock import patch, MagicMock
from chaosengineer.llm import create_decision_maker


class TestFactory:
    def test_claude_code_backend(self, tmp_path):
        dm = create_decision_maker("claude-code", _make_spec(), tmp_path)
        assert isinstance(dm, LLMDecisionMaker)
        from chaosengineer.llm.claude_code import ClaudeCodeHarness
        assert isinstance(dm.harness, ClaudeCodeHarness)

    def test_sdk_backend(self, tmp_path):
        with patch("chaosengineer.llm.sdk.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = MagicMock()
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
                dm = create_decision_maker("sdk", _make_spec(), tmp_path)
        assert isinstance(dm, LLMDecisionMaker)
        from chaosengineer.llm.sdk import SDKHarness
        assert isinstance(dm.harness, SDKHarness)

    def test_unknown_backend_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown backend"):
            create_decision_maker("openai", _make_spec(), tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/alex/CODE/OSS/autoresearch && uv run pytest tests/test_llm_decision_maker.py::TestFactory -v`
Expected: ImportError — `create_decision_maker` not in `chaosengineer.llm`.

- [ ] **Step 3: Implement factory function**

```python
# chaosengineer/llm/__init__.py
"""LLM harness abstraction for decision making."""

from __future__ import annotations

from pathlib import Path

from chaosengineer.llm.decision_maker import LLMDecisionMaker
from chaosengineer.llm.harness import LLMHarness, Usage
from chaosengineer.workloads.parser import WorkloadSpec

__all__ = ["LLMDecisionMaker", "LLMHarness", "Usage", "create_decision_maker"]


def create_decision_maker(
    backend: str,
    spec: WorkloadSpec,
    work_dir: Path,
) -> LLMDecisionMaker:
    """Create an LLMDecisionMaker with the specified backend.

    Args:
        backend: "claude-code" (default, uses subscription) or "sdk" (uses API key)
        spec: Workload specification for prompt context
        work_dir: Directory for LLM output files (must exist)
    """
    if backend == "claude-code":
        from chaosengineer.llm.claude_code import ClaudeCodeHarness
        harness = ClaudeCodeHarness()
    elif backend == "sdk":
        from chaosengineer.llm.sdk import SDKHarness
        harness = SDKHarness()
    else:
        raise ValueError(f"Unknown backend: '{backend}'. Use 'claude-code' or 'sdk'.")

    return LLMDecisionMaker(harness, spec, work_dir)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/alex/CODE/OSS/autoresearch && uv run pytest tests/test_llm_decision_maker.py::TestFactory -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/llm/__init__.py tests/test_llm_decision_maker.py
git commit -m "feat: add create_decision_maker factory with backend selection"
```

### Task 6: Add --llm-backend CLI flag

**Files:**
- Modify: `chaosengineer/cli.py:12-63`

- [ ] **Step 1: Add run subcommand with --llm-backend flag**

In `chaosengineer/cli.py`, after the test_parser block (line 31) and before the version subparser (line 34), add:

```python
    # Run command: execute a workload
    run_parser = subparsers.add_parser("run", help="Run a workload")
    run_parser.add_argument(
        "workload",
        type=Path,
        help="Path to workload spec markdown file",
    )
    run_parser.add_argument(
        "--llm-backend",
        choices=["claude-code", "sdk"],
        default="claude-code",
        help="LLM backend for coordinator decisions (default: claude-code)",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".chaosengineer/output"),
        help="Directory for run output",
    )
```

In the command dispatch section, after the `elif args.command == "test":` block (before `else:`), add:

```python
    elif args.command == "run":
        from chaosengineer.workloads.parser import parse_workload_spec
        from chaosengineer.llm import create_decision_maker

        args.output_dir.mkdir(parents=True, exist_ok=True)
        spec = parse_workload_spec(args.workload)

        llm_dir = args.output_dir / "llm_decisions"
        llm_dir.mkdir(parents=True, exist_ok=True)

        dm = create_decision_maker(args.llm_backend, spec, llm_dir)
        print(f"Created {args.llm_backend} decision maker for workload: {spec.name}")
        print("(Full coordinator integration is Sub-project C)")
```

- [ ] **Step 2: Verify CLI parses the flag**

Run: `cd /Users/alex/CODE/OSS/autoresearch && uv run chaosengineer run --help`
Expected: Shows `--llm-backend` option with `claude-code` and `sdk` choices.

- [ ] **Step 3: Commit**

```bash
git add chaosengineer/cli.py
git commit -m "feat: add 'run' subcommand with --llm-backend flag"
```

---

## Chunk 6: Final Verification

### Task 7: Full test suite and existing test regression check

- [ ] **Step 1: Run all new tests**

Run: `cd /Users/alex/CODE/OSS/autoresearch && uv run pytest tests/test_harness.py tests/test_llm_decision_maker.py -v`
Expected: All tests PASS.

- [ ] **Step 2: Run existing tests to verify no regressions**

Run: `cd /Users/alex/CODE/OSS/autoresearch && uv run pytest tests/ -v`
Expected: All 93 existing tests + new tests PASS. Zero failures.

- [ ] **Step 3: Verify shipped scenarios still pass**

Run: `cd /Users/alex/CODE/OSS/autoresearch && uv run chaosengineer test`
Expected: All 3 shipped scenarios PASS.

- [ ] **Step 4: Final commit if any cleanup needed**

Only if previous steps revealed issues that needed fixing.
