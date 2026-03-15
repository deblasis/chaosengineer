"""Tests for LLMDecisionMaker — prompt construction and response parsing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from unittest.mock import patch, MagicMock

from chaosengineer.core.interfaces import DimensionPlan
from chaosengineer.core.models import (
    Baseline,
    BudgetConfig,
    DimensionSpec,
    DimensionType,
)
from chaosengineer.llm import create_decision_maker
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


class TestSetPriorContext:
    def test_prior_context_stored(self, tmp_path):
        from unittest.mock import MagicMock
        from chaosengineer.llm.decision_maker import LLMDecisionMaker
        from chaosengineer.workloads.parser import WorkloadSpec
        from chaosengineer.core.models import BudgetConfig
        harness = MagicMock()
        spec = WorkloadSpec(name="test", primary_metric="loss", metric_direction="lower",
                            execution_command="echo", workers_available=1, budget=BudgetConfig(max_experiments=1))
        dm = LLMDecisionMaker(harness, spec, tmp_path)
        dm.set_prior_context("Explored: lr, bs. Best: 2.41")
        assert dm._prior_context == "Explored: lr, bs. Best: 2.41"

    def test_prior_context_prepended_to_prompt(self, tmp_path):
        harness = FakeHarness([{"done": True}])
        dm = LLMDecisionMaker(harness, _make_spec(), tmp_path)
        dm.set_prior_context("Prior state: explored lr. Best: 0.93")
        dm.pick_next_dimension(_make_dimensions(), _make_baselines(), [])

        user_prompt = harness.calls[0]["user"]
        assert user_prompt.startswith("Prior state: explored lr. Best: 0.93")

    def test_no_prior_context_prompt_unchanged(self, tmp_path):
        harness = FakeHarness([{"done": True}])
        dm = LLMDecisionMaker(harness, _make_spec(), tmp_path)
        dm.pick_next_dimension(_make_dimensions(), _make_baselines(), [])

        user_prompt = harness.calls[0]["user"]
        assert user_prompt.startswith("Workload:")


class TestDecisionLoggerWiring:
    def test_dimension_selected_logged(self, tmp_path):
        """DecisionLogger.log_dimension_selected called after pick_next_dimension."""
        from chaosengineer.core.decision_log import DecisionLogger
        import json

        harness = FakeHarness([
            {"dimension_name": "learning_rate", "values": [{"learning_rate": 0.02}]}
        ])
        logger = DecisionLogger(tmp_path)
        dm = LLMDecisionMaker(harness, _make_spec(), tmp_path, decision_logger=logger)
        dm.pick_next_dimension(_make_dimensions(), _make_baselines(), [])

        entries = []
        with open(tmp_path / "decisions.jsonl") as f:
            for line in f:
                entries.append(json.loads(line))
        assert len(entries) == 1
        assert entries[0]["type"] == "dimension_selected"
        assert entries[0]["dimension"] == "learning_rate"
        assert "activation" in entries[0]["alternatives"]

    def test_diverse_options_logged(self, tmp_path):
        """DecisionLogger.log_diverse_options called after discover_diverse_options."""
        from chaosengineer.core.decision_log import DecisionLogger
        import json

        harness = FakeHarness([
            {"options": ["chain-of-thought", "few-shot"], "saturated": True}
        ])
        logger = DecisionLogger(tmp_path)
        dm = LLMDecisionMaker(harness, _make_spec(), tmp_path, decision_logger=logger)
        dm.discover_diverse_options("prompt_strategy", "LLM eval")

        entries = []
        with open(tmp_path / "decisions.jsonl") as f:
            for line in f:
                entries.append(json.loads(line))
        assert len(entries) == 1
        assert entries[0]["type"] == "diverse_options_generated"
        assert entries[0]["options"] == ["chain-of-thought", "few-shot"]

    def test_no_logger_no_error(self, tmp_path):
        """No decision logger still works (default None)."""
        harness = FakeHarness([
            {"dimension_name": "learning_rate", "values": [{"learning_rate": 0.02}]}
        ])
        dm = LLMDecisionMaker(harness, _make_spec(), tmp_path)
        plan = dm.pick_next_dimension(_make_dimensions(), _make_baselines(), [])
        assert plan is not None


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
