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
