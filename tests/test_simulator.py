"""Tests for decision maker and executor interfaces."""

import pytest
from chaosengineer.core.models import DimensionType, DimensionSpec, ExperimentResult
from chaosengineer.core.interfaces import (
    DecisionMaker,
    DimensionPlan,
    ExperimentExecutor,
)
from chaosengineer.testing.simulator import ScriptedDecisionMaker
from chaosengineer.testing.executor import ScriptedExecutor


class TestScriptedDecisionMaker:
    def test_returns_scripted_dimensions(self):
        plans = [
            DimensionPlan(
                dimension_name="lr",
                values=[{"lr": 0.02}, {"lr": 0.08}],
            ),
            DimensionPlan(
                dimension_name="depth",
                values=[{"depth": 6}, {"depth": 12}],
            ),
        ]
        dm = ScriptedDecisionMaker(plans)
        plan = dm.pick_next_dimension(
            dimensions=[], baselines=[], history=[]
        )
        assert plan.dimension_name == "lr"
        assert len(plan.values) == 2

    def test_exhausted_returns_none(self):
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}]),
        ]
        dm = ScriptedDecisionMaker(plans)
        dm.pick_next_dimension([], [], [])
        result = dm.pick_next_dimension([], [], [])
        assert result is None

    def test_pick_diverse_options(self):
        dm = ScriptedDecisionMaker(
            plans=[],
            diverse_options={"strategy": ["A", "B", "C"]},
        )
        options = dm.discover_diverse_options("strategy", context="")
        assert options == ["A", "B", "C"]


class TestSetPriorContext:
    def test_scripted_decision_maker_accepts_prior_context(self):
        dm = ScriptedDecisionMaker(plans=[])
        dm.set_prior_context("Prior state: explored lr, bs. Baseline: 2.41")
        assert dm.pick_next_dimension([], [], []) is None


class TestScriptedExecutor:
    def test_returns_scripted_result(self):
        results = {
            "exp-001": ExperimentResult(primary_metric=0.93, duration_seconds=300),
            "exp-002": ExperimentResult(primary_metric=0.95, duration_seconds=300),
        }
        executor = ScriptedExecutor(results)
        result = executor.run_experiment(
            experiment_id="exp-001",
            params={"lr": 0.02},
            command="echo",
            baseline_commit="abc",
        )
        assert result.primary_metric == 0.93

    def test_missing_experiment_raises(self):
        executor = ScriptedExecutor({})
        with pytest.raises(KeyError):
            executor.run_experiment("unknown", {}, "echo", "abc")
