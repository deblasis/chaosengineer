"""Tests for production interfaces in core.interfaces."""

import pytest
from chaosengineer.core.interfaces import (
    DecisionMaker,
    DimensionPlan,
    ExperimentExecutor,
)
from chaosengineer.core.models import ExperimentResult


class TestDimensionPlan:
    def test_creation(self):
        plan = DimensionPlan(
            dimension_name="lr",
            values=[{"lr": 0.02}, {"lr": 0.08}],
        )
        assert plan.dimension_name == "lr"
        assert len(plan.values) == 2

    def test_empty_values(self):
        plan = DimensionPlan(dimension_name="lr", values=[])
        assert plan.values == []


class TestDecisionMakerIsAbstract:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            DecisionMaker()


class TestExperimentExecutorIsAbstract:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            ExperimentExecutor()
