"""Scripted decision maker for testing."""

from __future__ import annotations

from typing import Any

from chaosengineer.core.interfaces import DecisionMaker, DimensionPlan
from chaosengineer.core.models import Baseline, DimensionSpec


class ScriptedDecisionMaker(DecisionMaker):
    """Returns pre-scripted decisions for testing."""

    def __init__(
        self,
        plans: list[DimensionPlan],
        diverse_options: dict[str, list[str]] | None = None,
    ):
        self._plans = list(plans)
        self._plan_index = 0
        self._diverse_options = diverse_options or {}

    def pick_next_dimension(
        self,
        dimensions: list[DimensionSpec],
        baselines: list[Baseline],
        history: list[dict],
    ) -> DimensionPlan | None:
        if self._plan_index >= len(self._plans):
            return None
        plan = self._plans[self._plan_index]
        self._plan_index += 1
        return plan

    def discover_diverse_options(
        self, dimension_name: str, context: str
    ) -> list[str]:
        return self._diverse_options.get(dimension_name, [])
