"""Decision maker interface and scripted implementation for testing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from chaosengineer.core.models import Baseline, DimensionSpec


@dataclass
class DimensionPlan:
    """A plan to explore one dimension."""
    dimension_name: str
    values: list[dict[str, Any]]  # one dict per worker


class DecisionMaker(ABC):
    """Interface for making experiment planning decisions.

    In real mode, this calls an LLM. In test mode, returns scripted responses.
    """

    @abstractmethod
    def pick_next_dimension(
        self,
        dimensions: list[DimensionSpec],
        baselines: list[Baseline],
        history: list[dict],
    ) -> DimensionPlan | None:
        """Pick the next dimension to explore. Returns None if done."""

    @abstractmethod
    def discover_diverse_options(
        self, dimension_name: str, context: str
    ) -> list[str]:
        """Discover the saturated set for a diverse dimension."""


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
