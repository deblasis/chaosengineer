"""Production interfaces for decision making and experiment execution.

These are ABCs that real implementations (ClaudeDecisionMaker, SubagentExecutor)
and test doubles (ScriptedDecisionMaker, ScriptedExecutor) both implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from chaosengineer.core.models import Baseline, DimensionSpec, ExperimentResult


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


class ExperimentExecutor(ABC):
    """Interface for running experiments.

    In real mode, this runs commands in worktrees. In test mode, returns
    scripted results instantly.
    """

    @abstractmethod
    def run_experiment(
        self,
        experiment_id: str,
        params: dict[str, Any],
        command: str,
        baseline_commit: str,
        resource: str = "",
    ) -> ExperimentResult:
        """Run an experiment and return its result."""
