"""Production interfaces for decision making and experiment execution.

These are ABCs that real implementations (ClaudeDecisionMaker, SubagentExecutor)
and test doubles (ScriptedDecisionMaker, ScriptedExecutor) both implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

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

    def set_prior_context(self, context: str) -> None:
        """Provide factual summary of prior run state for resume. Default no-op."""
        pass


@dataclass
class ExperimentTask:
    """Input packet for a single experiment."""
    experiment_id: str
    params: dict[str, Any]
    command: str
    baseline_commit: str
    resource: str = ""


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

    def run_experiments(
        self,
        tasks: list[ExperimentTask],
        on_worker_done: "Callable[[ExperimentTask, ExperimentResult, int, int], None] | None" = None,
    ) -> list[ExperimentResult]:
        """Run a batch of experiments. Default: sequential.

        Postcondition: always returns exactly one ExperimentResult per input task.
        Failures are captured as ExperimentResult with error_message set, never raised.

        Args:
            on_worker_done: Optional callback(task, result, completed_count, total_count)
                called after each experiment completes.
        """
        results = []
        total = len(tasks)
        for i, t in enumerate(tasks):
            try:
                result = self.run_experiment(
                    t.experiment_id, t.params, t.command, t.baseline_commit, t.resource
                )
            except Exception as e:
                result = ExperimentResult(
                    primary_metric=0.0,
                    error_message=f"Executor error for {t.experiment_id}: {e}",
                )
            results.append(result)
            if on_worker_done is not None:
                on_worker_done(t, result, i + 1, total)
        return results

    def kill_active(self) -> None:
        """Kill any in-flight experiments. Default: no-op."""
        pass
