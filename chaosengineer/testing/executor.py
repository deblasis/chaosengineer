"""Scripted experiment executor for testing."""

from __future__ import annotations

from typing import Any

from chaosengineer.core.interfaces import ExperimentExecutor
from chaosengineer.core.models import ExperimentResult


class ScriptedExecutor(ExperimentExecutor):
    """Returns pre-scripted results for testing."""

    def __init__(self, results: dict[str, ExperimentResult]):
        self._results = results

    def run_experiment(
        self,
        experiment_id: str,
        params: dict[str, Any],
        command: str,
        baseline_commit: str,
        resource: str = "",
    ) -> ExperimentResult:
        if experiment_id not in self._results:
            raise KeyError(f"No scripted result for experiment {experiment_id}")
        return self._results[experiment_id]
