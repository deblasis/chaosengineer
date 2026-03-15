"""Experiment execution backends."""

from __future__ import annotations

from pathlib import Path

import yaml

from chaosengineer.core.interfaces import ExperimentExecutor
from chaosengineer.core.models import ExperimentResult
from chaosengineer.workloads.parser import WorkloadSpec


def create_executor(
    backend: str,
    spec: WorkloadSpec,
    output_dir: Path,
    mode: str = "sequential",
    scripted_results: Path | None = None,
    run_id: str = "run-unknown",
) -> ExperimentExecutor:
    """Create an executor with the specified backend.

    Args:
        backend: "subagent" (default) or "scripted" (for testing/demos)
        spec: Workload specification
        output_dir: Directory for experiment output
        mode: "sequential" or "parallel"
        scripted_results: Path to YAML file or folder (required for scripted)
        run_id: Run identifier (used by subagent for branch naming)
    """
    if backend == "subagent":
        from chaosengineer.execution.subagent import SubagentExecutor
        return SubagentExecutor(spec, output_dir, mode, run_id=run_id)

    elif backend == "scripted":
        if scripted_results is None:
            raise ValueError(
                "scripted_results path is required when using --executor=scripted"
            )
        results = _load_scripted_results(scripted_results)
        from chaosengineer.testing.executor import ScriptedExecutor
        return ScriptedExecutor(results)

    else:
        raise ValueError(
            f"Unknown executor backend: '{backend}'. Use 'subagent' or 'scripted'."
        )


def _load_scripted_results(path: Path) -> dict[str, ExperimentResult]:
    """Load scripted results from a YAML file or folder of YAML files."""
    if path.is_dir():
        merged: dict[str, dict] = {}
        for yaml_file in sorted(path.glob("*.yaml")):
            data = yaml.safe_load(yaml_file.read_text())
            if data:
                merged.update(data)
        raw = merged
    else:
        raw = yaml.safe_load(path.read_text()) or {}

    results = {}
    for exp_id, data in raw.items():
        results[exp_id] = ExperimentResult(
            primary_metric=data["primary_metric"],
            secondary_metrics=data.get("secondary_metrics", {}),
            artifacts=data.get("artifacts", []),
            commit_hash=data.get("commit_hash"),
            duration_seconds=data.get("duration_seconds", 0),
            error_message=data.get("error_message"),
        )
    return results
