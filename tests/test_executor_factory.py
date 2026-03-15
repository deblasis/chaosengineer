"""Tests for executor factory."""

import json
import pytest
from pathlib import Path

import yaml

from chaosengineer.execution import create_executor
from chaosengineer.execution.subagent import SubagentExecutor
from chaosengineer.testing.executor import ScriptedExecutor
from chaosengineer.core.models import ExperimentResult
from chaosengineer.workloads.parser import WorkloadSpec


def _make_spec() -> WorkloadSpec:
    return WorkloadSpec(
        name="test",
        primary_metric="val_bpb",
        metric_direction="lower",
        execution_command="echo test",
    )


class TestCreateExecutor:
    def test_subagent_backend(self, tmp_path):
        executor = create_executor("subagent", _make_spec(), tmp_path, "sequential")
        assert isinstance(executor, SubagentExecutor)

    def test_scripted_from_file(self, tmp_path):
        results_file = tmp_path / "results.yaml"
        results_file.write_text(yaml.dump({
            "exp-0-0": {"primary_metric": 0.91},
            "exp-0-1": {"primary_metric": 0.95},
        }))

        executor = create_executor(
            "scripted", _make_spec(), tmp_path, "sequential",
            scripted_results=results_file,
        )
        assert isinstance(executor, ScriptedExecutor)

        result = executor.run_experiment("exp-0-0", {}, "", "")
        assert result.primary_metric == 0.91

    def test_scripted_from_folder(self, tmp_path):
        folder = tmp_path / "results"
        folder.mkdir()
        (folder / "batch1.yaml").write_text(yaml.dump({
            "exp-0-0": {"primary_metric": 0.91},
        }))
        (folder / "batch2.yaml").write_text(yaml.dump({
            "exp-0-1": {"primary_metric": 0.95},
        }))

        executor = create_executor(
            "scripted", _make_spec(), tmp_path, "sequential",
            scripted_results=folder,
        )
        assert isinstance(executor, ScriptedExecutor)

        r0 = executor.run_experiment("exp-0-0", {}, "", "")
        r1 = executor.run_experiment("exp-0-1", {}, "", "")
        assert r0.primary_metric == 0.91
        assert r1.primary_metric == 0.95

    def test_scripted_requires_path(self):
        with pytest.raises(ValueError, match="scripted_results"):
            create_executor("scripted", _make_spec(), Path("/tmp"), "sequential")

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            create_executor("unknown", _make_spec(), Path("/tmp"), "sequential")
