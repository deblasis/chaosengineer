"""Tests for production interfaces in core.interfaces."""

import pytest
from chaosengineer.core.interfaces import (
    DecisionMaker,
    DimensionPlan,
    ExperimentExecutor,
    ExperimentTask,
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


class TestExperimentTask:
    def test_creation(self):
        task = ExperimentTask(
            experiment_id="exp-0-0",
            params={"lr": 0.02},
            command="echo test",
            baseline_commit="abc123",
        )
        assert task.experiment_id == "exp-0-0"
        assert task.params == {"lr": 0.02}
        assert task.command == "echo test"
        assert task.baseline_commit == "abc123"
        assert task.resource == ""

    def test_creation_with_resource(self):
        task = ExperimentTask(
            experiment_id="exp-0-0",
            params={"lr": 0.02},
            command="echo test",
            baseline_commit="abc123",
            resource="gpu:0",
        )
        assert task.resource == "gpu:0"


class TestRunExperimentsDefault:
    """Test that the default run_experiments calls run_experiment sequentially."""

    def test_default_sequential_batch(self):
        from chaosengineer.testing.executor import ScriptedExecutor

        results_map = {
            "exp-0-0": ExperimentResult(primary_metric=0.91),
            "exp-0-1": ExperimentResult(primary_metric=0.95),
        }
        executor = ScriptedExecutor(results_map)

        tasks = [
            ExperimentTask("exp-0-0", {"lr": 0.02}, "echo", "abc"),
            ExperimentTask("exp-0-1", {"lr": 0.08}, "echo", "abc"),
        ]
        results = executor.run_experiments(tasks)

        assert len(results) == 2
        assert results[0].primary_metric == 0.91
        assert results[1].primary_metric == 0.95


class TestRunExperimentsCallback:
    def test_callback_called_per_result(self):
        from unittest.mock import MagicMock
        from chaosengineer.core.interfaces import ExperimentTask, ExperimentExecutor
        from chaosengineer.core.models import ExperimentResult

        class SimpleExecutor(ExperimentExecutor):
            def run_experiment(self, experiment_id, params, command, baseline_commit, resource=""):
                return ExperimentResult(primary_metric=1.0)

        executor = SimpleExecutor()
        tasks = [
            ExperimentTask("e-0", {"lr": 0.1}, "echo", "abc"),
            ExperimentTask("e-1", {"lr": 0.2}, "echo", "abc"),
        ]
        callback = MagicMock()
        results = executor.run_experiments(tasks, on_worker_done=callback)

        assert len(results) == 2
        assert callback.call_count == 2
        assert callback.call_args_list[0][0][2] == 1  # completed count
        assert callback.call_args_list[0][0][3] == 2  # total count
        assert callback.call_args_list[1][0][2] == 2

    def test_callback_none_is_fine(self):
        class SimpleExecutor(ExperimentExecutor):
            def run_experiment(self, experiment_id, params, command, baseline_commit, resource=""):
                return ExperimentResult(primary_metric=1.0)

        executor = SimpleExecutor()
        tasks = [ExperimentTask("e-0", {"lr": 0.1}, "echo", "abc")]
        results = executor.run_experiments(tasks)
        assert len(results) == 1

    def test_kill_active_default_noop(self):
        class SimpleExecutor(ExperimentExecutor):
            def run_experiment(self, experiment_id, params, command, baseline_commit, resource=""):
                return ExperimentResult(primary_metric=1.0)

        executor = SimpleExecutor()
        executor.kill_active()  # Should not raise
