"""Tests for coordinator human-in-the-loop evaluation wiring."""
from unittest.mock import MagicMock, patch
import threading

import pytest

from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.models import Baseline, BudgetConfig, DimensionSpec, DimensionType
from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.interfaces import DimensionPlan
from chaosengineer.tui.eval_gate import EvaluationGate
from chaosengineer.workloads.parser import WorkloadSpec


def _make_spec(evaluation_type="automatic", max_experiments=1):
    return WorkloadSpec(
        name="test",
        context="test workload",
        dimensions=[DimensionSpec(name="lr", dim_type=DimensionType.DIRECTIONAL, current_value=0.01)],
        execution_command="echo test",
        evaluation_type=evaluation_type,
        primary_metric="accuracy",
        metric_direction="higher",
        metric_parse_command="echo 0.5",
        budget=BudgetConfig(max_experiments=max_experiments),
    )


def _make_coordinator(spec, eval_gate=None):
    dm = MagicMock()
    dm.pick_next_dimension.return_value = DimensionPlan(
        dimension_name="lr",
        values=[{"lr": 0.1}],
    )
    dm.discover_diverse_options = MagicMock(return_value=[])

    executor = MagicMock()
    result = MagicMock()
    result.primary_metric = 0.5
    result.cost_usd = 0.10
    result.error_message = None
    result.commit_hash = None
    result.to_dict.return_value = {"primary_metric": 0.5}
    executor.run_experiments.return_value = [result]

    logger = MagicMock()
    budget = BudgetTracker(spec.budget)
    baseline = Baseline(commit="HEAD", metric_value=0.0, metric_name="accuracy")

    return Coordinator(
        spec=spec,
        decision_maker=dm,
        executor=executor,
        logger=logger,
        budget=budget,
        initial_baseline=baseline,
        eval_gate=eval_gate,
    )


class TestCoordinatorEvalGate:
    def test_human_eval_calls_eval_gate(self):
        """When evaluation_type=human and eval_gate is set, coordinator calls request_evaluation."""
        spec = _make_spec(evaluation_type="human")
        gate = EvaluationGate()

        # Pre-submit evaluation so the gate doesn't block forever
        def submit_later():
            gate.evaluation_needed.wait(timeout=5)
            gate.submit_evaluation(0.85, "looks good")

        t = threading.Thread(target=submit_later, daemon=True)
        t.start()

        coord = _make_coordinator(spec, eval_gate=gate)
        coord.run()
        t.join(timeout=5)

        # The experiment should have used the human score
        assert coord.best_baseline.metric_value == 0.85

    def test_automatic_eval_does_not_call_gate(self):
        """When evaluation_type=automatic, eval_gate is not used even if provided."""
        spec = _make_spec(evaluation_type="automatic")
        gate = EvaluationGate()
        coord = _make_coordinator(spec, eval_gate=gate)
        coord.run()

        # Should use the executor's metric, not block on gate
        assert not gate.evaluation_needed.is_set()

    def test_human_eval_skip_marks_failed(self):
        """When human skips evaluation (score=None), experiment is treated as failed."""
        spec = _make_spec(evaluation_type="human")
        gate = EvaluationGate()

        def skip_later():
            gate.evaluation_needed.wait(timeout=5)
            gate.skip_evaluation()

        t = threading.Thread(target=skip_later, daemon=True)
        t.start()

        coord = _make_coordinator(spec, eval_gate=gate)
        coord.run()
        t.join(timeout=5)

        # Skipped = no improvement from baseline of 0.0
        assert coord.best_baseline.metric_value == 0.0

    def test_human_eval_no_gate_logs_warning(self):
        """When evaluation_type=human but no eval_gate, experiment is skipped with warning."""
        spec = _make_spec(evaluation_type="human")
        coord = _make_coordinator(spec, eval_gate=None)
        coord.run()

        # Should complete without blocking, baseline unchanged
        assert coord.best_baseline.metric_value == 0.0
