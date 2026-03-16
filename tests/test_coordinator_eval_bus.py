"""Tests for coordinator handling submit_evaluation bus command."""
from unittest.mock import MagicMock

import pytest

from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.models import Baseline, BudgetConfig
from chaosengineer.tui.eval_gate import EvaluationGate
from chaosengineer.workloads.parser import WorkloadSpec


class TestSubmitEvaluationCommand:
    def test_poll_commands_handles_submit_evaluation(self):
        """_poll_bus_commands should call eval_gate.submit_evaluation on submit_evaluation command."""
        spec = WorkloadSpec(
            name="test", evaluation_type="human",
            primary_metric="score", metric_direction="higher",
            budget=BudgetConfig(max_experiments=5),
        )
        gate = EvaluationGate()
        gate.evaluation_needed.set()  # Simulate waiting for evaluation

        logger = MagicMock()
        logger.poll_commands.return_value = [
            {"command": "submit_evaluation", "experiment_id": "exp-0-0", "score": 0.75, "note": "ok"},
        ]

        coord = Coordinator(
            spec=spec,
            decision_maker=MagicMock(),
            executor=MagicMock(),
            logger=logger,
            budget=MagicMock(),
            initial_baseline=Baseline("HEAD", 0.0, "score"),
            eval_gate=gate,
        )
        coord._poll_bus_commands()

        assert gate.score == 0.75
        assert gate.note == "ok"
        assert gate.evaluation_ready.is_set()

    def test_poll_commands_ignores_eval_without_gate(self):
        """submit_evaluation command is ignored when no eval_gate is set."""
        spec = WorkloadSpec(name="test", budget=BudgetConfig(max_experiments=5))
        logger = MagicMock()
        logger.poll_commands.return_value = [
            {"command": "submit_evaluation", "score": 0.75},
        ]

        coord = Coordinator(
            spec=spec,
            decision_maker=MagicMock(),
            executor=MagicMock(),
            logger=logger,
            budget=MagicMock(),
            initial_baseline=Baseline("HEAD", 0.0, "score"),
        )
        # Should not raise
        coord._poll_bus_commands()
