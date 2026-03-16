"""Tests for Coordinator integration with TUI (PauseGate, ViewManager)."""
import threading
import time
from unittest.mock import MagicMock

import pytest

from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.interfaces import DimensionPlan
from chaosengineer.core.models import Baseline, BudgetConfig, ExperimentResult
from chaosengineer.core.pause import PauseController
from chaosengineer.metrics.logger import EventLogger
from chaosengineer.testing.executor import ScriptedExecutor
from chaosengineer.testing.simulator import ScriptedDecisionMaker
from chaosengineer.tui.pause_gate import PauseGate
from chaosengineer.workloads.parser import WorkloadSpec


def _make_spec(budget=None):
    return WorkloadSpec(
        name="test", primary_metric="loss", metric_direction="lower",
        execution_command="echo 1", workers_available=1,
        budget=budget or BudgetConfig(max_experiments=10),
    )


class TestCoordinatorPauseGate:
    def test_uses_pause_gate_when_tui_active(self, tmp_path):
        """Coordinator uses PauseGate instead of interactive menu when TUI is active."""
        spec = _make_spec()
        plans = [DimensionPlan("lr", [{"lr": 0.01}])]
        results = {"exp-0-0": ExperimentResult(primary_metric=2.5)}

        pc = PauseController()
        pc.pause_requested = True

        gate = PauseGate()
        view_manager = MagicMock()
        view_manager.tui_active = True

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_path / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
            pause_controller=pc,
            view_manager=view_manager,
            pause_gate=gate,
        )

        # Run coordinator in thread (it will block on PauseGate)
        coord_result = []
        def run_coord():
            coordinator.run()
            coord_result.append("done")

        t = threading.Thread(target=run_coord)
        t.start()

        # Wait for gate to signal decision needed
        gate.decision_needed.wait(timeout=5.0)
        assert gate.decision_needed.is_set()
        assert "pause" in gate.options

        # Submit pause decision
        gate.submit_decision("pause")
        t.join(timeout=5.0)

        assert coord_result == ["done"]

    def test_falls_back_to_menu_when_tui_not_active(self, tmp_path):
        """When view_manager exists but tui_active=False, uses normal menu."""
        spec = _make_spec()
        plans = [DimensionPlan("lr", [{"lr": 0.01}])]
        results = {"exp-0-0": ExperimentResult(primary_metric=2.5)}

        pc = PauseController()
        pc.pause_requested = True
        pc.show_post_iteration_menu = MagicMock(return_value="pause")

        view_manager = MagicMock()
        view_manager.tui_active = False

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=EventLogger(tmp_path / "events.jsonl"),
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("aaa", 3.0, "loss"),
            pause_controller=pc,
            view_manager=view_manager,
            pause_gate=PauseGate(),
        )

        coordinator.run()
        pc.show_post_iteration_menu.assert_called()
