"""Integration tests: coordinator thread + EventBridge + consumer."""
import threading
import time

import pytest

from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.interfaces import DimensionPlan
from chaosengineer.core.models import Baseline, BudgetConfig, ExperimentResult
from chaosengineer.core.pause import PauseController
from chaosengineer.core.status import StatusDisplay
from chaosengineer.metrics.publisher import EventPublisher
from chaosengineer.testing.executor import ScriptedExecutor
from chaosengineer.testing.simulator import ScriptedDecisionMaker
from chaosengineer.bus import EventBridge
from chaosengineer.tui.pause_gate import PauseGate
from chaosengineer.workloads.parser import WorkloadSpec


def _make_spec(budget=None):
    return WorkloadSpec(
        name="test", primary_metric="loss", metric_direction="lower",
        execution_command="echo 1", workers_available=2,
        budget=budget or BudgetConfig(max_experiments=4),
    )


class TestCoordinatorBridgeIntegration:
    def test_events_flow_through_bridge(self, tmp_path):
        """Coordinator -> EventPublisher -> EventBridge -> consumer queue."""
        spec = _make_spec()
        plans = [
            DimensionPlan("lr", [{"lr": 0.01}, {"lr": 0.1}]),
            DimensionPlan("batch", [{"batch": 32}, {"batch": 64}]),
        ]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=2.5, cost_usd=0.5),
            "exp-0-1": ExperimentResult(primary_metric=2.0, cost_usd=0.5),
            "exp-1-0": ExperimentResult(primary_metric=1.8, cost_usd=0.5),
            "exp-1-1": ExperimentResult(primary_metric=1.5, cost_usd=0.5),
        }

        bridge = EventBridge()
        publisher = EventPublisher(
            path=tmp_path / "events.jsonl", bridge=bridge,
        )

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=publisher,
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("HEAD", 3.0, "loss"),
            status_display=StatusDisplay(),
        )

        coordinator.run()

        events = bridge.snapshot()
        event_types = [e["event"] for e in events]
        assert "run_started" in event_types
        assert "iteration_started" in event_types
        assert "worker_completed" in event_types

    def test_pause_via_gate_from_thread(self, tmp_path):
        """Coordinator in thread, consumer submits pause via PauseGate."""
        spec = _make_spec(BudgetConfig(max_experiments=20))
        plans = [DimensionPlan("lr", [{"lr": v} for v in [0.01, 0.1, 0.001, 0.5]])] * 5
        results = {
            f"exp-{i}-{j}": ExperimentResult(primary_metric=2.0 - i * 0.1 - j * 0.01, cost_usd=0.1)
            for i in range(5) for j in range(4)
        }

        bridge = EventBridge()
        gate = PauseGate()
        pc = PauseController()
        pc.pause_requested = True

        view_manager_mock = type("VM", (), {"tui_active": True})()

        publisher = EventPublisher(
            path=tmp_path / "events.jsonl", bridge=bridge,
        )

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=publisher,
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("HEAD", 3.0, "loss"),
            pause_controller=pc,
            view_manager=view_manager_mock,
            pause_gate=gate,
        )

        coord_done = threading.Event()

        def run_coord():
            coordinator.run()
            coord_done.set()

        t = threading.Thread(target=run_coord)
        t.start()

        # Wait for pause decision request
        gate.decision_needed.wait(timeout=10.0)
        assert gate.decision_needed.is_set()

        gate.submit_decision("pause")
        t.join(timeout=10.0)

        events = bridge.snapshot()
        event_types = [e["event"] for e in events]
        assert "run_paused" in event_types

    def test_extend_budget_from_consumer(self, tmp_path):
        """Consumer extends budget via coordinator.extend_budget()."""
        spec = _make_spec(BudgetConfig(max_api_cost=1.0, max_experiments=2))
        plans = [DimensionPlan("lr", [{"lr": 0.01}, {"lr": 0.1}])]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=2.5, cost_usd=0.3),
            "exp-0-1": ExperimentResult(primary_metric=2.0, cost_usd=0.3),
        }

        bridge = EventBridge()
        publisher = EventPublisher(
            path=tmp_path / "events.jsonl", bridge=bridge,
        )

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=publisher,
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("HEAD", 3.0, "loss"),
        )

        # Extend budget before run
        coordinator.extend_budget(add_cost=5.0, add_experiments=10)
        assert coordinator.budget.config.max_api_cost == 6.0
        assert coordinator.budget.config.max_experiments == 12
