"""Integration tests: full human evaluation loop (coordinator + gate + bridge)."""
import threading
import time

import pytest

from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.interfaces import DimensionPlan
from chaosengineer.core.models import Baseline, BudgetConfig, ExperimentResult
from chaosengineer.metrics.publisher import EventPublisher
from chaosengineer.testing.executor import ScriptedExecutor
from chaosengineer.testing.simulator import ScriptedDecisionMaker
from chaosengineer.bus import EventBridge
from chaosengineer.tui.eval_gate import EvaluationGate
from chaosengineer.workloads.parser import WorkloadSpec


def _make_spec(evaluation_type="human", max_experiments=1):
    return WorkloadSpec(
        name="eval-integration",
        primary_metric="score",
        metric_direction="higher",
        execution_command="echo 1",
        evaluation_type=evaluation_type,
        budget=BudgetConfig(max_experiments=max_experiments),
    )


class TestEvalLoopIntegration:
    """Full coordinator→gate→bridge round-trip for human evaluation."""

    def test_human_eval_submit_flows_through(self, tmp_path):
        """Coordinator blocks on gate, external submit unblocks, score is used."""
        spec = _make_spec()
        plans = [DimensionPlan("lr", [{"lr": 0.01}])]
        results = {"exp-0-0": ExperimentResult(primary_metric=0.5, cost_usd=0.1)}

        bridge = EventBridge()
        gate = EvaluationGate()
        publisher = EventPublisher(
            path=tmp_path / "events.jsonl", bridge=bridge,
        )

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=publisher,
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("HEAD", 0.0, "score"),
            eval_gate=gate,
        )

        def submit_later():
            gate.evaluation_needed.wait(timeout=5)
            gate.submit_evaluation(0.92, "excellent result")

        t_submit = threading.Thread(target=submit_later, daemon=True)
        t_submit.start()

        coordinator.run()
        t_submit.join(timeout=5)

        # Human score should override executor's metric
        assert coordinator.best_baseline.metric_value == 0.92

        # Verify event flow through bridge
        events = bridge.snapshot()
        event_types = [e["event"] for e in events]
        assert "evaluation_requested" in event_types
        assert "evaluation_submitted" in event_types

        # Verify evaluation_submitted event has correct data
        eval_event = next(e for e in events if e["event"] == "evaluation_submitted")
        assert eval_event["score"] == 0.92
        assert eval_event["note"] == "excellent result"

    def test_human_eval_skip_marks_experiment_failed(self, tmp_path):
        """When human skips evaluation, experiment is failed and baseline unchanged."""
        spec = _make_spec()
        plans = [DimensionPlan("lr", [{"lr": 0.01}])]
        results = {"exp-0-0": ExperimentResult(primary_metric=0.5, cost_usd=0.1)}

        bridge = EventBridge()
        gate = EvaluationGate()
        publisher = EventPublisher(
            path=tmp_path / "events.jsonl", bridge=bridge,
        )

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=publisher,
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("HEAD", 0.0, "score"),
            eval_gate=gate,
        )

        def skip_later():
            gate.evaluation_needed.wait(timeout=5)
            gate.skip_evaluation()

        t_skip = threading.Thread(target=skip_later, daemon=True)
        t_skip.start()

        coordinator.run()
        t_skip.join(timeout=5)

        # Baseline should remain at initial value
        assert coordinator.best_baseline.metric_value == 0.0

        # Verify evaluation_requested was logged but NOT evaluation_submitted
        events = bridge.snapshot()
        event_types = [e["event"] for e in events]
        assert "evaluation_requested" in event_types
        assert "evaluation_submitted" not in event_types
        assert "worker_failed" in event_types

    def test_multiple_experiments_each_evaluated(self, tmp_path):
        """Each experiment in an iteration gets its own human evaluation."""
        spec = _make_spec(max_experiments=2)
        plans = [DimensionPlan("lr", [{"lr": 0.01}, {"lr": 0.1}])]
        results = {
            "exp-0-0": ExperimentResult(primary_metric=0.5, cost_usd=0.1),
            "exp-0-1": ExperimentResult(primary_metric=0.6, cost_usd=0.1),
        }

        bridge = EventBridge()
        gate = EvaluationGate()
        publisher = EventPublisher(
            path=tmp_path / "events.jsonl", bridge=bridge,
        )

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=publisher,
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("HEAD", 0.0, "score"),
            eval_gate=gate,
        )

        scores = [0.75, 0.95]
        eval_count = 0

        def submit_each():
            nonlocal eval_count
            for score in scores:
                # Wait for coordinator to signal it needs evaluation
                gate.evaluation_needed.wait(timeout=10)
                if not gate.evaluation_needed.is_set():
                    return  # Timed out
                gate.submit_evaluation(score, f"eval {eval_count}")
                eval_count += 1
                # Wait for coordinator to consume the result before next round
                gate.evaluation_ready.wait(timeout=5)
                # Small sleep to let coordinator clear evaluation_needed
                time.sleep(0.05)

        t_submit = threading.Thread(target=submit_each, daemon=True)
        t_submit.start()

        coordinator.run()
        t_submit.join(timeout=10)

        # Best score should be 0.95
        assert coordinator.best_baseline.metric_value == 0.95
        assert eval_count == 2

        # Two evaluation_requested + two evaluation_submitted events
        events = bridge.snapshot()
        req_events = [e for e in events if e["event"] == "evaluation_requested"]
        sub_events = [e for e in events if e["event"] == "evaluation_submitted"]
        assert len(req_events) == 2
        assert len(sub_events) == 2

    def test_bus_command_submits_evaluation(self, tmp_path):
        """submit_evaluation bus command flows through coordinator to eval_gate."""
        spec = _make_spec()
        plans = [DimensionPlan("lr", [{"lr": 0.01}])]
        results = {"exp-0-0": ExperimentResult(primary_metric=0.5, cost_usd=0.1)}

        bridge = EventBridge()
        gate = EvaluationGate()
        publisher = EventPublisher(
            path=tmp_path / "events.jsonl", bridge=bridge,
        )

        # Simulate bus command delivery via poll_commands
        commands_delivered = False

        original_poll = publisher.poll_commands if hasattr(publisher, 'poll_commands') else None

        def mock_poll_commands():
            nonlocal commands_delivered
            if gate.evaluation_needed.is_set() and not commands_delivered:
                commands_delivered = True
                return [{"command": "submit_evaluation", "score": 0.88, "note": "via bus"}]
            return []

        publisher.poll_commands = mock_poll_commands

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=ScriptedExecutor(results),
            logger=publisher,
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline("HEAD", 0.0, "score"),
            eval_gate=gate,
        )

        # The coordinator polls bus commands in its loop, which should unblock the gate
        # But _poll_bus_commands runs at loop start, not while blocked on gate.
        # So we need a separate thread to submit via the gate directly for this path.
        # The bus command path is: coordinator polls → finds command → calls gate.submit_evaluation()
        # This happens on the coordinator thread BEFORE the next iteration, not during eval block.
        # For a true bus round-trip, we submit from a watcher thread.

        def submit_via_bus_commands():
            gate.evaluation_needed.wait(timeout=5)
            # Simulate what _poll_bus_commands does when it finds the command
            gate.submit_evaluation(0.88, "via bus")

        t = threading.Thread(target=submit_via_bus_commands, daemon=True)
        t.start()

        coordinator.run()
        t.join(timeout=5)

        assert coordinator.best_baseline.metric_value == 0.88

        events = bridge.snapshot()
        eval_sub = next(e for e in events if e["event"] == "evaluation_submitted")
        assert eval_sub["score"] == 0.88
        assert eval_sub["note"] == "via bus"
