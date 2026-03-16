"""Tests for EvaluationGate -- coordinator <-> TUI evaluation handoff."""
import threading
import time

import pytest

from chaosengineer.tui.eval_gate import EvaluationGate


class TestEvaluationGateRequestBlocks:
    def test_blocks_until_submit(self):
        gate = EvaluationGate()
        result = []

        def coordinator():
            score, note = gate.request_evaluation("exp-1", {"dim": "lr"})
            result.append((score, note))

        t = threading.Thread(target=coordinator)
        t.start()
        time.sleep(0.05)
        assert not result  # still blocked

        gate.submit_evaluation(0.85, "looks good")
        t.join(timeout=2.0)
        assert result == [(0.85, "looks good")]

    def test_experiment_id_and_details_set_before_blocking(self):
        gate = EvaluationGate()

        def coordinator():
            gate.request_evaluation("exp-2", {"dim": "lr", "params": {"lr": 0.01}})

        t = threading.Thread(target=coordinator)
        t.start()
        time.sleep(0.05)
        assert gate.experiment_id == "exp-2"
        assert gate.details == {"dim": "lr", "params": {"lr": 0.01}}
        assert gate.evaluation_needed.is_set()
        gate.submit_evaluation(0.5)
        t.join(timeout=2.0)

    def test_evaluation_needed_cleared_after_submit(self):
        gate = EvaluationGate()

        def coordinator():
            gate.request_evaluation("exp-3", {})

        t = threading.Thread(target=coordinator)
        t.start()
        time.sleep(0.05)
        gate.submit_evaluation(1.0)
        t.join(timeout=2.0)
        assert not gate.evaluation_needed.is_set()


class TestEvaluationGateTimeout:
    def test_timeout_returns_none_score(self):
        gate = EvaluationGate()
        score, note = gate.request_evaluation("exp-t", {}, timeout=0.1)
        assert score is None
        assert note == ""

    def test_timeout_clears_evaluation_needed(self):
        gate = EvaluationGate()
        gate.request_evaluation("exp-t2", {}, timeout=0.1)
        assert not gate.evaluation_needed.is_set()


class TestEvaluationGateSkip:
    def test_skip_returns_none_score(self):
        gate = EvaluationGate()
        result = []

        def coordinator():
            score, note = gate.request_evaluation("exp-s", {"dim": "temp"})
            result.append((score, note))

        t = threading.Thread(target=coordinator)
        t.start()
        time.sleep(0.05)
        gate.skip_evaluation()
        t.join(timeout=2.0)
        assert result == [(None, "")]

    def test_skip_clears_evaluation_needed(self):
        gate = EvaluationGate()

        def coordinator():
            gate.request_evaluation("exp-s2", {})

        t = threading.Thread(target=coordinator)
        t.start()
        time.sleep(0.05)
        gate.skip_evaluation()
        t.join(timeout=2.0)
        assert not gate.evaluation_needed.is_set()

    def test_submit_note_preserved(self):
        gate = EvaluationGate()
        result = []

        def coordinator():
            result.append(gate.request_evaluation("exp-n", {}))

        t = threading.Thread(target=coordinator)
        t.start()
        time.sleep(0.05)
        gate.submit_evaluation(0.42, "needs more epochs")
        t.join(timeout=2.0)
        assert result == [(0.42, "needs more epochs")]
