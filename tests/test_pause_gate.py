"""Tests for PauseGate — coordinator <-> TUI decision handoff."""
import threading
import time

import pytest

from chaosengineer.tui.pause_gate import PauseGate


class TestPauseGateRequestDecision:
    def test_blocks_until_decision_submitted(self):
        gate = PauseGate()
        result = []

        def coordinator():
            choice = gate.request_decision(["continue", "pause"])
            result.append(choice)

        t = threading.Thread(target=coordinator)
        t.start()
        time.sleep(0.05)
        assert not result  # still blocked

        gate.submit_decision("pause")
        t.join(timeout=2.0)
        assert result == ["pause"]

    def test_options_are_set_before_blocking(self):
        gate = PauseGate()

        def coordinator():
            gate.request_decision(["a", "b", "c"])

        t = threading.Thread(target=coordinator)
        t.start()
        time.sleep(0.05)
        assert gate.options == ["a", "b", "c"]
        assert gate.decision_needed.is_set()
        gate.submit_decision("a")
        t.join(timeout=2.0)

    def test_decision_needed_cleared_after_decision(self):
        gate = PauseGate()

        def coordinator():
            gate.request_decision(["x"])

        t = threading.Thread(target=coordinator)
        t.start()
        time.sleep(0.05)
        gate.submit_decision("x")
        t.join(timeout=2.0)
        assert not gate.decision_needed.is_set()
