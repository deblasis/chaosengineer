"""Tests for PauseController — state machine and signal handling."""

from __future__ import annotations

import signal
from unittest.mock import MagicMock, patch

import pytest

from chaosengineer.core.pause import PauseController


class TestPauseControllerStateMachine:
    def test_initial_state(self):
        pc = PauseController()
        assert pc.pause_requested is False
        assert pc.force_kill is False
        assert pc.wait_then_ask is False
        assert pc.kill_issued is False

    def test_first_sigint_sets_pause_requested(self):
        pc = PauseController()
        pc.on_sigint(signal.SIGINT, None)
        assert pc.pause_requested is True
        assert pc.force_kill is False

    def test_second_sigint_sets_force_kill(self):
        pc = PauseController()
        pc.on_sigint(signal.SIGINT, None)
        pc.on_sigint(signal.SIGINT, None)
        assert pc.force_kill is True

    def test_reset_clears_all_flags(self):
        pc = PauseController()
        pc.on_sigint(signal.SIGINT, None)
        pc.wait_then_ask = True
        pc.kill_issued = True
        pc.reset()
        assert pc.pause_requested is False
        assert pc.wait_then_ask is False
        assert pc.force_kill is False
        assert pc.kill_issued is False

    def test_should_show_menu_when_pause_requested(self):
        pc = PauseController()
        assert pc.should_show_menu() is False
        pc.pause_requested = True
        assert pc.should_show_menu() is True

    def test_should_show_menu_when_wait_then_ask(self):
        pc = PauseController()
        pc.wait_then_ask = True
        assert pc.should_show_menu() is True

    def test_should_show_menu_false_when_kill_issued(self):
        pc = PauseController()
        pc.pause_requested = True
        pc.kill_issued = True
        assert pc.should_show_menu() is False


class TestInstallUninstall:
    def test_install_registers_handler(self):
        pc = PauseController()
        original = signal.getsignal(signal.SIGINT)
        try:
            pc.install()
            assert signal.getsignal(signal.SIGINT) == pc.on_sigint
        finally:
            pc.uninstall()
            assert signal.getsignal(signal.SIGINT) == original

    def test_second_sigint_restores_default(self):
        pc = PauseController()
        pc.install()
        try:
            pc.on_sigint(signal.SIGINT, None)
            assert signal.getsignal(signal.SIGINT) == pc.on_sigint
            pc.on_sigint(signal.SIGINT, None)
            assert signal.getsignal(signal.SIGINT) == signal.default_int_handler
        finally:
            signal.signal(signal.SIGINT, signal.default_int_handler)


class TestSetExecutor:
    def test_set_executor_stores_reference(self):
        pc = PauseController()
        executor = MagicMock()
        pc.set_executor(executor)
        assert pc._executor is executor
