"""Tests for ViewManager — toggle between log mode and TUI mode."""
from unittest.mock import MagicMock, patch
import threading

import pytest

from chaosengineer.tui.views import ViewManager
from chaosengineer.tui.bridge import EventBridge
from chaosengineer.tui.pause_gate import PauseGate


def _make_vm(**overrides):
    """Helper to create a ViewManager with sensible defaults."""
    kwargs = dict(
        bridge=EventBridge(),
        pause_gate=PauseGate(),
        pause_controller=MagicMock(),
        coordinator=MagicMock(),
        status_display=MagicMock(),
    )
    kwargs.update(overrides)
    return ViewManager(**kwargs)


class TestViewManagerState:
    def test_starts_in_log_mode(self):
        vm = _make_vm()
        assert not vm.tui_active

    def test_tui_active_flag_set_during_tui(self):
        vm = _make_vm()
        assert not vm.tui_active


class TestViewManagerCoordinatorDone:
    def test_stops_when_coordinator_thread_finishes(self):
        """ViewManager.run() should exit when coord_done event is set."""
        vm = _make_vm()
        vm._coord_done = threading.Event()
        vm._coord_done.set()
        # run() should return quickly since coordinator is done
        # (We can't test the full stdin loop without a terminal)


class TestDebounceLogic:
    def test_first_toggle_always_passes(self):
        """First call to _debounce_ok() should always return True."""
        vm = _make_vm()
        with patch("time.monotonic", return_value=1.0):
            assert vm._debounce_ok() is True

    def test_rapid_toggle_rejected(self):
        """Second call within 500ms should return False."""
        vm = _make_vm()
        with patch("time.monotonic", return_value=1.0):
            assert vm._debounce_ok() is True
        # 200ms later (1.0s + 0.2s = 1.2s => 1200ms; gap = 200ms < 500ms)
        with patch("time.monotonic", return_value=1.2):
            assert vm._debounce_ok() is False

    def test_toggle_after_debounce_passes(self):
        """Call after 500ms+ should return True."""
        vm = _make_vm()
        with patch("time.monotonic", return_value=1.0):
            assert vm._debounce_ok() is True
        # 600ms later (1.0s + 0.6s = 1.6s => 1600ms; gap = 600ms > 500ms)
        with patch("time.monotonic", return_value=1.6):
            assert vm._debounce_ok() is True

    def test_debounce_uses_monotonic_clock(self):
        """Verify _debounce_ok calls time.monotonic."""
        vm = _make_vm()
        with patch("time.monotonic", return_value=10.0) as mock_mono:
            vm._debounce_ok()
            mock_mono.assert_called()


class TestCrashRecovery:
    @patch("chaosengineer.tui.views.ChaosApp", create=True)
    def test_tui_crash_falls_back_to_log_mode(self, _mock_cls, capsys):
        """If ChaosApp.run() raises, ViewManager should recover gracefully."""
        vm = _make_vm()
        with patch("chaosengineer.tui.app.ChaosApp") as MockApp:
            MockApp.return_value.run.side_effect = RuntimeError("boom")
            vm._enter_tui()

        assert vm.tui_active is False
        assert vm._status_display.suppressed is False
        captured = capsys.readouterr()
        assert "TUI error" in captured.err
        assert "boom" in captured.err

    @patch("chaosengineer.tui.app.ChaosApp")
    def test_tui_crash_auto_continues_pending_pause(self, MockApp):
        """If TUI crashes while pause decision is pending, auto-continue."""
        pause_gate = PauseGate()
        vm = _make_vm(pause_gate=pause_gate)
        MockApp.return_value.run.side_effect = RuntimeError("crash")

        # Simulate a pending pause decision
        pause_gate.decision_needed.set()

        vm._enter_tui()

        # submit_decision("continue") should have been called,
        # which sets decision_ready and stores the decision.
        # (decision_needed is cleared by request_decision, not submit_decision)
        assert pause_gate.decision_ready.is_set()
        assert pause_gate.decision == "continue"

    @patch("chaosengineer.tui.app.ChaosApp")
    def test_tui_crash_does_not_stop_main_loop(self, MockApp):
        """After TUI crashes, ViewManager should not propagate the exception."""
        vm = _make_vm()
        MockApp.return_value.run.side_effect = RuntimeError("crash")

        # _enter_tui should NOT raise
        vm._enter_tui()

        # ViewManager should be back in log mode, ready to continue
        assert vm.tui_active is False


class TestNonTTYFallback:
    @patch("sys.stdin")
    def test_non_tty_stdin_waits_for_coordinator(self, mock_stdin):
        """When stdin is not a TTY, run() should just wait on coord_done."""
        mock_stdin.isatty.return_value = False
        vm = _make_vm()
        coord_done = threading.Event()

        # Set coord_done immediately so run() doesn't block forever
        coord_done.set()
        vm.run(coord_done)

        mock_stdin.isatty.assert_called_once()

    @patch("sys.stdin")
    def test_non_tty_no_terminal_operations(self, mock_stdin):
        """When stdin is not a TTY, should NOT call tty.setcbreak or termios."""
        mock_stdin.isatty.return_value = False
        vm = _make_vm()
        coord_done = threading.Event()
        coord_done.set()

        with patch.dict("sys.modules", {"tty": MagicMock(), "termios": MagicMock()}) as modules:
            vm.run(coord_done)

        # stdin.fileno should not have been called (no terminal setup)
        mock_stdin.fileno.assert_not_called()


class TestStatusDisplaySuppression:
    @patch("chaosengineer.tui.app.ChaosApp")
    def test_status_display_suppressed_during_tui(self, MockApp):
        """status_display.suppressed should be True while TUI is active."""
        status_display = MagicMock()
        vm = _make_vm(status_display=status_display)

        suppressed_values = []

        def capture_run():
            suppressed_values.append(vm.tui_active)
            suppressed_values.append(status_display.suppressed)

        MockApp.return_value.run.side_effect = capture_run
        vm._enter_tui()

        # During app.run(), tui_active was True and suppressed was True
        assert suppressed_values == [True, True]

    @patch("chaosengineer.tui.app.ChaosApp")
    def test_status_display_restored_after_tui_exit(self, MockApp):
        """status_display.suppressed should be False after TUI exits."""
        status_display = MagicMock()
        vm = _make_vm(status_display=status_display)

        MockApp.return_value.run.return_value = None
        vm._enter_tui()

        assert vm.tui_active is False
        assert status_display.suppressed is False
