"""ViewManager — toggle between log mode and TUI mode."""
from __future__ import annotations

import sys
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chaosengineer.core.coordinator import Coordinator
    from chaosengineer.core.pause import PauseController
    from chaosengineer.core.status import StatusDisplay
    from chaosengineer.tui.bridge import EventBridge
    from chaosengineer.tui.pause_gate import PauseGate


class ViewManager:
    """Manages toggling between log mode (stderr) and TUI mode (Textual)."""

    DEBOUNCE_MS = 500

    def __init__(
        self,
        bridge: "EventBridge",
        pause_gate: "PauseGate",
        pause_controller: "PauseController",
        coordinator: "Coordinator",
        status_display: "StatusDisplay",
    ):
        self._bridge = bridge
        self._pause_gate = pause_gate
        self._pause_controller = pause_controller
        self._coordinator = coordinator
        self._status_display = status_display
        self.tui_active: bool = False
        self._coord_done = threading.Event()
        self._last_toggle: float = 0

    def run(self, coord_done: threading.Event) -> None:
        """Main loop. Runs on the main thread. Blocks until coordinator finishes."""
        import tty
        import termios

        self._coord_done = coord_done
        print("Press 't' to open TUI dashboard", file=sys.stderr)

        if not sys.stdin.isatty():
            coord_done.wait()
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._coord_done.is_set():
                if self._coord_done.wait(timeout=0.2):
                    break
                if self._check_stdin_for_toggle():
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    self._enter_tui()
                    tty.setcbreak(fd)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _check_stdin_for_toggle(self) -> bool:
        """Check if 't' was pressed on stdin. Terminal must be in cbreak mode."""
        try:
            import select
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if ready:
                ch = sys.stdin.read(1)
                if ch == "t" and self._debounce_ok():
                    return True
        except Exception:
            pass
        return False

    def _debounce_ok(self) -> bool:
        now = time.monotonic() * 1000
        if now - self._last_toggle < self.DEBOUNCE_MS:
            return False
        self._last_toggle = now
        return True

    def _enter_tui(self) -> None:
        """Switch to TUI mode."""
        from chaosengineer.tui.app import ChaosApp

        self.tui_active = True
        self._status_display.suppressed = True

        app = ChaosApp(
            bridge=self._bridge,
            pause_gate=self._pause_gate,
            coordinator=self._coordinator,
            pause_controller=self._pause_controller,
        )
        try:
            app.run()
        except Exception as e:
            print(f"\nTUI error: {e}. Falling back to log mode.", file=sys.stderr)
        finally:
            self.tui_active = False
            self._status_display.suppressed = False
            # Unblock coordinator if a pause decision was pending when TUI exited
            if self._pause_gate.decision_needed.is_set():
                self._pause_gate.submit_decision("continue")
