"""PauseController — signal handler and interactive pause menus."""

from __future__ import annotations

import signal
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chaosengineer.core.interfaces import ExperimentExecutor


class PauseController:
    """Manages graceful pause state and interactive menus."""

    def __init__(self) -> None:
        self.pause_requested: bool = False
        self.force_kill: bool = False
        self.wait_then_ask: bool = False
        self.kill_issued: bool = False
        self._executor: ExperimentExecutor | None = None
        self._original_handler = None

    def install(self) -> None:
        self._original_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.on_sigint)

    def uninstall(self) -> None:
        if self._original_handler is not None:
            signal.signal(signal.SIGINT, self._original_handler)
            self._original_handler = None

    def set_executor(self, executor: "ExperimentExecutor") -> None:
        self._executor = executor

    def on_sigint(self, signum: int, frame) -> None:
        if self.pause_requested:
            self.force_kill = True
            signal.signal(signal.SIGINT, signal.default_int_handler)
            print("\nForce kill armed — next Ctrl+C will terminate.", file=sys.stderr)
        else:
            self.pause_requested = True
            print("\nPause requested — will pause after current work finishes.", file=sys.stderr)

    def should_show_menu(self) -> bool:
        if self.kill_issued:
            return False
        return self.pause_requested or self.wait_then_ask

    def reset(self) -> None:
        self.pause_requested = False
        self.force_kill = False
        self.wait_then_ask = False
        self.kill_issued = False

    def show_mid_iteration_menu(self, completed: int, total: int) -> str:
        from chaosengineer.cli_menu import select
        options = [
            "[W] Wait for remaining workers, then decide",
            "[K] Kill workers and pause now",
            "[C] Continue running",
        ]
        idx = select(
            f"Pause requested\n\n  {completed}/{total} workers completed this iteration.",
            options,
        )
        return ["wait", "kill", "continue"][idx]

    def show_post_iteration_menu(self, summary: str = "") -> str:
        from chaosengineer.cli_menu import select
        prompt = summary or "Pause requested"
        options = ["[P] Pause now", "[C] Continue running"]
        idx = select(prompt, options)
        return ["pause", "continue"][idx]
