"""PauseGate — coordinator <-> TUI pause decision handoff."""
from __future__ import annotations

import threading


class PauseGate:
    """Shared object for blocking coordinator until TUI user makes a decision."""

    def __init__(self):
        self.decision: str | None = None
        self.decision_ready = threading.Event()
        self.decision_needed = threading.Event()
        self.options: list[str] = []

    def request_decision(self, options: list[str]) -> str:
        """Called from coordinator thread. Blocks until TUI user decides."""
        self.options = options
        self.decision = None
        self.decision_ready.clear()
        self.decision_needed.set()
        if not self.decision_ready.wait(timeout=300):
            # Timeout — TUI may have crashed. Default to continue.
            self.decision = "continue"
        self.decision_needed.clear()
        return self.decision

    def submit_decision(self, choice: str) -> None:
        """Called from TUI thread when user picks an option."""
        self.decision = choice
        self.decision_ready.set()
