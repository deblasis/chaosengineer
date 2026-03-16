"""EvaluationGate -- coordinator <-> TUI human-in-the-loop evaluation handoff."""
from __future__ import annotations

import threading


class EvaluationGate:
    """Shared object for blocking coordinator until TUI user submits an evaluation.

    The coordinator thread calls request_evaluation() which blocks until the
    TUI user submits a score (or timeout expires).  The TUI thread calls
    submit_evaluation() or skip_evaluation() once the user has acted.
    """

    #: Default timeout (seconds) before auto-skipping.
    DEFAULT_TIMEOUT: float = 300.0

    def __init__(self) -> None:
        self.evaluation_needed = threading.Event()
        self.evaluation_ready = threading.Event()
        self.experiment_id: str | None = None
        self.details: dict = {}
        self.score: float | None = None
        self.note: str = ""

    def request_evaluation(
        self,
        experiment_id: str,
        details: dict,
        timeout: float | None = None,
    ) -> tuple[float | None, str]:
        """Block coordinator until TUI user submits evaluation.

        Returns:
            Tuple of (score, note).  score is ``None`` when skipped or
            timed-out.
        """
        if timeout is None:
            timeout = self.DEFAULT_TIMEOUT

        self.experiment_id = experiment_id
        self.details = details
        self.score = None
        self.note = ""
        self.evaluation_ready.clear()
        self.evaluation_needed.set()

        if not self.evaluation_ready.wait(timeout=timeout):
            # Timeout — return skip sentinel.
            self.score = None
            self.note = ""

        self.evaluation_needed.clear()
        return (self.score, self.note)

    def submit_evaluation(self, score: float, note: str = "") -> None:
        """Called from TUI thread when user submits a score."""
        self.score = score
        self.note = note
        self.evaluation_ready.set()

    def skip_evaluation(self) -> None:
        """Called from TUI thread when user skips evaluation."""
        self.score = None
        self.note = ""
        self.evaluation_ready.set()
