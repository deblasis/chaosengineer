"""Tests for the evaluation modal in ChaosApp."""
from unittest.mock import MagicMock

import pytest

from chaosengineer.tui.app import ChaosApp, EvaluationModal
from chaosengineer.bus import EventBridge
from chaosengineer.tui.eval_gate import EvaluationGate
from chaosengineer.tui.pause_gate import PauseGate
from textual.widgets import Button, Input, RichLog


@pytest.fixture
def eval_gate():
    return EvaluationGate()


@pytest.fixture
def app(eval_gate):
    bridge = EventBridge()
    gate = PauseGate()
    coordinator = MagicMock()
    pause_controller = MagicMock()
    pause_controller.pause_requested = False
    return ChaosApp(
        bridge=bridge,
        pause_gate=gate,
        coordinator=coordinator,
        pause_controller=pause_controller,
        eval_gate=eval_gate,
    )


class TestEvaluationRequestedEvent:
    async def test_evaluation_requested_shows_modal(self, app):
        """evaluation_requested event should push an EvaluationModal screen."""
        async with app.run_test() as pilot:
            app._handle_event({
                "ts": "2026-03-16T14:00:00Z",
                "event": "evaluation_requested",
                "experiment_id": "exp-eval-1",
                "dimension": "visual_quality",
            })
            await pilot.pause()
            # The modal should be the top screen
            assert len(app.screen_stack) > 1
            assert isinstance(app.screen_stack[-1], EvaluationModal)

    async def test_evaluation_requested_logs_message(self, app):
        """evaluation_requested event should log a notification."""
        async with app.run_test() as pilot:
            app._handle_event({
                "ts": "2026-03-16T14:00:00Z",
                "event": "evaluation_requested",
                "experiment_id": "exp-eval-2",
            })
            log = app.query_one("#event-log", RichLog)
            assert len(log.lines) > 0


class TestEvaluationModalSubmit:
    async def test_submit_score_dismisses_modal(self, app, eval_gate):
        """Submitting a score should close the modal and call eval_gate."""
        async with app.run_test() as pilot:
            app._handle_event({
                "event": "evaluation_requested",
                "experiment_id": "exp-sub-1",
                "dimension": "coherence",
            })
            await pilot.pause()

            # Fill in score
            score_input = app.screen.query_one("#score-input", Input)
            score_input.value = "0.85"

            # Fill in note
            note_input = app.screen.query_one("#note-input", Input)
            note_input.value = "looks great"

            # Press submit
            submit_btn = app.screen.query_one("#eval-submit", Button)
            await pilot.click(submit_btn)
            await pilot.pause()

            # Modal should be dismissed
            assert len(app.screen_stack) == 1
            # eval_gate should have received the score
            assert eval_gate.score == 0.85
            assert eval_gate.note == "looks great"

    async def test_submit_empty_score_skips(self, app, eval_gate):
        """Submitting with empty score should act as skip."""
        async with app.run_test() as pilot:
            app._handle_event({
                "event": "evaluation_requested",
                "experiment_id": "exp-sub-2",
            })
            await pilot.pause()

            # Leave score empty, press submit
            submit_btn = app.screen.query_one("#eval-submit", Button)
            await pilot.click(submit_btn)
            await pilot.pause()

            assert len(app.screen_stack) == 1
            assert eval_gate.score is None


class TestEvaluationModalSkip:
    async def test_skip_button_dismisses_modal(self, app, eval_gate):
        """Skip button should close modal and call skip on eval_gate."""
        async with app.run_test() as pilot:
            app._handle_event({
                "event": "evaluation_requested",
                "experiment_id": "exp-skip-1",
            })
            await pilot.pause()

            skip_btn = app.screen.query_one("#eval-skip", Button)
            await pilot.click(skip_btn)
            await pilot.pause()

            assert len(app.screen_stack) == 1
            assert eval_gate.score is None

    async def test_skip_logs_message(self, app, eval_gate):
        """Skipping should log a message."""
        async with app.run_test() as pilot:
            app._handle_event({
                "event": "evaluation_requested",
                "experiment_id": "exp-skip-2",
            })
            await pilot.pause()

            skip_btn = app.screen.query_one("#eval-skip", Button)
            await pilot.click(skip_btn)
            await pilot.pause()

            log = app.query_one("#event-log", RichLog)
            # Should have at least 2 lines: the event log + skip message
            assert len(log.lines) >= 2


class TestEvaluationModalWithoutGate:
    async def test_no_gate_still_works(self):
        """App without eval_gate should handle evaluation events gracefully."""
        bridge = EventBridge()
        gate = PauseGate()
        coordinator = MagicMock()
        pause_controller = MagicMock()
        pause_controller.pause_requested = False
        app_no_gate = ChaosApp(
            bridge=bridge,
            pause_gate=gate,
            coordinator=coordinator,
            pause_controller=pause_controller,
            # no eval_gate
        )
        async with app_no_gate.run_test() as pilot:
            app_no_gate._handle_event({
                "event": "evaluation_requested",
                "experiment_id": "exp-no-gate",
            })
            await pilot.pause()
            # Modal should still be shown
            assert len(app_no_gate.screen_stack) > 1

            # Skip it
            skip_btn = app_no_gate.screen.query_one("#eval-skip", Button)
            await pilot.click(skip_btn)
            await pilot.pause()
            assert len(app_no_gate.screen_stack) == 1
