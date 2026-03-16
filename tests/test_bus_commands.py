# tests/test_bus_commands.py
"""Tests for coordinator command polling from the message bus."""
from unittest.mock import MagicMock, patch

import pytest

from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.models import (
    Baseline,
    BudgetConfig,
    DimensionSpec,
    DimensionType,
)
from chaosengineer.core.budget import BudgetTracker
from chaosengineer.metrics.logger import Event, EventLogger


def make_coordinator(logger, budget_config=None):
    """Create a minimal coordinator for testing command polling."""
    from chaosengineer.core.pause import PauseController

    spec = MagicMock()
    spec.name = "test"
    spec.dimensions = []
    spec.budget = budget_config or BudgetConfig(max_experiments=100)
    spec.primary_metric = "loss"
    spec.metric_direction = "lower"
    spec.execution_command = "echo test"
    spec.spec_hash.return_value = "abc123"

    budget = BudgetTracker(spec.budget)
    pause_controller = PauseController()

    coordinator = Coordinator(
        spec=spec,
        decision_maker=MagicMock(),
        executor=MagicMock(),
        logger=logger,
        budget=budget,
        initial_baseline=Baseline("HEAD", 1.0, "loss"),
        pause_controller=pause_controller,
    )
    return coordinator, pause_controller


class TestPollBusCommands:
    def test_pause_command_sets_flag(self):
        logger = MagicMock()
        logger.poll_commands.return_value = [
            {"command": "pause", "run_id": "run-1"}
        ]

        coord, pause_ctrl = make_coordinator(logger)
        coord._poll_bus_commands()

        assert pause_ctrl.pause_requested is True

    def test_extend_budget_cost(self):
        logger = MagicMock()
        logger.poll_commands.return_value = [
            {"command": "extend_budget", "add_cost_usd": 10.0}
        ]

        coord, _ = make_coordinator(
            logger, BudgetConfig(max_api_cost=5.0, max_experiments=10)
        )
        coord._poll_bus_commands()

        assert coord.budget.config.max_api_cost == 15.0
        assert coord.budget.config.max_experiments == 10  # unchanged

    def test_extend_budget_experiments(self):
        logger = MagicMock()
        logger.poll_commands.return_value = [
            {"command": "extend_budget", "add_experiments": 5}
        ]

        coord, _ = make_coordinator(
            logger, BudgetConfig(max_experiments=10)
        )
        coord._poll_bus_commands()

        assert coord.budget.config.max_experiments == 15

    def test_extend_budget_time(self):
        logger = MagicMock()
        logger.poll_commands.return_value = [
            {"command": "extend_budget", "add_time_seconds": 300.0}
        ]

        coord, _ = make_coordinator(
            logger, BudgetConfig(max_wall_time_seconds=600.0)
        )
        coord._poll_bus_commands()

        assert coord.budget.config.max_wall_time_seconds == 900.0

    def test_extend_budget_from_none(self):
        """Extending a None budget field should start from 0."""
        logger = MagicMock()
        logger.poll_commands.return_value = [
            {"command": "extend_budget", "add_cost_usd": 10.0}
        ]

        coord, _ = make_coordinator(
            logger, BudgetConfig(max_api_cost=None)
        )
        coord._poll_bus_commands()

        assert coord.budget.config.max_api_cost == 10.0

    def test_no_poll_when_logger_lacks_method(self):
        """EventLogger doesn't have poll_commands — should be a no-op."""
        logger = EventLogger("/dev/null")
        coord, pause_ctrl = make_coordinator(logger)
        coord._poll_bus_commands()  # Should not raise

        assert pause_ctrl.pause_requested is False

    def test_multiple_commands_in_one_poll(self):
        logger = MagicMock()
        logger.poll_commands.return_value = [
            {"command": "extend_budget", "add_cost_usd": 5.0},
            {"command": "pause", "run_id": "run-1"},
        ]

        coord, pause_ctrl = make_coordinator(
            logger, BudgetConfig(max_api_cost=10.0)
        )
        coord._poll_bus_commands()

        assert coord.budget.config.max_api_cost == 15.0
        assert pause_ctrl.pause_requested is True
