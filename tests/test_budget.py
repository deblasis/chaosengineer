"""Tests for budget tracking and enforcement."""

import time
import pytest
from chaosengineer.core.models import BudgetConfig
from chaosengineer.core.budget import BudgetTracker


class TestBudgetTracker:
    def test_no_limits(self):
        tracker = BudgetTracker(BudgetConfig())
        assert not tracker.is_exhausted()

    def test_cost_tracking(self):
        tracker = BudgetTracker(BudgetConfig(max_api_cost=10.0))
        tracker.add_cost(3.0)
        assert tracker.spent_usd == 3.0
        assert tracker.remaining_cost == 7.0
        assert not tracker.is_exhausted()

    def test_cost_exhaustion(self):
        tracker = BudgetTracker(BudgetConfig(max_api_cost=10.0))
        tracker.add_cost(10.0)
        assert tracker.is_exhausted()

    def test_experiment_count_tracking(self):
        tracker = BudgetTracker(BudgetConfig(max_experiments=5))
        tracker.record_experiment()
        tracker.record_experiment()
        assert tracker.experiments_run == 2
        assert tracker.remaining_experiments == 3
        assert not tracker.is_exhausted()

    def test_experiment_count_exhaustion(self):
        tracker = BudgetTracker(BudgetConfig(max_experiments=2))
        tracker.record_experiment()
        tracker.record_experiment()
        assert tracker.is_exhausted()

    def test_time_exhaustion(self):
        tracker = BudgetTracker(BudgetConfig(max_wall_time_seconds=1.0))
        tracker.start()
        # Simulate elapsed time by backdating start
        tracker._start_time = time.monotonic() - 2.0
        assert tracker.is_exhausted()

    def test_time_not_started(self):
        tracker = BudgetTracker(BudgetConfig(max_wall_time_seconds=100.0))
        # Not started yet, should not be exhausted
        assert not tracker.is_exhausted()

    def test_plateau_tracking(self):
        tracker = BudgetTracker(BudgetConfig(max_plateau_iterations=3))
        tracker.record_no_improvement()
        tracker.record_no_improvement()
        assert not tracker.is_exhausted()
        tracker.record_no_improvement()
        assert tracker.is_exhausted()

    def test_plateau_resets_on_improvement(self):
        tracker = BudgetTracker(BudgetConfig(max_plateau_iterations=3))
        tracker.record_no_improvement()
        tracker.record_no_improvement()
        tracker.record_improvement()
        assert tracker.consecutive_no_improvement == 0
        tracker.record_no_improvement()
        assert not tracker.is_exhausted()

    def test_multiple_limits_any_exhausts(self):
        tracker = BudgetTracker(BudgetConfig(max_api_cost=100.0, max_experiments=2))
        tracker.record_experiment()
        tracker.record_experiment()
        # Experiments exhausted, even though cost is fine
        assert tracker.is_exhausted()

    def test_snapshot(self):
        tracker = BudgetTracker(BudgetConfig(max_api_cost=50.0, max_experiments=100))
        tracker.add_cost(5.0)
        tracker.record_experiment()
        snap = tracker.snapshot()
        assert snap["spent_usd"] == 5.0
        assert snap["remaining_cost"] == 45.0
        assert snap["experiments_run"] == 1
        assert snap["remaining_experiments"] == 99
