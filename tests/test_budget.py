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


class TestBudgetFromSnapshot:
    def test_from_snapshot_restores_state(self):
        config = BudgetConfig(max_experiments=20, max_api_cost=10.0)
        tracker = BudgetTracker.from_snapshot(config=config, experiments_run=5, cost_spent=3.50,
                                              elapsed_offset=120.0, consecutive_no_improvement=2)
        assert tracker.experiments_run == 5
        assert tracker.spent_usd == 3.50
        assert tracker.consecutive_no_improvement == 2
        assert tracker.remaining_experiments == 15
        assert tracker.remaining_cost == 6.50

    def test_elapsed_offset_added_to_elapsed_seconds(self):
        config = BudgetConfig(max_wall_time_seconds=300)
        tracker = BudgetTracker.from_snapshot(config=config, experiments_run=0, cost_spent=0.0,
                                              elapsed_offset=100.0, consecutive_no_improvement=0)
        tracker.start()
        assert tracker.elapsed_seconds >= 100.0
        assert tracker.remaining_time <= 200.0

    def test_exhausted_with_offset(self):
        config = BudgetConfig(max_wall_time_seconds=100)
        tracker = BudgetTracker.from_snapshot(config=config, experiments_run=0, cost_spent=0.0,
                                              elapsed_offset=100.0, consecutive_no_improvement=0)
        tracker.start()
        assert tracker.is_exhausted()


class TestExhaustionReason:
    def test_experiment_exhaustion(self):
        config = BudgetConfig(max_experiments=5)
        tracker = BudgetTracker(config)
        for _ in range(5):
            tracker.record_experiment()
        assert tracker.exhaustion_reason == "budget_exhausted"

    def test_cost_exhaustion(self):
        config = BudgetConfig(max_api_cost=10.0)
        tracker = BudgetTracker(config)
        tracker.add_cost(10.0)
        assert tracker.exhaustion_reason == "budget_exhausted"

    def test_plateau_exhaustion(self):
        config = BudgetConfig(max_plateau_iterations=3)
        tracker = BudgetTracker(config)
        for _ in range(3):
            tracker.record_no_improvement()
        assert tracker.exhaustion_reason == "plateau"

    def test_not_exhausted(self):
        config = BudgetConfig(max_experiments=10)
        tracker = BudgetTracker(config)
        assert tracker.exhaustion_reason is None
