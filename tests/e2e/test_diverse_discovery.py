"""E2E test: diverse dimension discovery integration.

Validates that the coordinator discovers options for DIVERSE dimensions
before running experiments, using a mixed workload with DIRECTIONAL,
ENUM, and DIVERSE dimension types.
"""

import pytest

from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.models import (
    Baseline,
    BudgetConfig,
    DimensionSpec,
    DimensionType,
    ExperimentResult,
)
from chaosengineer.core.interfaces import DimensionPlan
from chaosengineer.metrics.logger import EventLogger
from chaosengineer.testing.simulator import ScriptedDecisionMaker
from chaosengineer.testing.executor import ScriptedExecutor
from chaosengineer.workloads.parser import WorkloadSpec


def _make_spec():
    return WorkloadSpec(
        name="diverse-discovery-test",
        context="Testing diverse dimension discovery with mixed dimension types.",
        primary_metric="val_bpb",
        metric_direction="lower",
        execution_command="echo test",
        workers_available=1,
        budget=BudgetConfig(max_experiments=5),
        dimensions=[
            DimensionSpec(name="lr", dim_type=DimensionType.DIRECTIONAL, current_value=0.04),
            DimensionSpec(name="activation", dim_type=DimensionType.ENUM, options=["relu", "gelu"]),
            DimensionSpec(name="optimizer", dim_type=DimensionType.DIVERSE),
        ],
    )


PLANS = [
    DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}]),
    DimensionPlan(dimension_name="optimizer", values=[{"optimizer": "adam"}]),
    DimensionPlan(dimension_name="optimizer", values=[{"optimizer": "sgd"}]),
    DimensionPlan(dimension_name="activation", values=[{"activation": "gelu"}]),
    DimensionPlan(dimension_name="optimizer", values=[{"optimizer": "rmsprop"}]),
]

RESULTS = {
    "exp-0-0": ExperimentResult(primary_metric=0.92),  # lr=0.02 — breakthrough
    "exp-1-0": ExperimentResult(primary_metric=0.88),  # adam — breakthrough
    "exp-2-0": ExperimentResult(primary_metric=0.90),  # sgd — no breakthrough
    "exp-3-0": ExperimentResult(primary_metric=0.85),  # gelu — breakthrough
    "exp-4-0": ExperimentResult(primary_metric=0.87),  # rmsprop — no breakthrough
}


class TestDiverseDimensionDiscovery:
    """Full E2E test: diverse discovery + mixed dimension experiments."""

    def _build_coordinator(self, tmp_path):
        spec = _make_spec()
        dm = ScriptedDecisionMaker(
            PLANS,
            diverse_options={"optimizer": ["adam", "sgd", "rmsprop"]},
        )
        logger = EventLogger(tmp_path / "events.jsonl")
        coordinator = Coordinator(
            spec=spec,
            decision_maker=dm,
            executor=ScriptedExecutor(RESULTS),
            logger=logger,
            budget=BudgetTracker(spec.budget),
            initial_baseline=Baseline(commit="HEAD", metric_value=1.0, metric_name="val_bpb"),
        )
        return coordinator, spec, logger

    def test_diverse_options_discovered(self, tmp_path):
        coordinator, spec, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        # DIVERSE dimension got its options populated
        diverse_dim = [d for d in spec.dimensions if d.dim_type == DimensionType.DIVERSE][0]
        assert diverse_dim.options == ["adam", "sgd", "rmsprop"]

    def test_discovery_event_logged(self, tmp_path):
        coordinator, spec, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        events = logger.read_events(event_type="diverse_discovered")
        assert len(events) == 1
        assert events[0]["dimension"] == "optimizer"
        assert events[0]["options"] == ["adam", "sgd", "rmsprop"]
        assert events[0]["count"] == 3

    def test_discovery_event_ordering(self, tmp_path):
        coordinator, spec, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        events = logger.read_events()
        event_types = [e["event"] for e in events]

        run_started_idx = event_types.index("run_started")
        discovered_idx = event_types.index("diverse_discovered")
        first_iteration_idx = event_types.index("iteration_started")

        assert run_started_idx < discovered_idx < first_iteration_idx

    def test_all_experiments_run(self, tmp_path):
        coordinator, spec, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        assert coordinator.budget.experiments_run == 5

    def test_breakthroughs(self, tmp_path):
        coordinator, spec, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        events = logger.read_events(event_type="breakthrough")
        # Breakthroughs: 0.92 < 1.0, 0.88 < 0.92, 0.85 < 0.88
        assert len(events) == 3
        assert coordinator.best_baseline.metric_value == pytest.approx(0.85)

    def test_non_diverse_dims_unchanged(self, tmp_path):
        coordinator, spec, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        lr_dim = [d for d in spec.dimensions if d.name == "lr"][0]
        act_dim = [d for d in spec.dimensions if d.name == "activation"][0]

        # DIRECTIONAL and ENUM dimensions were not touched by discovery
        assert lr_dim.options is None
        assert act_dim.options == ["relu", "gelu"]
