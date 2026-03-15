"""End-to-end tests: full pipeline with scripted executor."""

import json
import pytest
from pathlib import Path

from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.interfaces import DimensionPlan
from chaosengineer.core.models import Baseline, BudgetConfig
from chaosengineer.execution import create_executor
from chaosengineer.metrics.logger import EventLogger
from chaosengineer.testing.simulator import ScriptedDecisionMaker
from chaosengineer.workloads.parser import parse_workload_spec


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestScriptedPipelineFromFile:
    """Full pipeline using scripted results from a single YAML file."""

    def test_full_run(self, tmp_path):
        spec = parse_workload_spec(FIXTURES_DIR / "simple_workload.md")
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}, {"lr": 0.08}]),
        ]

        executor = create_executor(
            "scripted", spec, tmp_path, "sequential",
            scripted_results=FIXTURES_DIR / "simple_results.yaml",
        )

        logger = EventLogger(tmp_path / "events.jsonl")
        budget = BudgetTracker(spec.budget)

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=executor,
            logger=logger,
            budget=budget,
            initial_baseline=Baseline(commit="abc", metric_value=0.97, metric_name="val_bpb"),
        )

        coordinator.run()

        # Verify results
        assert coordinator.best_baseline.metric_value == 0.91
        assert coordinator.budget.experiments_run == 2

        # Verify event log
        events = logger.read_events()
        assert any(e["event"] == "run_started" for e in events)
        assert any(e["event"] == "run_completed" for e in events)
        assert any(e["event"] == "breakthrough" for e in events)
        assert any(e["event"] == "worker_completed" for e in events)


class TestScriptedPipelineFromFolder:
    """Full pipeline using scripted results from a folder of YAML files."""

    def test_full_run_from_folder(self, tmp_path):
        spec = parse_workload_spec(FIXTURES_DIR / "simple_workload.md")
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}, {"lr": 0.08}]),
        ]

        executor = create_executor(
            "scripted", spec, tmp_path, "sequential",
            scripted_results=FIXTURES_DIR / "results_folder",
        )

        logger = EventLogger(tmp_path / "events.jsonl")
        budget = BudgetTracker(spec.budget)

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=executor,
            logger=logger,
            budget=budget,
            initial_baseline=Baseline(commit="abc", metric_value=0.97, metric_name="val_bpb"),
        )

        coordinator.run()

        assert coordinator.best_baseline.metric_value == 0.91
        assert coordinator.budget.experiments_run == 2


class TestBudgetExhaustion:
    """Verify budget limits terminate the run correctly."""

    def test_max_experiments_stops_run(self, tmp_path):
        spec = parse_workload_spec(FIXTURES_DIR / "simple_workload.md")
        # Budget is max 4 experiments from workload, but only provide 2 results
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}, {"lr": 0.08}]),
        ]

        executor = create_executor(
            "scripted", spec, tmp_path, "sequential",
            scripted_results=FIXTURES_DIR / "simple_results.yaml",
        )

        logger = EventLogger(tmp_path / "events.jsonl")
        budget = BudgetTracker(BudgetConfig(max_experiments=2))

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=executor,
            logger=logger,
            budget=budget,
            initial_baseline=Baseline(commit="abc", metric_value=0.97, metric_name="val_bpb"),
        )

        coordinator.run()

        assert coordinator.budget.experiments_run == 2
        assert coordinator.budget.is_exhausted()

        # Verify run_completed event
        completed = logger.read_events(event_type="run_completed")
        assert len(completed) == 1


class TestEventLogCompleteness:
    """Verify the event log contains all expected events in order."""

    def test_event_sequence(self, tmp_path):
        spec = parse_workload_spec(FIXTURES_DIR / "simple_workload.md")
        plans = [
            DimensionPlan(dimension_name="lr", values=[{"lr": 0.02}, {"lr": 0.08}]),
        ]

        executor = create_executor(
            "scripted", spec, tmp_path, "sequential",
            scripted_results=FIXTURES_DIR / "simple_results.yaml",
        )

        logger = EventLogger(tmp_path / "events.jsonl")
        budget = BudgetTracker(spec.budget)

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=executor,
            logger=logger,
            budget=budget,
            initial_baseline=Baseline(commit="abc", metric_value=0.97, metric_name="val_bpb"),
        )

        coordinator.run()

        events = logger.read_events()
        event_types = [e["event"] for e in events]

        # Must start with run_started and end with run_completed
        assert event_types[0] == "run_started"
        assert event_types[-1] == "run_completed"

        # Must have iteration_started
        assert "iteration_started" in event_types

        # Must have worker_completed for each successful experiment
        completed = [e for e in events if e["event"] == "worker_completed"]
        assert len(completed) == 2


class TestAutoresearchScenario:
    """The original autoresearch workload run e2e with scripted results."""

    def test_autoresearch_workload(self, tmp_path):
        sample_workload = Path(__file__).parents[1] / "fixtures" / "sample_workload.md"
        spec = parse_workload_spec(sample_workload)
        plans = [
            DimensionPlan(
                dimension_name="learning_rate",
                values=[{"learning_rate": 0.02}, {"learning_rate": 0.08}],
            ),
        ]

        executor = create_executor(
            "scripted", spec, tmp_path, "sequential",
            scripted_results=FIXTURES_DIR / "autoresearch_results.yaml",
        )

        logger = EventLogger(tmp_path / "events.jsonl")
        budget = BudgetTracker(spec.budget)

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=executor,
            logger=logger,
            budget=budget,
            initial_baseline=Baseline(
                commit="abc", metric_value=0.97, metric_name="val_bpb",
            ),
        )

        coordinator.run()

        assert coordinator.best_baseline.metric_value == 0.92
        assert coordinator.budget.experiments_run == 2

        # Verify secondary metrics are in the event log
        completed = logger.read_events(event_type="worker_completed")
        assert any(
            "train_loss" in e.get("result", {}).get("secondary_metrics", {})
            for e in completed
        )
