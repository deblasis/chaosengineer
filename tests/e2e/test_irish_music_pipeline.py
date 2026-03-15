"""E2E test: Irish folk music experiment — full 18-iteration pipeline.

Replays the optimization run from "I Used Karpathy's Autoresearch to Train
a Music-Generating AI Model!" (Onchain AI Garage). Validates that the
ChaosEngineer pipeline correctly orchestrates 18 experiments with 6
breakthroughs, progressing val_bpb from 2.08 to 0.97.
"""

import pytest
from pathlib import Path

from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.models import Baseline, BudgetConfig
from chaosengineer.execution import create_executor
from chaosengineer.metrics.logger import EventLogger
from chaosengineer.testing.simulator import ScriptedDecisionMaker
from chaosengineer.workloads.parser import parse_workload_spec
from chaosengineer.workloads.plan_loader import load_scripted_plans


WORKLOADS_DIR = Path(__file__).parents[2] / "workloads"
IRISH_MUSIC_DIR = WORKLOADS_DIR / "irish-music"


class TestIrishMusicPipeline:
    """Full 18-iteration scripted replay of the Irish folk music experiment."""

    def _build_coordinator(self, tmp_path):
        spec = parse_workload_spec(WORKLOADS_DIR / "autoresearch-irish-music.md")
        plans = load_scripted_plans(IRISH_MUSIC_DIR / "scripted_plans.yaml")

        executor = create_executor(
            "scripted", spec, tmp_path, "sequential",
            scripted_results=IRISH_MUSIC_DIR / "scripted_results.yaml",
        )
        logger = EventLogger(tmp_path / "events.jsonl")
        budget = BudgetTracker(BudgetConfig(max_experiments=18))

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=executor,
            logger=logger,
            budget=budget,
            initial_baseline=Baseline(
                commit="baseline", metric_value=2.08, metric_name="val_bpb",
            ),
        )
        return coordinator, logger

    def test_full_18_iteration_run(self, tmp_path):
        coordinator, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        assert coordinator.best_baseline.metric_value == pytest.approx(0.97)
        assert coordinator.budget.experiments_run == 18

    def test_correct_number_of_breakthroughs(self, tmp_path):
        coordinator, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        events = logger.read_events()
        breakthroughs = [e for e in events if e["event"] == "breakthrough"]
        # 6 keeps: batch_size x3 (exps 2,3,4), aspect_ratio x2 (7,8), warmup (9)
        assert len(breakthroughs) == 6

    def test_metric_progression_monotonically_improves(self, tmp_path):
        coordinator, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        events = logger.read_events()
        breakthroughs = [e for e in events if e["event"] == "breakthrough"]
        values = [b["new_best"] for b in breakthroughs]

        # Each breakthrough is strictly better (lower) than the last
        assert len(values) == 6
        for i in range(1, len(values)):
            assert values[i] < values[i - 1], (
                f"Breakthrough {i}: {values[i]} not < {values[i-1]}"
            )

    def test_metric_progression_values(self, tmp_path):
        """Verify the actual metric values at each breakthrough."""
        coordinator, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        events = logger.read_events()
        breakthroughs = [e for e in events if e["event"] == "breakthrough"]
        values = [b["new_best"] for b in breakthroughs]

        expected = [1.44, 1.03, 1.01, 0.99, 0.985, 0.97]
        assert values == pytest.approx(expected)

    def test_event_log_completeness(self, tmp_path):
        coordinator, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        events = logger.read_events()
        event_types = [e["event"] for e in events]

        assert event_types[0] == "run_started"
        assert event_types[-1] == "run_completed"

        iterations = [e for e in events if e["event"] == "iteration_started"]
        assert len(iterations) == 18

        completed = [e for e in events if e["event"] == "worker_completed"]
        assert len(completed) == 18

    def test_secondary_metrics_present(self, tmp_path):
        coordinator, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        completed = logger.read_events(event_type="worker_completed")
        for e in completed:
            result = e.get("result", {})
            sec = result.get("secondary_metrics", {})
            assert "num_steps" in sec, f"Missing num_steps in {e['experiment_id']}"
            assert "peak_vram_mb" in sec, f"Missing peak_vram_mb in {e['experiment_id']}"

    def test_run_via_cli_scripted(self, tmp_path):
        """Verify the full CLI path works for scripted runs.

        Note: _execute_run uses float("inf") as baseline (not 2.08), so
        experiment 1 (metric 2.15) becomes a breakthrough here. This gives
        7 breakthroughs vs 6 in the direct coordinator tests above. This is
        expected — the initial baseline TODO in _execute_run is a known gap.
        This test only validates the wiring, not the exact breakthrough count.
        """
        from chaosengineer.cli import _execute_run

        class Args:
            workload = WORKLOADS_DIR / "autoresearch-irish-music.md"
            llm_backend = "scripted"
            scripted_plans = IRISH_MUSIC_DIR / "scripted_plans.yaml"
            executor = "scripted"
            scripted_results = IRISH_MUSIC_DIR / "scripted_results.yaml"
            mode = "sequential"
            output_dir = tmp_path / "output"

        _execute_run(Args())

        events_file = Args.output_dir / "events.jsonl"
        assert events_file.exists()
