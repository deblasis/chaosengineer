"""Tests for the scenario runner."""

import pytest
from pathlib import Path
from chaosengineer.testing.runner import ScenarioRunner, load_scenario, ScenarioResult

SCENARIOS_DIR = Path(__file__).parent.parent / "chaosengineer" / "testing" / "scenarios"


class TestLoadScenario:
    def test_load_breakthrough_scenario(self):
        scenario = load_scenario(SCENARIOS_DIR / "breakthrough.yaml")
        assert scenario["scenario"] == "breakthrough triggers baseline update"
        assert len(scenario["plans"]) == 2
        assert len(scenario["results"]) == 4

    def test_load_from_string(self):
        yaml_str = """
scenario: "test"
initial_baseline:
  commit: "abc"
  metric_value: 1.0
  metric_name: "score"
workload:
  name: "test"
  primary_metric: "score"
  metric_direction: "lower"
  execution_command: "echo"
  workers_available: 1
  budget:
    max_experiments: 1
plans:
  - dimension_name: "x"
    values:
      - { x: 1 }
results:
  "exp-0-0": { primary_metric: 0.9 }
expected:
  final_best_metric: 0.9
  total_experiments: 1
"""
        scenario = load_scenario(content=yaml_str)
        assert scenario["scenario"] == "test"


class TestScenarioRunner:
    def test_run_breakthrough_scenario(self, tmp_output_dir):
        runner = ScenarioRunner(output_dir=tmp_output_dir)
        result = runner.run_scenario(SCENARIOS_DIR / "breakthrough.yaml")

        assert result.passed
        assert result.final_best_metric == 0.88
        assert result.total_experiments == 4

    def test_run_scenario_with_expected_checks(self, tmp_output_dir):
        runner = ScenarioRunner(output_dir=tmp_output_dir)
        result = runner.run_scenario(SCENARIOS_DIR / "breakthrough.yaml")

        assert result.passed
        # Check expectations from YAML
        assert result.final_best_metric == result.expected["final_best_metric"]
        assert result.total_experiments == result.expected["total_experiments"]

    def test_run_produces_event_log(self, tmp_output_dir):
        runner = ScenarioRunner(output_dir=tmp_output_dir)
        result = runner.run_scenario(SCENARIOS_DIR / "breakthrough.yaml")

        assert result.event_log_path.exists()
        # Should have events
        import json
        events = [json.loads(line) for line in result.event_log_path.read_text().strip().split("\n")]
        assert len(events) > 0
        assert events[0]["event"] == "run_started"
