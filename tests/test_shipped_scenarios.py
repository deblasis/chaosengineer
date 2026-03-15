"""Tests that run all shipped scenarios and verify expectations pass."""

import pytest
from pathlib import Path
from chaosengineer.testing.runner import ScenarioRunner

SCENARIOS_DIR = Path(__file__).parent.parent / "chaosengineer" / "testing" / "scenarios"


def _scenario_files():
    return sorted(SCENARIOS_DIR.glob("*.yaml"))


@pytest.mark.parametrize("scenario_path", _scenario_files(), ids=lambda p: p.stem)
def test_shipped_scenario(scenario_path, tmp_output_dir):
    runner = ScenarioRunner(output_dir=tmp_output_dir)
    result = runner.run_scenario(scenario_path)

    assert result.passed, (
        f"Scenario '{result.scenario_name}' failed:\n"
        + "\n".join(f"  - {e}" for e in result.errors)
    )
