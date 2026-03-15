"""Tests for workload spec parser."""

import pytest
from pathlib import Path
from chaosengineer.workloads.parser import parse_workload_spec, WorkloadSpec
from chaosengineer.core.models import DimensionType


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestParseWorkloadSpec:
    def test_parse_name(self):
        spec = parse_workload_spec(FIXTURES_DIR / "sample_workload.md")
        assert spec.name == "Neural Network Architecture Search"

    def test_parse_context(self):
        spec = parse_workload_spec(FIXTURES_DIR / "sample_workload.md")
        assert "climbmix-400b" in spec.context

    def test_parse_dimensions(self):
        spec = parse_workload_spec(FIXTURES_DIR / "sample_workload.md")
        dims = {d.name: d for d in spec.dimensions}
        assert "learning_rate" in dims
        assert dims["learning_rate"].dim_type == DimensionType.DIRECTIONAL
        assert dims["learning_rate"].current_value == 0.04
        assert "depth" in dims
        assert dims["depth"].dim_type == DimensionType.DIRECTIONAL
        assert dims["depth"].current_value == 8.0
        assert "activation" in dims
        assert dims["activation"].dim_type == DimensionType.ENUM
        assert dims["activation"].options == ["GeLU", "SiLU", "ReLU"]
        assert "attention_mechanism" in dims
        assert dims["attention_mechanism"].dim_type == DimensionType.DIVERSE

    def test_parse_execution(self):
        spec = parse_workload_spec(FIXTURES_DIR / "sample_workload.md")
        assert "uv run train.py" in spec.execution_command
        assert spec.time_budget_seconds == 300  # 5 minutes

    def test_parse_evaluation(self):
        spec = parse_workload_spec(FIXTURES_DIR / "sample_workload.md")
        assert spec.evaluation_type == "automatic"
        assert spec.primary_metric == "val_bpb"
        assert spec.metric_direction == "lower"
        assert 'grep "^val_bpb:" run.log' in spec.metric_parse_command

    def test_parse_resources(self):
        spec = parse_workload_spec(FIXTURES_DIR / "sample_workload.md")
        assert spec.workers_available == 4

    def test_parse_budget(self):
        spec = parse_workload_spec(FIXTURES_DIR / "sample_workload.md")
        assert spec.budget.max_api_cost == 50.0
        assert spec.budget.max_experiments == 100
        assert spec.budget.max_wall_time_seconds == 28800  # 8h

    def test_parse_constraints(self):
        spec = parse_workload_spec(FIXTURES_DIR / "sample_workload.md")
        assert "train.py" in spec.modifiable_files
        assert "prepare.py" in spec.constraints_text

    def test_parse_from_string(self):
        md = """# Workload: Simple Test

## Context
A simple test workload.

## Experiment Space
- Directional: "value" (currently 10)

## Execution
- Command: `echo hello`
- Time budget per experiment: 1 minute

## Evaluation
- Type: automatic
- Metric: score (higher is better)
- Parse: `grep score output.txt`

## Resources
- Available: 2

## Budget
- Max experiments: 5
"""
        spec = parse_workload_spec(content=md)
        assert spec.name == "Simple Test"
        assert len(spec.dimensions) == 1
        assert spec.metric_direction == "higher"
        assert spec.workers_available == 2

    def test_parse_secondary_metrics(self):
        spec = parse_workload_spec(FIXTURES_DIR / "sample_workload.md")
        assert spec.secondary_metrics == ["train_loss", "perplexity"]

    def test_parse_no_secondary_metrics(self):
        md = """# Workload: Simple
## Experiment Space
## Execution
- Command: `echo`
## Evaluation
- Type: automatic
- Metric: score (higher is better)
## Resources
- Available: 1
"""
        spec = parse_workload_spec(content=md)
        assert spec.secondary_metrics == []

    def test_is_better_lower(self):
        spec = parse_workload_spec(FIXTURES_DIR / "sample_workload.md")
        assert spec.is_better(0.91, 0.95)
        assert not spec.is_better(0.95, 0.91)

    def test_is_better_higher(self):
        md = """# Workload: Score Test

## Experiment Space

## Execution
- Command: `echo`

## Evaluation
- Type: automatic
- Metric: accuracy (higher is better)

## Resources
- Available: 1
"""
        spec = parse_workload_spec(content=md)
        assert spec.is_better(0.95, 0.91)
        assert not spec.is_better(0.91, 0.95)

    def test_parse_hyphenated_dimension_name(self):
        md = """# Workload: Test
## Experiment Space
- Directional: "learning-rate" (currently 0.04)
- Enum: "batch-size" options: 16, 32, 64
- Diverse: "attention-type"
## Execution
- Command: `echo`
## Evaluation
- Metric: score (lower is better)
## Resources
- Available: 1
"""
        spec = parse_workload_spec(content=md)
        dims = {d.name: d for d in spec.dimensions}
        assert "learning-rate" in dims
        assert "batch-size" in dims
        assert "attention-type" in dims

    def test_parse_spaced_dimension_name(self):
        md = """# Workload: Test
## Experiment Space
- Directional: "learning rate" (currently 0.04)
## Execution
- Command: `echo`
## Evaluation
- Metric: score (lower is better)
## Resources
- Available: 1
"""
        spec = parse_workload_spec(content=md)
        assert spec.dimensions[0].name == "learning rate"
