"""Tests for YAML plan loader."""

import pytest
from pathlib import Path
from textwrap import dedent

from chaosengineer.core.interfaces import DimensionPlan
from chaosengineer.workloads.plan_loader import load_scripted_plans


class TestLoadScriptedPlans:
    """Load DimensionPlan list from YAML."""

    def test_loads_single_plan(self, tmp_path):
        yaml_file = tmp_path / "plans.yaml"
        yaml_file.write_text(dedent("""\
            plans:
              - dimension_name: depth
                values:
                  - depth: 12
        """))

        plans = load_scripted_plans(yaml_file)

        assert len(plans) == 1
        assert plans[0].dimension_name == "depth"
        assert plans[0].values == [{"depth": 12}]

    def test_loads_multiple_plans(self, tmp_path):
        yaml_file = tmp_path / "plans.yaml"
        yaml_file.write_text(dedent("""\
            plans:
              - dimension_name: depth
                values:
                  - depth: 12
              - dimension_name: batch_size
                values:
                  - batch_size: 131072
              - dimension_name: batch_size
                values:
                  - batch_size: 32768
        """))

        plans = load_scripted_plans(yaml_file)

        assert len(plans) == 3
        assert plans[0].dimension_name == "depth"
        assert plans[1].dimension_name == "batch_size"
        assert plans[1].values == [{"batch_size": 131072}]
        assert plans[2].values == [{"batch_size": 32768}]

    def test_loads_plan_with_multiple_values(self, tmp_path):
        yaml_file = tmp_path / "plans.yaml"
        yaml_file.write_text(dedent("""\
            plans:
              - dimension_name: learning_rate
                values:
                  - learning_rate: 0.02
                  - learning_rate: 0.08
        """))

        plans = load_scripted_plans(yaml_file)

        assert len(plans) == 1
        assert len(plans[0].values) == 2
        assert plans[0].values[0] == {"learning_rate": 0.02}
        assert plans[0].values[1] == {"learning_rate": 0.08}

    def test_preserves_order(self, tmp_path):
        yaml_file = tmp_path / "plans.yaml"
        yaml_file.write_text(dedent("""\
            plans:
              - dimension_name: alpha
                values:
                  - alpha: 1
              - dimension_name: beta
                values:
                  - beta: 2
              - dimension_name: gamma
                values:
                  - gamma: 3
        """))

        plans = load_scripted_plans(yaml_file)
        names = [p.dimension_name for p in plans]
        assert names == ["alpha", "beta", "gamma"]

    def test_empty_plans_raises(self, tmp_path):
        yaml_file = tmp_path / "plans.yaml"
        yaml_file.write_text("plans: []\n")

        with pytest.raises(ValueError, match="No plans found"):
            load_scripted_plans(yaml_file)

    def test_missing_plans_key_raises(self, tmp_path):
        yaml_file = tmp_path / "plans.yaml"
        yaml_file.write_text("something_else: true\n")

        with pytest.raises(ValueError, match="Missing 'plans' key"):
            load_scripted_plans(yaml_file)

    def test_boolean_and_string_values(self, tmp_path):
        yaml_file = tmp_path / "plans.yaml"
        yaml_file.write_text(dedent("""\
            plans:
              - dimension_name: value_embeddings
                values:
                  - value_embeddings: true
              - dimension_name: window_pattern
                values:
                  - window_pattern: SSSS
        """))

        plans = load_scripted_plans(yaml_file)

        assert plans[0].values == [{"value_embeddings": True}]
        assert plans[1].values == [{"window_pattern": "SSSS"}]
