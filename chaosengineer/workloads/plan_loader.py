"""Load scripted dimension plans from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from chaosengineer.core.interfaces import DimensionPlan


def load_scripted_plans(path: Path) -> list[DimensionPlan]:
    """Load a list of DimensionPlan from a YAML file.

    Expected format:
        plans:
          - dimension_name: depth
            values:
              - depth: 12
          - dimension_name: batch_size
            values:
              - batch_size: 131072
    """
    data = yaml.safe_load(path.read_text())

    if not isinstance(data, dict) or "plans" not in data:
        raise ValueError(f"Missing 'plans' key in {path}")

    raw_plans = data["plans"]
    if not raw_plans:
        raise ValueError(f"No plans found in {path}")

    return [
        DimensionPlan(
            dimension_name=entry["dimension_name"],
            values=entry["values"],
        )
        for entry in raw_plans
    ]
