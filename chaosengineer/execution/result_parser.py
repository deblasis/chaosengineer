"""Reads and validates experiment result JSON files."""

from __future__ import annotations

import json
from pathlib import Path

from chaosengineer.core.models import ExperimentResult


class ResultParser:
    """Reads result.json, validates, and builds ExperimentResult."""

    def parse(
        self,
        result_file: Path,
        experiment_id: str,
        duration_seconds: float,
    ) -> ExperimentResult:
        """Parse a result file into an ExperimentResult.

        Returns an error ExperimentResult if the file is missing, malformed,
        or missing required fields. Never raises.
        """
        if not result_file.exists():
            return ExperimentResult(
                primary_metric=0.0,
                duration_seconds=duration_seconds,
                error_message=f"Result file not found for {experiment_id}: {result_file}",
            )

        try:
            raw = result_file.read_text()
            data = json.loads(raw)
        except (json.JSONDecodeError, OSError) as e:
            return ExperimentResult(
                primary_metric=0.0,
                duration_seconds=duration_seconds,
                error_message=f"Failed to parse result for {experiment_id}: {e}",
            )

        if "primary_metric" not in data:
            return ExperimentResult(
                primary_metric=0.0,
                duration_seconds=duration_seconds,
                error_message=f"Missing primary_metric in result for {experiment_id}",
            )

        try:
            primary = float(data["primary_metric"])
        except (TypeError, ValueError) as e:
            return ExperimentResult(
                primary_metric=0.0,
                duration_seconds=duration_seconds,
                error_message=f"Invalid primary_metric for {experiment_id}: {e}",
            )

        return ExperimentResult(
            primary_metric=primary,
            secondary_metrics=data.get("secondary_metrics", {}),
            artifacts=data.get("artifacts", []),
            commit_hash=data.get("commit_hash"),
            duration_seconds=duration_seconds,
            error_message=data.get("error_message"),
        )
