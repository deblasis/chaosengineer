"""DecisionLogger — persists LLM reasoning for observability."""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any


class DecisionLogger:
    """Writes LLM decision reasoning to decisions.jsonl. Write-only, not used by resume."""

    def __init__(self, output_dir: Path):
        self.path = output_dir / "decisions.jsonl"

    def log_dimension_selected(self, dimension: str, reasoning: str, alternatives: list[str]) -> None:
        self._append({"type": "dimension_selected", "dimension": dimension,
                      "reasoning": reasoning, "alternatives": alternatives})

    def log_results_evaluated(self, dimension: str, reasoning: str, winner: str | None,
                              metrics: dict[str, Any]) -> None:
        self._append({"type": "results_evaluated", "dimension": dimension,
                      "reasoning": reasoning, "winner": winner, "metrics": metrics})

    def log_diverse_options(self, dimension: str, reasoning: str, options: list[str]) -> None:
        self._append({"type": "diverse_options_generated", "dimension": dimension,
                      "reasoning": reasoning, "options": options})

    def _append(self, entry: dict[str, Any]) -> None:
        entry["timestamp"] = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
