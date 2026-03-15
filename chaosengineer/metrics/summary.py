"""Per-run summary generation."""

from __future__ import annotations

from typing import Any

from chaosengineer.metrics.logger import EventLogger


def generate_summary(logger: EventLogger) -> dict[str, Any]:
    """Generate a summary dict from the event log."""
    events = logger.read_events()

    completed_events = [e for e in events if e["event"] == "run_completed"]
    run_data = completed_events[-1] if completed_events else {}

    breakthroughs = [e for e in events if e["event"] == "breakthrough"]
    iterations = [e for e in events if e["event"] == "iteration_started"]
    worker_completions = [e for e in events if e["event"] == "worker_completed"]
    failures = [e for e in events if e["event"] == "worker_failed"]

    return {
        "best_metric": run_data.get("best_metric"),
        "total_experiments": run_data.get("total_experiments", len(worker_completions) + len(failures)),
        "total_cost_usd": run_data.get("total_cost_usd", 0),
        "breakthroughs": len(breakthroughs),
        "iterations": len(iterations),
        "failures": len(failures),
    }


def summary_to_text(summary: dict[str, Any]) -> str:
    """Format summary as human-readable text."""
    lines = [
        "=== Run Summary ===",
        f"Best metric:       {summary.get('best_metric', 'N/A')}",
        f"Total experiments:  {summary.get('total_experiments', 0)}",
        f"Breakthroughs:      {summary.get('breakthroughs', 0)}",
        f"Iterations:         {summary.get('iterations', 0)}",
        f"Failures:           {summary.get('failures', 0)}",
        f"Total cost (USD):   ${summary.get('total_cost_usd', 0):.2f}",
    ]
    return "\n".join(lines)
