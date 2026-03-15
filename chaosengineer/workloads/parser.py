"""Markdown workload spec parser."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from chaosengineer.core.models import BudgetConfig, DimensionSpec, DimensionType


@dataclass
class WorkloadSpec:
    """Parsed workload specification."""
    name: str = ""
    context: str = ""
    dimensions: list[DimensionSpec] = field(default_factory=list)
    execution_command: str = ""
    time_budget_seconds: float | None = None
    evaluation_type: str = "automatic"  # "automatic" | "human"
    primary_metric: str = ""
    metric_direction: str = "lower"  # "lower" | "higher"
    metric_parse_command: str = ""
    secondary_metrics: list[str] = field(default_factory=list)
    workers_available: int = 1
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    modifiable_files: list[str] = field(default_factory=list)
    constraints_text: str = ""
    baseline_metric_value: float | None = None
    raw_markdown: str = ""

    def is_better(self, new_value: float, old_value: float) -> bool:
        if self.metric_direction == "lower":
            return new_value < old_value
        return new_value > old_value


def _extract_sections(markdown: str) -> dict[str, str]:
    """Split markdown into {heading: content} pairs."""
    sections: dict[str, str] = {}
    current_heading = ""
    current_lines: list[str] = []

    for line in markdown.split("\n"):
        heading_match = re.match(r"^##\s+(.+)$", line)
        if heading_match:
            if current_heading:
                sections[current_heading] = "\n".join(current_lines).strip()
            current_heading = heading_match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_heading:
        sections[current_heading] = "\n".join(current_lines).strip()

    return sections


def _parse_name(markdown: str) -> str:
    match = re.search(r"^#\s+Workload:\s*(.+)$", markdown, re.MULTILINE)
    return match.group(1).strip() if match else ""


def _parse_dimensions(text: str) -> list[DimensionSpec]:
    dims = []
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("-"):
            continue

        line = line.lstrip("- ").strip()

        # Directional: "name" (currently value)
        dir_match = re.match(
            r'Directional:\s*"([^"]+)"\s*\(currently\s+([\d.]+)\)', line
        )
        if dir_match:
            dims.append(DimensionSpec(
                name=dir_match.group(1),
                dim_type=DimensionType.DIRECTIONAL,
                current_value=float(dir_match.group(2)),
            ))
            continue

        # Enum: "name" options: A, B, C
        enum_match = re.match(
            r'Enum:\s*"([^"]+)"\s*options:\s*(.+)', line
        )
        if enum_match:
            options = [o.strip() for o in enum_match.group(2).split(",")]
            dims.append(DimensionSpec(
                name=enum_match.group(1),
                dim_type=DimensionType.ENUM,
                options=options,
            ))
            continue

        # Diverse: "name"
        diverse_match = re.match(r'Diverse:\s*"([^"]+)"', line)
        if diverse_match:
            dims.append(DimensionSpec(
                name=diverse_match.group(1),
                dim_type=DimensionType.DIVERSE,
            ))
            continue

    return dims


def _parse_time_budget(text: str) -> float | None:
    """Parse time budget, returns seconds or None if absent."""
    match = re.search(r"Time budget.*?:\s*(\d+)\s*(minutes?|seconds?|hours?)", text, re.IGNORECASE)
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("hour"):
        return value * 3600
    if unit.startswith("minute"):
        return value * 60
    return float(value)


def _parse_command(text: str) -> str:
    match = re.search(r"Command:\s*`([^`]+)`", text)
    return match.group(1) if match else ""


def _parse_evaluation(text: str) -> tuple[str, str, str, str]:
    """Returns (eval_type, metric_name, direction, parse_command)."""
    eval_type = "automatic"
    if re.search(r"Type:\s*human", text, re.IGNORECASE):
        eval_type = "human"

    metric_name = ""
    direction = "lower"
    metric_match = re.search(r"Metric:\s*(\w+)\s*\((\w+)\s+is\s+better\)", text)
    if metric_match:
        metric_name = metric_match.group(1)
        direction = metric_match.group(2).lower()

    parse_cmd = ""
    parse_match = re.search(r"Parse:\s*`([^`]+)`", text)
    if parse_match:
        parse_cmd = parse_match.group(1)

    return eval_type, metric_name, direction, parse_cmd


def _parse_secondary_metrics(text: str) -> list[str]:
    """Parse secondary metric names from evaluation section."""
    match = re.search(r"Secondary metrics?:\s*(.+)", text, re.IGNORECASE)
    if not match:
        return []
    return [m.strip() for m in match.group(1).split(",") if m.strip()]


def _parse_workers_available(text: str) -> int:
    match = re.search(r"Available:\s*(\d+)", text)
    return int(match.group(1)) if match else 1


def _parse_budget(text: str) -> BudgetConfig:
    max_cost = None
    cost_match = re.search(r"Max API cost:\s*\$?([\d.]+)", text)
    if cost_match:
        max_cost = float(cost_match.group(1))

    max_experiments = None
    exp_match = re.search(r"Max experiments:\s*(\d+)", text)
    if exp_match:
        max_experiments = int(exp_match.group(1))

    max_time = None
    time_match = re.search(r"Max wall time:\s*(\d+)\s*(h|hours?|m|minutes?|s|seconds?)", text, re.IGNORECASE)
    if time_match:
        value = int(time_match.group(1))
        unit = time_match.group(2).lower()
        if unit.startswith("h"):
            max_time = value * 3600.0
        elif unit.startswith("m"):
            max_time = value * 60.0
        else:
            max_time = float(value)

    return BudgetConfig(
        max_api_cost=max_cost,
        max_experiments=max_experiments,
        max_wall_time_seconds=max_time,
    )


def _parse_modifiable_files(text: str) -> list[str]:
    files = []
    match = re.search(r"Files workers may modify:\s*(.+)", text)
    if match:
        files = [f.strip() for f in match.group(1).split(",")]
    return files


def _parse_baseline_metric(text: str) -> float | None:
    match = re.search(
        r"Metric\s+value:\s*([-+]?[\d.]+(?:[eE][-+]?\d+)?)", text, re.IGNORECASE
    )
    if not match:
        return None
    return float(match.group(1))


def parse_workload_spec(
    path: Path | str | None = None,
    content: str | None = None,
) -> WorkloadSpec:
    """Parse a markdown workload spec from a file path or string content."""
    if content is None:
        if path is None:
            raise ValueError("Either path or content must be provided")
        content = Path(path).read_text()

    sections = _extract_sections(content)

    # Execution
    exec_section = sections.get("Execution", "")
    command = _parse_command(exec_section)
    time_budget = _parse_time_budget(exec_section)

    # Evaluation
    eval_section = sections.get("Evaluation", "")
    eval_type, metric_name, direction, parse_cmd = _parse_evaluation(eval_section)
    secondary = _parse_secondary_metrics(eval_section)

    # Resources
    resources_section = sections.get("Resources", "")
    workers = _parse_workers_available(resources_section)

    # Budget
    budget_section = sections.get("Budget", "")
    budget = _parse_budget(budget_section)

    # Constraints
    constraints_section = sections.get("Constraints", "")
    modifiable_files = _parse_modifiable_files(constraints_section)

    # Baseline
    baseline_section = sections.get("Baseline", "")
    baseline_value = _parse_baseline_metric(baseline_section)

    return WorkloadSpec(
        name=_parse_name(content),
        context=sections.get("Context", ""),
        dimensions=_parse_dimensions(sections.get("Experiment Space", "")),
        execution_command=command,
        time_budget_seconds=time_budget,
        evaluation_type=eval_type,
        primary_metric=metric_name,
        metric_direction=direction,
        metric_parse_command=parse_cmd,
        secondary_metrics=secondary,
        workers_available=workers,
        budget=budget,
        modifiable_files=modifiable_files,
        constraints_text=constraints_section,
        baseline_metric_value=baseline_value,
        raw_markdown=content,
    )
