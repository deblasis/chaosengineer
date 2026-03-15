"""LLMDecisionMaker — real LLM-backed implementation of DecisionMaker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chaosengineer.core.interfaces import DecisionMaker, DimensionPlan
from chaosengineer.core.models import Baseline, DimensionSpec
from chaosengineer.llm.harness import LLMHarness
from chaosengineer.workloads.parser import WorkloadSpec

PICK_SYSTEM_PROMPT = """\
You are a coordinator for an automated experimentation framework. Your job is to \
analyze the experiment space and history, then decide which dimension to explore next.

The framework uses coordinate descent: one dimension is explored per iteration, with \
multiple values tested in parallel from the same baseline. You pick the dimension that \
is most likely to yield improvement given what has been tried so far.

Respond with ONLY a JSON object — no other text. Either:
- {"dimension_name": "<name>", "values": [{"<param>": <val>}, ...]} to explore a dimension
- {"done": true} if no dimensions remain worth exploring"""

DISCOVER_SYSTEM_PROMPT = """\
You are generating maximally diverse options for an experiment dimension. Your goal is \
to produce a saturated set: options that are genuinely different from each other, covering \
the full space of reasonable approaches.

Think through what options exist. For each, check: is it truly distinct from the others, \
or just a variation? Keep only those that represent fundamentally different approaches. \
Stop when you cannot think of a genuinely novel option.

Respond with ONLY a JSON object — no other text:
{"options": ["option1", "option2", ...], "saturated": true}"""


class LLMDecisionMaker(DecisionMaker):
    """Real LLM-backed decision maker.

    Owns prompt construction, response parsing, and validation.
    Delegates LLM transport to an LLMHarness.
    """

    def __init__(self, harness: LLMHarness, spec: WorkloadSpec, work_dir: Path,
                 decision_logger: "DecisionLogger | None" = None):
        self.harness = harness
        self.spec = spec
        self.work_dir = work_dir
        self._call_count = 0
        self._prior_context: str | None = None
        self._decision_logger = decision_logger

    @property
    def last_cost_usd(self) -> float:
        """Cost of the most recent LLM call. 0.0 for ClaudeCode harness."""
        return self.harness.last_usage.cost_usd

    def _next_output_file(self) -> Path:
        self._call_count += 1
        return self.work_dir / f"decision_{self._call_count:03d}.json"

    def pick_next_dimension(
        self,
        dimensions: list[DimensionSpec],
        baselines: list[Baseline],
        history: list[dict],
    ) -> DimensionPlan | None:
        user_prompt = self._build_pick_prompt(dimensions, baselines, history)
        output_file = self._next_output_file()

        response = self.harness.complete(PICK_SYSTEM_PROMPT, user_prompt, output_file)

        if response.get("done"):
            return None

        plan = self._validate_pick_response(response, dimensions)

        if self._decision_logger:
            alternatives = [d.name for d in dimensions if d.name != plan.dimension_name]
            self._decision_logger.log_dimension_selected(
                dimension=plan.dimension_name,
                reasoning="",
                alternatives=alternatives,
            )

        return plan

    def discover_diverse_options(
        self, dimension_name: str, context: str
    ) -> list[str]:
        user_prompt = (
            f"Dimension: {dimension_name}\n\n"
            f"Context:\n{context}\n\n"
            f"Workload: {self.spec.name}\n"
            f"{self.spec.context}"
        )
        output_file = self._next_output_file()

        response = self.harness.complete(DISCOVER_SYSTEM_PROMPT, user_prompt, output_file)

        options = response.get("options", [])
        if not options:
            raise ValueError(f"LLM returned no options for dimension '{dimension_name}'")

        if self._decision_logger:
            self._decision_logger.log_diverse_options(
                dimension=dimension_name,
                reasoning="",
                options=options,
            )

        return options

    def set_prior_context(self, context: str) -> None:
        self._prior_context = context

    def _build_pick_prompt(
        self,
        dimensions: list[DimensionSpec],
        baselines: list[Baseline],
        history: list[dict],
    ) -> str:
        parts = []

        if self._prior_context is not None:
            parts.append(self._prior_context)
            parts.append("")

        parts.append(f"Workload: {self.spec.name}")
        if self.spec.context:
            parts.append(f"Context: {self.spec.context}")
        parts.append(f"Metric: {self.spec.primary_metric} ({self.spec.metric_direction} is better)")
        parts.append("")

        parts.append("## Available Dimensions")
        for d in dimensions:
            line = f"- {d.name} (type: {d.dim_type.value})"
            if d.current_value is not None:
                line += f", current: {d.current_value}"
            if d.options:
                line += f", options: {d.options}"
            if d.description:
                line += f" — {d.description}"
            parts.append(line)
        parts.append("")

        parts.append("## Active Baselines")
        for b in baselines:
            parts.append(f"- {b.metric_name}={b.metric_value} (commit: {b.commit})")
        parts.append("")

        if history:
            parts.append("## Experiment History")
            parts.append(json.dumps(history, indent=2, default=str))

        return "\n".join(parts)

    def _validate_pick_response(
        self, response: dict, dimensions: list[DimensionSpec]
    ) -> DimensionPlan:
        dim_name = response.get("dimension_name")
        values = response.get("values")

        known_names = {d.name for d in dimensions}
        if dim_name not in known_names:
            raise ValueError(
                f"Unknown dimension '{dim_name}' in LLM response. "
                f"Known dimensions: {known_names}"
            )

        if not values:
            raise ValueError(
                f"LLM returned empty values for dimension '{dim_name}'"
            )

        return DimensionPlan(dimension_name=dim_name, values=values)
