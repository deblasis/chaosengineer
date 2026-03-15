"""LLM harness abstraction for decision making."""

from __future__ import annotations

from pathlib import Path

from chaosengineer.llm.decision_maker import LLMDecisionMaker
from chaosengineer.llm.harness import LLMHarness, Usage
from chaosengineer.workloads.parser import WorkloadSpec

__all__ = ["LLMDecisionMaker", "LLMHarness", "Usage", "create_decision_maker"]


def create_decision_maker(
    backend: str,
    spec: WorkloadSpec,
    work_dir: Path,
) -> LLMDecisionMaker:
    """Create an LLMDecisionMaker with the specified backend.

    Args:
        backend: "claude-code" (default, uses subscription) or "sdk" (uses API key)
        spec: Workload specification for prompt context
        work_dir: Directory for LLM output files (must exist)
    """
    if backend == "claude-code":
        from chaosengineer.llm.claude_code import ClaudeCodeHarness
        harness = ClaudeCodeHarness()
    elif backend == "sdk":
        from chaosengineer.llm.sdk import SDKHarness
        harness = SDKHarness()
    else:
        raise ValueError(f"Unknown backend: '{backend}'. Use 'claude-code' or 'sdk'.")

    return LLMDecisionMaker(harness, spec, work_dir)
