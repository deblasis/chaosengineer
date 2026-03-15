"""LLM harness abstraction — transport layer for LLM calls."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Usage:
    """Token usage and cost from an LLM call."""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


class LLMHarness(ABC):
    """Abstract base for LLM transport.

    Implementations handle how to send a prompt and get a response.
    All harnesses must write the response JSON to output_file for audit.
    The caller guarantees the parent directory of output_file exists.
    """

    @abstractmethod
    def complete(self, system: str, user: str, output_file: Path) -> dict:
        """Send prompt to LLM, return parsed JSON dict."""

    @property
    def last_usage(self) -> Usage:
        """Token/cost data from the most recent call. Override in subclasses that track cost."""
        return Usage()


def extract_json(text: str) -> dict:
    """Extract the first JSON object from text that may contain prose or code fences.

    Raises ValueError if no valid JSON object is found.
    """
    # Try parsing the whole string first (fast path for pure JSON)
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    # Try extracting from code fences
    fence_match = re.search(r"```(?:json)?\s*\n(\{.*?\})\s*\n```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Find the first { and try to parse from there
    for match in re.finditer(r"\{", text):
        start = match.start()
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"No JSON object found in response: {text[:200]}")
