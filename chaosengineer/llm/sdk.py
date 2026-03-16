"""SDKHarness — uses Anthropic Python SDK for LLM calls."""

from __future__ import annotations

import json
import os
from pathlib import Path

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

from chaosengineer.llm.harness import LLMHarness, Usage, extract_json


# Rough cost per token for estimation (Claude Sonnet 4 pricing).
# Users on alternative providers may have different rates — this is best-effort.
_INPUT_COST_PER_TOKEN = 3.0 / 1_000_000   # $3 per 1M input tokens
_OUTPUT_COST_PER_TOKEN = 15.0 / 1_000_000  # $15 per 1M output tokens


class SDKHarness(LLMHarness):
    """Sends prompts via the Anthropic Python SDK.

    Reads configuration from constructor args or environment variables:
    - ANTHROPIC_API_KEY
    - ANTHROPIC_BASE_URL (for alternative Anthropic-compatible providers,
      e.g. OpenRouter, Z.AI, Kimi)
    - ANTHROPIC_MODEL (default: claude-sonnet-4-20250514)
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        if anthropic is None:
            raise ImportError(
                "The anthropic package is required for SDKHarness. "
                "Install it with: uv pip install 'chaosengineer[sdk]'"
            )

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No API key provided. Pass api_key= or set ANTHROPIC_API_KEY."
            )
        resolved_base = base_url or os.environ.get("ANTHROPIC_BASE_URL") or None

        self._client = anthropic.Anthropic(
            api_key=resolved_key,
            base_url=resolved_base,
        )
        self._model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        self._last_usage = Usage()

    @property
    def last_usage(self) -> Usage:
        return self._last_usage

    def complete(self, system: str, user: str, output_file: Path) -> dict:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        text = response.content[0].text
        tokens_in = response.usage.input_tokens
        tokens_out = response.usage.output_tokens
        cost = (tokens_in * _INPUT_COST_PER_TOKEN) + (tokens_out * _OUTPUT_COST_PER_TOKEN)

        self._last_usage = Usage(
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
        )

        parsed = extract_json(text)

        # Write to output_file for audit trail
        output_file.write_text(json.dumps(parsed, indent=2))

        return parsed
