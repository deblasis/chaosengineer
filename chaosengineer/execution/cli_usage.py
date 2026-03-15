"""Parse Claude CLI stream-json output for usage/cost data."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CliUsage:
    """Cost and token usage extracted from Claude CLI output."""

    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0


def parse_cli_usage(stdout: str | None) -> CliUsage:
    """Extract cost/token data from Claude CLI stream-json stdout.

    Scans lines in reverse for the last ``{"type":"result",...}`` event.
    Returns ``CliUsage()`` (all zeros) on any failure — never raises.
    """
    if not stdout:
        return CliUsage()

    # Scan in reverse — result event is the last line
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        if '"type":"result"' not in line and '"type": "result"' not in line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("Failed to parse CLI result line: %s", line[:200])
            continue
        if data.get("type") != "result":
            continue
        usage = data.get("usage", {})
        return CliUsage(
            cost_usd=float(data.get("total_cost_usd", 0.0)),
            tokens_in=int(usage.get("input_tokens", 0)),
            tokens_out=int(usage.get("output_tokens", 0)),
        )

    logger.debug("No result event found in CLI output (%d bytes)", len(stdout))
    return CliUsage()
