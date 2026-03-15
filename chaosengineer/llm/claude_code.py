"""ClaudeCodeHarness — runs claude -p as a subprocess."""

from __future__ import annotations

import subprocess
from pathlib import Path

from chaosengineer.llm.harness import LLMHarness, extract_json


class ClaudeCodeHarness(LLMHarness):
    """Sends prompts via Claude Code's -p flag.

    Default backend — uses the user's Claude Code subscription.
    No cost tracking (flat rate).
    """

    def __init__(self, model: str | None = None):
        self._model = model

    def complete(self, system: str, user: str, output_file: Path) -> dict:
        prompt = (
            f"{system}\n\n{user}\n\n"
            f"Write ONLY a JSON object to the file: {output_file}\n"
            f"Do not write any other text to the file — only valid JSON."
        )

        cmd = ["claude", "-p", prompt, "--allowedTools", "Write"]
        if self._model:
            cmd.extend(["--model", self._model])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            raise RuntimeError(
                f"Claude Code failed (exit {result.returncode}): {result.stderr[:500]}"
            )

        if not output_file.exists():
            raise FileNotFoundError(
                f"Claude Code did not write output file: {output_file}"
            )

        raw = output_file.read_text()
        return extract_json(raw)
