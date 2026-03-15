# tests/e2e/test_tmux_pause.py
"""E2E test: Ctrl+C pause via tmux.

Requires: tmux installed, chaos CLI available.
Skipped if tmux is not available.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest


pytestmark = pytest.mark.e2e

TMUX = shutil.which("tmux")
CHAOS = shutil.which("chaos")


@pytest.mark.skipif(not TMUX, reason="tmux not installed")
@pytest.mark.skipif(not CHAOS, reason="chaos CLI not in PATH")
class TestTmuxPause:
    def test_ctrl_c_pause_menu_and_pause(self, tmp_path):
        """Full signal chain: Ctrl+C → menu → Pause → run_paused event."""
        session = f"test-pause-{time.monotonic_ns()}"
        output_dir = tmp_path / "output"
        workload = tmp_path / "workload.md"

        # Create a trivial workload spec that will run for a while
        workload.write_text(
            "# Test Workload\n"
            "## Primary Metric\n"
            "loss (lower)\n"
            "## Budget\n"
            "max_experiments: 10\n"
            "## Execution Command\n"
            "sleep 30 && echo '{\"loss\": 1.0}'\n"
            "## Dimensions\n"
            "- lr: directional, current=0.01\n"
        )

        try:
            # Start chaos run in tmux
            subprocess.run(
                [TMUX, "new-session", "-d", "-s", session,
                 CHAOS, "run", str(workload), "--output-dir", str(output_dir),
                 "--executor", "subagent", "--llm-backend", "claude-code"],
                check=True,
            )

            # Wait for status line to appear (run started)
            events_path = output_dir / "events.jsonl"
            for _ in range(30):
                time.sleep(1)
                if events_path.exists():
                    break
            else:
                pytest.fail("events.jsonl never created")

            # Send Ctrl+C
            subprocess.run([TMUX, "send-keys", "-t", session, "C-c"], check=True)
            time.sleep(2)

            # Send "P" to select Pause
            subprocess.run([TMUX, "send-keys", "-t", session, "P"], check=True)
            time.sleep(3)

            # Verify run_paused event
            events = []
            with open(events_path) as f:
                for line in f:
                    events.append(json.loads(line.strip()))

            paused = [e for e in events if e.get("event") == "run_paused"]
            assert len(paused) >= 1, f"Expected run_paused event, got events: {[e['event'] for e in events]}"
            assert paused[0]["reason"] == "user_requested"

        finally:
            # Cleanup tmux session
            subprocess.run(
                [TMUX, "kill-session", "-t", session],
                capture_output=True,
            )
