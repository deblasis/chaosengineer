"""Tests for StatusDisplay — progress output formatting."""

from __future__ import annotations

import io
from unittest.mock import patch, MagicMock

import pytest

from chaosengineer.core.status import StatusDisplay
from chaosengineer.core.models import BudgetConfig


class TestStatusFormatting:
    def test_format_progress_line(self):
        sd = StatusDisplay()
        line = sd._format_progress(
            iteration=1, completed=2, total=4,
            cost=0.31, elapsed=222.0,
        )
        assert "iter 1" in line
        assert "2/4 workers done" in line
        assert "$0.31" in line
        assert "00:03:42" in line
        assert "Ctrl+C to pause" in line

    def test_on_worker_done_writes_stderr(self):
        sd = StatusDisplay()
        sd._iteration = 1
        sd._cost = 0.31
        sd._start_time = 0.0
        buf = io.StringIO()
        with patch("chaosengineer.core.status.time") as mock_time, \
             patch("sys.stderr", buf):
            mock_time.monotonic.return_value = 10.0
            task = MagicMock()
            result = MagicMock()
            result.error_message = None
            result.cost_usd = 0.0
            sd.on_worker_done(task, result, 2, 4)
        output = buf.getvalue()
        assert "2/4 workers done" in output

    def test_on_iteration_done_prints_newline(self):
        sd = StatusDisplay()
        sd._iteration = 1
        sd._cost = 0.50
        sd._start_time = 0.0
        buf = io.StringIO()
        with patch("chaosengineer.core.status.time") as mock_time, \
             patch("sys.stderr", buf):
            mock_time.monotonic.return_value = 60.0
            sd.on_iteration_done(iteration=1, best_metric=2.5)
        output = buf.getvalue()
        assert "\n" in output
        assert "iter 1" in output

    def test_on_run_start(self):
        sd = StatusDisplay()
        buf = io.StringIO()
        with patch("chaosengineer.core.status.time") as mock_time, \
             patch("sys.stderr", buf):
            mock_time.monotonic.return_value = 0.0
            sd.on_run_start(BudgetConfig(max_experiments=10))
        output = buf.getvalue()
        assert "max_experiments=10" in output
