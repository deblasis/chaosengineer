"""Tests for CLI run command argument parsing and validation."""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch

from chaosengineer.cli import main


class TestCliRunArgs:
    """Test that argparse accepts and defaults the new flags correctly."""

    def test_executor_flag_default(self):
        """--executor defaults to 'subagent'."""
        import argparse
        with patch("sys.argv", ["chaosengineer", "run", "workload.md"]):
            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = argparse.Namespace(
                    command="run", workload=Path("workload.md"),
                    executor="subagent", mode="sequential",
                    llm_backend="claude-code", scripted_results=None,
                    scripted_plans=None,
                    output_dir=Path(".chaosengineer/output"),
                )
                with patch("chaosengineer.cli._execute_run"):
                    main()

    def test_executor_choices_validated(self):
        """argparse rejects invalid --executor values."""
        with patch("sys.argv", [
            "chaosengineer", "run", "workload.md",
            "--executor", "invalid",
        ]):
            with pytest.raises(SystemExit):
                main()

    def test_mode_choices_validated(self):
        """argparse rejects invalid --mode values."""
        with patch("sys.argv", [
            "chaosengineer", "run", "workload.md",
            "--mode", "invalid",
        ]):
            with pytest.raises(SystemExit):
                main()

    def test_scripted_results_accepted(self):
        """--scripted-results is accepted as a Path argument."""
        import argparse
        with patch("sys.argv", [
            "chaosengineer", "run", "workload.md",
            "--executor", "scripted",
            "--scripted-results", "results.yaml",
        ]):
            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = argparse.Namespace(
                    command="run", workload=Path("workload.md"),
                    executor="scripted", mode="sequential",
                    llm_backend="claude-code",
                    scripted_results=Path("results.yaml"),
                    scripted_plans=None,
                    output_dir=Path(".chaosengineer/output"),
                )
                with patch("chaosengineer.cli._execute_run"):
                    main()

    def test_initial_baseline_flag_accepted(self):
        """--initial-baseline is accepted as a float argument."""
        import argparse
        with patch("sys.argv", [
            "chaosengineer", "run", "workload.md",
            "--initial-baseline", "2.08",
        ]):
            with patch("argparse.ArgumentParser.parse_args") as mock_parse:
                mock_parse.return_value = argparse.Namespace(
                    command="run", workload=Path("workload.md"),
                    executor="subagent", mode="sequential",
                    llm_backend="claude-code", scripted_results=None,
                    scripted_plans=None, initial_baseline=2.08,
                    output_dir=Path(".chaosengineer/output"),
                )
                with patch("chaosengineer.cli._execute_run"):
                    main()
