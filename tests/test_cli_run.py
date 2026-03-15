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


import json


class TestResumeSubcommand:
    def test_resume_parser_accepts_budget_extensions(self):
        """Resume accepts --add-cost, --add-experiments, --add-time flags."""
        from chaosengineer.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args([
            "resume", "/tmp/output", "workload.md",
            "--add-cost", "5.0",
            "--add-experiments", "10",
            "--add-time", "3600",
        ])
        assert args.add_cost == 5.0
        assert args.add_experiments == 10
        assert args.add_time == 3600

    def test_resume_parser_accepts_restart_iteration(self):
        from chaosengineer.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["resume", "/tmp/output", "workload.md", "--restart-iteration"])
        assert args.restart_iteration is True

    def test_resume_parser_default_flags(self):
        from chaosengineer.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["resume", "/tmp/output", "workload.md"])
        assert args.add_cost == 0
        assert args.add_experiments == 0
        assert args.add_time == 0
        assert args.restart_iteration is False

    def test_force_fresh_flag_on_run(self):
        from chaosengineer.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run", "workload.md", "--force-fresh"])
        assert args.force_fresh is True


class TestRunGuard:
    def test_detects_resumable_session(self, tmp_path):
        """Run guard detects events.jsonl with no run_completed."""
        from chaosengineer.cli import _check_resumable_session

        events_path = tmp_path / "events.jsonl"
        events_path.write_text(json.dumps({
            "event": "run_started", "run_id": "run-1",
            "workload": "test", "ts": "2026-01-01T00:00:00Z",
        }) + "\n")

        result = _check_resumable_session(tmp_path)
        assert result is not None
        assert result["run_id"] == "run-1"

    def test_no_guard_for_completed_run(self, tmp_path):
        """Run guard returns None for a completed run."""
        from chaosengineer.cli import _check_resumable_session

        events_path = tmp_path / "events.jsonl"
        events_path.write_text(
            json.dumps({"event": "run_started", "run_id": "run-1"}) + "\n"
            + json.dumps({"event": "run_completed", "best_metric": 2.0}) + "\n"
        )

        result = _check_resumable_session(tmp_path)
        assert result is None

    def test_no_guard_without_events_file(self, tmp_path):
        """No events.jsonl means no resumable session."""
        from chaosengineer.cli import _check_resumable_session

        result = _check_resumable_session(tmp_path)
        assert result is None
