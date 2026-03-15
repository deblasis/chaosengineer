"""Tests for CLI scripted decision maker support."""

import json
import pytest
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch, MagicMock

from chaosengineer.cli import _execute_run
from chaosengineer.workloads.parser import WorkloadSpec


class FakeArgs:
    """Minimal args namespace for testing _execute_run."""
    def __init__(self, tmp_path, plans_path, results_path, workload_path):
        self.workload = workload_path
        self.llm_backend = "scripted"
        self.executor = "scripted"
        self.mode = "sequential"
        self.scripted_plans = plans_path
        self.scripted_results = results_path
        self.output_dir = tmp_path / "output"
        self.initial_baseline = None


class TestScriptedBackend:
    """CLI wires ScriptedDecisionMaker when --llm-backend=scripted."""

    def _write_fixtures(self, tmp_path):
        workload = tmp_path / "workload.md"
        workload.write_text(dedent("""\
            # Workload: Test

            ## Context
            Test workload.

            ## Experiment Space
            - Directional: "lr" (currently 0.04)

            ## Execution
            - Command: `echo test`

            ## Evaluation
            - Type: automatic
            - Metric: val_bpb (lower is better)

            ## Baseline
            - Metric value: 1.0

            ## Budget
            - Max experiments: 2
        """))

        plans = tmp_path / "plans.yaml"
        plans.write_text(dedent("""\
            plans:
              - dimension_name: lr
                values:
                  - lr: 0.02
        """))

        results = tmp_path / "results.yaml"
        results.write_text(dedent("""\
            "exp-0-0":
              primary_metric: 0.91
        """))

        return workload, plans, results

    def test_scripted_backend_runs_without_llm(self, tmp_path):
        workload, plans, results = self._write_fixtures(tmp_path)
        args = FakeArgs(tmp_path, plans, results, workload)

        # Should run without any LLM calls
        _execute_run(args)

    def test_scripted_backend_requires_plans(self, tmp_path):
        workload, _, results = self._write_fixtures(tmp_path)
        args = FakeArgs(tmp_path, None, results, workload)

        with pytest.raises(SystemExit):
            _execute_run(args)

    def test_scripted_backend_produces_event_log(self, tmp_path):
        workload, plans, results = self._write_fixtures(tmp_path)
        args = FakeArgs(tmp_path, plans, results, workload)

        _execute_run(args)

        # Verify events were logged with the plan consumed
        events_file = args.output_dir / "events.jsonl"
        assert events_file.exists()
        events = [json.loads(line) for line in events_file.read_text().splitlines()]
        completed = [e for e in events if e["event"] == "worker_completed"]
        assert len(completed) == 1

    def test_cli_baseline_overrides_spec(self, tmp_path):
        """--initial-baseline flag takes priority over spec baseline."""
        workload, plans, results = self._write_fixtures(tmp_path)
        # Add baseline section to workload
        content = workload.read_text()
        workload.write_text(content + "\n## Baseline\n- Metric value: 99.0\n")

        args = FakeArgs(tmp_path, plans, results, workload)
        args.initial_baseline = 5.0  # CLI override

        _execute_run(args)

        events_file = args.output_dir / "events.jsonl"
        events = [json.loads(line) for line in events_file.read_text().splitlines()]
        run_started = [e for e in events if e["event"] == "run_started"][0]
        assert run_started["baseline"]["metric_value"] == 5.0

    def test_spec_baseline_used_when_no_cli_flag(self, tmp_path):
        """Workload spec baseline used when no --initial-baseline flag."""
        workload, plans, results = self._write_fixtures(tmp_path)
        content = workload.read_text()
        workload.write_text(content + "\n## Baseline\n- Metric value: 3.14\n")

        args = FakeArgs(tmp_path, plans, results, workload)
        args.initial_baseline = None

        _execute_run(args)

        events_file = args.output_dir / "events.jsonl"
        events = [json.loads(line) for line in events_file.read_text().splitlines()]
        run_started = [e for e in events if e["event"] == "run_started"][0]
        assert run_started["baseline"]["metric_value"] == pytest.approx(3.14)

    def test_scripted_executor_requires_baseline(self, tmp_path):
        """--executor=scripted without baseline in spec or CLI flag should error."""
        workload, plans, results = self._write_fixtures(tmp_path)
        # Overwrite workload without baseline section
        workload.write_text(dedent("""\
            # Workload: Test

            ## Context
            Test workload.

            ## Experiment Space
            - Directional: "lr" (currently 0.04)

            ## Execution
            - Command: `echo test`

            ## Evaluation
            - Type: automatic
            - Metric: val_bpb (lower is better)

            ## Budget
            - Max experiments: 2
        """))
        args = FakeArgs(tmp_path, plans, results, workload)
        args.initial_baseline = None

        with pytest.raises(SystemExit):
            _execute_run(args)


class TestDetectBaseline:
    """Tests for detect_baseline subprocess auto-detection."""

    def _make_spec(self):
        return WorkloadSpec(
            name="test",
            execution_command="echo test",
            primary_metric="val_bpb",
            metric_direction="lower",
            metric_parse_command="echo 2.08",
        )

    def test_detect_baseline_runs_commands(self):
        from chaosengineer.cli import detect_baseline

        spec = self._make_spec()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="2.08\n")
            result = detect_baseline(spec)

        assert result == pytest.approx(2.08)
        assert mock_run.call_count == 2  # execution + parse

    def test_detect_baseline_exits_on_execution_failure(self):
        from chaosengineer.cli import detect_baseline

        spec = self._make_spec()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="error")
            with pytest.raises(SystemExit):
                detect_baseline(spec)

    def test_detect_baseline_exits_on_unparseable_metric(self):
        from chaosengineer.cli import detect_baseline

        spec = self._make_spec()
        with patch("subprocess.run") as mock_run:
            # First call (execution) succeeds, second call (parse) returns garbage
            mock_run.side_effect = [
                MagicMock(returncode=0),
                MagicMock(returncode=0, stdout="not-a-number\n"),
            ]
            with pytest.raises(SystemExit):
                detect_baseline(spec)

    def test_detect_baseline_exits_on_parse_failure(self):
        from chaosengineer.cli import detect_baseline

        spec = self._make_spec()
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=0),
                MagicMock(returncode=1, stderr="parse error"),
            ]
            with pytest.raises(SystemExit):
                detect_baseline(spec)
