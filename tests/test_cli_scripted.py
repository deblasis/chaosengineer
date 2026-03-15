"""Tests for CLI scripted decision maker support."""

import json
import pytest
from pathlib import Path
from textwrap import dedent

from chaosengineer.cli import _execute_run


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
