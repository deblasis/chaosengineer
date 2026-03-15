"""Tests for LLM harness utilities."""

import json
import pytest
from unittest.mock import patch, MagicMock

from chaosengineer.llm.harness import extract_json
from chaosengineer.llm.claude_code import ClaudeCodeHarness


class TestExtractJson:
    def test_pure_json(self):
        raw = '{"dimension_name": "lr", "values": [{"lr": 0.02}]}'
        assert extract_json(raw) == {"dimension_name": "lr", "values": [{"lr": 0.02}]}

    def test_json_in_code_fence(self):
        raw = 'Here is my answer:\n```json\n{"done": true}\n```\n'
        assert extract_json(raw) == {"done": True}

    def test_json_with_surrounding_prose(self):
        raw = 'I think we should explore lr.\n{"dimension_name": "lr", "values": [{"lr": 0.1}]}\nGood luck!'
        assert extract_json(raw) == {"dimension_name": "lr", "values": [{"lr": 0.1}]}

    def test_nested_json(self):
        raw = '{"options": ["a", "b"], "meta": {"count": 2}}'
        result = extract_json(raw)
        assert result == {"options": ["a", "b"], "meta": {"count": 2}}

    def test_no_json_raises(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            extract_json("no json here at all")

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            extract_json("{not valid json at all}")


class TestClaudeCodeHarness:
    def test_happy_path(self, tmp_path):
        output_file = tmp_path / "decision_001.json"
        response_data = {"dimension_name": "lr", "values": [{"lr": 0.02}]}

        def fake_run(args, **kwargs):
            output_file.write_text(json.dumps(response_data))
            return MagicMock(returncode=0, stdout="Done", stderr="")

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            harness = ClaudeCodeHarness()
            result = harness.complete(
                system="You are a coordinator.",
                user="Pick a dimension.",
                output_file=output_file,
            )

        assert result == response_data
        assert mock_run.called
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "claude"
        assert "-p" in call_args

    def test_model_override(self, tmp_path):
        output_file = tmp_path / "decision_001.json"
        response_data = {"done": True}

        def fake_run(args, **kwargs):
            output_file.write_text(json.dumps(response_data))
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=fake_run) as mock_run:
            harness = ClaudeCodeHarness(model="claude-opus-4-6")
            harness.complete("sys", "usr", output_file)

        call_args = mock_run.call_args[0][0]
        assert "--model" in call_args
        assert "claude-opus-4-6" in call_args

    def test_missing_output_file_raises(self, tmp_path):
        output_file = tmp_path / "decision_001.json"

        with patch("subprocess.run", return_value=MagicMock(returncode=0, stdout="", stderr="")):
            harness = ClaudeCodeHarness()
            with pytest.raises(FileNotFoundError):
                harness.complete("sys", "usr", output_file)

    def test_nonzero_exit_raises(self, tmp_path):
        output_file = tmp_path / "decision_001.json"

        with patch("subprocess.run", return_value=MagicMock(returncode=1, stdout="", stderr="error")):
            harness = ClaudeCodeHarness()
            with pytest.raises(RuntimeError, match="Claude Code failed"):
                harness.complete("sys", "usr", output_file)

    def test_timeout_propagates(self, tmp_path):
        import subprocess as sp
        output_file = tmp_path / "decision_001.json"

        with patch("subprocess.run", side_effect=sp.TimeoutExpired(cmd="claude", timeout=300)):
            harness = ClaudeCodeHarness()
            with pytest.raises(sp.TimeoutExpired):
                harness.complete("sys", "usr", output_file)

    def test_last_usage_always_zero(self):
        harness = ClaudeCodeHarness()
        usage = harness.last_usage
        assert usage.cost_usd == 0.0
        assert usage.tokens_in == 0
