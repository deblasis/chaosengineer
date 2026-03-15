"""Tests for LLM harness utilities."""

import json
import pytest
from unittest.mock import patch, MagicMock

from chaosengineer.llm.harness import extract_json
from chaosengineer.llm.claude_code import ClaudeCodeHarness
from chaosengineer.llm.sdk import SDKHarness


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


class TestSDKHarness:
    def _mock_response(self, text, input_tokens=100, output_tokens=50):
        """Build a mock Anthropic API response."""
        content_block = MagicMock()
        content_block.text = text
        response = MagicMock()
        response.content = [content_block]
        response.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
        return response

    def test_happy_path(self, tmp_path):
        output_file = tmp_path / "decision_001.json"
        response_json = '{"dimension_name": "lr", "values": [{"lr": 0.02}]}'
        mock_response = self._mock_response(response_json)

        with patch("chaosengineer.llm.sdk.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = mock_client

            harness = SDKHarness(api_key="test-key")
            result = harness.complete("system prompt", "user prompt", output_file)

        assert result == {"dimension_name": "lr", "values": [{"lr": 0.02}]}
        assert output_file.read_text()  # file was written for audit

    def test_env_vars(self, tmp_path):
        output_file = tmp_path / "decision_001.json"
        mock_response = self._mock_response('{"done": true}')

        with patch("chaosengineer.llm.sdk.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = mock_client

            with patch.dict("os.environ", {
                "ANTHROPIC_API_KEY": "env-key",
                "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
                "ANTHROPIC_MODEL": "glm-4.7",
            }):
                harness = SDKHarness()
                harness.complete("sys", "usr", output_file)

            # Check Anthropic client was created with env base_url
            mock_anthropic.Anthropic.assert_called_with(
                api_key="env-key",
                base_url="https://api.z.ai/api/anthropic",
            )
            # Check model from env was used
            create_call = mock_client.messages.create.call_args
            assert create_call.kwargs["model"] == "glm-4.7"

    def test_cost_tracking(self, tmp_path):
        output_file = tmp_path / "decision_001.json"
        mock_response = self._mock_response('{"done": true}', input_tokens=500, output_tokens=200)

        with patch("chaosengineer.llm.sdk.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = mock_client

            harness = SDKHarness(api_key="test-key")
            harness.complete("sys", "usr", output_file)

        usage = harness.last_usage
        assert usage.tokens_in == 500
        assert usage.tokens_out == 200
        assert usage.cost_usd > 0  # some estimated cost

    def test_no_api_key_raises(self):
        with patch("chaosengineer.llm.sdk.anthropic") as mock_anthropic:
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(ValueError, match="No API key"):
                    SDKHarness()

    def test_api_error_propagates(self, tmp_path):
        output_file = tmp_path / "decision_001.json"

        with patch("chaosengineer.llm.sdk.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.side_effect = Exception("API rate limited")
            mock_anthropic.Anthropic.return_value = mock_client

            harness = SDKHarness(api_key="test-key")
            with pytest.raises(Exception, match="API rate limited"):
                harness.complete("sys", "usr", output_file)

    def test_explicit_params_override_env(self, tmp_path):
        output_file = tmp_path / "decision_001.json"
        mock_response = self._mock_response('{"done": true}')

        with patch("chaosengineer.llm.sdk.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = mock_client

            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key", "ANTHROPIC_MODEL": "env-model"}):
                harness = SDKHarness(api_key="explicit-key", model="explicit-model")
                harness.complete("sys", "usr", output_file)

            mock_anthropic.Anthropic.assert_called_with(api_key="explicit-key", base_url=None)
            create_call = mock_client.messages.create.call_args
            assert create_call.kwargs["model"] == "explicit-model"
