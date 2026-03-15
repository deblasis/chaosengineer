"""Tests for ResultParser."""

import json
import pytest
from pathlib import Path

from chaosengineer.execution.result_parser import ResultParser


class TestResultParserValidJSON:
    def test_parse_complete_result(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps({
            "primary_metric": 0.91,
            "secondary_metrics": {"train_loss": 1.5},
            "artifacts": ["model.pt"],
            "commit_hash": "abc123",
            "error_message": None,
        }))

        parser = ResultParser()
        result = parser.parse(result_file, "exp-0-0", duration_seconds=42.5)

        assert result.primary_metric == 0.91
        assert result.secondary_metrics == {"train_loss": 1.5}
        assert result.artifacts == ["model.pt"]
        assert result.commit_hash == "abc123"
        assert result.error_message is None
        assert result.duration_seconds == 42.5

    def test_parse_minimal_result(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps({"primary_metric": 0.91}))

        parser = ResultParser()
        result = parser.parse(result_file, "exp-0-0", duration_seconds=10.0)

        assert result.primary_metric == 0.91
        assert result.secondary_metrics == {}
        assert result.artifacts == []
        assert result.commit_hash is None
        assert result.error_message is None

    def test_parse_error_result(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps({
            "primary_metric": 0.0,
            "error_message": "OOM during training",
        }))

        parser = ResultParser()
        result = parser.parse(result_file, "exp-0-0", duration_seconds=5.0)

        assert result.error_message == "OOM during training"

    def test_extra_fields_ignored(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps({
            "primary_metric": 0.91,
            "some_unknown_field": "ignored",
        }))

        parser = ResultParser()
        result = parser.parse(result_file, "exp-0-0", duration_seconds=1.0)

        assert result.primary_metric == 0.91


class TestResultParserErrors:
    def test_missing_file_returns_error_result(self, tmp_path):
        result_file = tmp_path / "nonexistent.json"

        parser = ResultParser()
        result = parser.parse(result_file, "exp-0-0", duration_seconds=0.0)

        assert result.primary_metric == 0.0
        assert result.error_message is not None
        assert "not found" in result.error_message.lower() or "missing" in result.error_message.lower()

    def test_malformed_json_returns_error_result(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text("not valid json {{{")

        parser = ResultParser()
        result = parser.parse(result_file, "exp-0-0", duration_seconds=0.0)

        assert result.primary_metric == 0.0
        assert result.error_message is not None

    def test_missing_primary_metric_returns_error_result(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps({"secondary_metrics": {}}))

        parser = ResultParser()
        result = parser.parse(result_file, "exp-0-0", duration_seconds=0.0)

        assert result.primary_metric == 0.0
        assert result.error_message is not None
        assert "primary_metric" in result.error_message.lower()

    def test_primary_metric_not_numeric_returns_error(self, tmp_path):
        result_file = tmp_path / "result.json"
        result_file.write_text(json.dumps({"primary_metric": "not a number"}))

        parser = ResultParser()
        result = parser.parse(result_file, "exp-0-0", duration_seconds=0.0)

        assert result.error_message is not None
