"""Tests for CLI usage parser."""

import json

import pytest

from chaosengineer.execution.cli_usage import CliUsage, parse_cli_usage


def _result_line(cost: float = 0.5, tokens_in: int = 1000, tokens_out: int = 200) -> str:
    return json.dumps({
        "type": "result",
        "subtype": "success",
        "total_cost_usd": cost,
        "num_turns": 3,
        "usage": {
            "input_tokens": tokens_in,
            "output_tokens": tokens_out,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    })


class TestCliUsageDefaults:
    def test_frozen_dataclass(self):
        usage = CliUsage()
        assert usage.cost_usd == 0.0
        assert usage.tokens_in == 0
        assert usage.tokens_out == 0
        with pytest.raises(AttributeError):
            usage.cost_usd = 1.0


class TestParseCliUsage:
    def test_valid_result_event(self):
        stdout = _result_line(cost=0.123, tokens_in=5000, tokens_out=300)
        usage = parse_cli_usage(stdout)
        assert usage.cost_usd == 0.123
        assert usage.tokens_in == 5000
        assert usage.tokens_out == 300

    def test_result_among_other_events(self):
        lines = [
            '{"type":"system","subtype":"init"}',
            '{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}}',
            _result_line(cost=0.5),
        ]
        usage = parse_cli_usage("\n".join(lines))
        assert usage.cost_usd == 0.5

    def test_error_subtype_still_has_cost(self):
        line = json.dumps({
            "type": "result",
            "subtype": "error",
            "total_cost_usd": 0.08,
            "usage": {"input_tokens": 100, "output_tokens": 10},
        })
        usage = parse_cli_usage(line)
        assert usage.cost_usd == 0.08
        assert usage.tokens_in == 100

    def test_none_input(self):
        assert parse_cli_usage(None) == CliUsage()

    def test_empty_string(self):
        assert parse_cli_usage("") == CliUsage()

    def test_no_result_event(self):
        stdout = '{"type":"assistant","message":{}}\n{"type":"system"}'
        assert parse_cli_usage(stdout) == CliUsage()

    def test_malformed_json(self):
        assert parse_cli_usage("not json at all") == CliUsage()

    def test_result_missing_cost_fields(self):
        line = json.dumps({"type": "result", "subtype": "success"})
        usage = parse_cli_usage(line)
        assert usage.cost_usd == 0.0
        assert usage.tokens_in == 0

    def test_uses_last_result_event(self):
        lines = [
            _result_line(cost=0.1),
            _result_line(cost=0.5),
        ]
        usage = parse_cli_usage("\n".join(lines))
        assert usage.cost_usd == 0.5
