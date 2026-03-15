"""Tests for LLM harness utilities."""

import json
import pytest

from chaosengineer.llm.harness import extract_json


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
