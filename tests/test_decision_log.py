import json
from chaosengineer.core.decision_log import DecisionLogger


def _read_jsonl(path):
    entries = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


class TestDecisionLogger:
    def test_log_dimension_selected(self, tmp_path):
        logger = DecisionLogger(tmp_path)
        logger.log_dimension_selected(dimension="learning_rate",
            reasoning="High potential", alternatives=["batch_size", "dropout"])
        entries = _read_jsonl(tmp_path / "decisions.jsonl")
        assert len(entries) == 1
        assert entries[0]["type"] == "dimension_selected"
        assert entries[0]["dimension"] == "learning_rate"
        assert "timestamp" in entries[0]

    def test_log_results_evaluated(self, tmp_path):
        logger = DecisionLogger(tmp_path)
        logger.log_results_evaluated(dimension="lr", reasoning="0.001 gave best loss",
            winner="0.001", metrics={"exp-0-0": 2.5, "exp-0-1": 2.8})
        entries = _read_jsonl(tmp_path / "decisions.jsonl")
        assert entries[0]["type"] == "results_evaluated"
        assert entries[0]["winner"] == "0.001"

    def test_log_diverse_options(self, tmp_path):
        logger = DecisionLogger(tmp_path)
        logger.log_diverse_options(dimension="augmentation",
            reasoning="Exploring strategies", options=["cutmix", "mixup"])
        entries = _read_jsonl(tmp_path / "decisions.jsonl")
        assert entries[0]["type"] == "diverse_options_generated"
        assert entries[0]["options"] == ["cutmix", "mixup"]

    def test_multiple_entries_appended(self, tmp_path):
        logger = DecisionLogger(tmp_path)
        logger.log_dimension_selected("lr", "reason1", [])
        logger.log_dimension_selected("bs", "reason2", [])
        entries = _read_jsonl(tmp_path / "decisions.jsonl")
        assert len(entries) == 2
