"""Tests for core data models."""

import pytest
from chaosengineer.core.models import (
    ExperimentStatus,
    WorkerStatus,
    DimensionType,
    DimensionSpec,
    Experiment,
    WorkerState,
    Baseline,
    BudgetConfig,
    Run,
)


class TestDimensionSpec:
    def test_directional(self):
        d = DimensionSpec(
            name="learning_rate",
            dim_type=DimensionType.DIRECTIONAL,
            current_value=0.04,
        )
        assert d.name == "learning_rate"
        assert d.dim_type == DimensionType.DIRECTIONAL
        assert d.current_value == 0.04
        assert d.options is None

    def test_enum(self):
        d = DimensionSpec(
            name="activation",
            dim_type=DimensionType.ENUM,
            options=["GeLU", "SiLU", "ReLU"],
        )
        assert d.dim_type == DimensionType.ENUM
        assert len(d.options) == 3

    def test_diverse(self):
        d = DimensionSpec(
            name="prompt_strategy",
            dim_type=DimensionType.DIVERSE,
        )
        assert d.dim_type == DimensionType.DIVERSE
        assert d.options is None  # discovered at runtime


class TestExperiment:
    def test_creation(self):
        exp = Experiment(
            experiment_id="exp-001",
            dimension="learning_rate",
            params={"learning_rate": 0.08},
            baseline_commit="abc1234",
        )
        assert exp.status == ExperimentStatus.PLANNED
        assert exp.result is None
        assert exp.worker_id is None

    def test_to_dict_roundtrip(self):
        exp = Experiment(
            experiment_id="exp-001",
            dimension="learning_rate",
            params={"learning_rate": 0.08},
            baseline_commit="abc1234",
        )
        d = exp.to_dict()
        assert d["experiment_id"] == "exp-001"
        assert d["status"] == "planned"
        assert d["params"] == {"learning_rate": 0.08}


class TestWorkerState:
    def test_creation(self):
        w = WorkerState(worker_id="w1", resource="CUDA_VISIBLE_DEVICES=0")
        assert w.status == WorkerStatus.IDLE
        assert w.current_experiment_id is None

    def test_to_dict(self):
        w = WorkerState(worker_id="w1", resource="CUDA_VISIBLE_DEVICES=0")
        d = w.to_dict()
        assert d["worker_id"] == "w1"
        assert d["status"] == "idle"


class TestBaseline:
    def test_creation(self):
        b = Baseline(commit="abc1234", metric_value=0.95, metric_name="val_bpb")
        assert b.commit == "abc1234"
        assert b.metric_value == 0.95


class TestBudgetConfig:
    def test_defaults(self):
        b = BudgetConfig()
        assert b.max_api_cost is None
        assert b.max_experiments is None
        assert b.max_wall_time_seconds is None
        assert b.max_plateau_iterations is None

    def test_with_limits(self):
        b = BudgetConfig(max_api_cost=50.0, max_experiments=100, max_wall_time_seconds=28800)
        assert b.max_api_cost == 50.0
        assert b.max_experiments == 100
        assert b.max_wall_time_seconds == 28800


class TestRun:
    def test_creation(self):
        budget = BudgetConfig(max_experiments=10)
        run = Run(
            run_id="run-001",
            workload_name="nn-arch-search",
            budget=budget,
        )
        assert run.experiments == []
        assert run.workers == []
        assert len(run.baselines) == 0
