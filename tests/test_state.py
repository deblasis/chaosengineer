"""Tests for state machine transitions."""

import pytest
from chaosengineer.core.models import (
    Experiment, ExperimentStatus, WorkerState, WorkerStatus,
)
from chaosengineer.core.state import (
    InvalidTransitionError,
    assign_experiment,
    start_experiment,
    complete_experiment,
    fail_experiment,
    kill_experiment,
    assign_worker,
    release_worker,
    terminate_worker,
)


class TestExperimentTransitions:
    def _make_experiment(self, status=ExperimentStatus.PLANNED):
        exp = Experiment(
            experiment_id="exp-001",
            dimension="lr",
            params={"lr": 0.08},
            baseline_commit="abc1234",
        )
        exp.status = status
        return exp

    def test_assign_from_planned(self):
        exp = self._make_experiment()
        assign_experiment(exp, worker_id="w1")
        assert exp.status == ExperimentStatus.ASSIGNED
        assert exp.worker_id == "w1"

    def test_assign_from_running_raises(self):
        exp = self._make_experiment(ExperimentStatus.RUNNING)
        with pytest.raises(InvalidTransitionError):
            assign_experiment(exp, worker_id="w1")

    def test_start_from_assigned(self):
        exp = self._make_experiment(ExperimentStatus.ASSIGNED)
        start_experiment(exp)
        assert exp.status == ExperimentStatus.RUNNING

    def test_start_from_planned_raises(self):
        exp = self._make_experiment(ExperimentStatus.PLANNED)
        with pytest.raises(InvalidTransitionError):
            start_experiment(exp)

    def test_complete_from_running(self):
        from chaosengineer.core.models import ExperimentResult
        exp = self._make_experiment(ExperimentStatus.RUNNING)
        result = ExperimentResult(primary_metric=0.93)
        complete_experiment(exp, result)
        assert exp.status == ExperimentStatus.COMPLETED
        assert exp.result.primary_metric == 0.93

    def test_fail_from_running(self):
        from chaosengineer.core.models import ExperimentResult
        exp = self._make_experiment(ExperimentStatus.RUNNING)
        result = ExperimentResult(primary_metric=0.0, error_message="OOM")
        fail_experiment(exp, result)
        assert exp.status == ExperimentStatus.FAILED

    def test_kill_from_running(self):
        exp = self._make_experiment(ExperimentStatus.RUNNING)
        kill_experiment(exp)
        assert exp.status == ExperimentStatus.KILLED

    def test_kill_from_assigned(self):
        exp = self._make_experiment(ExperimentStatus.ASSIGNED)
        kill_experiment(exp)
        assert exp.status == ExperimentStatus.KILLED

    def test_complete_from_planned_raises(self):
        from chaosengineer.core.models import ExperimentResult
        exp = self._make_experiment(ExperimentStatus.PLANNED)
        with pytest.raises(InvalidTransitionError):
            complete_experiment(exp, ExperimentResult(primary_metric=0.0))


class TestWorkerTransitions:
    def _make_worker(self, status=WorkerStatus.IDLE):
        w = WorkerState(worker_id="w1", resource="GPU:0")
        w.status = status
        return w

    def test_assign_from_idle(self):
        w = self._make_worker()
        assign_worker(w, experiment_id="exp-001")
        assert w.status == WorkerStatus.BUSY
        assert w.current_experiment_id == "exp-001"

    def test_assign_from_busy_raises(self):
        w = self._make_worker(WorkerStatus.BUSY)
        with pytest.raises(InvalidTransitionError):
            assign_worker(w, experiment_id="exp-002")

    def test_release_from_busy(self):
        w = self._make_worker(WorkerStatus.BUSY)
        w.current_experiment_id = "exp-001"
        release_worker(w)
        assert w.status == WorkerStatus.IDLE
        assert w.current_experiment_id is None

    def test_terminate_from_busy(self):
        w = self._make_worker(WorkerStatus.BUSY)
        terminate_worker(w)
        assert w.status == WorkerStatus.TERMINATED

    def test_terminate_from_idle(self):
        w = self._make_worker(WorkerStatus.IDLE)
        terminate_worker(w)
        assert w.status == WorkerStatus.TERMINATED

    def test_assign_terminated_raises(self):
        w = self._make_worker(WorkerStatus.TERMINATED)
        with pytest.raises(InvalidTransitionError):
            assign_worker(w, experiment_id="exp-001")
