"""State machine transitions for experiments and workers."""

from __future__ import annotations

from chaosengineer.core.models import (
    Experiment,
    ExperimentResult,
    ExperimentStatus,
    WorkerState,
    WorkerStatus,
)


class InvalidTransitionError(Exception):
    """Raised when a state transition is not allowed."""

    def __init__(self, entity_id: str, current: str, target: str):
        super().__init__(
            f"Invalid transition for {entity_id}: {current} -> {target}"
        )


# Valid transitions for experiments
_EXPERIMENT_TRANSITIONS: dict[ExperimentStatus, set[ExperimentStatus]] = {
    ExperimentStatus.PLANNED: {ExperimentStatus.ASSIGNED, ExperimentStatus.KILLED},
    ExperimentStatus.ASSIGNED: {ExperimentStatus.RUNNING, ExperimentStatus.KILLED},
    ExperimentStatus.RUNNING: {
        ExperimentStatus.COMPLETED,
        ExperimentStatus.FAILED,
        ExperimentStatus.KILLED,
    },
    ExperimentStatus.COMPLETED: set(),
    ExperimentStatus.FAILED: set(),
    ExperimentStatus.KILLED: set(),
}

# Valid transitions for workers
_WORKER_TRANSITIONS: dict[WorkerStatus, set[WorkerStatus]] = {
    WorkerStatus.IDLE: {WorkerStatus.BUSY, WorkerStatus.TERMINATED},
    WorkerStatus.BUSY: {WorkerStatus.IDLE, WorkerStatus.TERMINATED},
    WorkerStatus.TERMINATED: set(),
}


def _check_experiment_transition(
    exp: Experiment, target: ExperimentStatus
) -> None:
    allowed = _EXPERIMENT_TRANSITIONS.get(exp.status, set())
    if target not in allowed:
        raise InvalidTransitionError(exp.experiment_id, exp.status.value, target.value)


def _check_worker_transition(
    worker: WorkerState, target: WorkerStatus
) -> None:
    allowed = _WORKER_TRANSITIONS.get(worker.status, set())
    if target not in allowed:
        raise InvalidTransitionError(worker.worker_id, worker.status.value, target.value)


# --- Experiment transitions ---


def assign_experiment(exp: Experiment, worker_id: str) -> None:
    _check_experiment_transition(exp, ExperimentStatus.ASSIGNED)
    exp.status = ExperimentStatus.ASSIGNED
    exp.worker_id = worker_id


def start_experiment(exp: Experiment) -> None:
    _check_experiment_transition(exp, ExperimentStatus.RUNNING)
    exp.status = ExperimentStatus.RUNNING


def complete_experiment(exp: Experiment, result: ExperimentResult) -> None:
    _check_experiment_transition(exp, ExperimentStatus.COMPLETED)
    exp.status = ExperimentStatus.COMPLETED
    exp.result = result


def fail_experiment(exp: Experiment, result: ExperimentResult) -> None:
    _check_experiment_transition(exp, ExperimentStatus.FAILED)
    exp.status = ExperimentStatus.FAILED
    exp.result = result


def kill_experiment(exp: Experiment) -> None:
    _check_experiment_transition(exp, ExperimentStatus.KILLED)
    exp.status = ExperimentStatus.KILLED


# --- Worker transitions ---


def assign_worker(worker: WorkerState, experiment_id: str) -> None:
    _check_worker_transition(worker, WorkerStatus.BUSY)
    worker.status = WorkerStatus.BUSY
    worker.current_experiment_id = experiment_id


def release_worker(worker: WorkerState) -> None:
    _check_worker_transition(worker, WorkerStatus.IDLE)
    worker.status = WorkerStatus.IDLE
    worker.current_experiment_id = None


def terminate_worker(worker: WorkerState) -> None:
    _check_worker_transition(worker, WorkerStatus.TERMINATED)
    worker.status = WorkerStatus.TERMINATED
