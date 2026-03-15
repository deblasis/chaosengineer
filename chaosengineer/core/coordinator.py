"""Coordinator: the central orchestration loop."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from chaosengineer.core.models import (
    Baseline,
    DimensionType,
    Experiment,
    ExperimentResult,
    ExperimentStatus,
    Run,
    WorkerState,
)
from chaosengineer.core.state import (
    assign_experiment,
    assign_worker,
    complete_experiment,
    fail_experiment,
    release_worker,
    start_experiment,
)
from chaosengineer.core.budget import BudgetTracker
from chaosengineer.metrics.logger import EventLogger, Event
from chaosengineer.core.interfaces import DecisionMaker, DimensionPlan, ExperimentExecutor, ExperimentTask
from chaosengineer.workloads.parser import WorkloadSpec


class Coordinator:
    """Runs the experiment loop: pick dimension, allocate workers, collect results."""

    def __init__(
        self,
        spec: WorkloadSpec,
        decision_maker: DecisionMaker,
        executor: ExperimentExecutor,
        logger: EventLogger,
        budget: BudgetTracker,
        initial_baseline: Baseline,
        tie_threshold_pct: float = 1.0,
        run_id: str | None = None,
    ):
        self.spec = spec
        self.decision_maker = decision_maker
        self.executor = executor
        self.logger = logger
        self.budget = budget
        self.best_baseline = initial_baseline
        self.tie_threshold_pct = tie_threshold_pct
        self.run_state = Run(
            run_id=run_id or f"run-{uuid.uuid4().hex[:8]}",
            workload_name=spec.name,
            budget=spec.budget,
            baselines=[initial_baseline],
        )
        self._iteration = 0
        self._history: list[dict] = []

    def _log(self, event: Event) -> None:
        """Log an event and append to in-memory history."""
        self.logger.log(event)
        ts = event.ts or datetime.now(timezone.utc).isoformat()
        record = {"ts": ts, "event": event.event, **event.data}
        self._history.append(record)

    def _discover_diverse_dimensions(self) -> None:
        """Discover options for DIVERSE dimensions before the main loop."""
        for dim in self.spec.dimensions:
            if dim.dim_type == DimensionType.DIVERSE and dim.options is None:
                try:
                    options = self.decision_maker.discover_diverse_options(
                        dim.name, self.spec.context,
                    )
                except Exception as e:
                    self._log(Event(
                        event="diverse_discovery_failed",
                        data={"dimension": dim.name, "error": str(e)},
                    ))
                    continue
                if not options:
                    self._log(Event(
                        event="diverse_discovery_failed",
                        data={"dimension": dim.name, "error": "empty options returned"},
                    ))
                    continue
                dim.options = options
                self._log(Event(
                    event="diverse_discovered",
                    data={"dimension": dim.name, "options": options, "count": len(options)},
                ))

    def run(self) -> None:
        """Execute the coordinator loop until budget or dimensions exhausted."""
        self._log(Event(
            event="run_started",
            data={
                "workload": self.spec.name,
                "budget": self.budget.config.to_dict(),
                "baseline": self.best_baseline.to_dict(),
            },
        ))
        self.budget.start()
        self.run_state.start_time = time.time()

        self._discover_diverse_dimensions()

        active_baselines = [self.best_baseline]

        while not self.budget.is_exhausted():
            # For each active baseline (beam search: may be >1 after ties),
            # ask the decision maker for a plan and run it.
            next_active: list[Baseline] = []
            for baseline in active_baselines:
                if self.budget.is_exhausted():
                    break

                plan = self.decision_maker.pick_next_dimension(
                    dimensions=self.spec.dimensions,
                    baselines=[baseline],
                    history=self._history,
                )
                if plan is None:
                    continue  # this branch has no more dimensions

                # Check if budget can accommodate this iteration
                if (
                    self.budget.config.max_experiments is not None
                    and self.budget.experiments_run + len(plan.values)
                    > self.budget.config.max_experiments
                ):
                    remaining = (
                        self.budget.config.max_experiments
                        - self.budget.experiments_run
                    )
                    if remaining <= 0:
                        break
                    original_count = len(plan.values)
                    plan = DimensionPlan(
                        dimension_name=plan.dimension_name,
                        values=plan.values[:remaining],
                    )
                    self._log(Event(
                        event="budget_trim",
                        data={
                            "dimension": plan.dimension_name,
                            "original_count": original_count,
                            "trimmed_count": remaining,
                        },
                    ))

                self._log(Event(
                    event="iteration_started",
                    data={
                        "dimension": plan.dimension_name,
                        "num_workers": len(plan.values),
                        "iteration": self._iteration,
                        "branch_id": baseline.branch_id,
                    },
                ))

                iteration_results = self._run_iteration(plan, baseline)

                branch_baselines = self._evaluate_iteration(
                    plan, iteration_results, [baseline]
                )
                next_active.extend(branch_baselines)

                self._log(Event(
                    event="budget_checkpoint",
                    data=self.budget.snapshot(),
                ))

                self._iteration += 1
                self.run_state.current_iteration = self._iteration
                self.run_state.dimensions_explored.append(plan.dimension_name)

            if not next_active:
                break  # all branches exhausted
            active_baselines = next_active

        self.run_state.end_time = time.time()
        self.run_state.total_experiments_run = self.budget.experiments_run
        self.run_state.total_cost_usd = self.budget.spent_usd

        self._log(Event(
            event="run_completed",
            data={
                "best_metric": self.best_baseline.metric_value,
                "total_experiments": self.budget.experiments_run,
                "total_cost_usd": self.budget.spent_usd,
            },
        ))

    def _run_iteration(
        self, plan: DimensionPlan, baseline: Baseline
    ) -> list[tuple[Experiment, ExperimentResult]]:
        """Run all experiments for one dimension sweep from a given baseline."""
        # Phase 1: Build Experiment objects and task list
        tasks: list[ExperimentTask] = []
        experiment_workers: list[tuple[Experiment, WorkerState]] = []

        for i, params in enumerate(plan.values):
            exp_id = f"exp-{self._iteration}-{i}"
            exp = Experiment(
                experiment_id=exp_id,
                dimension=plan.dimension_name,
                params=params,
                baseline_commit=baseline.commit,
                branch_id=baseline.branch_id,
            )
            self.run_state.experiments.append(exp)

            worker = WorkerState(worker_id=f"w-{self._iteration}-{i}")
            assign_experiment(exp, worker.worker_id)
            assign_worker(worker, exp.experiment_id)
            start_experiment(exp)

            tasks.append(ExperimentTask(
                exp_id, params, self.spec.execution_command, baseline.commit,
            ))
            experiment_workers.append((exp, worker))

        # Phase 2: Execute batch
        batch_results = self.executor.run_experiments(tasks)

        # Phase 3: Handle results
        results: list[tuple[Experiment, ExperimentResult]] = []
        for (exp, worker), result in zip(experiment_workers, batch_results):
            if result.error_message:
                fail_experiment(exp, result)
                self._log(Event(
                    event="worker_failed",
                    data={"experiment_id": exp.experiment_id, "error": result.error_message},
                ))
            else:
                complete_experiment(exp, result)
                self._log(Event(
                    event="worker_completed",
                    data={
                        "experiment_id": exp.experiment_id,
                        "params": exp.params,
                        "result": result.to_dict(),
                    },
                ))
            release_worker(worker)
            self.budget.record_experiment()
            self.budget.add_cost(result.cost_usd)

            results.append((exp, result))

        return results

    def _evaluate_iteration(
        self,
        plan: DimensionPlan,
        results: list[tuple[Experiment, ExperimentResult]],
        active_baselines: list[Baseline],
    ) -> list[Baseline]:
        """Evaluate iteration results. Returns updated active baselines."""
        # Filter to successful experiments (completed, no error)
        valid = [
            (exp, res)
            for exp, res in results
            if exp.status == ExperimentStatus.COMPLETED
            and res.error_message is None
        ]

        if not valid:
            self.budget.record_no_improvement()
            return active_baselines

        # Find the best result(s)
        if self.spec.metric_direction == "lower":
            valid.sort(key=lambda x: x[1].primary_metric)
        else:
            valid.sort(key=lambda x: -x[1].primary_metric)

        best_exp, best_result = valid[0]
        best_metric = best_result.primary_metric

        # Check if best beats current baseline
        if self.spec.is_better(best_metric, self.best_baseline.metric_value):
            new_baseline = Baseline(
                commit=best_result.commit_hash or best_exp.baseline_commit,
                metric_value=best_metric,
                metric_name=self.spec.primary_metric,
            )
            self.best_baseline = new_baseline
            self.run_state.baselines.append(new_baseline)
            self.budget.record_improvement()

            self._log(Event(
                event="breakthrough",
                data={
                    "new_best": best_metric,
                    "previous_best": active_baselines[0].metric_value
                    if active_baselines
                    else None,
                    "from_experiment": best_exp.experiment_id,
                },
            ))

            # Check for ties (beam search)
            tied = self._find_ties(valid, best_metric)
            if len(tied) > 1:
                # Multiple tied winners become parallel baselines
                return [
                    Baseline(
                        commit=r.commit_hash or e.baseline_commit,
                        metric_value=r.primary_metric,
                        metric_name=self.spec.primary_metric,
                        branch_id=e.experiment_id,
                    )
                    for e, r in tied
                ]

            return [new_baseline]
        else:
            self.budget.record_no_improvement()
            return active_baselines

    _TIE_EPSILON = 0.01  # minimum absolute tie threshold

    def _find_ties(
        self,
        sorted_results: list[tuple[Experiment, ExperimentResult]],
        best_metric: float,
    ) -> list[tuple[Experiment, ExperimentResult]]:
        """Find results within tie threshold of the best."""
        if self.tie_threshold_pct <= 0 or not sorted_results:
            return [sorted_results[0]]

        threshold = max(
            abs(best_metric) * (self.tie_threshold_pct / 100.0),
            self._TIE_EPSILON,
        )
        tied = []
        for exp, result in sorted_results:
            if abs(result.primary_metric - best_metric) <= threshold:
                tied.append((exp, result))
        return tied if tied else [sorted_results[0]]
