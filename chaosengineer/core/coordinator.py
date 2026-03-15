"""Coordinator: the central orchestration loop."""

from __future__ import annotations

import uuid

from chaosengineer.core.models import (
    Baseline,
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
from chaosengineer.core.interfaces import DecisionMaker, DimensionPlan, ExperimentExecutor
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

    def run(self) -> None:
        """Execute the coordinator loop until budget or dimensions exhausted."""
        self.logger.log(Event(
            event="run_started",
            data={
                "workload": self.spec.name,
                "budget": self.budget.config.to_dict(),
                "baseline": self.best_baseline.to_dict(),
            },
        ))
        self.budget.start()

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
                    history=self.logger.read_events(),
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
                    plan = DimensionPlan(
                        dimension_name=plan.dimension_name,
                        values=plan.values[:remaining],
                    )

                self.logger.log(Event(
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

                self.logger.log(Event(
                    event="budget_checkpoint",
                    data=self.budget.snapshot(),
                ))

                self._iteration += 1

            if not next_active:
                break  # all branches exhausted
            active_baselines = next_active

        self.logger.log(Event(
            event="run_completed",
            data={
                "best_metric": self.best_baseline.metric_value,
                "total_experiments": self.budget.experiments_run,
                "total_cost_usd": self.budget.spent_usd,
            },
        ))

    def _run_iteration(
        self, plan: DimensionPlan, baseline: Baseline
    ) -> list[tuple[Experiment, ExperimentResult | None]]:
        """Run all experiments for one dimension sweep from a given baseline."""
        results: list[tuple[Experiment, ExperimentResult | None]] = []
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

            # Create a temporary worker
            worker = WorkerState(worker_id=f"w-{self._iteration}-{i}")

            # State transitions
            assign_experiment(exp, worker.worker_id)
            assign_worker(worker, exp.experiment_id)
            start_experiment(exp)

            # Execute
            result: ExperimentResult | None = None
            try:
                result = self.executor.run_experiment(
                    experiment_id=exp_id,
                    params=params,
                    command=self.spec.execution_command,
                    baseline_commit=baseline.commit,
                )
                if result.error_message:
                    fail_experiment(exp, result)
                    self.logger.log(Event(
                        event="worker_failed",
                        data={"experiment_id": exp_id, "error": result.error_message},
                    ))
                else:
                    complete_experiment(exp, result)
                    self.logger.log(Event(
                        event="worker_completed",
                        data={
                            "experiment_id": exp_id,
                            "params": params,
                            "result": result.to_dict(),
                        },
                    ))
            except Exception as e:
                error_result = ExperimentResult(
                    primary_metric=0.0, error_message=str(e)
                )
                fail_experiment(exp, error_result)
                self.logger.log(Event(
                    event="worker_failed",
                    data={"experiment_id": exp_id, "error": str(e)},
                ))
                result = None

            release_worker(worker)
            self.budget.record_experiment()
            self.budget.add_cost(result.cost_usd if result else 0.0)

            results.append((exp, result))

        return results

    def _evaluate_iteration(
        self,
        plan: DimensionPlan,
        results: list[tuple[Experiment, ExperimentResult | None]],
        active_baselines: list[Baseline],
    ) -> list[Baseline]:
        """Evaluate iteration results. Returns updated active baselines."""
        # Filter to successful experiments (completed, no error)
        valid = [
            (exp, res)
            for exp, res in results
            if exp.status == ExperimentStatus.COMPLETED
            and res is not None
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

            self.logger.log(Event(
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

    def _find_ties(
        self,
        sorted_results: list[tuple[Experiment, ExperimentResult]],
        best_metric: float,
    ) -> list[tuple[Experiment, ExperimentResult]]:
        """Find results within tie threshold of the best."""
        if self.tie_threshold_pct <= 0 or not sorted_results:
            return [sorted_results[0]]

        threshold = abs(best_metric) * (self.tie_threshold_pct / 100.0)
        tied = []
        for exp, result in sorted_results:
            if abs(result.primary_metric - best_metric) <= threshold:
                tied.append((exp, result))
        return tied if tied else [sorted_results[0]]
