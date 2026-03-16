"""Coordinator: the central orchestration loop."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from chaosengineer.core.models import (
    Baseline,
    BudgetConfig,
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
        pause_controller: "PauseController | None" = None,
        status_display: "StatusDisplay | None" = None,
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
        self._pause_controller = pause_controller
        self._status_display = status_display

    def _log(self, event: Event) -> None:
        """Log an event and append to in-memory history."""
        self.logger.log(event)
        ts = event.ts or datetime.now(timezone.utc).isoformat()
        record = {"ts": ts, "event": event.event, **event.data}
        self._history.append(record)

    def _poll_bus_commands(self) -> None:
        """Poll message bus for remote commands (pause, extend_budget)."""
        if not hasattr(self.logger, "poll_commands"):
            return
        for cmd in self.logger.poll_commands():
            if cmd.get("command") == "pause" and self._pause_controller:
                self._pause_controller.pause_requested = True
            elif cmd.get("command") == "extend_budget":
                bc = self.budget.config
                if cmd.get("add_cost_usd"):
                    bc = BudgetConfig(
                        max_api_cost=(bc.max_api_cost or 0) + cmd["add_cost_usd"],
                        max_experiments=bc.max_experiments,
                        max_wall_time_seconds=bc.max_wall_time_seconds,
                        max_plateau_iterations=bc.max_plateau_iterations,
                    )
                if cmd.get("add_experiments"):
                    bc = BudgetConfig(
                        max_api_cost=bc.max_api_cost,
                        max_experiments=(bc.max_experiments or 0) + cmd["add_experiments"],
                        max_wall_time_seconds=bc.max_wall_time_seconds,
                        max_plateau_iterations=bc.max_plateau_iterations,
                    )
                if cmd.get("add_time_seconds"):
                    bc = BudgetConfig(
                        max_api_cost=bc.max_api_cost,
                        max_experiments=bc.max_experiments,
                        max_wall_time_seconds=(bc.max_wall_time_seconds or 0) + cmd["add_time_seconds"],
                        max_plateau_iterations=bc.max_plateau_iterations,
                    )
                self.budget.config = bc

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
            # Pause check during diverse discovery
            if self._pause_controller and self._pause_controller.pause_requested:
                return  # Will be caught at top of _run_loop

    def run(self) -> None:
        """Execute the coordinator loop until budget or dimensions exhausted."""
        self._log(Event(
            event="run_started",
            data={
                "workload": self.spec.name,
                "budget": self.budget.config.to_dict(),
                "baseline": self.best_baseline.to_dict(),
                "run_id": self.run_state.run_id,
                "mode": self.run_state.mode,
                "metric_direction": self.spec.metric_direction,
                "workload_spec_hash": self.spec.spec_hash(),
            },
        ))
        self.budget.start()
        self.run_state.start_time = time.time()

        if self._status_display:
            self._status_display.on_run_start(self.spec.budget)

        self._discover_diverse_dimensions()

        active_baselines = [self.best_baseline]
        self._run_loop(active_baselines)

    def _run_loop(self, active_baselines: list[Baseline]) -> None:
        """Main coordinator loop. Shared by run() and resume_from_snapshot()."""
        all_dimensions_exhausted = True

        while not self.budget.is_exhausted():
            self._poll_bus_commands()
            # Pause check: before starting new iteration
            if self._pause_controller and self._pause_controller.pause_requested and self._pause_controller.should_show_menu():
                choice = self._pause_controller.show_post_iteration_menu()
                if choice == "pause":
                    self._log_user_pause(active_baselines)
                    return
                else:
                    self._pause_controller.reset()

            next_active: list[Baseline] = []
            for baseline in active_baselines:
                if self.budget.is_exhausted():
                    all_dimensions_exhausted = False
                    break

                plan = self.decision_maker.pick_next_dimension(
                    dimensions=self.spec.dimensions,
                    baselines=[baseline],
                    history=self._history,
                )
                if plan is None:
                    continue

                # Budget trim (existing logic)
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
                        all_dimensions_exhausted = False
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

                iteration_tasks = [
                    {
                        "experiment_id": f"exp-{self._iteration}-{i}",
                        "params": params,
                        "command": self.spec.execution_command,
                        "baseline_commit": baseline.commit,
                    }
                    for i, params in enumerate(plan.values)
                ]
                self._log(Event(
                    event="iteration_started",
                    data={
                        "dimension": plan.dimension_name,
                        "num_workers": len(plan.values),
                        "iteration": self._iteration,
                        "branch_id": baseline.branch_id,
                        "tasks": iteration_tasks,
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

                # Status display: iteration done
                if self._status_display:
                    self._status_display.on_iteration_done(
                        self._iteration - 1, self.best_baseline.metric_value,
                    )

                self._poll_bus_commands()
                # Pause check: auto-pause after kill
                if self._pause_controller and self._pause_controller.kill_issued:
                    self._log_user_pause(active_baselines)
                    return
                # Pause check: after iteration (wait_then_ask or pause_requested)
                if self._pause_controller and self._pause_controller.should_show_menu():
                    choice = self._pause_controller.show_post_iteration_menu(
                        f"Iteration {self._iteration - 1} complete. "
                        f"{self.spec.primary_metric}={self.best_baseline.metric_value}"
                    )
                    if choice == "pause":
                        self._log_user_pause(active_baselines)
                        return
                    else:
                        self._pause_controller.reset()

            if not next_active:
                break
            active_baselines = next_active

        # Exit branching: paused vs completed
        self.run_state.end_time = time.time()
        self.run_state.total_experiments_run = self.budget.experiments_run
        self.run_state.total_cost_usd = self.budget.spent_usd

        if all_dimensions_exhausted and not self.budget.is_exhausted():
            self._log(Event(
                event="run_completed",
                data={
                    "best_metric": self.best_baseline.metric_value,
                    "total_experiments": self.budget.experiments_run,
                    "total_cost_usd": self.budget.spent_usd,
                },
            ))
        else:
            reason = self.budget.exhaustion_reason or "unknown"
            self._log(Event(
                event="run_paused",
                data={
                    "reason": reason,
                    "last_iteration": self._iteration,
                    "budget_state": self.budget.snapshot(),
                    "active_baselines": [b.to_dict() for b in active_baselines],
                },
            ))

    def resume_from_snapshot(self, snapshot: "RunSnapshot", restart_iteration: bool = False,
                             budget_extensions: dict | None = None) -> None:
        """Resume a run from a reconstructed snapshot."""
        from chaosengineer.core.snapshot import RunSnapshot, IncompleteIteration

        # Restore budget tracker with prior state
        self.budget = BudgetTracker.from_snapshot(
            config=snapshot.budget_config,
            experiments_run=snapshot.total_experiments_run,
            cost_spent=snapshot.total_cost_usd,
            elapsed_offset=snapshot.elapsed_time,
            consecutive_no_improvement=snapshot.consecutive_no_improvement,
        )
        self.budget.start()

        # Restore baselines and iteration counter
        active_baselines = list(snapshot.active_baselines)
        self.best_baseline = active_baselines[0] if active_baselines else self.best_baseline
        self._iteration = len(snapshot.dimensions_explored)
        self._history = list(snapshot.history)

        # Restore run_id
        self.run_state.run_id = snapshot.run_id

        # Restore DIVERSE discovered options
        for dim in self.spec.dimensions:
            if dim.name in snapshot.discovered_dimensions:
                dim.options = snapshot.discovered_dimensions[dim.name]

        # Set prior context on decision maker
        context_lines = ["Previous run state (resuming):"]
        for d in snapshot.dimensions_explored:
            context_lines.append(f"- Explored {d.name}: winner={d.winner}")
        bl_str = ", ".join(f"{b.metric_value}" for b in active_baselines)
        context_lines.append(f"- Active baselines: {bl_str}")
        context_lines.append(f"- Experiments run: {snapshot.total_experiments_run}")
        context_lines.append(f"- Budget spent: ${snapshot.total_cost_usd:.2f}")
        self.decision_maker.set_prior_context("\n".join(context_lines))

        # Log run_resumed event
        self._log(Event(
            event="run_resumed",
            data={
                "original_run_id": snapshot.run_id,
                "budget_extensions": budget_extensions or {},
                "restart_iteration": restart_iteration,
                "snapshot_summary": {
                    "dimensions_explored": len(snapshot.dimensions_explored),
                    "experiments_completed": snapshot.total_experiments_run,
                },
            },
        ))

        # Handle incomplete iteration
        if snapshot.incomplete_iteration and not restart_iteration:
            self._complete_partial_iteration(snapshot.incomplete_iteration, active_baselines)
            self._iteration += 1
        elif snapshot.incomplete_iteration and restart_iteration:
            pass  # Dimension returned to pool, LLM will re-pick

        # Enter normal run loop
        self._run_loop(active_baselines=active_baselines)

    def _complete_partial_iteration(
        self, incomplete: "IncompleteIteration", active_baselines: list[Baseline]
    ) -> None:
        """Run missing workers from an interrupted iteration and evaluate."""
        from chaosengineer.core.snapshot import IncompleteIteration

        new_results = self.executor.run_experiments(incomplete.missing_tasks)

        baseline_commit = incomplete.missing_tasks[0].baseline_commit if incomplete.missing_tasks else ""

        all_pairs: list[tuple[Experiment, ExperimentResult]] = []
        for exp_summary in incomplete.completed_experiments:
            exp = Experiment(
                experiment_id=exp_summary.experiment_id,
                dimension=exp_summary.dimension,
                params=exp_summary.params,
                baseline_commit=baseline_commit,
            )
            result = ExperimentResult(
                primary_metric=exp_summary.metric if exp_summary.metric is not None else 0.0,
                cost_usd=exp_summary.cost_usd,
            )
            all_pairs.append((exp, result))

        for task, result in zip(incomplete.missing_tasks, new_results):
            exp = Experiment(
                experiment_id=task.experiment_id,
                dimension=incomplete.dimension,
                params=task.params,
                baseline_commit=task.baseline_commit,
            )
            all_pairs.append((exp, result))
            self.budget.record_experiment()
            self.budget.add_cost(result.cost_usd)
            self._history.append({
                "experiment_id": task.experiment_id,
                "dimension": incomplete.dimension,
                "params": task.params,
                "metric": result.primary_metric,
                "status": "completed" if result.error_message is None else "failed",
            })

        self._log(Event(
            event="iteration_gap_completed",
            data={
                "dimension": incomplete.dimension,
                "original_completed": len(incomplete.completed_experiments),
                "gap_filled": len(new_results),
            },
        ))

        plan = DimensionPlan(
            dimension_name=incomplete.dimension,
            values=[task.params for task in incomplete.missing_tasks]
                   + [e.params for e in incomplete.completed_experiments],
        )
        active_baselines[:] = self._evaluate_iteration(plan, all_pairs, active_baselines)

    def _log_user_pause(self, active_baselines: list[Baseline]) -> None:
        """Log run_paused with reason user_requested."""
        self.run_state.end_time = time.time()
        self.run_state.total_experiments_run = self.budget.experiments_run
        self.run_state.total_cost_usd = self.budget.spent_usd
        self._log(Event(
            event="run_paused",
            data={
                "reason": "user_requested",
                "last_iteration": self._iteration,
                "budget_state": self.budget.snapshot(),
                "active_baselines": [b.to_dict() for b in active_baselines],
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

        # Build callback for status display and pause
        callback = None
        if self._status_display or self._pause_controller:
            def _on_worker_done(task, result, completed, total):
                if self._status_display:
                    self._status_display.on_worker_done(task, result, completed, total)
                if (self._pause_controller
                        and self._pause_controller.pause_requested
                        and not self._pause_controller.kill_issued
                        and completed < total):
                    choice = self._pause_controller.show_mid_iteration_menu(completed, total)
                    if choice == "kill":
                        self._pause_controller.kill_issued = True
                        self.executor.kill_active()
                    elif choice == "wait":
                        self._pause_controller.wait_then_ask = True
                        self._pause_controller.pause_requested = False
                    elif choice == "continue":
                        self._pause_controller.reset()
            callback = _on_worker_done

        # Phase 2: Execute batch
        batch_results = self.executor.run_experiments(tasks, on_worker_done=callback)

        # Phase 3: Handle results
        results: list[tuple[Experiment, ExperimentResult]] = []
        for (exp, worker), result in zip(experiment_workers, batch_results):
            if result.error_message:
                fail_experiment(exp, result)
                self._log(Event(
                    event="worker_failed",
                    data={
                        "experiment_id": exp.experiment_id,
                        "error": result.error_message,
                        "dimension": exp.dimension,
                        "params": exp.params,
                        "cost_usd": result.cost_usd,
                    },
                ))
            else:
                complete_experiment(exp, result)
                self._log(Event(
                    event="worker_completed",
                    data={
                        "experiment_id": exp.experiment_id,
                        "params": exp.params,
                        "result": result.to_dict(),
                        "dimension": exp.dimension,
                        "metric": result.primary_metric,
                        "cost_usd": result.cost_usd,
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
                    "commit": best_result.commit_hash or best_exp.baseline_commit,
                    "metric": best_metric,
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
