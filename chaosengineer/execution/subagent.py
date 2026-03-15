"""SubagentExecutor — spawns Claude Code subagents for real experiments."""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from chaosengineer.core.interfaces import ExperimentExecutor, ExperimentTask
from chaosengineer.core.models import ExperimentResult
from chaosengineer.execution.cli_usage import parse_cli_usage
from chaosengineer.execution.result_parser import ResultParser
from chaosengineer.execution.task_packet import TaskPacketBuilder
from chaosengineer.execution.worktree import WorktreeManager
from chaosengineer.workloads.parser import WorkloadSpec

_GRACE_SECONDS = 60


def _get_repo_root() -> Path:
    """Find the git repo root from CWD."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return Path.cwd()
    return Path(result.stdout.strip())


class SubagentExecutor(ExperimentExecutor):
    """Runs experiments by spawning Claude Code subagents in git worktrees."""

    def __init__(
        self,
        spec: WorkloadSpec,
        output_dir: Path,
        mode: str = "sequential",
        run_id: str = "run-unknown",
        repo_root: Path | None = None,
    ):
        self._spec = spec
        self._output_dir = output_dir
        self._mode = mode
        self._run_id = run_id
        self._worktree_mgr = WorktreeManager(
            repo_root=repo_root if repo_root is not None else _get_repo_root()
        )
        self._task_builder = TaskPacketBuilder()
        self._result_parser = ResultParser()
        self._active_processes: list[subprocess.Popen] = []
        self._process_lock = threading.Lock()

    def run_experiment(
        self,
        experiment_id: str,
        params: dict[str, Any],
        command: str,
        baseline_commit: str,
        resource: str = "",
    ) -> ExperimentResult:
        """Run a single experiment via Claude Code subagent."""
        task = ExperimentTask(experiment_id, params, command, baseline_commit, resource)
        return self._run_single(task)

    def run_experiments(
        self,
        tasks: list[ExperimentTask],
        on_worker_done: "Callable[[ExperimentTask, ExperimentResult, int, int], None] | None" = None,
    ) -> list[ExperimentResult]:
        """Run a batch of experiments with as_completed for reactive callbacks."""
        max_workers = len(tasks) if self._mode == "parallel" else 1

        result_map: dict[str, ExperimentResult] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_task = {
                pool.submit(self._run_single, task): task for task in tasks
            }
            completed_count = 0
            total = len(tasks)
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = ExperimentResult(
                        primary_metric=0.0,
                        error_message=f"Thread crashed for {task.experiment_id}: {e}",
                    )
                result_map[task.experiment_id] = result
                completed_count += 1
                if on_worker_done is not None:
                    on_worker_done(task, result, completed_count, total)

        # Reorder to match input task order
        return [result_map[t.experiment_id] for t in tasks]

    def _run_single(self, task: ExperimentTask) -> ExperimentResult:
        """Execute the full pipeline for one experiment."""
        result_file = self._output_dir / task.experiment_id / "result.json"

        # 1. Create worktree
        worktree_path = self._worktree_mgr.create(
            task.baseline_commit, self._run_id, task.experiment_id
        )

        try:
            # 2. Build task file
            task_file = self._task_builder.build(
                task, self._spec, worktree_path, result_file,
                run_id=self._run_id, output_dir=self._output_dir,
            )

            # 3. Invoke claude subprocess
            start = time.monotonic()
            invoke_result = self._invoke(
                task_file, worktree_path, task.resource
            )
            duration = time.monotonic() - start

            if isinstance(invoke_result, ExperimentResult):
                return invoke_result

            # 4. Check for non-zero exit code
            if invoke_result.returncode != 0:
                error_result = ExperimentResult(
                    primary_metric=0.0,
                    duration_seconds=duration,
                    error_message=(
                        f"Claude process failed for {task.experiment_id} "
                        f"(exit {invoke_result.returncode}): "
                        f"{(invoke_result.stderr or '')[:500]}"
                    ),
                )
                usage = parse_cli_usage(invoke_result.stdout)
                error_result.tokens_in = usage.tokens_in
                error_result.tokens_out = usage.tokens_out
                error_result.cost_usd = usage.cost_usd
                return error_result

            # 5. Parse result
            result = self._result_parser.parse(
                result_file, task.experiment_id, duration
            )
            usage = parse_cli_usage(invoke_result.stdout)
            result.tokens_in = usage.tokens_in
            result.tokens_out = usage.tokens_out
            result.cost_usd = usage.cost_usd
            return result
        finally:
            # 6. Cleanup worktree
            self._worktree_mgr.cleanup(worktree_path)

    def _invoke(
        self,
        task_file: Path,
        worktree_path: Path,
        resource: str,
    ) -> subprocess.CompletedProcess | ExperimentResult:
        """Spawn claude -p subprocess using Popen for kill support."""
        env = {**os.environ}
        if resource:
            gpu_id = _parse_gpu_id(resource)
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id

        timeout = None
        if self._spec.time_budget_seconds is not None:
            timeout = self._spec.time_budget_seconds + _GRACE_SECONDS

        prompt = task_file.read_text()
        cmd = [
            "claude", "-p", prompt,
            "--allowedTools", "Edit,Write,Bash,Read",
            "--output-format", "stream-json",
            "--verbose",
        ]

        proc = subprocess.Popen(
            cmd,
            cwd=str(worktree_path),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        with self._process_lock:
            self._active_processes.append(proc)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=proc.returncode,
                stdout=stdout,
                stderr=stderr,
            )
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            return ExperimentResult(
                primary_metric=0.0,
                error_message=(
                    f"Experiment timed out after {timeout}s "
                    f"(budget: {self._spec.time_budget_seconds}s + {_GRACE_SECONDS}s grace)"
                ),
            )
        finally:
            with self._process_lock:
                if proc in self._active_processes:
                    self._active_processes.remove(proc)

    def kill_active(self) -> None:
        """Terminate all active subprocesses."""
        with self._process_lock:
            processes = list(self._active_processes)
        for proc in processes:
            try:
                proc.terminate()
            except OSError:
                pass  # Already dead


def _parse_gpu_id(resource: str) -> str | None:
    """Extract GPU device ID from resource string like 'gpu:2'."""
    match = re.match(r"gpu:(\d+)", resource)
    return match.group(1) if match else None
