# SubagentExecutor Design — Phase 2C

## Overview

SubagentExecutor implements `ExperimentExecutor` to run real experiments via Claude Code subagents. Each experiment gets an isolated git worktree where a subagent applies parameter changes, runs the experiment command, and reports results. Supports sequential and parallel execution modes.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Subagent behavior | Hybrid | Gets params + modifiable_files + constraints; uses LLM reasoning to apply params, then executes command |
| Git workflow | Worktree + commit + branch | Named branch per experiment (`chaosengineer/<run_id>/<experiment_id>`), coordinator merges breakthroughs |
| Parallelism | Batch method on executor | `run_experiments(tasks) -> results`; coordinator stays synchronous, executor owns concurrency |
| Invocation | `claude -p` with task file | Write task to markdown file, avoids shell escaping, serves as audit trail |
| Result collection | Output file | Subagent writes `result.json` to experiment output directory |
| Timeouts | Spec-driven optional | `time_budget_seconds` + 60s grace if set; no timeout if unset |
| Human evaluation | Deferred | Automatic evaluation only for 2C; human-in-the-loop is a future phase |
| CLI | `--executor` + `--mode` flags | `--executor subagent\|scripted`, `--mode sequential\|parallel`; scripted supports YAML file or folder |

## ABC Changes

### ExperimentTask dataclass

New dataclass in `core/interfaces.py` to bundle experiment inputs for the batch method:

```python
@dataclass
class ExperimentTask:
    experiment_id: str
    params: dict[str, Any]
    command: str
    baseline_commit: str
    resource: str = ""
```

### ExperimentExecutor batch method

New method on the ABC with a default sequential implementation:

```python
class ExperimentExecutor(ABC):
    @abstractmethod
    def run_experiment(self, experiment_id, params, command, baseline_commit, resource="") -> ExperimentResult:
        ...

    def run_experiments(self, tasks: list[ExperimentTask]) -> list[ExperimentResult]:
        """Run a batch of experiments. Default: sequential."""
        return [
            self.run_experiment(t.experiment_id, t.params, t.command, t.baseline_commit, t.resource)
            for t in tasks
        ]
```

`ScriptedExecutor` inherits the default implementation for free.

## Module Structure

```
chaosengineer/execution/
├── __init__.py          # create_executor() factory, re-exports
├── subagent.py          # SubagentExecutor — orchestration + parallelism
├── task_packet.py       # TaskPacketBuilder — constructs task markdown files
├── worktree.py          # WorktreeManager — git worktree lifecycle + branching
└── result_parser.py     # ResultParser — reads result.json, validates, builds ExperimentResult
```

### SubagentExecutor (`subagent.py`)

Implements `run_experiment()` and overrides `run_experiments()`. Wires together TaskPacketBuilder, WorktreeManager, and ResultParser. For `run_experiments()`, uses `concurrent.futures.ThreadPoolExecutor` with `max_workers` matching batch size in parallel mode, or 1 in sequential mode. Each `Future.result()` is wrapped in try/except to convert thread-level crashes into error `ExperimentResult` objects, mirroring the coordinator's existing exception handling pattern.

### TaskPacketBuilder (`task_packet.py`)

Takes `ExperimentTask` + full `WorkloadSpec` and generates a markdown task file. Uses fields: `modifiable_files`, `constraints_text`, `metric_parse_command`, `time_budget_seconds`, `primary_metric`, `metric_direction`, `execution_command`. Owns the prompt template. Writes to experiment output directory.

### WorktreeManager (`worktree.py`)

Creates worktree from `baseline_commit`, creates experiment branch (`chaosengineer/<run_id>/<experiment_id>`), cleans up worktree after results are collected. The branch persists even after worktree removal. The `run_id` prefix prevents branch name collisions across runs.

### ResultParser (`result_parser.py`)

Reads `result.json` from the experiment output directory, validates required fields (at minimum `primary_metric`), builds `ExperimentResult`. Returns error result if file missing or malformed.

### Factory (`__init__.py`)

```python
def create_executor(
    backend: str,
    spec: WorkloadSpec,
    output_dir: Path,
    mode: str = "sequential",
    scripted_results: Path | None = None,
) -> ExperimentExecutor:
```

- `backend="subagent"` → `SubagentExecutor(spec, output_dir, mode)`
- `backend="scripted"` → loads results from YAML file or folder of YAML files → `ScriptedExecutor(results)`

When `scripted_results` is a directory, all YAML files in the folder are loaded and merged.

## Per-Experiment Pipeline

Each experiment (potentially running in a thread) follows this pipeline:

```
1. WorktreeManager.create(baseline_commit, run_id, experiment_id)
   → git worktree add .chaosengineer/worktrees/<experiment_id> -b chaosengineer/<run_id>/<experiment_id> <baseline_commit>
   → returns worktree_path

2. TaskPacketBuilder.build(task, spec, worktree_path, result_file_path)
   → writes .chaosengineer/output/<experiment_id>/task.md
   → returns task_file_path

3. SubagentExecutor._invoke(task_file_path, worktree_path, timeout, resource)
   → prompt = task_file_path.read_text()  # read file in Python, not shell expansion
   → env = {**os.environ}; if resource: env["CUDA_VISIBLE_DEVICES"] = parse_gpu(resource)
   → subprocess.run(["claude", "-p", prompt, "--allowedTools", "Edit,Write,Bash,Read"],
                     cwd=worktree_path, env=env, timeout=timeout)
   → if time_budget_seconds set: timeout = budget + 60s grace
   → if time_budget_seconds is None: timeout = None (no limit)
   → if TimeoutExpired: kill process, return error ExperimentResult
   → returns subprocess CompletedProcess

4. ResultParser.parse(result_file_path, experiment_id, duration_seconds)
   → reads .chaosengineer/output/<experiment_id>/result.json
   → validates primary_metric exists
   → sets duration_seconds from wall-clock timing (measured by SubagentExecutor)
   → if file missing/malformed: returns ExperimentResult(primary_metric=0, error_message="...")
   → returns ExperimentResult

5. WorktreeManager.cleanup(worktree_path)
   → git worktree remove .chaosengineer/worktrees/<experiment_id>
   → branch chaosengineer/<run_id>/<experiment_id> persists with the commit
```

**Resource handling:** If `resource` is non-empty (e.g., `"gpu:0"`), `CUDA_VISIBLE_DEVICES` is set in the subprocess environment. The numeric suffix is extracted from the resource string. Resource assignment is currently out of scope — the `resource` field will be `""` for all workers. GPU assignment can be added in a future phase by mapping `workers_available` to device IDs.

**Timing and cost:** `SubagentExecutor` measures wall-clock `duration_seconds` around the subprocess call and passes it to `ResultParser`. `tokens_in`, `tokens_out`, and `cost_usd` remain 0 for Claude Code subscription-based execution (no per-call cost tracking). This means `max_api_cost` budget limits will not trigger for subagent execution — only `max_experiments`, `max_wall_time_seconds`, and `max_plateau_iterations` are effective budget constraints.

**Tool permissions:** The subagent gets `Edit,Write,Bash,Read` (broader than ClaudeCodeHarness's `Write`-only) because experiment execution requires modifying code, running shell commands, and reading files — not just writing JSON output.

**Timeout defaults:** `WorkloadSpec.time_budget_seconds` defaults to 300 in the parser. To support long-running experiments with no per-experiment limit, the parser will be updated to treat an absent `Time budget` field as `None` instead of defaulting to 300. When `None`, no subprocess timeout is applied.

## Task Prompt Template

The markdown file generated by TaskPacketBuilder:

```markdown
# Experiment: {experiment_id}

## Objective
Apply the following parameter changes and run the experiment command.
Report the results as a JSON file.

## Parameters
{params as YAML block}

## Working Directory
You are working in a git worktree at: {worktree_path}
Branch: chaosengineer/{run_id}/{experiment_id}

## Modifiable Files
{modifiable_files list, or "Any files" if empty}

## Constraints
{constraints_text, or "None" if empty}

## Instructions
1. Study the codebase to understand how the parameters above map to code
2. Modify the relevant files to apply these parameter values
3. Commit your changes with message: "experiment {experiment_id}: {one-line param summary}"
4. Run the experiment command: `{execution_command}`
5. {if metric_parse_command: "Parse metrics by running: `{metric_parse_command}`"}
   {else: "Extract the primary metric '{primary_metric}' from the command output"}
6. Write results to: {result_file_path}

## Time Budget
{if time_budget_seconds: "Complete within {time_budget_seconds} seconds"}
{else: "No time limit — run to completion"}

## Result Format
Write ONLY this JSON to {result_file_path}:

    {
      "primary_metric": <float>,
      "secondary_metrics": {"metric_name": <float>, ...},
      "artifacts": ["path/to/artifact", ...],
      "commit_hash": "<git commit hash of your changes>",
      "error_message": null
    }

If the experiment fails, set error_message to a description of what went wrong.
Primary metric name: {primary_metric} ({metric_direction} is better)
```

## Coordinator Changes

The existing `_run_iteration` loop iterates `for i, params in enumerate(plan.values)`, creates an `Experiment` and ephemeral `WorkerState` per experiment, then calls `self.executor.run_experiment()` inline. The batch refactor splits this into two phases: prepare tasks, then execute and handle results.

```python
def _run_iteration(self, plan, baseline):
    results = []

    # Phase 1: Build Experiment objects and task list
    tasks = []
    experiments = []
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

        tasks.append(ExperimentTask(exp_id, params, self.spec.execution_command, baseline.commit))
        experiments.append((exp, worker))

    # Phase 2: Execute batch (sequential or parallel depending on executor)
    batch_results = self.executor.run_experiments(tasks)

    # Phase 3: Handle results one at a time (same logic as before)
    for (exp, worker), result in zip(experiments, batch_results):
        if result.error_message:
            fail_experiment(exp, result)
            self._log(Event(event="worker_failed", data={...}))
        else:
            complete_experiment(exp, result)
            self._log(Event(event="worker_completed", data={...}))
        release_worker(worker)
        self.budget.record_experiment()
        self.budget.add_cost(result.cost_usd if result else 0.0)
        results.append((exp, result))

    return results
```

**Behavioral changes from batching:**
- All experiments in an iteration are started (state: RUNNING) before any complete. This is accurate — they are conceptually running simultaneously.
- Budget tracking (`record_experiment`, `add_cost`) happens after the batch returns, not between experiments. This means the coordinator cannot abort mid-iteration if budget is exceeded — it completes the full iteration then checks. This is acceptable: budget checks between iterations already exist in the outer `run()` loop.
- Event logging timestamps for `worker_completed`/`worker_failed` will reflect when results are processed (post-batch), not when individual experiments finish. The actual experiment duration is captured in `result.duration_seconds`.

## CLI Changes

New flags on the `run` subcommand:

```python
run_parser.add_argument("--executor", choices=["subagent", "scripted"], default="subagent")
run_parser.add_argument("--mode", choices=["sequential", "parallel"], default="sequential")
run_parser.add_argument("--scripted-results", type=Path, help="YAML file or folder with canned results")
```

Validation: `--executor=scripted` requires `--scripted-results`.

Full wiring replaces the current placeholder:

```python
spec = parse_workload_spec(args.workload)
dm = create_decision_maker(args.llm_backend, spec, llm_dir)
executor = create_executor(args.executor, spec, args.output_dir, args.mode,
                           scripted_results=args.scripted_results)
logger = EventLogger(args.output_dir / "events.jsonl")
budget = BudgetTracker(spec.budget)

coordinator = Coordinator(
    spec=spec, decision_maker=dm, executor=executor,
    logger=logger, budget=budget,
)
coordinator.run()
```

## Testing Strategy

| Layer | Scope | Count |
|-------|-------|-------|
| Unit | TaskPacketBuilder, WorktreeManager, ResultParser, SubagentExecutor (mocked deps) | ~20 |
| Integration | Coordinator + batch `run_experiments()` with ScriptedExecutor | ~8 |
| End-to-end | CLI → Coordinator → ScriptedExecutor, file and folder-based scripted results | ~5 |
| Existing | All 117 current tests pass unchanged (fix 2 collection errors in test_runner.py, test_shipped_scenarios.py) | 117 |

**~33 new tests, ~150 total.**

### Unit tests

- **`test_task_packet.py`** — correct markdown for various param combos, handles missing optional fields
- **`test_worktree.py`** — creates/cleans up worktrees, creates named branches, handles missing worktree cleanup. Mocks git subprocess calls.
- **`test_result_parser.py`** — valid JSON, missing file, malformed JSON, missing primary_metric, extra fields ignored
- **`test_subagent_executor.py`** — wires components correctly, parallelizes with ThreadPoolExecutor, handles timeout and subprocess failure

### Integration tests

- **`test_batch_coordinator.py`** — coordinator calls `run_experiments()` with correct task list, handles mixed success/failure in batch, budget tracking across batches

### End-to-end tests

- Full pipeline: CLI → parse spec → create executor (scripted) → coordinator.run() → verify events and results
- Scripted results from single YAML file
- Scripted results from folder of YAML files
- Validates event log completeness, budget exhaustion, breakthrough detection
- **Autoresearch scenario**: The original autoresearch workload (neural network training) run end-to-end with scripted results. Validates the full pipeline against the real use case this project was built for — the workload spec, dimension space, and expected optimization flow, with only the actual GPU execution replaced by scripted results.

## Out of Scope

- Human-in-the-loop evaluation (`evaluation_type: "human"`)
- Docker-based isolation (alternative to git worktrees)
- Multi-machine distributed execution
- Artifact storage/management beyond file paths
- Result caching / experiment deduplication
