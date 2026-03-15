# ChaosEngineer Design Spec

**Project:** deblasis/chaosengineer (fork of autoresearch)
**Date:** 2026-03-15
**Status:** Approved

## Vision

ChaosEngineer is a general-purpose parallel experimentation framework. It evolves autoresearch's single-agent sequential experiment loop into a coordinator/worker architecture that can run multiple experiments simultaneously across arbitrary workloads.

The framework supports two modes:
- **Sequential** (`--mode sequential`): one worker, sequential execution, full worker autonomy. Backward-compatible with autoresearch's existing behavior.
- **Parallel** (`--mode parallel`): coordinator LLM plans experiments, spawns multiple workers, manages lifecycle with selective reassignment on breakthroughs.

## Core Concepts

### WorkloadSpec

A human-authored markdown document that describes the experiment domain to the coordinator. Analogous to autoresearch's `program.md`. Contains: domain context, experiment space (dimensions to explore), execution instructions, evaluation criteria, resource requirements, budget constraints, and safety rails.

The coordinator can optionally generate a structured JSON schema from the spec, which is passed to workers as an unambiguous execution contract.

### Run

A complete experimentation session. One run = one workload spec + N experiments across multiple iterations. Tracks aggregate metrics and maintains a results log (JSONL).

### Experiment

A single unit of work with specific parameters assigned by the coordinator. State machine:

```
planned -> assigned -> running -> completed
                               -> failed
                               -> killed
```

### Worker

A Claude Code subagent running in an isolated git worktree with an assigned resource (e.g., GPU). Each worker executes exactly one experiment: it receives a task packet, runs the experiment, returns a result packet, and terminates. Workers are stateless and disposable — they have no memory of previous experiments and no awareness of other workers.

In parallel mode, workers do NOT decide what to try, interpret results, communicate with other workers, or retry on failure without coordinator approval. In sequential mode, the single worker operates autonomously per the original autoresearch contract (full decision-making authority over what to try next).

### Coordinator

The main Claude Code session. Reads the workload spec, plans experiment iterations, spawns workers, monitors results, handles breakthroughs, and decides when to stop.

## Coordinator Architecture

### Experiment Planning: One Dimension at a Time

The coordinator follows a disciplined search strategy rather than freely assigning diverse experiments:

1. Pick one **dimension** (variable/parameter) to explore per iteration
2. Allocate workers based on that dimension's type and cardinality
3. Run all workers in parallel from the same baseline
4. Compare results (all directly comparable since only one thing changed)
5. Pick winner, adopt as new baseline, move to next dimension

This is coordinate descent with parallel sweep per dimension. The LLM decides which dimension is most promising to explore next.

**Known limitation (v1):** Coordinate descent assumes dimensions are mostly independent. When dimensions interact strongly (e.g., depth and learning rate are coupled), optimizing one at a time may miss the best combination. The beam search on ties partially mitigates this by exploring combinations when results are ambiguous. A future enhancement could add an explicit "interaction probing" phase where the coordinator tests promising dimension pairs jointly.

### Dimension Types

| Type | Description | Example | Workers |
|------|-------------|---------|---------|
| **Directional** | Numerical value that can increase or decrease | Learning rate, model depth | 2 (one per direction) |
| **Enum** | Finite set of known options | Activation function: GeLU, SiLU, ReLU | N = cardinality |
| **Diverse** | Large qualitative space of non-numerical options | Marketing strategies, prompt variations, thumbnail concepts | N = saturated set size (budget-constrained) |

### Diverse Dimension Discovery

For diverse dimensions, the coordinator runs a discovery phase before testing:

1. Ask the LLM to generate diverse options (no target count)
2. Ask the LLM to critique: "are any overlapping? is there a fundamentally different approach not covered?"
3. Repeat until the LLM can't produce genuinely novel options (starts recycling themes or declares saturation)
4. The exhaustion point = the natural cardinality = the **saturated set**
5. Coordinator fits the set to available resources/budget (test all if affordable, sample if too many)

This is cheap (text generation only, no experiment execution) and produces the largest non-overlapping option set before committing resources. Note: while no experiment execution occurs, the discovery phase does consume LLM tokens. These costs count toward the run's API budget.

### Branching on Ties (Beam Search)

When a batch completes and multiple options produce results within a **tie threshold** of each other, they are all treated as winners:

- **Tie detection**: The workload spec can define a threshold (e.g., "results within 1% of each other"). If unspecified, the coordinator LLM makes a judgment call based on domain context (e.g., for val_bpb, a difference of 0.001 may be noise). Exact equality is not required.
- All tied winners become **parallel baselines** for the next iteration
- The next dimension is explored independently from each baseline
- This creates a branching tree of exploration

```
Iteration 1: Test dimension A (5 options)
  -> Options A2 and A4 tie for best

Iteration 2: Fork -- explore dimension B from both
  Branch A2: Test dimension B (3 options) -> B1 wins
  Branch A4: Test dimension B (3 options) -> B3 wins

Iteration 3: Compare A2+B1 vs A4+B3 -> single winner, branches merge
```

The beam widens when results tie (exploring combinations) and narrows when a clear winner emerges. Worker count per iteration = (active branches) x (dimension cardinality). If this exceeds resources or budget, the coordinator prunes weakest branches, reduces sample size, or serializes some branches.

**Branch merging**: When branches are compared, the coordinator evaluates them on the primary metric. The best-performing branch becomes the sole baseline going forward. The losing branch's worktree is discarded (its results are still recorded in the metrics log for analysis). If branches tie again at the merge point, the beam stays wide and another dimension is explored from each — the coordinator does not branch indefinitely, as budget constraints naturally limit beam width.

### Coordinator Loop

```
1. Read workload spec, detect resources, parse budget constraints
2. Analyze experiment space -> identify dimensions to explore
3. Prioritize dimensions (LLM reasoning about impact)
4. FOR each dimension (until budget/stopping criteria):
   a. If diverse type: run discovery phase to find saturated set
   b. Determine cardinality -> worker count (capped by resources/budget)
   c. Assign one value per worker
   d. Spawn workers in parallel (background subagents in worktrees)
   e. Collect all results
   f. Evaluate: pick winner (automatic metric OR human-in-the-loop)
   g. If tie: branch (beam search)
   h. If clear winner: update baseline, merge branches if applicable
   i. Record metrics (cost, tokens, time)
   j. Re-prioritize remaining dimensions given what was learned
5. Report final results
```

### Selective Reassignment

A **breakthrough** is any result that establishes a new best baseline — i.e., an experiment produces a better primary metric than the current best. When a breakthrough occurs within a batch (one worker finishes before others), the coordinator flags remaining running workers whose experiments are now stale (exploring a region made irrelevant by the breakthrough). Since workers are single-experiment-and-terminate, "selective reassignment" means: the coordinator simply does not spawn follow-up work based on stale plans. Workers already running finish naturally (their results are still recorded), but the coordinator plans the next iteration from the new baseline rather than the old one. True mid-execution kill is a future enhancement.

### Baseline Management

The coordinator maintains a "current best" state tied to a specific git commit. Each experiment forks from a baseline commit. When a breakthrough happens, that commit becomes the new baseline. Workers running from an older baseline are marked stale -- their results are still recorded (informative) but won't become the new baseline unless they beat the updated best.

### Stopping Criteria

- Token/cost budget exhausted
- Max experiment count reached
- Wall-clock time limit
- N consecutive iterations with no improvement (plateau)
- User-defined criteria from the workload spec (e.g., "stop when val_bpb < 0.95")

## Worker Architecture

### Lifecycle

1. Coordinator spawns worker as a background Claude Code subagent with `isolation: "worktree"`
2. Worker receives a **task packet** containing:
   - Baseline commit to start from
   - Exact parameters/changes to apply
   - Execution command (from schema or workload spec)
   - How to parse results
   - Resource assignment (e.g., `CUDA_VISIBLE_DEVICES=2`)
3. Worker executes:
   - Applies specified changes (edit files, set params)
   - Commits the change in its worktree
   - Runs the experiment command
   - Parses results from output
   - Writes a **result packet** (metrics, artifacts, commit hash, timing, token usage)
4. Worker returns result packet to coordinator and terminates
5. Worktree is cleaned up (or preserved if the experiment was a winner)

### On Failure

Worker reports the failure (crash log, error message) back to the coordinator. The coordinator decides whether to retry, skip, or adjust.

### Resource Isolation

- **Git**: each worker in its own worktree (no conflicts)
- **GPU**: coordinator assigns `CUDA_VISIBLE_DEVICES` per worker
- **Filesystem**: worktree provides full isolation
- **Results**: each worker writes its own result packet, coordinator aggregates

## Workload Specification Format

```markdown
# Workload: <name>

## Context
What this workload does, domain knowledge the coordinator needs.

## Experiment Space
Dimensions that can be varied, organized by type:
- Directional: "learning rate (currently 0.04)"
- Enum: "activation function: GeLU, SiLU, ReLU, GELU-approx"
- Diverse: "attention mechanism design"
Constraints between dimensions if any.

## Execution
- Command: `uv run train.py > run.log 2>&1`
- Time budget per experiment: 5 minutes
- Setup steps, teardown steps

## Evaluation
- Type: automatic | human
- Metric: val_bpb (lower is better)
- Parse: `grep "^val_bpb:" run.log`
- Secondary metrics: peak_vram_mb, training_seconds
- For human: what artifacts to produce, scoring criteria

## Resources
- Per worker: 1 GPU, ~44GB VRAM
- Available: auto-detect | user-specified count

## Budget
- Max API cost: $50
- Max experiments: 100
- Max wall time: 8h
- Priority: cost | speed | thoroughness

## Constraints
- Files workers may modify
- Files that must not change
- Safety rails
```

### Backward Compatibility

The existing `program.md` format is accepted in sequential mode. The coordinator detects unstructured specs and runs as a single autonomous agent (autoresearch behavior). To unlock parallel mode, users add the structured sections.

## Human-in-the-Loop Evaluation

For workloads where results cannot be scored automatically:

1. The workload spec declares `evaluation: human`
2. After a batch completes, workers produce **artifacts** (files, screenshots, logs, diffs) in a structured review directory
3. The system writes a **review manifest** listing each experiment with its artifacts and a blank score field
4. The coordinator pauses and notifies the user
5. The user reviews artifacts at their convenience, fills in scores
6. The coordinator detects scores, parses them, resumes

For overnight runs: the system batches all pending human reviews so the user wakes up to a single review queue. The coordinator continues exploring dimensions that don't require human evaluation while reviews are pending.

## Testing Framework

### Purpose

Validate orchestration logic without LLM calls or real experiment execution. Scenarios complete in seconds.

### What Gets Mocked

| Real component | Simulator replacement |
|---|---|
| LLM API (coordinator reasoning) | Scripted decisions |
| LLM API (worker execution) | Scripted task execution |
| Workload command (e.g., `uv run train.py`) | Instant return with scripted metrics |
| Git worktrees | Lightweight temp directories or skipped |

### Scenario Format

Test scenarios are YAML files that script an entire run:

```yaml
scenario: "breakthrough triggers reassignment"
workload_spec: specs/test-workload.md

dimensions:
  - name: learning_rate
    type: directional
    baseline: 0.04

simulated_results:
  - worker: 1
    params: { learning_rate: 0.08 }
    result: { val_bpb: 0.95, status: completed }
  - worker: 2
    params: { learning_rate: 0.02 }
    result: { val_bpb: 0.91, status: completed }

expected_behavior:
  - new_baseline_from: worker_2
  - next_dimension: depth
```

### What Can Be Tested

- Coordinator loop logic (dimension selection, worker allocation, budget tracking)
- Branching on ties (beam search behavior)
- Selective reassignment after breakthroughs
- Budget enforcement
- Human-in-the-loop flow (pause, wait for scores, resume)
- Metric collection and aggregation
- Edge cases: all workers crash, all tie, budget exhausted mid-batch

### Shipped Scenarios

A library of scenarios ships with the framework covering common patterns: breakthrough, plateau, tie-branching, budget exhaustion, worker failure, human review.

## Metrics & Observability

### Tracked Metrics

| Category | Metrics |
|---|---|
| **Cost** | LLM tokens (in/out per call), API cost per worker, per iteration, coordinator reasoning cost, cumulative total (coordinator + all workers) |
| **Experiments** | Total, completed, failed, killed, per-dimension breakdown |
| **Performance** | Experiment duration, coordinator think-time, worker spin-up time, idle time |
| **Search progress** | Best metric over time, improvement rate, dimensions explored/remaining |
| **Resources** | Active workers, GPU utilization (if available), worktree count |
| **Budget** | Remaining cost/experiments/time, projected exhaustion |

### Export

**JSONL event log** (source of truth) -- one event per significant action:

```jsonl
{"ts":"...","event":"run_started","workload":"nn-arch-search","budget":{"max_cost":50,"max_experiments":100}}
{"ts":"...","event":"iteration_started","dimension":"depth","type":"directional","workers":2}
{"ts":"...","event":"worker_completed","worker_id":"w1","params":{"depth":12},"result":{"val_bpb":0.93},"tokens":{"in":1200,"out":800},"cost_usd":0.04,"duration_s":312}
{"ts":"...","event":"breakthrough","new_best":0.93,"previous_best":0.97,"from_worker":"w1"}
{"ts":"...","event":"budget_checkpoint","spent_usd":1.20,"remaining_usd":48.80,"experiments_run":8}
```

**Prometheus endpoint and OTLP export** are planned for a future version alongside the web UI. In v1, the JSONL log is the sole export format. Tooling to convert JSONL to Prometheus/OTLP formats can be added incrementally without changing the core logging.

The JSONL log ensures metrics are available even without a running collector -- post-hoc analysis of completed runs is always possible.

### Per-Run Summary

On completion, the coordinator writes a human-readable and machine-parseable summary: total cost, best result, experiment breakdown, dimension-by-dimension results.

## Project Structure

```
chaosengineer/
  core/           # coordinator, worker manager, experiment state machine
  metrics/        # JSONL logger (OTel/Prometheus exporters planned)
  testing/        # LLM simulator, scenario runner, shipped scenarios
  workloads/      # workload spec loader, schema generator
program.md        # workload definition (user-authored)
train.py          # autoresearch workload (first workload)
prepare.py        # autoresearch data prep (unchanged)
```

## Modes

| Flag | Behavior |
|------|----------|
| `--mode sequential` | Single worker, sequential execution, full worker autonomy. Backward-compatible with autoresearch. |
| `--mode parallel` | Coordinator plans, multiple workers, structured dimensions, beam search, budget enforcement. |
