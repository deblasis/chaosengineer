# Design: Real-World Workloads & Scripted Irish Music Experiment

## Goal

Validate ChaosEngineer can orchestrate realistic multi-iteration optimization runs by:

1. Creating runnable workload specs that replicate the original autoresearch workflow
2. Building a scripted 18-experiment replay of a real optimization run (Irish folk music ABC notation training) for demo and validation purposes

## Deliverables

### File Layout

```
workloads/
├── README.md                          # How to run both workloads
├── autoresearch-climbmix.md           # Original autoresearch workload spec
├── autoresearch-irish-music.md        # Irish folk music variant
└── irish-music/
    ├── scripted_plans.yaml            # 18 DimensionPlans in sequence
    └── scripted_results.yaml          # 18 experiment results with video metrics

tests/e2e/
└── test_irish_music_pipeline.py       # E2E test: full 18-iteration run
```

`workloads/` is a new top-level directory containing real, runnable workload specs — not test fixtures.

### Workload Spec: autoresearch-climbmix.md

Maps the original `program.md` into ChaosEngineer's workload format. Targets `train.py` modification, `val_bpb` minimization on the climbmix-400b dataset with a 5-minute training budget.

Dimensions: batch_size, depth, aspect_ratio, learning_rate, warmup_fraction, cooldown_fraction, weight_decay, head_dim (all Directional), activation and window_pattern (Enum), optimizer_strategy (Diverse).

Budget: 100 experiments, 8h wall time. Workers: 1 (sequential by default, overridable to parallel via CLI).

Constraints: only `train.py` modifiable. `prepare.py` is read-only.

### Workload Spec: autoresearch-irish-music.md

Same structure, targeting the Sanderwoods Irishman ABC notation dataset. Based on the experiment documented by Onchain AI Garage.

Key differences from climbmix:
- Context describes ABC notation (limited charset, rigid syntax, repeated patterns) and the insight that smaller/faster models win
- Budget: 20 experiments, 2h (matching the video's ~18 experiments)
- `prepare.py` is modifiable (agent adjusted it to load the Irishman dataset)
- Adds `value_embeddings` as an Enum dimension (options: true, false) — explored in experiment 15

### Scripted Irish Music Experiment

Faithful reproduction of all 18 experiments from the video.

**scripted_plans.yaml** — 18 dimension plans in order:

| # | Dimension | Value | Result |
|---|-----------|-------|--------|
| 1 | depth | 12 | 2.15 (discard) |
| 2 | batch_size | 131072 (2^17) | 1.44 (keep) |
| 3 | batch_size | 32768 (2^15) | 1.03 (keep) |
| 4 | batch_size | 16384 (2^14) | 1.01 (keep) |
| 5 | depth | 10 | 1.05 (discard) |
| 6 | learning_rate | 0.08 | 1.03 (discard) |
| 7 | aspect_ratio | 32 | 0.99 (keep) |
| 8 | aspect_ratio | 24 | 0.985 (keep) |
| 9 | warmup_fraction | 0.05 | 0.97 (keep) |
| 10 | warmup_fraction | 0.10 | 0.98 (discard) |
| 11 | cooldown_fraction | 0.30 | 0.975 (discard) |
| 12 | window_pattern | SSSS | 0.98 (discard) |
| 13 | weight_decay | 0.01 | 0.975 (discard) |
| 14 | weight_decay | 0.1 | 0.99 (discard) |
| 15 | value_embeddings | true | 0.972 (discard) |
| 16 | head_dim | 64 | 0.98 (discard) |
| 17 | head_dim | 96 | 0.975 (discard) |
| 18 | aspect_ratio | 28 | 0.978 (discard) |

Progression: 2.08 (baseline) → 0.97 (best). 6 keeps (experiments 2, 3, 4, 7, 8, 9), 12 discards.

**scripted_results.yaml** — keyed by `exp-{iteration}-{index}` (0-indexed: experiment 1 → `exp-0-0`, experiment 2 → `exp-1-0`, ..., experiment 18 → `exp-17-0`). Each entry has primary_metric, secondary_metrics (num_steps, peak_vram_mb).

### E2E Test: test_irish_music_pipeline.py

Wires scripted plans + results through the full coordinator loop. Validates:
- All 18 experiments execute
- Final best metric is 0.97
- 6 breakthroughs with monotonically improving values
- Complete event log (run_started, 18x iteration_started, 18x worker_completed, run_completed)
- Secondary metrics flow through the pipeline

### CLI Change: Fully Scripted Runs

Add `--llm-backend scripted` alongside `claude-code` and `sdk`. When used, requires `--scripted-plans <path>` pointing to a plans YAML file. The `scripted` branch in `_execute_run` instantiates `ScriptedDecisionMaker` directly from `chaosengineer.testing.simulator`, bypassing `create_decision_maker` entirely (since `ScriptedDecisionMaker` is not an LLM backend).

Fully scripted invocation:
```bash
chaosengineer run workloads/autoresearch-irish-music.md \
  --llm-backend scripted \
  --scripted-plans workloads/irish-music/scripted_plans.yaml \
  --executor scripted \
  --scripted-results workloads/irish-music/scripted_results.yaml \
  --mode sequential
```

This requires:
- New `--scripted-plans` argument in cli.py
- New branch in `_execute_run` for `args.llm_backend == "scripted"`
- A YAML→DimensionPlan loader in `chaosengineer/workloads/plan_loader.py`

**Plan YAML schema:**
```yaml
plans:
  - dimension_name: depth
    values:
      - depth: 12
  - dimension_name: batch_size
    values:
      - batch_size: 131072
```

Each entry maps to `DimensionPlan(dimension_name=..., values=[...])` where `values` is a list of single-key dicts matching the dimension name.

### README: workloads/README.md

Documents three run modes:
1. **Climbmix live** (GPU required) — sequential or parallel
2. **Irish music scripted** (no GPU) — deterministic replay
3. **Irish music live** (GPU required) — sequential or parallel

Includes a table comparing sequential vs parallel trade-offs.

## Sequential vs Parallel

Both workload specs support either mode via CLI flags:
- **Sequential**: one experiment per iteration, matches original autoresearch loop. Cost-efficient.
- **Parallel**: multiple experiments per iteration via git worktrees. Faster wall-clock but higher API cost since some experiments may be discarded.

## What Does NOT Change

No modifications to: coordinator, executor, budget tracker, parser, event logger, or any existing test. The existing infrastructure handles everything — we're only adding workload specs, scripted data, a test, and a small CLI enhancement.
