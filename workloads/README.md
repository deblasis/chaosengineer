# Autoresearch Workloads

Ready-to-run workload specs for ChaosEngineer, based on Andrej Karpathy's
autoresearch framework.

## Prerequisites

- ChaosEngineer installed: `uv pip install -e .`
- For live runs: NVIDIA GPU, CUDA toolkit, data prepared via `uv run prepare.py`
- For scripted demo: no GPU needed

## Workloads

### 1. Original Climbmix (live GPU required)

The original autoresearch workflow: optimize a small language model on the
climbmix-400b dataset to minimize val_bpb within a 5-minute training budget.

**Setup:**
```bash
uv run prepare.py  # download data + train tokenizer (~2 min)
```

**Sequential (original autoresearch behavior):**
```bash
chaosengineer run workloads/autoresearch-climbmix.md \
  --llm-backend claude-code \
  --executor subagent \
  --mode sequential
```

**Parallel (faster, higher API cost):**
```bash
chaosengineer run workloads/autoresearch-climbmix.md \
  --llm-backend claude-code \
  --executor subagent \
  --mode parallel
```

### 2. Irish Folk Music — Scripted Demo (no GPU needed)

Replay of a real 18-experiment optimization run training on ABC notation
Irish folk music (Sanderwoods Irishman dataset). Based on the experiment
by Onchain AI Garage.

val_bpb progression: 2.08 → 0.97 (53% improvement over 18 experiments).

```bash
chaosengineer run workloads/autoresearch-irish-music.md \
  --llm-backend scripted \
  --scripted-plans workloads/irish-music/scripted_plans.yaml \
  --executor scripted \
  --scripted-results workloads/irish-music/scripted_results.yaml \
  --mode sequential
```

### 3. Irish Folk Music — Live (GPU required)

Run the Irish folk music experiment for real with LLM-driven decisions:

**Setup:**
Modify prepare.py to download the Sanderwoods Irishman dataset instead of
climbmix (see autoresearch-irish-music.md for details).

**Sequential:**
```bash
chaosengineer run workloads/autoresearch-irish-music.md \
  --llm-backend claude-code \
  --executor subagent \
  --mode sequential
```

**Parallel:**
```bash
chaosengineer run workloads/autoresearch-irish-music.md \
  --llm-backend claude-code \
  --executor subagent \
  --mode parallel
```

## Sequential vs Parallel

| Mode | Behavior | Trade-off |
|------|----------|-----------|
| Sequential | One experiment per iteration, decide, advance/discard. Matches original autoresearch loop. | Slower but cost-efficient — no wasted experiments. |
| Parallel | Multiple experiments per iteration using git worktrees. | Faster wall-clock time but higher API cost — some experiments may be discarded even if they ran successfully. |

## Output

All runs produce:
- `events.jsonl` — full event log (iterations, breakthroughs, completions)
- `llm_decisions/` — LLM decision maker prompts and responses (live runs only)
