# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. I'm not 100% sure that I want to take this on personally right now. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

Seeing as there seems to be a lot of interest in tinkering with autoresearch on much smaller compute platforms than an H100, a few extra words. If you're going to try running autoresearch on smaller computers (Macbooks etc.), I'd recommend one of the forks below. On top of this, here are some recommendations for how to tune the defaults for much smaller models for aspiring forks:

1. To get half-decent results I'd use a dataset with a lot less entropy, e.g. this [TinyStories dataset](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). These are GPT-4 generated short stories. Because the data is a lot narrower in scope, you will see reasonable results with a lot smaller models (if you try to sample from them after training).
2. You might experiment with decreasing `vocab_size`, e.g. from 8192 down to 4096, 2048, 1024, or even - simply byte-level tokenizer with 256 possibly bytes after utf-8 encoding.
3. In `prepare.py`, you'll want to lower `MAX_SEQ_LEN` a lot, depending on the computer even down to 256 etc. As you lower `MAX_SEQ_LEN`, you may want to experiment with increasing `DEVICE_BATCH_SIZE` in `train.py` slightly to compensate. The number of tokens per fwd/bwd pass is the product of these two.
4. Also in `prepare.py`, you'll want to decrease `EVAL_TOKENS` so that your validation loss is evaluated on a lot less data.
5. In `train.py`, the primary single knob that controls model complexity is the `DEPTH` (default 8, here). A lot of variables are just functions of this, so e.g. lower it down to e.g. 4.
6. You'll want to most likely use `WINDOW_PATTERN` of just "L", because "SSSL" uses alternating banded attention pattern that may be very inefficient for you. Try it.
7. You'll want to lower `TOTAL_BATCH_SIZE` a lot, but keep it powers of 2, e.g. down to `2**14` (~16K) or so even, hard to tell.

I think these would be the reasonable hyperparameters to play with. Ask your favorite coding agent for help and copy paste them this guide, as well as the full source code.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

---

## ChaosEngineer

ChaosEngineer is an evolution of autoresearch into a general-purpose parallel experimentation framework. While autoresearch gives you a single agent running experiments in a loop, ChaosEngineer adds LLM-driven experiment planning, parallel execution, budget tracking, a TUI dashboard, human-in-the-loop evaluation, and pause/resume — turning the original concept into something you can point at any optimization problem and let run unattended.

### What it adds

- **LLM-driven experiment planning** — instead of the agent deciding what to try next on its own, a dedicated decision-maker LLM picks which dimension of the experiment space to explore and generates parameter variations using a coordinate-descent strategy.
- **Parallel execution** — run multiple experiments simultaneously using git worktrees for isolation. Each experiment gets its own branch and working directory.
- **Workload specs** — define your optimization problem in a simple Markdown format: what to run, what to measure, what parameters to explore, and budget limits. ChaosEngineer handles the rest.
- **Budget tracking** — set limits on API cost, experiment count, wall-clock time, or plateau iterations. ChaosEngineer stops gracefully when any limit is hit.
- **Pause and resume** — Ctrl+C triggers a graceful pause. Resume later with `chaosengineer resume`, optionally extending the budget.
- **TUI dashboard** — toggle a live terminal dashboard (press `t`) to see experiment progress, metrics, and budget status in real time.
- **Human-in-the-loop evaluation** — for workloads where quality can't be measured automatically (creative output, UX, etc.), ChaosEngineer can pause after each experiment and ask a human to score the result via the TUI.
- **Event bus** — an in-process event bus streams experiment events (iterations, breakthroughs, completions) to the TUI dashboard in real time.
- **Scripted/demo mode** — replay recorded experiment runs without a GPU or API key, useful for testing and demos.

### Installation

```bash
# Install with uv (recommended)
uv pip install -e .

# For Anthropic SDK backend (alternative to Claude Code CLI)
uv pip install -e '.[sdk]'
```

### Quick start

```bash
# Run the original autoresearch workflow via ChaosEngineer (sequential, one experiment at a time)
chaosengineer run workloads/autoresearch-climbmix.md \
  --llm-backend claude-code \
  --executor subagent \
  --mode sequential

# Run in parallel (faster, higher API cost)
chaosengineer run workloads/autoresearch-climbmix.md \
  --llm-backend claude-code \
  --executor subagent \
  --mode parallel

# Try the scripted demo (no GPU or API key needed)
chaosengineer run workloads/autoresearch-irish-music.md \
  --llm-backend scripted \
  --scripted-plans workloads/irish-music/scripted_plans.yaml \
  --executor scripted \
  --scripted-results workloads/irish-music/scripted_results.yaml
```

### CLI commands

| Command | Description |
|---------|-------------|
| `chaosengineer run WORKLOAD` | Start a new experiment run from a workload spec |
| `chaosengineer resume OUTPUT_DIR WORKLOAD` | Resume a paused or crashed run, optionally extending budget |
| `chaosengineer test [SCENARIO_YAML]` | Run a scenario YAML file, or all built-in scenarios if omitted |
| `chaosengineer version` | Print version |

Key flags for `run` and `resume`:

| Flag | Options | Default |
|------|---------|---------|
| `--llm-backend` | `claude-code`, `sdk`, `scripted` | `claude-code` |
| `--executor` | `subagent`, `scripted` | `subagent` |
| `--mode` | `sequential`, `parallel` | `sequential` |
| `--tui` | enable TUI dashboard on start (toggle with `t` during run) | off |
| `--initial-baseline` | override the baseline metric value | auto-detect |
| `--output-dir` | where to write run artifacts | `.chaosengineer/output` |
| `--force-fresh` | skip interactive resume prompt, start fresh | off |

Resume-specific flags: `--add-cost`, `--add-experiments`, `--add-time`, `--restart-iteration`.

### LLM backends

ChaosEngineer supports multiple LLM backends for the decision-making layer:

- **`claude-code`** (default) — invokes the [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI as a subprocess. Uses your existing Claude subscription, no separate API key needed.
- **`sdk`** — uses the Anthropic Python SDK directly. Requires `ANTHROPIC_API_KEY`. Supports alternative Anthropic-compatible providers via `ANTHROPIC_BASE_URL` (e.g. [OpenRouter](https://openrouter.ai), [Z.AI](https://z.ai), [Kimi](https://kimi.ai)). Override the model with `ANTHROPIC_MODEL` (default: `claude-sonnet-4-20250514`).
- **`scripted`** — replays pre-recorded plans from a YAML file. No LLM calls, no API key. Useful for testing and demos.

### Writing a workload spec

A workload spec is a Markdown file that describes your optimization problem. Here's the structure:

```markdown
# Workload: My Optimization Task

## Context
What you're optimizing and why. This is passed to the LLM
for context when planning experiments.

## Experiment Space
- Directional: "learning_rate" (currently 0.001)
- Directional: "batch_size" (currently 64)
- Enum: "optimizer" options: Adam, AdamW, SGD
- Diverse: "architecture_style"

## Execution
- Command: `python train.py > run.log 2>&1`
- Time budget per experiment: 5 minutes

## Evaluation
- Type: automatic
- Metric: val_loss (lower is better)
- Parse: `grep "^val_loss:" run.log | awk '{print $2}'`
- Secondary metrics: peak_vram_mb, throughput

## Resources
- Per worker: 1 GPU
- Available: 2

## Budget
- Max experiments: 50
- Max wall time: 4h
- Max API cost: $20

## Baseline
- Metric value: 0.92

## Constraints
- Files workers may modify: train.py, config.yaml
- Do not modify evaluate.py
```

The `## Baseline` section is optional for live runs (ChaosEngineer can auto-detect the baseline by running the command once), but required when using `--executor scripted` unless you pass `--initial-baseline`.

**Dimension types:**
- **Directional** — numeric parameter to sweep up/down from a current value
- **Enum** — discrete choices from a fixed list
- **Diverse** — the LLM generates creative options on the fly (useful for open-ended dimensions like "architecture style" or "optimizer strategy")

**Evaluation types:**
- **automatic** — metric extracted from command output via the parse command
- **human** — ChaosEngineer pauses after each experiment and prompts for a human score via the TUI

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                    CLI (cli.py)                      │
├─────────────────────────────────────────────────────┤
│              Coordinator (core/)                     │
│  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐ │
│  │ Budget   │ │ Snapshot │ │ Pause  │ │ State    │ │
│  │ Tracker  │ │ & Resume │ │ Control│ │ Machine  │ │
│  └──────────┘ └──────────┘ └────────┘ └──────────┘ │
├──────────────────┬──────────────────────────────────┤
│  LLM Layer       │  Execution Layer                 │
│  ┌────────────┐  │  ┌─────────────┐ ┌────────────┐ │
│  │ Decision   │  │  │ Subagent    │ │ Worktree   │ │
│  │ Maker      │  │  │ Executor    │ │ Manager    │ │
│  ├────────────┤  │  ├─────────────┤ └────────────┘ │
│  │ Claude Code│  │  │ Task Packet │                 │
│  │ SDK        │  │  │ Builder     │                 │
│  │ Scripted   │  │  │ Result      │                 │
│  └────────────┘  │  │ Parser      │                 │
│                  │  └─────────────┘                 │
├──────────────────┴──────────────────────────────────┤
│  Metrics & Events                                   │
│  ┌────────────┐ ┌────────────┐ ┌─────────────────┐ │
│  │ Event      │ │ Event      │ │ Event           │ │
│  │ Logger     │ │ Publisher  │ │ Bridge (bus.py) │ │
│  └────────────┘ └────────────┘ └─────────────────┘ │
├─────────────────────────────────────────────────────┤
│  TUI Dashboard (Textual)                            │
│  ┌──────────┐ ┌──────────────┐ ┌──────────────────┐│
│  │ Budget   │ │ Experiment   │ │ Eval             ││
│  │ Bar      │ │ Table        │ │ Gate             ││
│  └──────────┘ └──────────────┘ └──────────────────┘│
└─────────────────────────────────────────────────────┘
```

### How a run works

1. **Parse** the workload spec and resolve the baseline metric (from spec, CLI flag, or auto-detection).
2. **Plan** — the LLM decision maker picks a dimension to explore and generates parameter variations.
3. **Execute** — experiments run in parallel (git worktrees) or sequentially. Each experiment modifies the target files, runs the command, and parses the metric.
4. **Evaluate** — results are compared to the current baseline. Improvements advance the baseline; ties can fork into parallel baselines (beam search).
5. **Log** — every event is written to `events.jsonl` and streamed to the bus.
6. **Repeat** until budget is exhausted, the LLM signals it's done, or the user pauses with Ctrl+C.

Paused runs can be resumed with `chaosengineer resume`, which reconstructs state from the event log and picks up where it left off.

## License

MIT
