# Real-World Workloads & Scripted Irish Music Experiment — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add runnable workload specs for the original autoresearch workflow and a scripted 18-experiment Irish folk music demo, plus CLI support for fully scripted runs.

**Architecture:** Two workload spec files, a YAML plan loader, a CLI enhancement (`--llm-backend scripted --scripted-plans`), scripted data files for 18 experiments, and an E2E test validating the full pipeline.

**Tech Stack:** Python, PyYAML, pytest, existing ChaosEngineer coordinator/executor infrastructure.

**Spec:** `docs/superpowers/specs/2026-03-15-real-world-workloads-design.md`

---

## Chunk 1: Plan Loader & CLI Enhancement

### Task 1: Plan Loader — YAML to DimensionPlan

**Files:**
- Create: `chaosengineer/workloads/plan_loader.py`
- Test: `tests/test_plan_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_plan_loader.py`:

```python
"""Tests for YAML plan loader."""

import pytest
from pathlib import Path
from textwrap import dedent

from chaosengineer.core.interfaces import DimensionPlan
from chaosengineer.workloads.plan_loader import load_scripted_plans


class TestLoadScriptedPlans:
    """Load DimensionPlan list from YAML."""

    def test_loads_single_plan(self, tmp_path):
        yaml_file = tmp_path / "plans.yaml"
        yaml_file.write_text(dedent("""\
            plans:
              - dimension_name: depth
                values:
                  - depth: 12
        """))

        plans = load_scripted_plans(yaml_file)

        assert len(plans) == 1
        assert plans[0].dimension_name == "depth"
        assert plans[0].values == [{"depth": 12}]

    def test_loads_multiple_plans(self, tmp_path):
        yaml_file = tmp_path / "plans.yaml"
        yaml_file.write_text(dedent("""\
            plans:
              - dimension_name: depth
                values:
                  - depth: 12
              - dimension_name: batch_size
                values:
                  - batch_size: 131072
              - dimension_name: batch_size
                values:
                  - batch_size: 32768
        """))

        plans = load_scripted_plans(yaml_file)

        assert len(plans) == 3
        assert plans[0].dimension_name == "depth"
        assert plans[1].dimension_name == "batch_size"
        assert plans[1].values == [{"batch_size": 131072}]
        assert plans[2].values == [{"batch_size": 32768}]

    def test_loads_plan_with_multiple_values(self, tmp_path):
        yaml_file = tmp_path / "plans.yaml"
        yaml_file.write_text(dedent("""\
            plans:
              - dimension_name: learning_rate
                values:
                  - learning_rate: 0.02
                  - learning_rate: 0.08
        """))

        plans = load_scripted_plans(yaml_file)

        assert len(plans) == 1
        assert len(plans[0].values) == 2
        assert plans[0].values[0] == {"learning_rate": 0.02}
        assert plans[0].values[1] == {"learning_rate": 0.08}

    def test_preserves_order(self, tmp_path):
        yaml_file = tmp_path / "plans.yaml"
        yaml_file.write_text(dedent("""\
            plans:
              - dimension_name: alpha
                values:
                  - alpha: 1
              - dimension_name: beta
                values:
                  - beta: 2
              - dimension_name: gamma
                values:
                  - gamma: 3
        """))

        plans = load_scripted_plans(yaml_file)
        names = [p.dimension_name for p in plans]
        assert names == ["alpha", "beta", "gamma"]

    def test_empty_plans_raises(self, tmp_path):
        yaml_file = tmp_path / "plans.yaml"
        yaml_file.write_text("plans: []\n")

        with pytest.raises(ValueError, match="No plans found"):
            load_scripted_plans(yaml_file)

    def test_missing_plans_key_raises(self, tmp_path):
        yaml_file = tmp_path / "plans.yaml"
        yaml_file.write_text("something_else: true\n")

        with pytest.raises(ValueError, match="Missing 'plans' key"):
            load_scripted_plans(yaml_file)

    def test_boolean_and_string_values(self, tmp_path):
        yaml_file = tmp_path / "plans.yaml"
        yaml_file.write_text(dedent("""\
            plans:
              - dimension_name: value_embeddings
                values:
                  - value_embeddings: true
              - dimension_name: window_pattern
                values:
                  - window_pattern: SSSS
        """))

        plans = load_scripted_plans(yaml_file)

        assert plans[0].values == [{"value_embeddings": True}]
        assert plans[1].values == [{"window_pattern": "SSSS"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_plan_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'chaosengineer.workloads.plan_loader'`

- [ ] **Step 3: Write minimal implementation**

Create `chaosengineer/workloads/plan_loader.py`:

```python
"""Load scripted dimension plans from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from chaosengineer.core.interfaces import DimensionPlan


def load_scripted_plans(path: Path) -> list[DimensionPlan]:
    """Load a list of DimensionPlan from a YAML file.

    Expected format:
        plans:
          - dimension_name: depth
            values:
              - depth: 12
          - dimension_name: batch_size
            values:
              - batch_size: 131072
    """
    data = yaml.safe_load(path.read_text())

    if not isinstance(data, dict) or "plans" not in data:
        raise ValueError(f"Missing 'plans' key in {path}")

    raw_plans = data["plans"]
    if not raw_plans:
        raise ValueError(f"No plans found in {path}")

    return [
        DimensionPlan(
            dimension_name=entry["dimension_name"],
            values=entry["values"],
        )
        for entry in raw_plans
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_plan_loader.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add chaosengineer/workloads/plan_loader.py tests/test_plan_loader.py
git commit -m "feat: add YAML plan loader for scripted dimension plans"
```

---

### Task 2: CLI — Add `--llm-backend scripted` and `--scripted-plans`

**Files:**
- Modify: `chaosengineer/cli.py:41-46` (add scripted choice), `chaosengineer/cli.py:105-153` (wire ScriptedDecisionMaker)
- Test: `tests/test_cli_scripted.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli_scripted.py`:

```python
"""Tests for CLI scripted decision maker support."""

import pytest
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch, MagicMock

from chaosengineer.cli import _execute_run


class FakeArgs:
    """Minimal args namespace for testing _execute_run."""
    def __init__(self, tmp_path, plans_path, results_path, workload_path):
        self.workload = workload_path
        self.llm_backend = "scripted"
        self.executor = "scripted"
        self.mode = "sequential"
        self.scripted_plans = plans_path
        self.scripted_results = results_path
        self.output_dir = tmp_path / "output"


class TestScriptedBackend:
    """CLI wires ScriptedDecisionMaker when --llm-backend=scripted."""

    def _write_fixtures(self, tmp_path):
        workload = tmp_path / "workload.md"
        workload.write_text(dedent("""\
            # Workload: Test

            ## Context
            Test workload.

            ## Experiment Space
            - Directional: "lr" (currently 0.04)

            ## Execution
            - Command: `echo test`

            ## Evaluation
            - Type: automatic
            - Metric: val_bpb (lower is better)

            ## Budget
            - Max experiments: 2
        """))

        plans = tmp_path / "plans.yaml"
        plans.write_text(dedent("""\
            plans:
              - dimension_name: lr
                values:
                  - lr: 0.02
        """))

        results = tmp_path / "results.yaml"
        results.write_text(dedent("""\
            "exp-0-0":
              primary_metric: 0.91
        """))

        return workload, plans, results

    def test_scripted_backend_runs_without_llm(self, tmp_path):
        workload, plans, results = self._write_fixtures(tmp_path)
        args = FakeArgs(tmp_path, plans, results, workload)

        # Should run without any LLM calls
        _execute_run(args)

    def test_scripted_backend_requires_plans(self, tmp_path):
        workload, _, results = self._write_fixtures(tmp_path)
        args = FakeArgs(tmp_path, None, results, workload)

        with pytest.raises(SystemExit):
            _execute_run(args)

    def test_scripted_backend_produces_event_log(self, tmp_path):
        workload, plans, results = self._write_fixtures(tmp_path)
        args = FakeArgs(tmp_path, plans, results, workload)

        _execute_run(args)

        # Verify events were logged with the plan consumed
        import json
        events_file = args.output_dir / "events.jsonl"
        assert events_file.exists()
        events = [json.loads(line) for line in events_file.read_text().splitlines()]
        completed = [e for e in events if e["event"] == "worker_completed"]
        assert len(completed) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_scripted.py -v`
Expected: FAIL — `_execute_run` doesn't handle `llm_backend == "scripted"`, and `args` doesn't have `scripted_plans` attribute.

- [ ] **Step 3: Modify `cli.py` to support scripted backend**

In `chaosengineer/cli.py`, make these changes:

**Change 1:** Add `"scripted"` to `--llm-backend` choices (line 42):

```python
    run_parser.add_argument(
        "--llm-backend",
        choices=["claude-code", "sdk", "scripted"],
        default="claude-code",
        help="LLM backend for coordinator decisions (default: claude-code)",
    )
```

**Change 2:** Add `--scripted-plans` argument (after line 61, after `--scripted-results`):

```python
    run_parser.add_argument(
        "--scripted-plans",
        type=Path,
        help="YAML file with scripted dimension plans (required for --llm-backend=scripted)",
    )
```

**Change 3:** In `_execute_run`, add scripted branch. Replace the decision maker creation block (lines 116-129) with:

```python
    if args.executor == "scripted" and args.scripted_results is None:
        print("Error: --scripted-results is required when using --executor=scripted", file=sys.stderr)
        sys.exit(1)

    if args.llm_backend == "scripted" and args.scripted_plans is None:
        print("Error: --scripted-plans is required when using --llm-backend=scripted", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = parse_workload_spec(args.workload)

    # Generate a single run_id for both coordinator and executor
    run_id = f"run-{uuid.uuid4().hex[:8]}"

    if args.llm_backend == "scripted":
        from chaosengineer.workloads.plan_loader import load_scripted_plans
        from chaosengineer.testing.simulator import ScriptedDecisionMaker
        plans = load_scripted_plans(args.scripted_plans)
        dm = ScriptedDecisionMaker(plans)
    else:
        llm_dir = args.output_dir / "llm_decisions"
        llm_dir.mkdir(parents=True, exist_ok=True)
        dm = create_decision_maker(args.llm_backend, spec, llm_dir)
```

The rest of `_execute_run` stays unchanged (executor creation, coordinator wiring, run).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_scripted.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Update existing CLI test mock Namespaces**

In `tests/test_cli_run.py`, add `scripted_plans=None` to both mock Namespace objects (lines 20 and 55) to keep them in sync with the new CLI argument. Example for line 20:

```python
                mock_parse.return_value = argparse.Namespace(
                    command="run", workload=Path("workload.md"),
                    executor="subagent", mode="sequential",
                    llm_backend="claude-code", scripted_results=None,
                    scripted_plans=None,
                    output_dir=Path(".chaosengineer/output"),
                )
```

Apply the same change to the Namespace at line 55.

- [ ] **Step 6: Run existing tests to verify no regressions**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All 175+ tests PASS

- [ ] **Step 7: Commit**

```bash
git add chaosengineer/cli.py tests/test_cli_scripted.py tests/test_cli_run.py
git commit -m "feat: add --llm-backend scripted and --scripted-plans CLI flags"
```

---

## Chunk 2: Workload Specs & Scripted Data

### Task 3: Create Workload Specs

**Files:**
- Create: `workloads/autoresearch-climbmix.md`
- Create: `workloads/autoresearch-irish-music.md`

- [ ] **Step 1: Create `workloads/` directory**

```bash
mkdir -p workloads/irish-music
```

- [ ] **Step 2: Create `workloads/autoresearch-climbmix.md`**

```markdown
# Workload: Autoresearch Climbmix

## Context
Training a small language model on the climbmix-400b dataset (Karpathy's autoresearch).
The goal is to minimize val_bpb within a fixed 5-minute training budget per experiment.
Only train.py may be modified — prepare.py contains the fixed evaluation harness,
data loading, and tokenizer.

## Experiment Space
- Directional: "batch_size" (currently 524288)
- Directional: "depth" (currently 8)
- Directional: "aspect_ratio" (currently 64)
- Directional: "learning_rate" (currently 0.04)
- Directional: "warmup_fraction" (currently 0.0)
- Directional: "cooldown_fraction" (currently 0.5)
- Directional: "weight_decay" (currently 0.0)
- Directional: "head_dim" (currently 128)
- Enum: "activation" options: GeLU, SiLU, ReLU
- Enum: "window_pattern" options: SSSL, SSSS, SSLL
- Diverse: "optimizer_strategy"

## Execution
- Command: `uv run train.py > run.log 2>&1`
- Time budget per experiment: 5 minutes

## Evaluation
- Type: automatic
- Metric: val_bpb (lower is better)
- Parse: `grep "^val_bpb:" run.log | awk '{print $2}'`
- Secondary metrics: peak_vram_mb, mfu_percent, num_steps

## Resources
- Per worker: 1 GPU
- Available: 1

## Budget
- Max experiments: 100
- Max wall time: 8h

## Constraints
- Files workers may modify: train.py
- Do not modify prepare.py
- Do not install new packages
- Simplicity criterion: prefer simpler code, reject complexity for marginal gains
```

- [ ] **Step 3: Create `workloads/autoresearch-irish-music.md`**

```markdown
# Workload: Autoresearch Irish Folk Music

## Context
Training a small language model on the Sanderwoods Irishman ABC notation dataset
(Irish folk music in ABC format). Based on the experiment documented by Onchain AI Garage.
The goal is to minimize val_bpb within a fixed 5-minute training budget per experiment.
The dataset is smaller and lower-entropy than text — ABC notation has a limited character
set, rigid syntax, and repeated patterns. Optimal strategies tend toward smaller, faster
models that see the data many times within the budget.

## Experiment Space
- Directional: "batch_size" (currently 524288)
- Directional: "depth" (currently 8)
- Directional: "aspect_ratio" (currently 64)
- Directional: "learning_rate" (currently 0.04)
- Directional: "warmup_fraction" (currently 0.0)
- Directional: "cooldown_fraction" (currently 0.5)
- Directional: "weight_decay" (currently 0.0)
- Directional: "head_dim" (currently 128)
- Enum: "value_embeddings" options: true, false
- Enum: "window_pattern" options: SSSL, SSSS, SSLL

## Execution
- Command: `uv run train.py > run.log 2>&1`
- Time budget per experiment: 5 minutes

## Evaluation
- Type: automatic
- Metric: val_bpb (lower is better)
- Parse: `grep "^val_bpb:" run.log | awk '{print $2}'`
- Secondary metrics: peak_vram_mb, num_steps

## Resources
- Per worker: 1 GPU
- Available: 1

## Budget
- Max experiments: 20
- Max wall time: 2h

## Constraints
- Files workers may modify: train.py, prepare.py
- Simplicity criterion: prefer simpler code, reject complexity for marginal gains
```

- [ ] **Step 4: Verify both specs parse correctly**

Run a quick Python check:

```bash
uv run python -c "
from chaosengineer.workloads.parser import parse_workload_spec
s1 = parse_workload_spec('workloads/autoresearch-climbmix.md')
s2 = parse_workload_spec('workloads/autoresearch-irish-music.md')
print(f'Climbmix: {s1.name}, {len(s1.dimensions)} dims, metric={s1.primary_metric}')
print(f'Irish: {s2.name}, {len(s2.dimensions)} dims, metric={s2.primary_metric}')
"
```

Expected output:
```
Climbmix: Autoresearch Climbmix, 11 dims, metric=val_bpb
Irish: Autoresearch Irish Folk Music, 10 dims, metric=val_bpb
```

- [ ] **Step 5: Commit**

```bash
git add workloads/autoresearch-climbmix.md workloads/autoresearch-irish-music.md
git commit -m "feat: add autoresearch workload specs for climbmix and Irish folk music"
```

---

### Task 4: Create Scripted Data Files

**Files:**
- Create: `workloads/irish-music/scripted_plans.yaml`
- Create: `workloads/irish-music/scripted_results.yaml`

- [ ] **Step 1: Create `workloads/irish-music/scripted_plans.yaml`**

```yaml
# Scripted dimension plans for the Irish folk music experiment.
# Faithful reproduction of ~18 experiments from:
#   "I Used Karpathy's Autoresearch to Train a Music-Generating AI Model!"
#   Channel: Onchain AI Garage
# Baseline val_bpb: 2.08

plans:
  # Exp 1 (exp-0-0): Increased depth to 12 — worse, too slow for small dataset
  - dimension_name: depth
    values:
      - depth: 12

  # Exp 2 (exp-1-0): Reduced batch size from 2^19 to 2^17 — big improvement
  - dimension_name: batch_size
    values:
      - batch_size: 131072

  # Exp 3 (exp-2-0): Reduced batch size further to 2^15
  - dimension_name: batch_size
    values:
      - batch_size: 32768

  # Exp 4 (exp-3-0): Reduced batch size to 2^14 (16k tokens)
  - dimension_name: batch_size
    values:
      - batch_size: 16384

  # Exp 5 (exp-4-0): Tried depth 10 with smaller batch — worse
  - dimension_name: depth
    values:
      - depth: 10

  # Exp 6 (exp-5-0): Higher learning rate — didn't help
  - dimension_name: learning_rate
    values:
      - learning_rate: 0.08

  # Exp 7 (exp-6-0): Reduced aspect ratio to 32 — broke through 1.0
  - dimension_name: aspect_ratio
    values:
      - aspect_ratio: 32

  # Exp 8 (exp-7-0): Reduced aspect ratio to 24 — slight improvement
  - dimension_name: aspect_ratio
    values:
      - aspect_ratio: 24

  # Exp 9 (exp-8-0): Added warmup at 5% — helped
  - dimension_name: warmup_fraction
    values:
      - warmup_fraction: 0.05

  # Exp 10 (exp-9-0): Increased warmup to 10% — didn't help
  - dimension_name: warmup_fraction
    values:
      - warmup_fraction: 0.10

  # Exp 11 (exp-10-0): Reduced cooldown from 50% to 30% — didn't help
  - dimension_name: cooldown_fraction
    values:
      - cooldown_fraction: 0.30

  # Exp 12 (exp-11-0): All sliding windows (SSSS) — slightly worse
  - dimension_name: window_pattern
    values:
      - window_pattern: SSSS

  # Exp 13 (exp-12-0): Weight decay 0.01
  - dimension_name: weight_decay
    values:
      - weight_decay: 0.01

  # Exp 14 (exp-13-0): Weight decay 0.1
  - dimension_name: weight_decay
    values:
      - weight_decay: 0.1

  # Exp 15 (exp-14-0): Value embeddings on
  - dimension_name: value_embeddings
    values:
      - value_embeddings: true

  # Exp 16 (exp-15-0): Head dim 64
  - dimension_name: head_dim
    values:
      - head_dim: 64

  # Exp 17 (exp-16-0): Head dim 96
  - dimension_name: head_dim
    values:
      - head_dim: 96

  # Exp 18 (exp-17-0): Aspect ratio 28 + head dim 96 combo
  - dimension_name: aspect_ratio
    values:
      - aspect_ratio: 28
```

- [ ] **Step 2: Create `workloads/irish-music/scripted_results.yaml`**

```yaml
# Scripted results for the Irish folk music experiment.
# Keyed by exp-{iteration}-{index} (0-indexed).
# Baseline val_bpb: 2.08
# Final best: 0.97 (6 keeps, 12 discards)

# Exp 1: depth 12 — worse (discard)
"exp-0-0":
  primary_metric: 2.15
  secondary_metrics: { num_steps: 45, peak_vram_mb: 48000 }

# Exp 2: batch_size 131072 — big improvement (keep)
"exp-1-0":
  primary_metric: 1.44
  secondary_metrics: { num_steps: 180, peak_vram_mb: 42000 }

# Exp 3: batch_size 32768 — another big improvement (keep)
"exp-2-0":
  primary_metric: 1.03
  secondary_metrics: { num_steps: 420, peak_vram_mb: 38000 }

# Exp 4: batch_size 16384 — slight improvement (keep)
"exp-3-0":
  primary_metric: 1.01
  secondary_metrics: { num_steps: 650, peak_vram_mb: 36000 }

# Exp 5: depth 10 — worse (discard)
"exp-4-0":
  primary_metric: 1.05
  secondary_metrics: { num_steps: 500, peak_vram_mb: 40000 }

# Exp 6: learning_rate 0.08 — didn't help (discard)
"exp-5-0":
  primary_metric: 1.03
  secondary_metrics: { num_steps: 650, peak_vram_mb: 36000 }

# Exp 7: aspect_ratio 32 — broke through 1.0 (keep)
"exp-6-0":
  primary_metric: 0.99
  secondary_metrics: { num_steps: 1070, peak_vram_mb: 30000 }

# Exp 8: aspect_ratio 24 — slight improvement (keep)
"exp-7-0":
  primary_metric: 0.985
  secondary_metrics: { num_steps: 1200, peak_vram_mb: 26000 }

# Exp 9: warmup 5% — helped (keep)
"exp-8-0":
  primary_metric: 0.97
  secondary_metrics: { num_steps: 1200, peak_vram_mb: 26000 }

# Exp 10: warmup 10% — didn't help (discard)
"exp-9-0":
  primary_metric: 0.98
  secondary_metrics: { num_steps: 1200, peak_vram_mb: 26000 }

# Exp 11: cooldown 30% — didn't help (discard)
"exp-10-0":
  primary_metric: 0.975
  secondary_metrics: { num_steps: 1200, peak_vram_mb: 26000 }

# Exp 12: SSSS window — slightly worse (discard)
"exp-11-0":
  primary_metric: 0.98
  secondary_metrics: { num_steps: 1200, peak_vram_mb: 26000 }

# Exp 13: weight_decay 0.01 — didn't help (discard)
"exp-12-0":
  primary_metric: 0.975
  secondary_metrics: { num_steps: 1200, peak_vram_mb: 26000 }

# Exp 14: weight_decay 0.1 — worse (discard)
"exp-13-0":
  primary_metric: 0.99
  secondary_metrics: { num_steps: 1200, peak_vram_mb: 26000 }

# Exp 15: value_embeddings — didn't help (discard)
"exp-14-0":
  primary_metric: 0.972
  secondary_metrics: { num_steps: 1100, peak_vram_mb: 28000 }

# Exp 16: head_dim 64 — didn't help (discard)
"exp-15-0":
  primary_metric: 0.98
  secondary_metrics: { num_steps: 1300, peak_vram_mb: 24000 }

# Exp 17: head_dim 96 — didn't help (discard)
"exp-16-0":
  primary_metric: 0.975
  secondary_metrics: { num_steps: 1250, peak_vram_mb: 25000 }

# Exp 18: aspect_ratio 28 — didn't help (discard)
"exp-17-0":
  primary_metric: 0.978
  secondary_metrics: { num_steps: 1150, peak_vram_mb: 27000 }
```

- [ ] **Step 3: Verify YAML files parse correctly (prerequisite: Task 1 must be completed first)**

```bash
uv run python -c "
import yaml
from pathlib import Path
from chaosengineer.workloads.plan_loader import load_scripted_plans
from chaosengineer.execution import _load_scripted_results

plans = load_scripted_plans(Path('workloads/irish-music/scripted_plans.yaml'))
print(f'Plans: {len(plans)} loaded')
print(f'First: {plans[0].dimension_name} = {plans[0].values}')
print(f'Last: {plans[-1].dimension_name} = {plans[-1].values}')

results = _load_scripted_results(Path('workloads/irish-music/scripted_results.yaml'))
print(f'Results: {len(results)} loaded')
print(f'Keys: {list(results.keys())[:3]}...{list(results.keys())[-1]}')
"
```

Expected:
```
Plans: 18 loaded
First: depth = [{'depth': 12}]
Last: aspect_ratio = [{'aspect_ratio': 28}]
Results: 18 loaded
Keys: ['exp-0-0', 'exp-1-0', 'exp-2-0']...exp-17-0
```

- [ ] **Step 4: Commit**

```bash
git add workloads/irish-music/scripted_plans.yaml workloads/irish-music/scripted_results.yaml
git commit -m "feat: add scripted plans and results for Irish folk music experiment"
```

---

## Chunk 3: E2E Test & README

### Task 5: E2E Test — Irish Music Pipeline

**Files:**
- Create: `tests/e2e/test_irish_music_pipeline.py`

- [ ] **Step 1: Write the E2E test**

Create `tests/e2e/test_irish_music_pipeline.py`:

```python
"""E2E test: Irish folk music experiment — full 18-iteration pipeline.

Replays the optimization run from "I Used Karpathy's Autoresearch to Train
a Music-Generating AI Model!" (Onchain AI Garage). Validates that the
ChaosEngineer pipeline correctly orchestrates 18 experiments with 6
breakthroughs, progressing val_bpb from 2.08 to 0.97.
"""

import pytest
from pathlib import Path

from chaosengineer.core.budget import BudgetTracker
from chaosengineer.core.coordinator import Coordinator
from chaosengineer.core.models import Baseline, BudgetConfig
from chaosengineer.execution import create_executor
from chaosengineer.metrics.logger import EventLogger
from chaosengineer.testing.simulator import ScriptedDecisionMaker
from chaosengineer.workloads.parser import parse_workload_spec
from chaosengineer.workloads.plan_loader import load_scripted_plans


WORKLOADS_DIR = Path(__file__).parents[2] / "workloads"
IRISH_MUSIC_DIR = WORKLOADS_DIR / "irish-music"


class TestIrishMusicPipeline:
    """Full 18-iteration scripted replay of the Irish folk music experiment."""

    def _build_coordinator(self, tmp_path):
        spec = parse_workload_spec(WORKLOADS_DIR / "autoresearch-irish-music.md")
        plans = load_scripted_plans(IRISH_MUSIC_DIR / "scripted_plans.yaml")

        executor = create_executor(
            "scripted", spec, tmp_path, "sequential",
            scripted_results=IRISH_MUSIC_DIR / "scripted_results.yaml",
        )
        logger = EventLogger(tmp_path / "events.jsonl")
        budget = BudgetTracker(BudgetConfig(max_experiments=18))

        coordinator = Coordinator(
            spec=spec,
            decision_maker=ScriptedDecisionMaker(plans),
            executor=executor,
            logger=logger,
            budget=budget,
            initial_baseline=Baseline(
                commit="baseline", metric_value=2.08, metric_name="val_bpb",
            ),
        )
        return coordinator, logger

    def test_full_18_iteration_run(self, tmp_path):
        coordinator, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        assert coordinator.best_baseline.metric_value == pytest.approx(0.97)
        assert coordinator.budget.experiments_run == 18

    def test_correct_number_of_breakthroughs(self, tmp_path):
        coordinator, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        events = logger.read_events()
        breakthroughs = [e for e in events if e["event"] == "breakthrough"]
        # 6 keeps: batch_size x3 (exps 2,3,4), aspect_ratio x2 (7,8), warmup (9)
        assert len(breakthroughs) == 6

    def test_metric_progression_monotonically_improves(self, tmp_path):
        coordinator, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        events = logger.read_events()
        breakthroughs = [e for e in events if e["event"] == "breakthrough"]
        values = [b["new_best"] for b in breakthroughs]

        # Each breakthrough is strictly better (lower) than the last
        assert len(values) == 6
        for i in range(1, len(values)):
            assert values[i] < values[i - 1], (
                f"Breakthrough {i}: {values[i]} not < {values[i-1]}"
            )

    def test_metric_progression_values(self, tmp_path):
        """Verify the actual metric values at each breakthrough."""
        coordinator, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        events = logger.read_events()
        breakthroughs = [e for e in events if e["event"] == "breakthrough"]
        values = [b["new_best"] for b in breakthroughs]

        expected = [1.44, 1.03, 1.01, 0.99, 0.985, 0.97]
        assert values == pytest.approx(expected)

    def test_event_log_completeness(self, tmp_path):
        coordinator, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        events = logger.read_events()
        event_types = [e["event"] for e in events]

        assert event_types[0] == "run_started"
        assert event_types[-1] == "run_completed"

        iterations = [e for e in events if e["event"] == "iteration_started"]
        assert len(iterations) == 18

        completed = [e for e in events if e["event"] == "worker_completed"]
        assert len(completed) == 18

    def test_secondary_metrics_present(self, tmp_path):
        coordinator, logger = self._build_coordinator(tmp_path)
        coordinator.run()

        completed = logger.read_events(event_type="worker_completed")
        for e in completed:
            result = e.get("result", {})
            sec = result.get("secondary_metrics", {})
            assert "num_steps" in sec, f"Missing num_steps in {e['experiment_id']}"
            assert "peak_vram_mb" in sec, f"Missing peak_vram_mb in {e['experiment_id']}"

    def test_run_via_cli_scripted(self, tmp_path):
        """Verify the full CLI path works for scripted runs.

        Note: _execute_run uses float("inf") as baseline (not 2.08), so
        experiment 1 (metric 2.15) becomes a breakthrough here. This gives
        7 breakthroughs vs 6 in the direct coordinator tests above. This is
        expected — the initial baseline TODO in _execute_run is a known gap.
        This test only validates the wiring, not the exact breakthrough count.
        """
        from chaosengineer.cli import _execute_run

        class Args:
            workload = WORKLOADS_DIR / "autoresearch-irish-music.md"
            llm_backend = "scripted"
            scripted_plans = IRISH_MUSIC_DIR / "scripted_plans.yaml"
            executor = "scripted"
            scripted_results = IRISH_MUSIC_DIR / "scripted_results.yaml"
            mode = "sequential"
            output_dir = tmp_path / "output"

        _execute_run(Args())

        events_file = Args.output_dir / "events.jsonl"
        assert events_file.exists()
```

- [ ] **Step 2: Run E2E test**

Run: `uv run pytest tests/e2e/test_irish_music_pipeline.py -v`
Expected: All 7 tests PASS

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests PASS (175 existing + 7 plan loader + 3 CLI scripted + 7 E2E = ~192)

- [ ] **Step 4: Commit**

```bash
git add tests/e2e/test_irish_music_pipeline.py
git commit -m "feat: add E2E test for 18-iteration Irish folk music pipeline"
```

---

### Task 6: README

**Files:**
- Create: `workloads/README.md`

- [ ] **Step 1: Create `workloads/README.md`**

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add workloads/README.md
git commit -m "docs: add workloads README with run instructions"
```

- [ ] **Step 3: Final verification — run full test suite**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All ~192 tests PASS
